import argparse
import json
import queue
import time
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from statistics import mean

from run_static_hybrid import (
    WORKERS,
    Worker as SharedWorker,
    read_jsonl,
)

from run_dynamic_hybrid import (
    wait_ready,
    warm_one as shared_warm_one,
)

from run_pure_qwen import (
    Worker as QwenWorker,
    wait_for as qwen_wait_for,
)


ALL_MODELS = {
    "qwen",
    "distilbert",
    "resnet50",
    "paddleocr",
    "whisper",
}


MODEL_TASKS = {
    "qwen": {
        "text",
        "image",
        "ocr",
        "asr",
        "visual_qa",
        "lid",
    },
    "distilbert": {
        "text",
    },
    "resnet50": {
        "image",
    },
    "paddleocr": {
        "ocr",
    },
    "whisper": {
        "asr",
        "lid",
    },
}


# Canonical saturation workload used when a
# co-resident model supports multiple task types.
CANONICAL_CO_TASKS = {
    "qwen": [
        "text",
        "image",
        "ocr",
        "asr",
        "visual_qa",
        "lid",
    ],
    "distilbert": [
        "text",
    ],
    "resnet50": [
        "image",
    ],
    "paddleocr": [
        "ocr",
    ],
    "whisper": [
        "asr",
        "lid",
    ],
}


def qwen_warm_one(
    worker,
    item,
    request_id,
    timeout_s,
):
    worker.send({
        "request_id": request_id,
        "pool_id":
            "warmup:" + item["pool_id"],
        "arrival_index": -1,
        "task_type": item["task_type"],
        "sample": item["sample"],
        "submit_ns": time.time_ns(),
    })

    qwen_wait_for(
        worker,
        "RESULT",
        timeout_s,
    )

    print(
        f"Warmup qwen        "
        f"{item['task_type']:>10s}: OK"
    )


def merge_intervals(intervals):
    if not intervals:
        return []

    intervals = sorted(intervals)

    merged = [
        list(intervals[0])
    ]

    for start, end in intervals[1:]:

        last = merged[-1]

        if start <= last[1]:
            last[1] = max(
                last[1],
                end,
            )
        else:
            merged.append(
                [start, end]
            )

    return merged


def overlap_with_union(
    start,
    end,
    intervals,
):
    overlap = 0

    for a, b in intervals:

        if b <= start:
            continue

        if a >= end:
            break

        overlap += max(
            0,
            min(end, b)
            - max(start, a),
        )

    return overlap


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pool",
        required=True,
    )

    parser.add_argument(
        "--target-worker",
        required=True,
        choices=sorted(ALL_MODELS),
    )

    parser.add_argument(
        "--target-task",
        required=True,
        choices=[
            "text",
            "image",
            "ocr",
            "asr",
            "visual_qa",
            "lid",
        ],
    )

    parser.add_argument(
        "--co-worker",
        required=True,
        choices=sorted(ALL_MODELS),
    )

    parser.add_argument(
        "--co-tasks",
        default=None,
        help=(
            "Comma-separated co-resident task list. "
            "Default: canonical workload for model."
        ),
    )

    parser.add_argument(
        "--target-count",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--co-backlog",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--co-gap-ms",
        type=float,
        default=0.0,
        help=(
            "Idle gap in milliseconds before "
            "submitting the next co-resident request. "
            "Use --co-backlog 1 when gap > 0."
        ),
    )

    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--timeout-s",
        type=float,
        default=3600.0,
    )

    args = parser.parse_args()

    if args.co_gap_ms < 0:
        raise ValueError(
            "--co-gap-ms must be >= 0."
        )

    if (
        args.co_gap_ms > 0
        and args.co_backlog != 1
    ):
        raise ValueError(
            "--co-gap-ms > 0 requires "
            "--co-backlog 1, otherwise queued "
            "co-resident requests hide the gap."
        )

    target_worker = args.target_worker
    target_task = args.target_task
    co_worker = args.co_worker

    if target_worker == co_worker:
        raise ValueError(
            "target-worker and co-worker "
            "must differ."
        )

    if (
        target_task
        not in MODEL_TASKS[target_worker]
    ):
        raise ValueError(
            f"{target_worker} does not support "
            f"{target_task}."
        )

    if args.target_count <= 0:
        raise ValueError(
            "--target-count must be positive."
        )

    if args.co_backlog <= 0:
        raise ValueError(
            "--co-backlog must be positive."
        )

    if not (
        0.0 <= args.min_overlap <= 1.0
    ):
        raise ValueError(
            "--min-overlap must be in [0, 1]."
        )

    if args.co_tasks:

        co_tasks = [
            x.strip()
            for x in args.co_tasks.split(",")
            if x.strip()
        ]

    else:

        co_tasks = list(
            CANONICAL_CO_TASKS[
                co_worker
            ]
        )

    if not co_tasks:
        raise ValueError(
            "No co-resident tasks."
        )

    for task in co_tasks:

        if task not in MODEL_TASKS[co_worker]:
            raise ValueError(
                f"{co_worker} does not support "
                f"{task}."
            )

    all_tasks = read_jsonl(
        args.pool
    )

    by_task = defaultdict(list)

    for item in all_tasks:
        by_task[
            item["task_type"]
        ].append(item)

    if not by_task[target_task]:
        raise RuntimeError(
            f"Pool missing target task: "
            f"{target_task}"
        )

    for task in co_tasks:

        if not by_task[task]:
            raise RuntimeError(
                f"Pool missing co task: {task}"
            )

    target_source = (
        by_task[target_task]
    )

    # If target-count exceeds number of samples,
    # reuse them cyclically. Smoke pool therefore
    # remains usable for arbitrary test counts.
    target_items = [
        target_source[
            i % len(target_source)
        ]
        for i in range(
            args.target_count
        )
    ]

    co_sources = {
        task: cycle(
            by_task[task]
        )
        for task in co_tasks
    }

    co_task_cycle = cycle(
        co_tasks
    )

    events = queue.Queue()

    models = [
        target_worker,
        co_worker,
    ]

    shared_models = [
        m
        for m in models
        if m != "qwen"
    ]

    shared_workers = {
        name: SharedWorker(
            name,
            WORKERS[name],
            events,
        )
        for name in shared_models
    }

    qwen = (
        QwenWorker()
        if "qwen" in models
        else None
    )

    target_results = []
    co_results = []

    target_completed = set()

    co_submitted = 0
    co_completed = 0

    TARGET_ID_BASE = 0
    CO_ID_BASE = 1_000_000

    benchmark_start_ns = None
    benchmark_end_ns = None

    def send_request(
        worker_name,
        request,
    ):
        if worker_name == "qwen":
            qwen.send(request)
        else:
            shared_workers[
                worker_name
            ].send(request)

    def submit_co():
        nonlocal co_submitted

        task = next(
            co_task_cycle
        )

        item = next(
            co_sources[task]
        )

        rid = (
            CO_ID_BASE
            + co_submitted
        )

        request = {
            "request_id": rid,
            "pool_id":
                f"co:{co_submitted}:"
                + item["pool_id"],
            "arrival_index":
                co_submitted,
            "task_type":
                task,
            "sample":
                item["sample"],
            "submit_ns":
                time.time_ns(),
        }

        send_request(
            co_worker,
            request,
        )

        co_submitted += 1

    def submit_targets():
        for i, item in enumerate(
            target_items
        ):

            rid = (
                TARGET_ID_BASE + i
            )

            request = {
                "request_id": rid,
                "pool_id":
                    f"target:{i}:"
                    + item["pool_id"],
                "arrival_index": i,
                "task_type":
                    target_task,
                "sample":
                    item["sample"],
                "submit_ns":
                    time.time_ns(),
            }

            send_request(
                target_worker,
                request,
            )

    def get_event():
        # Shared workers use one common queue.
        try:
            (
                worker_name,
                kind,
                payload,
            ) = events.get_nowait()

            return (
                worker_name,
                kind,
                payload,
            )

        except queue.Empty:
            pass

        # Qwen owns its own queue.
        if qwen is not None:

            try:
                kind, payload = (
                    qwen.events.get_nowait()
                )

                return (
                    "qwen",
                    kind,
                    payload,
                )

            except queue.Empty:
                pass

        return None

    def check_alive():
        for name, worker in (
            shared_workers.items()
        ):
            if (
                worker.process is not None
                and worker.process.poll()
                is not None
            ):
                raise RuntimeError(
                    f"{name} exited with "
                    f"return code "
                    f"{worker.process.returncode}"
                )

        if (
            qwen is not None
            and qwen.process is not None
            and qwen.process.poll()
            is not None
        ):
            raise RuntimeError(
                "qwen exited with "
                f"return code "
                f"{qwen.process.returncode}"
            )

    def handle_result(
        worker_name,
        kind,
        payload,
        allow_refill,
    ):
        nonlocal co_completed

        if kind == "ERROR":
            raise RuntimeError(
                f"{worker_name}: {payload}"
            )

        if kind != "RESULT":
            return

        rid = int(
            payload["request_id"]
        )

        payload[
            "assigned_worker"
        ] = worker_name

        if rid >= CO_ID_BASE:

            payload[
                "profile_role"
            ] = "co_resident"

            co_results.append(
                payload
            )

            co_completed += 1

            if allow_refill:

                if args.co_gap_ms > 0:
                    time.sleep(
                        args.co_gap_ms
                        / 1000.0
                    )

                submit_co()

        elif rid >= TARGET_ID_BASE:

            if worker_name != target_worker:
                raise RuntimeError(
                    "Target request returned from "
                    f"wrong worker: {worker_name}"
                )

            payload[
                "profile_role"
            ] = "target"

            target_results.append(
                payload
            )

            target_completed.add(
                rid
            )

    try:

        print(
            "===== ACTIVE PAIR PROFILE ====="
        )

        print(
            "target:",
            target_worker,
            "->",
            target_task,
        )

        print(
            "co-resident:",
            co_worker,
            "tasks =",
            co_tasks,
        )

        print(
            "target_count:",
            args.target_count,
        )

        print(
            "co_backlog:",
            args.co_backlog,
        )

        # ----------------------------------
        # Start workers
        # ----------------------------------

        for name in shared_models:

            print(
                f"Starting {name}..."
            )

            shared_workers[
                name
            ].start()

            wait_ready(
                name,
                shared_workers,
                events,
                args.timeout_s,
            )

            print(
                f"{name}: READY"
            )

        if qwen is not None:

            print(
                "Starting qwen..."
            )

            qwen.start()

            qwen_wait_for(
                qwen,
                "READY",
                args.timeout_s,
            )

            print(
                "qwen: READY"
            )

        # ----------------------------------
        # Warm target and all co paths
        # ----------------------------------

        print()
        print(
            "===== WARMUP ====="
        )

        warm_id = -1

        target_warm_item = (
            by_task[target_task][0]
        )

        if target_worker == "qwen":

            qwen_warm_one(
                qwen,
                target_warm_item,
                warm_id,
                args.timeout_s,
            )

        else:

            shared_warm_one(
                target_worker,
                target_warm_item,
                warm_id,
                shared_workers,
                events,
                args.timeout_s,
            )

        warm_id -= 1

        for task in co_tasks:

            item = by_task[
                task
            ][0]

            if co_worker == "qwen":

                qwen_warm_one(
                    qwen,
                    item,
                    warm_id,
                    args.timeout_s,
                )

            else:

                shared_warm_one(
                    co_worker,
                    item,
                    warm_id,
                    shared_workers,
                    events,
                    args.timeout_s,
                )

            warm_id -= 1

        # ----------------------------------
        # Prime co-resident backlog
        # ----------------------------------

        print()
        print(
            "===== PRIME CO-RESIDENT ====="
        )

        for _ in range(
            args.co_backlog
        ):
            submit_co()

        # Do not start target until at least
        # one real co-resident inference has
        # completed. Since a backlog already
        # exists, another request should be
        # executing immediately behind it.
        deadline = (
            time.time()
            + args.timeout_s
        )

        while co_completed == 0:

            if time.time() >= deadline:
                raise TimeoutError(
                    "Timed out while priming "
                    "co-resident worker."
                )

            event = get_event()

            if event is None:
                check_alive()
                time.sleep(0.001)
                continue

            handle_result(
                *event,
                allow_refill=True,
            )

        print(
            "co-resident active; "
            "starting target requests"
        )

        # ----------------------------------
        # Submit target workload
        # ----------------------------------

        print()
        print(
            "===== TARGET BENCHMARK ====="
        )

        benchmark_start_ns = (
            time.time_ns()
        )

        submit_targets()

        # ----------------------------------
        # Run target while continuously
        # replenishing co-resident backlog.
        # ----------------------------------

        while (
            len(target_completed)
            < args.target_count
        ):

            if time.time() >= deadline:
                raise TimeoutError(
                    "Active pair profile timed out."
                )

            event = get_event()

            if event is None:
                check_alive()
                time.sleep(0.001)
                continue

            handle_result(
                *event,
                allow_refill=True,
            )

        benchmark_end_ns = (
            time.time_ns()
        )

        # ----------------------------------
        # Stop replenishing. Collect enough
        # co results to know the interference
        # interval through the final target.
        # ----------------------------------

        last_target_end = max(
            r["end_ns"]
            for r in target_results
        )

        latest_co_end = max(
            (
                r["end_ns"]
                for r in co_results
            ),
            default=0,
        )

        while latest_co_end < last_target_end:

            if time.time() >= deadline:
                raise TimeoutError(
                    "Timed out while finalizing "
                    "co-resident overlap."
                )

            event = get_event()

            if event is None:
                check_alive()
                time.sleep(0.001)
                continue

            handle_result(
                *event,
                allow_refill=False,
            )

            latest_co_end = max(
                (
                    r["end_ns"]
                    for r in co_results
                ),
                default=0,
            )

        # ----------------------------------
        # Compute exact target overlap.
        # ----------------------------------

        co_intervals = merge_intervals([
            (
                r["start_ns"],
                r["end_ns"],
            )
            for r in co_results
        ])

        total_target_ns = 0
        total_overlap_ns = 0

        accepted = []

        for row in target_results:

            duration = (
                row["end_ns"]
                - row["start_ns"]
            )

            overlap = (
                overlap_with_union(
                    row["start_ns"],
                    row["end_ns"],
                    co_intervals,
                )
            )

            ratio = (
                overlap / duration
                if duration > 0
                else 0.0
            )

            row[
                "active_overlap_ns"
            ] = overlap

            row[
                "active_overlap_ratio"
            ] = ratio

            total_target_ns += (
                duration
            )

            total_overlap_ns += (
                overlap
            )

            if ratio >= args.min_overlap:
                accepted.append(
                    row
                )

        weighted_overlap = (
            total_overlap_ns
            / total_target_ns
            if total_target_ns > 0
            else 0.0
        )

        target_results.sort(
            key=lambda r:
                int(r["request_id"])
        )

        co_results.sort(
            key=lambda r:
                int(r["request_id"])
        )

        output = Path(
            args.output
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w",
            encoding="utf-8",
        ) as f:

            for row in (
                target_results
                + co_results
            ):
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        accepted_ms = [
            r["worker_latency_ms"]
            for r in accepted
        ]

        all_target_ms = [
            r["worker_latency_ms"]
            for r in target_results
        ]

        summary = {
            "protocol":
                "active_pair_profile_v1",

            "target_worker":
                target_worker,

            "target_task":
                target_task,

            "co_resident":
                co_worker,

            "co_resident_tasks":
                co_tasks,

            "target_count":
                len(target_results),

            "co_completed_count":
                len(co_results),

            "co_submitted_count":
                co_submitted,

            "co_backlog":
                args.co_backlog,

            "min_overlap":
                args.min_overlap,

            "accepted_target_count":
                len(accepted),

            "accepted_fraction":
                (
                    len(accepted)
                    / len(target_results)
                    if target_results
                    else 0.0
                ),

            "weighted_overlap_ratio":
                weighted_overlap,

            "mean_target_latency_ms_all":
                (
                    mean(all_target_ms)
                    if all_target_ms
                    else None
                ),

            "mean_target_latency_ms_accepted":
                (
                    mean(accepted_ms)
                    if accepted_ms
                    else None
                ),

            "wall_time_s":
                (
                    (
                        benchmark_end_ns
                        - benchmark_start_ns
                    )
                    / 1e9
                    if (
                        benchmark_start_ns
                        is not None
                        and benchmark_end_ns
                        is not None
                    )
                    else None
                ),

            "valid_for_gamma":
                (
                    len(accepted)
                    == len(target_results)
                    and len(target_results) > 0
                    and weighted_overlap
                    >= args.min_overlap
                ),
        }

        summary_path = Path(
            str(output)
            + ".summary.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "===== ACTIVE PAIR RESULT ====="
        )

        print(
            "target samples =",
            len(target_results),
        )

        print(
            "accepted samples =",
            len(accepted),
        )

        print(
            "accepted fraction =",
            f"{summary['accepted_fraction']:.3f}",
        )

        print(
            "weighted overlap =",
            f"{weighted_overlap:.3f}",
        )

        print(
            "mean target latency all =",
            f"{summary['mean_target_latency_ms_all']:.3f}",
            "ms",
        )

        if accepted_ms:

            print(
                "mean target latency accepted =",
                f"{summary['mean_target_latency_ms_accepted']:.3f}",
                "ms",
            )

        else:

            print(
                "mean target latency accepted = NONE"
            )

        print(
            "valid_for_gamma =",
            summary[
                "valid_for_gamma"
            ],
        )

        print(
            "Saved:",
            output,
        )

        print(
            "Summary:",
            summary_path,
        )

    finally:

        # There may intentionally still be
        # queued co-resident requests.
        # Terminate rather than waiting for
        # the entire saturation backlog.
        for worker in (
            shared_workers.values()
        ):
            try:
                worker.terminate()
            except Exception:
                pass

        if qwen is not None:
            try:
                qwen.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
