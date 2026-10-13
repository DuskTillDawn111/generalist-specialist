import argparse
import json
import queue
import time
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
    warm_one as qwen_warm_one,
)


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


def collect_qwen(
    worker,
    count,
    timeout_s,
):

    results = []

    deadline = (
        time.time()
        + timeout_s
    )

    while len(results) < count:

        remaining = (
            deadline
            - time.time()
        )

        if remaining <= 0:
            raise TimeoutError(
                "Qwen benchmark timeout."
            )

        try:

            kind, payload = (
                worker.events.get(
                    timeout=min(
                        remaining,
                        1.0,
                    )
                )
            )

        except queue.Empty:

            if (
                worker.process.poll()
                is not None
            ):
                raise RuntimeError(
                    "Qwen worker exited."
                )

            continue

        if kind == "RESULT":

            results.append(
                payload
            )

        elif kind == "ERROR":

            raise RuntimeError(
                payload
            )

    return results


def collect_shared(
    worker_name,
    worker,
    events,
    count,
    timeout_s,
):

    results = []

    deadline = (
        time.time()
        + timeout_s
    )

    while len(results) < count:

        remaining = (
            deadline
            - time.time()
        )

        if remaining <= 0:
            raise TimeoutError(
                f"{worker_name} benchmark timeout."
            )

        try:

            name, kind, payload = (
                events.get(
                    timeout=min(
                        remaining,
                        1.0,
                    )
                )
            )

        except queue.Empty:

            if (
                worker.process.poll()
                is not None
            ):
                raise RuntimeError(
                    f"{worker_name} worker exited."
                )

            continue

        if name != worker_name:
            continue

        if kind == "RESULT":

            results.append(
                payload
            )

        elif kind == "ERROR":

            raise RuntimeError(
                payload
            )

    return results


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pool",
        required=True,
    )

    parser.add_argument(
        "--target-worker",
        required=True,
        choices=sorted(
            MODEL_TASKS
        ),
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
        "--target-count",
        type=int,
        default=30,
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

    if (
        args.target_task
        not in MODEL_TASKS[
            args.target_worker
        ]
    ):
        raise ValueError(
            f"{args.target_worker} does not "
            f"support {args.target_task}"
        )

    pool = read_jsonl(
        args.pool
    )

    task_rows = [
        row
        for row in pool
        if row["task_type"]
        == args.target_task
    ]

    if (
        len(task_rows)
        < args.target_count
    ):
        raise RuntimeError(
            f"Need {args.target_count} "
            f"{args.target_task} samples, "
            f"got {len(task_rows)}."
        )

    targets = (
        task_rows[
            :args.target_count
        ]
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    worker = None
    shared_events = None

    try:

        print(
            "===== ACTIVE SINGLE PROFILE ====="
        )

        print(
            "target:",
            args.target_worker,
            "->",
            args.target_task,
        )

        print(
            "target_count:",
            args.target_count,
        )

        # ====================================
        # Start worker
        # ====================================

        if args.target_worker == "qwen":

            worker = QwenWorker()

            print(
                "Starting qwen..."
            )

            worker.start()

            qwen_wait_for(
                worker,
                "READY",
                args.timeout_s,
            )

            print(
                "qwen: READY"
            )

        else:

            shared_events = (
                queue.Queue()
            )

            worker = SharedWorker(
                args.target_worker,
                WORKERS[
                    args.target_worker
                ],
                shared_events,
            )

            print(
                f"Starting "
                f"{args.target_worker}..."
            )

            worker.start()

            wait_ready(
                args.target_worker,
                {
                    args.target_worker:
                        worker
                },
                shared_events,
                args.timeout_s,
            )

            print(
                f"{args.target_worker}: "
                f"READY"
            )

        # ====================================
        # Warmup exactly target task
        # ====================================

        print()
        print(
            "===== WARMUP ====="
        )

        if args.target_worker == "qwen":

            qwen_warm_one(
                worker,
                targets[0],
                -1,
                args.timeout_s,
            )

            print(
                f"Warmup qwen "
                f"{args.target_task}: OK"
            )

        else:

            shared_warm_one(
                args.target_worker,
                targets[0],
                -1,
                {
                    args.target_worker:
                        worker
                },
                shared_events,
                args.timeout_s,
            )

        # ====================================
        # Submit same 30 targets
        # ====================================

        print()
        print(
            "===== TARGET-ONLY BENCHMARK ====="
        )

        for request_id, item in enumerate(
            targets
        ):

            worker.send({
                "request_id":
                    request_id,

                "pool_id":
                    item["pool_id"],

                "arrival_index":
                    item["arrival_index"],

                "task_type":
                    item["task_type"],

                "sample":
                    item["sample"],

                "submit_ns":
                    time.time_ns(),
            })

        # ====================================
        # Collect
        # ====================================

        if args.target_worker == "qwen":

            results = collect_qwen(
                worker,
                args.target_count,
                args.timeout_s,
            )

        else:

            results = collect_shared(
                args.target_worker,
                worker,
                shared_events,
                args.target_count,
                args.timeout_s,
            )

        results.sort(
            key=lambda row:
                int(
                    row[
                        "request_id"
                    ]
                )
        )

        # ====================================
        # Verify exact same pool IDs
        # ====================================

        expected_ids = [
            row["pool_id"]
            for row in targets
        ]

        actual_ids = [
            row["pool_id"]
            for row in results
        ]

        if (
            expected_ids
            != actual_ids
        ):
            raise RuntimeError(
                "Result pool IDs do not "
                "match target pool IDs."
            )

        # ====================================
        # Save
        # ====================================

        with output.open(
            "w",
            encoding="utf-8",
        ) as f:

            for row in results:

                row[
                    "profile_role"
                ] = "single_target"

                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        latencies = [
            float(
                row[
                    "worker_latency_ms"
                ]
            )
            for row in results
        ]

        summary = {
            "protocol":
                "active_single_profile_v1",

            "target_worker":
                args.target_worker,

            "target_task":
                args.target_task,

            "target_count":
                len(results),

            "mean_worker_latency_ms":
                mean(latencies),

            "min_worker_latency_ms":
                min(latencies),

            "max_worker_latency_ms":
                max(latencies),

            "output":
                str(output),
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
            "===== ACTIVE SINGLE RESULT ====="
        )

        print(
            "samples =",
            len(results),
        )

        print(
            "mean worker latency ms =",
            round(
                mean(latencies),
                6,
            ),
        )

        print(
            "min ms =",
            round(
                min(latencies),
                6,
            ),
        )

        print(
            "max ms =",
            round(
                max(latencies),
                6,
            ),
        )

        print(
            "Saved:",
            output,
        )

    finally:

        if worker is not None:

            try:
                worker.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
