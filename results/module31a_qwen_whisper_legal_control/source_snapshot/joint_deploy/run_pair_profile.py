import argparse
import json
import queue
import time
from collections import defaultdict
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


SPECIALIST_TASK = {
    "distilbert": "text",
    "resnet50": "image",
    "paddleocr": "ocr",
    "whisper": "asr",
}

ALL_MODELS = {
    "qwen",
    "distilbert",
    "resnet50",
    "paddleocr",
    "whisper",
}


PROFILE_TASKS = {
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


def qwen_warm_one(
    worker,
    item,
    request_id,
    timeout_s,
):
    worker.send({
        "request_id": request_id,
        "pool_id": item["pool_id"],
        "arrival_index": item["arrival_index"],
        "task_type": item["task_type"],
        "sample": item["sample"],
        "submit_ns": time.time_ns(),
    })

    qwen_wait_for(
        worker,
        "RESULT",
        timeout_s,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pool",
        required=True,
    )

    parser.add_argument(
        "--model-a",
        required=True,
        choices=sorted(ALL_MODELS),
    )

    parser.add_argument(
        "--model-b",
        required=True,
        choices=sorted(ALL_MODELS),
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

    parser.add_argument(
        "--qwen-asr-count",
        type=int,
        default=0,
        help=(
            "For qwen+whisper only: route this many "
            "ASR samples to Qwen and the remainder "
            "to Whisper."
        ),
    )

    parser.add_argument(
        "--qwen-native-count",
        type=int,
        default=0,
        help=(
            "For qwen+specialist calibration: "
            "route this many samples of the "
            "specialist native task to Qwen."
        ),
    )

    parser.add_argument(
        "--profile-worker",
        choices=sorted(ALL_MODELS),
        default=None,
        help=(
            "Explicit worker whose task latency "
            "should be profiled."
        ),
    )

    parser.add_argument(
        "--profile-task",
        choices=[
            "text",
            "image",
            "ocr",
            "asr",
            "visual_qa",
            "lid",
        ],
        default=None,
        help=(
            "Explicit task routed to "
            "--profile-worker."
        ),
    )

    args = parser.parse_args()

    if args.model_a == args.model_b:
        raise ValueError(
            "The two models must differ."
        )

    models = [
        args.model_a,
        args.model_b,
    ]

    if (
        (args.profile_worker is None)
        != (args.profile_task is None)
    ):
        raise ValueError(
            "--profile-worker and --profile-task "
            "must be provided together."
        )

    if args.profile_worker is not None:

        if args.profile_worker not in models:
            raise ValueError(
                "profile-worker must belong to the pair."
            )

        if (
            args.profile_task
            not in PROFILE_TASKS[
                args.profile_worker
            ]
        ):
            raise ValueError(
                f"{args.profile_worker} does not support "
                f"task {args.profile_task}."
            )

    if models.count("qwen") > 1:
        raise RuntimeError(
            "Invalid Qwen pair."
        )

    all_tasks = read_jsonl(
        args.pool
    )

    # --------------------------------------
    # Define canonical pair workload
    #
    # Qwen + specialist:
    #   specialist receives native task;
    #   Qwen receives other three tasks.
    #
    # specialist + specialist:
    #   each receives its own native task.
    # --------------------------------------

    routing = {}

    if "qwen" in models:

        specialist = (
            args.model_b
            if args.model_a == "qwen"
            else args.model_a
        )

        native = SPECIALIST_TASK[
            specialist
        ]

        for task in [
            "text",
            "image",
            "ocr",
            "asr",
        ]:
            routing[task] = (
                specialist
                if task == native
                else "qwen"
            )

    else:

        for model in models:
            routing[
                SPECIALIST_TASK[model]
            ] = model

    # --------------------------------------
    # Explicit profiling target.
    #
    # Keep the original canonical pair workload,
    # but additionally route the requested task
    # to the requested worker.
    # --------------------------------------

    if args.profile_worker is not None:

        routing[
            args.profile_task
        ] = args.profile_worker

        print(
            "Explicit profiling target:",
            args.profile_worker,
            "->",
            args.profile_task,
        )

    tasks = [
        item
        for item in all_tasks
        if item["task_type"]
        in routing
    ]

    # Optional deterministic ASR split for
    # Qwen + Whisper calibration.
    asr_override = {}

    if (
        set(models)
        == {"qwen", "whisper"}
        and args.qwen_asr_count > 0
    ):
        asr_items = [
            item
            for item in tasks
            if item["task_type"] == "asr"
        ]

        n = len(asr_items)

        if not (
            0 < args.qwen_asr_count < n
        ):
            raise ValueError(
                "--qwen-asr-count must be "
                f"between 1 and {n - 1}"
            )

        qwen_n = args.qwen_asr_count

        # Even deterministic spread rather than
        # one contiguous sample block.
        chosen = set()

        for i in range(qwen_n):
            idx = int(
                (i + 0.5)
                * n
                / qwen_n
            )

            idx = min(
                idx,
                n - 1,
            )

            chosen.add(
                asr_items[idx]["pool_id"]
            )

        # Guard against any index collision.
        if len(chosen) != qwen_n:
            chosen = set(
                item["pool_id"]
                for item in asr_items[
                    :qwen_n
                ]
            )

        for item in asr_items:
            asr_override[
                item["pool_id"]
            ] = (
                "qwen"
                if item["pool_id"] in chosen
                else "whisper"
            )


    # --------------------------------------
    # Optional native-task split for:
    #
    # Qwen + DistilBERT
    # Qwen + ResNet50
    # Qwen + PaddleOCR
    #
    # Used only for pairwise interference
    # calibration.
    # --------------------------------------

    native_override = {}

    if (
        "qwen" in models
        and "whisper" not in models
        and args.qwen_native_count > 0
    ):

        specialist = (
            args.model_b
            if args.model_a == "qwen"
            else args.model_a
        )

        native_task = SPECIALIST_TASK[
            specialist
        ]

        native_items = [
            item
            for item in tasks
            if item["task_type"]
            == native_task
        ]

        n = len(native_items)

        qwen_n = (
            args.qwen_native_count
        )

        if not (
            0 < qwen_n < n
        ):
            raise ValueError(
                "--qwen-native-count must "
                f"be between 1 and {n - 1}"
            )

        chosen = set()

        for i in range(qwen_n):

            idx = int(
                (i + 0.5)
                * n
                / qwen_n
            )

            idx = min(
                idx,
                n - 1,
            )

            chosen.add(
                native_items[
                    idx
                ]["pool_id"]
            )

        if len(chosen) != qwen_n:
            raise RuntimeError(
                "Native split produced "
                "duplicate sample indices."
            )

        for item in native_items:

            native_override[
                item["pool_id"]
            ] = (
                "qwen"
                if item["pool_id"]
                in chosen
                else specialist
            )

        print(
            "native split:",
            native_task,
            "qwen =",
            qwen_n,
            specialist,
            "=",
            n - qwen_n,
        )

    first_by_task = {}

    for item in tasks:
        first_by_task.setdefault(
            item["task_type"],
            item,
        )

    print(
        "===== PAIR PROFILE ====="
    )

    print(
        "models:",
        models
    )

    print(
        "routing:",
        routing
    )

    print(
        "requests:",
        len(tasks)
    )

    # --------------------------------------
    # Workers
    # --------------------------------------

    events = queue.Queue()

    shared_models = [
        m
        for m in models
        if m != "qwen"
    ]

    shared_workers = {
        name:
            SharedWorker(
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

    results = []

    try:

        # ----------------------------------
        # Start
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
        # Warm each used path
        # ----------------------------------

        print()
        print(
            "===== WARMUP ====="
        )

        warm_id = -1

        for task, target in (
            routing.items()
        ):

            item = first_by_task[
                task
            ]

            if target == "qwen":

                qwen_warm_one(
                    qwen,
                    item,
                    warm_id,
                    args.timeout_s,
                )

            else:

                shared_warm_one(
                    target,
                    item,
                    warm_id,
                    shared_workers,
                    events,
                    args.timeout_s,
                )

            warm_id -= 1

        # If a specialist native task is split,
        # warm the extra Qwen execution path.
        if (
            native_override
            and args.qwen_native_count > 0
        ):

            specialist = (
                args.model_b
                if args.model_a == "qwen"
                else args.model_a
            )

            native_task = (
                SPECIALIST_TASK[
                    specialist
                ]
            )

            item = first_by_task[
                native_task
            ]

            print(
                "Warmup extra path: "
                f"qwen -> {native_task}"
            )

            qwen_warm_one(
                qwen,
                item,
                warm_id,
                args.timeout_s,
            )

            warm_id -= 1

        # If ASR is split between Qwen and Whisper,
        # warm the additional Qwen-ASR execution path.
        if (
            set(models)
            == {"qwen", "whisper"}
            and args.qwen_asr_count > 0
        ):
            item = first_by_task[
                "asr"
            ]

            print(
                "Warmup extra path: "
                "qwen -> asr"
            )

            qwen_warm_one(
                qwen,
                item,
                warm_id,
                args.timeout_s,
            )

            warm_id -= 1

        # ----------------------------------
        # Submit
        # ----------------------------------

        print()
        print(
            "===== BENCHMARK ====="
        )

        benchmark_start_ns = (
            time.time_ns()
        )

        for request_id, item in enumerate(
            tasks
        ):

            target = routing[
                item["task_type"]
            ]

            if (
                item["task_type"] == "asr"
                and item["pool_id"]
                in asr_override
            ):
                target = asr_override[
                    item["pool_id"]
                ]


            if (
                item["pool_id"]
                in native_override
            ):
                target = (
                    native_override[
                        item["pool_id"]
                    ]
                )

            req = {
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
            }

            if target == "qwen":
                qwen.send(req)

            else:
                shared_workers[
                    target
                ].send(req)

        # ----------------------------------
        # Collect
        # ----------------------------------

        completed = set()

        deadline = (
            time.time()
            + args.timeout_s
        )

        while len(completed) < len(tasks):

            if time.time() >= deadline:
                raise TimeoutError(
                    "Pair profile timed out."
                )

            progress = False

            try:
                (
                    worker,
                    kind,
                    payload,
                ) = events.get_nowait()

                progress = True

                if kind == "RESULT":

                    rid = int(
                        payload[
                            "request_id"
                        ]
                    )

                    if rid >= 0:

                        payload[
                            "assigned_worker"
                        ] = worker

                        results.append(
                            payload
                        )

                        completed.add(rid)

                elif kind == "ERROR":
                    raise RuntimeError(
                        payload
                    )

            except queue.Empty:
                pass

            if qwen is not None:

                try:
                    kind, payload = (
                        qwen.events
                        .get_nowait()
                    )

                    progress = True

                    if kind == "RESULT":

                        rid = int(
                            payload[
                                "request_id"
                            ]
                        )

                        if rid >= 0:

                            payload[
                                "assigned_worker"
                            ] = "qwen"

                            results.append(
                                payload
                            )

                            completed.add(rid)

                    elif kind == "ERROR":
                        raise RuntimeError(
                            payload
                        )

                except queue.Empty:
                    pass

            for name, worker in (
                shared_workers.items()
            ):
                if (
                    worker.process
                    is not None
                    and worker.process.poll()
                    is not None
                    and len(completed)
                    < len(tasks)
                ):
                    raise RuntimeError(
                        f"{name} exited."
                    )

            if (
                qwen is not None
                and qwen.process is not None
                and qwen.process.poll()
                is not None
                and len(completed)
                < len(tasks)
            ):
                raise RuntimeError(
                    "qwen exited."
                )

            if not progress:
                time.sleep(0.002)

        benchmark_end_ns = (
            time.time_ns()
        )

        # ----------------------------------
        # Save
        # ----------------------------------

        results.sort(
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

            for row in results:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        grouped = defaultdict(list)

        for row in results:
            grouped[
                (
                    row["assigned_worker"],
                    row["task_type"],
                )
            ].append(
                float(
                    row[
                        "worker_latency_ms"
                    ]
                )
            )

        summary = {
            "models":
                models,

            "routing":
                routing,

            "num_results":
                len(results),

            "wall_time_s":
                (
                    benchmark_end_ns
                    - benchmark_start_ns
                ) / 1e9,

            "paths": {},
        }

        print()
        print(
            "===== PAIR PROFILE RESULT ====="
        )

        for (
            worker,
            task
        ), values in sorted(
            grouped.items()
        ):

            key = (
                f"{worker}:{task}"
            )

            x = {
                "count":
                    len(values),

                "mean_worker_latency_ms":
                    mean(values),
            }

            summary[
                "paths"
            ][key] = x

            print(
                f"{worker:<12}"
                f"{task:<8}"
                f"N={len(values):>3}  "
                f"mean="
                f"{mean(values):>9.3f} ms"
            )

        summary_path = Path(
            str(output)
            + ".summary.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print(
            "wall_time_s =",
            summary["wall_time_s"]
        )

        print(
            "Saved:",
            output
        )

        print(
            "Summary:",
            summary_path
        )

    finally:

        for worker in (
            shared_workers.values()
        ):
            try:
                worker.shutdown()
            except Exception:
                pass

        if qwen is not None:
            try:
                qwen.shutdown()
            except Exception:
                pass

        time.sleep(1)

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
