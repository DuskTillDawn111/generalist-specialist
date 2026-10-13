import argparse
import json
import queue
import time
from collections import Counter
from pathlib import Path

from run_static_hybrid import (
    ROOT,
    WORKERS,
    SPECIALIST,
    Worker,
    read_jsonl,
)

from run_dynamic_hybrid import (
    TASK_ORDER,
    percentile,
    wait_ready,
    warm_one,
)


SPECIALIST_WORKERS = [
    "distilbert",
    "resnet50",
    "whisper",
    "paddleocr",
]


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pool",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--timeout-s",
        type=float,
        default=3600.0
    )

    args = parser.parse_args()

    tasks = read_jsonl(
        args.pool
    )

    if not tasks:
        raise RuntimeError(
            "Task pool is empty."
        )

    # ---------------------------------------
    # Find one warmup sample per task
    # ---------------------------------------

    first_by_task = {}

    for item in tasks:
        first_by_task.setdefault(
            item["task_type"],
            item
        )

    for task_type in TASK_ORDER:
        if task_type not in first_by_task:
            raise RuntimeError(
                f"Pool missing task: {task_type}"
            )

    events = queue.Queue()

    workers = {
        name: Worker(
            name,
            WORKERS[name],
            events
        )
        for name in SPECIALIST_WORKERS
    }

    results = []
    errors = []

    try:

        # =======================================
        # 1. Start 4 specialist workers
        # =======================================

        print(
            "===== STARTING SPECIALISTS ====="
        )

        for name in SPECIALIST_WORKERS:

            print(
                f"Starting {name}..."
            )

            workers[
                name
            ].start()

            wait_ready(
                name,
                workers,
                events,
                args.timeout_s
            )

            print(
                f"{name}: READY"
            )

        print()
        print(
            "ALL 4 SPECIALIST MODELS ARE RESIDENT."
        )

        # =======================================
        # 2. Warmup
        # =======================================

        print()
        print(
            "===== WARMUP ====="
        )

        warmup_id = -1

        for task_type in TASK_ORDER:

            target = SPECIALIST[
                task_type
            ]

            warm_one(
                target,
                first_by_task[
                    task_type
                ],
                warmup_id,
                workers,
                events,
                args.timeout_s
            )

            warmup_id -= 1

        print()
        print(
            "ALL SPECIALIST PATHS WARM."
        )

        # =======================================
        # 3. Fixed specialist routing
        # =======================================

        routing = Counter()

        routing_by_task = {
            task_type: Counter()
            for task_type
            in TASK_ORDER
        }

        print()
        print(
            "===== PURE SPECIALIST ROUTING ====="
        )

        for task_type in TASK_ORDER:
            print(
                f"{task_type:>5} -> "
                f"{SPECIALIST[task_type]}"
            )

        # Benchmark begins after warmup.
        benchmark_start_ns = (
            time.time_ns()
        )

        # ---------------------------------------
        # Dispatch all 800 requests
        # ---------------------------------------

        for request_id, item in enumerate(
            tasks
        ):

            task_type = item[
                "task_type"
            ]

            target = SPECIALIST[
                task_type
            ]

            submit_ns = time.time_ns()

            request = {
                "request_id":
                    request_id,

                "pool_id":
                    item[
                        "pool_id"
                    ],

                "arrival_index":
                    item[
                        "arrival_index"
                    ],

                "task_type":
                    task_type,

                "sample":
                    item[
                        "sample"
                    ],

                "submit_ns":
                    submit_ns,
            }

            routing[
                target
            ] += 1

            routing_by_task[
                task_type
            ][
                target
            ] += 1

            workers[
                target
            ].send(
                request
            )

        print()
        print(
            "Submitted:",
            len(tasks)
        )

        # =======================================
        # 4. Collect all results
        # =======================================

        completed = 0

        deadline = (
            time.time()
            + args.timeout_s
        )

        while completed < len(tasks):

            remaining = (
                deadline
                - time.time()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "Pure-specialist run timed out."
                )

            try:
                (
                    worker_name,
                    kind,
                    payload
                ) = events.get(
                    timeout=min(
                        remaining,
                        1.0
                    )
                )

            except queue.Empty:

                for name, worker in (
                    workers.items()
                ):

                    if (
                        worker.process
                        is not None
                        and worker.process.poll()
                        is not None
                    ):
                        raise RuntimeError(
                            f"Worker {name} exited."
                        )

                continue

            if kind == "RESULT":

                results.append(
                    payload
                )

                completed += 1

                print(
                    f"["
                    f"{completed:03d}/"
                    f"{len(tasks):03d}"
                    f"] "
                    f"{payload['task_type']:>5} "
                    f"-> "
                    f"{worker_name:<10} "
                    f"run="
                    f"{payload['worker_latency_ms']:.2f} "
                    f"ms "
                    f"wait="
                    f"{payload.get('queue_wait_ms', 0):.2f} "
                    f"ms"
                )

            elif kind == "ERROR":

                errors.append(
                    payload
                )

                completed += 1

                print(
                    "ERROR:",
                    payload
                )

        benchmark_end_ns = (
            time.time_ns()
        )

        # =======================================
        # 5. Save results
        # =======================================

        results.sort(
            key=lambda row:
                row[
                    "arrival_index"
                ]
        )

        output = Path(
            args.output
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output,
            "w",
            encoding="utf-8"
        ) as f:

            for row in results:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        # =======================================
        # 6. Per-task metrics
        # =======================================

        per_task = {}

        for task_type in TASK_ORDER:

            rows = [
                row
                for row in results
                if row[
                    "task_type"
                ] == task_type
            ]

            latencies = [
                float(
                    row[
                        "worker_latency_ms"
                    ]
                )
                for row in rows
            ]

            waits = [
                float(
                    row.get(
                        "queue_wait_ms",
                        0.0
                    )
                )
                for row in rows
            ]

            task_summary = {
                "count":
                    len(rows),

                "routing":
                    dict(
                        Counter(
                            row[
                                "worker"
                            ]
                            for row in rows
                        )
                    ),

                "worker_latency_mean_ms":
                    mean(
                        latencies
                    ),

                "worker_latency_p95_ms":
                    percentile(
                        latencies,
                        95
                    ),

                "queue_wait_mean_ms":
                    mean(
                        waits
                    ),

                "queue_wait_p95_ms":
                    percentile(
                        waits,
                        95
                    ),
            }

            if task_type != "asr":

                correctness = [
                    bool(
                        row[
                            "correct"
                        ]
                    )
                    for row in rows
                    if "correct" in row
                ]

                task_summary[
                    "accuracy"
                ] = (
                    sum(
                        correctness
                    )
                    / len(
                        correctness
                    )
                    if correctness
                    else None
                )

            else:

                word_errors = sum(
                    int(
                        row.get(
                            "word_errors",
                            0
                        )
                    )
                    for row in rows
                )

                reference_words = sum(
                    int(
                        row.get(
                            "reference_words",
                            0
                        )
                    )
                    for row in rows
                )

                task_summary[
                    "corpus_wer"
                ] = (
                    word_errors
                    / reference_words
                    if reference_words
                    else None
                )

            per_task[
                task_type
            ] = task_summary

        # =======================================
        # 7. Final summary
        # =======================================

        wall_time_s = (
            benchmark_end_ns
            - benchmark_start_ns
        ) / 1e9

        summary = {
            "policy":
                "pure_specialists",

            "tasks":
                len(tasks),

            "results":
                len(results),

            "errors":
                len(errors),

            "models": [
                "distilbert",
                "resnet50",
                "paddleocr",
                "whisper",
            ],

            "routing":
                dict(
                    routing
                ),

            "routing_by_task":
                {
                    task:
                        dict(
                            routing_by_task[
                                task
                            ]
                        )
                    for task in TASK_ORDER
                },

            "wall_time_s":
                wall_time_s,

            "throughput_tasks_per_s":
                (
                    len(results)
                    / wall_time_s
                    if wall_time_s > 0
                    else None
                ),

            "per_task":
                per_task,
        }

        summary_path = (
            output.with_suffix(
                ".summary.json"
            )
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                summary,
                f,
                ensure_ascii=False,
                indent=2
            )

        print()
        print(
            "===== SUMMARY ====="
        )

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2
            )
        )

        print()
        print(
            "Results:",
            output
        )

        print(
            "Summary:",
            summary_path
        )

    finally:

        for worker in (
            workers.values()
        ):
            worker.shutdown()

        time.sleep(
            1.0
        )

        for worker in (
            workers.values()
        ):
            worker.terminate()


if __name__ == "__main__":
    main()
