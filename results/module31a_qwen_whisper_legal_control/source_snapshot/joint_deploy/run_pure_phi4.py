import argparse
import json
import queue
import time
from collections import Counter
from pathlib import Path

from run_static_hybrid import (
    WORKERS,
    Worker,
    read_jsonl,
)

from run_dynamic_hybrid import (
    TASK_ORDER,
    percentile,
    wait_ready,
    warm_one,
)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


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

    first_by_task = {}

    for item in tasks:
        first_by_task.setdefault(
            item["task_type"],
            item
        )

    events = queue.Queue()

    workers = {
        "phi4": Worker(
            "phi4",
            WORKERS["phi4"],
            events
        )
    }

    results = []
    errors = []

    try:

        print(
            "===== STARTING PHI-4 ====="
        )

        workers[
            "phi4"
        ].start()

        wait_ready(
            "phi4",
            workers,
            events,
            args.timeout_s
        )

        print(
            "phi4: READY"
        )

        # -------------------------
        # Warm every modality
        # -------------------------

        print()
        print(
            "===== WARMUP ====="
        )

        warmup_id = -1

        for task_type in TASK_ORDER:

            warm_one(
                "phi4",
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
            "ALL PHI-4 TASK PATHS WARM."
        )

        print()
        print(
            "===== PURE GENERALIST ROUTING ====="
        )

        for task_type in TASK_ORDER:
            print(
                f"{task_type:>5} -> phi4"
            )

        benchmark_start_ns = (
            time.time_ns()
        )

        # -------------------------
        # Dispatch entire pool
        # -------------------------

        for request_id, item in enumerate(
            tasks
        ):

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
                    item[
                        "task_type"
                    ],

                "sample":
                    item[
                        "sample"
                    ],

                "submit_ns":
                    time.time_ns(),
            }

            workers[
                "phi4"
            ].send(
                request
            )

        print()
        print(
            "Submitted:",
            len(tasks)
        )

        deadline = (
            time.time()
            + args.timeout_s
        )

        completed = 0

        while completed < len(tasks):

            remaining = (
                deadline
                - time.time()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "Phi-only run timed out."
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

                worker = workers[
                    "phi4"
                ]

                if (
                    worker.process
                    is not None
                    and worker.process.poll()
                    is not None
                ):
                    raise RuntimeError(
                        "Phi-4 worker exited."
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
                    f"-> phi4 "
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

        # -------------------------
        # Per-task summary
        # -------------------------

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

                "routing": {
                    "phi4":
                        len(rows)
                },

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

        wall_time_s = (
            benchmark_end_ns
            - benchmark_start_ns
        ) / 1e9

        summary = {
            "policy":
                "pure_generalist_phi4",

            "tasks":
                len(tasks),

            "results":
                len(results),

            "errors":
                len(errors),

            "models": [
                "phi4"
            ],

            "routing": {
                "phi4":
                    len(tasks)
            },

            "routing_by_task": {
                task: {
                    "phi4":
                        sum(
                            1
                            for item in tasks
                            if item[
                                "task_type"
                            ] == task
                        )
                }
                for task
                in TASK_ORDER
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

        workers[
            "phi4"
        ].shutdown()

        time.sleep(
            1.0
        )

        workers[
            "phi4"
        ].terminate()


if __name__ == "__main__":
    main()
