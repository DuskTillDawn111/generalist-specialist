import argparse
import json
import queue
import time
from pathlib import Path
from statistics import mean

from run_static_hybrid import (
    WORKERS,
    SPECIALIST,
    Worker,
    read_jsonl,
)

from run_dynamic_hybrid import (
    wait_ready,
    warm_one,
)


def percentile(xs, p):
    xs = sorted(xs)

    if not xs:
        return None

    if len(xs) == 1:
        return xs[0]

    k = (
        (len(xs) - 1)
        * p / 100.0
    )

    lo = int(k)
    hi = min(
        lo + 1,
        len(xs) - 1
    )

    frac = k - lo

    return (
        xs[lo] * (1 - frac)
        + xs[hi] * frac
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pool",
        required=True,
    )

    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "text",
            "image",
            "ocr",
            "asr",
        ],
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

    all_tasks = read_jsonl(
        args.pool
    )

    tasks = [
        item
        for item in all_tasks
        if item["task_type"]
        == args.task
    ]

    if not tasks:
        raise RuntimeError(
            f"No {args.task} tasks found."
        )

    worker_name = SPECIALIST[
        args.task
    ]

    events = queue.Queue()

    workers = {
        worker_name:
            Worker(
                worker_name,
                WORKERS[
                    worker_name
                ],
                events,
            )
    }

    worker = workers[
        worker_name
    ]

    results = []

    try:

        print(
            "===== SINGLE SPECIALIST ====="
        )

        print(
            "task:",
            args.task
        )

        print(
            "worker:",
            worker_name
        )

        print(
            "samples:",
            len(tasks)
        )

        print()

        print(
            f"Starting {worker_name}..."
        )

        worker.start()

        wait_ready(
            worker_name,
            workers,
            events,
            args.timeout_s,
        )

        print(
            f"{worker_name}: READY"
        )

        print()
        print(
            "===== WARMUP ====="
        )

        warm_one(
            worker_name,
            tasks[0],
            -1,
            workers,
            events,
            args.timeout_s,
        )

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

            request = {
                "request_id":
                    request_id,

                "pool_id":
                    item["pool_id"],

                "arrival_index":
                    item[
                        "arrival_index"
                    ],

                "task_type":
                    item[
                        "task_type"
                    ],

                "sample":
                    item["sample"],

                "submit_ns":
                    time.time_ns(),
            }

            worker.send(
                request
            )

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
                    "Single specialist run "
                    "timed out."
                )

            try:
                (
                    name,
                    kind,
                    payload,
                ) = events.get(
                    timeout=min(
                        remaining,
                        1.0,
                    )
                )

            except queue.Empty:

                if (
                    worker.process
                    is not None
                    and worker.process.poll()
                    is not None
                ):
                    raise RuntimeError(
                        f"{worker_name} exited."
                    )

                continue

            if kind == "RESULT":

                request_id = int(
                    payload[
                        "request_id"
                    ]
                )

                if request_id < 0:
                    continue

                payload[
                    "assigned_worker"
                ] = worker_name

                results.append(
                    payload
                )

                completed += 1

            elif kind == "ERROR":
                raise RuntimeError(
                    payload
                )

        benchmark_end_ns = (
            time.time_ns()
        )

        wall_time_s = (
            benchmark_end_ns
            - benchmark_start_ns
        ) / 1e9

        results.sort(
            key=lambda r:
                int(
                    r["request_id"]
                )
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

        latencies = [
            float(
                r["worker_latency_ms"]
            )
            for r in results
        ]

        summary = {
            "task":
                args.task,

            "worker":
                worker_name,

            "num_results":
                len(results),

            "wall_time_s":
                wall_time_s,

            "throughput_tasks_per_s":
                len(results)
                / wall_time_s,

            "mean_worker_latency_ms":
                mean(latencies),

            "p95_worker_latency_ms":
                percentile(
                    latencies,
                    95,
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
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print(
            "===== SINGLE SPECIALIST RESULT ====="
        )

        print(
            "worker =",
            worker_name
        )

        print(
            "task =",
            args.task
        )

        print(
            "results =",
            len(results)
        )

        print(
            "wall_time_s =",
            wall_time_s
        )

        print(
            "throughput =",
            summary[
                "throughput_tasks_per_s"
            ]
        )

        print(
            "mean_worker_latency_ms =",
            summary[
                "mean_worker_latency_ms"
            ]
        )

        print(
            "p95_worker_latency_ms =",
            summary[
                "p95_worker_latency_ms"
            ]
        )

    finally:

        try:
            worker.shutdown()
        except Exception:
            pass

        time.sleep(1.0)

        try:
            worker.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
