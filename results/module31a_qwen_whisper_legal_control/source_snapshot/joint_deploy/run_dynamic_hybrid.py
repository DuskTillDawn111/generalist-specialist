import argparse
import json
import queue
import time
from collections import Counter, defaultdict
from pathlib import Path

from run_static_hybrid import (
    ROOT,
    WORKERS,
    SPECIALIST,
    Worker,
    read_jsonl,
)


# --------------------------------------------------
# Formal baseline E2E service-time prior
# Unit: ms
# --------------------------------------------------

SERVICE_MS = {
    "distilbert": {
        "text": 1.731,
    },

    "resnet50": {
        "image": 4.605,
    },

    "paddleocr": {
        "ocr": 4.849,
    },

    "whisper": {
        "asr": 122.748,
    },

    "phi4": {
        "text": 53.680,
        "image": 201.265,
        "ocr": 170.076,
        "asr": 569.887,
    },
}


TASK_ORDER = [
    "text",
    "image",
    "ocr",
    "asr",
]


def percentile(values, q):
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    pos = (
        (len(values) - 1)
        * q
        / 100.0
    )

    lo = int(pos)
    hi = min(
        lo + 1,
        len(values) - 1
    )

    frac = pos - lo

    return (
        values[lo] * (1.0 - frac)
        + values[hi] * frac
    )


def wait_ready(
    name,
    workers,
    events,
    timeout_s
):
    deadline = (
        time.time()
        + timeout_s
    )

    while True:
        remaining = (
            deadline
            - time.time()
        )

        if remaining <= 0:
            raise TimeoutError(
                f"{name} READY timeout"
            )

        if (
            workers[name]
            .process
            .poll()
            is not None
        ):
            raise RuntimeError(
                f"{name} exited before READY "
                f"with return code "
                f"{workers[name].process.returncode}"
            )

        try:
            worker_name, kind, payload = (
                events.get(
                    timeout=min(
                        remaining,
                        1.0
                    )
                )
            )

        except queue.Empty:
            continue

        if (
            worker_name == name
            and kind == "READY"
        ):
            return

        if (
            worker_name == name
            and kind == "ERROR"
        ):
            raise RuntimeError(
                payload
            )


def warm_one(
    worker_name,
    item,
    request_id,
    workers,
    events,
    timeout_s
):
    request = {
        "request_id":
            request_id,

        "pool_id":
            "warmup:"
            + item["pool_id"],

        "arrival_index":
            -1,

        "task_type":
            item["task_type"],

        "sample":
            item["sample"],

        "submit_ns":
            time.time_ns(),
    }

    workers[
        worker_name
    ].send(
        request
    )

    deadline = (
        time.time()
        + timeout_s
    )

    while True:
        remaining = (
            deadline
            - time.time()
        )

        if remaining <= 0:
            raise TimeoutError(
                f"Warmup timeout: "
                f"{worker_name}"
            )

        worker, kind, payload = (
            events.get(
                timeout=remaining
            )
        )

        if (
            worker == worker_name
            and kind == "RESULT"
            and payload[
                "request_id"
            ] == request_id
        ):
            print(
                f"Warmup "
                f"{worker_name:12s} "
                f"{item['task_type']:>5s}: "
                f"OK"
            )
            return

        if (
            worker == worker_name
            and kind == "ERROR"
        ):
            raise RuntimeError(
                f"Warmup failed: "
                f"{payload}"
            )


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
        default=600.0
    )

    args = parser.parse_args()

    tasks = read_jsonl(
        args.pool
    )

    if not tasks:
        raise RuntimeError(
            "Task pool is empty."
        )

    first_by_task = {}

    for item in tasks:
        first_by_task.setdefault(
            item["task_type"],
            item
        )

    for task_type in TASK_ORDER:
        if task_type not in first_by_task:
            raise RuntimeError(
                f"Pool missing task: "
                f"{task_type}"
            )

    events = queue.Queue()

    workers = {
        name:
            Worker(
                name,
                config,
                events
            )
        for name, config
        in WORKERS.items()
    }

    startup_order = [
        "distilbert",
        "resnet50",
        "whisper",
        "paddleocr",
        "phi4",
    ]

    try:

        # ==========================================
        # Start all resident models
        # ==========================================

        print(
            "===== STARTING WORKERS ====="
        )

        for name in startup_order:
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
            "ALL 5 MODELS ARE RESIDENT."
        )

        # ==========================================
        # Warm up specialists
        # ==========================================

        print()
        print(
            "===== WARMUP ====="
        )

        warmup_id = -1

        for task_type in TASK_ORDER:
            target = (
                SPECIALIST[
                    task_type
                ]
            )

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

        # ==========================================
        # Warm up Phi on all 4 modalities
        # ==========================================

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
            "ALL MODEL/TASK PATHS WARM."
        )

        # ==========================================
        # Dynamic list scheduling
        #
        # virtual_work_ms means predicted remaining
        # service work assigned to that worker.
        # ==========================================

        virtual_work_ms = defaultdict(
            float
        )

        routing = Counter()

        routing_by_task = {
            task:
                Counter()
            for task in TASK_ORDER
        }

        assignments = []

        start_ns = (
            time.time_ns()
        )

        for request_id, item in enumerate(
            tasks
        ):
            task_type = (
                item[
                    "task_type"
                ]
            )

            specialist = (
                SPECIALIST[
                    task_type
                ]
            )

            specialist_service = (
                SERVICE_MS[
                    specialist
                ][
                    task_type
                ]
            )

            phi_service = (
                SERVICE_MS[
                    "phi4"
                ][
                    task_type
                ]
            )

            specialist_eta = (
                virtual_work_ms[
                    specialist
                ]
                + specialist_service
            )

            phi_eta = (
                virtual_work_ms[
                    "phi4"
                ]
                + phi_service
            )

            if (
                specialist_eta
                <= phi_eta
            ):
                target = specialist
                predicted_finish = (
                    specialist_eta
                )
                service_ms = (
                    specialist_service
                )

            else:
                target = "phi4"
                predicted_finish = (
                    phi_eta
                )
                service_ms = (
                    phi_service
                )

            virtual_work_ms[
                target
            ] += service_ms

            routing[
                target
            ] += 1

            routing_by_task[
                task_type
            ][
                target
            ] += 1

            assignments.append(
                {
                    "request_id":
                        request_id,

                    "pool_id":
                        item["pool_id"],

                    "task_type":
                        task_type,

                    "target":
                        target,

                    "predicted_finish_ms":
                        predicted_finish,

                    "predicted_service_ms":
                        service_ms,
                }
            )

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
                    task_type,

                "sample":
                    item["sample"],

                "submit_ns":
                    time.time_ns(),
            }

            workers[
                target
            ].send(
                request
            )

        print()
        print(
            "===== DYNAMIC ROUTING ====="
        )

        for task_type in TASK_ORDER:
            print(
                f"{task_type:>5}: "
                f"{dict(routing_by_task[task_type])}"
            )

        print()
        print(
            "Total:",
            dict(routing)
        )

        # ==========================================
        # Collect
        # ==========================================

        expected = len(
            tasks
        )

        results = []
        errors = []

        deadline = (
            time.time()
            + args.timeout_s
        )

        while (
            len(results)
            + len(errors)
            < expected
        ):
            remaining = (
                deadline
                - time.time()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "Result collection timeout"
                )

            worker_name, kind, payload = (
                events.get(
                    timeout=remaining
                )
            )

            if kind == "RESULT":
                results.append(
                    payload
                )

                print(
                    f"["
                    f"{len(results):03d}/"
                    f"{expected:03d}"
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

                print(
                    "ERROR:",
                    worker_name,
                    payload.get(
                        "error"
                    )
                )

        end_ns = (
            time.time_ns()
        )

        results.sort(
            key=lambda x:
                x["arrival_index"]
        )

        # ==========================================
        # Save raw results
        # ==========================================

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

        assignment_path = (
            output.with_name(
                output.stem
                + ".assignments.jsonl"
            )
        )

        with open(
            assignment_path,
            "w",
            encoding="utf-8"
        ) as f:

            for row in assignments:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        # ==========================================
        # Per-task summary
        # ==========================================

        per_task = {}

        for task_type in TASK_ORDER:

            rows = [
                r
                for r in results
                if r[
                    "task_type"
                ] == task_type
            ]

            worker_latencies = [
                float(
                    r[
                        "worker_latency_ms"
                    ]
                )
                for r in rows
            ]

            queue_waits = [
                float(
                    r.get(
                        "queue_wait_ms",
                        0
                    )
                )
                for r in rows
            ]

            task_summary = {
                "count":
                    len(rows),

                "routing":
                    dict(
                        Counter(
                            r["worker"]
                            for r in rows
                        )
                    ),

                "worker_latency_mean_ms":
                    (
                        sum(
                            worker_latencies
                        )
                        / len(
                            worker_latencies
                        )
                        if worker_latencies
                        else None
                    ),

                "worker_latency_p95_ms":
                    percentile(
                        worker_latencies,
                        95
                    ),

                "queue_wait_mean_ms":
                    (
                        sum(
                            queue_waits
                        )
                        / len(
                            queue_waits
                        )
                        if queue_waits
                        else None
                    ),

                "queue_wait_p95_ms":
                    percentile(
                        queue_waits,
                        95
                    ),
            }

            if task_type != "asr":

                correctness = [
                    r["correct"]
                    for r in rows
                    if "correct" in r
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
                    r.get(
                        "word_errors",
                        0
                    )
                    for r in rows
                )

                reference_words = sum(
                    r.get(
                        "reference_words",
                        0
                    )
                    for r in rows
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
            end_ns
            - start_ns
        ) / 1e9

        summary = {
            "policy":
                "predicted_load_balance_v1",

            "tasks":
                len(tasks),

            "results":
                len(results),

            "errors":
                len(errors),

            "routing":
                dict(routing),

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

            "service_time_prior_ms":
                SERVICE_MS,
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
            "Assignments:",
            assignment_path
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
