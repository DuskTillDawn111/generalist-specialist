import argparse
import copy
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

from run_dynamic_hybrid import (
    SERVICE_MS,
    TASK_ORDER,
    percentile,
    wait_ready,
    warm_one,
)


def mean(values):
    if not values:
        return None

    return (
        sum(values)
        / len(values)
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
        "--alpha",
        type=float,
        default=0.25,
        help=(
            "EWMA update coefficient. "
            "new=(1-alpha)*old+alpha*actual"
        )
    )

    parser.add_argument(
        "--max-outstanding",
        type=int,
        default=32,
        help=(
            "Maximum number of admitted "
            "but unfinished requests."
        )
    )

    parser.add_argument(
        "--timeout-s",
        type=float,
        default=3600.0
    )

    args = parser.parse_args()

    if not (
        0.0
        < args.alpha
        <= 1.0
    ):
        raise ValueError(
            "--alpha must be in (0, 1]"
        )

    if args.max_outstanding <= 0:
        raise ValueError(
            "--max-outstanding must be > 0"
        )

    tasks = read_jsonl(
        args.pool
    )

    if not tasks:
        raise RuntimeError(
            "Task pool is empty."
        )

    # ---------------------------------------
    # Warmup samples
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

    # ---------------------------------------
    # Mutable online service-time estimates
    # ---------------------------------------

    service_estimate = copy.deepcopy(
        SERVICE_MS
    )

    initial_service_estimate = (
        copy.deepcopy(
            SERVICE_MS
        )
    )

    # Estimated outstanding service work
    # already assigned to each worker.
    remaining_work_ms = defaultdict(
        float
    )

    routing = Counter()

    routing_by_task = {
        task:
            Counter()
        for task in TASK_ORDER
    }

    results = []
    errors = []
    assignments = []
    updates = []

    # request_id -> assignment metadata
    outstanding = {}

    next_request_id = 0

    try:

        # =======================================
        # 1. Start resident workers
        # =======================================

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

        # =======================================
        # 2. Same warmup as previous experiments
        # =======================================

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

        print()
        print(
            "===== ONLINE ADAPTIVE CONFIG ====="
        )

        print(
            "alpha:",
            args.alpha
        )

        print(
            "max outstanding:",
            args.max_outstanding
        )

        # =======================================
        # 3. Online scheduling
        # =======================================

        benchmark_start_ns = (
            time.time_ns()
        )

        deadline = (
            time.time()
            + args.timeout_s
        )

        def choose_worker(
            item
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
                service_estimate[
                    specialist
                ][
                    task_type
                ]
            )

            phi_service = (
                service_estimate[
                    "phi4"
                ][
                    task_type
                ]
            )

            specialist_eta = (
                remaining_work_ms[
                    specialist
                ]
                + specialist_service
            )

            phi_eta = (
                remaining_work_ms[
                    "phi4"
                ]
                + phi_service
            )

            # Tie -> specialist.
            if (
                specialist_eta
                <= phi_eta
            ):
                return (
                    specialist,
                    specialist_service,
                    specialist_eta,
                    phi_eta,
                )

            return (
                "phi4",
                phi_service,
                specialist_eta,
                phi_eta,
            )

        def admit_one(
            request_id
        ):
            item = tasks[
                request_id
            ]

            task_type = (
                item[
                    "task_type"
                ]
            )

            (
                target,
                predicted_service_ms,
                specialist_eta_ms,
                phi_eta_ms,
            ) = choose_worker(
                item
            )

            specialist = (
                SPECIALIST[
                    task_type
                ]
            )

            admission_ns = (
                time.time_ns()
            )

            remaining_before_ms = (
                remaining_work_ms[
                    target
                ]
            )

            estimate_before_ms = (
                service_estimate[
                    target
                ][
                    task_type
                ]
            )

            remaining_work_ms[
                target
            ] += (
                predicted_service_ms
            )

            routing[
                target
            ] += 1

            routing_by_task[
                task_type
            ][
                target
            ] += 1

            record = {
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

                "target":
                    target,

                "specialist":
                    specialist,

                "predicted_service_ms":
                    predicted_service_ms,

                "estimate_before_ms":
                    estimate_before_ms,

                "remaining_work_before_ms":
                    remaining_before_ms,

                "specialist_eta_ms":
                    specialist_eta_ms,

                "phi_eta_ms":
                    phi_eta_ms,

                "admission_ns":
                    admission_ns,

                "batch_admission_wait_ms":
                    (
                        admission_ns
                        - benchmark_start_ns
                    ) / 1e6,
            }

            assignments.append(
                record
            )

            outstanding[
                request_id
            ] = record

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

                # Worker-side queue waiting
                # starts only after admission.
                "submit_ns":
                    admission_ns,
            }

            workers[
                target
            ].send(
                request
            )

        # ---------------------------------------
        # Initial admission window
        # ---------------------------------------

        while (
            next_request_id
            < len(tasks)
            and len(outstanding)
            < args.max_outstanding
        ):
            admit_one(
                next_request_id
            )

            next_request_id += 1

        print()
        print(
            "Initial admitted:",
            len(outstanding)
        )

        # ---------------------------------------
        # Completion-driven scheduling loop
        # ---------------------------------------

        completed = 0

        while (
            completed
            < len(tasks)
        ):
            remaining_timeout = (
                deadline
                - time.time()
            )

            if remaining_timeout <= 0:
                raise TimeoutError(
                    "Online scheduling timeout"
                )

            try:
                (
                    worker_name,
                    kind,
                    payload
                ) = events.get(
                    timeout=min(
                        remaining_timeout,
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
                        and any(
                            meta[
                                "target"
                            ]
                            == name
                            for meta
                            in outstanding.values()
                        )
                    ):
                        raise RuntimeError(
                            f"Worker {name} "
                            f"exited while requests "
                            f"were outstanding."
                        )

                continue

            if kind not in (
                "RESULT",
                "ERROR",
            ):
                continue

            request_id = (
                payload.get(
                    "request_id"
                )
            )

            if (
                request_id
                not in outstanding
            ):
                continue

            meta = outstanding.pop(
                request_id
            )

            task_type = (
                meta[
                    "task_type"
                ]
            )

            target = (
                meta[
                    "target"
                ]
            )

            predicted_service_ms = (
                meta[
                    "predicted_service_ms"
                ]
            )

            # Request no longer contributes
            # predicted work to this worker.
            remaining_work_ms[
                target
            ] = max(
                0.0,
                remaining_work_ms[
                    target
                ]
                - predicted_service_ms
            )

            if kind == "RESULT":

                actual_service_ms = float(
                    payload[
                        "worker_latency_ms"
                    ]
                )

                old_estimate_ms = float(
                    service_estimate[
                        target
                    ][
                        task_type
                    ]
                )

                new_estimate_ms = (
                    (
                        1.0
                        - args.alpha
                    )
                    * old_estimate_ms
                    + args.alpha
                    * actual_service_ms
                )

                service_estimate[
                    target
                ][
                    task_type
                ] = (
                    new_estimate_ms
                )

                completion_ns = int(
                    payload[
                        "end_ns"
                    ]
                )

                batch_admission_wait_ms = (
                    meta[
                        "batch_admission_wait_ms"
                    ]
                )

                worker_queue_wait_ms = float(
                    payload.get(
                        "queue_wait_ms",
                        0.0
                    )
                )

                payload[
                    "batch_admission_wait_ms"
                ] = (
                    batch_admission_wait_ms
                )

                payload[
                    "worker_queue_wait_ms"
                ] = (
                    worker_queue_wait_ms
                )

                payload[
                    "total_wait_from_batch_start_ms"
                ] = (
                    batch_admission_wait_ms
                    + worker_queue_wait_ms
                )

                payload[
                    "batch_response_time_ms"
                ] = (
                    completion_ns
                    - benchmark_start_ns
                ) / 1e6

                payload[
                    "predicted_service_ms"
                ] = (
                    predicted_service_ms
                )

                payload[
                    "estimate_before_ms"
                ] = (
                    old_estimate_ms
                )

                payload[
                    "estimate_after_ms"
                ] = (
                    new_estimate_ms
                )

                results.append(
                    payload
                )

                updates.append(
                    {
                        "request_id":
                            request_id,

                        "task_type":
                            task_type,

                        "worker":
                            target,

                        "actual_service_ms":
                            actual_service_ms,

                        "old_estimate_ms":
                            old_estimate_ms,

                        "new_estimate_ms":
                            new_estimate_ms,

                        "alpha":
                            args.alpha,

                        "remaining_work_after_ms":
                            remaining_work_ms[
                                target
                            ],
                    }
                )

                completed += 1

                print(
                    f"["
                    f"{completed:03d}/"
                    f"{len(tasks):03d}"
                    f"] "
                    f"{task_type:>5} "
                    f"-> "
                    f"{target:<10} "
                    f"run="
                    f"{actual_service_ms:.2f} "
                    f"ms "
                    f"worker_wait="
                    f"{worker_queue_wait_ms:.2f} "
                    f"ms "
                    f"est="
                    f"{old_estimate_ms:.1f}"
                    f"->"
                    f"{new_estimate_ms:.1f}"
                    f" ms"
                )

            else:

                errors.append(
                    payload
                )

                completed += 1

                print(
                    f"ERROR "
                    f"{request_id}: "
                    f"{target} "
                    f"{payload.get('error')}"
                )

            # -----------------------------------
            # Admission control:
            # one completion -> refill window
            # -----------------------------------

            while (
                next_request_id
                < len(tasks)
                and len(outstanding)
                < args.max_outstanding
            ):
                admit_one(
                    next_request_id
                )

                next_request_id += 1

        benchmark_end_ns = (
            time.time_ns()
        )

        # =======================================
        # 4. Save raw outputs
        # =======================================

        results.sort(
            key=lambda x:
                x[
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

        update_path = (
            output.with_name(
                output.stem
                + ".updates.jsonl"
            )
        )

        with open(
            update_path,
            "w",
            encoding="utf-8"
        ) as f:

            for row in updates:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        # =======================================
        # 5. Summary
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

            worker_latencies = [
                float(
                    row[
                        "worker_latency_ms"
                    ]
                )
                for row in rows
            ]

            worker_waits = [
                float(
                    row.get(
                        "worker_queue_wait_ms",
                        0.0
                    )
                )
                for row in rows
            ]

            admission_waits = [
                float(
                    row.get(
                        "batch_admission_wait_ms",
                        0.0
                    )
                )
                for row in rows
            ]

            response_times = [
                float(
                    row.get(
                        "batch_response_time_ms",
                        0.0
                    )
                )
                for row in rows
            ]

            summary = {
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
                        worker_latencies
                    ),

                "worker_latency_p95_ms":
                    percentile(
                        worker_latencies,
                        95
                    ),

                "worker_queue_wait_mean_ms":
                    mean(
                        worker_waits
                    ),

                "worker_queue_wait_p95_ms":
                    percentile(
                        worker_waits,
                        95
                    ),

                "batch_admission_wait_mean_ms":
                    mean(
                        admission_waits
                    ),

                "batch_admission_wait_p95_ms":
                    percentile(
                        admission_waits,
                        95
                    ),

                "batch_response_time_mean_ms":
                    mean(
                        response_times
                    ),

                "batch_response_time_p95_ms":
                    percentile(
                        response_times,
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

                summary[
                    "accuracy"
                ] = (
                    (
                        sum(
                            correctness
                        )
                        / len(
                            correctness
                        )
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

                summary[
                    "corpus_wer"
                ] = (
                    (
                        word_errors
                        / reference_words
                    )
                    if reference_words
                    else None
                )

            per_task[
                task_type
            ] = summary

        wall_time_s = (
            benchmark_end_ns
            - benchmark_start_ns
        ) / 1e9

        final_summary = {
            "policy":
                "online_adaptive_v2",

            "alpha":
                args.alpha,

            "max_outstanding":
                args.max_outstanding,

            "tasks":
                len(tasks),

            "results":
                len(results),

            "errors":
                len(errors),

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

            "initial_service_estimate_ms":
                initial_service_estimate,

            "final_service_estimate_ms":
                service_estimate,

            "num_estimate_updates":
                len(updates),
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
                final_summary,
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
                final_summary,
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
            "Updates:",
            update_path
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
