import argparse
import json
import math
import queue
import time
from collections import Counter, defaultdict
from pathlib import Path

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


TASK_ORDER = [
    "text",
    "image",
    "ocr",
    "asr",
]

SUPPORTED_RUNTIME_MODELS = {
    "qwen",
    "distilbert",
    "resnet50",
    "paddleocr",
    "whisper",
}


def load_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def select_plan(
    placement,
    budget_gib,
    max_models,
):
    matches = []

    for row in placement["results"]:
        if (
            abs(
                float(row["budget_gib"])
                - float(budget_gib)
            )
            < 1e-9
            and int(row["max_models"])
            == int(max_models)
        ):
            matches.append(row)

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one placement row for "
            f"budget={budget_gib}, "
            f"max_models={max_models}; "
            f"found {len(matches)}"
        )

    row = matches[0]

    if row["best"] is None:
        raise RuntimeError(
            "Selected placement is infeasible."
        )

    return row


def allocate_integer_counts(
    total,
    fractions,
):
    """
    Convert LP routing fractions to exact integer
    request counts using largest remainder.
    """

    names = list(
        fractions.keys()
    )

    raw = {
        name:
            float(fractions[name])
            * total
        for name in names
    }

    counts = {
        name:
            int(math.floor(raw[name]))
        for name in names
    }

    remaining = (
        total
        - sum(counts.values())
    )

    order = sorted(
        names,
        key=lambda name: (
            raw[name] - counts[name],
            fractions[name],
            name,
        ),
        reverse=True,
    )

    for name in order[:remaining]:
        counts[name] += 1

    assert sum(counts.values()) == total

    return counts


def build_assignments(
    tasks,
    routing_plan,
):
    """
    Deterministically assign tasks using optimizer fractions.

    For each task type:
      1. preserve original arrival order
      2. convert fractions to exact counts
      3. allocate requested number to each worker
    """

    by_task = {
        task: []
        for task in TASK_ORDER
    }

    for request_id, item in enumerate(tasks):
        task_type = item["task_type"]

        if task_type not in by_task:
            raise RuntimeError(
                f"Unknown task type: {task_type}"
            )

        by_task[task_type].append(
            (request_id, item)
        )

    assignments = {}

    exact_counts = {}

    for task_type in TASK_ORDER:

        items = by_task[task_type]

        if task_type not in routing_plan:
            raise RuntimeError(
                f"Routing missing task: {task_type}"
            )

        route = routing_plan[
            task_type
        ]

        fractions = {
            model:
                float(info["fraction"])
            for model, info
            in route.items()
        }

        if not fractions:
            raise RuntimeError(
                f"No executor for task {task_type}"
            )

        fraction_sum = sum(
            fractions.values()
        )

        if abs(fraction_sum - 1.0) > 1e-5:
            raise RuntimeError(
                f"Routing fractions for {task_type} "
                f"sum to {fraction_sum}"
            )

        counts = allocate_integer_counts(
            len(items),
            fractions,
        )

        exact_counts[
            task_type
        ] = counts

        # ---------------------------------
        # Deterministic interleaved routing
        #
        # Do not assign one contiguous block
        # to each worker. Spread the exact
        # integer counts across the task stream.
        # This is important for ASR, where
        # samples may differ in duration.
        # ---------------------------------

        assigned = {
            model: 0
            for model in counts
        }

        total = len(items)

        for position, (
            request_id,
            _
        ) in enumerate(
            items,
            start=1,
        ):

            candidates = [
                model
                for model in counts
                if assigned[model]
                < counts[model]
            ]

            if not candidates:
                raise RuntimeError(
                    f"No remaining executor "
                    f"for {task_type}"
                )

            # Weighted-fair cumulative target:
            #
            # desired_i(m)
            #   = i * exact_count(m) / total
            #
            # Pick the model currently most
            # behind its desired cumulative
            # allocation.
            target = max(
                candidates,
                key=lambda model: (
                    (
                        position
                        * counts[model]
                        / total
                    )
                    - assigned[model],
                    counts[model]
                    - assigned[model],
                    model,
                ),
            )

            assignments[
                request_id
            ] = target

            assigned[
                target
            ] += 1

        assert assigned == counts

    assert len(assignments) == len(tasks)

    return assignments, exact_counts


def qwen_warm_one(
    worker,
    item,
    request_id,
    timeout_s,
):
    """
    Same worker protocol as the measured requests.
    Warmup result is discarded.
    """

    request = {
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

    worker.send(request)

    qwen_wait_for(
        worker,
        "RESULT",
        timeout_s,
    )


def recompute_nominal_prediction(
    exact_counts,
    profile,
):
    """
    Recompute nominal makespan using exact integer
    request counts used by this run.
    """

    models = profile["models"]

    load_ms = defaultdict(float)

    for task_type, route_counts in (
        exact_counts.items()
    ):
        for model, count in (
            route_counts.items()
        ):

            service_ms = (
                models[model]
                ["tasks"]
                [task_type]
                ["service_ms"]
            )

            load_ms[model] += (
                count
                * float(service_ms)
            )

    makespan_ms = (
        max(load_ms.values())
        if load_ms
        else 0.0
    )

    total = sum(
        sum(x.values())
        for x in exact_counts.values()
    )

    throughput = (
        total
        / (makespan_ms / 1000.0)
        if makespan_ms > 0
        else 0.0
    )

    return {
        "model_load_ms":
            dict(load_ms),

        "makespan_ms":
            makespan_ms,

        "throughput_tasks_per_s":
            throughput,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--placement",
        required=True,
    )

    parser.add_argument(
        "--profile",
        default=(
            "benchmark/profiles/"
            "model_placement_profile.json"
        ),
    )

    parser.add_argument(
        "--budget-gib",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--max-models",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--pool",
        required=True,
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

    placement = load_json(
        args.placement
    )

    profile = load_json(
        args.profile
    )

    selected_row = select_plan(
        placement,
        args.budget_gib,
        args.max_models,
    )

    plan = selected_row["best"]

    resident_models = list(
        plan["models"]
    )

    unsupported = (
        set(resident_models)
        - SUPPORTED_RUNTIME_MODELS
    )

    if unsupported:
        raise RuntimeError(
            "This validation runner currently supports "
            "Qwen + specialists only. Unsupported: "
            + ", ".join(sorted(unsupported))
        )

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
            item,
        )

    for task_type in TASK_ORDER:
        if task_type not in first_by_task:
            raise RuntimeError(
                f"Pool missing task: {task_type}"
            )

    assignments, exact_counts = (
        build_assignments(
            tasks,
            plan["routing"],
        )
    )

    integer_prediction = (
        recompute_nominal_prediction(
            exact_counts,
            profile,
        )
    )

    print(
        "===== SELECTED PLACEMENT ====="
    )

    print(
        "budget:",
        args.budget_gib,
        "GiB"
    )

    print(
        "max models:",
        args.max_models
    )

    print(
        "resident models:",
        resident_models
    )

    print(
        "profile memory:",
        plan["memory_mib"],
        "MiB"
    )

    print()
    print(
        "===== EXACT ROUTING COUNTS ====="
    )

    for task_type in TASK_ORDER:
        print(
            f"{task_type:>5} ->",
            exact_counts[task_type],
        )

    print()
    print(
        "LP predicted makespan:",
        plan["predicted_makespan_ms"],
        "ms"
    )

    print(
        "LP predicted throughput:",
        plan["predicted_throughput_qps"],
        "tasks/s"
    )

    print(
        "Integer-route nominal makespan:",
        round(
            integer_prediction[
                "makespan_ms"
            ],
            3,
        ),
        "ms"
    )

    print(
        "Integer-route nominal throughput:",
        round(
            integer_prediction[
                "throughput_tasks_per_s"
            ],
            3,
        ),
        "tasks/s"
    )

    # -------------------------------------------------
    # Worker construction
    # -------------------------------------------------

    shared_events = queue.Queue()

    shared_models = [
        model
        for model in resident_models
        if model != "qwen"
    ]

    shared_workers = {
        name:
            SharedWorker(
                name,
                WORKERS[name],
                shared_events,
            )
        for name in shared_models
    }

    qwen_worker = (
        QwenWorker()
        if "qwen" in resident_models
        else None
    )

    results = []
    errors = []

    try:

        # ==========================================
        # 1. Start resident workers
        # ==========================================

        print()
        print(
            "===== STARTING RESIDENT WORKERS ====="
        )

        startup_order = [
            x
            for x in [
                "distilbert",
                "resnet50",
                "paddleocr",
                "whisper",
            ]
            if x in shared_workers
        ]

        for name in startup_order:

            print(
                f"Starting {name}..."
            )

            shared_workers[
                name
            ].start()

            wait_ready(
                name,
                shared_workers,
                shared_events,
                args.timeout_s,
            )

            print(
                f"{name}: READY"
            )

        if qwen_worker is not None:

            print(
                "Starting qwen..."
            )

            qwen_worker.start()

            qwen_wait_for(
                qwen_worker,
                "READY",
                args.timeout_s,
            )

            print(
                "qwen: READY"
            )

        print()
        print(
            "ALL SELECTED MODELS ARE RESIDENT."
        )

        # ==========================================
        # 2. Warm only model/task paths actually used
        # ==========================================

        print()
        print(
            "===== WARMUP USED PATHS ====="
        )

        warmup_id = -1

        for task_type in TASK_ORDER:

            for model, count in (
                exact_counts[
                    task_type
                ].items()
            ):

                if count <= 0:
                    continue

                print(
                    f"warm {task_type} -> {model}"
                )

                if model == "qwen":

                    qwen_warm_one(
                        qwen_worker,
                        first_by_task[
                            task_type
                        ],
                        warmup_id,
                        args.timeout_s,
                    )

                else:

                    shared_warm_one(
                        model,
                        first_by_task[
                            task_type
                        ],
                        warmup_id,
                        shared_workers,
                        shared_events,
                        args.timeout_s,
                    )

                warmup_id -= 1

        print()
        print(
            "ALL USED MODEL/TASK PATHS WARM."
        )

        # ==========================================
        # 3. Fixed placement routing
        # ==========================================

        routing = Counter()

        routing_by_task = {
            task:
                Counter()
            for task in TASK_ORDER
        }

        request_to_worker = {}

        benchmark_start_ns = (
            time.time_ns()
        )

        # Same saturated-batch protocol as
        # previous pure-system experiments:
        # submit the complete task pool.
        for request_id, item in enumerate(
            tasks
        ):

            target = assignments[
                request_id
            ]

            task_type = item[
                "task_type"
            ]

            request = {
                "request_id":
                    request_id,

                "pool_id":
                    item["pool_id"],

                "arrival_index":
                    item["arrival_index"],

                "task_type":
                    task_type,

                "sample":
                    item["sample"],

                "submit_ns":
                    time.time_ns(),
            }

            request_to_worker[
                request_id
            ] = target

            routing[target] += 1

            routing_by_task[
                task_type
            ][target] += 1

            if target == "qwen":
                qwen_worker.send(
                    request
                )
            else:
                shared_workers[
                    target
                ].send(
                    request
                )

        print()
        print(
            "Submitted:",
            len(tasks)
        )

        # ==========================================
        # 4. Collect from both event systems
        # ==========================================

        completed_ids = set()

        deadline = (
            time.time()
            + args.timeout_s
        )

        while (
            len(completed_ids)
            < len(tasks)
        ):

            if time.time() >= deadline:
                raise TimeoutError(
                    "Placement validation timed out."
                )

            made_progress = False

            # ------------------------------
            # Shared specialist events
            # ------------------------------

            try:
                (
                    worker_name,
                    kind,
                    payload,
                ) = shared_events.get_nowait()

                made_progress = True

                if kind == "RESULT":

                    request_id = int(
                        payload[
                            "request_id"
                        ]
                    )

                    if request_id >= 0:

                        payload[
                            "assigned_worker"
                        ] = worker_name

                        results.append(
                            payload
                        )

                        completed_ids.add(
                            request_id
                        )

                elif kind == "ERROR":

                    errors.append({
                        "worker":
                            worker_name,
                        "payload":
                            payload,
                    })

                    if isinstance(
                        payload,
                        dict,
                    ):
                        request_id = (
                            payload.get(
                                "request_id"
                            )
                        )

                        if (
                            request_id
                            is not None
                            and int(request_id)
                            >= 0
                        ):
                            completed_ids.add(
                                int(request_id)
                            )

            except queue.Empty:
                pass

            # ------------------------------
            # Qwen events
            # ------------------------------

            if qwen_worker is not None:

                try:
                    kind, payload = (
                        qwen_worker.events
                        .get_nowait()
                    )

                    made_progress = True

                    if kind == "RESULT":

                        request_id = int(
                            payload[
                                "request_id"
                            ]
                        )

                        if request_id >= 0:

                            payload[
                                "assigned_worker"
                            ] = "qwen"

                            results.append(
                                payload
                            )

                            completed_ids.add(
                                request_id
                            )

                    elif kind == "ERROR":

                        errors.append({
                            "worker":
                                "qwen",
                            "payload":
                                payload,
                        })

                        if isinstance(
                            payload,
                            dict,
                        ):
                            request_id = (
                                payload.get(
                                    "request_id"
                                )
                            )

                            if (
                                request_id
                                is not None
                                and int(
                                    request_id
                                ) >= 0
                            ):
                                completed_ids.add(
                                    int(request_id)
                                )

                except queue.Empty:
                    pass

            # ------------------------------
            # Check worker failures
            # ------------------------------

            for name, worker in (
                shared_workers.items()
            ):

                if (
                    worker.process
                    is not None
                    and worker.process.poll()
                    is not None
                ):
                    raise RuntimeError(
                        f"Worker {name} exited "
                        "before benchmark completed."
                    )

            if (
                qwen_worker is not None
                and qwen_worker.process
                is not None
                and qwen_worker.process.poll()
                is not None
                and len(completed_ids)
                < len(tasks)
            ):
                raise RuntimeError(
                    "Qwen worker exited before "
                    "benchmark completed."
                )

            if not made_progress:
                time.sleep(0.002)

        benchmark_end_ns = (
            time.time_ns()
        )

        wall_time_s = (
            benchmark_end_ns
            - benchmark_start_ns
        ) / 1e9

        results.sort(
            key=lambda x:
                int(
                    x["request_id"]
                )
        )

        # ==========================================
        # 5. Save raw enriched results
        # ==========================================

        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
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

        # ==========================================
        # 6. Prediction-vs-measurement summary
        # ==========================================

        measured_qps = (
            len(tasks)
            / wall_time_s
            if wall_time_s > 0
            else 0.0
        )

        integer_pred_s = (
            integer_prediction[
                "makespan_ms"
            ]
            / 1000.0
        )

        wall_error_pct = (
            abs(
                wall_time_s
                - integer_pred_s
            )
            / wall_time_s
            * 100.0
            if wall_time_s > 0
            else None
        )

        throughput_error_pct = (
            abs(
                measured_qps
                - integer_prediction[
                    "throughput_tasks_per_s"
                ]
            )
            / measured_qps
            * 100.0
            if measured_qps > 0
            else None
        )

        summary = {
            "placement_source":
                args.placement,

            "profile_source":
                args.profile,

            "pool":
                args.pool,

            "budget_gib":
                args.budget_gib,

            "max_models":
                args.max_models,

            "resident_models":
                resident_models,

            "profile_memory_mib":
                plan["memory_mib"],

            "num_tasks":
                len(tasks),

            "num_results":
                len(results),

            "num_errors":
                len(errors),

            "optimizer_fractional_prediction": {
                "makespan_ms":
                    plan[
                        "predicted_makespan_ms"
                    ],

                "throughput_tasks_per_s":
                    plan[
                        "predicted_throughput_qps"
                    ],

                "routing":
                    plan["routing"],

                "predicted_quality":
                    plan[
                        "predicted_quality"
                    ],
            },

            "integer_route_prediction":
                integer_prediction,

            "exact_routing_counts":
                exact_counts,

            "measured": {
                "wall_time_s":
                    wall_time_s,

                "throughput_tasks_per_s":
                    measured_qps,

                "routing":
                    dict(routing),

                "routing_by_task": {
                    task:
                        dict(
                            routing_by_task[
                                task
                            ]
                        )
                    for task in TASK_ORDER
                },
            },

            "prediction_error": {
                "wall_time_percent":
                    wall_error_pct,

                "throughput_percent":
                    throughput_error_pct,
            },

            "errors":
                errors,
        }

        summary_path = Path(
            str(output_path)
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
            "===== MEASURED RESULT ====="
        )

        print(
            "tasks:",
            len(tasks)
        )

        print(
            "results:",
            len(results)
        )

        print(
            "errors:",
            len(errors)
        )

        print(
            "wall time:",
            round(
                wall_time_s,
                3,
            ),
            "s"
        )

        print(
            "throughput:",
            round(
                measured_qps,
                3,
            ),
            "tasks/s"
        )

        print(
            "nominal predicted throughput:",
            round(
                integer_prediction[
                    "throughput_tasks_per_s"
                ],
                3,
            ),
            "tasks/s"
        )

        print(
            "throughput prediction error:",
            round(
                throughput_error_pct,
                2,
            ),
            "%"
        )

        print()
        print(
            "routing:",
            dict(routing)
        )

        print(
            "routing by task:"
        )

        for task_type in TASK_ORDER:
            print(
                " ",
                task_type,
                dict(
                    routing_by_task[
                        task_type
                    ]
                ),
            )

        print()
        print(
            "Saved results:",
            output_path
        )

        print(
            "Saved summary:",
            summary_path
        )

    finally:

        # ==========================================
        # Cleanup
        # ==========================================

        for worker in (
            shared_workers.values()
        ):
            try:
                worker.shutdown()
            except Exception:
                pass

        if qwen_worker is not None:
            try:
                qwen_worker.shutdown()
            except Exception:
                pass

        time.sleep(1.0)

        for worker in (
            shared_workers.values()
        ):
            try:
                worker.terminate()
            except Exception:
                pass

        if qwen_worker is not None:
            try:
                qwen_worker.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
