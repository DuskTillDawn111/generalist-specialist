import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


TASKS = [
    "text",
    "image",
    "ocr",
    "asr",
]


# ------------------------------------------------------------
# v2 canonical search space
#
# Interference profiling is complete only for:
# Qwen + four task specialists.
# ------------------------------------------------------------

ALLOWED_MODELS = [
    "qwen",
    "distilbert",
    "resnet50",
    "paddleocr",
    "whisper",
]


# ------------------------------------------------------------
# Same mixed_formal_800 single-resident service baseline.
#
# Do NOT use the old single-task service_ms here because
# pairwise gamma was estimated against these same-workload
# baselines.
# ------------------------------------------------------------

SINGLE_SERVICE_MS = {
    ("qwen", "text"):
        43.760553445,

    ("qwen", "image"):
        137.328372035,

    ("qwen", "ocr"):
        94.207280995,

    ("qwen", "asr"):
        621.703561445,

    ("distilbert", "text"):
        2.51614193,

    ("resnet50", "image"):
        5.58187836,

    ("paddleocr", "ocr"):
        5.80345425,

    ("whisper", "asr"):
        195.270631865,
}


SLA = {
    "text": {
        "metric": "accuracy",
        "op": ">=",
        "value": 0.925,
    },

    "image": {
        "metric": "accuracy",
        "op": ">=",
        "value": 1.0,
    },

    "ocr": {
        "metric": "exact_match",
        "op": ">=",
        "value": 0.96,
    },

    "asr": {
        "metric": "wer",
        "op": "<=",
        "value": 0.042181,
    },
}


def load_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def compatible(
    model_info,
    task,
):
    return (
        task
        in model_info[
            "supported_tasks"
        ]
    )


def load_gamma(path):
    rows = load_json(path)

    return {
        (
            r["worker"],
            r["task"],
            r["co_resident"],
        ): float(
            r["gamma"]
        )
        for r in rows
    }


def subset_memory(
    subset,
    models,
):
    return sum(
        float(
            models[m][
                "peak_process_vram_mib"
            ]
        )
        for m in subset
    )


def covers_all_tasks(
    subset,
    models,
):
    return all(
        any(
            compatible(
                models[m],
                task,
            )
            for m in subset
        )
        for task in TASKS
    )


def effective_service_ms(
    worker,
    task,
    subset,
    gamma,
):
    base = SINGLE_SERVICE_MS[
        (worker, task)
    ]

    total_gamma = 0.0

    for other in subset:

        if other == worker:
            continue

        key = (
            worker,
            task,
            other,
        )

        if key not in gamma:
            raise RuntimeError(
                "Missing interference "
                f"coefficient: {key}"
            )

        total_gamma += gamma[key]

    factor = (
        1.0
        + total_gamma
    )

    if factor <= 0:
        raise RuntimeError(
            f"Invalid interference "
            f"factor for "
            f"{worker}:{task} "
            f"in {subset}: {factor}"
        )

    return (
        base
        * factor
    )


def solve_routing(
    subset,
    models,
    gamma,
    workload,
):
    pairs = []

    for task in TASKS:

        for model in subset:

            if compatible(
                models[model],
                task,
            ):
                pairs.append(
                    (
                        task,
                        model,
                    )
                )

    pair_index = {
        pair: i
        for i, pair
        in enumerate(pairs)
    }

    t_index = len(pairs)
    nvar = len(pairs) + 1

    # Objective: minimize makespan T.
    c = np.zeros(
        nvar,
        dtype=float,
    )

    c[t_index] = 1.0


    # --------------------------------------------------------
    # Every task is completely assigned.
    # --------------------------------------------------------

    A_eq = []
    b_eq = []

    for task in TASKS:

        row = np.zeros(
            nvar,
            dtype=float,
        )

        for model in subset:

            pair = (
                task,
                model,
            )

            if pair in pair_index:
                row[
                    pair_index[pair]
                ] = 1.0

        A_eq.append(row)
        b_eq.append(1.0)


    # --------------------------------------------------------
    # Worker load <= makespan.
    # --------------------------------------------------------

    A_ub = []
    b_ub = []

    effective = {}

    for model in subset:

        row = np.zeros(
            nvar,
            dtype=float,
        )

        for task in TASKS:

            pair = (
                task,
                model,
            )

            if pair not in pair_index:
                continue

            service_ms = (
                effective_service_ms(
                    worker=model,
                    task=task,
                    subset=subset,
                    gamma=gamma,
                )
            )

            effective[
                (
                    model,
                    task,
                )
            ] = service_ms

            row[
                pair_index[pair]
            ] = (
                workload[task]
                * service_ms
            )

        row[t_index] = -1.0

        A_ub.append(row)
        b_ub.append(0.0)


    # --------------------------------------------------------
    # Quality SLA.
    # --------------------------------------------------------

    for task, constraint in (
        SLA.items()
    ):

        expected_metric = (
            constraint["metric"]
        )

        threshold = float(
            constraint["value"]
        )

        row = np.zeros(
            nvar,
            dtype=float,
        )

        for model in subset:

            pair = (
                task,
                model,
            )

            if pair not in pair_index:
                continue

            task_info = (
                models[model][
                    "tasks"
                ][task]
            )

            if (
                task_info[
                    "quality_metric"
                ]
                != expected_metric
            ):
                raise RuntimeError(
                    "Metric mismatch: "
                    f"{task}/{model}"
                )

            row[
                pair_index[pair]
            ] = float(
                task_info[
                    "quality"
                ]
            )

        if (
            constraint["op"]
            == ">="
        ):
            A_ub.append(
                -row
            )
            b_ub.append(
                -threshold
            )

        elif (
            constraint["op"]
            == "<="
        ):
            A_ub.append(row)
            b_ub.append(
                threshold
            )

        else:
            raise ValueError(
                constraint["op"]
            )


    bounds = (
        [
            (0.0, 1.0)
            for _ in pairs
        ]
        + [
            (0.0, None)
        ]
    )


    result = linprog(
        c=c,
        A_ub=np.asarray(
            A_ub
        ),
        b_ub=np.asarray(
            b_ub
        ),
        A_eq=np.asarray(
            A_eq
        ),
        b_eq=np.asarray(
            b_eq
        ),
        bounds=bounds,
        method="highs",
    )


    if not result.success:
        return None


    x = result.x

    routing = {}

    predicted_quality = {}

    worker_load_ms = {
        model: 0.0
        for model in subset
    }


    for task in TASKS:

        routing[task] = {}

        quality = 0.0

        for model in subset:

            pair = (
                task,
                model,
            )

            if pair not in pair_index:
                continue

            frac = float(
                x[
                    pair_index[pair]
                ]
            )

            if frac < 1e-8:
                continue

            service_ms = effective[
                (
                    model,
                    task,
                )
            ]

            requests = (
                workload[task]
                * frac
            )

            routing[task][model] = {
                "fraction":
                    frac,

                "requests":
                    requests,

                "effective_service_ms":
                    service_ms,
            }

            worker_load_ms[
                model
            ] += (
                requests
                * service_ms
            )

            quality += (
                frac
                * float(
                    models[model][
                        "tasks"
                    ][task][
                        "quality"
                    ]
                )
            )

        predicted_quality[
            task
        ] = quality


    makespan_ms = float(
        x[t_index]
    )

    total_requests = sum(
        workload.values()
    )

    throughput = (
        total_requests
        / (
            makespan_ms
            / 1000.0
        )
    )


    return {
        "predicted_makespan_ms":
            makespan_ms,

        "predicted_throughput_qps":
            throughput,

        "routing":
            routing,

        "predicted_quality":
            predicted_quality,

        "worker_load_ms":
            worker_load_ms,

        "effective_service_ms": {
            f"{model}:{task}":
                value
            for (
                model,
                task
            ), value
            in effective.items()
        },
    }


def optimize(
    models,
    gamma,
    workload,
    budget_mib,
    max_models,
    max_generalists,
):
    names = [
        name
        for name in ALLOWED_MODELS
        if name in models
    ]

    feasible = []


    for k in range(
        1,
        min(
            max_models,
            len(names),
        )
        + 1,
    ):

        for subset_tuple in (
            itertools.combinations(
                names,
                k,
            )
        ):

            subset = list(
                subset_tuple
            )

            generalists = sum(
                models[m]["role"]
                == "generalist"
                for m in subset
            )

            if (
                generalists
                > max_generalists
            ):
                continue

            memory_mib = (
                subset_memory(
                    subset,
                    models,
                )
            )

            if (
                memory_mib
                > budget_mib
            ):
                continue

            if not covers_all_tasks(
                subset,
                models,
            ):
                continue

            result = solve_routing(
                subset=subset,
                models=models,
                gamma=gamma,
                workload=workload,
            )

            if result is None:
                continue

            feasible.append({
                "models":
                    subset,

                "model_count":
                    len(subset),

                "memory_mib":
                    memory_mib,

                **result,
            })


    if not feasible:
        return None, []


    feasible.sort(
        key=lambda r:
            (
                r[
                    "predicted_makespan_ms"
                ],
                r["memory_mib"],
                r["model_count"],
            )
    )


    return (
        feasible[0],
        feasible,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        default=(
            "benchmark/profiles/"
            "model_placement_profile.json"
        ),
    )

    parser.add_argument(
        "--gamma",
        default=(
            "benchmark/results/placement/"
            "pairwise_interference_gamma_complete.json"
        ),
    )

    parser.add_argument(
        "--budget-gib",
        type=float,
        default=24.0,
    )

    parser.add_argument(
        "--slots",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            3,
            4,
        ],
    )

    parser.add_argument(
        "--max-generalists",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--count-per-task",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--output",
        default=(
            "benchmark/results/placement/"
            "placement_v2_interference_aware.json"
        ),
    )

    args = parser.parse_args()


    profile = load_json(
        args.profile
    )

    models = profile[
        "models"
    ]

    gamma = load_gamma(
        args.gamma
    )


    workload = {
        task:
            args.count_per_task
        for task in TASKS
    }


    budget_mib = (
        args.budget_gib
        * 1024.0
    )


    output_rows = []


    print(
        "===== INTERFERENCE-AWARE "
        "PLACEMENT OPTIMIZER V2 ====="
    )

    print(
        "budget:",
        args.budget_gib,
        "GiB"
    )

    print(
        "workload:",
        workload
    )

    print(
        "search models:",
        ALLOWED_MODELS
    )

    print()


    for slots in args.slots:

        best, feasible = optimize(
            models=models,
            gamma=gamma,
            workload=workload,
            budget_mib=budget_mib,
            max_models=slots,
            max_generalists=(
                args.max_generalists
            ),
        )


        print(
            "=" * 70
        )

        print(
            f"MAX MODELS = {slots}"
        )


        if best is None:

            print(
                "INFEASIBLE"
            )

            output_rows.append({
                "max_models":
                    slots,

                "feasible":
                    False,
            })

            continue


        print(
            "selected models =",
            best["models"]
        )

        print(
            "memory_mib =",
            best["memory_mib"]
        )

        print(
            "predicted wall_s =",
            best[
                "predicted_makespan_ms"
            ]
            / 1000.0
        )

        print(
            "predicted qps =",
            best[
                "predicted_throughput_qps"
            ]
        )

        print(
            "worker loads:"
        )

        for model, load in sorted(
            best[
                "worker_load_ms"
            ].items()
        ):
            print(
                f"  {model:<12}"
                f"{load / 1000:>9.3f} s"
            )

        print(
            "routing:"
        )

        for task in TASKS:
            print(
                f"  {task:<6}",
                best[
                    "routing"
                ][task]
            )

        print(
            "quality:",
            best[
                "predicted_quality"
            ]
        )

        print(
            "feasible candidates =",
            len(feasible)
        )


        output_rows.append({
            "max_models":
                slots,

            "feasible":
                True,

            "best":
                best,

            "num_feasible":
                len(feasible),
        })


    out = {
        "schema_version": 2,

        "model":
            "directional_pairwise_interference",

        "service_formula": (
            "single_service_ms * "
            "(1 + sum(gamma_worker_task_other))"
        ),

        "budget_gib":
            args.budget_gib,

        "workload":
            workload,

        "allowed_models":
            ALLOWED_MODELS,

        "results":
            output_rows,
    }


    out_path = Path(
        args.output
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(
        json.dumps(
            out,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


    print()
    print(
        "Saved:",
        out_path
    )


if __name__ == "__main__":
    main()
