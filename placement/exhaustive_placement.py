import argparse
import csv
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


def subset_memory(
    subset,
    models,
):
    return sum(
        models[m][
            "peak_process_vram_mib"
        ]
        for m in subset
    )


def covers_all_tasks(
    subset,
    models,
):
    for task in TASKS:

        if not any(
            compatible(
                models[m],
                task,
            )
            for m in subset
        ):
            return False

    return True


def solve_routing(
    subset,
    models,
    workload,
    quality_constraints,
):
    # ---------------------------------
    # Variables:
    #
    # y_(task,model)
    #   routing fraction
    #
    # T
    #   predicted makespan in ms
    # ---------------------------------

    pairs = []

    for task in TASKS:
        for model in subset:

            if compatible(
                models[model],
                task,
            ):
                pairs.append(
                    (task, model)
                )

    pair_index = {
        pair: i
        for i, pair
        in enumerate(pairs)
    }

    t_index = len(pairs)
    nvar = len(pairs) + 1

    # Objective: min T
    c = np.zeros(
        nvar,
        dtype=float,
    )

    c[t_index] = 1.0

    # ---------------------------------
    # Equality:
    # every task must be fully routed
    # ---------------------------------

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

    # ---------------------------------
    # Inequality constraints
    # ---------------------------------

    A_ub = []
    b_ub = []

    # Model load <= T
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
                models[model][
                    "tasks"
                ][task][
                    "service_ms"
                ]
            )

            row[
                pair_index[pair]
            ] = (
                workload[task]
                * service_ms
            )

        row[t_index] = -1.0

        A_ub.append(row)
        b_ub.append(0.0)

    # ---------------------------------
    # Quality SLA
    # ---------------------------------

    for task, constraint in (
        quality_constraints.items()
    ):

        expected_metric = (
            constraint["metric"]
        )

        op = constraint["op"]
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

            actual_metric = (
                task_info[
                    "quality_metric"
                ]
            )

            if (
                actual_metric
                != expected_metric
            ):
                raise RuntimeError(
                    f"Quality metric mismatch: "
                    f"{task}/{model}: "
                    f"{actual_metric} "
                    f"!= {expected_metric}"
                )

            row[
                pair_index[pair]
            ] = float(
                task_info["quality"]
            )

        if op == ">=":

            A_ub.append(
                -row
            )
            b_ub.append(
                -threshold
            )

        elif op == "<=":

            A_ub.append(
                row
            )
            b_ub.append(
                threshold
            )

        else:
            raise ValueError(
                f"Unsupported quality "
                f"operator: {op}"
            )

    bounds = (
        [(0.0, 1.0)]
        * len(pairs)
        + [(0.0, None)]
    )

    result = linprog(
        c=c,
        A_ub=(
            np.asarray(A_ub)
            if A_ub
            else None
        ),
        b_ub=(
            np.asarray(b_ub)
            if b_ub
            else None
        ),
        A_eq=np.asarray(A_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return None

    x = result.x

    routing = {}
    predicted_quality = {}

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

            routing[task][model] = {
                "fraction": frac,
                "requests": (
                    frac
                    * workload[task]
                ),
            }

            quality += (
                frac
                * models[model][
                    "tasks"
                ][task][
                    "quality"
                ]
            )

        predicted_quality[task] = (
            quality
        )

    model_load_ms = {}

    for model in subset:

        load = 0.0

        for task in TASKS:

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

            load += (
                workload[task]
                * frac
                * models[model][
                    "tasks"
                ][task][
                    "service_ms"
                ]
            )

        model_load_ms[model] = load

    makespan_ms = float(
        x[t_index]
    )

    total_requests = sum(
        workload.values()
    )

    throughput = (
        total_requests
        / (makespan_ms / 1000.0)
        if makespan_ms > 0
        else 0.0
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

        "model_load_ms":
            model_load_ms,
    }


def optimize_one(
    models,
    workload,
    quality_constraints,
    budget_mib,
    max_models,
    max_generalists,
):
    names = list(
        models.keys()
    )

    feasible = []

    subsets_enumerated = 0

    # 7 models => all 2^7 subsets
    for mask in range(
        1 << len(names)
    ):

        subsets_enumerated += 1

        subset = [
            names[i]
            for i in range(
                len(names)
            )
            if (
                mask
                & (1 << i)
            )
        ]

        if not subset:
            continue

        if (
            max_models is not None
            and len(subset)
            > max_models
        ):
            continue

        generalist_count = sum(
            1
            for m in subset
            if (
                models[m]["role"]
                == "generalist"
            )
        )

        specialist_count = (
            len(subset)
            - generalist_count
        )

        if (
            max_generalists is not None
            and generalist_count
            > max_generalists
        ):
            continue

        memory_mib = subset_memory(
            subset,
            models,
        )

        if memory_mib > budget_mib:
            continue

        if not covers_all_tasks(
            subset,
            models,
        ):
            continue

        routing_result = solve_routing(
            subset=subset,
            models=models,
            workload=workload,
            quality_constraints=(
                quality_constraints
            ),
        )

        if routing_result is None:
            continue

        record = {
            "models": subset,
            "model_count":
                len(subset),

            "generalist_count":
                generalist_count,

            "specialist_count":
                specialist_count,

            "memory_mib":
                memory_mib,

            **routing_result,
        }

        feasible.append(record)

    feasible.sort(
        key=lambda x: (
            x[
                "predicted_makespan_ms"
            ],
            x["memory_mib"],
            x["model_count"],
            ",".join(x["models"]),
        )
    )

    return {
        "subsets_enumerated":
            subsets_enumerated,

        "feasible_count":
            len(feasible),

        "best": (
            feasible[0]
            if feasible
            else None
        ),

        "top10":
            feasible[:10],
    }


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
        "--scenario",
        default=(
            "benchmark/profiles/"
            "placement_scenario.json"
        ),
    )

    parser.add_argument(
        "--quality-preset",
        default="none",
    )

    parser.add_argument(
        "--budget-gib",
        type=float,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--max-models",
        type=int,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--max-generalists",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    profile = load_json(
        args.profile
    )

    scenario = load_json(
        args.scenario
    )

    models = profile["models"]

    workload = scenario[
        "workload"
    ]

    presets = scenario[
        "quality_presets"
    ]

    if (
        args.quality_preset
        not in presets
    ):
        raise KeyError(
            "Unknown quality preset: "
            + args.quality_preset
        )

    quality_constraints = (
        presets[
            args.quality_preset
        ]
    )

    results = []

    for budget_gib in (
        args.budget_gib
    ):

        budget_mib = (
            budget_gib
            * 1024.0
        )

        for max_models in (
            args.max_models
        ):

            solved = optimize_one(
                models=models,
                workload=workload,
                quality_constraints=(
                    quality_constraints
                ),
                budget_mib=budget_mib,
                max_models=max_models,
                max_generalists=(
                    args.max_generalists
                ),
            )

            row = {
                "budget_gib":
                    budget_gib,

                "budget_mib":
                    budget_mib,

                "max_models":
                    max_models,

                "max_generalists":
                    args.max_generalists,

                "quality_preset":
                    args.quality_preset,

                **solved,
            }

            results.append(row)

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "profile":
            args.profile,

        "scenario":
            args.scenario,

        "quality_preset":
            args.quality_preset,

        "workload":
            workload,

        "results":
            results,
    }

    output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    csv_path = (
        output.with_suffix(".csv")
    )

    csv_rows = []

    print()
    print(
        "===== PLACEMENT RESULTS ====="
    )

    for r in results:

        best = r["best"]

        if best is None:

            print(
                f"budget={r['budget_gib']:>5g} GiB "
                f"slots={r['max_models']} "
                f"-> INFEASIBLE"
            )

            csv_rows.append({
                "budget_gib":
                    r["budget_gib"],
                "max_models":
                    r["max_models"],
                "quality_preset":
                    r["quality_preset"],
                "status":
                    "infeasible",
            })

            continue

        model_str = "+".join(
            best["models"]
        )

        print(
            f"budget={r['budget_gib']:>5g} GiB "
            f"slots={r['max_models']} "
            f"-> {model_str}"
        )

        print(
            f"    G={best['generalist_count']} "
            f"S={best['specialist_count']} "
            f"mem={best['memory_mib']:.1f} MiB "
            f"T={best['predicted_makespan_ms']:.2f} ms "
            f"QPS={best['predicted_throughput_qps']:.3f}"
        )

        for task in TASKS:

            parts = []

            for model, info in (
                best[
                    "routing"
                ][task].items()
            ):

                parts.append(
                    f"{model}:"
                    f"{info['fraction']:.3f}"
                )

            print(
                f"    {task:5s} -> "
                + ", ".join(parts)
                + " | quality="
                + (
                    f"{best['predicted_quality'][task]:.6f}"
                )
            )

        csv_rows.append({
            "budget_gib":
                r["budget_gib"],

            "max_models":
                r["max_models"],

            "quality_preset":
                r["quality_preset"],

            "status":
                "feasible",

            "models":
                model_str,

            "generalist_count":
                best["generalist_count"],

            "specialist_count":
                best["specialist_count"],

            "memory_mib":
                best["memory_mib"],

            "predicted_makespan_ms":
                best[
                    "predicted_makespan_ms"
                ],

            "predicted_throughput_qps":
                best[
                    "predicted_throughput_qps"
                ],

            "text_quality":
                best[
                    "predicted_quality"
                ]["text"],

            "image_quality":
                best[
                    "predicted_quality"
                ]["image"],

            "ocr_quality":
                best[
                    "predicted_quality"
                ]["ocr"],

            "asr_quality":
                best[
                    "predicted_quality"
                ]["asr"],
        })

    fields = [
        "budget_gib",
        "max_models",
        "quality_preset",
        "status",
        "models",
        "generalist_count",
        "specialist_count",
        "memory_mib",
        "predicted_makespan_ms",
        "predicted_throughput_qps",
        "text_quality",
        "image_quality",
        "ocr_quality",
        "asr_quality",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(
            csv_rows
        )

    print()
    print("Saved:")
    print(output)
    print(csv_path)


if __name__ == "__main__":
    main()
