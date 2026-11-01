#!/usr/bin/env python3
"""CPU-only lower-bound audit and identical-domain MILP/DFS validation.

Default work only inspects evidence and emits a small-domain config. Explicit
--solve-small/--solve-full enable bounded CPU searches; this script never runs
models or uses GPU APIs. Existing frozen packets are never edited.
"""

import argparse
import copy
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsdeploy import DeploymentSystem, SystemConfig
from gsdeploy.performance import service_lower_ms
from gsdeploy.prediction import predict
from gsdeploy.quality import sample_totals
from gsdeploy.trace import make_trace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lower_bound_audit(c):
    d = c.catalog
    models = sorted(d["models"])
    tasks = sorted(c.workload)
    per_model_memory = {m: min(d["safe_factor"] * value for n, value in
                              d["memory_table_mib"][m].items() if int(n) > 0) for m in models}
    memory_bound = None
    cover = None
    if d.get("memory_mode") != "profiled":
        for mask in itertools.product([False, True], repeat=len(models)):
            chosen = [m for m, yes in zip(models, mask) if yes]
            if not all(any(t in d["models"][m]["supported_tasks"] for m in chosen) for t in tasks):
                continue
            value = sum(per_model_memory[m] for m in chosen)
            if memory_bound is None or value < memory_bound:
                memory_bound, cover = value, chosen
    per_task = {}
    trace = make_trace(c.workload, c.horizon_seconds, c.random_seed, c.arrival_process)
    quality = d.get("quality", {})
    for task in tasks:
        supported = [m for m in models if task in d["models"][m]["supported_tasks"]]
        native_service = min(service_lower_ms(d, m, task) * int(n) / factor
                             for m in supported for n, factor in d["capacity_factors"][m].items()
                             if int(n) > 0)
        path_profiles = {}
        for model in supported:
            entries = d.get("task_replica_profiles", {}).get(model, {}).get(task, {})
            observed = [(int(n), p.get("mean_service_ms")) for n, p in entries.items()
                        if p.get("mean_service_ms") is not None]
            path_profiles[model] = dict(
                observed_mean_service_min_ms=min((v for _, v in observed), default=None),
                observed_counts=[n for n, _ in observed],
                scope="standalone recorded replica means, not a physical mixed-service bound",
            )
        qbound, qscope = 0.0, "zero: incomplete or unsupported quality evidence"
        paths = {m: quality.get("paths", {}).get(m, {}).get(task, {}) for m in supported}
        try:
            if quality.get("mode") == "aggregate_proxy":
                values = [p["value"] for p in paths.values()]
                qbound = min(1 - v if task in {"text", "image"} else v for v in values)
                qscope = "minimum aggregate path loss, valid for the declared convex quality proxy"
            elif quality.get("mode") == "per_sample_replay":
                order = list(quality["input_order"][task])
                random.Random(f"{c.random_seed}:samples:{task}").shuffle(order)
                numerator = denominator = 0.0
                count = sum(t == task for _, t in trace)
                for index in range(count):
                    sid = order[index % len(order)]
                    scores = [sample_totals(task, p["samples"][sid]) for p in paths.values()]
                    lengths = {b for _, b in scores}
                    if len(lengths) != 1:
                        raise ValueError("reference lengths differ")
                    numerator += min(a for a, _ in scores)
                    denominator += scores[0][1]
                qbound = numerator / denominator
                qscope = "per-input best-path oracle on frozen cohort; relaxes routing consistency, valid quality-loss lower bound"
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
        per_task[task] = dict(supported_models=supported, service_lower_ms=native_service,
                              service_bound_scope="declared predictor only; zero means unavailable",
                              quality_loss_lower_bound=qbound, quality_bound_scope=qscope,
                              recorded_replica_profiles=path_profiles)
    total = sum(c.workload.values())
    quality_cost = c.weights["Q"] * sum(c.workload[t] / total * p["quality_loss_lower_bound"]
                                       for t, p in per_task.items())
    latency_cost = sum(c.workload[t] / total / c.latency_reference_ms[t]
                       * (c.weights["L"] + c.weights["P"]) * p["service_lower_ms"]
                       for t, p in per_task.items())
    memory_cost = c.weights["M"] * (memory_bound or 0) / c.gpu_memory_mib
    return dict(
        memory_mode=d.get("memory_mode", "standalone_proxy"),
        minimum_covering_memory_lower_mib=memory_bound if memory_bound is not None else 0.0,
        memory_cover_relaxation=cover, per_task=per_task,
        utility_upper_bound=-(quality_cost + latency_cost + memory_cost),
        lower_cost_contributions=dict(Q=quality_cost, T=0, L_plus_P=latency_cost, M=memory_cost, E=0),
        scope="optimistic frozen-model relaxation; neither a predicted score nor physical GPU bound",
        limitations="Ignores aggregate capacity, queues, energy and route consistency; no witness or winner is inferred",
    )


def small_problem(c):
    d = copy.deepcopy(c.catalog)
    chosen = ["generalist_2", "text"]
    for field in ["models", "capacity_factors", "memory_table_mib"]:
        d[field] = {m: d[field][m] for m in chosen}
    for field in ["capacity_factors", "memory_table_mib"]:
        d[field] = {m: {n: value for n, value in row.items() if n in {"0", "1"}}
                    for m, row in d[field].items()}
    return replace(c, catalog=d, route_grid=4, milp_capacity_witness=False,
                   milp_time_limit_seconds=10, search_max_states=1000000)


def path_table(c):
    rows = []
    for model, spec in c.catalog["models"].items():
        for task in spec["supported_tasks"]:
            q = c.catalog.get("quality", {}).get("paths", {}).get(model, {}).get(task, {})
            replicas = {}
            for n in [1, 2]:
                profile = c.catalog.get("task_replica_profiles", {}).get(model, {}).get(task, {}).get(str(n), {})
                memory = c.catalog["memory_table_mib"][model].get(str(n))
                replicas[str(n)] = dict(
                    mean_worker_service_ms=profile.get("mean_service_ms"),
                    safe_residency_proxy_mib=memory * c.catalog["safe_factor"] if memory is not None else None,
                    sampled_board_peak_mib=profile.get("sampled_board_peak_mib"),
                    median_rps=profile.get("median_rps"), source=profile.get("source"),
                    scope=profile.get("scope"))
            rows.append(dict(model=model, task=task, quality_metric=q.get("metric"),
                             offline_quality_value=q.get("value"), quality_source=q.get("source"),
                             quality_scope="recorded offline metric; quality is not remeasured per replica count",
                             replicas=replicas))
    return rows


def screen_partial(c, budget_seconds=5):
    """A bounded named-pattern screen, never an exhaustive optimum claim."""
    start = time.monotonic()
    models = c.catalog["models"]
    generals = [m for m, s in models.items() if s["role"] == "generalist"]
    specialists = {t: next(m for m, s in models.items()
                          if s["role"] == "specialist" and t in s["supported_tasks"])
                   for t in c.workload}
    tasks = sorted(c.workload)
    plans = []
    for counts in itertools.product([1, 2], repeat=len(tasks)):
        x = {specialists[t]: n for t, n in zip(tasks, counts)}
        route = {t: {specialists[t]: q} for t, q in c.workload.items()}
        plans.append(("specialist_control", x, route))
    for model, count in itertools.product(generals, [1, 2]):
        for active in itertools.product([False, True], repeat=len(tasks)):
            selected = [t for t, yes in zip(tasks, active) if yes]
            for share in ([0] if not selected else [1.0, 0.75]):
                x = {model: count, **{specialists[t]: 1 for t in selected}}
                route = {t: {model: q} for t, q in c.workload.items()}
                for task in selected:
                    q = c.workload[task]
                    route[task] = {m: v for m, v in [(model, q * (1 - share)),
                                                    (specialists[task], q * share)] if v > 0}
                plans.append(("generalist_or_mixed_pattern", x, route))
    evaluated, rejected = [], []
    context = DeploymentSystem(c).context()
    trace = make_trace(c.workload, c.horizon_seconds, c.random_seed, c.arrival_process)
    for label, x, route in plans:
        if time.monotonic() - start > budget_seconds:
            break
        try:
            p = predict(c.catalog, x, route, c.workload, c.gpu_memory_mib,
                        c.latency_reference_ms, c.weights, trace=trace, **context)
        except (KeyError, ValueError) as exc:
            rejected.append(dict(deployment=x, reason=str(exc)))
            continue
        record = dict(label=label, deployment=x, routing=route,
                      constraints_eligible=p["constraints_eligible"], reason=p["reason"],
                      full_U=p["U"], partial_U=p.get("diagnostic_partial_U"),
                      normalized={k: p.get(k + "_norm") for k in "QTLPME"},
                      per_task=p.get("per_task"), max_rho=p.get("max_rho"))
        if p["constraints_eligible"] and record["partial_U"] is not None:
            evaluated.append(record)
        else:
            rejected.append(record)
    evaluated.sort(key=lambda p: -p["partial_U"])
    pure = [p for p in evaluated if p["label"] == "specialist_control"]
    mixed = [p for p in evaluated if any(models[m]["role"] == "specialist" for m in p["deployment"])
             and any(models[m]["role"] == "generalist" for m in p["deployment"])]
    return dict(
        scope="bounded named n1/n2 pattern screen; not full-domain search or physical GPU ranking",
        generated=len(plans), evaluated_eligible=len(evaluated), rejected=len(rejected),
        time_limit_seconds=budget_seconds, elapsed_seconds=time.monotonic() - start,
        search_completed=len(evaluated) + len(rejected) == len(plans),
        best_screened_specialist_partial=pure[0] if pure else None,
        most_competitive_screened_mixed_partial=mixed[:10], highest_partial_candidates=evaluated[:10],
        caution="Partial U omits unknown energy; partial ranking can reverse. Unmeasured G1/G3/multi-replica applicability requires new calibration or independent confirmation.",
    )


def compare(c, seconds):
    results = {}
    for solver in ["milp", "dfs"]:
        result = DeploymentSystem(replace(c, solver=solver,
                                         milp_time_limit_seconds=seconds)).plan()
        results[solver] = result
    a, b = results["milp"], results["dfs"]
    av = a["winner"]["score"] if a["winner"] else None
    bv = b["winner"]["score"] if b["winner"] else None
    return dict(results=results,
                observed_objectives_agree=av is not None and bv is not None and abs(av - bv) <= 1e-6,
                exact_optimality_agrees=a["optimality_certified"] and b["optimality_certified"]
                and av is not None and bv is not None and abs(av - bv) <= 1e-6,
                scope="same frozen-model domain and scoring; tied deployments allowed; limited/unknown runs retain their own gaps")


def render_report(report):
    audit = report["lower_bound_audit"]
    lines = ["# Frozen-pool CPU search preparation", "",
             "No GPU was used. Resource/service/quality bounds refer to the declared frozen predictor; they are not measured mixed-GPU guarantees.", "",
             f"Optimistic utility upper bound: {audit['utility_upper_bound']:.8f}. It relaxes queues, routing consistency and energy; it is not a score or winner.",
             f"Minimum additive residency covering all tasks: {audit['minimum_covering_memory_lower_mib']:.3f} MiB. Cover: {audit['memory_cover_relaxation']}.", "",
             "## Recorded n=1 / n=2 paths", "",
             "Quality is the recorded offline metric, shared across replica counts. Service is the observed standalone worker mean, not a mixed-run bound. Memory is the safety-adjusted residency proxy.", "",
             "| Model | Task | Metric | Offline value | n1 service/ms | n2 service/ms | n1 memory/MiB | n2 memory/MiB |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    def value(v):
        return "unknown" if v is None else f"{v:.4f}"
    for row in report["path_n1_n2_comparison"]:
        a, b = row["replicas"]["1"], row["replicas"]["2"]
        lines.append(f"| {row['model']} | {row['task']} | {row['quality_metric']} | {value(row['offline_quality_value'])} | "
                     f"{value(a['mean_worker_service_ms'])} | {value(b['mean_worker_service_ms'])} | "
                     f"{value(a['safe_residency_proxy_mib'])} | {value(b['safe_residency_proxy_mib'])} |")
    if "partial_pattern_screen" in report:
        screen = report["partial_pattern_screen"]
        lines += ["", "## Bounded pattern screen", "",
                  f"Generated {screen['generated']} named patterns; scored {screen['evaluated_eligible']} eligible and rejected {screen['rejected']} within {screen['elapsed_seconds']:.3f} seconds. Complete: {screen['search_completed']}.",
                  "Unknown energy is omitted from the diagnostic partial objective. Rankings can reverse; this is not exhaustive search or a six-term winner.", "",
                  "| Pattern | Deployment | Partial U | Q | T | L | P | M | E |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        rows = ([screen["best_screened_specialist_partial"]]
                if screen["best_screened_specialist_partial"] else [])
        rows += screen["most_competitive_screened_mixed_partial"][:5]
        for row in rows:
            lines.append(f"| {row['label']} | {row['deployment']} | {value(row['partial_U'])} | "
                         + " | ".join(value(row["normalized"][k]) for k in "QTLPME") + " |")
        lines += ["", screen["caution"]]
    lines += ["", "## Formal restricted validation domain", "",
              "The emitted small-domain-config.json preserves the four-task load, inputs, weights and evaluator. It limits model choices to G2/text at 0/1 replicas and K=4, yielding eight full identities before evidence validity. This validates the implementation on a stated restricted domain, not the retained seven-model optimum.", "",
              "Use --solve-small or --solve-full only when the corresponding evaluator is ready; limited and unknown runs preserve their gaps. Full details and source identities are in report.json.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("artifacts/static-study-v6-20261021/inputs/system-config.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/static-study-gpu-20261022/cpu-search-validation"))
    parser.add_argument("--solve-small", action="store_true")
    parser.add_argument("--solve-full", action="store_true")
    parser.add_argument("--time-limit", type=float, default=10)
    parser.add_argument("--screen", action="store_true")
    args = parser.parse_args()
    c = SystemConfig.load(args.config)
    if c.route_pair_denominators_by_task:
        raise ValueError("PAIR_ROUTE_DOMAIN_UNSUPPORTED_BY_LEGACY_STATIC_VALIDATION: use a versioned domain audit")
    small = small_problem(c)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "small-domain-config.json").write_text(json.dumps(small.as_dict(), indent=2) + "\n")
    r = dict(schema_version=1, gpu_started=False, source_config=str(args.config.resolve()),
             source_config_sha256=sha(args.config), source_catalog_sha256=hashlib.sha256(
                 json.dumps(c.catalog, sort_keys=True).encode()).hexdigest(),
             lower_bound_audit=lower_bound_audit(c),
             path_n1_n2_comparison=path_table(c),
             small_domain=dict(models=["generalist_2", "text"], counts=[0, 1],
                               workload=c.workload, K=4, expected_complete_plans=8,
                               scope="restricted domain, not full retained pool",
                               note="G2 mandatory for image/OCR/ASR; G2 alone plus seven G2+text rational QA shares"))
    if args.screen:
        r["partial_pattern_screen"] = screen_partial(c)
    if args.solve_small:
        r["small_search"] = compare(small, args.time_limit)
    if args.solve_full:
        r["full_search"] = compare(c, args.time_limit)
    (args.output / "report.json").write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
    (args.output / "report.md").write_text(render_report(r))
    print(json.dumps(dict(output=str(args.output.resolve()), gpu_started=False,
                          small_search="small_search" in r, full_search="full_search" in r)))


if __name__ == "__main__":
    main()
