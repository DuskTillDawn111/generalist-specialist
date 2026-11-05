"""Paired CPU experiments, unchanged historical baselines and raw ledgers."""

import copy
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .. import DeploymentSystem
from ..route_domain import fraction_rows
from ..trace import make_trace
from .contracts import Budget, ContinuousProblem, TraceSpec, digest, load_experiment
from .evaluator import ContinuousRoutingEvaluator
from .predictor_adapter import LegacyPredictorOracle, SyntheticColocationOracle
from .de_optimizer import DERoutingOptimizer


def oracle_from_spec(spec):
    mode = spec.get("oracle", {}).get("kind", "legacy")
    if mode == "synthetic_fixture":
        return SyntheticColocationOracle(**spec["oracle"].get("parameters", {}))
    if mode != "legacy":
        raise ValueError("Unknown oracle; supply a complete-candidate adapter through the Python API")
    return LegacyPredictorOracle()


def materialize_grid_catalog(problem, oracle, K, budget, trace):
    """Build exact fixture evidence, not predicted scores, for the old planner.

    Preparation time counts toward the historical planner's total deadline.
    Legacy mode needs no generated evidence and keeps its original catalog.
    """
    if not isinstance(oracle, SyntheticColocationOracle):
        return copy.deepcopy(problem.config.catalog), 0
    c = problem.config
    data = copy.deepcopy(c.catalog)
    data.update(performance_mode="profiled", memory_mode="profiled",
                colocation_profiles={}, energy_profiles={})
    models = tuple(problem.states)
    count = 0
    for ns in itertools.product(*(problem.states[m] for m in models)):
        budget.check()
        x = {m: n for m, n in zip(models, ns) if n}
        options = []
        for task, rate in sorted(c.workload.items()):
            eligible = [m for m in x if task in c.catalog["models"][m]["supported_tasks"]]
            options.append([{m: rate * f.numerator / f.denominator for m, f in row}
                            for row in fraction_rows(eligible, K)])
        for rows in itertools.product(*options):
            budget.check()
            routing = dict(zip(sorted(c.workload), rows))
            generated = oracle.catalog_for(problem, x, routing, trace)
            data["colocation_profiles"].update(generated["colocation_profiles"])
            data["energy_profiles"].update(generated["energy_profiles"])
            count += 1
    return data, count


def legacy_baseline(problem, oracle, solver, K, seconds, calls):
    from .contracts import BudgetExhausted
    if len(problem.train_traces) != 1:
        return dict(algorithm=f"{solver}_K{K}", status="UNSUPPORTED_MULTITRACE_TRAINING_OBJECTIVE",
                    winner=None, evidence_kind=oracle.evidence_kind)
    trace = problem.train_traces[0]
    c = problem.config
    # Old public planners generate their own trace. Never silently substitute it.
    if trace.arrivals != make_trace(c.workload, c.horizon_seconds, trace.seed, c.arrival_process):
        return dict(algorithm=f"{solver}_K{K}", status="UNSUPPORTED_CUSTOM_TRACE",
                    winner=None, evidence_kind=oracle.evidence_kind)
    budget = Budget(seconds=seconds, max_predictions=calls)
    try:
        data, materialized = materialize_grid_catalog(problem, oracle, K, budget, trace)
        remaining = budget.deadline-time.perf_counter()
        if remaining <= 0:
            raise BudgetExhausted("PREPARATION_TIME_LIMIT")
        baseline_config = replace(c, catalog=data, solver=solver, route_grid=K,
            random_seed=trace.seed, milp_capacity_witness=False,
            route_pair_denominators_by_task={}, milp_max_iterations=calls,
            milp_time_limit_seconds=remaining,
            # DFS leaves consume search states, so this conservative cap also
            # limits prediction calls, with the actual state cost disclosed.
            search_max_states=calls if solver == "dfs" else c.search_max_states)
        raw = DeploymentSystem(baseline_config).plan(replica_options={m: [n for n in ns if n]
            for m, ns in problem.states.items()})
        actual_calls = raw["metrics"]["full_prediction_calls"]
        winner = raw["winner"]
        if winner and winner["prediction"].get("U") is None:
            winner = None
        convergence = []
        best = None
        for i, row in enumerate(raw.get("history", [])):
            value = row.get("candidate_U")
            if value is not None:
                best = value if best is None else max(best, value)
            convergence.append(dict(full_prediction_calls=i+1, best_feasible_utility=best))
        return dict(algorithm=f"{solver}_K{K}", status=raw["status"], winner=winner,
            finite_grid_certified=raw["optimality_certified"], continuous_optimality=False,
            full_prediction_calls=actual_calls, predictor_leaf_calls=3*actual_calls,
            milp_solve_calls=raw["metrics"].get("milp_calls", 0),
            elapsed_seconds=time.perf_counter()-budget.start,
            fixture_profiles_prepared=materialized, preparation_included=True,
            dfs_state_budget_is_stricter_than_prediction_budget=solver == "dfs",
            convergence=convergence, raw=raw, evidence_kind=oracle.evidence_kind)
    except BudgetExhausted as exc:
        return dict(algorithm=f"{solver}_K{K}", status=str(exc), winner=None,
            elapsed_seconds=time.perf_counter()-budget.start,
            full_prediction_calls=0, evidence_kind=oracle.evidence_kind)


def make_problem(config, trace_seeds, replica_options=None):
    traces = [TraceSpec(f"{split}-{seed}", seed, make_trace(config.workload,
        config.horizon_seconds, seed, config.arrival_process), split)
        for split, seeds in trace_seeds.items() for seed in seeds]
    return ContinuousProblem(config, traces, replica_options)


def compact_winner(result):
    w = result.get("winner")
    if not w:
        return None
    if "best_feasible_utility" in w:
        return dict(deployment=w["deployment"], routing=w["routing_qps"],
                    utility=w["best_feasible_utility"])
    return dict(deployment=w["deployment"], routing=w["routing"], utility=w["score"])


def evaluate_holdout(problem, oracle, winner, seconds, calls):
    traces = [t for t in problem.traces if t.split != "train"]
    if not traces or not winner:
        return None
    e = ContinuousRoutingEvaluator(problem, oracle, budget=Budget(seconds, calls))
    # Report each split independently; never select a different D or route.
    from .contracts import BudgetExhausted
    output = {}
    for split in ("validation", "test"):
        selected = [t for t in traces if t.split == split]
        if selected:
            try:
                output[split] = e.candidate(winner["deployment"], winner["routing"], traces=selected)
            except BudgetExhausted as exc:
                output[split] = dict(status=str(exc))
    output["cost"] = e.budget.report()
    return output


def run_experiment(config_path, output, *, algorithms=None, seeds=None,
                   outer=None, predictions=None, seconds=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    spec, original = load_experiment(config_path)
    algorithms = algorithms or spec.get("algorithms", ["de_enumeration", "de_milp", "milp_K2", "dfs_K2"])
    seeds = seeds or spec.get("optimizer_seeds", [11, 29, 47])
    settings = spec.get("budget", {})
    outer = outer or settings.get("outer", 64)
    predictions = predictions or settings.get("predictions", 2000)
    seconds = seconds or settings.get("seconds", 30)
    output.mkdir(parents=True)
    (output/"configuration.json").write_text(json.dumps(dict(spec=spec,
        overrides=dict(algorithms=algorithms, seeds=seeds, outer=outer,
                       predictions=predictions, seconds=seconds)), indent=2))
    records = []
    problems = {}
    start = time.perf_counter()
    for workload_name, workload in spec.get("workloads", {"configured": original.config.workload}).items():
        config = replace(original.config, workload=workload)
        problem = make_problem(config, spec.get("trace_seeds", {"train": [config.random_seed]}), spec.get("replica_options"))
        problems[workload_name] = dict(identity=problem.identity, objective_id=problem.objective_id,
            config=problem.config.as_dict(), states=problem.states,
            traces=[asdict(t) for t in problem.traces])
        (output/"problems.json").write_text(json.dumps(problems,indent=2,allow_nan=False))
        for seed in seeds:
            for algorithm in algorithms:
                run_start = time.perf_counter()
                oracle = oracle_from_spec(spec)
                if algorithm.startswith("de_"):
                    budget = Budget(seconds, predictions, outer)
                    e = ContinuousRoutingEvaluator(problem, oracle,
                        inner=algorithm.removeprefix("de_"), budget=budget,
                        cache=spec.get("cache", True), inner_seconds=spec.get("inner_seconds"),
                        inner_bounds=spec.get("inner_bounds", True),
                        inner_warm_start=spec.get("inner_warm_start", False))
                    fit = DERoutingOptimizer(e, seed=seed, **spec.get("de", {})).solve()
                    fit["evaluations"] = e.history
                    fit.update(full_prediction_calls=budget.report()["full_prediction_calls"],
                        predictor_leaf_calls=budget.report()["predictor_leaf_calls"],
                        milp_solve_calls=budget.report()["milp_solve_calls"],
                        elapsed_seconds=budget.report()["elapsed_seconds"])
                else:
                    solver, k = algorithm.split("_K")
                    if solver not in {"milp", "dfs"}:
                        raise ValueError("UNKNOWN_BASELINE")
                    fit = legacy_baseline(problem, oracle, solver, int(k), seconds, predictions)
                winner = compact_winner(fit)
                fit.update(workload=workload_name, optimizer_seed=seed,
                    problem_sha256=problem.identity, oracle_version=oracle.version,
                    objective_id=problem.objective_id,
                    holdout=evaluate_holdout(problem, oracle, winner, seconds, predictions))
                filename = f"{workload_name}-{algorithm}-seed{seed}.json"
                (output/filename).write_text(json.dumps(fit, indent=2, allow_nan=False))
                row = dict(workload=workload_name, algorithm=algorithm, seed=seed,
                    status=fit["status"], utility=winner["utility"] if winner else None,
                    deployment=winner["deployment"] if winner else None,
                    routing=winner["routing"] if winner else None,
                    full_prediction_calls=fit.get("full_prediction_calls", 0),
                    predictor_leaf_calls=fit.get("predictor_leaf_calls", 0),
                    milp_solve_calls=fit.get("milp_solve_calls", 0),
                    elapsed_seconds=fit.get("elapsed_seconds"),
                    end_to_end_seconds=time.perf_counter()-run_start,
                    evidence_kind=oracle.evidence_kind, raw_file=filename)
                records.append(row)
                with (output/"results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
    import csv
    with (output/"summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = dict(records=records, total_wall_seconds=time.perf_counter()-start,
        numpy=np.__version__, oracle_kind=spec.get("oracle", {}).get("kind", "legacy"),
        conclusion="CPU algorithm evidence only; no physical predictor accuracy or native gains",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        implementation_sha256=digest({p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")}))
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary
