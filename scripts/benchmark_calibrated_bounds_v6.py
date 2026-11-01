#!/usr/bin/env python3
"""CPU fixture comparison of frozen-v1 structural bounds, with fresh processes.

The ablation keeps the exact same predictor/domain/objective, and only returns
to the previous zero service/energy and additive memory search relaxations.
It is not a GPU result or a claim of speed superiority on the actual pool.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def child(algorithm, variant, grid):
    from tests.test_calibrated_search_bounds_v6 import search_fixture, context
    from gsdeploy.milp import JointMilpOptimizer
    from gsdeploy.optimizer import UtilityOptimizer

    config = search_fixture()
    cls = JointMilpOptimizer if algorithm == "milp" else UtilityOptimizer
    search_context = context(config)
    if variant == "plain_dfs":
        if algorithm != "dfs":
            raise ValueError("plain_dfs is a DFS-only variant")
        search_context["dfs_bound_pruning"] = False
    start = time.perf_counter()
    solver = cls(config.catalog, config.workload, config.gpu_memory_mib,
                 config.latency_reference_ms, config.weights, **search_context)
    if variant == "zero_service_energy":
        # Version validation already ran. Only the mathematical search floor is
        # disabled, retaining known complete E and its normal incumbent logic.
        solver.surrogate = None
        solver.separable_service = False
        solver.profiled_memory = False
    result = solver.solve(K=grid)
    total = time.perf_counter() - start
    print(json.dumps(dict(algorithm=algorithm, variant=variant,
        total_search_seconds=total, solve_seconds=result["elapsed_seconds"],
        status=result["status"], optimality_certified=result["optimality_certified"],
        score=result["winner"]["score"] if result["winner"] else None,
        metrics=result["metrics"], utility_upper_bound=result["utility_upper_bound"],
        optimality_gap=result["optimality_gap"], domain=result["domain"],
        cpu_affinity=sorted(os.sched_getaffinity(0)))))


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        "gsdeploy/milp.py", "gsdeploy/optimizer.py", "gsdeploy/surrogate_bounds.py",
        "gsdeploy/calibration.py", "gsdeploy/prediction.py",
        "tests/test_calibrated_search_bounds_v6.py", "scripts/benchmark_calibrated_bounds_v6.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--plain-comparison", action="store_true",
                        help="compare proposed MILP, B&B DFS and exhaustive DFS on the same domain")
    parser.add_argument("--algorithm", choices=["milp", "dfs"])
    parser.add_argument("--variant", choices=["structural_v1", "zero_service_energy", "plain_dfs"])
    args = parser.parse_args()
    if args.child:
        child(args.algorithm, args.variant, args.grid)
        return
    if args.output is None:
        parser.error("--output is required")
    before, runs = hashes(), []
    combinations = ([("milp", "structural_v1"), ("dfs", "structural_v1"), ("dfs", "plain_dfs")]
                    if args.plain_comparison else
                    [(a, v) for a in ("milp", "dfs")
                     for v in ("structural_v1", "zero_service_energy")])
    for repeat in range(args.repeats):
        for algorithm, variant in (combinations if repeat % 2 == 0 else list(reversed(combinations))):
            started = time.perf_counter()
            raw = subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
                "--child", "--algorithm", algorithm, "--variant", variant,
                "--grid", str(args.grid)], cwd=ROOT, text=True)
            row = json.loads(raw)
            row.update(repeat=repeat, process_wall_seconds=time.perf_counter() - started)
            runs.append(row)
    summaries = []
    for algorithm, variant in combinations:
        selected = [r for r in runs if (r["algorithm"], r["variant"]) == (algorithm, variant)]
        summaries.append(dict(algorithm=algorithm, variant=variant,
            median_search_seconds=statistics.median(r["total_search_seconds"] for r in selected),
            median_process_seconds=statistics.median(r["process_wall_seconds"] for r in selected),
            median_predictions=statistics.median(r["metrics"]["full_prediction_calls"] for r in selected),
            median_milp_calls=statistics.median(r["metrics"].get("milp_calls", 0) for r in selected),
            median_utility_bound_evaluations=statistics.median(r["metrics"].get("utility_bound_evaluations", 0)
                                                             for r in selected),
            certificates=sum(r["optimality_certified"] for r in selected), repeats=len(selected)))
    after = hashes()
    scores = [r["score"] for r in runs if r["score"] is not None]
    all_certified = all(r["optimality_certified"] for r in runs)
    unchanged = before == after
    agrees = len(scores) == len(runs) and max(scores) - min(scores) <= 1e-8
    report = dict(scope="synthetic CPU fixture only; frozen predictor optimum, not physical GPU optimum",
        fixture="two tasks, two models, measured replica states 0/1/2, all six utility terms",
        timing="fresh process each run; shared imports and synthetic fit excluded; search construction included",
        source_before=before, source_after=after, source_unchanged=unchanged,
        benchmark_valid=unchanged and all_certified and agrees,
        objective_agrees=agrees,
        comparison=("proposed MILP vs B&B DFS vs exhaustive DFS" if args.plain_comparison
                    else "structural lower bound ablation"),
        all_certified=all_certified,
        score_range=[min(scores), max(scores)] if scores else None,
        summaries=summaries, runs=runs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("runs", "source_before", "source_after")}, indent=2))


if __name__ == "__main__":
    main()
