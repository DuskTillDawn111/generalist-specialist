"""Compare the target BO/MADS bilevel algorithm with retained MILP baselines.

CPU fixtures only: this is software verification, not GPU qualification.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsdeploy.config import SystemConfig
from gsdeploy.continuous import (ActivationAwareBilevelOptimizer, Budget,
    ContinuousRoutingEvaluator, ImmutableContinuousProblem, SessionPredictorOracle)
from gsdeploy.continuous.predictor_adapter import source_identity
from experiments.benchmark_c1_lite import run as run_baseline, utility


METHODS = ("Joint-K1", "Joint-K2", "C1", "C1-Lite-budget", "Activation-BO-MADS",
           "Activation-BO", "Activation-MADS", "Activation-BO-MADS-Joint")


def run(config, method, seed, seconds, predictions, routes):
    if not method.startswith("Activation-"):
        return run_baseline(config, method, seed, seconds=seconds, predictions=predictions, routes=routes)
    budget = Budget(seconds, predictions, routes)
    problem = ImmutableContinuousProblem(config)
    with SessionPredictorOracle(problem) as oracle:
        evaluator = ContinuousRoutingEvaluator(problem, oracle, budget=budget)
        result = ActivationAwareBilevelOptimizer(evaluator, seed=seed,
            initial_endpoints=4, inner_seconds=min(1., seconds), inner_predictions=4,
            inner_milp_calls=8, poll_evaluations=4,
            use_bo=method != "Activation-MADS", use_mads=method != "Activation-BO",
            joint_seed_k=1 if method.endswith("-Joint") else None).solve()
        result["engine_stats"] = oracle.engine_stats()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/paper_fifo_reference_v2_cpu_fixture.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=4.)
    parser.add_argument("--predictions", type=int, default=96)
    parser.add_argument("--routes", type=int, default=48)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    args = parser.parse_args()
    c = SystemConfig.load(args.config)
    if not c.catalog.get("paper_profiled_fifo_v1", {}).get("allow_cpu_fixture"):
        parser.error("Explicit CPU fixture required; real qualification is a separate experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    paths = sorted((ROOT/"gsdeploy/continuous").glob("*.py"))+[
        Path(__file__), ROOT/"experiments/benchmark_c1_lite.py", args.config.resolve()]
    manifest = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save("source-manifest.json", dict(files=manifest, predictor_source_identity=source_identity()))
    save("configuration.json", dict(config=c.as_dict(), methods=args.methods, seeds=args.seeds,
        seconds=args.seconds, predictions=args.predictions, routes=args.routes,
        scope="CPU fixture; shared limits, distinct proposal trajectories; not physical qualification"))
    records, raw_runs = [], []
    for seed in args.seeds:
        offset = seed % len(args.methods)
        methods = args.methods[offset:]+args.methods[:offset]
        for method in methods:
            raw = run(c, method, seed, args.seconds, args.predictions, args.routes)
            save(f"{method}-{seed}.json", raw)
            b = raw["budget"]
            row = dict(method=method, seed=seed, utility=utility(raw), status=raw["status"],
                total_seconds=b["elapsed_seconds"], full_predictions=b["full_prediction_calls"],
                milp_calls=b["milp_solve_calls"], outer_evaluations=b["outer_evaluations"],
                gp_certified_observations=raw.get("surrogate", {}).get("training_observations"),
                excluded_partial=raw.get("surrogate", {}).get("excluded_partial"),
                activation_spaces=raw.get("activation_spaces", {}).get("discovered"),
                mads_evaluations=sum(r["source"] == "mads_poll" for r in raw.get("events", [])),
                native_qualified=False)
            records.append(row); raw_runs.append((method, seed, raw))
            print(json.dumps(row, ensure_ascii=False), flush=True)
    save("comparison.json", dict(rows=records, physical_qualification=False,
        activation_and_outer_initialization_differ=True, continuous_optimality=False,
        joint_initializer_optional=True))
    with (args.output/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method, seed, raw in raw_runs:
        if seed != args.seeds[0]:
            continue
        events = [r for r in raw.get("convergence", []) if r.get("best_feasible_utility") is not None]
        if events:
            axes[0].step([r["elapsed_seconds"] for r in events], [r["best_feasible_utility"] for r in events],
                         where="post", label=method)
            axes[1].step([r["full_prediction_calls"] for r in events], [r["best_feasible_utility"] for r in events],
                         where="post", label=method)
    axes[0].set_xlabel("Total elapsed seconds"); axes[1].set_xlabel("Full prediction queries")
    for ax in axes:
        ax.set_ylabel("Best U (CPU fixture)"); ax.grid(alpha=.2)
    axes[1].legend(fontsize=7); fig.tight_layout()
    fig.savefig(args.output/"comparison.png", dpi=150); plt.close(fig)
    assert all(hashlib.sha256((ROOT/p if not Path(p).is_absolute() else Path(p)).read_bytes()).hexdigest() == h
               for p,h in manifest.items()), "SOURCE_CHANGED_DURING_COMPARISON"


if __name__ == "__main__":
    main()
