"""Small CPU engineering comparison; physical qualification is never implied."""

import argparse
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsdeploy.config import SystemConfig
from gsdeploy.continuous import (Budget, ImmutableContinuousProblem, SessionPredictorOracle,
    ContinuousRoutingEvaluator, DERoutingOptimizer, C1LiteOptimizer)
from gsdeploy.continuous.multi_candidate import JointDeploymentCandidates, MultiCandidateContinuousRefinement
from gsdeploy.continuous.predictor_adapter import source_identity
from gsdeploy.continuous.contracts import BudgetExhausted


def utility(raw):
    winner = raw.get("winner")
    if winner is None:
        return None
    for name in ("best_feasible_utility", "utility", "score"):
        if name in winner:
            return winner[name]
    raise ValueError("UNRECOGNIZED_WINNER")


def run(c, method, seed, *, seconds, predictions, routes):
    if method == "C3":
        return MultiCandidateContinuousRefinement(c, k=3, seconds=seconds,
            max_predictions=predictions, max_routes=routes, seed=seed).solve()
    budget = Budget(seconds, predictions, routes)
    problem = ImmutableContinuousProblem(c)
    with SessionPredictorOracle(problem) as oracle:
        if method.startswith("Joint-K"):
            extractor = JointDeploymentCandidates(problem, oracle, budget)
            raw = extractor.solve_one(K=int(method.split("K")[1]), excluded=[])
            convergence, best = [], None
            for event in extractor.events:
                if event["utility"] is not None:
                    best = max(best, event["utility"]) if best is not None else event["utility"]
                convergence.append(dict(elapsed_seconds=event["seconds"], full_prediction_calls=event["predictions"],
                                        best_feasible_utility=best))
            return dict(algorithm=method, winner=extractor.best, status=raw["status"],
                finite_domain_certified=raw["optimality_certified"], continuous_optimality=False,
                convergence=convergence,
                budget=budget.report(), raw=raw, evidence_kind="CPU_fixture")
        e = ContinuousRoutingEvaluator(problem, oracle, inner="milp_bounds", budget=budget)
        if method == "C1":
            raw = DERoutingOptimizer(e, seed=seed, popsize=3).solve()
            raw["history"] = e.history
        else:
            policy = method.rsplit("-", 1)[1] if method.rsplit("-", 1)[1] in {"always", "gap", "budget"} else "budget"
            raw = C1LiteOptimizer(e, seed=seed, policy=policy, population_size=6,
                endpoint_limit=2, face_limit=1, inner_seconds=min(1., seconds),
                inner_predictions=4, inner_milp_calls=8,
                pool_max_size=1 if method == "C1-Lite-pool1" else 16,
                exact_inner=method == "C1-Lite-exact",
                use_bounds=method != "C1-Lite-no-bounds").solve()
        raw["engine_stats"] = oracle.engine_stats()
        return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/paper_fifo_reference_v2_cpu_fixture.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=4.)
    parser.add_argument("--predictions", type=int, default=64)
    parser.add_argument("--routes", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--methods", nargs="+", default=["Joint-K1", "Joint-K2", "C1",
        "C1-Lite-exact", "C1-Lite-always", "C1-Lite-gap", "C1-Lite-budget", "C1-Lite-pool1", "C1-Lite-no-bounds", "C3"])
    args = parser.parse_args()
    supported = {"Joint-K1", "Joint-K2", "C1", "C1-Lite-exact", "C1-Lite-always",
                 "C1-Lite-gap", "C1-Lite-budget", "C1-Lite-pool1", "C1-Lite-no-bounds", "C3"}
    if set(args.methods)-supported:
        parser.error("Unknown comparison method")
    c = SystemConfig.load(args.config)
    # This entry point is deliberately software-only. A future qualified real
    # comparison needs an explicit qualification manifest, not this flag.
    if not c.catalog.get("paper_profiled_fifo_v1", {}).get("allow_cpu_fixture"):
        parser.error("This runner only accepts explicitly declared CPU fixtures")
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    save("configuration.json", dict(config=c.as_dict(), source=str(args.config.resolve()),
        seconds=args.seconds, max_predictions=args.predictions, max_routes=args.routes,
        seeds=args.seeds, methods=args.methods, physical_qualification=False))
    rows, all_raw = [], []
    for seed in args.seeds:
        # Rotate execution order to avoid always charging cold imports to Joint.
        methods = args.methods[seed % len(args.methods):]+args.methods[:seed % len(args.methods)]
        for method in methods:
            raw = run(c, method, seed, seconds=args.seconds, predictions=args.predictions, routes=args.routes)
            save(f"{method}-{seed}.json", raw)
            b = raw["budget"]
            row = dict(method=method, seed=seed, utility=utility(raw), status=raw["status"],
                total_seconds=b["elapsed_seconds"], full_predictions=b["full_prediction_calls"],
                milp_calls=b["milp_solve_calls"], outer_evaluations=b["outer_evaluations"],
                inner_sessions=raw.get("inner_sessions"), heuristic_selections=raw.get("heuristic_selections"),
                pool_size=raw.get("pool", {}).get("size"), native_qualified=False)
            rows.append(row); all_raw.append((row, raw))
            print(json.dumps(row, ensure_ascii=False), flush=True)
    targets = {seed: max((r["utility"] for r in rows if r["seed"] == seed and r["method"].startswith("Joint")
                         and r["utility"] is not None), default=None) for seed in args.seeds}
    for row, raw in all_raw:
        target = targets[row["seed"]]
        hit = [r["elapsed_seconds"] for r in raw.get("convergence", [])
               if target is not None and r.get("best_feasible_utility") is not None
               and r["best_feasible_utility"] >= target-1e-9]
        row.update(joint_target=target, first_target_seconds=min(hit) if hit else None,
                   target_time_available=bool(hit))
    save("comparison.json", dict(scope="CPU engineering only", rows=rows,
        real_predictor_qualified=False, hybrid_gate_passed=False,
        original_c1_and_progressive_de_have_different_initialization=True))
    with (args.output/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    paths = list((ROOT/"gsdeploy/continuous").glob("*.py"))+[Path(__file__), ROOT/"gsdeploy/c1_qualification.py"]
    save("source-manifest.json", dict(predictor_source_identity=source_identity(), files={str(p.relative_to(ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for row, raw in all_raw:
        if row["seed"] != args.seeds[0]:
            continue
        events = [r for r in raw.get("convergence", []) if r.get("best_feasible_utility") is not None]
        if events:
            axes[0].step([r["elapsed_seconds"] for r in events], [r["best_feasible_utility"] for r in events],
                         where="post", label=row["method"])
            axes[1].step([r["full_prediction_calls"] for r in events], [r["best_feasible_utility"] for r in events],
                         where="post", label=row["method"])
    axes[0].set_xlabel("Total elapsed seconds"); axes[1].set_xlabel("Full prediction queries")
    for ax in axes:
        ax.set_ylabel("Best U (CPU fixture)"); ax.grid(alpha=.2)
    axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(args.output/"comparison.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
