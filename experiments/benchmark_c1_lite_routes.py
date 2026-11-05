"""Same-route exact-inner ablation, separate from outer search comparisons."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsdeploy.config import SystemConfig
from gsdeploy.continuous import (Budget, ImmutableContinuousProblem, SessionPredictorOracle,
    ContinuousRoutingEvaluator, DeploymentCandidatePool, ProgressiveFixedRoutingSolver)
from gsdeploy.continuous.predictor_adapter import source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/paper_fifo_reference_v2_cpu_fixture.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c = SystemConfig.load(args.config)
    if not c.catalog.get("paper_profiled_fifo_v1", {}).get("allow_cpu_fixture"):
        parser.error("Explicit CPU fixture required")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for method in ("original_fixed_milp", "pool_always_exact", "pool_budget_then_refine"):
        budget = Budget(8., 128, 64)
        p = ImmutableContinuousProblem(c)
        with SessionPredictorOracle(p) as oracle:
            e = ContinuousRoutingEvaluator(p, oracle, inner="milp_bounds", budget=budget)
            solver = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(p), policy="budget",
                                                   inner_predictions=2, inner_milp_calls=2)
            routes = [{t: {models[0]: ratio, models[-1]: 1.-ratio} for t, models in e.domain.edges.items()}
                      for ratio in (0., 1., .3141592653589793, .6180339887498948, .41, .73)]
            rows = []
            for shares in routes:
                if method == "original_fixed_milp":
                    row = e.evaluate(shares)
                elif method == "pool_always_exact":
                    row = solver.evaluate(shares, exact=True)
                else:
                    row = solver.evaluate(shares)
                    if not row["certified"]:
                        row = solver.evaluate(shares, exact=True)
                rows.append(dict(routing_shares=shares, utility=row["best_feasible_utility"],
                                 deployment=row["deployment"], certified=row["certified"]))
            results.append(dict(method=method, routes=rows, budget=budget.report()))
    reference = results[0]["routes"]
    for method in results[1:]:
        for a, b in zip(reference, method["routes"]):
            if not (a["certified"] and b["certified"] and abs(a["utility"]-b["utility"]) < 1e-10):
                raise AssertionError("Same-route inner mismatch")
    output = dict(scope="CPU same-route ablation; independent of outer initialization",
        exact_results_equal=True, results=results, predictor_source_identity=source_identity(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physical_qualification=False)
    (args.output/"comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"exact_results_equal": True, "methods": [{"method": r["method"], **r["budget"]} for r in results]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
