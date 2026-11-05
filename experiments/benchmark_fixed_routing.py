"""Paired fixed-route inner costs and finite-trace routing plateaus."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsdeploy.continuous import Budget, ContinuousRoutingEvaluator, ContinuousRouteDomain
from gsdeploy.continuous.contracts import load_experiment
from gsdeploy.continuous.benchmark import oracle_from_spec
from gsdeploy.continuous.predictor_adapter import assignment_digest


def run(config, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    spec, problem = load_experiment(config)
    output.mkdir(parents=True)
    domain = ContinuousRouteDomain(problem)
    routes = list(domain.endpoints()) + [domain.decode([u] * domain.dimension)
                                        for u in (.1732050807568877, .3819660112501051, .6180339887498948)]
    rows = []
    for i, route in enumerate(routes):
        pair = []
        # Alternate ordering to avoid consistently rewarding the warmed solver.
        order = ("enumeration", "milp") if i % 2 == 0 else ("milp", "enumeration")
        for inner in order:
            evaluator = ContinuousRoutingEvaluator(problem, oracle_from_spec(spec), inner,
                Budget(30, 10000, 10), cache=False)
            row = evaluator.evaluate(route)
            pair.append(row)
            (output/f"route-{i}-{inner}.json").write_text(json.dumps(row, indent=2, allow_nan=False))
            rows.append(dict(route=i, inner=inner, utility=row["best_feasible_utility"],
                certified=row["certified"], seconds=row["evaluation_seconds"],
                predictions=row["full_prediction_calls"], milp_calls=row["milp_solve_calls"],
                domain_size=row["deployment_domain_size"],
                active_models=len(row["activation"])))
        if pair[0]["certified"] != pair[1]["certified"] or pair[0]["best_feasible_utility"] != pair[1]["best_feasible_utility"]:
            raise AssertionError("Paired inner solvers disagree")
    with (output/"inner-cost.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Diagnostic only, not an optimizer: tiny perturbations of continuous shares
    # show that assignment equality does not justify whole-prediction caching.
    oracle = oracle_from_spec(spec)
    evaluator = ContinuousRoutingEvaluator(problem, oracle, budget=Budget(30,10000,1000))
    deployment = {m: 1 for m in problem.states}
    plateau = []
    for i in range(41):
        u = .4 + (i-20)*1e-5
        shares = domain.decode([u]*domain.dimension)
        routing = domain.to_qps(shares)
        candidate = evaluator.candidate(deployment, routing)
        plateau.append(dict(coordinate=u, assignment=assignment_digest(deployment,routing,problem.train_traces[0]),
                            utility=candidate["utility"]))
    (output/"plateaus.json").write_text(json.dumps(plateau,indent=2,allow_nan=False))
    (output/"summary.json").write_text(json.dumps(dict(pairs=len(routes), all_agree=True,
        evidence_kind=oracle.evidence_kind, rows=rows,
        plateau_assignment_classes=len({r["assignment"] for r in plateau}),
        plateau_utility_classes=len({r["utility"] for r in plateau})),indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    run(args.config,args.output)
