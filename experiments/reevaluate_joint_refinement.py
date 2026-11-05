"""Small Joint + continuous-route re-evaluation with one declared predictor.

Uses an unchanged Joint optimizer and the existing independent DE/local
refinement. No profiling, predictor fitting, deployment neighbourhood or GP-BO.
"""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsdeploy.config import SystemConfig
from gsdeploy.continuous import (Budget, DERoutingOptimizer,
    FixedDeploymentRoutingEvaluator, ImmutableContinuousProblem,
    JointMILPContinuousRefinement, SessionPredictorOracle)
from gsdeploy.continuous.refinement import project_shares
from gsdeploy.search_session import SearchSession
from experiments.joint_scalability import probes
from experiments.joint_continuous_refinement import off_grid
from experiments.diagnose_continuous_phase1 import assignment


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def engine_stats(session):
    if session.config.catalog.get("performance_mode") != "paper_profiled_fifo_v1":
        return None
    engine = session.module("paper_forecast").session(session.config.catalog)
    return dict(event_engine_calls=engine.engine_calls, uncached_scenario_queries=engine.calls,
                cache_hits=engine.hits)


def mapping(config, winner):
    if config.catalog.get("performance_mode") == "paper_profiled_fifo_v1":
        rows = winner["prediction"]["request_mapping"]
        return dict(requests=rows, counts=dict(Counter(r["alias"] for r in rows)))
    return assignment(config, winner["deployment"], winner["routing"])


def fixed_diagnostic(config, deployment, seed_routing, seconds, routes, seed):
    """Explicit supplied deployment diagnostic, never a Joint-selected result."""
    budget = Budget(seconds, max_predictions=routes*2, max_outer=routes)
    problem = ImmutableContinuousProblem(config)
    with SessionPredictorOracle(problem) as oracle:
        evaluator = FixedDeploymentRoutingEvaluator(problem, oracle, deployment, budget)
        initial = project_shares(problem, deployment, seed_routing)
        raw = DERoutingOptimizer(evaluator, seed=seed, popsize=3, maxiter=100,
                                initial_shares=initial).solve()
        return dict(scope="SUPPLIED_DEPLOYMENT_DIAGNOSTIC_NOT_JOINT_WINNER",
            deployment=deployment, dimension=evaluator.domain.dimension,
            engine_stats=engine_stats(oracle.session), history=evaluator.history, result=raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/paper_fifo_cpu_fixture.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--replicas", type=int)
    parser.add_argument("--seconds", type=float, default=4.)
    parser.add_argument("--routes", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11,29,47])
    parser.add_argument("--diagnostic-deployment", help="Optional explicit JSON deployment")
    args = parser.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = SystemConfig.load(args.config)
    if args.models:
        catalog = dict(config.catalog)
        for field in ("models", "capacity_factors", "memory_table_mib"):
            catalog[field] = {m: catalog[field][m] for m in args.models}
        config = replace(config, catalog=catalog)
    config = replace(config, milp_time_limit_seconds=args.seconds)
    states = ({m:[args.replicas] for m in config.catalog["models"]} if args.replicas else None)
    # Freeze source evidence before any run; report the effective domain.
    sources = [ROOT/"gsdeploy"/"milp.py", ROOT/"gsdeploy"/"optimizer.py",
               Path(__file__), args.config.resolve()]
    sources += list((ROOT/"gsdeploy/continuous").rglob("*.py"))
    sources += [ROOT/"gsdeploy/search_session.py",ROOT/"gsdeploy/paper_forecast.py"]
    sources += list((ROOT/"gsdeploy/forecasting").rglob("*.py"))
    manifest = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    save(out/"source-manifest.json", manifest)
    save(out/"effective-config.json", config.as_dict())
    save(out/"plan.json", dict(mode=config.catalog["performance_mode"], replica_options=states,
        seconds_per_pipeline=args.seconds, max_routes=args.routes, seeds=args.seeds,
        deployment_neighborhood=False, new_measurements=0, fitting=False,
        CPU_fixture=config.catalog.get("paper_profiled_fifo_v1",{}).get("allow_cpu_fixture",False)))
    rows = []
    at = time.perf_counter()
    with SearchSession(ROOT, config.as_dict()) as session:
        optimizer = session.optimizer("milp")
        optimizer.no_split = False
        setup = time.perf_counter()-at
        # The Joint solve gets the same remaining pipeline time, including setup.
        optimizer.time_limit = max(.001,args.seconds-setup)
        with probes(session.module("milp"), optimizer) as timing:
            baseline = optimizer.solve(K=config.route_grid,replica_options=states)
        counters = engine_stats(session)
    baseline_seconds = time.perf_counter()-at
    winner = baseline.get("winner")
    save(out/"joint.json",dict(result=baseline,timing=timing,setup_seconds=setup,
                              total_seconds=baseline_seconds,engine_stats=counters))
    rows.append(dict(algorithm="joint",seed=None,total_seconds=baseline_seconds,
        utility=winner["score"] if winner else None,improvement=0. if winner else None,
        prediction_bundles=timing["prediction_calls"],milp_solves=len(timing["rounds"]),
        refinement_milp_solves=0,route_evaluations=0,fixed_route_dimension=None,
        status=baseline["status"],engine_stats=counters))
    if winner:
        save(out/"joint-routing.json",dict(deployment=winner["deployment"],routing=winner["routing"],
            off_grid=off_grid(config,winner["routing"]),fifo=mapping(config,winner)))
    for method in ("de","local"):
        for seed in args.seeds:
            raw = JointMILPContinuousRefinement(config, states, seconds=args.seconds,
                max_predictions=max(64,args.routes*4), max_routes=args.routes,
                seed=seed,method=method,neighborhood=False).solve()
            save(out/f"refine-{method}-seed{seed}.json", raw)
            w = raw["winner"]
            fixed_milp = sum(h["milp_calls"] for h in raw["histories"])
            if w:
                assert w["utility"] >= raw["initial_winner"]["utility"]
                assert raw["deployment_preserved"] and fixed_milp == 0
                save(out/f"routing-{method}-seed{seed}.json",dict(deployment=w["deployment"],
                    routing=w["routing"],off_grid=off_grid(config,w["routing"]),fifo=mapping(config,w)))
            row = dict(algorithm="joint+"+method,seed=seed,total_seconds=raw["total_seconds"],
                utility=w["utility"] if w else None,improvement=raw["utility_improvement"],
                prediction_bundles=raw["budget"]["full_prediction_calls"],
                milp_solves=raw["budget"]["milp_solve_calls"],refinement_milp_solves=fixed_milp,
                route_evaluations=raw["budget"]["outer_evaluations"],
                fixed_route_dimension=raw["fixed_route_dimension"],status=raw["status"],
                engine_stats=raw["predictor_engine_stats"])
            rows.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    if args.diagnostic_deployment and winner:
        d = json.loads(args.diagnostic_deployment)
        diagnostic = fixed_diagnostic(config,d,winner["routing"],args.seconds,args.routes,args.seeds[0])
        save(out/"supplied-deployment-diagnostic.json",diagnostic)
    assert manifest == {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = dict(schema="gsdeploy.joint_refinement_reevaluation.v1",
        prediction_mode=config.catalog["performance_mode"],rows=rows,
        source_unchanged=True,new_gpu_measurements=0,independent_new_predictor_accuracy_qualified=False,
        final_real_gain_claim=False,elapsed_seconds=time.perf_counter()-started,
        scope="Software mechanism check for CPU fixture; legacy replay results remain predictions under its original contract")
    save(out/"report.json", report)
    print(json.dumps(dict(output=str(out),seconds=report["elapsed_seconds"]),ensure_ascii=False))


if __name__ == "__main__":
    main()
