"""Solve six frozen comparison roles after surrogate validation; CPU only."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsdeploy.config import SystemConfig
from gsdeploy.evidence import digest
from gsdeploy.io import save
from gsdeploy.quality import quality_digest
from gsdeploy.system import DeploymentSystem


def subset_config(config, models):
    """Restrict alternatives while retaining each model's frozen baseline rows."""
    data = copy.deepcopy(config.catalog)
    for field in ("models", "capacity_factors", "memory_table_mib"):
        data[field] = {model: data[field][model] for model in models}
    return replace(config, catalog=data)


def solve_roles(config, output):
    """Save each completed result so a bounded search never loses evidence."""
    if config.route_pair_denominators_by_task:
        raise ValueError("PAIR_ROUTE_DOMAIN_UNSUPPORTED_BY_LEGACY_STATIC_ROLES: use SearchSession")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "system-config.json", config.as_dict())
    candidates = []

    def solve(label, problem=config, **restrictions):
        started = time.monotonic()
        result = DeploymentSystem(problem).plan(**restrictions)
        result["external_wall_seconds"] = time.monotonic() - started
        save(output / (label + ".json"), result)
        candidates.append({"label": label, "search": result})
        print(json.dumps({"role": label, "status": result["status"],
                          "optimality_certified": result.get("optimality_certified", False),
                          "U": result["winner"]["score"] if result.get("winner") else None,
                          "elapsed_seconds": result["external_wall_seconds"]}), flush=True)
        return result

    best = solve("free_search_winner")
    winner = best.get("winner")
    if winner:
        solve("strongest_alternative_deployment", exclude_deployments=[winner["deployment"]])
        solve("strongest_alternative_route", deployment=winner["deployment"], exclude_plans=[winner])
        solve("best_no_split", no_split=True)
        for role, label in (("generalist", "best_generalist_only"), ("specialist", "best_specialist_only")):
            models = [model for model, spec in config.catalog["models"].items() if spec["role"] == role]
            solve(label, subset_config(config, models))
    bundle = {"schema": "gsdeploy.static_search_roles.v1", "config_sha256": quality_digest(config.as_dict()),
              "candidates": candidates, "gpu_started": False,
              "scope": "Frozen calibrated predictor; bounded searches retain gaps and unknown domains",
              "physical_global_optimality_certified": False}
    save(output / "search-results.json", bundle)
    save(output / "manifest.json", {"files": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
                                    "gpu_started": False})
    return bundle


def compare_same_domain(config, output, *, repeats=5, seconds=10):
    """Run interleaved MILP/DFS on an explicitly restricted identical domain."""
    if config.route_pair_denominators_by_task:
        raise ValueError("PAIR_ROUTE_DOMAIN_UNSUPPORTED_BY_LEGACY_STATIC_COMPARISON: use SearchSession")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    small = subset_config(config, ["generalist_2", "text"])
    small = replace(small, route_grid=4, milp_time_limit_seconds=seconds, search_max_states=1000000)
    options = {model: [1] for model in small.catalog["models"]}
    records = []
    for repeat in range(repeats):
        for solver in (["milp", "dfs", "plain_dfs"] if repeat % 2 == 0 else ["plain_dfs", "dfs", "milp"]):
            effective = replace(small, solver="dfs" if solver == "plain_dfs" else solver)
            started = time.monotonic()
            if solver == "plain_dfs":
                from gsdeploy.optimizer import UtilityOptimizer
                result = UtilityOptimizer(
                    effective.catalog, effective.workload, effective.gpu_memory_mib,
                    effective.latency_reference_ms, effective.weights,
                    search_max_states=effective.search_max_states, search_policy=effective.search_policy,
                    milp_capacity_witness=False, dfs_time_limit_seconds=seconds,
                    dfs_bound_pruning=False, **DeploymentSystem(effective).context(),
                ).solve(replica_options=options, K=effective.route_grid)
            else:
                result = DeploymentSystem(effective).plan(replica_options=options)
            records.append({"repeat": repeat, "solver": solver, "external_wall_seconds": time.monotonic() - started,
                            "result": result})
    objective_agrees, certified_agrees, identities_agree = [], [], []
    for repeat in range(repeats):
        pair = [r["result"] for r in records if r["repeat"] == repeat]
        values = [r["winner"]["score"] if r.get("winner") else None for r in pair]
        agree = all(v is not None for v in values) and max(values) - min(values) <= config.milp_absolute_gap
        objective_agrees.append(agree)
        certified_agrees.append(agree and all(r.get("optimality_certified") for r in pair))
        plan_ids = [quality_digest({key: r["winner"][key] for key in ("deployment", "routing")})
                    if r.get("winner") else None for r in pair]
        identities_agree.append(all(key is not None for key in plan_ids) and len(set(plan_ids)) == 1)
    timing = {solver: statistics.median(r["external_wall_seconds"] for r in records if r["solver"] == solver)
              for solver in ("milp", "dfs", "plain_dfs")}
    report = {"schema": "gsdeploy.same_domain_solver_comparison.v1", "config": small.as_dict(),
              "replica_options": options, "records": records, "objective_agrees": objective_agrees,
              "certified_objective_agrees": certified_agrees, "deployment_route_identities_agree": identities_agree,
              "median_external_wall_seconds": timing,
              "observed_speedup_milp_over_plain_dfs": timing["plain_dfs"] / timing["milp"],
              "observed_speedup_milp_over_bounded_dfs": timing["dfs"] / timing["milp"], "gpu_started": False,
              "scope": "Seven-model study's restricted g2/text n0/n1 domain; no full-pool proof inferred",
              "timing_scope": "Interleaved wall clock on this host; concurrent GPU jobs and host contention may affect timings"}
    save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--route-grid", type=int)
    parser.add_argument("--compare-small", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    config = SystemConfig.load(args.config)
    if config.catalog.get("performance_mode") == "calibrated_replay_v2":
        from gsdeploy.calibration_v2 import validated_replay_spec
        validated_replay_spec(config.catalog, rates=config.workload)
    else:
        from gsdeploy.calibration import validated_spec
        validated_spec(config.catalog, rates=config.workload)
    changes = {}
    if args.time_limit is not None:
        changes["milp_time_limit_seconds"] = args.time_limit
    if args.route_grid is not None:
        changes["route_grid"] = args.route_grid
    config = replace(config, **changes)
    if args.compare_small:
        report = compare_same_domain(config, args.output, repeats=args.repeats,
                                     seconds=config.milp_time_limit_seconds)
        result = {"objective_agrees": report["objective_agrees"],
                  "certified_objective_agrees": report["certified_objective_agrees"]}
    else:
        result = {"roles": len(solve_roles(config, args.output)["candidates"])}
    print(json.dumps({**result, "gpu_started": False}))


if __name__ == "__main__":
    main()
