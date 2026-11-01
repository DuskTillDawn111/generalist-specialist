#!/usr/bin/env python3
"""Six static comparison roles with isolated memo and frozen predictor sources."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_wrapper(path):
    spec = importlib.util.spec_from_file_location("isolated_study_session", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def solve_roles(config, output, source_root, *, algorithm="milp", wrapper_path=None):
    output, source_root = Path(output), Path(source_root).resolve()
    wrapper_path = (Path(wrapper_path) if wrapper_path else source_root / "gsdeploy" / "search_session.py").resolve()
    wrapper = load_wrapper(wrapper_path)
    source_before = {str(p.relative_to(source_root)): digest(p) for p in sorted((source_root / "gsdeploy").rglob("*.py"))}
    wrapper_hash = digest(wrapper_path)
    with wrapper.SearchSession(source_root, config) as validated:
        original = json.loads(json.dumps(validated.config.as_dict()))
    output.mkdir(parents=True, exist_ok=False)
    save(output / "system-config.json", original)
    candidates = []

    def solve(label, models=None, **restrictions):
        started = time.perf_counter()
        with wrapper.SearchSession(source_root, original, models=models) as session:
            c = session.config
            result = session.optimizer(algorithm).solve(K=c.route_grid, **restrictions)
            result["session_cache"] = session.stats()
            # JSON round trip explicitly returns ordinary dict/list objects;
            # preserve complete public problem fields for runtime acceptance.
            result["problem"] = json.loads(json.dumps(c.as_dict()))
            result["search_restrictions"] = {
                "fixed_deployment": restrictions.get("deployment"),
                "exclude_deployments": restrictions.get("exclude_deployments", []),
                "exclude_plans": restrictions.get("exclude_plans", []),
                "no_split": restrictions.get("no_split", False),
                "replica_options": restrictions.get("replica_options"),
            }
            if result.get("winner"):
                result["constraints"] = c.validate_plan(result["winner"]["deployment"], result["winner"]["routing"])
        result["external_wall_seconds"] = time.perf_counter() - started
        save(output / (label + ".json"), result)
        candidates.append({"label": label, "search": result})
        print(json.dumps(dict(role=label, status=result["status"],
            optimality_certified=result["optimality_certified"],
            U=result["winner"]["score"] if result.get("winner") else None,
            elapsed_seconds=result["external_wall_seconds"])), flush=True)
        return result

    best = solve("free_search_winner")
    winner = best.get("winner")
    if winner:
        solve("strongest_alternative_deployment", exclude_deployments=[winner["deployment"]])
        solve("strongest_alternative_route", deployment=winner["deployment"], exclude_plans=[winner])
        solve("best_no_split", no_split=True)
        for role, label in (("generalist", "best_generalist_only"), ("specialist", "best_specialist_only")):
            models = [m for m, spec in original["catalog"]["models"].items() if spec["role"] == role]
            solve(label, models=models)
    source_after = {str(p.relative_to(source_root)): digest(p) for p in sorted((source_root / "gsdeploy").rglob("*.py"))}
    bundle = {"schema": "gsdeploy.static_search_roles.v1",
        "config_sha256": hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":"),
                                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
        "candidates": candidates, "gpu_started": False,
        "scope": "Frozen calibrated predictor with immutable session digest memo; bounded searches retain gaps",
        "physical_global_optimality_certified": False,
        "engine": {"source_root": str(source_root), "source_before": source_before, "source_after": source_after,
            "source_unchanged": source_before == source_after, "wrapper_path": str(wrapper_path),
            "wrapper_sha256": wrapper_hash, "wrapper_unchanged": wrapper_hash == digest(wrapper_path)}}
    save(output / "search-results.json", bundle)
    save(output / "manifest.json", {"files": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
                                   "gpu_started": False})
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--wrapper", type=Path, help="Defaults to source-root/gsdeploy/search_session.py")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algorithm", choices=["milp", "dfs", "plain_dfs"], default="milp")
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--route-grid", type=int)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text())
    if isinstance(raw["catalog"], str):
        raw["catalog"] = json.loads(Path(raw["catalog"]).read_text())
    if args.time_limit is not None:
        raw["milp_time_limit_seconds"] = args.time_limit
    if args.route_grid is not None:
        raw["route_grid"] = args.route_grid
    bundle = solve_roles(raw, args.output, args.source_root, algorithm=args.algorithm, wrapper_path=args.wrapper)
    print(json.dumps(dict(roles=len(bundle["candidates"]), gpu_started=False)))


if __name__ == "__main__":
    main()
