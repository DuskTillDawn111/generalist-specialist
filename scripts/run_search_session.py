#!/usr/bin/env python3
"""CPU search with isolated immutable digest memo and explicit frozen sources."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--wrapper", type=Path, help="Defaults to source-root/gsdeploy/search_session.py")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algorithm", choices=["milp", "dfs", "plain_dfs"], default="milp")
    parser.add_argument("--route-grid", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=120.)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--replicas", type=int, nargs="+", help="Explicit positive replica options for every retained model")
    args = parser.parse_args()
    if args.time_limit <= 0 or not 1 <= args.route_grid <= 16:
        parser.error("Positive time limit and grid1..16 required")
    if args.replicas and any(n <= 0 for n in args.replicas):
        parser.error("Replica options must be positive")
    root = args.source_root.resolve()
    before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((root / "gsdeploy").rglob("*.py"))}
    raw = json.loads(args.config.read_text())
    if isinstance(raw["catalog"], str):
        raw["catalog"] = json.loads(Path(raw["catalog"]).read_text())
    raw.update(route_grid=args.route_grid, milp_time_limit_seconds=args.time_limit,
        milp_capacity_witness=False, search_max_states=100000000, milp_max_iterations=1000000)
    module_path = (args.wrapper or root / "gsdeploy" / "search_session.py").resolve()
    spec = importlib.util.spec_from_file_location("isolated_search_session", module_path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    started = time.perf_counter()
    with wrapper.SearchSession(root, raw, models=args.models) as session:
        options = ({m: args.replicas for m in session.config.catalog["models"]}
                   if args.replicas else None)
        solver = session.optimizer(args.algorithm)
        result = solver.solve(replica_options=options, K=args.route_grid)
        elapsed, cache = time.perf_counter() - started, session.stats()
    after = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((root / "gsdeploy").rglob("*.py"))}
    report = dict(algorithm=args.algorithm, result=result, cache=cache,
        session_wall_seconds=elapsed, replica_options=options,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        mathematical_source_before=before, mathematical_source_after=after,
        mathematical_source_unchanged=before == after,
        wrapper_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(), gpu_started=False,
        scope="Unmodified frozen predictor and declared search grid; immutable session digest memo only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], optimality_certified=result["optimality_certified"],
        U=result["winner"]["score"] if result.get("winner") else None,
        session_wall_seconds=elapsed, cache=cache)))


if __name__ == "__main__":
    main()
