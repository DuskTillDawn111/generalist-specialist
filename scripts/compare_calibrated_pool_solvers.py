#!/usr/bin/env python3
"""Fresh-process CPU comparison on a declared validated calibrated pool.

Always supply the frozen mathematical source root explicitly. No execution,
fitting, measurement or GPU APIs are invoked. Default domain is G2 plus four
specialists, with absent/one-replica states and the union of grids 1..K.
"""

import argparse
import copy
from dataclasses import replace
from fractions import Fraction
import hashlib
import itertools
import importlib.util
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

DEFAULT_MODELS = ["generalist_2", "text", "image", "ocr", "asr"]
ALGORITHMS = ["milp", "dfs", "plain_dfs"]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(root):
    return {str(p.relative_to(root)): fingerprint(p) for p in sorted((root / "gsdeploy").rglob("*.py"))}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(args):
    sys.path.insert(0, str(args.source_root))
    from gsdeploy.config import SystemConfig
    from gsdeploy.system import DeploymentSystem
    from gsdeploy.quality import quality_digest
    from gsdeploy.surrogate_bounds import surrogate_bounds

    config = SystemConfig.load(args.config)
    mode = config.catalog.get("performance_mode")
    if mode == "calibrated_replay_v2":
        from gsdeploy.calibration_v2 import validated_replay_spec
        validated_replay_spec(config.catalog, config.workload)
    elif mode == "calibrated_surrogate":
        from gsdeploy.calibration import validated_spec
        validated_spec(config.catalog, config.workload)
    else:
        raise ValueError("Comparison requires an explicitly validated v1/v2 calibrated predictor")
    original_hash = quality_digest(config.as_dict())
    missing = set(args.models) - set(config.catalog["models"])
    if missing:
        raise ValueError("Missing requested models: " + repr(sorted(missing)))
    catalog = copy.deepcopy(config.catalog)
    # Keep every original row for retained models. No count/profile truncation.
    for field in ("models", "capacity_factors", "memory_table_mib"):
        catalog[field] = {m: catalog[field][m] for m in args.models}
    for m in args.models:
        if "1" not in catalog["capacity_factors"][m]:
            raise ValueError("Unmeasured one-replica state: " + m)
    config = replace(config, catalog=catalog, route_grid=args.route_grid,
        milp_time_limit_seconds=args.time_limit, milp_capacity_witness=False,
        search_max_states=100000000, milp_max_iterations=1000000)
    bounds = surrogate_bounds(config.catalog, config.workload, **DeploymentSystem(config).context())
    if bounds is None:
        raise ValueError("Validated source cannot provide shared calibrated bounds")
    return config, original_hash, bounds


def audit_domain(config, models, grid, bounds):
    if config.route_pair_denominators_by_task:
        raise ValueError("PAIR_ROUTE_DOMAIN_UNSUPPORTED_BY_LEGACY_POOL_AUDIT: use a versioned domain audit")
    shares = {Fraction(u, d) for d in range(1, grid + 1) for u in range(d + 1)}
    deployments, identities, memory_pass, memory_identities = 0, 0, 0, 0
    per_deployment = []
    for bits in itertools.product((0, 1), repeat=len(models)):
        x = {m: 1 for m, bit in zip(models, bits) if bit}
        routes = 1
        for task in config.workload:
            paths = [m for m in x if task in config.catalog["models"][m]["supported_tasks"]]
            if len(paths) > 2:
                raise ValueError("This benchmark count audit supports at most two paths per task")
            if not paths:
                routes = 0
                break
            routes *= len(shares) if len(paths) == 2 else 1
        if not routes:
            continue
        memory = config.catalog["safe_factor"] * bounds.memory_multiplier * sum(
            config.catalog["memory_table_mib"][m]["1"] for m in x)
        passed = memory <= config.gpu_memory_mib + 1e-6
        deployments += 1
        identities += routes
        memory_pass += passed
        memory_identities += routes if passed else 0
        per_deployment.append(dict(deployment=x, route_identities=routes,
            memory_lower_mib=memory, memory_lower_budget_pass=passed))
    return dict(grid=grid, grid_contract="union of denominators 1..K; no capacity witnesses",
        binary_route_shares=[str(v) for v in sorted(shares)], covering_deployments=deployments,
        all_route_deployment_identities=identities, memory_lower_pass_deployments=memory_pass,
        memory_lower_pass_identity_upper_bound=memory_identities,
        per_deployment=per_deployment,
        scope="Counts before rho/drain/unknown evidence rejection; no complete feasibility inferred")


def child(args):
    config, original_hash, bounds = prepare(args)
    from gsdeploy.system import DeploymentSystem
    from gsdeploy.quality import quality_digest
    from gsdeploy.milp import JointMilpOptimizer
    from gsdeploy.optimizer import UtilityOptimizer
    from gsdeploy.trace import make_trace, trace_digest

    effective_hash = quality_digest(config.as_dict())
    audit = audit_domain(config, args.models, args.route_grid, bounds)
    trace = make_trace(config.workload, config.horizon_seconds, config.random_seed, config.arrival_process)
    envelope = dict(original_problem_sha256=original_hash, effective_problem_sha256=effective_hash,
        predictor_schema=bounds.spec["schema"], predictor_version=bounds.spec["version_sha256"],
        source_root=str(args.source_root), domain=audit, replica_options={m: [1] for m in args.models},
        trace_sha256=trace_digest(trace), offered_requests=len(trace), gpu_started=False,
        fairness=dict(time_limit_seconds=args.time_limit, search_max_states=config.search_max_states,
            milp_max_iterations=config.milp_max_iterations, milp_max_columns=config.milp_max_columns,
            absolute_gap=config.milp_absolute_gap,
            policy=config.search_policy, capacity_witness=False,
            task_replica_profiles_retained=True, baseline_count_rows_retained=True))
    if args.audit_only:
        print(json.dumps(envelope, allow_nan=False))
        return
    options = dict(search_max_states=config.search_max_states, search_policy=config.search_policy,
        milp_capacity_witness=False, **DeploymentSystem(config).context())
    if args.algorithm == "milp":
        cls = JointMilpOptimizer
        options.update(milp_time_limit_seconds=args.time_limit,
            milp_max_iterations=config.milp_max_iterations, milp_max_columns=config.milp_max_columns,
            milp_absolute_gap=config.milp_absolute_gap)
    else:
        cls = UtilityOptimizer
        options.update(dfs_time_limit_seconds=args.time_limit, dfs_bound_pruning=args.algorithm != "plain_dfs")
    started = time.perf_counter()
    if args.session:
        wrapper_path = args.wrapper or args.source_root / "gsdeploy" / "search_session.py"
        spec = importlib.util.spec_from_file_location("isolated_benchmark_session", wrapper_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with module.SearchSession(args.source_root, config.as_dict()) as session:
            solver = session.optimizer(args.algorithm)
            result = solver.solve(replica_options=envelope["replica_options"], K=args.route_grid)
            envelope["session_cache"] = session.stats()
        envelope["wrapper_path"] = str(wrapper_path)
        envelope["wrapper_sha256"] = fingerprint(wrapper_path)
    else:
        solver = cls(config.catalog, config.workload, config.gpu_memory_mib,
                     config.latency_reference_ms, config.weights, **options)
        result = solver.solve(replica_options=envelope["replica_options"], K=args.route_grid)
    elapsed = time.perf_counter() - started
    envelope.update(algorithm=args.algorithm, search_wall_seconds=elapsed, result=result,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cpu_affinity=sorted(os.sched_getaffinity(0)))
    print(json.dumps(envelope, allow_nan=False))


def command(args, algorithm=None, audit=False):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--source-root", str(args.source_root),
        "--config", str(args.config), "--route-grid", str(args.route_grid), "--time-limit", str(args.time_limit),
        "--models", *args.models]
    if audit:
        cmd.append("--audit-only")
    if algorithm:
        cmd.extend(["--algorithm", algorithm])
    if args.session:
        cmd.append("--session")
    if args.wrapper:
        cmd.extend(["--wrapper", str(args.wrapper)])
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--route-grid", type=int, default=4, choices=range(1, 17))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=120.)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Explicit model subset; default G2 plus all four specialists")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--session", action="store_true", help="Use identical isolated digest memo for all three algorithms")
    parser.add_argument("--wrapper", type=Path, help="Frozen session wrapper; default source-root/gsdeploy/search_session.py")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--algorithm", choices=ALGORITHMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source_root, args.config = args.source_root.resolve(), args.config.resolve()
    if args.wrapper:
        args.wrapper = args.wrapper.resolve()
    if args.repeats < 1 or args.time_limit <= 0 or len(set(args.models)) != len(args.models):
        parser.error("Positive repeats/time-limit and unique models are required")
    if not (args.source_root / "gsdeploy" / "milp.py").is_file():
        parser.error("--source-root must contain the explicit frozen gsdeploy source")
    if args.child:
        child(args)
        return
    before = sources(args.source_root)
    benchmark_source_before = fingerprint(__file__)
    wrapper_path = args.wrapper or args.source_root / "gsdeploy" / "search_session.py"
    wrapper_before = fingerprint(wrapper_path) if args.session else None
    config_hash = fingerprint(args.config)
    # Gate and count in the same explicit source process before any search.
    checked = subprocess.run(command(args, audit=True), text=True, capture_output=True)
    if checked.returncode:
        raise SystemExit("Validated-predictor gate rejected config:\n" + checked.stderr)
    audit = json.loads(checked.stdout)
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return
    if args.output is None:
        parser.error("--output is required for measured comparisons")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "preflight.json", audit)
    records = []
    for repeat in range(args.repeats):
        order = ALGORITHMS if repeat % 2 == 0 else list(reversed(ALGORITHMS))
        for algorithm in order:
            started = time.perf_counter()
            completed = subprocess.run(command(args, algorithm), text=True, capture_output=True)
            row = dict(repeat=repeat, algorithm=algorithm,
                       parent_process_wall_seconds=time.perf_counter() - started)
            if completed.returncode:
                row.update(process_failed=True, returncode=completed.returncode,
                           stderr=completed.stderr, stdout=completed.stdout)
            else:
                row.update(json.loads(completed.stdout))
            records.append(row)
            write(args.output / f"run-{repeat}-{algorithm}.json", row)
            print(json.dumps(dict(repeat=repeat, algorithm=algorithm,
                status=row.get("result", {}).get("status", "PROCESS_FAILED"),
                parent_wall=row["parent_process_wall_seconds"])), flush=True)
    consistency = []
    for repeat in range(args.repeats):
        rows = [r for r in records if r["repeat"] == repeat]
        results = [r.get("result", {}) for r in rows]
        scores = [r.get("winner", {}).get("score") if r.get("winner") else None for r in results]
        objective = (all(v is not None for v in scores)
                     and max(scores) - min(scores) <= audit["fairness"]["absolute_gap"])
        plans = [json.dumps({k: r["winner"][k] for k in ("deployment", "routing")}, sort_keys=True)
                 if r.get("winner") else None for r in results]
        consistency.append(dict(repeat=repeat, objective_agrees=objective,
            all_certified=all(r.get("optimality_certified", False) for r in results),
            certified_objective_agrees=objective and all(r.get("optimality_certified", False) for r in results),
            plan_identity_agrees=None not in plans and len(set(plans)) == 1))
    summary = []
    for algorithm in ALGORITHMS:
        rows = [r for r in records if r["algorithm"] == algorithm and "result" in r]
        summary.append(dict(algorithm=algorithm, completed_processes=len(rows),
            certificates=sum(r["result"]["optimality_certified"] for r in rows),
            median_search_wall_seconds=statistics.median(r["search_wall_seconds"] for r in rows) if rows else None,
            median_parent_wall_seconds=statistics.median(r["parent_process_wall_seconds"] for r in rows) if rows else None,
            median_full_predictions=statistics.median(r["result"]["metrics"]["full_prediction_calls"] for r in rows) if rows else None,
            max_peak_rss_kib=max((r["peak_rss_kib"] for r in rows), default=None)))
    after = sources(args.source_root)
    fair = all(r.get("effective_problem_sha256") == audit["effective_problem_sha256"]
               and r.get("trace_sha256") == audit["trace_sha256"] and r.get("fairness") == audit["fairness"]
               for r in records)
    report = dict(preflight=audit, records=records, summaries=summary, per_repeat=consistency,
        source_before=before, source_after=after, source_unchanged=before == after,
        benchmark_source_before=benchmark_source_before, benchmark_source_after=fingerprint(__file__),
        session_enabled=args.session, session_wrapper_sha256=wrapper_before,
        session_wrapper_unchanged=(wrapper_before == fingerprint(wrapper_path) if args.session else True),
        config_unchanged=config_hash == fingerprint(args.config), same_contract= fair,
        gpu_started=False, scope="Declared calibrated finite predictor domain only; no physical global optimum claim",
        timing_scope="Fresh serial processes; child wall includes solver construction and solve; parent wall also includes imports, config validation and result transport; source audit excluded",
        benchmark_complete=(all(r["certified_objective_agrees"] for r in consistency) and fair and before == after
                            and config_hash == fingerprint(args.config)
                            and benchmark_source_before == fingerprint(__file__)
                            and (not args.session or wrapper_before == fingerprint(wrapper_path))))
    write(args.output / "report.json", report)
    print(json.dumps(dict(summaries=summary, per_repeat=consistency,
                         benchmark_complete=report["benchmark_complete"]), indent=2))


if __name__ == "__main__":
    main()
