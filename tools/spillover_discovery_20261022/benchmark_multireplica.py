#!/usr/bin/env python3
"""Serial fresh-process comparison on a declared, validated transfer subdomain.

No GPU, fitting, or native measurement APIs are used. Absence (zero replicas)
is always allowed; --replica-options declares the positive measured counts.
The default five-model domain is separate from any seven-model certificate.
"""
import argparse
import copy
from dataclasses import replace
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from target_specialist import reject_extended_route_domain
from target_specialist import TARGET_SPECIALISTS, validate_target

ALGORITHMS = ("milp", "dfs", "plain_dfs")
MODELS = ("generalist_2", "text", "image", "ocr", "asr")
SCHEMA = "gsdeploy.spillover_multireplica_benchmark.v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sources(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted((root / "gsdeploy").rglob("*.py"))}


def parse_options(value, models, target_specialist='asr'):
    if value is None:
        result = {m: ([1, 2] if m == target_specialist else [1]) for m in models}
    else:
        result = json.loads(Path(value[1:]).read_text() if value.startswith("@") else value)
    if not isinstance(result, dict) or set(result) != set(models):
        raise ValueError("replica-options must explicitly cover exactly the declared models")
    for model, counts in result.items():
        if (not isinstance(counts, list) or not counts or
                any(type(n) is not int or n <= 0 for n in counts) or len(counts) != len(set(counts))):
            raise ValueError("Positive unique integer counts required for " + model)
    return {m: sorted(result[m]) for m in models}


def prepare(args):
    sys.path.insert(0, str(args.source_root))
    from gsdeploy.config import SystemConfig
    from gsdeploy.quality import quality_digest
    from gsdeploy.replay_transfer import validated_transfer_spec, check_scoring
    from gsdeploy.system import DeploymentSystem
    from gsdeploy.surrogate_bounds import surrogate_bounds
    original = SystemConfig.load(args.config)
    reject_extended_route_domain(original.as_dict())
    if original.catalog.get("performance_mode") != "replay_transfer_v1":
        raise ValueError("Requires replay_transfer_v1; old predictors are not accepted")
    spec = validated_transfer_spec(original.catalog, original.workload)
    check_scoring(original)
    target=getattr(args,'target_specialist','asr')
    validate_target(original.catalog,target)
    if target not in args.models:raise ValueError('Target specialist must remain in declared benchmark models')
    if original.random_seed not in spec["seed_roles"]["development"]:
        raise ValueError("Efficiency search requires a declared development seed")
    missing = set(args.models) - set(original.catalog["models"])
    if missing:
        raise ValueError("Requested models missing: " + repr(sorted(missing)))
    catalog = copy.deepcopy(original.catalog)
    # Keep all profiles and count rows for retained models; options restrict the
    # search, never truncate the predictor's measurement evidence.
    for field in ("models", "capacity_factors", "memory_table_mib"):
        catalog[field] = {m: catalog[field][m] for m in args.models}
    for model, counts in args.replica_options.items():
        for count in counts:
            if (str(count) not in catalog["capacity_factors"][model] or
                    str(count) not in catalog["memory_table_mib"][model]):
                raise ValueError("Unmeasured replica count: %s n%d" % (model, count))
    effective = replace(original, catalog=catalog, route_grid=args.route_grid,
        milp_time_limit_seconds=args.time_limit, milp_capacity_witness=False,
        search_max_states=100000000, milp_max_iterations=1000000)
    validated_transfer_spec(effective.catalog, effective.workload)
    check_scoring(effective)
    bounds = surrogate_bounds(effective.catalog, effective.workload,
                              **DeploymentSystem(effective).context())
    if bounds is None:
        raise ValueError("Validated predictor did not supply shared bounds")
    return original, effective, bounds, quality_digest


def domain_audit(config, options, grid, bounds):
    shares = sorted({Fraction(u, d) for d in range(1, grid + 1) for u in range(d + 1)})
    rows = []
    models = list(options)
    for vector in itertools.product(*([0] + options[m] for m in models)):
        deployment = {m: n for m, n in zip(models, vector) if n}
        routes = 1
        for task in config.workload:
            paths = [m for m in deployment if task in config.catalog["models"][m]["supported_tasks"]]
            if len(paths) > 2:
                raise ValueError("Count audit supports at most two model paths per task")
            if not paths:
                routes = 0
                break
            routes *= len(shares) if len(paths) == 2 else 1
        if not routes:
            continue
        memory = config.catalog["safe_factor"] * bounds.memory_multiplier * math.fsum(
            config.catalog["memory_table_mib"][m][str(n)] for m, n in deployment.items())
        rows.append(dict(deployment=deployment, route_identities=routes,
            memory_lower_mib=memory, memory_lower_budget_pass=memory <= config.gpu_memory_mib + 1e-6))
    return dict(models=models, replica_options=options, absence_allowed=True, route_grid=grid,
        route_contract="union of denominators 1..K; capacity witnesses disabled",
        binary_route_shares=[str(q) for q in shares], covering_deployments=len(rows),
        route_deployment_identities=sum(r["route_identities"] for r in rows),
        memory_lower_pass_deployments=sum(r["memory_lower_budget_pass"] for r in rows),
        memory_lower_pass_identity_upper_bound=sum(r["route_identities"] for r in rows if r["memory_lower_budget_pass"]),
        per_deployment=rows, winner_structure_restricted=False,
        scope="Declared subdomain only; counts precede rho/drain/evidence gates; no large-domain certificate transfer")


def child(args):
    original, config, bounds, digest = prepare(args)
    from gsdeploy.trace import make_trace, trace_digest
    domain = domain_audit(config, args.replica_options, args.route_grid, bounds)
    wrapper = args.source_root / "gsdeploy" / "search_session.py"
    envelope = dict(schema=SCHEMA, original_config=original.as_dict(), effective_config=config.as_dict(),
        original_problem_sha256=digest(original.as_dict()), effective_problem_sha256=digest(config.as_dict()),
        input_file_sha256=sha(args.config), source_root=str(args.source_root), source_sha256=sources(args.source_root),
        benchmark_sha256=sha(__file__), session_wrapper_sha256=sha(wrapper), domain=domain,
        predictor_version=bounds.spec["version_sha256"], predictor_schema=bounds.spec["schema"],
        trace_sha256=trace_digest(make_trace(config.workload, config.horizon_seconds,
                                           config.random_seed, config.arrival_process)),
        fairness=dict(time_limit_seconds=args.time_limit, search_max_states=config.search_max_states,
            milp_max_iterations=config.milp_max_iterations, milp_max_columns=config.milp_max_columns,
            absolute_gap=config.milp_absolute_gap, policy=config.search_policy,
            capacity_witness=False, isolated_session_per_algorithm=True), gpu_started=False)
    envelope.update(target_specialist=args.target_specialist,target_task=args.target_specialist)
    envelope['domain'].update(target_specialist=args.target_specialist)
    if args.audit_only:
        print(json.dumps(envelope, allow_nan=False))
        return
    spec = importlib.util.spec_from_file_location("_spillover_benchmark_session", wrapper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wall, cpu = time.perf_counter(), time.process_time()
    with module.SearchSession(args.source_root, config.as_dict()) as session:
        result = session.optimizer(args.algorithm).solve(replica_options=args.replica_options, K=args.route_grid)
        cache = session.stats()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    envelope.update(algorithm=args.algorithm, result=result, session_cache=cache,
        search_wall_seconds=time.perf_counter() - wall, search_cpu_seconds=time.process_time() - cpu,
        process_user_cpu_seconds=usage.ru_utime, process_system_cpu_seconds=usage.ru_stime,
        peak_rss_kib=usage.ru_maxrss, cpu_affinity=sorted(os.sched_getaffinity(0)))
    print(json.dumps(envelope, allow_nan=False))


def command(args, algorithm=None):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--source-root", str(args.source_root),
        "--config", str(args.config), "--route-grid", str(args.route_grid), "--time-limit", str(args.time_limit),
        "--replica-options", json.dumps(args.replica_options), "--models", *args.models]
    cmd += ['--target-specialist',args.target_specialist]
    cmd += ["--algorithm", algorithm] if algorithm else ["--audit-only"]
    return cmd


def agreement(records, repeats, tolerance):
    rows = []
    for repeat in range(repeats):
        runs = [r for r in records if r["repeat"] == repeat]
        results = [r.get("result", {}) for r in runs]
        values = [r.get("winner", {}).get("score") if r.get("winner") else None for r in results]
        finite = len(values) == len(ALGORITHMS) and all(type(v) in (int, float) and math.isfinite(v) for v in values)
        certified = len(results) == len(ALGORITHMS) and all(r.get("optimality_certified") is True for r in results)
        same = finite and max(values) - min(values) <= tolerance
        plans = [json.dumps({k: r["winner"][k] for k in ("deployment", "routing")}, sort_keys=True)
                 if r.get("winner") else None for r in results]
        rows.append(dict(repeat=repeat, all_certified=certified, objective_agrees=same,
                         certified_objective_agrees=certified and same,
                         plan_identity_agrees=None not in plans and len(set(plans)) == 1))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument('--target-specialist',choices=TARGET_SPECIALISTS,default='asr')
    parser.add_argument("--replica-options", help='JSON dict or @JSON-file; e.g. {"asr":[1,2],...}')
    parser.add_argument("--route-grid", type=int, choices=range(1, 17), default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=180.)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--algorithm", choices=ALGORITHMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source_root, args.config = args.source_root.resolve(), args.config.resolve()
    if not all((args.source_root / "gsdeploy" / name).is_file() for name in
               ("config.py", "replay_transfer.py", "search_session.py", "optimizer.py", "milp.py")):
        parser.error("--source-root must contain the explicit transfer/session/solver source")
    if args.repeats < 1 or not math.isfinite(args.time_limit) or args.time_limit <= 0 or len(set(args.models)) != len(args.models):
        parser.error("Positive repeats/time-limit and unique model names required")
    try:
        args.replica_options = parse_options(args.replica_options, args.models,args.target_specialist)
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    if args.child:
        child(args)
        return
    if not args.audit_only and args.output is None:
        parser.error("--output required for measured comparisons")
    before, benchmark_before, input_bytes = sources(args.source_root), sha(__file__), args.config.read_bytes()
    checked = subprocess.run(command(args), text=True, capture_output=True)
    if checked.returncode:
        raise SystemExit("Validated-transfer preflight rejected config:\n" + checked.stderr)
    audit = json.loads(checked.stdout)
    if args.audit_only:
        print(json.dumps(audit, indent=2, allow_nan=False))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "original-input.json").write_bytes(input_bytes)
    write(args.output / "effective-config.json", audit["effective_config"])
    write(args.output / "domain.json", audit["domain"])
    write(args.output / "preflight.json", audit)
    write(args.output / "source-sha256.json", before)
    write(args.output / "invocation.json", dict(argv=sys.argv, executable=sys.executable,
        python_version=sys.version, repeats=args.repeats,
        fresh_processes=True, serial=True, algorithm_order="forward/reverse alternation",
        process_watchdog_seconds=args.time_limit + 120, gpu_started=False))
    records = []
    for repeat in range(args.repeats):
        order = ALGORITHMS if repeat % 2 == 0 else tuple(reversed(ALGORITHMS))
        for algorithm in order:
            start = time.perf_counter()
            row = dict(repeat=repeat, algorithm=algorithm)
            try:
                completed = subprocess.run(command(args, algorithm), text=True, capture_output=True,
                                           timeout=args.time_limit + 120)
                (args.output / f"run-{repeat}-{algorithm}.stdout").write_text(completed.stdout)
                (args.output / f"run-{repeat}-{algorithm}.stderr").write_text(completed.stderr)
                if completed.returncode:
                    row.update(process_failed=True, returncode=completed.returncode, stderr=completed.stderr)
                else:
                    try:
                        row.update(json.loads(completed.stdout))
                    except ValueError:
                        row.update(process_failed=True, failure="INVALID_CHILD_JSON")
            except subprocess.TimeoutExpired:
                row.update(process_failed=True, failure="PROCESS_WATCHDOG_TIMEOUT",
                           watchdog_seconds=args.time_limit + 120)
            row["parent_process_wall_seconds"] = time.perf_counter() - start
            records.append(row)
            write(args.output / f"run-{repeat}-{algorithm}.json", row)
            print(json.dumps(dict(repeat=repeat, algorithm=algorithm,
                status=row.get("result", {}).get("status", "PROCESS_FAILED"))), flush=True)
    keys = ("target_specialist", "effective_problem_sha256", "original_problem_sha256", "trace_sha256", "domain", "fairness",
            "source_sha256", "benchmark_sha256", "session_wrapper_sha256", "input_file_sha256")
    same_contract = all(all(r.get(k) == audit[k] for k in keys) for r in records)
    per_repeat = agreement(records, args.repeats, audit["fairness"]["absolute_gap"])
    all_scores = [r["result"]["winner"]["score"] for r in records if r.get("result", {}).get("winner")]
    across_repeats = len(all_scores) == len(records) and max(all_scores) - min(all_scores) <= audit["fairness"]["absolute_gap"]
    unchanged = before == sources(args.source_root) and benchmark_before == sha(__file__) and input_bytes == args.config.read_bytes()
    complete = same_contract and unchanged and across_repeats and all(r["certified_objective_agrees"] for r in per_repeat)
    summaries = []
    for algorithm in ALGORITHMS:
        rows = [r for r in records if r["algorithm"] == algorithm and "result" in r]
        summaries.append(dict(algorithm=algorithm, completed_processes=len(rows),
            certificates=sum(r["result"].get("optimality_certified") is True for r in rows),
            **{f"median_{field}": statistics.median(r[field] for r in rows) if rows else None for field in
               ("search_wall_seconds", "search_cpu_seconds", "parent_process_wall_seconds", "peak_rss_kib")},
            median_full_predictions=statistics.median(r["result"]["metrics"]["full_prediction_calls"] for r in rows) if rows else None))
    ratios = None
    if complete:
        indexed = {r["algorithm"]: r for r in summaries}
        ratios = {f"{baseline}_over_milp": {field: indexed[baseline]["median_" + field] / indexed["milp"]["median_" + field]
                  for field in ("search_wall_seconds", "search_cpu_seconds", "parent_process_wall_seconds")}
                  for baseline in ("dfs", "plain_dfs")}
    report = dict(schema=SCHEMA, preflight=audit, records=records, summaries=summaries,
        per_repeat=per_repeat, objectives_agree_across_repeats=across_repeats, same_contract=same_contract,
        source_and_input_unchanged=unchanged, benchmark_complete=complete, speed_ratios=ratios,
        gpu_started=False, timing_scope="Fresh serial processes; search includes private session initialization and solve; parent includes imports/preflight/output; source scan excluded from search",
        scope="Declared finite predictor subdomain only; separate from seven-model proof and physical optimality")
    write(args.output / "report.json", report)
    print(json.dumps(dict(benchmark_complete=complete, summaries=summaries, speed_ratios=ratios), indent=2))


if __name__ == "__main__":
    main()
