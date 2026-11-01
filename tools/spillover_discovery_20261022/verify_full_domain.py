#!/usr/bin/env python3
"""CPU-only MILP/bounded-DFS agreement on the full seven-model replica domain.

Default: K8, every measured positive count plus absence, fresh serial processes
and equal per-solver budgets. No structural or routing restrictions are added.
--reference-result accepts only solve_roles.py's completed, unrestricted
free.raw.json with its sibling provenance bundle; then only DFS is run.
No timing speedup is claimed. Budget exhaustion remains UNVERIFIED.

CPU fixture checks use the repository's explicitly synthetic replay-transfer
fixture, temporarily relax only the seven-model count assertion to two, and
mock optimizer.solve (never launch a real search). They check full count
options, reference source/config/trace provenance, zero-gap agreement,
restricted-reference rejection, budget UNVERIFIED and raw-result retention.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from target_specialist import reject_extended_route_domain
from target_specialist import TARGET_SPECIALISTS, validate_target, recorded_target

SCHEMA = "gsdeploy.full_domain_solver_agreement.v1"
MODEL_COUNT = 7


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sources(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted((root / "gsdeploy").rglob("*.py"))}


def wrapper(root):
    spec = importlib.util.spec_from_file_location("_full_domain_private_session", root / "gsdeploy/search_session.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_receipt(row):
    if row != receipt(row["path"]):
        raise ValueError("Changed reference receipt: " + row["path"])
    return read(row["path"])


def prepare(args):
    module = wrapper(args.source_root)
    raw = read(args.config)
    reject_extended_route_domain(raw)
    if isinstance(raw["catalog"], str):
        path = Path(raw["catalog"])
        raw["catalog"] = read(path if path.is_absolute() else args.config.parent / path)
    if len(raw["catalog"]["models"]) != MODEL_COUNT:
        raise ValueError("Full-domain verification requires exactly seven models")
    if raw["catalog"].get("performance_mode") != "replay_transfer_v1":
        raise ValueError("Full-domain verification requires validated replay_transfer_v1")
    # Validate original before any operational budget/grid changes.
    with module.SearchSession(args.source_root, raw) as session:
        session.module("replay_transfer").check_scoring(session.config)
        original = json.loads(json.dumps(session.config.as_dict()))
        roles = session.config.catalog["replay_transfer_v1"]["seed_roles"]
        if session.config.random_seed not in roles["development"]:
            raise ValueError("Only predeclared development seeds may be searched")
    target=getattr(args,'target_specialist','asr')
    validate_target(original['catalog'],target)
    if args.seed is not None and original["random_seed"] != args.seed:
        raise ValueError("--seed must match the supplied config; no silent seed changes")
    if args.horizon is not None and original["horizon_seconds"] != args.horizon:
        raise ValueError("--horizon must match supplied config; no silent horizon changes")
    raw = dict(original, route_grid=args.route_grid, milp_time_limit_seconds=args.time_limit,
               milp_capacity_witness=False)
    # Declare every positive measured count explicitly; do not reuse restricted
    # replica domains or require any winner structure.
    options = {m: sorted(int(n) for n in raw["catalog"]["capacity_factors"][m] if int(n) > 0)
               for m in sorted(raw["catalog"]["models"])}
    supplied = args.replica_options
    if supplied:
        requested = read(supplied[1:]) if supplied.startswith("@") else json.loads(supplied)
        if requested != options:
            raise ValueError("Full-domain options must equal all measured positive counts for every model")
    if any(not values for values in options.values()):
        raise ValueError("A model has no measured positive replica count")
    with module.SearchSession(args.source_root, raw) as session:
        session.module("replay_transfer").check_scoring(session.config)
        config = json.loads(json.dumps(session.config.as_dict()))
        trace_module = session.module("trace")
        trace = trace_module.make_trace(session.config.workload, session.config.horizon_seconds,
                                       session.config.random_seed, session.config.arrival_process)
        trace_hash = trace_module.trace_digest(trace)
        transfer = session.config.catalog["replay_transfer_v1"]
        envelope = dict(schema=SCHEMA, original_config=original, effective_config=config,
            config_sha256=canonical(config), input_receipt=receipt(args.config),
            source_root=str(args.source_root), source_sha256=sources(args.source_root),
            tool_receipt=receipt(__file__), wrapper_receipt=receipt(args.source_root / "gsdeploy/search_session.py"),
            domain=dict(models=sorted(options), replica_options=options, absence_allowed=True,
                deployment_count_vectors=math.prod(len(values) + 1 for values in options.values()),
                K=args.route_grid, routing="union of denominators 1..K", capacity_witness=False,
                fixed_deployment=None, no_split=False, exclude_deployments=[], exclude_plans=[]),
            seed=config["random_seed"], horizon_seconds=config["horizon_seconds"], trace_sha256=trace_hash,
            predictor_version=transfer["version_sha256"], gpu_started=False,
            target_specialist=target,target_task=target,
            fairness=dict(solver_time_limit_seconds=args.time_limit, search_max_states=config["search_max_states"],
                milp_max_iterations=config["milp_max_iterations"], milp_max_columns=config["milp_max_columns"],
                absolute_gap=config["milp_absolute_gap"], isolated_fresh_process_per_solver=True))
        reference = reference_result(args, envelope, session) if args.reference_result else None
    return module, config, options, envelope, reference


def certified(result):
    winner = result.get("winner")
    return bool(result.get("optimality_certified") is True and result.get("search_completed") is True
        and type(result.get("optimality_gap")) in (int, float) and result["optimality_gap"] == 0
        and winner and type(winner.get("score")) in (int, float) and math.isfinite(winner["score"])
        and winner.get("prediction", {}).get("eligible") is True)


def reference_result(args, envelope, session):
    path = args.reference_result
    if path.name != "free.raw.json":
        raise ValueError("Reference must be solve_roles.py's unrestricted free.raw.json")
    bundle_path = path.parent / "search-results.json"
    bundle = read(bundle_path)
    recorded_target(bundle,getattr(args,'target_specialist','asr'))
    engine = bundle.get("engine", {})
    if (bundle.get("schema") != "gsdeploy.static_search_roles.v1"
            or engine.get("source_root") != str(args.source_root)
            or engine.get("source_before") != envelope["source_sha256"]
            or engine.get("source_after") != envelope["source_sha256"]
            or engine.get("source_unchanged") is not True
            or engine.get("wrapper_sha256") != envelope["wrapper_receipt"]["sha256"]):
        raise ValueError("Reference source/wrapper provenance is incomplete or differs")
    raw = read(path)
    item = next((row for row in bundle.get("candidates", []) if row.get("label") == "free"), None)
    if item is None or item.get("raw_result_receipt") != receipt(path):
        raise ValueError("Reference raw result is not bound as the free search")
    enriched = item["search"]
    if enriched.get("search_restrictions") != {} or any(enriched.get(k) != v for k, v in raw.items()):
        raise ValueError("Reference result is restricted or differs from recorded search")
    config = checked_receipt(bundle["config_receipt"])
    if (enriched.get("config_receipt") != bundle["config_receipt"]
            or canonical(config) != bundle.get("config_sha256")
            or bundle.get("K") != envelope["domain"]["K"]
            or bundle.get("seed") != envelope["seed"]
            or bundle.get("horizon_seconds") != envelope["horizon_seconds"]):
        raise ValueError("Reference config/K/seed/horizon provenance differs")
    # Time limits are operational; all other config fields must match exactly.
    strip_budget = lambda c: {k: v for k, v in c.items() if k != "milp_time_limit_seconds"}
    if strip_budget(config) != strip_budget(envelope["effective_config"]):
        raise ValueError("Reference mathematical config/domain differs")
    domain = raw.get("domain", {})
    if (raw.get("solver") != "milp" or not certified(raw)
            or raw.get("trace_sha256") != envelope["trace_sha256"]
            or domain.get("K") != envelope["domain"]["K"]
            or domain.get("capacity_boundary_witnesses") is not False):
        raise ValueError("Reference must have a completed zero-gap MILP certificate on this exact grid/trace")
    winner = raw["winner"]
    prediction = session.predict(winner["deployment"], winner["routing"])
    if prediction.get("eligible") is not True or abs(prediction["U"] - winner["score"]) > session.config.milp_absolute_gap:
        raise ValueError("Reference winner is not eligible with the same utility here")
    return dict(algorithm="milp", result=raw, reference_only=True,
        receipts=[receipt(path), receipt(bundle_path), bundle["config_receipt"]],
        reference_solver_budget_seconds=config["milp_time_limit_seconds"],
        scope="Reference certificate only; timings are not a fresh-process efficiency comparison")


def child(args):
    module, config, options, envelope, reference = prepare(args)
    if args.audit_only:
        envelope["reference"] = reference
        print(json.dumps(envelope, allow_nan=False))
        return
    wall, cpu = time.perf_counter(), time.process_time()
    with module.SearchSession(args.source_root, config) as session:
        result = session.optimizer(args.algorithm).solve(replica_options=options, K=args.route_grid)
        stats = session.stats()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    envelope.update(algorithm=args.algorithm, result=result, session_cache=stats,
        search_wall_seconds=time.perf_counter() - wall, search_cpu_seconds=time.process_time() - cpu,
        process_user_cpu_seconds=usage.ru_utime, process_system_cpu_seconds=usage.ru_stime,
        peak_rss_kib=usage.ru_maxrss, cpu_affinity=sorted(os.sched_getaffinity(0)))
    print(json.dumps(envelope, allow_nan=False))


def command(args, algorithm=None):
    result = [sys.executable, str(Path(__file__).resolve()), "--child", "--source-root", str(args.source_root),
        "--config", str(args.config), "--route-grid", str(args.route_grid), "--time-limit", str(args.time_limit)]
    for name in ("seed", "horizon", "replica_options", "reference_result"):
        value = getattr(args, name)
        if value is not None:
            result += ["--" + name.replace("_", "-"), str(value)]
    result += ['--target-specialist',getattr(args,'target_specialist','asr')]
    return result + (["--algorithm", algorithm] if algorithm else ["--audit-only"])


def compare(records, audit, unchanged):
    results = [row.get("result", {}) for row in records]
    complete = len(records) == 2 and {row["algorithm"] for row in records} == {"milp", "dfs"}
    all_certified = complete and all(certified(result) for result in results)
    values = [result["winner"]["score"] for result in results] if all_certified else []
    agrees = all_certified and max(values) - min(values) <= audit["fairness"]["absolute_gap"]
    fields = ("target_specialist", "config_sha256", "domain", "seed", "horizon_seconds", "trace_sha256", "source_sha256",
              "predictor_version", "wrapper_receipt", "tool_receipt", "fairness", "input_receipt")
    same = complete and all(row.get("reference_only") or all(row.get(k) == audit[k] for k in fields) for row in records)
    verified = agrees and same and unchanged
    return dict(status="SAME_FULL_DOMAIN_OPTIMUM_VERIFIED" if verified else "UNVERIFIED",
        same_full_domain_optimum_verified=verified, both_completed_zero_gap_certified=all_certified,
        certified_utilities_agree=agrees, same_contract=same, source_inputs_unchanged=unchanged,
        reason=None if verified else "Missing certificate, budget exhaustion, different U, changed provenance or failed process; restricted certificates are never substituted",
        gpu_started=False, physical_global_optimality_certified=False,
        speedup_claimed=False, scope="Full seven-model measured replica domain and declared finite grid only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument('--target-specialist',choices=TARGET_SPECIALISTS,default='asr')
    parser.add_argument("--output", type=Path)
    parser.add_argument("--route-grid", "--K", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--time-limit", type=float, default=180.)
    parser.add_argument("--seed", type=int, help="Assert config seed, rather than overriding it")
    parser.add_argument("--horizon", type=float, help="Assert config horizon, rather than overriding it")
    parser.add_argument("--replica-options", help="Optional JSON/@file assertion of full positive measured counts")
    parser.add_argument("--reference-result", type=Path, help="Verified provenance of formal free.raw.json; run only DFS")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--algorithm", choices=("milp", "dfs"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("source_root", "config", "output", "reference_result"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("Positive finite per-solver budget required")
    if not (args.source_root / "gsdeploy/search_session.py").is_file():
        parser.error("Explicit frozen SearchSession source required")
    if args.child:
        child(args)
        return
    before, tool, input_sha = sources(args.source_root), sha(__file__), sha(args.config)
    preflight = subprocess.run(command(args), capture_output=True, text=True)
    if preflight.returncode:
        raise SystemExit("Full-domain preflight rejected:\n" + preflight.stderr)
    audit = json.loads(preflight.stdout)
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return
    if args.output is None:
        parser.error("--output required for searches")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "original-input.json").write_bytes(args.config.read_bytes())
    write(args.output / "effective-config.json", audit["effective_config"])
    write(args.output / "preflight.json", audit)
    write(args.output / "invocation.json", dict(argv=sys.argv, python_version=sys.version,
        executable=sys.executable, source_root=str(args.source_root), gpu_started=False,
        algorithms=["dfs"] if audit["reference"] else ["milp", "dfs"], serial=True,
        fresh_processes=True, watchdog_seconds=args.time_limit + 180,
        reference_timing_not_comparable=bool(audit["reference"])))
    records = []
    if audit["reference"]:
        records.append(audit["reference"])
        write(args.output / "reference-milp.json", audit["reference"])
    for algorithm in (("dfs",) if audit["reference"] else ("milp", "dfs")):
        started = time.perf_counter()
        row = dict(algorithm=algorithm)
        try:
            process = subprocess.run(command(args, algorithm), capture_output=True, text=True, timeout=args.time_limit + 180)
            (args.output / (algorithm + ".stdout")).write_text(process.stdout)
            (args.output / (algorithm + ".stderr")).write_text(process.stderr)
            if process.returncode:
                row.update(process_failed=True, returncode=process.returncode, error=process.stderr)
            else:
                row.update(json.loads(process.stdout))
        except (subprocess.TimeoutExpired, ValueError) as exc:
            row.update(process_failed=True, error=str(exc))
        row["parent_wall_seconds"] = time.perf_counter() - started
        records.append(row)
        write(args.output / (algorithm + ".json"), row)
        print(json.dumps(dict(algorithm=algorithm, status=row.get("result", {}).get("status", "PROCESS_FAILED"))), flush=True)
    unchanged = before == sources(args.source_root) and tool == sha(__file__) and input_sha == sha(args.config)
    if audit["reference"]:
        unchanged = unchanged and all(row == receipt(row["path"]) for row in audit["reference"]["receipts"])
    report = dict(schema=SCHEMA, preflight=audit, records=records, **compare(records, audit, unchanged))
    write(args.output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "same_full_domain_optimum_verified", "gpu_started")}))


if __name__ == "__main__":
    main()
