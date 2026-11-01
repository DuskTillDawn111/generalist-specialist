#!/usr/bin/env python3
"""Materialize certified role results into a spillover confirmation contract.

CPU validation only: never solves, activates a transfer, or starts native jobs.
Rejected inputs produce an audit report and no consumable contract/candidates.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import traceback
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from target_specialist import reject_extended_route_domain
from target_specialist import (TARGET_SPECIALISTS, validate_target, recorded_target,
    one_specialist_restriction, target_structure, assert_one_specialist_plan, route_grid_membership)

ROLE_MAP = {
    "free": "free_search_winner",
    "best_global_no_split": "best_global_no_split",
    "best_same_deployment_no_split": "best_same_deployment_no_split",
    "strongest_alternative_deployment": "strongest_alternative_deployment",
    "best_one_specialist_max": "best_one_specialist_max",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def read_receipt(row):
    path = Path(row["path"])
    if not path.is_absolute() or sha(path) != row["sha256"]:
        raise ValueError("Source receipt changed: " + str(path))
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def wrapper(root):
    path = Path(root) / "gsdeploy/search_session.py"
    spec = importlib.util.spec_from_file_location("spillover_materialize_session", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def warmup_value(raw):
    if isinstance(raw.get("warmup"), dict) and "seconds" in raw["warmup"]:
        return raw["warmup"]["seconds"]
    if "warmup_seconds" in raw:
        return raw["warmup_seconds"]
    if isinstance(raw.get("contract"), dict) and "warmup_seconds" in raw["contract"]:
        return raw["contract"]["warmup_seconds"]
    values = [j["identity"]["warmup_seconds"] for j in raw.get("jobs", [])]
    if values and len(set(values)) == 1:
        return values[0]
    raise ValueError("Warmup source lacks a single explicit warmup duration")


def inspect_restrictions(role, restrictions, winner, target_specialist='asr'):
    allowed = {
        "free": {},
        "best_global_no_split": {"no_split": True},
        "best_same_deployment_no_split": {"deployment": winner["deployment"], "no_split": True},
        "strongest_alternative_deployment": {"exclude_deployments": [winner["deployment"]]},
        "best_one_specialist_max": one_specialist_restriction(target_specialist),
    }[role]
    if restrictions != allowed:
        raise ValueError("Unexpected search restrictions for " + role)


def certify(row, role, tolerance):
    """Check recorded certificate; full eligible plan math is rechecked later."""
    result = row["search"]
    raw_receipt = row.get("raw_result_receipt", result.get("raw_result_receipt"))
    if not raw_receipt:
        raise ValueError("Missing raw solver receipt for " + role)
    raw = read_receipt(raw_receipt)
    for key, value in raw.items():
        if result.get(key) != value:
            raise ValueError("Raw/enriched result mismatch for " + role + ": " + key)
    if result.get("error") or result.get("metrics", {}).get("unknown_candidates", 0):
        raise ValueError("Failed or unknown-evidence search role: " + role)
    candidate = result.get("winner")
    complete = result.get("search_completed") is True
    if candidate is not None:
        gap = result.get("optimality_gap")
        if (result.get("status") != "OPTIMAL_MILP_FINITE_DOMAIN"
                or result.get("optimality_certified") is not True or not complete
                or result.get("eligible_incumbent") is not True
                or type(gap) not in (int, float) or not math.isfinite(gap)
                or gap < 0 or gap > tolerance
                or not result.get("constraints", {}).get("constraints_valid")
                or not candidate.get("prediction", {}).get("eligible")
                or not candidate["prediction"].get("utility_complete")):
            raise ValueError("Uncertified/ineligible incumbent (budget results cannot be materialized): " + role)
        return dict(kind="CERTIFIED_FINITE_DOMAIN_OPTIMUM", raw_result_receipt=raw_receipt,
                    raw_result=raw, result=result, plan={k: copy.deepcopy(candidate[k]) for k in ("deployment", "routing")})
    # The solver reports optimality_certified=False for infeasible domains.
    # Its completed exhaustive finite-domain outcome, no incumbent and no
    # unknown candidates are the certificate; early column/budget exits fail.
    if (not complete or result.get("status") not in {"NO_FEASIBLE_GRID_PLAN", "INFEASIBLE_FINITE_DOMAIN"}
            or result.get("diagnostic_candidate") is not None
            or result.get("metrics", {}).get("unknown_candidates") != 0
            or not isinstance(result.get("domain"), dict)):
        raise ValueError("Absent candidate lacks completed finite-domain infeasibility certificate: " + role)
    return dict(kind="CERTIFIED_INFEASIBLE_FINITE_DOMAIN", raw_result_receipt=raw_receipt,
                raw_result=raw, result=result, plan=None)


def materialize(search_path, config_path, source_root, output, *, study_id,
                warmup_source=None, purpose="spillover_independent_confirmation_not_training", target_specialist='asr'):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(schema="gsdeploy.spillover_materialization_audit.v1", status="CHECKING",
        ready=False, gpu_started=False, search_performed=False, role_checks=[],
        physical_global_optimality_certified=False,
        sources=dict(search_results=receipt(search_path), config=receipt(config_path)),
        source_root=str(Path(source_root).resolve()))
    report.update(target_specialist=target_specialist,target_task=target_specialist)
    try:
        search = read_receipt(report["sources"]["search_results"])
        raw_config = read_receipt(report["sources"]["config"])
        reject_extended_route_domain(raw_config)
        if search.get("schema") != "gsdeploy.static_search_roles.v1":
            raise ValueError("Unexpected role search bundle schema")
        recorded_target(search,target_specialist)
        validate_target(raw_config['catalog'],target_specialist)
        if (search.get("config_sha256") != fingerprint(raw_config)
                or read_receipt(search["config_receipt"]) != raw_config
                or search["config_receipt"]["sha256"] != report["sources"]["config"]["sha256"]):
            raise ValueError("Search configuration receipt does not match exact supplied config")
        engine = search["engine"]
        current = {str(p.relative_to(source_root)): sha(p) for p in sorted((Path(source_root)/"gsdeploy").rglob("*.py"))}
        if (engine.get("source_unchanged") is not True or engine.get("source_before") != engine.get("source_after")
                or engine.get("source_after") != current):
            raise ValueError("Search snapshot source receipts are changed/incomplete")
        report["snapshot_sources"] = current
        if raw_config["horizon_seconds"] != 1800:
            raise ValueError("This confirmation requires the searched H1800 configuration")
        if raw_config.get("search_policy") != "nominal":
            raise ValueError("This materializer requires nominal six-term search scores")
        transfer = raw_config["catalog"].get("replay_transfer_v1", {})
        if (raw_config["catalog"].get("performance_mode") != "replay_transfer_v1"
                or transfer.get("status") != "VALIDATED_ON_NEW_WORKLOAD"):
            raise ValueError("Independent new-workload transfer validation required")
        if warmup_source is None:
            protocol_receipt = transfer["validation_protocol"]
            protocol = read_receipt(protocol_receipt)
            report["sources"]["transfer_validation_protocol"] = protocol_receipt
            warmup_receipt = protocol["packet"]
        else:
            warmup_receipt = receipt(warmup_source)
        warmup = warmup_value(read_receipt(warmup_receipt))
        if type(warmup) not in (int, float) or not math.isfinite(warmup) or warmup < 0:
            raise ValueError("Invalid explicit warmup duration")
        report["sources"]["warmup"] = warmup_receipt
        roles = {}
        for row in search["candidates"]:
            if row["label"] in roles:
                raise ValueError("Duplicate role label: " + row["label"])
            roles[row["label"]] = row
        missing = set(ROLE_MAP) - set(roles)
        if missing:
            raise ValueError("Missing required search roles: " + ",".join(sorted(missing)))
        certificates = {}
        tolerance = raw_config["milp_absolute_gap"] + 1e-12
        for name in ROLE_MAP:
            try:
                cert = certify(roles[name], name, tolerance)
                if cert['raw_result'].get('domain',{}).get('capacity_boundary_witnesses') is not False:
                    raise ValueError('Materialization requires the declared grid-only domain without capacity witnesses')
                if read_receipt(cert["result"]["config_receipt"]) != raw_config:
                    raise ValueError("Per-role config receipt changed")
                if cert["raw_result"].get("domain", {}).get("K") != raw_config["route_grid"]:
                    raise ValueError("Per-role route-grid certificate differs from searched config")
                certificates[name] = cert
                report["role_checks"].append(dict(role=name,status="PASS",certificate_kind=cert["kind"]))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                report["role_checks"].append(dict(role=name,status="REJECTED",reason=str(exc)))
        if len(certificates) != len(ROLE_MAP):
            raise ValueError("One or more required roles are uncertified, unresolved, or ineligible")
        winner = certificates["free"]["plan"]
        if winner is None:
            raise ValueError("Free domain is certified infeasible; target deployment unavailable")
        report["target_structure"] = target_structure(raw_config['catalog'],winner,target_specialist)
        if target_specialist=='asr':
            report['target_structure'].update(asr_generalist_rps=report['target_structure']['target_generalist_rps'],
                asr_specialist_rps=report['target_structure']['target_specialist_rps'])
        if not report["target_structure"]["achieved"]:
            report["status"] = "TARGET_NOT_ACHIEVED"
            raise ValueError("Certified free winner does not naturally deploy "+target_specialist+">=2 and route its task positively to specialist and generalist")
        contract = dict(schema="gsdeploy.spillover_confirmation_contract.v1",study_id=study_id,
            workload=copy.deepcopy(raw_config["workload"]),horizon_seconds=raw_config["horizon_seconds"],
            warmup_seconds=warmup,seed_roles=copy.deepcopy(transfer["seed_roles"]),
            training_forbidden=True,purpose=purpose,required_roles=list(ROLE_MAP.values()),
            winner_label="free_search_winner",target_task=target_specialist,target_specialist=target_specialist,
            search_source=report["sources"]["search_results"],
            timing_source=warmup_receipt,physical_global_optimality_certified=False)
        contract['routing_domain']=dict(K=raw_config['route_grid'],kind='union_of_denominators_1_through_K',
            capacity_boundary_witnesses=False,original_absolute_rates_preserved=True)
        bundle = dict(schema="gsdeploy.spillover_confirmation_candidates.v1",
            config_sha256=fingerprint(raw_config),study_contract_sha256=fingerprint(contract),candidates=[],
            source_search=report["sources"]["search_results"],source_config=report["sources"]["config"],
            gpu_started=False,physical_global_optimality_certified=False)
        bundle.update(target_specialist=target_specialist,target_task=target_specialist)
        session_module = wrapper(Path(source_root).resolve())
        with session_module.SearchSession(Path(source_root).resolve(), raw_config) as session:
            session.module("replay_transfer").check_scoring(session.config)
            consumer = session.module("spillover_confirmation")
            for name,label in ROLE_MAP.items():
                cert = certificates[name];result=cert["result"];plan=cert["plan"]
                inspect_restrictions(name,result["search_restrictions"],winner,target_specialist)
                recorded_target(result,target_specialist)
                if name=='best_one_specialist_max':assert_one_specialist_plan(plan,target_specialist)
                p = constraint = None
                membership=None
                if plan is not None:
                    membership=route_grid_membership(plan,raw_config['workload'],raw_config['route_grid'])
                    constraint = session.config.validate_plan(plan["deployment"],plan["routing"])
                    p = session.predict(plan["deployment"],plan["routing"])
                    if not constraint["constraints_valid"] or not p.get("eligible") or not p.get("utility_complete"):
                        raise ValueError("Fresh formal replay rejects recorded role incumbent: " + name)
                    recorded = result["winner"]["score"]
                    if type(recorded) not in (int,float) or not math.isclose(recorded,p["U"],rel_tol=1e-10,abs_tol=1e-10):
                        raise ValueError("Fresh prediction score differs from recorded score: " + name)
                bundle["candidates"].append(dict(label=label,plan=copy.deepcopy(plan),
                    status="CERTIFIED_INFEASIBLE_FINITE_DOMAIN" if plan is None else result["status"],
                    optimality_certified=plan is not None,infeasibility_certified=plan is None,
                    certificate_kind=cert["kind"],raw_solver_status=result["status"],
                    optimality_gap=result.get("optimality_gap"),utility_upper_bound=result.get("utility_upper_bound"),
                    search_completed=result.get("search_completed"),search_restrictions=result["search_restrictions"],
                    raw_search_receipt=cert["raw_result_receipt"],search_config_receipt=result["config_receipt"],
                    search_budget=result.get("budget"),domain=result.get("domain"),
                    constraints=constraint,predicted_U=p.get("U") if p else None,
                    source_role=name,source_search_certificate=copy.deepcopy(result)))
                bundle['candidates'][-1]['routing_domain_membership']=membership
            consumer._gate(session.config,contract)
            records,unique = consumer._roles(session.config,bundle,contract)
        report.update(status="READY_FOR_PACKET_PREPARATION",ready=True,
            format_checked_by_snapshot_consumer=True,unique_physical_candidates=len(unique),
            confirmation_seeds=contract["seed_roles"]["confirmation"],
            planned_gpu_jobs=len(unique)*len(contract["seed_roles"]["confirmation"]),
            extra_search_roles_ignored=sorted(set(roles)-set(ROLE_MAP)))
        # Emit consumable files only after every required check has passed.
        save(output/"system-config.json",raw_config)
        save(output/"contract.json",contract)
        save(output/"candidates.json",bundle)
        save(output/"deduplicated-plans.json",dict(candidates=unique,candidate_roles=records,gpu_started=False))
    except Exception as exc:
        if report["status"] != "TARGET_NOT_ACHIEVED":
            report["status"] = "REJECTED_NOT_READY"
        report.update(ready=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    report["elapsed_seconds"] = time.perf_counter()-started
    save(output/"materialization-report.json",report)
    save(output/"manifest.json",dict(files={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        script=receipt(__file__),gpu_started=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-results",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,required=True,help="Exact search snapshot including its spillover confirmation consumer")
    parser.add_argument("--output",type=Path,required=True,help="New output directory; never overwritten")
    parser.add_argument("--study-id",required=True)
    parser.add_argument("--warmup-source",type=Path,help="Explicit study contract or packet; default reads validated transfer protocol's packet")
    parser.add_argument("--purpose",default="spillover_independent_confirmation_not_training")
    parser.add_argument('--target-specialist',choices=TARGET_SPECIALISTS,default='asr')
    args=parser.parse_args()
    result=materialize(args.search_results,args.config,args.source_root,args.output,study_id=args.study_id,
        warmup_source=args.warmup_source,purpose=args.purpose,target_specialist=args.target_specialist)
    print(json.dumps(dict(status=result["status"],ready=result["ready"],gpu_started=False)))
    if not result["ready"]:raise SystemExit(2)


if __name__=="__main__":main()
