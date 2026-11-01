#!/usr/bin/env python3
"""CPU role searches using an unchanged snapshot and validated replay transfer.

Never activates, repairs or relabels a transfer envelope. Failed roles and
budget incumbents remain explicit; output is a proposal bundle, not GPU jobs.
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
from target_specialist import TARGET_SPECIALISTS, validate_target, one_specialist_restriction

DEFAULT_ROLES = ("free", "best_global_no_split", "best_same_deployment_no_split",
                 "strongest_alternative_deployment", "best_one_specialist_max")
ALL_ROLES = (*DEFAULT_ROLES, "strongest_alternative_route")
DEPENDENT_ROLES = {"best_same_deployment_no_split", "strongest_alternative_deployment",
                   "strongest_alternative_route"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def ordinary(value):
    return json.loads(json.dumps(value, allow_nan=False))


def save(path, value):
    # Keep a readable complete checkpoint if the process stops between roles.
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def source_receipts(source_root):
    return {str(p.relative_to(source_root)): digest(p)
            for p in sorted((source_root / "gsdeploy").rglob("*.py"))}


def load_wrapper(source_root):
    path = source_root / "gsdeploy" / "search_session.py"
    spec = importlib.util.spec_from_file_location("spillover_isolated_search_session", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    if isinstance(raw["catalog"], str):
        catalog = Path(raw["catalog"])
        if not catalog.is_absolute():
            catalog = path.parent / catalog
        raw["catalog"] = json.loads(catalog.read_text())
    return raw


def anchor_from_file(path):
    raw = json.loads(Path(path).read_text())
    if "search" in raw:
        raw = raw["search"]
    winner = raw.get("winner")
    if not winner:
        raise ValueError("Anchor result has no feasible incumbent candidate")
    return {"deployment": winner["deployment"], "routing": winner["routing"]}, {
        "path": str(Path(path).resolve()), "sha256": digest(path),
        "status": raw.get("status"), "optimality_certified": raw.get("optimality_certified", False)}


def solve_roles(config, output, source_root, *, K=8, time_limit=120., seed=51,
                horizon=1800., roles=DEFAULT_ROLES, anchor_result=None,
                source_config_receipt=None, target_specialist='asr'):
    reject_extended_route_domain(config)
    if (type(K) is not int or not 1 <= K <= 16 or not math.isfinite(time_limit)
            or time_limit <= 0 or not math.isfinite(horizon) or horizon <= 0):
        raise ValueError("K1..16, positive finite solver budget/horizon required")
    if len(set(roles)) != len(roles) or not roles or set(roles) - set(ALL_ROLES):
        raise ValueError("Known, unique, nonempty role list required")
    if "free" in roles and roles[0] != "free":
        raise ValueError("Free search must run first so dependent roles use its incumbent")
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    raw = copy.deepcopy(config)
    raw.update(route_grid=K, milp_time_limit_seconds=time_limit,
               random_seed=seed, horizon_seconds=horizon, solver="milp",
               milp_capacity_witness=False)
    save(output / "system-config.json", raw)
    config_receipt = {"path": str(output / "system-config.json"),
                      "sha256": digest(output / "system-config.json")}
    source_before = source_receipts(source_root)
    bundle = dict(schema="gsdeploy.static_search_roles.v1",
        study_kind="seven_model_replay_transfer_static_spillover",
        status="PREFLIGHT", config_sha256=canonical_digest(raw),
        config_receipt=config_receipt, source_config_receipt=source_config_receipt,
        requested_roles=list(roles), candidates=[], physical_candidates=[],
        budget=dict(per_role_solver_time_limit_seconds=time_limit,
                    external_wall_includes_import_validation_and_serialization=True),
        scope="Validated transfer predictor and declared finite grid only; GPU candidate proposals require separate packet materialization.",
        seed=seed, horizon_seconds=horizon, K=K, gpu_started=False,
        target_specialist=target_specialist, target_task=target_specialist,
        physical_global_optimality_certified=False,
        engine=dict(source_root=str(source_root), source_before=source_before))
    save(output / "search-results.json", bundle)
    wrapper = None
    anchor = anchor_receipt = None
    try:
        # Inspect the explicit state before importing/starting any optimizer.
        catalog = raw["catalog"]
        transfer = catalog.get("replay_transfer_v1", {})
        if (catalog.get("performance_mode") != "replay_transfer_v1"
                or catalog.get("memory_mode") != "replay_transfer_v1"
                or transfer.get("status") != "VALIDATED_ON_NEW_WORKLOAD"):
            raise ValueError("Formal role searches require replay_transfer_v1 VALIDATED_ON_NEW_WORKLOAD")
        if len(catalog["models"]) != 7:
            raise ValueError("This study requires the seven-model pool")
        validate_target(catalog,target_specialist)
        wrapper, wrapper_path = load_wrapper(source_root)
        bundle["engine"].update(wrapper_path=str(wrapper_path), wrapper_sha256=digest(wrapper_path))
        with wrapper.SearchSession(source_root, raw) as session:
            session.module("replay_transfer").check_scoring(session.config)
            # SearchSession invokes the snapshot's full formal validation gate,
            # including source, code, independent validation protocol and report.
            raw = ordinary(session.config.as_dict())
            if anchor_result is not None:
                anchor, anchor_receipt = anchor_from_file(anchor_result)
                session.config.validate_plan(anchor["deployment"], anchor["routing"])
                nominal = session.predict(anchor["deployment"], anchor["routing"])
                if not nominal.get("eligible"):
                    raise ValueError("External anchor is not eligible under this exact search configuration")
        bundle.update(status="RUNNING", transfer_version_sha256=transfer.get("version_sha256"),
                      anchor_receipt=anchor_receipt)
        save(output / "preflight.json", dict(status="PASS", config_receipt=config_receipt,
            transfer_version_sha256=transfer.get("version_sha256"), gpu_started=False))
        save(output / "search-results.json", bundle)

        for role in roles:
            role_started = time.perf_counter()
            restrictions = {}
            if role in DEPENDENT_ROLES and anchor is None:
                result = dict(status="SKIPPED_NO_FREE_INCUMBENT", winner=None,
                    optimality_certified=False, optimality_gap=None,
                    utility_upper_bound=None, search_completed=False)
                save(output / (role + ".raw.json"), result)
                enriched = {**result, "external_wall_seconds": 0., "constraints": None,
                    "search_restrictions": {}, "config_receipt": config_receipt,
                    "reason": "Dependent role needs a feasible free-search incumbent or explicit validated anchor result"}
            else:
                if role == "best_global_no_split":
                    restrictions["no_split"] = True
                elif role == "best_same_deployment_no_split":
                    restrictions.update(deployment=anchor["deployment"], no_split=True)
                elif role == "strongest_alternative_deployment":
                    restrictions["exclude_deployments"] = [anchor["deployment"]]
                elif role == "strongest_alternative_route":
                    restrictions.update(deployment=anchor["deployment"], exclude_plans=[anchor])
                elif role == "best_one_specialist_max":
                    # The optimizer adds state zero. Only the target's positive options
                    # change; every other family's full measured state set stays.
                    restrictions.update(one_specialist_restriction(target_specialist))
                raw_result = None
                try:
                    if source_receipts(source_root) != source_before:
                        raise ValueError("Snapshot mathematical sources changed during this run")
                    with wrapper.SearchSession(source_root, raw) as session:
                        c = session.config
                        raw_result = ordinary(session.optimizer("milp").solve(K=K, **restrictions))
                        save(output / (role + ".raw.json"), raw_result)
                        constraints = None
                        eligible_incumbent = False
                        if raw_result.get("winner"):
                            w = raw_result["winner"]
                            constraints = ordinary(c.validate_plan(w["deployment"], w["routing"]))
                            nominal = w.get("prediction", {})
                            eligible_incumbent = bool(constraints.get("constraints_valid")
                                and nominal.get("eligible") and nominal.get("utility_complete")
                                and w.get("score") is not None)
                        enriched = {**raw_result, "constraints": constraints,
                            "eligible_incumbent": eligible_incumbent,
                            "incumbent_interpretation": ("CERTIFIED_FINITE_DOMAIN_OPTIMUM"
                                if eligible_incumbent and raw_result.get("optimality_certified")
                                else "UNCERTIFIED_BUDGET_OR_PARTIAL_INCUMBENT" if eligible_incumbent
                                else "NO_ELIGIBLE_INCUMBENT"),
                            "session_cache": session.stats(), "config_receipt": config_receipt,
                            "search_restrictions": ordinary(restrictions),
                            "anchor_receipt": anchor_receipt if role in DEPENDENT_ROLES else None,
                            "budget": dict(solver_time_limit_seconds=time_limit,
                                max_iterations=raw["milp_max_iterations"],max_columns=raw["milp_max_columns"]),
                            "external_wall_seconds": time.perf_counter()-role_started}
                        if role == "free" and eligible_incumbent:
                            anchor = {"deployment": w["deployment"], "routing": w["routing"]}
                            anchor_receipt = dict(path=str(output/(role+".raw.json")),
                                sha256=digest(output/(role+".raw.json")),status=raw_result["status"],
                                optimality_certified=raw_result.get("optimality_certified", False))
                        elif role == "free":
                            anchor = anchor_receipt = None
                except Exception as exc:
                    if raw_result is None:
                        save(output / (role + ".raw.json"), dict(status="ROLE_EXCEPTION_BEFORE_RESULT",
                            exception_type=type(exc).__name__, error=str(exc)))
                    enriched = dict(status="ROLE_FAILED", winner=None, eligible_incumbent=False,
                        optimality_certified=False, optimality_gap=None, utility_upper_bound=None,
                        constraints=None, error_type=type(exc).__name__, error=str(exc),
                        traceback=traceback.format_exc(), search_restrictions=ordinary(restrictions),
                        raw_result_receipt=dict(path=str(output/(role+".raw.json")),
                                                sha256=digest(output/(role+".raw.json"))),
                        config_receipt=config_receipt, budget=dict(solver_time_limit_seconds=time_limit),
                        external_wall_seconds=time.perf_counter()-role_started)
                    if role == "free":
                        anchor = anchor_receipt = None
            enriched.setdefault("budget", dict(solver_time_limit_seconds=time_limit,
                max_iterations=raw["milp_max_iterations"], max_columns=raw["milp_max_columns"]))
            enriched.setdefault("eligible_incumbent", False)
            enriched["comparison_role"] = role
            enriched['target_specialist']=target_specialist
            enriched["role_scope"] = (target_specialist+" specialist replica count 0 or 1; every other model retains every measured count"
                if role == "best_one_specialist_max" else
                "Conditional on free-search incumbent deployment; incumbent need not be certified"
                if role in DEPENDENT_ROLES else "Full seven-model measured replica domain")
            save(output / (role + ".json"), enriched)
            item = dict(label=role, search=enriched,
                raw_result_receipt=dict(path=str(output/(role+".raw.json")),sha256=digest(output/(role+".raw.json"))))
            bundle["candidates"].append(item)
            # Dedupe proposals by deployment and absolute task rates; preserve
            # all originating roles without implying an incumbent is optimal.
            if enriched.get("eligible_incumbent"):
                w = enriched["winner"]
                plan = {"deployment":w["deployment"],"routing":w["routing"]}
                key = canonical_digest(plan)
                candidate = next((p for p in bundle["physical_candidates"] if p["plan_key"]==key),None)
                role_evidence = dict(label=role,status=enriched["status"],
                    optimality_certified=enriched.get("optimality_certified",False),
                    optimality_gap=enriched.get("optimality_gap"),utility_upper_bound=enriched.get("utility_upper_bound"))
                if candidate:
                    candidate["roles"].append(role_evidence)
                else:
                    bundle["physical_candidates"].append(dict(plan_key=key,plan=plan,roles=[role_evidence]))
            save(output / "search-results.json", bundle)
            print(json.dumps(dict(role=role,status=enriched["status"],
                U=enriched.get("winner",{}).get("score") if enriched.get("winner") else None,
                optimality_certified=enriched.get("optimality_certified",False),
                optimality_gap=enriched.get("optimality_gap"),external_wall_seconds=enriched["external_wall_seconds"])),flush=True)
        bundle["status"] = ("ROLES_COMPLETED_WITH_FAILURES_OR_SKIPS" if any(
            x["search"]["status"] in {"ROLE_FAILED","SKIPPED_NO_FREE_INCUMBENT"} for x in bundle["candidates"])
            else "REQUESTED_ROLES_COMPLETED")
    except Exception as exc:
        bundle.update(status="PREFLIGHT_FAILED",error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        save(output / "preflight.json",dict(status="FAILED",error_type=type(exc).__name__,error=str(exc),gpu_started=False))
    finally:
        after = source_receipts(source_root)
        bundle["engine"].update(source_after=after,source_unchanged=after==source_before)
        bundle["external_wall_seconds"] = time.perf_counter()-started
        save(output / "search-results.json",bundle)
        save(output / "candidate-bundle.json",dict(schema="spillover.materialization_candidates.v1",
            search_bundle_receipt=dict(path=str(output/"search-results.json"),sha256=digest(output/"search-results.json")),
            config_receipt=config_receipt,candidates=bundle["physical_candidates"],
            target_specialist=target_specialist,target_task=target_specialist,
            status="PROPOSALS_ONLY_NOT_AN_EXECUTION_PACKET",gpu_started=False,
            physical_global_optimality_certified=False))
        save(output / "manifest.json",dict(files={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!="manifest.json"},
            script=dict(path=str(Path(__file__).resolve()),sha256=digest(__file__)),
            source_unchanged=after==source_before,gpu_started=False))
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",required=True,type=Path,help="Frozen new gsdeploy snapshot root")
    parser.add_argument("--config",required=True,type=Path,help="Already VALIDATED_ON_NEW_WORKLOAD transfer JSON")
    parser.add_argument("--output",required=True,type=Path,help="New directory; existing outputs are never overwritten")
    parser.add_argument("--K","--route-grid",dest="K",type=int,default=8)
    parser.add_argument("--time-limit",type=float,default=120.,help="Solver budget per role; import/validation wall time recorded separately")
    parser.add_argument("--seed",type=int,default=51)
    parser.add_argument("--horizon",type=float,default=1800.)
    parser.add_argument("--roles",nargs="+",choices=ALL_ROLES,default=list(DEFAULT_ROLES))
    parser.add_argument("--extra-alternative-route",action="store_true",help="Append strongest_alternative_route as a sixth role")
    parser.add_argument("--anchor-result",type=Path,help="Free raw/enriched result for separately run dependent roles")
    parser.add_argument('--target-specialist',choices=TARGET_SPECIALISTS,default='asr')
    args = parser.parse_args()
    roles = list(args.roles)
    if args.extra_alternative_route and "strongest_alternative_route" not in roles:
        roles.append("strongest_alternative_route")
    if any(r in DEPENDENT_ROLES for r in roles) and "free" not in roles and args.anchor_result is None:
        parser.error("Dependent-only runs require --anchor-result")
    config_path = args.config.resolve()
    result = solve_roles(load_config(config_path),args.output,args.source_root,K=args.K,
        time_limit=args.time_limit,seed=args.seed,horizon=args.horizon,roles=roles,
        anchor_result=args.anchor_result,source_config_receipt=dict(path=str(config_path),sha256=digest(config_path)),
        target_specialist=args.target_specialist)
    print(json.dumps(dict(status=result["status"],roles=len(result["candidates"]),
        physical_candidates=len(result["physical_candidates"]),gpu_started=False)))
    if result["status"]=="PREFLIGHT_FAILED":
        raise SystemExit(2)


if __name__=="__main__":
    main()
