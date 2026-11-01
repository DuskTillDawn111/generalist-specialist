"""Assemble a receipt-bound static-study account; never solve, fit or use a GPU."""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


TERMS = ("Q", "T", "L", "P", "M", "E")
ROLES = ("free_search_winner", "strongest_alternative_deployment", "strongest_alternative_route",
         "best_no_split", "best_generalist_only", "best_specialist_only")
ROLE_ZH = ("自由搜索选中方案", "最强其他部署", "同部署最强其他路由", "最优不分流", "最优纯通", "最优纯专")
ALGORITHMS = ("milp", "dfs", "plain_dfs")
SCOPE = "冻结预测器及声明有限域；独立实测只覆盖实际确认的候选与种子；未证明物理全组合全局最优。"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def bound(item):
    path = Path(item["path"])
    if not path.is_absolute() or not path.is_file() or sha(path) != item["sha256"]:
        raise ValueError("Changed or unavailable receipt: " + str(path))
    return path


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def scoring(utility, weights):
    if utility.get("eligible") is not True or not finite(utility.get("U")):
        raise ValueError("Expected eligible complete six-term utility")
    values = {term: utility.get(term + "_norm") for term in TERMS}
    if not all(finite(v) for v in values.values()):
        raise ValueError("Missing or invalid utility component")
    costs = {term: weights[term] * values[term] for term in TERMS}
    if not math.isclose(utility["U"], -math.fsum(costs.values()), rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("U does not match frozen six-term weights")
    return costs


def role_title(role):
    """Keep machine role IDs frozen while making evidence strength explicit."""
    label = role["label"]
    if role.get("winner") is None:
        names = dict(zip(ROLES, ("自由搜索", "其他部署", "同部署其他路由", "不分流", "纯通", "纯专")))
        return names[label] + "无选中方案"
    if role.get("optimality_certified") is True:
        return dict(zip(ROLES, ROLE_ZH))[label]
    names = dict(zip(ROLES, ("自由搜索", "其他部署", "同部署其他路由", "不分流", "纯通", "纯专")))
    budget = any(word in role.get("status", "") for word in ("BUDGET", "LIMIT"))
    return names[label] + ("预算候选" if budget else "当前候选")


def role_summary(label, result, config):
    winner = result.get("winner")
    entry = {"label": label, "status": result["status"],
        "search_completed": result.get("search_completed"), "optimality_certified": result.get("optimality_certified"),
        "utility_upper_bound": result.get("utility_upper_bound"), "optimality_gap": result.get("optimality_gap"),
        "certificate_scope": result.get("certificate_scope"), "domain": result.get("domain"),
        "unknown_candidates": result.get("metrics", {}).get("unknown_candidates"),
        "unknown_domain_lower_cost": result.get("unknown_domain_lower_cost"),
        "unresolved_candidates": result.get("unresolved_candidates"), "metrics": result.get("metrics", {}),
        "full_prediction_calls": result.get("metrics", {}).get("full_prediction_calls"),
        "external_wall_seconds": result.get("external_wall_seconds"), "elapsed_seconds": result.get("elapsed_seconds"),
        "proof_completed_seconds": result.get("proof_completed_seconds"),
        "search_restrictions": result.get("search_restrictions"), "winner": None}
    if winner:
        plan = {k: winner[k] for k in ("deployment", "routing")}
        if set(plan["routing"]) != set(config["workload"]):
            raise ValueError("Winner does not cover frozen tasks")
        shares = {}
        for task, rate in config["workload"].items():
            routes = plan["routing"][task]
            if (not finite(rate) or rate <= 0 or not all(finite(v) and v >= 0 for v in routes.values())
                    or not math.isclose(sum(routes.values()), rate, rel_tol=1e-8, abs_tol=1e-10)
                    or any(m not in plan["deployment"] for m, v in routes.items() if v > 0)):
                raise ValueError("Winner route violates workload conservation")
            shares[task] = {m: v / rate for m, v in routes.items()}
        costs = scoring(winner["prediction"], config["weights"])
        for model, count in plan["deployment"].items():
            if (model not in config["catalog"]["models"] or type(count) is not int or count <= 0
                    or str(count) not in config["catalog"]["capacity_factors"][model]):
                raise ValueError("Winner has invalid resident model/count")
        if any(task not in config["catalog"]["models"][m]["supported_tasks"]
               for task, routes in plan["routing"].items() for m, rate in routes.items() if rate > 0):
            raise ValueError("Winner routes to an unsupported task path")
        entry["winner"] = {**plan, "plan_key": canonical(plan), "routing_share": shares,
            "U": winner["prediction"]["U"], "objective_score": winner.get("score"),
            "objective": winner.get("objective"), "weighted_costs": costs,
            "normalized_terms": {t: winner["prediction"][t + "_norm"] for t in TERMS}}
    entry["display_label"] = role_title(entry)
    return entry


def load_search(directory, validated):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = read(manifest_path)
    required = {"system-config.json", "search-results.json", *(r + ".json" for r in ROLES)}
    if not required.issubset(manifest.get("files", {})):
        raise ValueError("Search manifest lacks the complete six-role contract")
    receipts = [receipt(manifest_path)]
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha(path) != expected:
            raise ValueError("Changed search manifest member: " + name)
        receipts.append(receipt(path))
    config = read(directory / "system-config.json")
    if config.get("route_pair_denominators_by_task"):
        raise ValueError("PAIR_ROUTE_DOMAIN_UNSUPPORTED_BY_LEGACY_STUDY_REPORT: use a versioned domain-aware report")
    weights = config["weights"]
    if set(weights) != set(TERMS) or not all(finite(v) and v >= 0 for v in weights.values()):
        raise ValueError("Six finite nonnegative utility weights required")
    models = config["catalog"]["models"]
    if len(models) != 7 or {v["role"] for v in models.values()} != {"generalist", "specialist"}:
        raise ValueError("This study account requires the frozen seven-model pool")
    spec = config["catalog"].get("calibrated_replay_v2", {})
    if (config["catalog"].get("performance_mode") != "calibrated_replay_v2"
            or spec.get("version_sha256") != validated["version_sha256"]
            or spec.get("status") != "VALIDATED_ON_HELDOUT_CONDITIONS"):
        raise ValueError("Search does not use the supplied validated v2 predictor")
    bundle = read(directory / "search-results.json")
    if (bundle.get("schema") != "gsdeploy.static_search_roles.v1"
            or bundle.get("config_sha256") != canonical(config)
            or [c["label"] for c in bundle["candidates"]] != list(ROLES)
            or bundle.get("physical_global_optimality_certified") is not False):
        raise ValueError("Changed or incomplete frozen search bundle")
    summaries = []
    for candidate in bundle["candidates"]:
        label, result = candidate["label"], candidate["search"]
        if result != read(directory / (label + ".json")):
            raise ValueError("Bundled and individual role results differ")
        expected = copy.deepcopy(config)
        if label in {"best_generalist_only", "best_specialist_only"}:
            role = "generalist" if label == "best_generalist_only" else "specialist"
            keep = {m for m, spec in models.items() if spec["role"] == role}
            for field in ("models", "capacity_factors", "memory_table_mib"):
                expected["catalog"][field] = {m: v for m, v in expected["catalog"][field].items() if m in keep}
        if result.get("problem") != expected:
            raise ValueError("Role changed the frozen scoring/domain contract: " + label)
        if label in {"best_generalist_only", "best_specialist_only"}:
            restrictions = result.get("search_restrictions") or {}
            if (restrictions.get("fixed_deployment") or restrictions.get("exclude_deployments")
                    or restrictions.get("exclude_plans") or restrictions.get("no_split")):
                raise ValueError("Pure comparison unexpectedly restricts the pure search domain")
        summaries.append(role_summary(label, result, expected))
    selected = summaries[0]["winner"]
    for role in summaries:
        winner = role["winner"]
        if winner is None:
            continue
        if selected and role["label"] == "strongest_alternative_deployment" and winner["deployment"] == selected["deployment"]:
            raise ValueError("Alternative deployment repeats the selected deployment")
        if selected and role["label"] == "strongest_alternative_route" and (
                winner["deployment"] != selected["deployment"] or winner["plan_key"] == selected["plan_key"]):
            raise ValueError("Alternative route must differ on the selected deployment")
        if role["label"] == "best_no_split" and any(sum(v > 0 for v in row.values()) > 1 for row in winner["routing"].values()):
            raise ValueError("No-split control contains split traffic")
    catalog_options = {m: sorted(int(n) for n in config["catalog"]["capacity_factors"][m] if int(n) > 0) for m in models}
    domain = {"route_grid_K": config["route_grid"], "routing": "union of denominators 1..K",
        "continuous_optimality_certified": False, "absence_allowed": True,
        "catalog_positive_replica_options": catalog_options,
        "effective_replica_options": summaries[0]["search_restrictions"].get("replica_options") or catalog_options,
        "capacity_boundary_witnesses": config.get("milp_capacity_witness"),
        "search_max_states": config.get("search_max_states"), "time_limit_seconds": config.get("milp_time_limit_seconds"),
        "absolute_gap_tolerance": config.get("milp_absolute_gap"), "policy": config.get("search_policy")}
    pool = [{"id": m, "name": spec.get("name"), "role": spec["role"], "supported_tasks": spec.get("supported_tasks")} for m, spec in models.items()]
    return {"config_sha256": canonical(config), "search_bundle": receipt(directory / "search-results.json"), "models": pool, "weights": weights,
        "workload": config["workload"], "horizon_seconds": config["horizon_seconds"],
        "seed": config["random_seed"], "arrival_process": config["arrival_process"],
        "gpu_memory_mib": config["gpu_memory_mib"], "domain": domain, "roles": summaries,
        "pure_controls": {role: [m for m, s in models.items() if s["role"] == role] for role in ("generalist", "specialist")}}, config, receipts


def additional_search_evidence(directories, validated, main_search, main_config):
    """Retain nested-grid candidates without replacing the frozen confirmation roster."""
    # These limits affect search effort/termination, not the mathematical domain.
    effort_fields = {"route_grid", "milp_time_limit_seconds", "milp_max_iterations", "milp_max_columns",
        "search_max_states", "milp_absolute_gap", "solver"}
    baseline = {k: v for k, v in main_config.items() if k not in effort_fields}
    main_K = main_config["route_grid"]
    evidence, receipts = [], []
    seen = set()
    for directory in directories or []:
        directory = Path(directory).resolve()
        if directory in seen:
            raise ValueError("Duplicate additional search evidence directory")
        seen.add(directory)
        search, config, files = load_search(directory, validated)
        if ({k: v for k, v in config.items() if k not in effort_fields} != baseline
                or not 1 <= config["route_grid"] <= main_K
                or config.get("milp_capacity_witness") or main_config.get("milp_capacity_witness")):
            raise ValueError("Additional evidence is not a nested grid under the same frozen mathematical contract")
        if search["domain"]["effective_replica_options"] != main_search["domain"]["effective_replica_options"]:
            raise ValueError("Additional evidence changes the effective replica domain")
        differences = {k: {"additional": config.get(k), "main": main_config.get(k)}
            for k in set(config) | set(main_config) if config.get(k) != main_config.get(k)}
        evidence.append({"source_directory": str(directory), "source_bundle": search["search_bundle"],
            "grid_nested_in_main": True, "contract_differences": differences,
            "main_frozen_roles_replaced": False, "formal_confirmation_packet_changed": False,
            "search": search, "scope": "额外嵌套网格搜索证据；可改进已知预测候选下界，不能改变已冻结正式确认名单或把子域上界移作主域上界。"})
        receipts.extend(files)
    pure = []
    selected = main_search["roles"][0]["winner"]
    selected_score = selected.get("objective_score") if selected else None
    tolerance = main_config.get("milp_absolute_gap", 1e-6)
    for label in ("best_generalist_only", "best_specialist_only"):
        main = next(r for r in main_search["roles"] if r["label"] == label)
        options = []
        if main["winner"]:
            options.append({"winner": main["winner"], "source_bundle": main_search["search_bundle"],
                "source_K": main_K, "source_status": main["status"], "source_subdomain_optimality_certified": main["optimality_certified"],
                "source": "MAIN_FROZEN_ROLE"})
        for item in evidence:
            role = next(r for r in item["search"]["roles"] if r["label"] == label)
            if role["winner"]:
                shares = role["winner"]["routing_share"]
                # Each task must use one denominator from the declared union.
                if any(not any(all(math.isclose(v * d, round(v * d), abs_tol=1e-8) for v in row.values())
                                   for d in range(1, main_K + 1)) for row in shares.values()):
                    raise ValueError("Additional pure candidate is outside the main route grid")
                options.append({"winner": role["winner"], "source_bundle": item["source_bundle"],
                    "source_K": item["search"]["domain"]["route_grid_K"], "source_status": role["status"],
                    "source_subdomain_optimality_certified": role["optimality_certified"], "source": "ADDITIONAL_NESTED_GRID"})
        if any(not finite(item["winner"].get("objective_score")) for item in options):
            raise ValueError("Additional candidates need finite comparable objective scores")
        best = max(options, key=lambda item: item["winner"]["objective_score"]) if options else None
        upper = main["utility_upper_bound"]
        stronger = bool(best and main["winner"] and best["winner"]["objective_score"] > main["winner"]["objective_score"] + tolerance)
        excluded = finite(upper) and finite(selected_score) and upper < selected_score - tolerance
        pure.append({"label": label, "display_label": main["display_label"],
            "frozen_main_candidate": main["winner"], "main_pure_domain_optimality_certified": main["optimality_certified"],
            "main_pure_domain_upper_bound": upper, "best_reported_predictor_candidate": best,
            "additional_candidate_stronger_than_frozen_main": stronger,
            "all_reported_candidates": options,
            "excluded_from_full_pool_optimum_by_main_upper_bound": excluded,
            "full_pool_incumbent_objective_score": selected_score,
            "exclusion_margin": selected_score - upper if excluded else None,
            "pure_subdomain_proven_by_this_additional_evidence": False,
            "formal_confirmation_roster_unchanged": True,
            "interpretation": "纯域上界低于全池可行 incumbent 可排除纯域赢得全池最优；这不证明纯域内当前候选最优。额外更强候选仅作证据补充，未替换正式确认方案。"})
    return {"nested_searches": evidence, "pure_candidate_assessments": pure,
        "formal_confirmation_roster_unchanged": True}, receipts


def efficiency_summary(path, validated):
    path = Path(path).resolve()
    report = read(path)
    preflight = report["preflight"]
    if preflight.get("predictor_version") != validated["version_sha256"]:
        raise ValueError("Efficiency comparison uses a different predictor")
    receipts = [receipt(path)]
    if read(path.parent / "preflight.json") != preflight:
        raise ValueError("Efficiency preflight differs from the report")
    receipts.append(receipt(path.parent / "preflight.json"))
    records = report["records"]
    repeats = sorted({r["repeat"] for r in records})
    if not repeats or repeats != list(range(len(repeats))):
        raise ValueError("Missing benchmark repeats")
    for repeat in repeats:
        if sorted(r["algorithm"] for r in records if r["repeat"] == repeat) != sorted(ALGORITHMS):
            raise ValueError("Every repeat must retain all three algorithms, including failures")
    for row in records:
        run_path = path.parent / f"run-{row['repeat']}-{row['algorithm']}.json"
        if read(run_path) != row:
            raise ValueError("Efficiency run file differs from report")
        receipts.append(receipt(run_path))
    fair = all(r.get("effective_problem_sha256") == preflight["effective_problem_sha256"]
        and r.get("trace_sha256") == preflight["trace_sha256"] and r.get("fairness") == preflight["fairness"] for r in records)
    unchanged = (report.get("source_before") == report.get("source_after")
        and report.get("benchmark_source_before") == report.get("benchmark_source_after")
        and report.get("source_unchanged") is True and report.get("config_unchanged") is True)
    if report.get("same_contract") is not fair:
        raise ValueError("Efficiency fairness claim disagrees with retained records")
    comparisons = []
    for repeat in repeats:
        results = [r.get("result", {}) for r in records if r["repeat"] == repeat]
        scores = [r["winner"].get("score") if r.get("winner") else None for r in results]
        agrees = all(finite(v) for v in scores) and max(scores) - min(scores) <= preflight["fairness"]["absolute_gap"]
        plans = [canonical({k: r["winner"][k] for k in ("deployment", "routing")}) if r.get("winner") else None for r in results]
        all_certified = all(r.get("optimality_certified") is True for r in results)
        comparisons.append({"repeat": repeat, "objective_agrees": agrees, "all_certified": all_certified,
            "certified_objective_agrees": agrees and all_certified,
            "plan_identity_agrees": None not in plans and len(set(plans)) == 1})
    if report.get("per_repeat") != comparisons:
        raise ValueError("Efficiency agreement claim disagrees with retained results")
    complete = all(r["certified_objective_agrees"] for r in comparisons) and fair and unchanged
    if report.get("benchmark_complete") is not complete:
        raise ValueError("Efficiency completion claim disagrees with retained records")
    summaries = []
    outcomes = []
    for algorithm in ALGORITHMS:
        rows = [r for r in records if r["algorithm"] == algorithm and "result" in r]
        for row in rows:
            if not finite(row.get("search_wall_seconds")) or row["search_wall_seconds"] <= 0:
                raise ValueError("Invalid benchmark wall time")
        summaries.append({"algorithm": algorithm, "completed_processes": len(rows),
            "certificates": sum(r["result"].get("optimality_certified") is True for r in rows),
            "median_search_wall_seconds": statistics.median(r["search_wall_seconds"] for r in rows) if rows else None,
            "median_parent_wall_seconds": statistics.median(r["parent_process_wall_seconds"] for r in rows) if rows else None,
            "median_full_predictions": statistics.median(r["result"]["metrics"]["full_prediction_calls"] for r in rows) if rows else None,
            "max_peak_rss_kib": max((r["peak_rss_kib"] for r in rows), default=None)})
        for row in (r for r in records if r["algorithm"] == algorithm):
            result = row.get("result", {})
            outcomes.append({"repeat": row["repeat"], "algorithm": algorithm,
                "status": result.get("status", "PROCESS_FAILED"), "optimality_certified": result.get("optimality_certified", False),
                "incumbent_U": result["winner"].get("score") if result.get("winner") else None,
                "utility_upper_bound": result.get("utility_upper_bound"), "optimality_gap": result.get("optimality_gap"),
                "unknown_candidates": result.get("metrics", {}).get("unknown_candidates"),
                "full_prediction_calls": result.get("metrics", {}).get("full_prediction_calls"),
                "search_wall_seconds": row.get("search_wall_seconds"), "parent_process_wall_seconds": row.get("parent_process_wall_seconds"),
                "process_failed": row.get("process_failed", False)})
    if report.get("summaries") != summaries:
        raise ValueError("Efficiency medians/certificate counts disagree with retained records")
    times = {s["algorithm"]: s["median_search_wall_seconds"] for s in summaries}
    ratios = {rival: times[rival] / times["milp"] if times[rival] is not None and times["milp"] else None for rival in ("dfs", "plain_dfs")}
    return {"status": "CERTIFIED_SAME_OBJECTIVE" if complete else "INCOMPLETE_OR_UNCERTIFIED",
        "domain": preflight["domain"], "replica_options": preflight["replica_options"], "source_root": preflight["source_root"],
        "effective_problem_sha256": preflight["effective_problem_sha256"], "same_contract": fair,
        "source_and_config_unchanged": unchanged, "per_repeat": comparisons, "summaries": summaries, "outcomes": outcomes,
        "observed_wall_ratio_rival_over_milp": ratios,
        "certified_solve_speedup_rival_over_milp": ratios if complete else {"dfs": None, "plain_dfs": None},
        "milp_faster_than_both_at_certified_same_objective": complete and all(v is not None and v > 1 for v in ratios.values()),
        "timing_scope": report.get("timing_scope"), "scope": report.get("scope"),
        "interpretation": "仅在三算法同域、同优且均有证书时解释为求解加速；预算耗尽时只报告观测耗时比与 incumbent/gap。",
        "full_seven_model_domain_proved_by_this_comparison": False}, receipts


def confirmation_summary(path, search, config):
    if path is None:
        return {"status": "NOT_DONE", "completed_jobs": None, "planned_jobs": None,
            "paired": None, "measured_runs": None, "reason": "未提供正式独立确认报告；预测 U 不能替代 GPU 实测 U。"}, []
    path = Path(path).resolve()
    report = read(path)
    if (report.get("schema") != "gsdeploy.independent_confirmation_report.v1"
            or report.get("physical_global_optimality_certified") is not False
            or report.get("status") not in {"ALL_FROZEN_JOBS_VERIFIED", "INCOMPLETE_CONFIRMATION"}):
        raise ValueError("Expected strict independent confirmation report")
    packet_path = bound(report["packet"])
    packet = read(packet_path)
    if (packet.get("confirmation_sha256") != canonical({k: v for k, v in packet.items() if k != "confirmation_sha256"})
            or packet.get("config") != config):
        raise ValueError("Confirmation packet differs from the frozen search config")
    receipts = [receipt(path), receipt(packet_path)]
    if packet.get("frozen_search", {}).get("sha256") != search["search_bundle"]["sha256"]:
        raise ValueError("Confirmation was prepared from a different search bundle")
    receipts.append(receipt(bound(packet["frozen_search"])))
    if report.get("candidate_roles") != packet["candidate_roles"]:
        raise ValueError("Confirmation roles differ from packet")
    by_role = {r["label"]: r for r in search["roles"]}
    if [r["label"] for r in packet["candidate_roles"]] != list(ROLES):
        raise ValueError("Missing confirmation roles")
    for role in packet["candidate_roles"]:
        expected = by_role[role["label"]]
        key = expected["winner"]["plan_key"] if expected["winner"] else None
        if role["plan_key"] != key or role["status"] != expected["status"] or role["optimality_certified"] != expected["optimality_certified"]:
            raise ValueError("Confirmation changed the frozen search selection")
    jobs = {j["id"]: j for j in packet["jobs"]}
    seeds = packet["confirmation"]["seeds"]
    candidate_map = {c["plan_key"]: c for c in packet["candidates"]}
    candidates = set(candidate_map)
    expected_keys = {r["winner"]["plan_key"] for r in search["roles"] if r["winner"]}
    if (candidates != expected_keys or len(candidate_map) != len(packet["candidates"])
            or any(key != canonical(c["plan"]) for key, c in candidate_map.items())):
        raise ValueError("Confirmation changed the frozen candidate plans")
    if (seeds != [43, 44, 45] or len(jobs) != len(packet["jobs"]) or len(jobs) != len(candidates) * len(seeds)
            or {(j["candidate_key"], j["seed"]) for j in jobs.values()} != {(k, s) for k in candidates for s in seeds}):
        raise ValueError("Changed confirmation seed/job roster")
    seen, rows = set(), []
    for row in report["runs"]:
        job_id = row["job_id"]
        if job_id not in jobs or job_id in seen:
            raise ValueError("Unknown or duplicate confirmation job")
        seen.add(job_id)
        job = jobs[job_id]
        if (row["plan_key"] != job["candidate_key"] or row["seed"] != job["seed"]
                or job.get("purpose") != "independent_confirmation_not_training"
                or job.get("phase") != "confirmation" or job["id"] != canonical(job["identity"])
                or job["identity"].get("seed") != job["seed"]
                or job["plan"] != candidate_map[job["candidate_key"]]["plan"]):
            raise ValueError("Confirmation summary changed candidate/seed")
        raw_path = bound(row)
        raw = read(raw_path)
        if (raw.get("experiment_job_id") != job_id or raw.get("experiment_phase") != "confirmation"
                or raw.get("backend") != "native" or raw.get("status") != "completed"
                or raw.get("confirmation_packet_sha256") != packet["confirmation_sha256"]):
            raise ValueError("Wrong native confirmation identity/phase")
        actual = copy.deepcopy(raw.get("executed_measurement_identity", {}))
        if raw.get("executed_measurement_id") != canonical(actual) or raw.get("plan") != job["plan"]:
            raise ValueError("Changed native confirmation executed identity/plan")
        for worker in actual.get("workers", {}).values():
            worker.get("env", {}).pop("GS_CHECKPOINT_FILES_JSON", None)
        if actual != job["identity"]:
            raise ValueError("Confirmation did not execute the frozen job identity")
        costs = scoring(row["utility"], config["weights"])
        if any(not finite(row.get("weighted_costs", {}).get(t)) or not math.isclose(row["weighted_costs"][t], costs[t], rel_tol=1e-8, abs_tol=1e-8) for t in TERMS):
            raise ValueError("Changed measured confirmation costs")
        receipts.append(receipt(raw_path))
        rows.append({k: row[k] for k in ("job_id", "plan_key", "seed", "utility", "weighted_costs")})
    missing = sorted(set(jobs) - seen)
    if (report["missing_job_ids"] != missing or report["completed_jobs"] != len(seen) or report["planned_jobs"] != len(jobs)
            or report["status"] != ("INCOMPLETE_CONFIRMATION" if missing else "ALL_FROZEN_JOBS_VERIFIED")):
        raise ValueError("False confirmation completion claim")
    winner = by_role["free_search_winner"]["winner"]
    paired = report["paired"]
    if paired.get("winner_key") != (winner["plan_key"] if winner else None) or paired.get("planned_seeds") != seeds:
        raise ValueError("Paired comparison changed the frozen winner/seed blocks")
    groups = {(r["plan_key"], r["seed"]): r["utility"]["U"] for r in rows}
    pairs = paired.get("comparisons", [])
    if (len(pairs) != len(candidates) - 1 or {p["rival_key"] for p in pairs} != candidates - {paired["winner_key"]}):
        raise ValueError("Paired report omitted a frozen rival")
    if paired.get("all_planned_blocks_complete") is not (not missing):
        raise ValueError("Paired completion status disagrees with planned blocks")
    for pair in pairs:
        available = [s for s in seeds if (paired["winner_key"], s) in groups and (pair["rival_key"], s) in groups]
        deltas = [groups[paired["winner_key"], s] - groups[pair["rival_key"], s] for s in available]
        mean = statistics.fmean(deltas) if deltas else None
        observed_mean = pair.get("mean_U_difference")
        mean_agrees = observed_mean is None if mean is None else finite(observed_mean) and math.isclose(observed_mean, mean, rel_tol=1e-8, abs_tol=1e-8)
        if (pair.get("winner_key") != paired["winner_key"] or pair.get("paired_seeds") != available
                or pair.get("missing_seeds") != [s for s in seeds if s not in available]
                or pair.get("U_differences") != deltas
                or pair.get("status") != ("COMPLETE_PAIRED_BLOCKS" if len(available) == len(seeds) else "INCOMPLETE_BLOCKS")
                or not mean_agrees):
            raise ValueError("Paired differences disagree with measured seed utilities")
        if len(available) < len(seeds):
            if "student_interval" in pair or "exact_two_sided_sign_flip_p" in pair:
                raise ValueError("Incomplete pairs must remain descriptive")
        else:
            observed = abs(math.fsum(deltas))
            p = sum(abs(math.fsum(s * d for s, d in zip(signs, deltas))) >= observed - 1e-12
                    for signs in itertools.product((-1, 1), repeat=len(deltas))) / 2 ** len(deltas)
            if not finite(pair.get("exact_two_sided_sign_flip_p")) or not math.isclose(pair["exact_two_sided_sign_flip_p"], p, abs_tol=1e-12):
                raise ValueError("Paired exact sign-flip p disagrees with measured differences")
    return {"status": report["status"], "completed_jobs": len(seen), "planned_jobs": len(jobs),
        "missing_job_ids": missing, "paired": paired, "measured_runs": rows,
        "scope": "严格上游报告的配对统计及实测成本；此汇总核对种子差值与精确符号翻转 p，保留上游 Student 区间/Holm 校正，不重跑原生导入。三种子精确双侧符号翻转最小 p=0.25。"}, receipts


def assemble(calibration_report, validated_report, search_dir, efficiency_report, output, *, confirmation_report=None,
             additional_search_evidence_dirs=None, cpu_fixture=False):
    failed, validated = read(calibration_report), read(validated_report)
    if failed.get("schema") != "gsdeploy.calibrated_surrogate.v1" or failed.get("status") != "VALIDATION_FAILED":
        raise ValueError("Preserve the supplied failed v1 calibration report")
    if validated.get("schema") != "gsdeploy.calibrated_replay.v2" or validated.get("status") != "VALIDATED_ON_HELDOUT_CONDITIONS":
        raise ValueError("A validated v2 report is required")
    search, config, search_receipts = load_search(search_dir, validated)
    additional, additional_receipts = additional_search_evidence(additional_search_evidence_dirs, validated, search, config)
    efficiency, efficiency_receipts = efficiency_summary(efficiency_report, validated)
    confirmation, confirmation_receipts = confirmation_summary(confirmation_report, search, config)
    result = {"schema": "gsdeploy.static_study_summary.v2", "evidence_kind": "CPU_FIXTURE_NOT_MEASURED" if cpu_fixture else "RECEIPT_BOUND_STUDY_SUMMARY",
        "v1_failure": failed, "v2_validation": validated, "static_search": search, "efficiency": efficiency,
        "additional_search_evidence": additional,
        "confirmation": confirmation, "physical_global_optimality_certified": False, "gpu_started": False, "scope": SCOPE,
        "receipts": {"calibration_report": receipt(calibration_report), "validated_report": receipt(validated_report),
            "search_files": search_receipts, "additional_search_files": additional_receipts,
            "efficiency_files": efficiency_receipts, "confirmation_files": confirmation_receipts,
            "script": receipt(__file__)}}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    text = render_chinese(result)
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "report.zh.md").write_text(text)
    (output / "manifest.json").write_text(json.dumps({"files": {n: sha(output / n) for n in ("report.json", "report.zh.md")},
        "evidence_kind": result["evidence_kind"], "gpu_started": False}, indent=2) + "\n")
    return result


def render_chinese(result):
    search, efficiency, confirmation = (result[k] for k in ("static_search", "efficiency", "confirmation"))
    efficiency_domain = {k: efficiency["domain"].get(k) for k in ("grid", "grid_contract", "all_route_deployment_identities",
        "memory_lower_pass_identity_upper_bound", "scope")}
    def fmt(value):
        return "未提供/未完成" if value is None else (f"{value:.6g}" if type(value) is float else str(value))
    lines = ["# 静态任务负载研究汇总", "", result["scope"], "",
        "证据类型：" + result["evidence_kind"] + "。本入口仅核对并汇总已有文件，没有启动 GPU、拟合或搜索。", "",
        "## 校准历史", "", "v1：**VALIDATION_FAILED**，失败结果完整保存在 JSON，不能从最终记录中删去。",
        "v1 留出效用绝对误差：" + ", ".join(fmt(c.get("errors", {}).get("utility_absolute_error")) for c in result["v1_failure"].get("validation", {}).get("conditions", [])) + "。",
        "v1 排名检查：`" + json.dumps(result["v1_failure"].get("validation", {}).get("ranking"), ensure_ascii=False) + "`。",
        "v2：" + result["v2_validation"]["status"] + "；版本 " + result["v2_validation"]["version_sha256"] + "。",
        "v2 新留出效用绝对误差：" + ", ".join(fmt(c.get("errors", {}).get("utility_absolute_error")) for c in result["v2_validation"].get("validation", {}).get("conditions", [])) + "。", "",
        "## 冻结研究条件", "", f"模型池 {len(search['models'])} 个；K={search['domain']['route_grid_K']}；H={search['horizon_seconds']} 秒；seed={search['seed']}；到达过程 {search['arrival_process']}。",
        "权重：`" + json.dumps(search["weights"]) + "`。总效用 U 为六项加权成本之和的负数，越大越好。质量使用 Q 项损失计入。",
        "静态任务到达率：`" + json.dumps(search["workload"]) + "`；GPU 显存预算 " + fmt(search["gpu_memory_mib"]) + " MiB。",
        "路由域为分母 1..K 的并集，允许模型缺席；正副本域：`" + json.dumps(search["domain"]["effective_replica_options"]) + "`。", "",
        "| 模型 | 角色 | 冻结名称 |", "|---|---|---|"]
    lines += [f"| {m['id']} | {m['role']} | {m['name']} |" for m in search["models"]]
    lines += ["", "纯通控制池：" + ", ".join(search["pure_controls"]["generalist"]) + "；纯专控制池：" + ", ".join(search["pure_controls"]["specialist"]) + "。", "",
        "## 六角色搜索", "", "下表 U 是预测值。证书只适用于各自冻结预测器和有限搜索域；预算耗尽时选中方案仅是当前 incumbent。", "",
        "| 角色 | 状态 | 预测 U | 证书 | 上界 | gap | unknown | 完整评价次数 | 墙钟秒 |", "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for role in search["roles"]:
        title = role["display_label"]
        winner = role["winner"]
        lines.append("| " + " | ".join(map(fmt, (title, role["status"], winner["U"] if winner else None,
            role["optimality_certified"], role["utility_upper_bound"], role["optimality_gap"], role["unknown_candidates"],
            role["full_prediction_calls"], role["external_wall_seconds"]))) + " |")
    for role in search["roles"]:
        title = role["display_label"]
        winner = role["winner"]
        lines += ["", "### " + title, "", "部署：`" + json.dumps(winner["deployment"], ensure_ascii=False) + "`。" if winner else "无可行选中方案；保留搜索状态与 unknown，不能填写为零效用。"]
        if winner:
            lines += ["路由比例：`" + json.dumps(winner["routing_share"], ensure_ascii=False) + "`。",
                "六项加权成本：`" + json.dumps(winner["weighted_costs"]) + "`。"]
        lines.append("证书范围：" + fmt(role["certificate_scope"]) + "；未解决候选：`" + json.dumps(role["unresolved_candidates"]) + "`。")
    lines += ["", "## 补充搜索证据与纯部署限制", "", "主搜索与正式确认角色、方案和名单保持冻结；以下额外证据独立保存，不替换正式确认候选。"]
    additional = result["additional_search_evidence"]
    for item in additional["nested_searches"]:
        lines += ["", "嵌套 K=" + fmt(item["search"]["domain"]["route_grid_K"]) + " 来源：`" + item["source_directory"] + "`；bundle SHA=" + item["source_bundle"]["sha256"] + "。",
            "与主搜索差异：`" + json.dumps(item["contract_differences"], ensure_ascii=False) + "`。"]
    for pure in additional["pure_candidate_assessments"]:
        main = pure["frozen_main_candidate"]
        best = pure["best_reported_predictor_candidate"]
        title = "纯通" if pure["label"] == "best_generalist_only" else "纯专"
        lines += ["", title + "：主域冻结候选预测 U=" + fmt(main["U"] if main else None)
            + "；主域上界=" + fmt(pure["main_pure_domain_upper_bound"])
            + "；子域最优证书=" + fmt(pure["main_pure_domain_optimality_certified"]) + "。"]
        if best:
            lines.append("已报告最强" + title + "预测候选来自 K=" + fmt(best["source_K"]) + "，U=" + fmt(best["winner"]["U"])
                + "，来源=" + best["source"] + "；额外候选强于主域冻结候选=" + fmt(pure["additional_candidate_stronger_than_frozen_main"]) + "。")
        if pure["excluded_from_full_pool_optimum_by_main_upper_bound"]:
            lines.append(title + "主域上界=" + fmt(pure["main_pure_domain_upper_bound"]) + " < 全池可行方案目标值="
                + fmt(pure["full_pool_incumbent_objective_score"]) + "，因此已排除该纯域赢得全池预测最优；不能据此声称纯域当前候选本身最优。")
        else:
            lines.append("当前上界未排除该纯域赢得全池预测最优。")
        lines.append("纯域内是否最优以主域证书为准；嵌套小网格上界不用于主域上界，候选改进不改变 formal packet。")
    lines += ["", "## 三算法同域效率比较", "", "状态：" + efficiency["status"] + "；同域合同核对：" + fmt(efficiency["same_contract"]) + "。",
        "对比副本域：`" + json.dumps(efficiency["replica_options"]) + "`；路由域：`" + json.dumps(efficiency_domain, ensure_ascii=False) + "`。",
        "此对比域与七模型完整搜索域分别报告，不能互相替代证书。", "",
        "| 算法 | 完成进程 | 证书数 | 搜索墙钟中位秒 | 完整评价中位数 | 峰值 RSS KiB |", "|---|---:|---:|---:|---:|---:|"]
    lines += ["| " + " | ".join(fmt(s[k]) for k in ("algorithm", "completed_processes", "certificates", "median_search_wall_seconds", "median_full_predictions", "max_peak_rss_kib")) + " |" for s in efficiency["summaries"]]
    lines += ["", "逐轮同优/证书/方案身份：`" + json.dumps(efficiency["per_repeat"], ensure_ascii=False) + "`。",
        "观测墙钟比（竞争算法/MILP）：`" + json.dumps(efficiency["observed_wall_ratio_rival_over_milp"]) + "`。",
        "同优且全有证书的求解速度比：`" + json.dumps(efficiency["certified_solve_speedup_rival_over_milp"]) + "`。",
        "是否证明本对比中 MILP 比两种 DFS 均更快：" + fmt(efficiency["milp_faster_than_both_at_certified_same_objective"]) + "。",
        efficiency["interpretation"], "计时口径：" + fmt(efficiency["timing_scope"]) + "。", "",
        "## 正式独立确认", "", "状态：**" + confirmation["status"] + "**。"]
    if confirmation["status"] == "NOT_DONE":
        lines += [confirmation["reason"], "正式 GPU 效用、配对差、区间和 p 值均未完成，以 null 保存；不能用 0 或预测值代替。"]
    else:
        lines += [f"有效作业 {confirmation['completed_jobs']}/{confirmation['planned_jobs']}；缺失作业：`{json.dumps(confirmation['missing_job_ids'])}`。", confirmation["scope"], "",
            "| 对比方案 key | 已配对种子 | 平均 U 差（冻结选中减竞争） | Student 区间 | 精确符号翻转 p |", "|---|---|---:|---|---:|"]
        for pair in confirmation["paired"].get("comparisons", []):
            lines.append("| " + " | ".join((pair["rival_key"], str(pair["paired_seeds"]), fmt(pair.get("mean_U_difference")), str(pair.get("student_interval")), fmt(pair.get("exact_two_sided_sign_flip_p")))) + " |")
    lines += ["", "所有输入和逐文件 SHA receipt 保存在 report.json；输出文件 SHA 保存在 manifest.json。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--validated-report", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--efficiency-report", type=Path, required=True)
    parser.add_argument("--confirmation-report", type=Path)
    parser.add_argument("--additional-search-evidence", type=Path, nargs="+", help="Complete nested-grid six-role directories; retain evidence without changing frozen confirmation plans")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-fixture", action="store_true", help="Mark synthetic verification output CPU_FIXTURE_NOT_MEASURED")
    args = parser.parse_args()
    result = assemble(args.calibration_report, args.validated_report, args.search_dir, args.efficiency_report,
        args.output, confirmation_report=args.confirmation_report,
        additional_search_evidence_dirs=args.additional_search_evidence, cpu_fixture=args.cpu_fixture)
    print(json.dumps({"confirmation_status": result["confirmation"]["status"], "efficiency_status": result["efficiency"]["status"], "gpu_started": False}))
