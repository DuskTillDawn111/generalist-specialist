"""Read-only inventory of historical run receipts for external utility rescoring.

This deliberately imports no experiment runner and never rewrites old evidence.
Missing terms are unknown, not zero. Eligibility is copied, never recomputed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


TERMS = ("Q", "T", "L", "P", "M", "E")
TASKS = ("text", "image", "ocr", "asr")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def receipt(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            sha.update(chunk)
    return {"path": str(path.resolve()), "sha256": sha.hexdigest()}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def scalar_fields(value):
    return {k: v for k, v in (value or {}).items() if not isinstance(v, (dict, list))}


def stage_for(path, data):
    identity = data.get("executed_measurement_identity") or {}
    if not isinstance(identity, dict):
        identity = {}
    explicit = data.get("experiment_phase") or identity.get("phase")
    # Calibration status takes precedence over an embedded generic phase.
    parts = path.parts
    if any("calibration" in p for p in parts):
        return "calibration"
    if any("mock" in p or "acceptance" in p or "cli-check" in p for p in parts):
        return "acceptance_or_mock"
    if any("diagnostic" in p for p in parts):
        return "diagnostic"
    if explicit:
        return explicit
    for value in ("final_confirmation", "confirmation", "development", "screen"):
        if any(value in p for p in parts):
            return value
    if any("holdout" in p for p in parts):
        return "holdout_validation"
    return "unknown"


def referenced_receipts(data, cache):
    candidates = []
    if isinstance(data.get("source_packet"), dict):
        candidates.append(("source_packet", data["source_packet"]))
    identity = data.get("executed_measurement_identity") or {}
    if isinstance(identity, dict):
        for key, value in identity.items():
            if isinstance(value, dict) and isinstance(value.get("path"), str):
                candidates.append(("executed_measurement_identity." + key, value))
    result = []
    supplements = []
    seen = set()
    for source_field, ref in candidates:
        p = Path(ref["path"])
        if str(p) in seen:
            continue
        seen.add(str(p))
        if str(p) not in cache:
            if p.is_file():
                actual = receipt(p)
                try:
                    obj = json.loads(p.read_text()) if p.suffix == ".json" else None
                except (ValueError, UnicodeError):
                    obj = None
                cache[str(p)] = (actual, obj)
            else:
                cache[str(p)] = (None, None)
        actual, obj = cache[str(p)]
        result.append({"field": source_field, "declared": ref,
                       "actual": actual,
                       "sha256_matches": actual["sha256"] == ref.get("sha256") if actual and ref.get("sha256") else None})
        if isinstance(obj, dict):
            supplements.append((source_field, obj))
    return result, supplements


def packet_job_index(root):
    """Find matching job identities without trusting directory names as metadata."""
    index = defaultdict(list)
    for path in sorted(root.rglob("*packet*.json")):
        if "unified-weight-study" in str(path):
            continue
        try:
            obj = json.loads(path.read_bytes())
        except (ValueError, UnicodeError):
            continue
        if not isinstance(obj, dict):
            continue
        jobs = obj.get("jobs") or []
        if not isinstance(jobs, list):
            continue
        packet_receipt = None
        for job in jobs:
            if not isinstance(job, dict):
                continue
            job_id = job.get("id") or job.get("job_id")
            if not isinstance(job_id, str):
                continue
            if packet_receipt is None:
                packet_receipt = receipt(path)
            compact = {k: job[k] for k in ("id", "job_id", "labels", "label", "phase", "role",
                                           "purpose", "state", "seed", "job_identity_sha256") if k in job}
            index[job_id].append({"packet": packet_receipt, "matching_job": compact,
                                  "match_rule": "exact_raw_experiment_job_id_or_job.job_id"})
    return index


def inventory_row(root, path, cache, job_index):
    raw_bytes = path.read_bytes()
    raw = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw_bytes).hexdigest()}
    data = json.loads(raw_bytes)
    del raw_bytes
    label = str(path.relative_to(root))
    utility = (data.get("evaluation") or {}).get("utility") or {}
    terms = {k: utility.get(k + "_norm") for k in TERMS}
    evaluation = data.get("evaluation") or {}
    config = data.get("config") or {}
    transport = data.get("transport_config") or {}
    catalog = config.get("catalog") or {}
    plan = data.get("plan") or {}
    contract = data.get("policy_contract") or {}
    identity = data.get("executed_measurement_identity") or {}
    identity = identity if isinstance(identity, dict) else {}
    raw_job = data.get("job") or {}
    raw_job = raw_job if isinstance(raw_job, dict) else {}
    job_id = data.get("experiment_job_id") or raw_job.get("job_id")
    job_matches = job_index.get(job_id, [])
    source_receipts, supplements = referenced_receipts(data, cache)
    supplement_fields = {}
    # Supplement only when an explicit referenced packet contains the exact field.
    for source_field, obj in supplements:
        candidate_config = obj.get("config") or obj.get("base_config") or {}
        if not catalog and isinstance(candidate_config, dict) and candidate_config.get("catalog"):
            catalog = candidate_config["catalog"]
            supplement_fields["catalog"] = source_field + ".config.catalog"
    models = catalog.get("models") or {}
    pool_models = {k: {"name": v.get("name"), "role": v.get("role"),
                       "supported_tasks": v.get("supported_tasks")}
                   for k, v in models.items() if isinstance(v, dict)}
    quality = catalog.get("quality") or {}
    quality_evidence = utility.get("quality_evidence") or {}
    quality_identity = quality_evidence.get("identity") or quality.get("identity")
    energy = data.get("energy_evidence") or {}
    workload = config.get("workload") or transport.get("workload_rps") or identity.get("workload")
    workload = workload if isinstance(workload, dict) else {}
    workload = {**{k: workload.get(k) for k in TASKS}, **workload}
    deployment = plan.get("deployment") or contract.get("deployment") or identity.get("deployment")
    window = data.get("measurement_window") or {}
    horizon = window.get("duration_seconds") or transport.get("measurement_seconds") or config.get("horizon_seconds")
    seed = transport.get("random_seed", identity.get("seed", config.get("random_seed")))
    strategy = data.get("strategy") or identity.get("strategy") or contract.get("strategy")
    role = identity.get("role") or identity.get("selection_role") or raw_job.get("role")
    policy_summary = {k: contract.get(k) for k in
                      ("schema", "strategy", "mode", "guard", "refinement", "coupled", "probe", "background_routing")
                      if k in contract}
    declared_allow_spill = None
    spill_declaration_source = None
    if isinstance(contract.get("probe"), dict) and finite(contract["probe"].get("probability")):
        declared_allow_spill = contract["probe"]["probability"] > 0
        spill_declaration_source = "policy_contract.probe.probability_override"
    for key in ("coupled", "refinement"):
        if declared_allow_spill is None and isinstance(contract.get(key), dict) and isinstance(contract[key].get("allow_spill"), bool):
            declared_allow_spill = contract[key]["allow_spill"]
            spill_declaration_source = "policy_contract." + key + ".allow_spill"
            break
    if declared_allow_spill is None and strategy in ("no_split_swrr", "specialist_only", "queue_specialist_only"):
        declared_allow_spill = False
        spill_declaration_source = "policy_contract.strategy:" + strategy
    if declared_allow_spill is None and strategy == "fixed_1of32_swrr":
        declared_allow_spill = True
        spill_declaration_source = "policy_contract.strategy:fixed_1of32_swrr"
    if declared_allow_spill is None and strategy in ("queue_latency", "queue_quality"):
        guard = contract.get("guard") or {}
        if guard.get("task") == "text" and (guard.get("family") or "").startswith("generalist"):
            declared_allow_spill = (guard.get("rate_rps") or 0) > 0
            spill_declaration_source = "policy_contract.strategy_and_generalist_text_guard"
    if declared_allow_spill is None and not contract:
        text_route = (plan.get("routing") or {}).get("text")
        if isinstance(text_route, dict):
            declared_allow_spill = any(rate > 0 and ((models.get(family) or {}).get("role") == "generalist"
                                          or family.startswith("generalist"))
                                      for family, rate in text_route.items() if finite(rate))
            spill_declaration_source = "plan.routing.text_fixed_rates"
    stage = stage_for(path, data)
    records = data.get("request_records") or []
    alias_map = data.get("replica_to_family") or {}
    actual_counts = defaultdict(Counter)
    actual_completed_counts = defaultdict(Counter)
    task_counts = Counter()
    statuses = Counter()
    missing_identity = Counter()
    cohort = []
    spill_count = 0
    unknown_families = 0
    for record in records:
        task = record.get("task")
        alias = record.get("model")
        family = alias_map.get(alias)
        if family is None and alias in (contract.get("aliases") or {}):
            family = contract["aliases"][alias].get("family")
        if family is None and isinstance(alias, str):
            candidate = alias.split("#", 1)[0]
            if candidate in models:
                family = candidate
        family = family or "UNKNOWN"
        unknown_families += family == "UNKNOWN"
        actual_counts[task or "UNKNOWN"][family] += 1
        if record.get("status") == "completed":
            actual_completed_counts[task or "UNKNOWN"][family] += 1
        is_generalist = (models.get(family) or {}).get("role") == "generalist" or family.startswith("generalist")
        spill_count += task == "text" and is_generalist
        task_counts[task or "UNKNOWN"] += 1
        statuses[record.get("status") or "UNKNOWN"] += 1
        fields = ("request_id", "task", "pool_id", "sample_id", "scheduled_offset_seconds")
        cohort.append([record.get(k) for k in fields])
        for field in fields:
            if record.get(field) is None:
                missing_identity[field] += 1
    # Independent of routing and run-specific absolute timestamps.
    cohort.sort(key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False))
    cohort_signature = digest(cohort) if cohort else None
    normalized = {
        "utility_schema": utility.get("utility_schema") or config.get("utility_schema"),
        "term_directions": {k: "loss_or_cost_smaller_is_better" for k in TERMS},
        "quality_mode": quality.get("mode"), "quality_identity": quality_identity,
        "latency_reference_ms": config.get("latency_reference_ms"),
        "gpu_memory_mib": config.get("gpu_memory_mib"),
        "energy_reference_watts": config.get("energy_reference_watts"),
        "startup_amortization_seconds": config.get("startup_amortization_seconds"),
        "energy_scope": energy.get("scope"), "energy_scope_note": energy.get("scope_note"),
        "energy_kind": energy.get("kind"),
        "quality_task_weights": quality_evidence.get("task_weights"),
        "memory_scope": utility.get("memory_scope") or catalog.get("memory_scope"),
    }
    complete_terms = all(finite(v) for v in terms.values())
    declared_complete = utility.get("utility_complete")
    utility_complete = complete_terms and declared_complete is not False
    backend = data.get("backend")
    excluded = []
    if backend != "native":
        excluded.append("BACKEND_NOT_DECLARED" if backend is None else "NON_NATIVE_BACKEND:" + str(backend))
    if data.get("status") != "completed":
        excluded.append("RUN_NOT_COMPLETED:" + str(data.get("status")))
    if utility.get("eligible") is not True:
        excluded.append("ORIGINAL_ELIGIBLE_NOT_TRUE")
    if not utility_complete:
        excluded.append("INCOMPLETE_SIX_TERMS:" + ",".join(k for k, v in terms.items() if not finite(v)))
    if energy.get("kind") != "measured" or energy.get("coverage_valid") is not True:
        excluded.append("ENERGY_NOT_VERIFIED_MEASUREMENT")
    if stage in ("calibration", "acceptance_or_mock", "diagnostic", "unknown"):
        excluded.append("NOT_FORMAL_EFFICACY_PHASE:" + stage)
    row = {
        "raw": raw, "label": label, "study": path.relative_to(root).parts[0],
        "stage": stage, "phase": data.get("experiment_phase"),
        "path_stage": str(path.parent.relative_to(root)),
        "seed": seed, "status": data.get("status"), "backend": backend,
        "eligible": utility.get("eligible"), "utility_complete": utility_complete,
        "declared_utility_complete": declared_complete,
        "original_U": utility.get("U"), "original_weights": config.get("weights"),
        "terms": terms, "term_source": "evaluation.utility.{Q,T,L,P,M,E}_norm",
        "rescorable_numeric": utility.get("eligible") is True and utility_complete,
        "formal_efficacy_eligible": not excluded,
        "excluded_reason": excluded or None,
        "original_utility_reason": utility.get("reason"),
        "deployment": deployment, "planned_routing_rps": plan.get("routing"),
        "workload": workload, "horizon_seconds": horizon,
        "quality_identity": quality_identity,
        "normalization": normalized, "metric_definition_signature": digest(normalized),
        "energy": {k: energy.get(k) for k in ("kind", "scope", "coverage_valid", "run_joules", "startup_joules", "scope_note")},
        "strategy": strategy, "role": role,
        "packet_path": ((data.get("source_packet") or identity.get("base_packet") or {}).get("path")
                        or (job_matches[0]["packet"]["path"] if len(job_matches) == 1 else None)),
        "packet_job_label": [m["matching_job"].get("labels", m["matching_job"].get("label"))
                             for m in job_matches] or None,
        "packet_job_matches": job_matches,
        "raw_job_scalar_metadata": scalar_fields(raw_job),
        "raw_schema": data.get("schema_version") or data.get("schema"),
        "raw_protocol": data.get("protocol"),
        "declared_text_spill_allowed": declared_allow_spill,
        "spill_declaration_source": spill_declaration_source,
        "policy_control_summary": policy_summary or {"fixed_routing_rps": plan.get("routing")},
        "policy_identity": data.get("policy_identity"),
        "policy_contract_receipt": {"sha256": digest(contract), "embedded_in_raw": True} if contract else None,
        "policy_contract_scalar_fields": scalar_fields(contract),
        "pool_models": pool_models or None,
        "pool_signature": digest(pool_models) if pool_models else None,
        "actual_routing_task_family_counts": dict(actual_counts),
        "actual_completed_routing_task_family_counts": dict(actual_completed_counts),
        "text_generalist_spills": spill_count if not unknown_families else None,
        "cohort_signature": cohort_signature,
        "cohort_signature_fields": ["request_id", "task", "pool_id", "sample_id", "scheduled_offset_seconds"],
        "cohort_missing_identity_fields": dict(missing_identity),
        "cohort_identity_complete": bool(records) and not missing_identity,
        "counts": {"request_records": len(records), "task": dict(task_counts),
                   "status": dict(statuses), "unknown_routing_family": unknown_families,
                   "warmup_records": len(data.get("warmup_request_records") or []),
                   "priming_records": len(data.get("priming_request_records") or [])},
        "system_metrics": data.get("system_metrics"),
        "stability": scalar_fields(evaluation.get("stability")),
        "evaluation_scalar_metadata": scalar_fields(utility),
        "metadata_source_receipts": ([{"field": "raw", "actual": raw}] + source_receipts
                                    + [{"field": "matched_packet_job", "actual": m["packet"],
                                        "matching_job": m["matching_job"], "match_rule": m["match_rule"]}
                                       for m in job_matches]),
        "supplemented_fields": supplement_fields,
        "experiment_job_id": job_id,
        "measurement_identity": data.get("measurement_identity"),
        "executed_measurement_identity": identity or None,
    }
    row["comparison_group_signature"] = digest({
        "metric_definition": row["metric_definition_signature"],
        "pool": row["pool_signature"], "workload": workload, "horizon": horizon,
        "cohort": cohort_signature,
    })
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/unified-weight-study-20261023/inventory-v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing inventory directory: " + str(args.output))
    paths = sorted(p for p in args.root.rglob("run.json")
                   if "unified-weight-study" not in str(p))
    rows, cache = [], {}
    job_index = packet_job_index(args.root)
    for index, path in enumerate(paths, 1):
        rows.append(inventory_row(args.root, path, cache, job_index))
        print(f"{index}/{len(paths)} {path}", flush=True)
    assert len(rows) == len(paths)
    assert len({r["raw"]["path"] for r in rows}) == len(paths)
    counts = Counter(r["study"] for r in rows)
    numeric = Counter(r["study"] for r in rows if r["rescorable_numeric"])
    formal = Counter(r["study"] for r in rows if r["formal_efficacy_eligible"])
    summary = {
        "run_count": len(rows), "studies": dict(counts),
        "numeric_rescorable_count": sum(numeric.values()),
        "numeric_rescorable_by_study": dict(numeric),
        "formal_efficacy_count": sum(formal.values()),
        "formal_efficacy_by_study": dict(formal),
        "stages": dict(Counter(r["stage"] for r in rows)),
        "exclusion_reasons": dict(Counter(e for r in rows for e in r["excluded_reason"] or [])),
        "metric_definition_count": len({r["metric_definition_signature"] for r in rows}),
        "missing_model_pool_count": sum(r["pool_models"] is None for r in rows),
        "cohort_incomplete_count": sum(not r["cohort_identity_complete"] for r in rows),
    }
    result = {
        "schema": "gsdeploy.unified_weight_raw_inventory.v1", "complete": True,
        "root": str(args.root.resolve()), "tool": receipt(Path(__file__)),
        "scope": "All historical artifacts/**/run.json, excluding new unified-weight-study outputs.",
        "rules": ["Copy original eligibility; never infer infeasibility from a failed or missing run.",
                  "Q is quality loss; all six terms are costs and U=-sum(w[k]*terms[k]).",
                  "Numeric rescoring includes complete originally eligible rows; calibration remains calibration.",
                  "Formal efficacy excludes non-native, unverified-energy and calibration runs.",
                  "Compare rankings only with matched cohort, model pool, workload, horizon and metric definitions.",
                  "Missing or invalid energy is unknown, never zero.",
                  "No historical strategy is required to become profitable under the new weights."],
        "summary": summary, "rows": rows,
    }
    args.output.mkdir(parents=True)
    (args.output / "inventory.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    fields = ["label", "study", "stage", "role", "seed", "backend", "status", "eligible",
              "utility_complete", "rescorable_numeric", "formal_efficacy_eligible", "original_U",
              *TERMS, "deployment", "workload", "horizon_seconds", "strategy", "text_generalist_spills",
              "cohort_signature", "metric_definition_signature", "pool_signature", "excluded_reason", "raw"]
    with (args.output / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            flat = {k: row.get(k) for k in fields}
            flat.update(row["terms"])
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in flat.items()})
    readme = ["# 历史实验原始收据与统一换权重清单", "",
              f"完整收录 {len(rows)} 份 run.json；可数值重评分 {sum(numeric.values())} 份，"
              f"其中符合正式实测阶段标记的 {sum(formal.values())} 份。", "",
              "本清单只读取旧证据，保留每份原始文件的 SHA256。Q 是质量损失，六项均越小越好；"
              "新总效用应由 U = −Σ wᵢ termᵢ 计算，不能直接混合旧 U。", "",
              "| 实验系列 | 全部记录 | 原 eligible 且六项完整 | 正式实测阶段 |",
              "|---|---:|---:|---:|"]
    readme += [f"| {study} | {count} | {numeric[study]} | {formal[study]} |" for study, count in counts.items()]
    readme += ["", "`inventory.json` 保存完整紧凑元数据，`runs.csv` 用于查阅。"
               "原 eligible=True 且六项完整是数值重评分的条件；校准数据仍明确标成校准，不能冒充独立确认。", "",
               "按 metric_definition_signature、pool_signature、workload、horizon 和 cohort_signature 分组比较。"
               "不同任务比例、样本队列、模型池、质量身份或归一化口径不能混合排名。"
               "unknown 保持 null；实验失败或能源缺失不代表候选不可行。", "",
               "实际路由统计来自 measurement request_records，经 replica_to_family 和策略 alias 表映射。"
               "text_generalist_spills 统计文本任务实际进入通用模型的请求数；在纯通方案中它表示文本通用处理量，"
               "不自动解释为专用分流收益。", "",
               "cohort_signature 绑定请求 ID、任务、pool_id、sample_id 和计划到达相对偏移，排除实际路由与绝对时钟。"
               "cohort_identity_complete=False 时必须另外核对缺失身份。", "",
               "元数据补充使用原始收据直接引用的包，及与 raw job ID 精确匹配的历史 packet jobs；"
               "前者列出声明 SHA、实际 SHA 及其匹配状态，后者列出 packet 收据和匹配规则。"
               "所有旧策略继续保留作比较，不要求同一权重把每个历史策略都变为正收益。", "",
               "排除标记计数（同一记录可有多个原因）：", ""]
    readme += [f"- {reason}: {count}" for reason, count in summary["exclusion_reasons"].items()]
    (args.output / "README.zh.md").write_text("\n".join(readme) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
