"""Build auditable historical comparison maps; never infer control from outcomes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from unified_weight_inventory_20261023 import digest, receipt


def is_text_specialist(row, family):
    model = (row.get("pool_models") or {}).get(family) or {}
    return model.get("role") == "specialist" and "text" in (model.get("supported_tasks") or [])


def text_role(row):
    deployed = row.get("deployment") or {}
    specialist = any(count > 0 and is_text_specialist(row, family) for family, count in deployed.items())
    generalist = any(count > 0 and ((row.get("pool_models") or {}).get(family) or {}).get("role") == "generalist"
                     for family, count in deployed.items())
    allowed = row.get("declared_text_spill_allowed")
    if specialist and allowed is False:
        return "declared_no_text_spill"
    if specialist and generalist and allowed is True:
        return "declared_text_spill_with_specialist"
    return "other_or_unknown"


def compact(row):
    return {k: row.get(k) for k in (
        "label", "raw", "stage", "seed", "role", "strategy", "deployment", "planned_routing_rps",
        "declared_text_spill_allowed", "spill_declaration_source", "text_generalist_spills",
        "rescorable_numeric", "original_U", "excluded_reason", "packet_job_label")}


def differing_paths(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            path = prefix + ("." if prefix else "") + key
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(differing_paths(left[key], right[key], path))
        return result
    return [] if left == right else [prefix]


def full_contract(row, cache):
    path = row["raw"]["path"]
    if path not in cache:
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["raw"]["sha256"]:
            raise ValueError("Historical raw changed: " + path)
        obj = json.loads(raw)
        cache[path] = {"contract": obj.get("policy_contract") or {},
                       "config_signature": digest(obj.get("config") or {}),
                       "transport": {k: v for k, v in (obj.get("transport_config") or {}).items()
                                     if k != "routing_rps"}}
    return cache[path]


def pair_metadata(spill, control, cache):
    same_deployment = spill["deployment"] == control["deployment"]
    a, b = full_contract(spill, cache), full_contract(control, cache)
    diffs = differing_paths(a["contract"], b["contract"])
    policy_route_toggle_only = bool(a["contract"] and b["contract"]) and set(diffs).issubset(
        {"strategy", "coupled.allow_spill", "refinement.allow_spill", "probe.probability"})
    # Same role/strategy or same deployment alone does not certify a route ablation.
    if not same_deployment:
        kind = "different_deployment_structural_control"
    elif not a["contract"] and not b["contract"]:
        sa, cb = copy.deepcopy(spill.get("planned_routing_rps") or {}), copy.deepcopy(control.get("planned_routing_rps") or {})
        sa.pop("text", None)
        cb.pop("text", None)
        kind = "fixed_text_route_only_metadata_match" if sa == cb else "same_deployment_background_route_changed"
    elif policy_route_toggle_only:
        kind = "policy_allowed_text_route_toggle_metadata_match"
    else:
        kind = "same_deployment_different_policy_control"
    return {
        "spill": spill["label"], "control": control["label"],
        "comparison_kind": kind, "same_deployment": same_deployment,
        "full_policy_contract_differing_paths": diffs,
        "same_raw_config": a["config_signature"] == b["config_signature"],
        "same_transport_except_routing": a["transport"] == b["transport"],
        "both_numeric_rescorable": spill["rescorable_numeric"] and control["rescorable_numeric"],
        "spill_actual_text_to_generalist": spill["text_generalist_spills"],
        "control_actual_text_to_generalist": control["text_generalist_spills"],
        "causal_improvement_claim": False,
        "scope_note": "Metadata match only; measured outcomes, interference and noise remain empirical. "
                      "Route-toggle label requires full policy-contract differences confined to declared route toggles.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("artifacts/unified-weight-study-20261023/inventory-v3/inventory.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/unified-weight-study-20261023/historical-pair-map-v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite " + str(args.output))
    inventory = json.loads(args.inventory.read_text())
    grouped, ungroupable = defaultdict(list), []
    for row in inventory["rows"]:
        if not row.get("cohort_identity_complete") or not row.get("pool_signature"):
            ungroupable.append({"row": compact(row), "reason": "INCOMPLETE_COHORT_OR_POOL_IDENTITY"})
            continue
        key = digest({"study": row["study"], "conditions": row["comparison_group_signature"]})
        grouped[key].append(row)
    groups, cache = [], {}
    for key, rows in grouped.items():
        controls = [r for r in rows if text_role(r) == "declared_no_text_spill"]
        spills = [r for r in rows if text_role(r) == "declared_text_spill_with_specialist"]
        pairs = [pair_metadata(spill, control, cache) for spill in spills for control in controls]
        if any(r["text_generalist_spills"] != 0 for r in controls):
            raise ValueError("A declared no-spill control has nonzero measured spills")
        sample = rows[0]
        groups.append({
            "group_id": key, "study": sample["study"], "seeds": sorted({r["seed"] for r in rows}),
            "workload": sample["workload"], "horizon_seconds": sample["horizon_seconds"],
            "metric_definition_signature": sample["metric_definition_signature"],
            "normalization": sample["normalization"], "pool_signature": sample["pool_signature"],
            "cohort_signature": sample["cohort_signature"],
            "candidate_count": len(rows), "rows": [compact(r) for r in rows],
            "declared_no_spill": [r["label"] for r in controls],
            "declared_spill_with_text_specialist": [r["label"] for r in spills],
            "other_candidates": [r["label"] for r in rows if text_role(r) == "other_or_unknown"],
            "pairs": pairs,
            "status": "PAIRED" if pairs else ("SPILL_WITHOUT_DECLARED_MATCHED_CONTROL" if spills else "NO_TEXT_SPILL_PAIR"),
        })
    summary = {
        "raw_rows": len(inventory["rows"]), "groups": len(groups), "ungroupable_rows": len(ungroupable),
        "groups_with_pairs": sum(bool(g["pairs"]) for g in groups),
        "pairs": sum(len(g["pairs"]) for g in groups),
        "numeric_pairs": sum(p["both_numeric_rescorable"] for g in groups for p in g["pairs"]),
        "pair_kinds": dict(Counter(p["comparison_kind"] for g in groups for p in g["pairs"])),
        "paired_groups_by_study": dict(Counter(g["study"] for g in groups if g["pairs"])),
    }
    output = {"schema": "gsdeploy.unified_weight_historical_pair_map.v1", "inventory": receipt(args.inventory),
              "tool": receipt(Path(__file__)), "complete": True, "summary": summary,
              "rules": ["Within-study exact workload, horizon, metric, pool and request cohort identities.",
                        "A control requires predeclared no-spill; observed zero spills does not establish a control.",
                        "A spill arm requires a deployed text specialist and a deployed generalist.",
                        "Different deployment is a structural control, never a route-only ablation.",
                        "Calibration is calibration; failures remain unknown; no new utility or GPU measurement."],
              "groups": groups, "ungroupable": ungroupable}
    args.output.mkdir(parents=True)
    (args.output / "pair-map.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 历史实验可比候选与分流对照映射", "",
             f"收录 {summary['raw_rows']} 条原始记录，得到 {summary['groups']} 个兼容条件组；"
             f"其中 {summary['groups_with_pairs']} 组存在明确预声明的文本分流/不分流对照，"
             f"形成 {summary['pairs']} 个比较（{summary['numeric_pairs']} 个双方均可重评分）。", "",
             "分组严格匹配同一实验系列、负载、时长、归一化与质量身份、模型池、请求样本及计划到达序列。"
             "实际分流数为零只描述结果，不用它定义对照。存在文本专用与通用驻留、且预声明允许文本走通用，才列为文本分流臂。", "",
             "同驻留、同样本仍不自动构成严格分流消融。JSON逐对列出完整原始policy_contract的差异字段；"
             "只有差异限于策略路由开关时才标元数据匹配。不同驻留均标结构对照。此表没有宣称正收益或因果结论。", "",
             "| 实验系列/seed | text rps | 同条件候选数 | 明确不分流臂 | 明确分流臂（实际文本通用处理量） | 比较数 |",
             "|---|---:|---:|---|---|---:|"]
    for group in groups:
        if not group["pairs"]:
            continue
        short = lambda label: label.split("/")[-2]
        controls = "; ".join(short(v) for v in group["declared_no_spill"])
        spills = "; ".join(short(v) + f" ({next(r['text_generalist_spills'] for r in group['rows'] if r['label']==v)})"
                           for v in group["declared_spill_with_text_specialist"])
        lines.append(f"| {group['study']} / {','.join(map(str,group['seeds']))} | {group['workload']['text']} | "
                     f"{group['candidate_count']} | {controls} | {spills} | {len(group['pairs'])} |")
    lines += ["", "主要阶段的读法：", "",
              "- refined 205–206 是独立验证，207–208 是原固定选择的最终确认；各seed的 calibrated_spill 对 risk_matched_no_spill 单独配对。209分流臂能源缺失且配对对照未运行，不能补成失败或零收益。",
              "- queue-aware 113–114 的开发组保留固定1/32、queue_latency、queue_quality、动态专用平衡、不分流SWRR以及额外图像专用驻留等候选；71–73最终确认和81–82独立验证对比队列分流与动态专用平衡。",
              "- static text185的107–108组同时含纯专结构控制、1个文本专用+较高分流、2个文本专用+不分流及1/32分流；109–112是搜索前两名的同驻留路由比较。旧raw将这些声明为calibration，继续保留该标签。",
              "- static中ASR为主的实验不属于文本专→通分流；文本本来就由通用处理，不能把其text通用处理数当文本分流收益。完整候选和固定路由仍保留于JSON。",
              "- coupled 303–304的同条件候选同时包括coupled分流/禁分流、legacy与其他驻留；跨策略的零分流结果不得冒充同策略禁分流消融。", "",
              "所有组（含单候选、无明确对照、六项缺失）及原始SHA均列在pair-map.json；本步骤没有重跑GPU或进行权重求解。"]
    (args.output / "README.zh.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
