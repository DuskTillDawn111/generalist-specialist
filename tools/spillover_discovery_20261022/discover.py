#!/usr/bin/env python3
"""CPU-only exploratory interface preflight; never fills missing energy or U.

Uses the public unmodified task-replica proxy. If all six terms are unavailable,
full optimization is explicitly blocked rather than optimizing a partial score.
The hand-declared structural probes are not search winners.
"""

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsdeploy.calibration_v2 import validated_replay_spec
from gsdeploy.config import SystemConfig
from gsdeploy.prediction import predict

STATUS = "EXPLORATORY_NOT_VALIDATED"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_receipts():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "gsdeploy").rglob("*.py"))}


def inspect_domain(catalog, budget):
    states = {m: sorted(map(int, catalog["capacity_factors"][m])) for m in catalog["models"]}
    memory_feasible = target_feasible = 0
    names = list(states)
    for counts in itertools.product(*(states[m] for m in names)):
        if not any(counts):
            continue
        mem = catalog["safe_factor"] * sum(catalog["memory_table_mib"][m][str(n)]
                                            for m, n in zip(names, counts))
        if mem <= budget:
            memory_feasible += 1
            target_feasible += counts[names.index("asr")] >= 2 and any(
                n and catalog["models"][m]["role"] == "generalist"
                for m, n in zip(names, counts))
    capacities = {}
    receipts = {}
    for model, tasks in catalog["task_replica_profiles"].items():
        capacities[model] = {}
        for n, entry in tasks.get("asr", {}).items():
            capacities[model][n] = .8 * int(n) * 1000 / entry["mean_service_ms"]
            path = Path(entry["source"])
            receipts[str(path)] = {"exists": path.is_file(), "sha256": digest(path) if path.is_file() else None}
    return {"model_replica_states": states, "all_seven_models_retained": len(states) == 7,
            "raw_deployment_identities_including_empty": __import__("math").prod(map(len, states.values())),
            "additive_proxy_memory_feasible_nonempty_deployments": memory_feasible,
            "memory_feasible_asr_n_ge_2_and_generalist_deployments": target_feasible,
            "asr_rho_08_capacity_rps_by_measured_state": capacities,
            "max_single_family_asr_rps_relaxing_background_and_memory":
                max(v for row in capacities.values() for v in row.values()),
            "asr_profile_receipts": receipts,
            "interpretation": "Capacity/memory are standalone transfer proxies, not measured mixed-board hardware bounds."}


def probe_plans(rates):
    # Predeclared witnesses, not constraints on a free search. Include the old
    # winner structure, split candidates, and same-deployment no-split controls.
    old = {"deployment": {"generalist_2": 1, "image": 1, "ocr": 1},
           "routing": {"text": {"generalist_2": rates["text"]},
                       "image": {"image": rates["image"]},
                       "ocr": {"ocr": rates["ocr"]},
                       "asr": {"generalist_2": rates["asr"]}}}
    yield "previous_winner_structure", old
    for n, share in ((2, 2 / 3), (3, 3 / 4)):
        x = {"generalist_2": 1, "ocr": 1, "asr": n}
        background = {"text": {"generalist_2": rates["text"]},
                      "image": {"generalist_2": rates["image"]},
                      "ocr": {"ocr": rates["ocr"]}}
        yield f"declared_asr_n{n}_specialist_share_{share:.6f}", {
            "deployment": x, "routing": {**background, "asr": {
                "asr": rates["asr"] * share, "generalist_2": rates["asr"] * (1 - share)}}}
        yield f"same_deployment_asr_n{n}_no_split", {
            "deployment": x, "routing": {**background, "asr": {"asr": rates["asr"]}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "artifacts/static-study-gpu-20261022/search-v2-v5-K8/system-config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/static-spillover-study-20261022/cpu-discovery")
    parser.add_argument("--asr-rates", nargs="+", type=float, default=[2.5, 2.6, 2.7])
    parser.add_argument("--background-rate", type=float, default=.05)
    parser.add_argument("--point-seconds", type=float, default=120.)
    args = parser.parse_args()
    if not 0 < args.point_seconds <= 120 or min(args.asr_rates + [args.background_rate]) <= 0:
        parser.error("Positive rates and point budget <=120 seconds required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    before = source_receipts()
    original = json.loads(args.config.read_text())
    proxy = copy.deepcopy(original)
    catalog = proxy["catalog"]
    catalog["performance_mode"] = "task_replica_proxy"
    catalog["memory_mode"] = "standalone_proxy"
    # Preserve ordinary quality and real profiles. Do not retain a misleading
    # old VALIDATED label or use its auxiliary power/memory coefficients.
    catalog.pop("calibrated_replay_v2", None)
    catalog.pop("calibrated_surrogate", None)
    catalog.pop("colocation_profiles", None)
    catalog.pop("energy_profiles", None)
    catalog["exploration_status"] = STATUS
    protocol = {"status": STATUS, "asr_rates_rps": args.asr_rates,
        "background_rates_rps": {t: args.background_rate for t in ("text", "image", "ocr")},
        "route_grid": 8, "point_budget_seconds": args.point_seconds,
        "seed": original["random_seed"], "horizon_seconds": original["horizon_seconds"],
        "all_model_and_replica_domains_retained": True,
        "planned_comparison_roles": ["free_search", "global_no_split", "same_winner_deployment_no_split"],
        "early_stop_rule": "Missing any six-term utility input blocks optimization; report unknown, never optimize partial U.",
        "structural_probes": "Previous winner plus n2 share2/3 and n3 share3/4, each with same-deployment no-split.",
        "not_validated": ["New workload rates", "standalone task-replica means transferred to shared FIFO",
            "mixed-model/mixed-task GPU interference", "input-dependent service tails", "additive memory peak",
            "high-load open-loop stability", "historical quality reproduction", "energy unavailable"],
        "weights_and_reference_values_changed": False, "gpu_started": False}
    save(args.output / "protocol.json", protocol)
    save(args.output / "proxy-base-config.json", proxy)
    domain = inspect_domain(catalog, original["gpu_memory_mib"])
    save(args.output / "full-domain-inspection.json", domain)
    reports = []
    for rate in args.asr_rates:
        point_start = time.perf_counter()
        config = copy.deepcopy(proxy)
        config["workload"] = {t: args.background_rate for t in ("text", "image", "ocr")}
        config["workload"]["asr"] = rate
        config["route_grid"] = 8
        SystemConfig(**config)
        try:
            validated_replay_spec(original["catalog"], config["workload"])
            old_gate = "UNEXPECTED_PASS"
        except ValueError as exc:
            old_gate = str(exc)
        probes = []
        for label, plan in probe_plans(config["workload"]):
            if time.perf_counter() - point_start >= args.point_seconds:
                probes.append({"label": label, "status": "POINT_BUDGET_EXHAUSTED"})
                break
            pred = predict(config["catalog"], plan["deployment"], plan["routing"],
                config["workload"], config["gpu_memory_mib"], config["latency_reference_ms"], config["weights"],
                horizon=config["horizon_seconds"], seed=config["random_seed"],
                arrival_process=config["arrival_process"], drain_seconds=config["drain_seconds"],
                energy_reference_watts=config["energy_reference_watts"],
                startup_amortization_seconds=config["startup_amortization_seconds"])
            assert pred.get("E_norm") is None and pred.get("U") is None
            probes.append({"label": label, "status": STATUS, "plan": plan, "prediction": pred,
                           "is_search_winner": False, "energy_unknown": True})
        report = {"status": "BLOCKED_SIX_TERM_ENERGY_INTERFACE", "exploration_status": STATUS,
            "workload": config["workload"], "old_v2_workload_gate": old_gate,
            "probes": probes, "free_search_winner": None, "best_global_no_split": None,
            "same_winner_deployment_no_split": None, "optimality_certified": False,
            "gap": None, "gap_reason": "No complete feasible six-term U; no lower incumbent or evaluated optimum.",
            "optimization_attempted": False, "no_split_proxy_capacity_infeasible":
                rate > domain["max_single_family_asr_rps_relaxing_background_and_memory"],
            "elapsed_seconds": time.perf_counter() - point_start,
            "public_interface_obstacle": "task_replica_proxy predicts Q/T/L/P/M but energy_profiles is empty; E and total U unknown. SearchSession requires validated calibrated predictor.",
            "gpu_started": False}
        save(args.output / f"asr-{rate:g}-background-{args.background_rate:g}.json", report)
        reports.append(report)
        print(json.dumps({"asr_rate": rate, "status": report["status"], "probes": len(probes),
                          "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    after = source_receipts()
    summary = {"schema": "spillover_cpu_discovery_interface.v1", "status": STATUS,
        "source_config": {"path": str(args.config.resolve()), "sha256": digest(args.config)},
        "protocol": protocol, "domain": domain, "points": reports,
        "source_before": before, "source_after": after, "source_unchanged": before == after,
        "six_term_winner_found": False, "gpu_started": False,
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_decision": "Independent explicitly unvalidated exploratory energy/auxiliary predictor, or new-load GPU development measurements and validation. Do not relabel old v2."}
    save(args.output / "report.json", summary)
    rows = ["# ASR 高负载 CPU 探索接口预检", "", STATUS, "",
        "保留七模型全部画像副本域、六项等权及参考值。旧 v2 绑定原负载，各个新负载均被原门禁正确拒绝。",
        "公开 task_replica_proxy 可计算五项诊断，但无整板能耗 E，因此总 U、自由赢家及 gap 均为未知。未运行无完整效用的最优化。", "",
        "下列是预声明的结构探针，不是自由搜索结果；不以目标结构约束自由搜索。", "",
        "| ASR rps | 探针 | ASR 副本 | 专用流量占比 | proxy 约束通过 | max rho | Q | E | U |",
        "|---:|---|---:|---:|---|---:|---:|---|---|"]
    for point in reports:
        for probe in point["probes"]:
            if "prediction" not in probe:
                continue
            pred, plan = probe["prediction"], probe["plan"]
            fraction = plan["routing"]["asr"].get("asr", 0) / point["workload"]["asr"]
            rows.append(f"| {point['workload']['asr']:g} | {probe['label']} | {plan['deployment'].get('asr',0)} | {fraction:.6f} | {pred.get('constraints_eligible')} | {pred.get('max_rho',0):.6f} | {pred.get('Q_norm',0):.6f} | UNKNOWN | UNKNOWN |")
    rows += ["", "所有单一模型族在画像 rho≤0.8 下的 ASR 最大容量约为 "
             f"{domain['max_single_family_asr_rps_relaxing_background_and_memory']:.6f} rps，已忽略背景负载与共驻显存限制。",
             "因此 2.5／2.6／2.7 的全域不分流在该代理下均不可行；分流探针若可行，只支持代理下的可行性改善，不能证明双方可行时的总效用提升。",
             "真实高负载共驻干扰、开环稳定性、显存峰值及质量复现均未验证。未采用旧 v2 的服务系数、功耗系数或显存系数。",
             "", "下一步需要独立、明确未验证的新能耗／辅助预测接口，或新负载 GPU 开发测量；不能伪造 v2 验证状态、补零能耗、修改权重或以部分效用冒充完整目标。", ""]
    (args.output / "README.zh.md").write_text("\n".join(rows))
    save(args.output / "manifest.json", {"files": {p.name: digest(p) for p in args.output.iterdir() if p.is_file()},
        "script": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
        "source_unchanged": before == after, "gpu_started": False})


if __name__ == "__main__":
    main()
