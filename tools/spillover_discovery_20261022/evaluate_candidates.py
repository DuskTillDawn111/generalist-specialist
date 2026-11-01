#!/usr/bin/env python3
"""Limited CPU candidates: proxy FIFO plus external unvalidated energy score.

No production U, synthetic energy record, validation override or optimizer is
created. Historical power parameters are extrapolated under an explicit label.
"""

import argparse
import copy
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsdeploy.prediction import predict
from gsdeploy.trace import make_trace

STATUS = "EXPLORATORY_TRANSFER_NOT_VALIDATED"
FRACTIONS = [Fraction(1), Fraction(7, 8), Fraction(5, 6), Fraction(4, 5),
             Fraction(3, 4), Fraction(2, 3), Fraction(1, 2)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def compact(candidate):
    if candidate is None:
        return None
    return {k: candidate[k] for k in ("candidate_id", "deployment", "routing", "specialist_fraction",
        "exploratory_score", "six_external_terms", "proxy_constraints_eligible", "max_rho")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=ROOT / "artifacts/static-study-gpu-20261022/search-v2-v5-K8/system-config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/static-spillover-study-20261022/cpu-discovery/external-score-v1")
    parser.add_argument("--asr-rates", nargs="+", type=float, default=[1.6, 1.7, 1.8, 1.85, 2.5, 2.6, 2.7])
    parser.add_argument("--point-seconds", type=float, default=120.)
    args = parser.parse_args()
    if not 0 < args.point_seconds <= 120 or min(args.asr_rates) <= 0:
        parser.error("Positive rates and <=120s point budget required")
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.source_config.read_text())
    old = original["catalog"]["calibrated_replay_v2"]
    coeff = old["power_coefficients"]
    startup_worker = old["startup_joules_per_worker"]
    data = copy.deepcopy(original["catalog"])
    data["performance_mode"], data["memory_mode"] = "task_replica_proxy", "standalone_proxy"
    for key in ("calibrated_replay_v2", "calibrated_surrogate", "energy_profiles", "colocation_profiles"):
        data.pop(key, None)
    sources_before = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "gsdeploy").rglob("*.py"))}
    protocol = {"status": STATUS, "asr_rates": args.asr_rates, "background_each_rps": .05,
        "seed": original["random_seed"], "horizon_seconds": original["horizon_seconds"],
        "point_budget_seconds": args.point_seconds,
        "limited_deployments": {"asr": [1, 2], "generalist_2": [1, 2], "ocr": [1]},
        "specialist_fractions": [str(f) for f in FRACTIONS],
        "background_routing": {"text": "generalist_2", "image": "generalist_2", "ocr": "ocr"},
        "service_model": "Public unmodified constant measured task-replica mean FIFO proxy; no old v2 service multiplier or input-service model.",
        "memory_model": "Public standalone additive residency sum times original safe factor; no old v2 memory multiplier.",
        "power_model": "Externally calculate features=[1,sum(route_rps*measured_n_state_mean_ms/1000),worker_count]; dot frozen historical coefficients. No runtime energy evidence is created.",
        "power_coefficients": coeff, "startup_joules_per_worker": startup_worker,
        "power_source": {"path": str(args.source_config.resolve()), "sha256": sha(args.source_config),
            "original_version_sha256": old["version_sha256"], "original_workload": old["workload"],
            "source_training_feature_min": old["training_feature_min"],
            "source_training_feature_max": old["training_feature_max"]},
        "score_definition": "exploratory_score=-sum(original_weight[k]*external_term[k]); stored outside production U and evidence gates.",
        "original_six_weights": original["weights"], "original_references": original["latency_reference_ms"],
        "not_validated": ["new workload", "aggregate per-replica service transfer", "mixed-model interference",
            "open-loop latency tails", "additive memory", "quality reproduction", "historical power/startup extrapolation"],
        "all_pool_or_global_optimality_claim": False, "production_prediction_U_remains_null": True,
        "gpu_started": False}
    save(args.output / "protocol.json", protocol)
    summaries = []
    for rate in args.asr_rates:
        started = time.perf_counter()
        rates = dict(text=.05, image=.05, ocr=.05, asr=rate)
        trace = make_trace(rates, original["horizon_seconds"], original["random_seed"], original["arrival_process"])
        candidates = []
        exhausted = False
        for n_asr, n_g2, fraction in itertools.product((1, 2), (1, 2), FRACTIONS):
            if time.perf_counter() - started >= args.point_seconds:
                exhausted = True
                break
            x = dict(asr=n_asr, generalist_2=n_g2, ocr=1)
            row = {"asr": rate * float(fraction)}
            if fraction < 1:
                row["generalist_2"] = rate * (1 - float(fraction))
            routing = dict(text={"generalist_2": .05}, image={"generalist_2": .05}, ocr={"ocr": .05}, asr=row)
            predicted = predict(data, x, routing, rates, original["gpu_memory_mib"],
                original["latency_reference_ms"], original["weights"],
                horizon=original["horizon_seconds"], seed=original["random_seed"],
                arrival_process=original["arrival_process"], drain_seconds=original["drain_seconds"],
                energy_reference_watts=original["energy_reference_watts"],
                startup_amortization_seconds=original["startup_amortization_seconds"], trace=trace)
            assert predicted.get("U") is None and predicted.get("E_norm") is None
            demand = math.fsum(q * data["task_replica_profiles"][m][t][str(x[m])]["mean_service_ms"] / 1000
                               for t, route in routing.items() for m, q in route.items())
            features = [1., demand, float(sum(x.values()))]
            watts = sum(a * b for a, b in zip(features, coeff))
            startup = startup_worker * sum(x.values())
            external_energy = (watts + startup / original["startup_amortization_seconds"]) / original["energy_reference_watts"]
            terms = {k: predicted.get(k + "_norm") for k in ("Q", "T", "L", "P", "M")}
            terms["E"] = external_energy
            eligible = predicted.get("constraints_eligible", False)
            score = -math.fsum(original["weights"][k] * v for k, v in terms.items()) if eligible and all(v is not None for v in terms.values()) else None
            candidates.append({"candidate_id": f"asr{n_asr}-g2n{n_g2}-ocr1-share{fraction.numerator}of{fraction.denominator}",
                "status": STATUS, "deployment": x, "routing": routing,
                "specialist_fraction": str(fraction), "exploratory_score": score,
                "six_external_terms": terms, "proxy_constraints_eligible": eligible,
                "constraint_or_evidence_reason": predicted["reason"],
                "max_rho": predicted.get("max_rho"), "rho": predicted.get("rho"),
                "safe_memory_mib": predicted.get("safe_memory_mib"),
                "per_task": predicted.get("per_task"), "quality_evidence": predicted.get("quality_evidence"),
                "external_energy": {"status": STATUS, "power_features": features,
                    "feature_outside_historical_training_range": [v < a or v > b for v, a, b in zip(features, old["training_feature_min"], old["training_feature_max"])],
                    "board_watts": watts, "startup_joules": startup, "measured": False},
                "production_U": predicted.get("U"), "production_E": predicted.get("E_norm"),
                "production_eligible": predicted.get("eligible"), "production_reason": predicted.get("reason"),
                "offered_requests": predicted.get("offered_requests"), "trace_sha256": predicted.get("trace_sha256")})
        valid = [c for c in candidates if c["exploratory_score"] is not None]
        best = max(valid, key=lambda c: c["exploratory_score"], default=None)
        no_split = max((c for c in valid if c["specialist_fraction"] == "1"), key=lambda c: c["exploratory_score"], default=None)
        same = next((c for c in valid if best and c["deployment"] == best["deployment"] and c["specialist_fraction"] == "1"), None)
        summary = {"workload": rates, "status": STATUS, "planned_candidates": 28,
            "evaluated_candidates": len(candidates), "proxy_feasible_candidates": len(valid),
            "point_budget_exhausted": exhausted, "elapsed_seconds": time.perf_counter() - started,
            "best_among_listed_candidates_only": compact(best),
            "best_no_split_among_listed_candidates_only": compact(no_split),
            "same_selected_deployment_no_split": compact(same),
            "delta_vs_listed_best_no_split": best["exploratory_score"] - no_split["exploratory_score"] if best and no_split else None,
            "delta_vs_same_deployment_no_split": best["exploratory_score"] - same["exploratory_score"] if best and same else None,
            "selected_target_structure": bool(best and best["deployment"]["asr"] >= 2 and best["specialist_fraction"] != "1"),
            "global_free_search_performed": False, "optimality_certified": False, "global_gap": None}
        save(args.output / f"asr-{rate:g}.json", {**summary, "candidates": candidates})
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    sources_after = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "gsdeploy").rglob("*.py"))}
    save(args.output / "report.json", {"schema": "spillover_external_candidate_scores.v1", "protocol": protocol,
        "points": summaries, "source_unchanged": sources_before == sources_after,
        "source_before": sources_before, "source_after": sources_after, "gpu_started": False})
    lines = ["# ASR 高负载有限候选外部评分", "", STATUS, "",
        "仅评估每点 28 个预声明候选：ASR n1/2、G2 n1/2、OCR n1；背景 text/image→G2、OCR→专用，背景各 .05 rps。",
        "使用真实画像平均服务的公开 FIFO 代理及逐输入质量，再于生产接口之外外推旧功率参数计算六项探索分数。",
        "旧 v2 验证不迁移；生产 U/E 始终 UNKNOWN，未建立合成能耗证据或绕过门禁。不能称为全池自由搜索赢家。", "",
        "| ASR rps | 列表最佳 | 探索分数 | 列表最佳不分流 | 分数差 | 同部署不分流差 | ASR≥2且分流 |",
        "|---:|---|---:|---|---:|---:|---|"]
    for p in summaries:
        best, no_split = p["best_among_listed_candidates_only"], p["best_no_split_among_listed_candidates_only"]
        fmt = lambda v: "UNKNOWN" if v is None else f"{v:.6f}"
        lines.append(f"| {p['workload']['asr']:g} | {best['candidate_id'] if best else 'NONE'} | {fmt(best['exploratory_score'] if best else None)} | {no_split['candidate_id'] if no_split else 'NONE'} | {fmt(p['delta_vs_listed_best_no_split'])} | {fmt(p['delta_vs_same_deployment_no_split'])} | {p['selected_target_structure']} |")
    lines += ["", "列表内的最佳仅是有限扫描排名，不给全池最优性证书或全局 gap；未测真实干扰、稳定性、质量复现及高负载功率。",
        "2.5–2.7 的不分流不可行属于容量救援区，不作双方均可行时的效用收益证据。低负载若有正差，仍只是未验证外推下的 GPU 开发候选。",
        "旧功率模型的需求特征往往超过原训练范围；每候选均记录越界标记、参数来源、绝对路由流量和六项外部成本。", ""]
    (args.output / "README.zh.md").write_text("\n".join(lines))
    save(args.output / "manifest.json", {"files": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "source_unchanged": sources_before == sources_after, "gpu_started": False})


if __name__ == "__main__":
    main()
