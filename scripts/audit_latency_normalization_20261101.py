"""CPU-only rescoring of frozen observations at common latency references.

No optimization of reference values, deployment, or routing is performed.
Run as: python -m scripts.audit_latency_normalization_20261101 --output PATH
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from gsdeploy.prediction import components
from scripts.analyze_latency_references_20261026 import sections, sha

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("text", "image", "ocr", "asr")
COMMON_MS = (500, 1000, 2000, 5000)
TERMS = "QTLPME"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "artifacts/static-study-gpu-20261022/study-summary-final-v1/report.json"
    summary = json.loads(source.read_text())
    confirmation = summary["confirmation"]
    sources = [{"path": str(source), "sha256": sha(source)}]
    paths = {Path(r["path"]): r["sha256"] for r in summary["receipts"]["confirmation_files"]
             if Path(r["path"]).name == "run.json"}
    assert len(paths) == 15
    saved = {r["job_id"]: r for r in confirmation["measured_runs"]}
    rows, observations = [], []
    for path, digest in paths.items():
        assert sha(path) == digest, path
        raw = sections(path)
        config, utility = raw["config"], raw["evaluation"]["utility"]
        assert utility["eligible"] is True
        assert all(math.isclose(config["weights"][k], 1/6, abs_tol=1e-14) for k in TERMS)
        seed = config["random_seed"]
        job = next(r for job_id, r in saved.items() if job_id[:12] in path.parent.name)
        assert job["seed"] == seed
        assert math.isclose(utility["U"], job["utility"]["U"], abs_tol=1e-10)
        stats = {t: {"mean_ms": raw["per_task_metrics"][t]["mean_e2e_latency_ms"],
                     "p95_ms": raw["per_task_metrics"][t]["p95_e2e_latency_ms"]} for t in TASKS}

        def score(refs):
            return components(stats, config["workload"], refs, utility["T_norm"],
                              utility["M_norm"], config["weights"],
                              E=utility["E_norm"], Q=utility["Q_norm"])

        old = score(config["latency_reference_ms"])
        assert math.isclose(old["U"], utility["U"], abs_tol=1e-10)
        for k in TERMS:
            assert math.isclose(old[k+"_norm"], utility[k+"_norm"], abs_tol=1e-10)
        label = path.parent.name.split("-seed")[0]
        observations.append(dict(label=label, seed=seed, plan_key=job["plan_key"],
                                 workload=config["workload"], per_task=stats,
                                 original_utility={k+"_norm": utility[k+"_norm"] for k in TERMS},
                                 original_reference_ms=config["latency_reference_ms"]))
        for name, refs in [("historical", config["latency_reference_ms"]),
                           *[(f"common_{ms}ms", dict.fromkeys(TASKS, ms)) for ms in COMMON_MS]]:
            result = score(refs)
            rows.append(dict(scheme=name, label=label, seed=seed, plan_key=job["plan_key"],
                             **{k: result[k+"_norm"] for k in TERMS}, U=result["U"]))
        sources.append(dict(path=str(path), sha256=digest))

    winner = confirmation["paired"]["winner_key"]
    comparisons = []
    for ms in COMMON_MS:
        scheme = f"common_{ms}ms"
        lookup = {(r["plan_key"], r["seed"]): r for r in rows if r["scheme"] == scheme}
        for old_pair in confirmation["paired"]["comparisons"]:
            seeds = old_pair["paired_seeds"]
            rival = old_pair["rival_key"]
            deltas = [lookup[winner, s]["U"] - lookup[rival, s]["U"] for s in seeds]
            comparisons.append(dict(scheme=scheme, rival=lookup[rival, seeds[0]]["label"],
                                    paired_seeds=seeds, delta_U=deltas,
                                    mean_delta_U=math.fsum(deltas)/len(deltas),
                                    all_observed_positive=all(d > 0 for d in deltas)))

    sensitivity = ROOT / "artifacts/latency-reference-sensitivity-20261026-v1/analysis.json"
    prior = json.loads(sensitivity.read_text())
    sources.append(dict(path=str(sensitivity), sha256=sha(sensitivity)))
    dynamic_rows = []
    for p in prior["pairs"]:
        assert math.isclose(p["constant"]+math.fsum(p["task_coefficients"].values()),
                            p["original_delta_U"], abs_tol=1e-10)
        for ms in COMMON_MS:
            direct = p["constant"] + math.fsum(
                d["alpha"] * (d["mean_control_ms"] - d["mean_spill_ms"]
                              + d["p95_control_ms"] - d["p95_spill_ms"]) / ms / 6
                for d in p["details"].values())
            coefficient = p["constant"] + math.fsum(
                p["task_coefficients"][t]*prior["base_b_ms"][t]/ms for t in TASKS)
            assert math.isclose(direct, coefficient, abs_tol=1e-10)
            dynamic_rows.append(dict(scheme=f"common_{ms}ms", seed=p["seed"], study=p["study"],
                                     original_delta_U=p["original_delta_U"], delta_U=direct))
    assert len(prior["pairs"]) == 15
    report = dict(status="POSTHOC_FIXED_OBSERVATION_REFERENCE_SENSITIVITY",
                  common_reference_ms=list(COMMON_MS),
                  scope="Original completed observations and original eligibility only. No new GPU run, search, routing-policy replay, SLO validation, or optimality certificate.",
                  formula="U=-(Q+T+L+P+M+E)/6; L and P share one fixed reference per task.",
                  static_jobs=15, static_observations=observations, static_rescores=rows,
                  static_comparisons=comparisons, dynamic_rescores=dynamic_rows,
                  sources=sources, script_sha256=sha(__file__))
    write_json(output/"analysis.json", report)
    for filename, records in (("static-rescores.csv", rows), ("dynamic-rescores.csv", dynamic_rows)):
        with (output/filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    lines = ["# 统一延迟参考的离线重评分", "",
             "仅重评分既有固定部署/路由的观测；不是新目标下的重新搜索或在线策略重放。",
             "保留原始资格判定，全部六项等权，平均/P95 共用参考值。尺度在本次重评分前固定为 0.5/1/2/5 秒，主配置为用户指定的 1 秒。", "",
             "静态确认：5 个既有候选 × seeds 43/44/45。ΔU 为旧搜索选中方案减去对照；正值表示前者在新尺度下分数更高。三个种子仅作描述性比较。", "",
             "|共同参考 ms|对照|三个 seed 的 ΔU|平均 ΔU|", "|---:|---|---|---:|"]
    for c in comparisons:
        lines.append(f"|{c['scheme']}|{c['rival']}|" + ", ".join(f"{v:.7g}" for v in c["delta_U"]) + f"|{c['mean_delta_U']:.7g}|")
    lines += ["", "最近耦合 spillover 配对：ΔU = spill − control。历史 15 配对全部保留在 CSV/JSON；不跨策略阶段合并。", "",
              "|seed|旧分母 ΔU|共同 1 秒 ΔU|", "|---:|---:|---:|"]
    for row in dynamic_rows:
        if row["seed"] in (303, 304) and row["scheme"] == "common_1000ms":
            lines.append(f"|{row['seed']}|{row['original_delta_U']:.9g}|{row['delta_U']:.9g}|")
    lines += ["", "改变分母会改变偏好和排名，不改变延迟观测。旧候选集合并不穷尽新目标的可行域，旧最优性证书不能迁移。",
              "逐任务原始延迟、六项贡献、静态原始文件已核验的 SHA256 及动态系数文件来源见 analysis.json。动态配对由既有已核验系数/观测摘要重算，未在本脚本重新扫描全部动态原始大文件。"]
    (output/"report.zh.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"static_jobs":15, "dynamic_pairs":15, "output":str(output),
                      "checks":"Historical static six-term scores reproduced; raw static hashes checked; dynamic direct and coefficient formulas agree."}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
