#!/usr/bin/env python3
"""Read only the two completed seed51 development runs; never inspect holdouts."""
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "artifacts/static-spillover-study-20261022"
SOURCE = STUDY / "predictor-source-v1"
sys.path.insert(0, str(SOURCE))
from gsdeploy.calibration import _import_source
from gsdeploy.calibration_v2 import input_service_ms, replay_input_ids
from gsdeploy.config import SystemConfig
from gsdeploy.prediction import components, predict
from gsdeploy.replay_transfer import prepare_transfer, predict_validation
from gsdeploy.routing import Router
from gsdeploy.trace import expand_routes, make_trace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p95": float(np.percentile(values, 95)),
            "min": min(values), "max": max(values)}


def union_length(intervals):
    total = 0
    right = None
    for a, b in sorted(intervals):
        if right is None or a > right:
            total += b-a
        elif b > right:
            total += b-right
        right = b if right is None else max(right, b)
    return total


def pathkey(raw, record):
    return raw["replica_to_family"][record["model"]] + "/" + record["task"]


def fifo(raw, config, service_values):
    clocks = defaultdict(float)
    task_latencies = defaultdict(list)
    waits = []
    inwindow = 0
    for r, ms in zip(raw["request_records"], service_values):
        arrival = r["scheduled_offset_seconds"]
        start = max(arrival, clocks[r["model"]])
        finish = start + ms/1000
        clocks[r["model"]] = finish
        waits.append((start-arrival)*1000)
        task_latencies[r["task"]].append((finish-arrival)*1000)
        inwindow += finish <= raw["measurement_window"]["duration_seconds"]
    per_task = {t: {"mean_ms": statistics.mean(v), "p95_ms": float(np.percentile(v, 95))}
                for t, v in task_latencies.items()}
    measured = raw["evaluation"]["utility"]
    terms = components(per_task, config.workload, config.latency_reference_ms,
        1-inwindow/len(service_values), measured["M_norm"], config.weights,
        measured["E_norm"], measured["Q_norm"])
    return {"label": "fixed_service_FIFO_diagnostic_not_causal_or_formal_prediction",
            "per_task": per_task, "mean_wait_ms": statistics.mean(waits),
            "terms_with_measured_QME": terms}


def concurrency(raw, expected):
    records = raw["request_records"]
    rows = []
    intervals = [(r["worker_response"]["start_ns"], r["worker_response"]["end_ns"])
                 for r in records]
    for i, (r, (start, end)) in enumerate(zip(records, intervals)):
        overlaps = defaultdict(list)
        integral = defaultdict(int)
        for j, (other, (a, b)) in enumerate(zip(records, intervals)):
            if i == j or other["model"] == r["model"]:
                continue
            lo, hi = max(start, a), min(end, b)
            if hi <= lo:
                continue
            family = raw["replica_to_family"][other["model"]]
            for k in ("any_other_alias", family, pathkey(raw, other)):
                overlaps[k].append((lo, hi))
                integral[k] += hi-lo
        duration = end-start
        rows.append({"request_id": r["request_id"], "path": pathkey(raw, r),
            "alias": r["model"], "sample_id": r["sample_id"],
            "service_to_frozen_ratio": r["worker_latency_ms"]/expected[i],
            "worker_ms": r["worker_latency_ms"], "baseline_ms": expected[i],
            "overlap_fraction": {k: union_length(v)/duration for k, v in overlaps.items()},
            "mean_other_busy_aliases": {k: v/duration for k, v in integral.items()}})
    grouped = {}
    for key in sorted(set(row["path"] for row in rows)):
        group = [r for r in rows if r["path"] == key]
        summary = {"requests": len(group)}
        for family in ("any_other_alias", "asr", "generalist_2", "generalist_2/asr", "ocr"):
            vals = [r["overlap_fraction"].get(family, 0) for r in group]
            bins = {}
            for label, lo, hi in (("lt10pct", -1, .1), ("10to90pct", .1, .9), ("ge90pct", .9, 2)):
                subset = [r for r in group if lo <= r["overlap_fraction"].get(family, 0) < hi]
                bins[label] = {"ratio": stats([r["service_to_frozen_ratio"] for r in subset]),
                               "service_ms": stats([r["worker_ms"] for r in subset])}
            summary[family] = {"fraction": stats(vals),
                "time_weighted_union_fraction": sum(r["worker_ms"]*r["overlap_fraction"].get(family, 0) for r in group)/sum(r["worker_ms"] for r in group),
                "mean_other_busy_aliases": stats([r["mean_other_busy_aliases"].get(family, 0) for r in group]),
                "conditional_bins": bins}
        grouped[key] = summary
    return grouped, rows


def main():
    output = STUDY / "development-asr18-reversal-audit-v1"
    output.mkdir(exist_ok=False)
    development = STUDY / "development-asr18-v1"
    packet_path = development / "experiment-packet.json"
    packet = json.loads(packet_path.read_text())
    rates = packet["config"]["workload"]
    basis = ROOT / "artifacts/static-study-gpu-20261022/surrogate-v2/system-config.json"
    transfer = prepare_transfer(basis, rates)
    config = SystemConfig(**transfer)
    spec = transfer["catalog"]["replay_transfer_v1"]
    source_hashes = {str(p): sha(p) for p in SOURCE.rglob("*.py")}
    all_raws, conditions, overlap_rows = {}, {}, {}
    accepted = {j["identity"]["runtime_code_sha256"] for j in packet["jobs"]}
    # Explicit filenames: no traversal of validation/confirmation folders.
    for label, dirname in (("no_split", "asr18_n2_specialist_share_1p0-attempt1"),
                           ("split_25pct_G2", "asr18_n2_specialist_share_0p75-attempt1")):
        path = development / "runs" / dirname / "run.json"
        source = {"id": label, **receipt(path), "packet_path": str(packet_path),
                  "packet_sha256": sha(packet_path)}
        raw, measured, _ = _import_source(source, accepted)
        assert raw["transport_config"]["random_seed"] == 51
        assert raw["status"] == "completed"
        assert measured["eligible"]
        all_raws[label] = raw
        x, routing = raw["plan"]["deployment"], raw["plan"]["routing"]
        predicted = predict_validation(config, x, routing, horizon=600, seed=51)
        assert predicted["U"] is None and not predicted["eligible"]
        trace = make_trace(rates, 600, 51, config.arrival_process)
        expected_ids = replay_input_ids(config.catalog, trace, 51)
        assert expected_ids == [r["sample_id"] for r in raw["request_records"]]
        aliases, route_aliases = expand_routes(x, routing)
        router = Router(route_aliases)
        assert [router.choose(t) for _, t in trace] == [r["model"] for r in raw["request_records"]]
        expected = [input_service_ms(spec, aliases[r["model"]], r["task"], x[aliases[r["model"]]], r["sample_id"])
                    for r in raw["request_records"]]
        actual = [r["worker_latency_ms"] for r in raw["request_records"]]
        summaries = {}
        for grouping in ("path", "alias"):
            summaries[grouping] = {}
            keys = sorted(set(pathkey(raw, r) if grouping == "path" else r["model"] for r in raw["request_records"]))
            for key in keys:
                indices = [i for i, r in enumerate(raw["request_records"])
                           if (pathkey(raw, r) if grouping == "path" else r["model"]) == key]
                records = [raw["request_records"][i] for i in indices]
                summary = {"requests": len(indices), "unique_inputs": len(set(r["sample_id"] for r in records)),
                    "actual_to_frozen_service_sum_ratio": sum(actual[i] for i in indices)/sum(expected[i] for i in indices),
                    "frozen_service_ms": stats([expected[i] for i in indices]),
                    "worker_handler_ms": stats([actual[i] for i in indices]),
                    "busy_handler_seconds_per_600s": sum(actual[i] for i in indices)/1000,
                    "input_tokens": stats([r["worker_response"].get("input_tokens") for r in records]),
                    "generated_tokens": stats([r["worker_response"].get("generated_tokens") for r in records])}
                for field in ("worker_queue_wait_ms", "e2e_latency_ms", "schedule_delay_ms"):
                    summary[field] = stats([r.get(field) for r in records])
                summary["worker_service_ms"] = stats([r["worker_response"].get("service_ms") for r in records])
                summaries[grouping][key] = summary
        overlap, overlap_rows[label] = concurrency(raw, expected)
        raw_proxy = copy.deepcopy(raw["config"]["catalog"])
        for key in ("calibrated_replay_v2", "calibrated_surrogate", "replay_transfer_v1", "energy_profiles", "colocation_profiles"):
            raw_proxy.pop(key, None)
        raw_proxy["performance_mode"], raw_proxy["memory_mode"] = "task_replica_proxy", "standalone_proxy"
        constant = predict(raw_proxy, x, routing, rates, config.gpu_memory_mib, config.latency_reference_ms,
            config.weights, horizon=600, seed=51, arrival_process=config.arrival_process,
            drain_seconds=config.drain_seconds, energy_reference_watts=config.energy_reference_watts,
            startup_amortization_seconds=config.startup_amortization_seconds)
        assert constant["U"] is None and constant["E_norm"] is None
        demand = math.fsum(q*raw_proxy["task_replica_profiles"][m][t][str(x[m])]["mean_service_ms"]/1000
                          for t, row in routing.items() for m, q in row.items())
        features = [1., demand, float(sum(x.values()))]
        watts = sum(a*b for a,b in zip(features,spec["power_coefficients"]))
        external_E = (watts + spec["startup_joules_per_worker"]*sum(x.values())/config.startup_amortization_seconds)/config.energy_reference_watts
        constant_variants = {}
        for memory_label, memory in (("raw_additive_memory", constant["M_norm"]),
                                      ("old_learned_memory", predicted["M_norm"])):
            terms = {k: constant[k+"_norm"] for k in "QTLP"}
            terms.update(M=memory, E=external_E)
            constant_variants[memory_label] = {"six_terms": terms,
                "external_diagnostic_score": -sum(config.weights[k]*v for k,v in terms.items()),
                "formal_U": None, "eligible": False,
                "scope": "constant raw n-state means; old external power; no refitting or physical validation"}
        counterfactuals = {"frozen_service_measured_QME": fifo(raw,config,expected),
                           "all_observed_service_measured_QME": fifo(raw,config,actual)}
        for key in sorted(summaries["path"]):
            service = [actual[i] if pathkey(raw,r)==key else expected[i] for i,r in enumerate(raw["request_records"])]
            counterfactuals["replace_only_"+key] = fifo(raw,config,service)
        memory = raw["memory_evidence"]
        energy = raw["energy_evidence"]
        conditions[label] = {"source": source, "strict_import_passed": True,
            "measurement_id": raw["executed_measurement_id"], "plan": raw["plan"],
            "stability": raw["evaluation"]["stability"],
            "measured": measured, "prediction_unvalidated": predicted,
            "constant_mean_diagnostics": constant_variants, "path_and_alias": summaries,
            "concurrency": overlap, "counterfactual_fixed_service_FIFO": counterfactuals,
            "measurement_window": raw["measurement_window"],
            "memory_summary": {"safe_memory_mib": memory["safe_memory_mib"],
                "resource_valid": memory["resource_valid"], "scope": memory["scope"],
                "sampled_peak_owned_mib": max(s["owned_used_mib"] for s in memory["samples"]),
                "foreign_pid_observations": sum(bool(s.get("foreign_pids")) for s in memory["samples"])},
            "energy_summary": {k: energy.get(k) for k in ("run_joules", "startup_joules", "scope", "coverage_valid", "sampling_errors")},
            "total_service_relative_error": abs(sum(actual)-sum(expected))/sum(actual)}
        print(label, measured["U"], predicted["diagnostic_partial_U"], flush=True)
    a,b = (all_raws[k] for k in ("no_split","split_25pct_G2"))
    request_pairs = list(zip(a["request_records"], b["request_records"]))
    assert len(a["request_records"]) == len(b["request_records"])
    pairs = defaultdict(list)
    for ra,rb in request_pairs:
        if pathkey(a,ra)!=pathkey(b,rb):
            continue
        pairs[pathkey(a,ra)].append((ra,rb))
    paired = {}
    for key, group in pairs.items():
        paired[key] = {"requests":len(group),
            "same_request_sha_count":sum(ra["worker_response"].get("request_sha256")==rb["worker_response"].get("request_sha256") for ra,rb in group),
            "same_output_count":sum(ra["worker_response"].get("prediction")==rb["worker_response"].get("prediction") for ra,rb in group),
            "same_generated_token_count":sum(ra["worker_response"].get("generated_tokens")==rb["worker_response"].get("generated_tokens") for ra,rb in group),
            "split_minus_nosplit_worker_ms":stats([rb["worker_latency_ms"]-ra["worker_latency_ms"] for ra,rb in group]),
            "split_over_nosplit_worker_ratio":stats([rb["worker_latency_ms"]/ra["worker_latency_ms"] for ra,rb in group])}
    ca,cb=(conditions[k] for k in ("no_split","split_25pct_G2"))
    attribution={}
    total=sum(rates.values())
    for t in sorted(rates):
        ma,mb=a["per_task_metrics"][t],b["per_task_metrics"][t]
        attribution[t]={"delta_L_contribution":-config.weights["L"]*rates[t]/total/config.latency_reference_ms[t]*(mb["mean_e2e_latency_ms"]-ma["mean_e2e_latency_ms"]),
                        "delta_P_contribution":-config.weights["P"]*rates[t]/total/config.latency_reference_ms[t]*(mb["p95_e2e_latency_ms"]-ma["p95_e2e_latency_ms"]),
                        "no_split_mean_ms":ma["mean_e2e_latency_ms"],"split_mean_ms":mb["mean_e2e_latency_ms"],
                        "no_split_p95_ms":ma["p95_e2e_latency_ms"],"split_p95_ms":mb["p95_e2e_latency_ms"]}
    delta={k:cb["measured"][k]-ca["measured"][k] for k in ["U"]+[t+"_contribution" for t in "QTLPME"]}
    report={"schema":"spillover_development_reversal_audit.v1", "status":"READONLY_DEVELOPMENT_DIAGNOSIS",
        "scope":"Two seed51 development runs only; no validation or confirmation result was read; no fitting/GPU/threshold edits",
        "source":receipt(SOURCE/"manifest.json"), "basis":receipt(basis), "packet":receipt(packet_path),
        "conditions":conditions,"input_comparison":{
            "same_trace_sha":a["measurement_window"]["trace_sha256"]==b["measurement_window"]["trace_sha256"],
            "same_ordered_task_sample_schedule":all((ra["task"],ra["sample_id"],ra["scheduled_offset_seconds"])==(rb["task"],rb["sample_id"],rb["scheduled_offset_seconds"]) for ra,rb in request_pairs),
            "same_family_paired_requests":paired},
        "split_minus_nosplit_contributions":delta,"task_latency_contribution_deltas":attribution,
        "limitations":["Only one paired seed: no statistical superiority or physical optimum claim",
            "Worker handler interval overlap measures concurrency, not GPU kernel occupancy, utilization or causal slowdown",
            "Overlap bins are observational, affected by task length and queue selection; normalized input ratios do not remove all confounding",
            "Fixed-service FIFO substitutions diagnose model arithmetic only; observed service is itself route/load dependent",
            "Raw sampled peak memory cannot certify an unsampled hardware maximum"],
        "gpu_started":False,"parameters_refitted":False,"validation_results_read":False}
    report["frozen_python_sources_unchanged"]=all(sha(p)==h for p,h in source_hashes.items())
    assert report["frozen_python_sources_unchanged"]
    assert abs(sum(delta[k+"_contribution"] for k in "QTLPME")-delta["U"])<1e-12
    save(output/"audit.json",report)
    save(output/"request-concurrency.json",overlap_rows)
    write_chinese(output,report)
    save(output/"manifest.json",{"script":receipt(__file__),"files":{p.name:sha(p) for p in output.iterdir() if p.is_file()},
        "gpu_started":False,"parameters_refitted":False,"validation_results_read":False})
    print(str(output),flush=True)


def write_chinese(output,report):
    conditions=report["conditions"]
    lines=["# 开发 seed51：分流排序反转只读诊断", "", 
        "仅审计已完成的两个开发 raw。未读取 61/62 或最终确认结果，未启动 GPU、未改冻结模型参数或验证阈值。严格导入均通过，实测总效用均有效。", "",
        "## 结论", "",
        "冻结逐输入模型认为 25% ASR 分流到 G2 更好，实测相反。主要差额来自延迟，尤其 G2 接入 ASR 后引起的共享 FIFO 排队，以及若干路径的实际服务时间远高于冻结独立画像。该证据不支持宣布分流收益或物理最优。", "",
        "| 条件 | 实测 U | 逐输入未验证诊断分数 | L | P | Q质量损失 | M | E |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key,c in conditions.items():
        m=c["measured"]
        lines.append(f"| {key} | {m['U']:.9f} | {c['prediction_unvalidated']['diagnostic_partial_U']:.9f} | {m['L_norm']:.6f} | {m['P_norm']:.6f} | {m['Q_norm']:.6f} | {m['M_norm']:.6f} | {m['E_norm']:.6f} |")
    delta=report["split_minus_nosplit_contributions"]
    lines += ["",f"分流减不分流的实测 ΔU = {delta['U']:.9f}。六项效用贡献差（正值更好）：", "",
        "| Q | T | L | P | M | E |", "|---:|---:|---:|---:|---:|---:|",
        "| "+" | ".join(f"{delta[k+'_contribution']:.9f}" for k in "QTLPME")+" |", "",
        "Q 为错误/损失，越小越好。此开发对中分流的 Q 略有改善，不能把设计时专用整体精度更高解释成每个有限输入子集都严格更高。真实输出评分已保留。", "",
        "## 同输入与路径成本", "",
        f"两个 raw 的到达 trace 相同：{report['input_comparison']['same_trace_sha']}；有序任务、sample_id 和计划到达偏移相同：{report['input_comparison']['same_ordered_task_sample_schedule']}。路由改变导致 ASR 输入在模型间重分配，不能直接比较未校正的路径均值。以下用各路径实际输入逐个查冻结基线后求和比较。", "",
        "| 条件 | 路径 | 请求数 | 实际服务均值ms | 冻结逐输入均值ms | 实际/预测总服务 | 等待均值ms | 端到端均值ms |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name,c in conditions.items():
        for key,p in c["path_and_alias"]["path"].items():
            lines.append(f"| {name} | {key} | {p['requests']} | {p['worker_handler_ms']['mean']:.3f} | {p['frozen_service_ms']['mean']:.3f} | {p['actual_to_frozen_service_sum_ratio']:.3f} | {p['worker_queue_wait_ms']['mean']:.3f} | {p['e2e_latency_ms']['mean']:.3f} |")
    lines += ["", "同一路径、同一请求的配对比较可区分输入分配差异，详情见 audit.json 的 input_comparison。各别名服务忙时、输入长度、生成长度、服务/等待/p95、实际输出相同数均保存。", "",
        "## 哪些任务造成效用下降", "", "| 任务 | ΔL效用贡献 | ΔP效用贡献 | 不分流均值ms | 分流均值ms | 不分流p95ms | 分流p95ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key,p in report["task_latency_contribution_deltas"].items():
        lines.append(f"| {key} | {p['delta_L_contribution']:.9f} | {p['delta_P_contribution']:.9f} | {p['no_split_mean_ms']:.3f} | {p['split_mean_ms']:.3f} | {p['no_split_p95_ms']:.3f} | {p['split_p95_ms']:.3f} |")
    lines += ["", "图像背景虽仅 .05 rps，但延迟参考值仅 250ms，G2 共享队列变慢会放大归一化 p95 惩罚。低比例背景任务也必须纳入完整总效用，不能只看 ASR 专用队列是否下降。", "",
        "## 并发、代理模型与重放检查", "",
        "worker_response.start_ns/end_ns 使用共同 UNIX 时钟，逐请求与其他别名区间求交；不会混用 host monotonic 计划时间。保存其他 ASR 副本、G2 ASR、G2 总路径、OCR 的重叠比例和输入归一化服务比条件统计。这里测到的是 worker handler 并发，不能据此声称 GPU kernel 竞争已被因果识别。", "",
        "| 条件 | 路径 | ASR其他副本/路径时间重叠 | G2 ASR时间重叠 | 任意其他别名时间重叠 |", "|---|---|---:|---:|---:|"]
    for name,c in conditions.items():
        for key,p in c["concurrency"].items():
            lines.append(f"| {name} | {key} | {p['asr']['time_weighted_union_fraction']:.3f} | {p['generalist_2/asr']['time_weighted_union_fraction']:.3f} | {p['any_other_alias']['time_weighted_union_fraction']:.3f} |")
    lines += ["", "常量均值旧探索口径重新以同一 seed51/H600 计算，独立保留原始加性显存与旧学习显存两个版本：", "", "| 条件 | 常量均值+原始显存 | 常量均值+旧学习显存 | 实际服务FIFO诊断U（实测Q/M/E） |", "|---|---:|---:|---:|"]
    for name,c in conditions.items():
        d=c["constant_mean_diagnostics"]
        f=c["counterfactual_fixed_service_FIFO"]["all_observed_service_measured_QME"]["terms_with_measured_QME"]["U"]
        lines.append(f"| {name} | {d['raw_additive_memory']['external_diagnostic_score']:.9f} | {d['old_learned_memory']['external_diagnostic_score']:.9f} | {f:.9f} |")
    lines += ["", "这些都是诊断分数，正式预测 U 保持空。另保存“冻结服务+实测Q/M/E”、逐条路径单独替换实际服务的 FIFO 算术诊断；替换之间存在共享队列交互，单路径差额不能直接相加或作为改路由后的真实反事实。", "",
        "## 下一轮最少测量与新模型建议", "",
        "1. 保持 61/62 预声明的零重拟合验证按原协议结束，原模型即使失败也保留。此诊断完全不读取该结果，不把失败隐去或改阈值。暂不把 71/72/73 最终确认用于选参数或选路由。", 
        "2. 先用此开发对完成静态误差分解。G2 text/image 在不分流和分流下的偏差方向/幅度不同，OCR 同样明显偏慢，不能只对 G2 ASR 加一个乘数，更不能仅改一个全局服务系数。", 
        "3. 若只需检验 ASR/G2 共驻并发交互，最少新增两个开发控制：同一 ASR2+G2n1+OCR1 驻留和同一 seed53 输入下，(a) ASR2 执行而 G2 只按低频背景执行；(b) G2 执行 .45rps ASR+背景而 ASR 两副本驻留空闲。与已有混合同时执行观测相比能定位活动并发，但严格配对应补一条 seed53 同时活动条件，共三条。保持模型驻留/输入/暖机/电源状态/执行顺序随机化并记录实际温度与时钟；不要用暂停某进程改变内存常驻。该三条属于开发，不复用 61/62。", 
        "4. 两条隔离控制仍不足以拟合跨负载响应曲面。新模型可采用每路径逐输入基线 × 低维活动干扰系数：自身副本数、其他家族忙度、ASR/G2 成对活动项、G2 内任务混合；保留按别名共享 FIFO 产生排队。所用忙度必须由预测服务与到达自洽求解，不能使用待预测实测的 overlap 造成数据泄漏。仅在开发控制支持时启用相应交互项。", 
        "5. 优先直接标定当前固定部署/三条路由的低维局部模型，避免穷举所有混合组合；若要进入全池自由搜索，必须扩充结构覆盖和可验证的保守界限。现有低维局部测量不能保证所有部署可预测，也不能沿用旧全局证书。", 
        "6. 新模型另命名、另版本、另绑定训练来源与预声明误差阈值。若已查看 61/62 后据其改模型，它们只能保留为旧模型失败验证并标记后续开发参考；新模型验证用全新且不相交的 81/82 等种子。最终确认仍用未参与开发的保留种子；一旦看过用于调参则必须换新。", "",
        "这两条开发结果只说明原迁移模型在该条件下排名失真，不能证明所有混合部署都差；选择是否分流仍须让完整六项效用、共同可行性与独立证据决定。", ""]
    (output/"diagnosis.zh.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
