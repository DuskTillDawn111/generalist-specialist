#!/usr/bin/env python3
"""External CPU exploration fitted only on two seed51 development runs."""
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_development_reversal import STUDY, SOURCE, receipt, save, sha, fifo
from gsdeploy.calibration import _deployment_key
from gsdeploy.calibration_v2 import input_service_ms, replay_input_ids, replay_path_statistics
from gsdeploy.config import SystemConfig
from gsdeploy.prediction import components
from gsdeploy.quality import predict_quality
from gsdeploy.replay_transfer import prepare_transfer
from gsdeploy.routing import Router
from gsdeploy.trace import expand_routes, make_trace

X = {"asr": 2, "generalist_2": 1, "ocr": 1}
SPLITS = (0., .125, .25)
LOADS = (1.8, 1.9, 2., 2.1, 2.2)
BACKGROUNDS = (.005, .01, .05)


def main():
    output = STUDY / "cpu-discovery/empirical-two-run-scan-v1"
    output.mkdir(exist_ok=False)
    audit_path = STUDY / "development-asr18-reversal-audit-v1/audit.json"
    audit = json.loads(audit_path.read_text())
    basis = ROOT / "artifacts/static-study-gpu-20261022/surrogate-v2/system-config.json"
    original = json.loads(basis.read_text())
    spec = original["catalog"]["calibrated_replay_v2"]
    data = original["catalog"]
    source_hashes={str(p):sha(p) for p in SOURCE.rglob("*.py")}
    inputs = {}
    ratios = {}
    residuals = {}
    training_runs = {}
    for label, share in (("no_split", 0.), ("split_25pct_G2", .25)):
        condition = audit["conditions"][label]
        raw = json.loads(Path(condition["source"]["path"]).read_text())
        training_runs[label] = raw
        assert sha(condition["source"]["path"]) == condition["source"]["sha256"]
        inputs[label] = condition["source"]
        grouped = defaultdict(list)
        for r in raw["request_records"]:
            family = raw["replica_to_family"][r["model"]]
            key = family+"/"+r["task"]
            frozen = input_service_ms(spec, family, r["task"], X[family], r["sample_id"])
            grouped[key].append((r["worker_latency_ms"], frozen))
        ratios[str(share)] = {}
        residuals[label] = {}
        for key, values in grouped.items():
            ratio = sum(a for a,b in values)/sum(b for a,b in values)
            ratios[str(share)][key] = ratio
            differences = [a-ratio*b for a,b in values]
            residuals[label][key] = {"requests":len(values),"aggregate_service_residual_ms":sum(differences),
                "mean_absolute_request_residual_ms":float(np.mean(np.abs(differences))),
                "request_rmse_ms":float(np.sqrt(np.mean(np.square(differences)))),
                "p95_absolute_request_residual_ms":float(np.percentile(np.abs(differences),95))}
    def correction(key, share):
        if not 0 <= share <= .25:
            raise ValueError("No extrapolation beyond observed split fraction")
        if key == "generalist_2/asr":
            return ratios["0.25"][key]
        alpha = share/.25
        return (1-alpha)*ratios["0.0"][key] + alpha*ratios["0.25"][key]
    training_replay = {}
    for label, share in (("no_split",0.),("split_25pct_G2",.25)):
        raw=training_runs[label]
        service=[]
        for r in raw["request_records"]:
            family=raw["replica_to_family"][r["model"]]
            service.append(input_service_ms(spec,family,r["task"],X[family],r["sample_id"])*correction(family+"/"+r["task"],share))
        replay=fifo(raw,SystemConfig(**raw["config"]),service)
        replay["measured_U"]=raw["evaluation"]["utility"]["U"]
        replay["utility_residual_with_measured_QME"]=replay["terms_with_measured_QME"]["U"]-replay["measured_U"]
        training_replay[label]=replay
    empirical_memory = max(c["measured"]["M_norm"] for c in audit["conditions"].values())
    old_memory = max(data["safe_factor"]*sum(data["memory_table_mib"][m][str(n)] for m,n in X.items())*spec["memory_multiplier"],
        spec["training_deployment_memory"].get(_deployment_key(X),0))/original["gpu_memory_mib"]
    protocol = {"schema":"empirical_spillover_scan_protocol.v1", "status":"DEVELOPMENT_FIT_EXPLORATION_ONLY",
        "audit_source":receipt(audit_path),"development_sources":inputs,"basis":receipt(basis),
        "fixed_deployment":X,"ASR_rps":LOADS,"background_each_rps":BACKGROUNDS,"G2_ASR_fractions":SPLITS,
        "horizon_seconds":600,"seed":51,"arrival_process":"poisson",
        "service_rule":"For each model/task/n, sum(actual worker_ms)/sum(frozen per-input ms), separately at split0 and split.25. Four paths linearly interpolate in split; G2 ASR is held constant at its only observed .25 ratio. Input IDs use frozen pool order shuffled with seed51 before routing.",
        "split_extrapolation_forbidden":True,"workload_transfer_warning":"Only ASR1.8/background.05 observed. Other load/background points are explicitly unsupported workload transfer diagnostics, not interpolation in those dimensions.",
        "correction_ratios":ratios,"aggregate_free_parameters":9,
        "identifiability":"Four paths each have two endpoints (8 parameters); G2 ASR has only one observable endpoint (1 parameter), so its split slope is unidentifiable and not fitted. Aggregate training residuals vanish by construction; request residuals remain.",
        "training_residuals":residuals,"training_FIFO_replay_residuals":training_replay,
        "weights":original["weights"],"latency_references_ms":original["latency_reference_ms"],
        "quality":"Unmodified frozen per-input quality predictor; no measured-output substitution for unobserved candidate",
        "power":"External only: original NNLS coefficients applied to original constant n-state demand features, plus original per-worker startup amortization. Not refitted; not energy evidence.",
        "memory":"Report both old learned estimate and max observed safe memory of the two development runs; latter is local empirical diagnostic, not hardware bound",
        "model_or_production_files_edited":False,"production_U":None,"production_eligible":False,
        "gpu_started":False,"holdout_results_read":False,"overfitting_warning":True,
        "global_search":False,"global_optimality_claim":False}
    save(output/"protocol.json",protocol)
    candidates=[]
    for rate in LOADS:
        for bg in BACKGROUNDS:
            rates=dict(text=bg,image=bg,ocr=bg,asr=rate)
            trace=make_trace(rates,600,51,"poisson")
            ids=replay_input_ids(data,trace,51)
            for share in SPLITS:
                row={"asr":rate*(1-share)}
                if share:row["generalist_2"]=rate*share
                routing=dict(text={"generalist_2":bg},image={"generalist_2":bg},ocr={"ocr":bg},asr=row)
                aliases,routes=expand_routes(X,routing)
                router=Router(routes)
                clocks=defaultdict(float)
                taskstats=defaultdict(list)
                service=defaultdict(list)
                inwindow=late=0
                for (offset,task),sample in zip(trace,ids):
                    alias=router.choose(task)
                    family=aliases[alias]
                    svc=input_service_ms(spec,family,task,X[family],sample)*correction(family+"/"+task,share)
                    finish=max(offset,clocks[alias])+svc/1000
                    clocks[alias]=finish
                    taskstats[task].append((finish-offset)*1000)
                    service[family+"/"+task].append(svc)
                    inwindow+=finish<=600
                    late+=finish>900
                per_task={t:{"mean_ms":float(np.mean(v)),"p95_ms":float(np.percentile(v,95)),"requests":len(v)} for t,v in taskstats.items()}
                demand=math.fsum(q*data["task_replica_profiles"][m][t][str(X[m])]["mean_service_ms"]/1000
                    for t,route in routing.items() for m,q in route.items())
                features=[1.,demand,float(sum(X.values()))]
                watts=sum(a*b for a,b in zip(features,spec["power_coefficients"]))
                E=(watts+spec["startup_joules_per_worker"]*sum(X.values())/original["startup_amortization_seconds"])/original["energy_reference_watts"]
                Q,Qe=predict_quality(data,X,routing,rates,trace,51)
                rho={m:sum(q*replay_path_statistics(spec,m,t,X[m])["mean_ms"]*correction(m+"/"+t,share)/1000
                     for t,route in routing.items() for model,q in route.items() if model==m)/X[m] for m in X}
                scores={}
                if set(per_task)==set(rates):
                    for memory_label,M in (("old_learned_memory",old_memory),("max_observed_development_memory",empirical_memory)):
                        scores[memory_label]=components(per_task,rates,original["latency_reference_ms"],1-inwindow/len(trace),M,original["weights"],E,Q)
                candidate={"candidate_id":f"asr{rate:g}-bg{bg:g}-G2share{share:g}","deployment":X,"routing":routing,
                    "ASR_rps":rate,"background_each_rps":bg,"G2_ASR_fraction":share,
                    "per_task":per_task,"corrected_rho":rho,"max_corrected_rho":max(rho.values()),
                    "offered_requests":len(trace),"end_backlog_requests":len(trace)-inwindow,"incomplete_after_drain":late,
                    "per_path_actual_cohort_service_mean_ms":{k:float(np.mean(v)) for k,v in service.items()},
                    "external_six_term_diagnostics":scores,"quality_prediction_evidence":Qe,
                    "old_external_energy":{"features":features,"watts":watts,"E_norm":E},
                    "local_diagnostic_constraints_pass":bool(scores and max(rho.values())<=.8+1e-8 and not late and old_memory<=1),
                    "workload_dimension_transfer_unvalidated":rate!=1.8 or bg!=.05,
                    "fewer_than_20_requests_tasks":[t for t,p in per_task.items() if p["requests"]<20],
                    "production_U":None,"production_eligible":False,"global_optimality_claim":False}
                candidates.append(candidate)
    summaries=[]
    for rate in LOADS:
        for bg in BACKGROUNDS:
            group=[c for c in candidates if c["ASR_rps"]==rate and c["background_each_rps"]==bg]
            baseline=next(c for c in group if c["G2_ASR_fraction"]==0)
            def score(c):return c["external_six_term_diagnostics"].get("old_learned_memory",{}).get("U")
            feasible=[c for c in group if c["local_diagnostic_constraints_pass"] and score(c) is not None]
            best=max(feasible,key=score,default=None)
            summaries.append({"ASR_rps":rate,"background_each_rps":bg,"baseline_constraints_pass":baseline["local_diagnostic_constraints_pass"],
                "baseline_score":score(baseline),"best_among_three_feasible_diagnostics":best["candidate_id"] if best else None,
                "best_G2_fraction":best["G2_ASR_fraction"] if best else None,"best_score":score(best) if best else None,
                "delta_vs_no_split_if_both_feasible":score(best)-score(baseline) if best and baseline["local_diagnostic_constraints_pass"] else None,
                "scores":[{"share":c["G2_ASR_fraction"],"score":score(c),"rho":c["max_corrected_rho"],"local_constraints_pass":c["local_diagnostic_constraints_pass"]} for c in group],
                "workload_transfer_unvalidated":rate!=1.8 or bg!=.05})
    report={"schema":"empirical_spillover_scan.v1","protocol":protocol,"candidates":candidates,"summaries":summaries,
        "frozen_sources_unchanged":all(sha(p)==h for p,h in source_hashes.items()),"candidate_count":len(candidates),
        "production_U":None,"production_eligible":False,"gpu_started":False,"holdout_results_read":False}
    assert report["frozen_sources_unchanged"] and len(candidates)==45
    save(output/"report.json",report)
    lines=["# 两条开发运行拟合的经验 FIFO：只供减少盲测", "", "固定 ASR2+G2n1+OCR1，seed51/H600，原六权重、参考值与逐输入质量保持。分流份额仅在已观测的 0..25% 内插值；G2 ASR 仅有一个端点观测，其倍率保持常量，不能识别随份额变化的斜率。", "",
        "只有 ASR1.8、背景各.05 属于训练工作负载。其他负载/背景都是未经验证的工作负载迁移诊断，不能被叫做这几个维度的内插预测。9个参数拟合2条运行，聚合训练残差按构造为0，逐请求残差不为0，明显可能过拟合。所有候选 production_U=null、production_eligible=false；不进入正式搜索、验证、最优性证书。", "",
        "| ASR | 背景各 | 三个份额内局部可行最佳G2份额 | 诊断分数 | 不分流局部可行 | 双方可行ΔU |", "|---:|---:|---:|---:|---|---:|"]
    for s in summaries:
        fmt=lambda v:"UNKNOWN" if v is None else f"{v:.6f}"
        lines.append(f"| {s['ASR_rps']:g} | {s['background_each_rps']:g} | {fmt(s['best_G2_fraction'])} | {fmt(s['best_score'])} | {s['baseline_constraints_pass']} | {fmt(s['delta_vs_no_split_if_both_feasible'])} |")
    lines += ["", "局部可行仅指经验服务下 rho≤.8、旧预测显存和模型排空检查；实际稳定性/能耗/质量/显存均未验证。若不分流已不满足模型rho，该条件只能称容量救援候选，不能称双方可行时分流总效用优势。", "",
        "训练运行中，将每路径聚合服务总量拟合准确不等于复现延迟分布。保留实测Q/M/E时的经验FIFO训练效用残差："+
        "; ".join(f"{k}: {v['utility_residual_with_measured_QME']:.9f}" for k,v in training_replay.items())+"。这部分残差会限制预测微小正收益的可信度。", "",
        "背景 .005/.01 的 H600 样本很少，p95与逐输入质量会受有限样本影响；每候选保存少于20请求的任务。为了减少盲测，优先使用已预声明的 ASR1.8/1.9、12.5%小分流开发点检查模型弯曲和工作负载迁移，再依据其实际路径残差决定是否新增轻背景测量。", "",
        "不能仅选择本表中正收益最大的一行做结论；至少保留同条件不分流和小分流配对，另用新种子独立验证。61/62与71/72/73结果均未读取。本扫描不改变已冻结实验、生产预测器或门禁。", ""]
    (output/"README.zh.md").write_text("\n".join(lines))
    save(output/"manifest.json",{"script":receipt(__file__),"files":{p.name:sha(p) for p in output.iterdir() if p.is_file()},"gpu_started":False,"holdout_results_read":False})
    print(json.dumps({"output":str(output),"summaries":summaries},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
