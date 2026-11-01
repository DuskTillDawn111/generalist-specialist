#!/usr/bin/env python3
"""CPU-only audit of exactly five development points; missing files stay INCOMPLETE.

Exit codes: 0 complete audit, 2 incomplete inputs, 3 strict import/contract failure.
Never reads transfer validation or final confirmation output directories.
"""
import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_development_reversal import ROOT, STUDY, SOURCE, receipt, save, sha, stats, fifo, pathkey
from gsdeploy.calibration import _import_source, _baseline_identity
from gsdeploy.calibration_v2 import input_service_ms, replay_input_ids
from gsdeploy.config import SystemConfig
from gsdeploy.prediction import components
from gsdeploy.quality import quality_digest
from gsdeploy.replay_transfer import SCORING, prepare_transfer, predict_validation

X = {"asr": 2, "generalist_2": 1, "ocr": 1}
ROSTER = (
    ("asr1p8_G2share0", 1.8, 0., "initial_training_endpoint", "first", "experiment-packet.json", "runs", "asr18_n2_specialist_share_1p0"),
    ("asr1p8_G2share0p25", 1.8, .25, "initial_training_endpoint", "first", "experiment-packet.json", "runs", "asr18_n2_specialist_share_0p75"),
    ("asr1p8_G2share0p125", 1.8, .125, "development_interpolation_check_not_independent_validation", "small", "experiment-packet-asr1p8.json", "runs-asr1p8", "asr1p8_specialist_share_0p875"),
    ("asr1p9_G2share0", 1.9, 0., "development_workload_transfer_check_not_independent_validation", "small", "experiment-packet-asr1p9.json", "runs-asr1p9", "asr1p9_specialist_share_1p0"),
    ("asr1p9_G2share0p125", 1.9, .125, "development_workload_transfer_check_not_independent_validation", "small", "experiment-packet-asr1p9.json", "runs-asr1p9", "asr1p9_specialist_share_0p875"),
)


def diagnostic_terms(value):
    """Separate arithmetic from formal prediction eligibility explicitly."""
    result = dict(value)
    score = result.pop("U", None)
    result["diagnostic_score"] = result.get("diagnostic_partial_U", score)
    result["production_U"] = None
    result["production_eligible"] = False
    return result


def errors(predicted, measured):
    score = predicted.get("diagnostic_score")
    result = {}
    for key in ["U"] + [k+"_norm" for k in "QTLPME"]:
        a = score if key == "U" else predicted.get(key)
        b = measured.get(key)
        result[key] = {"signed": None if a is None or b is None else a-b,
            "absolute": None if a is None or b is None else abs(a-b),
            "relative_to_measured": None if a is None or b is None else abs(a-b)/max(abs(b),1e-9)}
    return result


def configure_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-development", type=Path, default=STUDY/"development-asr18-v1")
    parser.add_argument("--small-development", type=Path, default=STUDY/"development-small-spillover-v2")
    parser.add_argument("--basis-config", type=Path, default=ROOT/"artifacts/static-study-gpu-20261022/surrogate-v2/system-config.json")
    parser.add_argument("--output", type=Path, help="New output directory; never overwrite prior audits")
    parser.add_argument("--check-only", action="store_true", help="Check explicit input paths only; do not read raw contents")
    return parser


def availability(args):
    entries = []
    for key, rate, fraction, role, location, packet, runs, label in ROSTER:
        base = args.first_development if location == "first" else args.small_development
        entries.append({"id":key,"asr_rps":rate,"G2_ASR_fraction":fraction,"role":role,
            "packet_path":str((base/packet).resolve()),
            "run_path":str((base/runs/(label+"-attempt1")/"run.json").resolve())})
    paths = set([str(args.basis_config.resolve()), str(SOURCE/"manifest.json")])
    for entry in entries:
        paths.update((entry["packet_path"], entry["run_path"]))
    for path in paths:
        if any(part.startswith(("transfer-validation", "confirmation", "final-confirmation")) for part in Path(path).parts):
            raise ValueError("Holdout directories are forbidden")
    return entries, sorted(path for path in paths if not Path(path).is_file())


def run_audit(args, entries):
    code_before = {str(p):sha(p) for p in SOURCE.rglob("*.py")}
    raw_by_id, condition_by_id = {}, {}
    imported_receipts=[]
    for entry in entries:
        packet = json.loads(Path(entry["packet_path"]).read_text())
        expected = next((j for j in packet["jobs"] if j["labels"] == [Path(entry["run_path"]).parent.name.removesuffix("-attempt1")]),None)
        if expected is None:
            raise ValueError("Declared job missing from original packet: "+entry["id"])
        source = {"id":entry["id"],"path":entry["run_path"],"sha256":sha(entry["run_path"]),
            "packet_path":entry["packet_path"],"packet_sha256":sha(entry["packet_path"])}
        raw,measured,strict_receipt = _import_source(source,{expected["identity"]["runtime_code_sha256"]})
        rates=dict(text=.05,image=.05,ocr=.05,asr=entry["asr_rps"])
        fraction=raw["plan"]["routing"]["asr"].get("generalist_2",0)/entry["asr_rps"]
        if (raw["experiment_job_id"]!=expected["id"] or raw["config"]["workload"]!=rates
                or raw["plan"]["deployment"]!=X or not math.isclose(fraction,entry["G2_ASR_fraction"],abs_tol=1e-12)
                or raw["transport_config"]["random_seed"]!=51 or raw["measurement_window"]["duration_seconds"]!=600
                or raw.get("experiment_phase")!="calibration"
                or expected.get("purpose")!="new_load_exploratory_development_not_final_confirmation"):
            raise ValueError("Raw differs from predeclared five-point roster: "+entry["id"])
        raw_by_id[entry["id"]]=raw
        condition_by_id[entry["id"]]={**entry,"strict_import_passed":True,"source":source,
            "executed_measurement_id":raw["executed_measurement_id"],"workload":rates,"plan":raw["plan"],
            "measured_six_term_utility":measured,"stability":raw["evaluation"]["stability"],
            "measurement_window":raw["measurement_window"]}
        imported_receipts.append(strict_receipt)
    assert len({r["executed_measurement_id"] for r in raw_by_id.values()})==len(entries)
    # Training is fixed to first two conditions. The three new conditions never fit any coefficient.
    basis=json.loads(args.basis_config.read_text())
    spec=basis["catalog"]["calibrated_replay_v2"]
    ratios={}
    training_residuals={}
    for key in ("asr1p8_G2share0","asr1p8_G2share0p25"):
        raw=raw_by_id[key]
        grouped=defaultdict(list)
        for r in raw["request_records"]:
            model=raw["replica_to_family"][r["model"]]
            baseline=input_service_ms(spec,model,r["task"],X[model],r["sample_id"])
            grouped[pathkey(raw,r)].append((r["worker_latency_ms"],baseline))
        ratios[key]={p:sum(a for a,b in values)/sum(b for a,b in values) for p,values in grouped.items()}
        training_residuals[key]={p:{"requests":len(values),
            "sum_residual_ms":sum(a-ratios[key][p]*b for a,b in values),
            "request_RMSE_ms":float(np.sqrt(np.mean([(a-ratios[key][p]*b)**2 for a,b in values])))} for p,values in grouped.items()}
    def correction(path,share):
        if not 0<=share<=.25:
            raise ValueError("No split extrapolation permitted")
        if path=="generalist_2/asr":
            return ratios["asr1p8_G2share0p25"][path]
        a=share/.25
        return (1-a)*ratios["asr1p8_G2share0"][path]+a*ratios["asr1p8_G2share0p25"][path]
    for entry in entries:
        key=entry["id"]
        raw=raw_by_id[key]
        c=condition_by_id[key]
        config=SystemConfig(**prepare_transfer(args.basis_config,c["workload"]))
        native=raw["config"]
        if (any(native.get(k)!=getattr(config,k,None) for k in SCORING)
                or _baseline_identity(native["catalog"])!=_baseline_identity(config.catalog)
                or quality_digest(native["catalog"]["quality"])!=quality_digest(config.catalog["quality"])
                or native["catalog"]["safe_factor"]!=config.catalog["safe_factor"]
                or native["catalog"]["measurement_context"]["gpu_uuid"]!=config.catalog["measurement_context"]["gpu_uuid"]
                or any(native["catalog"]["measurement_context"]["runtime_checkpoint_sha256"].get(m)!=config.catalog["measurement_context"]["runtime_checkpoint_sha256"].get(m) for m in X)):
            raise ValueError("Development scoring/baseline/checkpoint context differs from frozen basis")
        pred=predict_validation(config,X,raw["plan"]["routing"],horizon=600,seed=51)
        assert pred.get("U") is None and pred.get("eligible") is not True
        c["frozen_zero_refit_diagnostic_prediction"]=diagnostic_terms(pred)
        c["frozen_prediction_errors"]=errors(c["frozen_zero_refit_diagnostic_prediction"],c["measured_six_term_utility"])
        trace=make_trace_for(config)
        assert replay_input_ids(config.catalog,trace,51)==[r["sample_id"] for r in raw["request_records"]]
        frozen_services=[]
        empirical_services=[]
        path_groups=defaultdict(list)
        for r in raw["request_records"]:
            model=raw["replica_to_family"][r["model"]]
            frozen=input_service_ms(spec,model,r["task"],X[model],r["sample_id"])
            empirical=frozen*correction(pathkey(raw,r),entry["G2_ASR_fraction"])
            frozen_services.append(frozen)
            empirical_services.append(empirical)
            path_groups[pathkey(raw,r)].append((r,frozen,empirical))
        actual_services=[r["worker_latency_ms"] for r in raw["request_records"]]
        c["service_errors"]={}
        for model_name,services in (("frozen_zero_refit",frozen_services),("two_endpoint_9_parameter",empirical_services)):
            c["service_errors"][model_name]={"actual_total_ms":sum(actual_services),"predicted_total_ms":sum(services),
                "total_relative_error":abs(sum(services)-sum(actual_services))/sum(actual_services),
                "request_RMSE_ms":float(np.sqrt(np.mean([(a-b)**2 for a,b in zip(services,actual_services)])))}
        c["per_path"]={p:{"requests":len(values),"unique_inputs":len({r["sample_id"] for r,f,e in values}),
            "actual_service_ms":stats([r["worker_latency_ms"] for r,f,e in values]),
            "frozen_service_ms":stats([f for r,f,e in values]),"empirical_service_ms":stats([e for r,f,e in values]),
            "actual_over_frozen_total":sum(r["worker_latency_ms"] for r,f,e in values)/sum(f for r,f,e in values),
            "wait_ms":stats([r["worker_queue_wait_ms"] for r,f,e in values]),
            "e2e_ms":stats([r["e2e_latency_ms"] for r,f,e in values])} for p,values in path_groups.items()}
        actual_task_groups={t:[r for r in raw["request_records"] if r["task"]==t] for t in config.workload}
        c["per_task"]={t:{"offered_requests":len(records),
            "mean_e2e_latency_ms":stats([r["e2e_latency_ms"] for r in records])["mean"],
            "p95_e2e_latency_ms":stats([r["e2e_latency_ms"] for r in records])["p95"],
            "mean_worker_latency_ms":stats([r["worker_latency_ms"] for r in records])["mean"],
            "p95_worker_latency_ms":stats([r["worker_latency_ms"] for r in records])["p95"],
            "mean_worker_queue_wait_ms":stats([r["worker_queue_wait_ms"] for r in records])["mean"],
            "p95_worker_queue_wait_ms":stats([r["worker_queue_wait_ms"] for r in records])["p95"],
            "completed_within_window":sum(r["completion_time_ns"]<=raw["measurement_window"]["end_time_ns"] for r in records),
            "completed_during_drain":sum(r["completion_time_ns"]>raw["measurement_window"]["end_time_ns"] for r in records),
            "actual_quality":c["measured_six_term_utility"]["quality_evidence"]["per_task"][t],
            "frozen_prediction":pred["per_task"][t]} for t,records in actual_task_groups.items()}
        # fifo uses cached QME only as diagnostics; replace with strict recomputed values on a local copy.
        replay_raw={**raw,"evaluation":{**raw["evaluation"],"utility":c["measured_six_term_utility"]}}
        replays={}
        for name,services in (("frozen_service_measured_QME",frozen_services),
                              ("empirical_9parameter_measured_QME",empirical_services),
                              ("actual_service_measured_QME",actual_services)):
            replay=fifo(replay_raw,config,services)
            replay["arithmetic_diagnostic"]=diagnostic_terms(replay.pop("terms_with_measured_QME"))
            replay["prediction_errors"]=errors(replay["arithmetic_diagnostic"],c["measured_six_term_utility"])
            replays[name]=replay
        c["FIFO_diagnostics_with_measured_QME"]=replays
        emp=replays["empirical_9parameter_measured_QME"]
        # Full external diagnostic keeps original frozen Q/M/E, so QME observations do not leak into prediction.
        T=emp["arithmetic_diagnostic"]["T_norm"]
        emp_terms=components(emp["per_task"],config.workload,config.latency_reference_ms,T,
            pred["M_norm"],config.weights,pred["E_norm"],pred["Q_norm"])
        c["empirical_9parameter_full_external_diagnostic"]={**diagnostic_terms(emp_terms),
            "per_task":emp["per_task"],"scope":"Development-fitted service; original frozen Q/M/E; no formal eligibility or new-model validation"}
        c["empirical_9parameter_errors"]=errors(c["empirical_9parameter_full_external_diagnostic"],c["measured_six_term_utility"])
    comparisons=[]
    for a,b in itertools.combinations(entries,2):
        if a["asr_rps"]!=b["asr_rps"]:
            continue
        ra,rb=raw_by_id[a["id"]],raw_by_id[b["id"]]
        ca,cb=condition_by_id[a["id"]],condition_by_id[b["id"]]
        records_a,records_b=ra["request_records"],rb["request_records"]
        paired=len(records_a)==len(records_b) and all((x["task"],x["sample_id"],x["scheduled_offset_seconds"])==(y["task"],y["sample_id"],y["scheduled_offset_seconds"]) for x,y in zip(records_a,records_b))
        if not paired:
            raise ValueError("Same-load development comparison is not paired")
        measured_delta=cb["measured_six_term_utility"]["U"]-ca["measured_six_term_utility"]["U"]
        compare={"a":a["id"],"b":b["id"],"direction":"b minus a", "asr_rps":a["asr_rps"],
            "same_ordered_task_sample_schedule":True,"measured_delta_U":measured_delta,
            "delta_six_contributions":{k:cb["measured_six_term_utility"][k+"_contribution"]-ca["measured_six_term_utility"][k+"_contribution"] for k in "QTLPME"},
            "task_mean_latency_delta_ms":{t:cb["per_task"][t]["mean_e2e_latency_ms"]-ca["per_task"][t]["mean_e2e_latency_ms"] for t in ca["per_task"]},
            "task_p95_latency_delta_ms":{t:cb["per_task"][t]["p95_e2e_latency_ms"]-ca["per_task"][t]["p95_e2e_latency_ms"] for t in ca["per_task"]},
            "scope":"Single-seed paired development effect; no significance or physical optimum claim"}
        for name,field in (("frozen","frozen_zero_refit_diagnostic_prediction"),("empirical_9parameter","empirical_9parameter_full_external_diagnostic")):
            delta=cb[field]["diagnostic_score"]-ca[field]["diagnostic_score"]
            compare[name]={"diagnostic_delta_U":delta,"delta_error":delta-measured_delta,
                "ordering_consistent":delta*measured_delta>=0}
        comparisons.append(compare)
    expected_pairs=sum(a["asr_rps"]==b["asr_rps"] for a,b in itertools.combinations(entries,2))
    assert len(comparisons)==expected_pairs
    assert all(sha(p)==h for p,h in code_before.items())
    return {"schema":"spillover_five_development_point_audit.v1",
        "status":"COMPLETE_DEVELOPMENT_AUDIT" if len(entries)==5 else "PARTIAL_INTERNAL_DIAGNOSTIC",
        "audited_condition_count":len(entries),
        "scope":f"{len(entries)} seed51 development conditions; no independent validation claim; no holdout results read",
        "basis":receipt(args.basis_config),"frozen_source":receipt(SOURCE/"manifest.json"),
        "strict_import_receipts":imported_receipts,"conditions":[condition_by_id[e["id"]] for e in entries],
        "same_load_paired_comparisons":comparisons,"training_model":{
            "training_ids":["asr1p8_G2share0","asr1p8_G2share0p25"],"parameters":9,"ratios":ratios,
            "rule":"Four shared path ratios linearly interpolate in G2 split0..25%; G2/asr constant at its only training endpoint; no refit on new three points",
            "training_residuals":training_residuals,"finer_per_input_correction_fitted":False,
            "new_condition_roles":"ASR1.8 split.125 checks interpolation; ASR1.9 pair checks workload transfer. All remain development, not independent holdouts."},
        "parameters_refitted_on_new_three_points":False,"frozen_sources_unchanged":True,
        "production_U":None,"production_eligible":False,"gpu_started":False,
        "validation_results_read":False,"confirmation_results_read":False,"global_optimality_claim":False}


def make_trace_for(config):
    from gsdeploy.trace import make_trace
    return make_trace(config.workload,600,51,config.arrival_process)


def markdown(report):
    lines=["# 五开发点只读审计", "", f"状态：{report['status']}", ""]
    if report["status"]!="COMPLETE_DEVELOPMENT_AUDIT":
        return "\n".join(lines+["输入不齐或严格导入失败，不补造观测、不拟合新参数、不输出完成结论。", "", "```json",json.dumps(report,ensure_ascii=False,indent=2),"```",""])
    lines += ["初始 ASR1.8 的0/25%分流两点固定拟合9参数；1.8/12.5%检查插值，1.9的配对检查工作负载迁移。后三点未参与拟合。全部是开发证据，没有读取61/62或最终确认结果。", "",
        "| 条件 | 角色 | 实测U | 冻结诊断U | 9参数外部诊断U | 冻结U绝对误差 | 9参数U绝对误差 |", "|---|---|---:|---:|---:|---:|---:|"]
    for c in report["conditions"]:
        lines.append(f"| {c['id']} | {c['role']} | {c['measured_six_term_utility']['U']:.9f} | {c['frozen_zero_refit_diagnostic_prediction']['diagnostic_score']:.9f} | {c['empirical_9parameter_full_external_diagnostic']['diagnostic_score']:.9f} | {c['frozen_prediction_errors']['U']['absolute']:.9f} | {c['empirical_9parameter_errors']['U']['absolute']:.9f} |")
    lines += ["", "| 配对方向 b−a | 实测ΔU | 冻结诊断ΔU | 9参数诊断ΔU | 9参数排序一致 |", "|---|---:|---:|---:|---|"]
    for c in report["same_load_paired_comparisons"]:
        lines.append(f"| {c['b']} − {c['a']} | {c['measured_delta_U']:.9f} | {c['frozen']['diagnostic_delta_U']:.9f} | {c['empirical_9parameter']['diagnostic_delta_U']:.9f} | {c['empirical_9parameter']['ordering_consistent']} |")
    lines += ["", "完整负载、路由、六项成本/贡献、逐任务服务/等待/均值/p95、路径服务偏差及逐项预测误差见 audit.json。保留实测Q/M/E的FIFO结果仅用于隔离服务/排队误差；完整外部诊断使用冻结预测Q/M/E，不能把前者称作无泄漏的预测。", "",
        "全部诊断 production_U=null、production_eligible=false；9参数服务模型只拟合初始两条，未拟合15参数或更细逐输入校正。单种子开发配对不证明统计优势、全池最优或独立验证通过。", ""]
    return "\n".join(lines)


def main():
    parser=configure_parser()
    args=parser.parse_args()
    if args.output is None and not args.check_only:
        parser.error("--output is required unless --check-only")
    entries,missing=availability(args)
    report={"schema":"spillover_five_development_point_audit.v1","status":"INCOMPLETE" if missing else "FILES_PRESENT_NOT_AUDITED",
        "expected_conditions":entries,"missing_files":missing,"gpu_started":False,"raw_contents_read":False,
        "validation_results_read":False,"confirmation_results_read":False,"production_U":None}
    if args.check_only:
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 2 if missing else 0
    args.output.mkdir(parents=True,exist_ok=False)
    code=2 if missing else 0
    if not missing:
        try:
            report=run_audit(args,entries)
        except Exception as exc:
            report={**report,"status":"FAILED_STRICT_IMPORT_OR_CONTRACT","error_type":type(exc).__name__,"error":str(exc),"raw_contents_read":True}
            code=3
    save(args.output/"audit.json",report)
    (args.output/"README.zh.md").write_text(markdown(report))
    save(args.output/"manifest.json",{"script":receipt(__file__),"helper":receipt(Path(__file__).with_name("audit_development_reversal.py")),
        "files":{p.name:sha(p) for p in args.output.iterdir() if p.is_file()},"gpu_started":False,"validation_results_read":False})
    print(json.dumps({"status":report["status"],"output":str(args.output.resolve()),"missing_files":missing},ensure_ascii=False))
    return code


if __name__=="__main__":
    sys.exit(main())
