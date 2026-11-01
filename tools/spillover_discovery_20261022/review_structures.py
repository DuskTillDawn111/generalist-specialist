#!/usr/bin/env python3
"""Audit external structure scans and summarize predeclared strong families."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gsdeploy.prediction import predict
from gsdeploy.quality import quality_digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories",type=Path,nargs="+")
    args=parser.parse_args()
    for base in args.directories:
        report=json.loads((base/"report.json").read_text())
        records=json.loads((base/"evaluated-candidates.json").read_text())["candidates"]
        memory=json.loads((base/"deployment-memory-inspection.json").read_text())
        source=Path(report["protocol"]["source_config"]["path"])
        assert sha(source)==report["protocol"]["source_config"]["sha256"]
        config=json.loads(source.read_text());old=config["catalog"]["calibrated_replay_v2"]
        data=copy.deepcopy(config["catalog"])
        for k in ("calibrated_replay_v2","calibrated_surrogate","energy_profiles","colocation_profiles"):
            data.pop(k,None)
        data["performance_mode"],data["memory_mode"]="task_replica_proxy","standalone_proxy"
        rates=report["workload"];budget=config["gpu_memory_mib"];weights=config["weights"]
        def simplify(c):
            if c is None:return None
            return {k:c[k] for k in ("deployment","routing","raw_exploratory_score","transferred_exploratory_score",
                "raw_safe_memory_mib","transferred_memory_mib","transferred_memory_feasible",
                "asr_specialist_fraction","target_asr_n_ge_2_and_split","max_rho")}
        predicates={
            "ASR2_G2n2_optional_background_specialists":lambda c:c["deployment"].get("asr")==2 and c["deployment"].get("generalist_2")==2,
            "ASR1_G2n2_OCR1":lambda c:c["deployment"]==dict(asr=1,generalist_2=2,ocr=1),
            "ASR3_any_background":lambda c:c["deployment"].get("asr")==3,
            "G2_G3_combinations":lambda c:c["deployment"].get("generalist_2",0)>0 and c["deployment"].get("generalist_3",0)>0,
            "G2_G3_ASR_without_ASR_specialist":lambda c:not c["deployment"].get("asr") and c["deployment"].get("generalist_2",0)>0 and c["deployment"].get("generalist_3",0)>0,
            "OCR_wholly_on_generalist":lambda c:all(data["models"][m]["role"]=="generalist" for m in c["routing"]["ocr"]),
            "image_specialist_used":lambda c:c["routing"]["image"].get("image",0)>0,
            "pure_specialists":lambda c:all(data["models"][m]["role"]=="specialist" for m in c["deployment"]),
            "all_ASR_no_split":lambda c:c["is_no_split"],
        }
        groups={};witnesses=[]
        for name,condition in predicates.items():
            members=[c for c in records if condition(c)]
            best={}
            for mode,field in (("raw_safe_sum","raw_exploratory_score"),("old_learned_multiplier","transferred_exploratory_score")):
                result=max((c for c in members if c[field] is not None),key=lambda c:c[field],default=None)
                best[mode]=simplify(result)
                if result:witnesses.append(result)
            groups[name]={"evaluated_route_count":len(members),"best_by_memory_interpretation":best}
        explicit=[]
        for n in (2,3,4):
            x=dict(asr=n,text=1,image=1,ocr=1)
            raw=data["safe_factor"]*math.fsum(data["memory_table_mib"][m][str(k)] for m,k in x.items())
            transfer=max(raw*old["memory_multiplier"],old["training_deployment_memory"].get(quality_digest(x),0))
            explicit.append(dict(deployment=x,raw_safe_memory_mib=raw,transferred_memory_mib=transfer,
                raw_memory_feasible=raw<=budget,transferred_memory_feasible=transfer<=budget,
                note="Background text/image/OCR each resident specialist; no generalist. Raw-memory infeasible deployments have no routing score."))
        # Independent scalar recomputation checks every evaluated candidate.
        for c in records:
            assert c["production_U"] is None and c["production_E"] is None
            assert c["max_rho"]<=.8+1e-8
            for task,row in c["routing"].items():assert math.isclose(sum(row.values()),rates[task],abs_tol=1e-12)
            raw=data["safe_factor"]*math.fsum(data["memory_table_mib"][m][str(n)] for m,n in c["deployment"].items())
            assert math.isclose(raw,c["raw_safe_memory_mib"],abs_tol=1e-9)
            transfer=max(raw*old["memory_multiplier"],old["training_deployment_memory"].get(quality_digest(c["deployment"]),0))
            assert math.isclose(transfer,c["transferred_memory_mib"],abs_tol=1e-9)
            demand=math.fsum(q*data["task_replica_profiles"][m][t][str(c["deployment"][m])]["mean_service_ms"]/1000
                            for t,row in c["routing"].items() for m,q in row.items())
            watts=math.fsum(a*b for a,b in zip([1,demand,sum(c["deployment"].values())],old["power_coefficients"]))
            energy=(watts+old["startup_joules_per_worker"]*sum(c["deployment"].values())/config["startup_amortization_seconds"])/config["energy_reference_watts"]
            assert math.isclose(energy,c["raw_external_terms"]["E"],abs_tol=1e-12)
            for field,terms in (("raw_exploratory_score","raw_external_terms"),("transferred_exploratory_score","transferred_external_terms")):
                if c[field] is not None:assert math.isclose(-math.fsum(weights[k]*v for k,v in c[terms].items()),c[field],abs_tol=1e-12)
            assert (c["transferred_exploratory_score"] is None)==(c["incomplete_after_drain"]>0 or transfer>budget)
        # Include top target and every strong family's best actual route in the
        # slower public evaluator, without supplying fabricated energy.
        witnesses.extend(report["summaries"][mode]["top20"][0] for mode in report["summaries"])
        checked=[];seen=set()
        for c in witnesses:
            key=quality_digest({"x":c["deployment"],"r":c["routing"]})
            if key in seen:continue
            seen.add(key)
            p=predict(data,c["deployment"],c["routing"],rates,budget,config["latency_reference_ms"],weights,
                horizon=config["horizon_seconds"],seed=config["random_seed"],arrival_process=config["arrival_process"],
                drain_seconds=config["drain_seconds"],energy_reference_watts=config["energy_reference_watts"],
                startup_amortization_seconds=config["startup_amortization_seconds"])
            differences={k:c["raw_external_terms"][k]-p[k+"_norm"] for k in "QTLPM"}
            assert all(abs(v)<1e-10 for v in differences.values()),differences
            assert p["U"] is None and p["E_norm"] is None
            checked.append(dict(deployment=c["deployment"],routing=c["routing"],differences=differences))
        audit=dict(status="PASS",source_report_sha256=sha(base/"report.json"),all_candidates_scalar_checked=len(records),
            public_predictor_additional_cross_checks=len(checked),cross_checks=checked,
            all_declared_finite_candidates_evaluated=report["all_declared_candidates_evaluated"],
            global_optimality_certified=False,production_U_and_E_remain_null=True,gpu_started=False)
        review=dict(status=report["status"],workload=rates,groups=groups,explicit_pure_specialist_memory_exclusions=explicit,
            audit=audit,scope="Strong-family rankings within declared finite routes; no full rational-grid or physical optimum.")
        save(base/"strong-alternative-review.json",review)
        lines=["# 强替代分组与独立复核","",report["status"],"",
            "两套显存解释保留不同可行域与外部分数。空值表示该组在相应解释下无已评可行候选。",
            "", "| 组 | 路由数 | 原始safe最佳 | raw分数 | 旧显存系数最佳 | learned分数 |",
            "|---|---:|---|---:|---|---:|"]
        def score(c,f):return "UNKNOWN" if c is None else f"{c[f]:.9f}"
        for name,g in groups.items():
            a=g["best_by_memory_interpretation"]["raw_safe_sum"];b=g["best_by_memory_interpretation"]["old_learned_multiplier"]
            lines.append(f"| {name} | {g['evaluated_route_count']} | {json.dumps(a['deployment']) if a else 'NONE'} | {score(a,'raw_exploratory_score')} | {json.dumps(b['deployment']) if b else 'NONE'} | {score(b,'transferred_exploratory_score')} |")
        lines.extend(["", "## ASR 扩容且背景全专用", "", "| 部署 | 原始安全MiB | old迁移MiB | raw可行 | learned可行 |",
                      "|---|---:|---:|---|---|"])
        for e in explicit:lines.append(f"| {json.dumps(e['deployment'])} | {e['raw_safe_memory_mib']:.3f} | {e['transferred_memory_mib']:.3f} | {e['raw_memory_feasible']} | {e['transferred_memory_feasible']} |")
        lines.extend(["",f"已复核全部 {len(records)} 候选的流量守恒、双显存、能耗外推及加权分数算术；另对 {len(checked)} 个强组最佳方案以公开预测器复核Q/T/L/P/M，误差均<1e-10，生产E/U保持空值。",
            "本次有限域全部已评，但背景仍只允许整任务路由，ASR只允许两模型族与所列分数，服务仍为画像常量均值，不能声称全池自由赢家或新逐输入迁移模型最优。",""])
        (base/"strong-alternative-review.zh.md").write_text("\n".join(lines))
        # Ensure each displayed Markdown table has a header after separating
        # prose. This is formatting only; numerical source reports stay intact.
        readme=base/"README.zh.md";s=readme.read_text()
        marker="| old_learned_multiplier | 1 |"
        header="| 显存解释 | 排名 | 部署 | ASR路由rps | 外部探索分数 | 原始安全MiB | 旧迁移MiB | 目标结构 |\n|---|---:|---|---|---:|---:|---:|---|\n"
        if marker in s and header+marker not in s:s=s.replace(marker,header+marker,1)
        s+="\n强替代各组最佳与显存排除见 [分组复核](strong-alternative-review.zh.md)，原始路径与完整路由见对应JSON。\n"
        readme.write_text(s)
        manifest=json.loads((base/"manifest.json").read_text())
        manifest["files"]={p.name:sha(p) for p in base.iterdir() if p.is_file() and p.name!="manifest.json"}
        manifest["review_script"]=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))
        save(base/"manifest.json",manifest)
        print(json.dumps(dict(directory=str(base),scalar_checks=len(records),public_predictor_cross_checks=len(checked),status="PASS")))


if __name__=="__main__":main()
