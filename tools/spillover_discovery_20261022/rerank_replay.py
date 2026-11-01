#!/usr/bin/env python3
"""Temporary unvalidated per-input replay rerank of listed strong candidates."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gsdeploy.config import SystemConfig
from gsdeploy.prediction import predict
from gsdeploy.quality import quality_digest
from gsdeploy.replay_transfer import prepare_transfer,predict_validation


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--seconds",type=float,default=110.)
    args=parser.parse_args()
    if not 0<args.seconds<=115:parser.error("Budget must be <=115 seconds")
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    previous=json.loads((args.structures/"report.json").read_text())
    groups=json.loads((args.structures/"strong-alternative-review.json").read_text())
    source=Path(previous["protocol"]["source_config"]["path"])
    rates=previous["workload"]
    # This mutable development configuration is intentionally temporary; it
    # is not the finally frozen transfer envelope or a validation protocol.
    raw=prepare_transfer(source,rates)
    config=SystemConfig(**raw)
    bindings=raw["catalog"]["replay_transfer_v1"]["code_bindings"]
    selected={}
    def add(c,label):
        plan={"deployment":c["deployment"],"routing":c["routing"]}
        key=quality_digest(plan)
        if key not in selected:selected[key]={"plan":plan,"labels":[],"previous_constant_mean_scores":{}}
        selected[key]["labels"].append(label)
        for field in ("raw_exploratory_score","transferred_exploratory_score"):
            if field in c:selected[key]["previous_constant_mean_scores"][field]=c[field]
    for mode,s in previous["summaries"].items():
        for i,c in enumerate(s["top20"],1):add(c,f"{mode}_top{i}")
        if s["best_no_split"]:add(s["best_no_split"],f"{mode}_best_no_split")
    for name,g in groups["groups"].items():
        for mode,c in g["best_by_memory_interpretation"].items():
            if c:add(c,f"strong_group_{name}_{mode}")
    # Three same-deployment routes are declared before this replay stage.
    for fraction in (1.,.75,2/3):
        row={"asr":rates["asr"]*fraction}
        if fraction<1:row["generalist_2"]=rates["asr"]*(1-fraction)
        add(dict(deployment=dict(asr=2,generalist_2=1,ocr=1),routing=dict(
            text={"generalist_2":.05},image={"generalist_2":.05},ocr={"ocr":.05},asr=row)),f"explicit_target_share_{fraction:.6f}")
    for n in (2,3,4):
        add(dict(deployment=dict(asr=n,text=1,image=1,ocr=1),routing={t:{t:q} for t,q in rates.items()}),f"pure_specialist_expansion_asr{n}")
    protocol=dict(status="TRANSFER_UNVALIDATED_DEVELOPMENT_DIAGNOSTIC",workload=rates,
        seed=51,horizon_seconds=1800,seconds_budget=args.seconds,selected_candidates=len(selected),
        selection_sources=[dict(path=str(args.structures/"report.json"),sha256=sha(args.structures/"report.json")),
                           dict(path=str(args.structures/"strong-alternative-review.json"),sha256=sha(args.structures/"strong-alternative-review.json"))],
        previous_model="Constant measured n-state mean FIFO + external old coefficients, seed42",
        current_model="Public replay_transfer_v1 diagnostic: per-input service medians * frozen old service multiplier, old learned memory/power, seed51",
        comparison_warning="Both service model and seed differ; ranking changes cannot be causally attributed to only one factor.",
        parameters_refitted=False,final_transfer_envelope_frozen=False,temporary_envelope_version=raw["catalog"]["replay_transfer_v1"]["version_sha256"],
        temporary_code_bindings=bindings,gpu_started=False,physical_global_optimality_certified=False)
    save(args.output/"protocol.json",protocol)
    results=[]
    for key,item in selected.items():
        if time.perf_counter()-started>=args.seconds:break
        c=item["plan"]
        p=predict_validation(config,c["deployment"],c["routing"],horizon=1800,seed=51)
        assert p.get("U") is None and p.get("eligible") is not True
        accepted=p.get("constraints_eligible",False) and p.get("utility_complete",False)
        results.append(dict(candidate_key=key,**item,diagnostic_score=p.get("diagnostic_partial_U") if accepted else None,
            constraint_eligible=p.get("constraints_eligible"),utility_complete=p.get("utility_complete"),
            diagnostic_prediction=p,asr_specialist_fraction=c["routing"]["asr"].get("asr",0)/rates["asr"],
            target_structure=c["deployment"].get("asr",0)>=2 and 0<c["routing"]["asr"].get("asr",0)<rates["asr"],
            asr_no_split=len(c["routing"]["asr"])==1))
        if len(results)%10==0:print(json.dumps(dict(evaluated=len(results),selected=len(selected),seconds=time.perf_counter()-started)),flush=True)
    ranking=sorted((r for r in results if r["diagnostic_score"] is not None),key=lambda r:r["diagnostic_score"],reverse=True)
    no_split=sorted((r for r in ranking if r["asr_no_split"]),key=lambda r:r["diagnostic_score"],reverse=True)
    # Formal public scoring stays rejected even though private validation
    # diagnostics have complete arithmetic.
    formal=None
    if ranking:
        c=ranking[0]["plan"]
        formal=predict(config.catalog,c["deployment"],c["routing"],rates,config.gpu_memory_mib,config.latency_reference_ms,config.weights,
            horizon=1800,seed=51,arrival_process=config.arrival_process,drain_seconds=config.drain_seconds,
            energy_reference_watts=config.energy_reference_watts,startup_amortization_seconds=config.startup_amortization_seconds)
        assert formal["U"] is None and not formal["eligible"]
    current={b["path"]:sha(ROOT/b["path"]) for b in bindings}
    unchanged=all(current[b["path"]]==b["sha256"] for b in bindings)
    report=dict(status="TRANSFER_UNVALIDATED_DEVELOPMENT_DIAGNOSTIC",protocol=protocol,results=results,
        ranking=ranking,best_selected_no_split=no_split[0] if no_split else None,
        delta_best_to_selected_no_split=ranking[0]["diagnostic_score"]-no_split[0]["diagnostic_score"] if ranking and no_split else None,
        selected_candidates=len(selected),evaluated_candidates=len(results),unevaluated_candidates=len(selected)-len(results),
        elapsed_seconds=time.perf_counter()-started,temporary_code_unchanged=unchanged,
        formal_public_prediction_on_top_candidate=formal,global_free_search=False,optimality_certified=False,gap=None,gpu_started=False)
    save(args.output/"report.json",report)
    lines=["# 逐输入迁移模型：有限强候选诊断重排","",report["status"],"",
        f"ASR {rates['asr']} rps，背景各 .05；H1800、开发种子51，候选{len(selected)}，已评{len(results)}。未封最终模型包络、未验证、未拟合、未执行GPU或全池自由搜索。",
        "候选来自常量均值模型两种显存榜top20、所有强结构组最佳及指定不分流/扩容方案。当前使用逐输入服务表及旧全局服务系数，显存采用旧额外学习系数。",
        "与前轮探索相比服务模型和seed均不同，不能将排序变化单独归因为某一个因素。以下只排列完整且满足预测约束的诊断分数，生产U仍为空。","",
        "| 名次 | 部署 | 背景text/image/ocr | ASR路由rps | 诊断分数 | 目标结构 |",
        "|---:|---|---|---|---:|---|"]
    for i,r in enumerate(ranking[:20],1):
        plan=r["plan"];background={t:plan["routing"][t] for t in ("text","image","ocr")}
        lines.append(f"| {i} | {json.dumps(plan['deployment'])} | {json.dumps(background)} | {json.dumps(plan['routing']['asr'])} | {r['diagnostic_score']:.9f} | {r['target_structure']} |")
    lines.extend(["",f"已评候选中最佳相对最佳不分流诊断差：{report['delta_best_to_selected_no_split']}。",
        "显存/rho/排空失败候选保留完整诊断但不参与排名；不能把不满足约束的diagnostic_partial_U当作可行分数。",
        f"公开正式评价门禁：{formal.get('reason') if formal else None}，U=None，eligible=False。代码绑定在诊断运行期间保持不变：{unchanged}。",
        "此处保留此前两榜top20形成的筛选限制，不能证明被常量均值阶段排除或未进入top20的逐输入候选更差。",""])
    (args.output/"README.zh.md").write_text("\n".join(lines))
    save(args.output/"manifest.json",dict(files={p.name:sha(p) for p in args.output.iterdir() if p.is_file()},
        script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),gpu_started=False))
    print(json.dumps(dict(output=str(args.output),evaluated=len(results),selected=len(selected),seconds=report['elapsed_seconds'],
        top=ranking[0]['plan'] if ranking else None,score=ranking[0]['diagnostic_score'] if ranking else None)),flush=True)


if __name__=="__main__":main()
