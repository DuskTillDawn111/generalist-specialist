#!/usr/bin/env python3
"""External limited-structure scan. No production gates, U or optimizer edits."""

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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsdeploy.quality import quality_digest, task_quality
from gsdeploy.routing import Router
from gsdeploy.trace import expand_routes, make_trace, trace_digest
from gsdeploy.prediction import predict

STATUS = "EXPLORATORY_TRANSFER_NOT_VALIDATED"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_receipts():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "gsdeploy").rglob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=1.8)
    parser.add_argument("--seconds", type=float, default=105.)
    parser.add_argument("--max-evaluations", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/static-spillover-study-20261022/cpu-discovery/strong-structures-asr1.8-v1")
    parser.add_argument("--source-config", type=Path, default=ROOT / "artifacts/static-study-gpu-20261022/search-v2-v5-K8/system-config.json")
    args = parser.parse_args()
    if not 0 < args.seconds <= 115 or args.rate <= 0:
        parser.error("Positive ASR rate and budget <=115s required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    before = source_receipts()
    original = json.loads(args.source_config.read_text())
    data, old = copy.deepcopy(original["catalog"]), original["catalog"]["calibrated_replay_v2"]
    for k in ("calibrated_replay_v2", "calibrated_surrogate", "energy_profiles", "colocation_profiles"):
        data.pop(k, None)
    data["performance_mode"], data["memory_mode"] = "task_replica_proxy", "standalone_proxy"
    rates = dict(text=.05, image=.05, ocr=.05, asr=args.rate)
    horizon, seed = original["horizon_seconds"], original["random_seed"]
    trace = make_trace(rates, horizon, seed, original["arrival_process"])
    budget, refs, weights = original["gpu_memory_mib"], original["latency_reference_ms"], original["weights"]
    total = math.fsum(rates.values())
    names = sorted(data["models"])
    fractions = sorted({Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(4, 5), Fraction(5, 6), Fraction(7, 8)} |
                       {1 - f for f in (Fraction(2, 3), Fraction(3, 4), Fraction(4, 5), Fraction(5, 6), Fraction(7, 8))})
    states = {m: sorted(map(int, data["capacity_factors"][m])) for m in names}
    protocol = {"status": STATUS, "workload": rates, "seconds_budget": args.seconds,
        "evaluation_limit": args.max_evaluations, "seed": seed, "horizon_seconds": horizon,
        "source_config": {"path": str(args.source_config.resolve()), "sha256": sha(args.source_config)},
        "declared_deployment_states": states, "background_route_domain": "Each background task goes wholly to one deployed executable model with a measured state profile.",
        "asr_route_domain": "One executable family, or two executable families with declared fractions. Includes specialist/generalist and generalist/generalist splits.",
        "pair_fractions": [str(f) for f in fractions],
        "used_resident_families_only": True, "quality_model": "Exact public historical per-input replay for each task routing column, cached after unchanged public evidence checks.",
        "service_model": "Measured n-state constant means through independent FIFO implementation cross-checked against public predictor. No service multiplier or input-service tails.",
        "raw_safe_memory": "original safe_factor * sum(original n-state residency)",
        "transferred_memory": "max(raw_safe_memory * old memory_multiplier, old seen-deployment peak if any)",
        "old_memory_multiplier": old["memory_multiplier"], "old_power_coefficients": old["power_coefficients"],
        "old_startup_joules_per_worker": old["startup_joules_per_worker"], "old_parameter_version": old["version_sha256"],
        "external_energy_model": "(dot([1,sum(route_rps*n_state_service_seconds),worker_count],old_power_coefficients)+old_startup_per_worker*worker_count/A)/reference_watts",
        "finite_candidate_order": "One low service-floor candidate per deployment first, then remaining candidates by optimistic service-only ranking. Floor is an ordering heuristic, no full-domain certificate.",
        "not_validated": ["new workload", "mixed-board interference", "memory extrapolation", "power extrapolation", "open-loop stability", "quality reproduction"],
        "global_free_search": False, "physical_optimality_certified": False, "gpu_started": False}
    save(args.output / "protocol.json", protocol)

    def service(m, t, n):
        return data["task_replica_profiles"].get(m, {}).get(t, {}).get(str(n), {}).get("mean_service_ms")

    def memories(x):
        raw = data["safe_factor"] * math.fsum(data["memory_table_mib"][m][str(n)] for m, n in x.items())
        learned = max(raw * old["memory_multiplier"], old["training_deployment_memory"].get(quality_digest(x), 0))
        return raw, learned

    deployment_records, candidates = [], []
    counters = dict(deployment_identities=0, raw_memory_rejected=0, learned_only_memory_rejected=0,
                    background_profile_incomplete=0, route_candidates_generated=0, rho_rejected=0, unused_resident_rejected=0)
    for ns in itertools.product(*(states[m] for m in names)):
        if not any(ns):
            continue
        counters["deployment_identities"] += 1
        x = {m:n for m,n in zip(names, ns) if n}
        raw, learned = memories(x)
        record = dict(deployment=x, raw_safe_memory_mib=raw, transferred_memory_mib=learned,
                      raw_memory_feasible=raw <= budget, transferred_memory_feasible=learned <= budget)
        deployment_records.append(record)
        if raw > budget:
            counters["raw_memory_rejected"] += 1
            continue
        if learned > budget:
            counters["learned_only_memory_rejected"] += 1
        bg = {t:[m for m,n in x.items() if service(m,t,n)] for t in ("text", "image", "ocr")}
        if any(not row for row in bg.values()):
            counters["background_profile_incomplete"] += 1
            continue
        active_asr = [m for m,n in x.items() if service(m,"asr",n)]
        asr_rows = [{m:args.rate} for m in active_asr]
        for a,b in itertools.combinations(active_asr,2):
            asr_rows.extend({a:args.rate*float(f),b:args.rate*(1-float(f))} for f in fractions)
        for bg_assign in itertools.product(*(bg[t] for t in bg)):
            background = {t:{m:rates[t]} for t,m in zip(bg,bg_assign)}
            for row in asr_rows:
                counters["route_candidates_generated"] += 1
                routing = {**background,"asr":row}
                if set(x) != {m for r in routing.values() for m in r}:
                    counters["unused_resident_rejected"] += 1
                    continue
                demands = {m:math.fsum(r.get(m,0)*service(m,t,x[m])/1000
                                      for t,r in routing.items() if m in r) for m in x}
                rho = {m:d/x[m] for m,d in demands.items()}
                if max(rho.values()) > .8+1e-8:
                    counters["rho_rejected"] += 1
                    continue
                features = [1.,math.fsum(demands.values()),float(sum(x.values()))]
                watts = math.fsum(a*b for a,b in zip(features,old["power_coefficients"]))
                energy = (watts+old["startup_joules_per_worker"]*sum(x.values())/original["startup_amortization_seconds"])/original["energy_reference_watts"]
                floor = weights["M"]*raw/budget+weights["E"]*energy
                floor += math.fsum((weights["L"]+weights["P"])*q/total*service(m,t,x[m])/refs[t]
                                  for t,r in routing.items() for m,q in r.items())
                candidates.append(dict(deployment=x,routing=routing,rho=rho,max_rho=max(rho.values()),
                    raw_safe_memory_mib=raw,transferred_memory_mib=learned,
                    transferred_memory_feasible=learned<=budget,external_energy=energy,
                    power_features=features,external_board_watts=watts,ordering_cost=floor))
    generation_seconds = time.perf_counter()-started
    save(args.output / "deployment-memory-inspection.json", {"status":STATUS,"budget_mib":budget,
        "memory_multiplier":old["memory_multiplier"],"deployments":deployment_records,"counts":counters})
    candidates.sort(key=lambda c:c["ordering_cost"])
    first,rest,seen = [],[],set()
    for c in candidates:
        key = tuple(c["deployment"].items())
        (rest if key in seen else first).append(c)
        seen.add(key)
    order = first+rest
    # Route-specific deterministic choices and quality are independent of other
    # tasks. Cache exact public column outputs; do not change evidence gates.
    column_cache = {}
    task_indices = {t:[i for i,(_,task) in enumerate(trace) if task==t] for t in rates}
    def column(x,t,row):
        key = (t,tuple(sorted((m,x[m],q) for m,q in row.items())))
        if key not in column_cache:
            loss,evidence = task_quality(data,t,x,row,trace,seed)
            if loss is None:
                raise ValueError("Missing quality column: "+str(evidence))
            aliases,routes = expand_routes(x,{t:row})
            router = Router(routes)
            choices = [router.choose(t) for _ in task_indices[t]]
            column_cache[key] = (loss,choices,{a:service(m,t,x[m])/1000 for a,m in aliases.items() if m in row})
        return column_cache[key]
    evaluated,cross_checks = [],[]
    def evaluate(c):
        x,routing = c["deployment"],c["routing"]
        tasks = {t:column(x,t,routing[t]) for t in rates}
        clocks = {a:0. for a in expand_routes(x,routing)[0]}
        indices = {t:0 for t in rates}; lat = {t:[] for t in rates}
        inwindow=late=0
        for offset,t in trace:
            _,choices,svcs=tasks[t]; alias=choices[indices[t]];indices[t]+=1
            finish=max(offset,clocks[alias])+svcs[alias];clocks[alias]=finish
            lat[t].append((finish-offset)*1000)
            inwindow+=finish<=horizon;late+=finish>horizon+original["drain_seconds"]
        stats={t:dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),request_count=len(v)) for t,v in lat.items()}
        terms=dict(Q=math.fsum(rates[t]/total*tasks[t][0] for t in rates),T=1-inwindow/len(trace),
            L=math.fsum(rates[t]/total*stats[t]["mean_ms"]/refs[t] for t in rates),
            P=math.fsum(rates[t]/total*stats[t]["p95_ms"]/refs[t] for t in rates),M=c["raw_safe_memory_mib"]/budget,E=c["external_energy"])
        score=-math.fsum(weights[k]*v for k,v in terms.items()) if not late else None
        learned_terms={**terms,"M":c["transferred_memory_mib"]/budget}
        learned_score=-math.fsum(weights[k]*v for k,v in learned_terms.items()) if not late and c["transferred_memory_feasible"] else None
        return {**c,"raw_external_terms":terms,"transferred_external_terms":learned_terms,
            "raw_exploratory_score":score,"transferred_exploratory_score":learned_score,
            "incomplete_after_drain":late,"per_task":stats,
            "asr_specialist_fraction":routing["asr"].get("asr",0)/args.rate,
            "target_asr_n_ge_2_and_split":x.get("asr",0)>=2 and 0<routing["asr"].get("asr",0)<args.rate,
            "is_no_split":len(routing["asr"])==1,"status":STATUS,"production_U":None,"production_E":None}
    for i,c in enumerate(order):
        if len(evaluated)>=args.max_evaluations or time.perf_counter()-started>=args.seconds:
            break
        result=evaluate(c);result["evaluation_index"]=i
        if len(cross_checks)<6:
            p=predict(data,c["deployment"],c["routing"],rates,budget,refs,weights,
                horizon=horizon,seed=seed,arrival_process=original["arrival_process"],drain_seconds=original["drain_seconds"],
                energy_reference_watts=original["energy_reference_watts"],startup_amortization_seconds=original["startup_amortization_seconds"],trace=trace)
            differences={k:result["raw_external_terms"][k]-p[k+"_norm"] for k in "QTLPM"}
            assert all(abs(v)<1e-10 for v in differences.values()),differences
            assert p["U"] is None and p["E_norm"] is None
            cross_checks.append(dict(evaluation_index=i,differences=differences,production_U_null=True,production_E_null=True))
        evaluated.append(result)
        if len(evaluated)%500==0:
            print(json.dumps(dict(evaluated=len(evaluated),candidate_count=len(candidates),seconds=time.perf_counter()-started)),flush=True)
    def ranking(field,selector=lambda c:True):
        return sorted((c for c in evaluated if c[field] is not None and selector(c)),key=lambda c:c[field],reverse=True)
    summaries={}
    for mode,field in (("raw_safe_sum","raw_exploratory_score"),("old_learned_multiplier","transferred_exploratory_score")):
        rows=ranking(field);nosplit=ranking(field,lambda c:c["is_no_split"])
        target=ranking(field,lambda c:c["target_asr_n_ge_2_and_split"])
        summaries[mode]={"top20":rows[:20],"best_no_split":nosplit[0] if nosplit else None,
                         "best_target":target[0] if target else None,
                         "best_score_gap_to_no_split":rows[0][field]-nosplit[0][field] if rows and nosplit else None}
    after=source_receipts()
    report=dict(status=STATUS,workload=rates,protocol=protocol,domain_counters=counters,
        generation_seconds=generation_seconds,total_seconds=time.perf_counter()-started,
        declared_rho_and_raw_memory_feasible_candidates=len(candidates),
        evaluated_candidates=len(evaluated),unevaluated_candidates=len(candidates)-len(evaluated),
        distinct_feasible_deployments=len(seen),distinct_evaluated_deployments=len({tuple(c["deployment"].items()) for c in evaluated}),
        selection_order="One candidate per deployment, then optimistic service-only cost ordering",
        summaries=summaries,cross_checks_against_public_predictor=cross_checks,
        column_cache_entries=len(column_cache),source_before=before,source_after=after,source_unchanged=before==after,
        all_declared_candidates_evaluated=len(evaluated)==len(candidates),
        global_free_search=False,optimality_certified=False,gap=None,gpu_started=False)
    save(args.output/"report.json",report)
    save(args.output/"evaluated-candidates.json",dict(status=STATUS,candidates=evaluated))
    lines=["# 强替代有限结构筛查","",STATUS,"",
        f"ASR={args.rate} rps，背景三任务各 .05；七模型副本原始域保留，背景仅整任务路由，ASR最多两模型族分流。仅评估使用到的驻留模型。",
        f"原始非空部署 {counters['deployment_identities']}；通过画像、rho 与原始安全显存的有限路由候选 {len(candidates)}，完整FIFO评价 {len(evaluated)}，未评 {len(candidates)-len(evaluated)}。",
        f"CPU墙钟 {report['total_seconds']:.3f}s；全池自由搜索未执行，最优性/gap未知。六条独立FIFO计算与公开预测器五项成本一致。",
        f"原始安全显存乘数是safe_factor={data['safe_factor']}；旧学习系数为额外 {old['memory_multiplier']:.12f}，两者分别评价。旧学习系数没有新负载硬件边界含义。","",
        "| 显存解释 | 排名 | 部署 | ASR路由rps | 外部探索分数 | 原始安全MiB | 旧迁移MiB | 目标结构 |",
        "|---|---:|---|---|---:|---:|---:|---|"]
    for mode,field in (("raw_safe_sum","raw_exploratory_score"),("old_learned_multiplier","transferred_exploratory_score")):
        for i,c in enumerate(summaries[mode]["top20"][:10],1):
            lines.append(f"| {mode} | {i} | {json.dumps(c['deployment'])} | {json.dumps(c['routing']['asr'])} | {c[field]:.6f} | {c['raw_safe_memory_mib']:.1f} | {c['transferred_memory_mib']:.1f} | {c['target_asr_n_ge_2_and_split']} |")
        no=summaries[mode]["best_no_split"]
        if no:
            lines.extend(["",f"{mode}：已评最佳不分流 {json.dumps(no['deployment'])}，ASR路由 {json.dumps(no['routing']['asr'])}，分数 {no[field]:.9f}；已评最佳对它差 {summaries[mode]['best_score_gap_to_no_split']:.9f}。",""])
    lines.extend(["", "逐部署显存记录包含所有原始副本身份，可区分原始安全显存拒绝和仅旧学习系数拒绝。完整候选包含全部背景绝对路由，不能只据ASR一行忽略OCR迁移代价。",
        "服务使用固定画像均值，不含逐输入服务尾；功率/启动沿用旧低负载系数作外部外推，质量为历史逐输入回放。生产U/E始终null，不创建能耗证据，不绕过任何门禁。",
        "当前排名仅针对实际已评的有限结构，不足以排除背景分流、其他ASR分数或未评价候选，也不等于新mode验证后全池自由搜索结果。",""])
    (args.output/"README.zh.md").write_text("\n".join(lines))
    save(args.output/"manifest.json",dict(files={p.name:sha(p) for p in args.output.iterdir() if p.is_file()},
        script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),gpu_started=False))
    print(json.dumps(dict(output=str(args.output),evaluated=len(evaluated),remaining=len(candidates)-len(evaluated),seconds=report['total_seconds'])),flush=True)


if __name__=="__main__":
    main()
