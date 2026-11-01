#!/usr/bin/env python3
"""108 predeclared CPU text candidates; external unvalidated energy diagnostic."""
import collections
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/static-study-gpu-20261022/execution-source-v5'
sys.path.insert(0, str(SOURCE))
from gsdeploy.calibration_v2 import replay_input_ids
from gsdeploy.quality import predict_quality
from gsdeploy.routing import Router
from gsdeploy.trace import expand_routes, make_trace, trace_digest

CONFIG = ROOT / 'artifacts/static-study-gpu-20261022/search-v2-v5-K8/system-config.json'
OUTPUT = ROOT / 'artifacts/static-spillover-study-20261022/cpu-discovery/text-load-screen-v1'
STATUS = 'EXPLORATORY_EXTERNAL_SCORE_NOT_VALIDATED'
RATES = [1.8, 1.9, 2.1, 2.3, 2.5, 2.7]
SHARES = [Fraction(1), Fraction(7, 8), Fraction(3, 4), Fraction(2, 3), Fraction(1, 2), Fraction(1, 3)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def compact(row):
    if row is None:
        return None
    return {k: row[k] for k in ('candidate_id', 'text_replicas', 'specialist_share',
        'external_diagnostic_score', 'six_external_terms', 'rho', 'historical_text_EM',
        'safe_residency_proxy_mib', 'per_task', 'text_path_unique_input_counts')}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    original = json.loads(CONFIG.read_text())
    data = original['catalog']
    old = data['calibrated_replay_v2']
    baseline = old['input_baselines']
    source_before = {str(p):sha(p) for p in sorted((SOURCE / 'gsdeploy').rglob('*.py'))}
    protocol = dict(status=STATUS, config_source=dict(path=str(CONFIG), sha256=sha(CONFIG)),
        rates=RATES, background_each_rps=.01, seed=51, horizon_seconds=600,
        text_replicas=[1, 2, 3], generalist_2_replicas=1, ocr_replicas=1,
        specialist_shares=[str(s) for s in SHARES], planned_candidates=108,
        background_routes=dict(image='generalist_2', asr='generalist_2', ocr='ocr'),
        service='Historical per-input medians for each measured n-state, no global multiplier, no load/colocation model; same frozen v5 router and input replay.',
        memory='Original safe_factor times standalone measured residency sum. No learned memory multiplier.',
        quality='Unchanged historical per-input output replay, actual outputs unverified; no seeds or quality values changed.',
        energy='External only: old coefficients dot [1,sum(route_rps*original_measured_n_state_mean_ms/1000),worker_count], plus old startup_per_worker*workers/amortization, divide original energy reference. No measured energy evidence or production scorer promotion.',
        power_coefficients=old['power_coefficients'], startup_joules_per_worker=old['startup_joules_per_worker'],
        power_parameter_version=old['version_sha256'], historical_training_feature_min=old['training_feature_min'],
        historical_training_feature_max=old['training_feature_max'],
        weights=original['weights'], references=original['latency_reference_ms'],
        production_U=None, production_E=None, full_pool_optimum=False, validated=False,
        gpu_started=False, holdout_results_read=False, runtime_source=str(SOURCE),
        limitations=['108 specified candidates only, seven-model catalog unchanged',
            'Single seed51 development diagnostic, no independent validation',
            'Power demand usually beyond historical training, mixed interference absent',
            'Fixed32-input cycle and deterministic routes can align quality and service coverage',
            'Scores for failed rho/memory/drain constraints excluded from selection'])
    save(OUTPUT / 'protocol.json', protocol)
    all_rows, summaries = [], []
    for rate in RATES:
        rates = dict(text=rate, image=.01, ocr=.01, asr=.01)
        trace = make_trace(rates, 600, 51, original['arrival_process'])
        ids = replay_input_ids(data, trace, 51)
        rows = []
        for n in [1, 2, 3]:
            for fraction in SHARES:
                share = float(fraction)
                x = dict(text=n, generalist_2=1, ocr=1)
                row = dict(text=rate*share)
                if fraction < 1:
                    row['generalist_2'] = rate*(1-share)
                routing = dict(text=row, image={'generalist_2': .01}, asr={'generalist_2': .01}, ocr={'ocr': .01})
                aliases, routes = expand_routes(x, routing)
                router, clocks = Router(routes), {a:0. for a in aliases}
                lat = {t:[] for t in rates}
                coverage = {m:collections.Counter() for m in ('text', 'generalist_2')}
                inwindow = late = 0
                for (arrival, task), sample in zip(trace, ids):
                    alias = router.choose(task)
                    model = aliases[alias]
                    svc = baseline[model][task][str(x[model])]['samples'][sample]['median_service_ms']
                    svc *= x[model] / data['capacity_factors'][model][str(x[model])]
                    finish = max(arrival, clocks[alias]) + svc / 1000
                    clocks[alias] = finish
                    lat[task].append((finish-arrival)*1000)
                    inwindow += finish <= 600
                    late += finish > 600+original['drain_seconds']
                    if task == 'text':
                        coverage[model][sample] += 1
                rho = {m:math.fsum(route.get(m, 0)*np.mean([z['median_service_ms'] for z in
                    baseline[m][t][str(x[m])]['samples'].values()])/1000
                    for t, route in routing.items() if m in route)/data['capacity_factors'][m][str(x[m])] for m in x}
                mem = data['safe_factor']*math.fsum(data['memory_table_mib'][m][str(nn)] for m, nn in x.items())
                stats = {t:dict(mean_ms=float(np.mean(v)), p95_ms=float(np.percentile(v,95)), count=len(v)) for t, v in lat.items()}
                quality, quality_evidence = predict_quality(data, x, routing, rates, trace, 51)
                if quality is None:
                    raise ValueError('Missing historical quality')
                demand = math.fsum(q*data['task_replica_profiles'][m][t][str(x[m])]['mean_service_ms']/1000
                    for t, route in routing.items() for m, q in route.items())
                features = [1., demand, float(sum(x.values()))]
                watts = math.fsum(a*b for a,b in zip(features,old['power_coefficients']))
                external_energy = (watts+old['startup_joules_per_worker']*sum(x.values())/original['startup_amortization_seconds'])/original['energy_reference_watts']
                total = math.fsum(rates.values())
                terms = dict(Q=quality,T=1-inwindow/len(trace),
                    L=math.fsum(rates[t]/total*stats[t]['mean_ms']/original['latency_reference_ms'][t] for t in rates),
                    P=math.fsum(rates[t]/total*stats[t]['p95_ms']/original['latency_reference_ms'][t] for t in rates),
                    M=mem/original['gpu_memory_mib'],E=external_energy)
                failed = []
                if max(rho.values()) > .8+1e-8: failed.append('RHO_GT_0P8')
                if mem > original['gpu_memory_mib']+1e-6: failed.append('MEMORY_GT_BUDGET')
                if late: failed.append('REQUESTS_BEYOND_DRAIN')
                external_score = -math.fsum(original['weights'][k]*v for k,v in terms.items())
                rows.append(dict(candidate_id=f'text{rate:g}-n{n}-share{fraction.numerator}of{fraction.denominator}',
                    workload=rates, text_replicas=n, specialist_share=str(fraction), deployment=x, routing=routing,
                    status=STATUS, production_U=None, production_E=None, production_eligible=False,
                    external_diagnostic_score=external_score if not failed else None,
                    external_arithmetic_even_if_infeasible=external_score, six_external_terms=terms,
                    proxy_constraints_pass=not failed, failed_constraints=failed, rho=rho,
                    safe_residency_proxy_mib=mem, per_task=stats, beyond_drain_requests=late,
                    historical_text_EM=1-quality_evidence['per_task']['text']['loss'],
                    text_path_unique_input_counts={m:len(c) for m,c in coverage.items()},
                    text_path_input_counts={m:dict(c) for m,c in coverage.items()},
                    external_power_features=features, external_board_watts=watts,
                    power_features_outside_historical_range=[v<a or v>b for v,a,b in zip(features,old['training_feature_min'],old['training_feature_max'])],
                    trace_sha256=trace_digest(trace)))
        feasible = [r for r in rows if r['proxy_constraints_pass']]
        best = lambda values:max(values,key=lambda r:r['external_diagnostic_score'],default=None)
        by_n = {str(n):best([r for r in feasible if r['text_replicas']==n]) for n in (1,2,3)}
        top = best(feasible)
        n2, n1 = by_n['2'], by_n['1']
        n3no = next(r for r in rows if r['text_replicas']==3 and r['specialist_share']=='1')
        summary = dict(text_rps=rate,evaluated=len(rows),feasible=len(feasible),
            best_among_listed=compact(top), best_by_text_replicas={k:compact(v) for k,v in by_n.items()},
            n2_minus_best_n1_external_score=n2['external_diagnostic_score']-n1['external_diagnostic_score'] if n2 and n1 else None,
            listed_best_is_n2_spillover=bool(top and top['text_replicas']==2 and top['specialist_share']!='1'),
            n3_no_split=dict(rho=n3no['rho'],failed_constraints=n3no['failed_constraints'],
                            external_diagnostic_score=n3no['external_diagnostic_score']),
            capacity_rescue_relative_to_n3_no_split=n3no['rho']['text']>.8+1e-8)
        summaries.append(summary)
        all_rows.extend(rows)
    source_after = {str(p):sha(p) for p in sorted((SOURCE/'gsdeploy').rglob('*.py'))}
    save(OUTPUT / 'report.json',dict(protocol=protocol,summaries=summaries,candidates=all_rows,
        source_unchanged=source_before==source_after,source_receipts=source_before,gpu_started=False))
    lines = ['# Text 多副本分流：108个候选CPU诊断','',STATUS,'',
        '固定seed51/H600、背景各.01。逐输入服务不乘旧全局倍率；显存不乘旧学习倍率。能耗仅外部旧参数外推，生产U/E为空。',
        '只比较text n1/n2/n3+G2n1+OCR1及六个指定比例，不能称为全池最优。','',
        '| text rps | 列表最佳 n/share | 外部诊断分数 | 最佳n2相对最佳n1 | n3不分流rho | n3不分流容量失败 |',
        '|---:|---|---:|---:|---:|---|']
    for s in summaries:
        top=s['best_among_listed'];delta=s['n2_minus_best_n1_external_score']
        lines.append(f"| {s['text_rps']:g} | {top['text_replicas']}/{top['specialist_share']} | {top['external_diagnostic_score']:.6f} | {delta:.6f} | {s['n3_no_split']['rho']['text']:.6f} | {s['capacity_rescue_relative_to_n3_no_split']} |")
    lines += ['', '实际共驻干扰、排队尾延迟、能耗和质量复现未经验证；固定输入循环与路由对齐可能影响排名。高负载的优势若伴随n3不分流rho失败，应明确属于容量救援情境。','']
    (OUTPUT/'README.zh.md').write_text('\n'.join(lines))
    save(OUTPUT/'manifest.json',dict(files={p.name:sha(p) for p in OUTPUT.iterdir() if p.is_file()},
        script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),gpu_started=False))
    print(json.dumps(dict(output=str(OUTPUT),evaluated=len(all_rows),source_unchanged=source_before==source_after,
        n2_spillover_list_winner_rates=[s['text_rps'] for s in summaries if s['listed_best_is_n2_spillover']]),ensure_ascii=False))


if __name__=='__main__':
    main()
