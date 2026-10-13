"""Audit numerical/scientific limits, summarize frozen scans and real-search targets."""
from .common import *
from .frontiers import first_axis, witness
from .solver import StructuralSolver
from module11b.convex_audit import reference_route
from collections import Counter
import itertools, html


def load_rows(stage):
    p=read(OUT/'experiment_freeze.json')
    return sum([read(OUT/f"{stage}_contexts/{w['id']}_{b['id']}.json")['rows'] for w in p['workloads'] for b in p['budgets']],[])


def validate():
    p=read(OUT/'experiment_freeze.json'); c,hw,roles,cal=baseline()
    old=copy.deepcopy(cal.capacity_factors);cc,ac=scaled(c,roles,cal,2.,2.5)
    for m in c:
        mult=2. if roles[m]=='specialist' else 1.
        assert cc[m].service_time_ms=={k:v*mult for k,v in c[m].service_time_ms.items()}
        assert cc[m].supported_tasks==c[m].supported_tasks
        assert ac.memory_table_mib[m]=={n:v*(2.5 if roles[m]=='specialist' else 1.) for n,v in cal.memory_table_mib[m].items()}
    assert ac.capacity_factors==old==cal.capacity_factors and ac.evidence_bounds==cal.evidence_bounds
    for family in p['families_in_speedup_order']:
        cc,ac=scaled(c,roles,cal,1.5,1.25,family['models'])
        for m in c:
            assert cc[m].service_time_ms=={k:v*(1.5 if m in family['models'] else 1.) for k,v in c[m].service_time_ms.items()}
    models=['distilbert','qwen'];toy={m:c[m] for m in models};tr={m:roles[m] for m in models}
    tc=ReplicaCalibration({m:{n:v for n,v in cal.capacity_factors[m].items() if n<=1} for m in models},
        {m:{n:v for n,v in cal.memory_table_mib[m].items() if n<=1} for m in models},
        {m:1 for m in models},{m:cal.model_evidence[m] for m in models},cal.source_hashes).validate(toy)
    ss=StructuralSolver();fit=ss.solve({'text':2.},toy,tr,tc,49140.)
    vals=[]
    for ns in itertools.product([0,1],repeat=2):
        x=dict(zip(models,ns))
        if not any(ns):continue
        fixed=reference_route(x,{'text':2.},toy,tc)
        if fixed:vals.append(fixed['evaluation']['J_seconds'])
    assert fit['global_lower_bound_seconds']<=min(vals)+1e-8 and abs(fit['J_seconds']-min(vals))<=1e-8
    again=ss.solve({'text':2.},toy,tr,tc,49140.)
    assert abs(again['J_seconds']-fit['J_seconds'])<=1e-8
    results=[]
    for path in (OUT/'query_cache').glob('*.json'):
        entry=read(path);r=entry['row'];f=entry['fit']
        if not r['feasible']:continue
        assert f['certificate']['passed'] and f['global_gap_seconds']<=1e-8
        assert r['max_rho']<=.8+1e-8 and r['safe_memory_mib']<=r['memory_budget_mib']+1e-6
        assert all(n<=cal.evidence_bounds[m] for m,n in r['deployment'].items())
        results.append(f['global_gap_seconds'])
    verify(p['protected_references']);verify([p['baseline_calibration']])
    out={'status':'PASS','counterfactual_scaling_checks':'PASS','Generalists_g_quality_capability_original_data_unchanged':True,
        'independent_small_discrete_exhaustive_check':'PASS','fresh_vs_reused_tangent_agreement':'PASS',
        'certified_feasible_cached_queries':len(results),'max_global_gap_seconds':max(results),
        'protected_file_hashes_verified':len(p['protected_references']),'GPU_runs':0,
        'scope':'Known empirical counts only; incomplete physical replica frontier is not filled or claimed'}
    save('cpu_validation.json',out);return out


def event_stats(rows,axis):
    return {e:{'contexts':len({(r['workload'],r['budget']) for r in rows if e in r['events']}),
        'first_observed_factor':min((r[axis] for r in rows if e in r['events']),default=None)} for e in ['T1','T2','T3','T4','T5','T6']}


def plots(rows):
    colors={'S_DOMINANT':'#70b6a1','HYBRID':'#f4bd63','G_DOMINANT':'#c88db7','G_ONLY':'#8d96d4','INFEASIBLE':'#dddddd'}
    workloads=['demand_balanced','mix_03_I1','mix_03_I2','mix_03_I3'];budgets=['B1','B3','B6'];parts=[]
    def text(x,y,s,size=12):return f'<text x="{x}" y="{y}" font-size="{size}">{html.escape(str(s))}</text>'
    parts.append('<svg xmlns="http://www.w3.org/2000/svg" width="1130" height="1190" viewBox="0 0 1130 1190"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#223344">')
    parts.append(text(28,30,'CPU counterfactual: global Specialist scaling',22))
    parts.append(text(28,53,'Cell: base phase; number: G/S replicas. Gray: infeasible. gamma increases right; mu increases up.'))
    for wi,w in enumerate(workloads):
        for bi,b in enumerate(budgets):
            ox=50+bi*360;oy=110+wi*258;parts.append(text(ox,oy-15,f'{w} / {b}',15))
            lookup={(r['gamma'],r['mu']):r for r in rows if r['workload']==w and r['budget']==b}
            for gi,g in enumerate(GAMMAS):
                parts.append(text(ox+gi*39+8,oy+222,g,10))
                for mi,m in enumerate(MUS):
                    y=oy+(6-mi)*29;x=ox+gi*39;r=lookup[g,m]
                    parts.append(f'<rect x="{x}" y="{y}" width="37" height="27" fill="{colors[r["base_phase"]]}"/>')
                    if r['feasible']:parts.append(text(x+7,y+18,f'{r["N_G"]}/{r["N_S"]}',10))
                    if gi==0:parts.append(text(x-28,y+18,m,10))
            parts.append(text(ox+120,oy+242,'gamma (latency ratio)',11))
    parts.append(text(30,1164,'Counterfactual planning evidence only. J objective unchanged; operational rho <= 0.8; measured counts only.',13))
    parts.append('</g></svg>');(OUT/'global_phase_map.svg').write_text(''.join(parts))


def main():
    validation=validate()
    plan=read(OUT/'experiment_freeze.json');globalrows=load_rows('global');prows=load_rows('per_task');front=read(OUT/'transition_frontiers.json')
    c,hw,roles,cal=baseline();targets={};family_stats=[];axis_records=[]
    for fam in plan['families_in_speedup_order']:
        name=fam['task'];ft={'baseline_speedup':fam['baseline_speedup'],'ratio_reference_model':fam['fastest_admitted_S'],
            'scaling_models':fam['models'],'kind':'REAL_MODEL_SEARCH_TARGET','axis_regions':{},'context_brackets':[]}
        for axis,a,other,grid in [('latency','gamma','mu',GAMMAS),('memory','mu','gamma',MUS)]:
            rr=[r for r in prows if r['task_family']==name and r['axis']==axis]
            stats=event_stats(rr,a);family_stats.append({'task':name,'axis':axis,'speedup':fam['baseline_speedup'],**stats})
            intervals=[];witness_status=Counter()
            for w in plan['workloads']:
                for b in plan['budgets']:
                    subset=[r for r in rr if r['workload']==w['id'] and r['budget']==b['id']]
                    # Memory axis omits shared (1,1) row in table; explicitly prepend it.
                    base=next(r for r in globalrows if r['workload']==w['id'] and r['budget']==b['id'] and r['gamma']==r['mu']==1)
                    if axis=='memory':subset=[{**base,'events':[]}]+subset
                    info=first_axis(subset,a);bracket=info['events']['ANY'];rec={'task':name,'axis':axis,'workload':w['id'],'budget':b['id'],**bracket}
                    if bracket['status']=='COARSE_BRACKET':
                        high=next(r for r in subset if r[a]==bracket['high']);wi=witness(high,base,w,b,fam['models'],c,roles,cal)
                        rec['witness']=wi;witness_status[wi['status']]+=1
                        supported=wi['structural_support'] or ('T5' in high['events'] and wi['route_support'])
                        rec['supported']=supported
                        if supported:intervals.append([bracket['low'],bracket['high']]);ft['context_brackets'].append(rec)
                    axis_records.append(rec)
            if intervals:
                first=min(intervals,key=lambda r:(r[1],r[0]));ft['axis_regions'][axis]={'status':'OBSERVED_COARSE_BRACKET','first_interval':first,
                    'context_intervals':sorted(set(tuple(x) for x in intervals)), 'interpretation':'first routing/replica transition; not necessarily identity removal or balanced composition'}
            else:ft['axis_regions'][axis]={'status':'RIGHT_CENSORED','search_lower_bound':grid[-1], 'interpretation':'no supported transition through scanned limit; no claim a larger ratio will transition'}
            stats['witness_status_counts']=dict(witness_status)
        targets[name]=ft
    save('real_model_search_targets.json',{'kind':'REAL_MODEL_SEARCH_TARGET','counterfactual_not_real_performance':True,
        'tasks':targets,'priority':['asr','lid','text','image','ocr'],
        'priority_reason':'Observed latency-driven split/replica transitions first (ASR/LID), then Text; Image/OCR latency axes remain censored through 4x. No 2G-based selection.',
        'ASR_LID_overlap':'Whole-model families overlap; effects are not additive or independent.',
        'distance_regions':'union of axis-only brackets (other ratio fixed at 1); no invented per-task 2D rectangle',
        'balanced_global_search_contexts':[{'workload':r['workload'],'budget':r['budget'],'gamma':r['gamma'],'mu':r['mu']} for r in globalrows if r['base_phase']=='HYBRID'],
        'real_selection_rule':'specialist_selection_rule.json; freeze Q/latency/memory Pareto and distance choice before final V3 outcomes'})
    csvsave('per_task_transition_brackets.csv',axis_records)
    latency=[r for r in globalrows if r['mu']==1];memory=[r for r in globalrows if r['gamma']==1]
    interaction=[r for r in globalrows if r['events'] and not next(s for s in globalrows if s['workload']==r['workload'] and s['budget']==r['budget'] and s['gamma']==r['gamma'] and s['mu']==1)['events'] and not next(s for s in globalrows if s['workload']==r['workload'] and s['budget']==r['budget'] and s['gamma']==1 and s['mu']==r['mu'])['events']]
    baseline_rows=[r for r in globalrows if r['gamma']==r['mu']==1]
    # Only same-mixture intensity sequence is used for workload phase attribution.
    sequences=[]
    for b in plan['budgets']:
        for g in GAMMAS:
            for m in MUS:
                seq=[next(r for r in globalrows if r['workload']==f'mix_03_I{i}' and r['budget']==b['id'] and r['gamma']==g and r['mu']==m) for i in [1,2,3]]
                sequences.append({'budget':b['id'],'gamma':g,'mu':m,'phases':[r['phase'] for r in seq],'G_S_split':[r.get('split_tasks') for r in seq],
                    'G_to_S_to_split':seq[0]['base_phase'] in ['G_ONLY','G_DOMINANT'] and seq[1]['base_phase']=='S_DOMINANT' and bool(seq[2].get('split_tasks'))})
    attrs={'kind':'COUNTERFACTUAL_CPU_ONLY','latency_only':event_stats(latency,'gamma'),'memory_only':event_stats(memory,'mu'),
        'global_event_context_counts':{e:len({(r['workload'],r['budget']) for r in globalrows if e in r['events']}) for e in ['T1','T2','T3','T4','T5','T6']},
        'joint_only_first_event_points':len(interaction),'per_task':family_stats,
        'baseline_phase_counts':dict(Counter(r['phase'] for r in baseline_rows)),
        'baseline_base_phase_counts':dict(Counter(r['base_phase'] for r in baseline_rows)),
        'baseline_multi_G_contexts':[(r['workload'],r['budget']) for r in baseline_rows if r['N_G']>=2],
        'replaced_S_model_counts':dict(Counter(m for r in globalrows for m in r.get('G_replaced_Specialists',[]))),
        'workload_sequences':sequences,'G_to_S_to_split_sequences':sum(s['G_to_S_to_split'] for s in sequences),
        'limitations':['No fixed residency cost or low-QPS GPU result enters J; no economic low-load claim.',
            'No quality scaling. No Generalist scaling. No unmeasured replica count extrapolation.',
            'N_G/N_S denote replicas; identity counts are separate.',
            'S_PLUS_G_OVERFLOW describes split, not physical exhaustion proof.',
            'First bracket is observed frozen-grid/refinement evidence; nonmonotone returns and numerical ties are retained.']}
    save('speed_vs_memory_attribution.json',attrs)
    plots(globalrows)
    summary(plan,globalrows,prows,front,attrs,targets,validation)


def summary(plan,rows,prows,front,attrs,targets,validation):
    lines=['# Module 12 Specialist Efficiency Sensitivity Audit','',
        '已完成 CPU-only counterfactual audit；GPU 未启动。当前真实 Specialist pool 继续冻结为 `EFFICIENCY_OPTIMAL_SPECIALIST_REGIME`。原始数据、Module 9、Module 11B 均未修改。',
        '',f"覆盖 9 个 development workloads × B1/B3/B6；global frozen grid {len(rows)} points。Per-task 仅 latency-only / memory-only，共 {len(prows)} 行，包含共享 baseline；未运行 per-task 二维网格。",
        '', '优化语义：在已测 replica counts 上全局最小化原 J，保持 g、admission、capability 不变，hard rho≤0.8。属于 CPU 结构参考，不是 frozen V3 bounded-search 的真实 outcome。N_G/N_S 是 replica 数；identity 数单独记录。',
        '', '## 必答结果','',
        f"1. (1,1)：27 个 context 的 base phase 全部 S_DOMINANT；primary phase 为 {attrs['baseline_phase_counts']}。其中 {len(attrs['baseline_multi_G_contexts'])} 个 B6 context 已存在 ≥2 个 G replicas，不能算 scaling 新触发第二个 G。",
        '2. Latency-only 首次 observed structure/routing change 为 pressured mix_03/I2/B1 的 ASR split：gamma bracket (1.2125,1.25]，其余比例不变。这里是 route transition，不是减少 Specialist identity。Latency-only 首次 T1 的 refined bracket 为 (1.9375,2.0]（mix_03/I3/B6，13→12 S replicas）。',
        '3. Memory-only 最早 T1 refined bracket 为 (1.00,1.05]：B3/B6 部分 context 减少 Specialist replicas，baseline deployment 在高端已 memory-infeasible。5% 是 refinement 的区间精度，不是性能门槛；不能说精确阈值就是 1.05。',
        '4. 二维 frontier 是带阶梯和 nonmonotone reentries 的离散边界，不是一条统一 gamma×mu 曲线。B1 的多数 mu>1 点直接 infeasible；B3/B6 多为 replica reduction。所有 event-specific Pareto-minimal 点和 reentries 见 transition_frontiers.json。',
        '5. Latency 驱动主要来自 ASR/LID overlapping speech families，Text 次之；memory 对所有 family 都可触发 replica reduction。类别相关，不可将 ASR/LID 效应相加。',
        '6. OCR/Image/Text 的真实最快 admitted G/S speedup 为 26.43×/25.39×/11.84×；OCR/Image latency-only 扫到 4× 无结构变化，支持其效率优势维持 Specialist-heavy。该关联不是 GPU 因果证据。',
        '7. 所有 global/per-task points 均未产生新增 T2/T3。第二 G 在部分 B6 baseline 已有；scaling 后没有新 context 从 <2 G 转为 ≥2 G。',
        '8. 首次减少 N_S：memory bracket (1,1.05]；latency refined bracket (1.9375,2.0]。此处减少主要是 replicas，不能一概写成减少 Specialist identities。',
        '9. 在本范围内 T6=0：没有 low-load task 完全撤掉其所有 dedicated Specialist paths 的 observed region。J 不包含实测 fixed-cost，不据此作经济结论。',
        f"10. 同一 mix_03 的 I1→I2→I3 上，G→S→S+G 序列 {attrs['G_to_S_to_split_sequences']} 个；已有 S-dominant→split 或 infeasible，未验证完整三阶段序列。",
        '11. 描述性 HYBRID 仅在 global 15/1512 points 出现：例如 mix_03/I1/B1 gamma=4,mu=1；mix_03/I1/B3 gamma=3,mu=2.5；低-load B3 gamma=4,mu=2.5。多为 ASR 向 G 迁移，OCR/Image 等仍走 S，不能称为所有任务 balanced，也不是证明真实 Specialist exhaustion。',
        '12. 各 task 的 REAL_MODEL_SEARCH_TARGET 如下。Memory 的首个区间代表 replica transition；latency censored 区间只是搜索下界，不能保证更大比例必然 transition。','',
        '| Task | Latency-only target | Memory-only target |','|---|---|---|']
    for t in plan['families_in_speedup_order']:
        obj=targets[t['task']]
        def fmt(a):
            v=obj['axis_regions'][a]
            return str(v['first_interval']) if v['status']=='OBSERVED_COARSE_BRACKET' else f">{v['search_lower_bound']}（right-censored）"
        lines.append(f"| {t['task'].upper()} | {fmt('latency')} | {fmt('memory')} |")
    lines += ['', '13. 下一步真实模型搜索优先 ASR/LID，其次 Text；Image/OCR 在当前 latency 范围仍未 transition。后续候选必须通过 frozen quality admission，实测 latency/memory，构造 Pareto frontier，再按已冻结的 region distance + quality/memory/model-ID tie-break 选择；本模块未检索或下载任何模型。',
        '', '## 数值与证据限制','',
        '10 ns global J certificate 使用原数值标准。Baseline deployment 在 counterfactual 下的 memory/capacity 或独立 convex lower-bound loss，及旧 route 的 objective loss，均单独审计。`NUMERICAL_TIE_OR_UNRESOLVED` 不被当作已证实的 composition threshold；route-only transition 的支持独立记录。global ANY bracket 的二分仅在第一个 coarse bracket 内、relative width≤5% 或≤6次；不细化 per-task axes。',
        f"保护文件 SHA 核对 {validation['protected_file_hashes_verified']} 个，PASS；全部 feasible cache certificate PASS；最大 J global gap {validation['max_global_gap_seconds']:.4g} s。完整 frontier 仍限于已测 counts；未测 counts 未补齐。",
        '', 'phase labels 用于描述，未参与 objective。G/S split 不等于 Specialist physical exhaustion。没有重开 Module 11B，没有 fixed-cost/low-QPS/multi-hotspot/Task-Rich runs。',
        '', '## 主要文件','',
        '- `global_sensitivity_grid.csv`, `phase_map.csv`, `global_phase_map.svg`',
        '- `transition_frontiers.json`, `transition_refinement.csv`, `transition_witnesses/`',
        '- `per_task_sensitivity.csv`, `per_task_transition_brackets.csv`',
        '- `speed_vs_memory_attribution.json`, `real_model_search_targets.json`',
        '- `specialist_selection_rule.json`, `cpu_validation.json`',
        '', 'Module 12 完成后停止；未启动下一阶段真实模型选择。']
    (OUT/'module12_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'COMPLETE','global_points':len(rows),'per_task_axis_rows':len(prows),'targets':{k:v['axis_regions'] for k,v in targets.items()},'validation':validation}),flush=True)

if __name__=='__main__':main()
