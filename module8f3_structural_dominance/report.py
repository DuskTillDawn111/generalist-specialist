"""Fact-based structural interpretation and frozen-artifact closeout."""
from .common import *
from collections import Counter
import statistics
import numpy as np

def records(name):
    out=[]
    for row in csv.DictReader((OUT/name).open()):
        parsed={}
        for k,v in row.items():
            if v=='':parsed[k]=None
            elif v in ['True','False']:parsed[k]=v=='True'
            elif v[:1] in ['{','[']:parsed[k]=json.loads(v)
            else:
                try:parsed[k]=float(v)
                except ValueError:parsed[k]=v
        out.append(parsed)
    return out

def expand_all_state_rows():
    file=OUT/'pareto_all_states.csv'
    with file.open() as f:
        reader=csv.DictReader(f);columns=reader.fieldnames;count=sum(1 for _ in reader)
    if count==13824*12:return
    assert count==4326*12
    invalid=[r for r in records('all_deployments_structural.csv') if not r['quality_coverage']]
    cells=read(F/'campaign_cells.json')['cells']
    with file.open('a',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=columns)
        for cell in cells:
            for s in invalid:
                ng,ns=s['N_G'],s['N_S']
                r={'cell_id':cell['cell_id'],'state_id':int(s['state_id']),'deployment_vector':s['vector'],'deployment':s['deployment'],'N_G':int(ng),'N_S':int(ns),'total_replicas':int(s['total_replicas']),'M_safe_mib':s['M_safe_mib'],'z_star':None,'routing':{},'quality_capability_covered':False,'memory_feasible':s['M_safe_mib']<=cell['memory_budget_mib'],'capacity_feasible':False,'cell_feasible':False,'Pareto_global':False,'physical_memory_feasible':s['physical_memory_feasible'],'regime':'Specialist-heavy' if ns>ng else 'Hybrid' if ng and ns else 'Generalist-heavy' if ng else 'Specialist-only','hybrid_flag':bool(ng and ns)}
                w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})

def main():
    structural=read(OUT/'structural_stage_results.json');cache=read(OUT/'lp_states_cache.json');oracles=structural['oracle_cells'];ids=cache['model_order'];cells=read(F/'campaign_cells.json')['cells'];bycell={c['cell_id']:c for c in cells};states={s['state_id']:s for s in cache['states']};front=[states[i] for i in structural['frontier_state_ids']];physical=[s for s in front if s['physical_memory_feasible']]
    search=[]
    for c in oracles:
        trace=read(DEV/c['cell_id']/'static_v2/search_trace.json');target=c['oracle_deployment'];matches=[]
        for iteration in trace['iterations']:
            for candidate in iteration['candidates']:
                x=candidate['mutation']['deployment']
                if all(x.get(m,0)==target[m] for m in ids):matches.append(candidate)
        search.append({'cell_id':c['cell_id'],'oracle_deployment':target,'oracle_N_G':c['oracle_N_G'],'oracle_N_S':c['oracle_N_S'],'oracle_is_seed':all(trace['seed']['deployment'].get(m,0)==target[m] for m in ids),'oracle_generated':bool(matches),'oracle_Top_B':any(r['selected_top_b'] for r in matches),'oracle_runtime_evaluated':bool(any(r.get('runtime') for r in matches) or (all(trace['seed']['deployment'].get(m,0)==target[m] for m in ids) and trace['seed_runtime'])), 'oracle_evaluated_as_candidate':bool(any(r.get('runtime') for r in matches)), 'oracle_is_runtime_final':c['v2_equals_oracle'],'oracle_accepted':any(r['status']=='ACCEPTED' for r in matches),'v2_status':trace['status'],'v2_iterations':trace['search_iterations'],'visited_states':trace['visited_deployment_count'],'seed_stability':trace['seed_runtime']['stability_state_v2'],'final_stability':trace['runtime']['stability_state_v2'],'candidate_top_B_statuses':[{'operation':r['mutation']['operation'],'status':r['status'],'stability':r['runtime']['stability_state_v2'] if r.get('runtime') else None} for it in trace['iterations'] for r in it['candidates'] if r.get('selected_top_b')],'structural_gap_is_runtime_gap':False})
    csvsave('search_structural_gap_audit.csv',search)
    def gap(method):
        a=[c[method+'_gap'] for c in oracles]
        return {'mean':statistics.mean(a),'median':statistics.median(a),'max':max(a),'min':min(a),'distribution_by_cell':dict(zip([c['cell_id'] for c in oracles],a)),'equal_oracle_deployment_cells':sum(c[method+'_equals_oracle'] for c in oracles),'equal_oracle_z_cells':sum(c[method+'_z_equals_oracle'] for c in oracles),'Pareto_cells':sum(c[method+'_Pareto'] for c in oracles),'strict_positive_gap_cells':sum(v>TOL for v in a)}
    RC=gap('RC_SG');V=gap('v2');removal=records('specialist_removal_cost.csv');greedy=records('rcsg_greedy_optimality.csv');task=records('task_level_dominance.csv');decomp=records('intensity_capacity_decomposition.csv');memory=read(OUT/'memory_counterfactual_results.json');speed=read(OUT/'speed_counterfactual_results.json');actual_memory=list(csv.DictReader((DEV/'memory_sweep_results.csv').open()))
    specialists=set(read(OUT/'local_audit_results.json')['fastest_unique_specialists']);equalS=0
    for c in oracles:
        a={m for m,n in c['RC_SG_deployment'].items() if n and m not in ['qwen','phi4mm','minicpm']};b={m for m,n in c['oracle_deployment'].items() if n and m not in ['qwen','phi4mm','minicpm']};equalS+=a==b
    valid_removal=[r for r in removal if r['primary_reference']];primary_cover_fail=sum(r['status']=='COVERAGE_INFEASIBLE' for r in valid_removal)
    residency_bias=all(s['N_S']>s['N_G'] for s in front) and all(r['speedup']>1 for r in task if r['speedup'] is not None) and not any(s['N_S']==0 for s in front)
    hidden_multi_G=[r for r in search if r['oracle_N_G']>=2 and not r['oracle_runtime_evaluated']]
    # Sustained gap means the same loose budget across all three frozen intensity
    # levels, not an invented percentage threshold.
    loose=[c for c in oracles if c['B_mib']==49140]
    structural_search_limitation=all(c['v2_gap']>TOL and c['oracle_N_G']>sum(c['v2_deployment'][m]>0 for m in ['qwen','phi4mm','minicpm']) for c in loose) and bool(hidden_multi_G)
    diagnosis='BOTH' if residency_bias and structural_search_limitation else 'STRUCTURAL_DOMINANCE_STRONG' if residency_bias else 'OPTIMIZER_WEAKNESS_STRONG' if structural_search_limitation else 'INCONCLUSIVE'
    facts={'RC_SG':RC,'v2':V,'RC_SG_specialist_identity_set_equals_oracle_cells':equalS,'frontier_all_bound_valid_count':len(front),'frontier_physical_count':len(physical),'frontier_physical_N_G_distribution':dict(Counter(s['N_G'] for s in physical)),'frontier_physical_N_S_distribution':dict(Counter(s['N_S'] for s in physical)),'all_frontier_Specialist_heavy':residency_bias,'Generalist_only_frontier_exists':any(s['N_S']==0 for s in front),'Generalists_more_than_Specialists_frontier_exists':any(s['N_G']>s['N_S'] for s in front),'primary_removal_coverage_infeasible_count':primary_cover_fail,'primary_removal_total':len(valid_removal),'hidden_multi_G_oracle_cells':hidden_multi_G,'loose_budget_sustained_v2_structural_gap':structural_search_limitation,'greedy_terminal_zero_demand_gain_positive_z_benefit_rows':sum(r['terminal_step'] and r['demand_saving']==0 and (r['marginal_z_benefit'] or 0)>TOL for r in greedy),'RC_SG_exact_oracle_majority':RC['equal_oracle_deployment_cells']>len(oracles)/2,'rich_non_Specialist_heavy_frontier':False,'nonzero_gap_is_not_a_claim_of_measured_E2E_superiority':True}
    save('structural_diagnosis.json',{'diagnosis':diagnosis,'STRUCTURAL_AUDIT_ONLY':True,'GPU_RUNS':0,'facts':facts,'interpretation':'Strong Specialist residency bias coexists with bounded-search/base-selection gaps in standalone pressure. All frontier states are also Hybrid under the overlapping frozen flag; hidden two-Generalist solutions remain Specialist-heavy. RC-SG is Pareto-optimal in all cells but not the minimum-pressure deployment in most cells. Not all bullets of either pure idealized category hold; the mixed diagnosis distinguishes these facts instead of inventing a near-optimal cutoff.','v3_implication':'Preserve complete 8F-2 and existing gates. Full 9A can address interference/congestion prediction and routing; 9B/9C can test whether bounded search finds hidden multi-Generalist/min-pressure compositions. CPU oracle cannot guarantee runtime wins. A later independently motivated real G/S trade-off regime may be needed; do not change existing workload/model pool or start it in this module.','complete_Module9A_started':False,'routing_ranking_method_benchmark_changed':False})
    lines=['# Module 8F-3 — Generalist–Specialist Structural Dominance Audit','',f'Execution PASS. Diagnosis: **{diagnosis}**. STRUCTURAL_AUDIT_ONLY=true; GPU_RUNS=0.','', '1. Frozen task-level advantage and marginal residency cost (adding S does not subtract resident G memory):','', '| Task | fastest admitted G | fastest admitted S | speedup | additional safe MiB |','|---|---|---|---:|---:|']
    for r in task:
        if r['cell_id']=='C03':lines.append(f'| {r["task"]} | {r["fastest_G"]} | {r["fastest_S"] or "NONE"} | {r["speedup"]:.9f} | {r["extra_S_safe_mib"]:.9f} |' if r['speedup'] is not None else f'| {r["task"]} | {r["fastest_G"]} | NONE | — | — |')
    lines+=['','2. ResNet50、PaddleOCR、DistilBERT 同时呈现排序最突出的速度/低显存组合；不定义任意“巨大”阈值。Whisper-small 对 LID 是最快 admitted Specialist，并同时覆盖 ASR。Distil-Whisper 对 ASR 最快，但 residency cost 更高。所有 12 cells 的 lambda-weighted demand saving 和 value/MiB 见 task_level_dominance.csv。',
'','3. Primary reference = Phi + DistilBERT/ResNet50/PaddleOCR/Whisper-small/Distil-Whisper。各 budget 下 reference 是否 fits 都明确记录；没有将预算外 reference 作为 serving baseline。Phi 不 admitted text/image/OCR/LID，因此移除前四个分别失去文本/图像/OCR/LID coverage。移除 Distil-Whisper 保留 Whisper-ASR fallback，增加 pressure。额外 Qwen/MiniCPM references 展示合法 G fallback；全部 task flows、demand multipliers 保存在 specialist_removal_cost.csv。','', '| C12 reference G | removed S | safe MiB saved | status | delta demand (s/s) | delta z |','|---|---|---:|---|---:|---:|']
    for r in removal:
        if r['cell_id']=='C12':lines.append(f'| {r["reference_G"]} | {r["removed_specialist"]} | {r["memory_saved_mib"]:.6f} | {r["status"]} | {r["delta_demand"] if r["delta_demand"] is not None else "undefined"} | {r["delta_z"] if r["delta_z"] is not None else "undefined"} |')
    lines +=['','4. 完整 13,824 向量；4,326 quality/capability-covered LP states。9 个全 bound-valid Pareto states，其中 8 个 <=49140 MiB。物理 frontier 的 G 数分布 '+str(facts['frontier_physical_N_G_distribution'])+'；S 数分布 '+str(facts['frontier_physical_N_S_distribution'])+'。全部满足 N_S>N_G。第 9 个状态含 3G，safe memory 超过物理预算，不能作为可执行解。',
'','5. 没有 Generalist-only 或 N_G>N_S 的 Pareto region。2G+4S/5S 解是实际 Hybrid（冻结 hybrid 与 Specialist-heavy predicates 重叠），不能将其误称 Generalist-heavy。',
'','6. I3 下 M1→M6 oracle compositions：','', '| Cell | budget MiB | Oracle deployment | N_G/N_S | z* | RC gap | v2 gap |','|---|---:|---|---|---:|---:|---:|']
    for c in oracles:
        if c['intensity']=='I3':lines.append(f'| {c["cell_id"]} | {c["B_mib"]:.9f} | {json.dumps({m:n for m,n in c["oracle_deployment"].items() if n},ensure_ascii=False)} | {c["oracle_N_G"]}/{c["oracle_N_S"]} | {c["oracle_z"]:.9f} | {c["RC_SG_gap"]:.9f} | {c["v2_gap"]:.9f} |')
    lines +=['','7. 8F-2 没有呈现类别转变有两个原因：结构最优本身全程 Specialist-heavy；同时 v2 在 I3 的全部 memory points 返回 Phi+4S seed。正式 runtime objective 在同一非 steady tier 内不会因更低 mean latency 自动接受；高 intensity candidates 的 unstable/rollback 证据和 oracle 是否进入 Top-B/被测量保存于 search_structural_gap_audit.csv。所有隐藏 2G oracle 未被 GPU evaluated。不能从低 z 推断它一定改善真实 latency。',
f'','8. RC-SG：oracle deployment 相同 {RC["equal_oracle_deployment_cells"]}/12；Pareto {RC["Pareto_cells"]}/12；严格正 structural gap {RC["strict_positive_gap_cells"]}/12。gap mean/median/max={RC["mean"]:.12f}/{RC["median"]:.12f}/{RC["max"]:.12f}。不定义“近最优”阈值；完整分布在 JSON/CSV。',
f'','9. v2：oracle deployment 相同 {V["equal_oracle_deployment_cells"]}/12；Pareto {V["Pareto_cells"]}/12；严格正 gap {V["strict_positive_gap_cells"]}/12。gap mean/median/max={V["mean"]:.12f}/{V["median"]:.12f}/{V["max"]:.12f}。',
'','10. Uniform load scaling: r(alpha)=alpha*r_base and z(alpha)=alpha*z_base. At fixed budget, structural-oracle composition cannot change with alpha while it remains feasible; if its minimum z exceeds rho_max no deployment is feasible. RC-SG base demand and all greedy values scale by the same alpha, so its choices are unchanged. Specialist throughput advantage is retained; absolute pressure difference grows with load, relative ratio does not. More load is not a mechanism for naturally hurting Specialists here.','', '| C12 method | z | feasible memory | feasible capacity | alpha capacity boundary |','|---|---:|---|---|---:|']
    for method in ['Generalist_only_fallback','Frozen_Generalist_centric','RC_SG','structural_oracle','v2']:
        r=next(r for r in decomp if r['cell_id']=='C12' and r['method']==method)
        lines.append(f'| {method} | {r["max_pressure"]:.9f} | {r["memory_feasible"]} | {r["capacity_feasible"]} | {r["alpha"]*r["capacity_scale_limit_relative_to_cell"]:.9f} |')
    lines +=['','11. Memory counterfactual events are exact affine residency/Pareto events, not selected coarse multipliers. Main event table:','', '| Budget | fastest-S residency lost above c_M | first oracle S count reduction | first forced G multi-task sharing |','|---|---:|---:|---:|']
    for B in sorted(set(c['memory_budget_mib'] for c in cells)):
        c=next(c for c in cells if c['memory_budget_mib']==B and c['intensity_level']=='I3');ev={r['event']:r for r in memory['events'] if r['cell_id']==c['cell_id']};a=ev['FIRST_LOSS_1G_ALL_FASTEST_SPECIALISTS_RESIDENCY'];b=ev['FIRST_ORACLE_SPECIALIST_COUNT_REDUCTION'];d=ev['FIRST_FORCED_GENERALIST_MULTI_TASK_SHARING_REGION'];lines.append(f'| {B:.9f} | {a["raw_residency_critical_c"]:.12f} ({a["status"]}) | {b["c_M"]} | {d["c_M"]} |')
    lines +=['','At M1, c_M>1 immediately loses minimum coverage residency: this is infeasibility, not a successful Generalist trade-off. M6 first identity switch is c_M=1.0601398985 (2G+5S remains); first S count reduction/forced sharing is c_M=1.8965280387 (2G+4S remains Specialist-heavy). Later Hybrid/Generalist-heavy counterfactual regions must be checked against the exported capacity flag, not presented as executable wins.','', '12. Speed counterfactual uses the fastest quality-admitted G per task and alpha multiplies the actual service-time difference. alpha=0 equalizes service time, but cheap additional Specialist workers still add capacity. Observed transition brackets:','', '| Budget | first identity change alpha bracket | first G/S count change alpha bracket |','|---|---|---|']
    for B in sorted(set(c['memory_budget_mib'] for c in cells)):
        rs=[r for r in speed['transitions'] if r['B_mib']==B and r['identity_changed']];cs=[r for r in rs if r['composition_changed']];a=max(rs,key=lambda r:r['alpha_speed_midpoint']) if rs else None;b=max(cs,key=lambda r:r['alpha_speed_midpoint']) if cs else None
        lines.append(f'| {B:.9f} | {[a["alpha_speed_lower"],a["alpha_speed_upper"]] if a else "No observed change"} | {[b["alpha_speed_lower"],b["alpha_speed_upper"]] if b else "No observed change"} |')
    lines +=['','257-point screen and <=1e-8 bisection brackets are documented. Continuous sub-grid nonmonotone islands are not exhaustively excluded; no unsupported event-completeness claim. Counterfactuals are explanatory diagnostics only.','',
f'13. RC-SG starts from Phi whose admission forces four fast Specialists as capability repairs; only Distil-Whisper is subsequently added in loose cells. Specialist final identity set matches oracle in {equalS}/12 cells. Text/image/OCR choices are largely separate, but speech paths overlap (ASR/LID), Generalists share tasks, and min-max pressure couples the routing problem. Greedy demand-saving is not a pressure objective; zero-demand-gain parallel workers can still lower z. Both the near-separable local repair structure and the remaining global interactions are explicitly recorded.',
f'','14. Diagnosis={diagnosis}. Specialist-heavy residency is structurally favored; RC-SG is not exactly pressure-optimal in most cells. Bounded v2 misses sustained loose-budget multi-Generalist alternatives. No Generalist-heavy Pareto region exists, so this is not evidence that hidden Generalist-heavy deployments beat Specialist-heavy ones.','',
'15. Implication: interference/congestion prediction、routing、robustness 与 bounded-search quality 是主要发展方向；也应测试隐藏 multi-Generalist Hybrid 的压力优势是否转化为真实 runtime 优势。完整 9A/9B/9C 不能凭空改变 frozen s0/memory 的结构偏向。后续独立设计真实 G/S trade-off regime 可能有必要，8F-2 永久保留。此处未开始完整 9A 或修改任何实验输入，完成后停止。',
'','Execution notes: all work CPU-only. New audit code initially encountered a frozen-file field-name compatibility error, heterogeneous CSV-schema error, and missing Workload import; original error logs retained. No original file/source/data was repaired or regenerated. All frozen references reverified at closeout.']
    (OUT/'module8f3_summary.md').write_text('\n'.join(lines)+'\n')
    expand_all_state_rows()
    verify(read(OUT/'audit_manifest.json')['frozen_refs'])
    required=['model_pool_audit.csv','task_level_dominance.csv','specialist_removal_cost.csv','all_deployments_structural.csv','pareto_all_states.csv','pareto_front_by_cell.csv','pareto_composition_summary.csv','structural_oracle_12cells.csv','intensity_capacity_decomposition.csv','counterfactual_memory_breakpoints.csv','counterfactual_speed_breakpoints.csv','counterfactual_tradeoff_map.csv','rcsg_greedy_optimality.csv','structural_diagnosis.json','module8f3_summary.md']
    assert all((OUT/f).is_file() for f in required)
    save('validation_results.json',{'status':'PASS','STRUCTURAL_AUDIT_ONLY':True,'GPU_RUNS':0,'enumerated_vectors':13824,'covered_LP_states':4326,'all_state_cell_rows':165888,'required_outputs':required,'all_frozen_refs_unchanged':True,'source_9A1_gate':'GATE_PARTIAL','source_9A2_gate':'REPEATABILITY_STRONG_SUPPORT','complete_Module9A_started':False,'counterfactuals_changed_benchmark':False})
    save('execution_source_hashes.json',{str(f.resolve()):sha(f) for f in Path(__file__).parent.glob('*.py')})
    save('artifact_hashes.json',{str(f.relative_to(OUT)):{'path':str(f.resolve()),'sha256':sha(f)} for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='artifact_hashes.json'})
    print('REPORT COMPLETE',diagnosis,RC,V,flush=True)

if __name__=='__main__': main()
