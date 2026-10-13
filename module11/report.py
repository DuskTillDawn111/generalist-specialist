"""Evidence-qualified report. Never turn empirical caps into exhaustion claims."""
from collections import Counter
from .common import *
from .envelope import fixed_lp

def main():
    c,hw,roles,cal=inputs();freeze=read(OUT/'input_freeze.json');bg=freeze['background']
    assert read(OUT/'cpu_validation.json')['status']=='PASS'
    windows=read(OUT/'windows.json')['rows'];fits=read(OUT/'capacity_witnesses.json')['rows']
    latency=read(OUT/'latency_witnesses.json')['rows'];structure=read(OUT/'structural_witnesses.json')['rows']
    budgets=read(OUT/'budget_structure_audit.json')['budgets'];states=read(OUT/'deployment_space.json')
    for r in latency:
        observed=r.get('observed_two_G_J_advantage',r['latency_beneficial_in_empirical_space'])
        r['observed_two_G_J_advantage']=observed
        r['latency_beneficial_in_empirical_space']=bool(observed and not any(r['unresolved_better_state_count'].values()))
    save('latency_witnesses.json',{'rows':latency});csvsave('oneG_vs_twoG_j.csv',latency)
    for b in budgets:
        two=sorted((s for s in states[b['budget']] if s['N_G']>=2),key=lambda s:s['safe_memory_mib'])
        witness=None
        for state in two:
            witness=fixed_lp(state['deployment'],bg,{},c,roles,cal,'2G',alpha=0.)
            if witness:break
        b['two_G_operational_background_witness']=witness
    save('budget_structure_audit.json',{'budgets':budgets,'GPU_runs':0});csvsave('budget_structure_audit.csv',budgets)
    positive=[w for w in windows if w['W_2G']>0]
    certified=[w for w in positive if w['replica_certificate']]
    latency_positive=[r for r in latency if r['latency_beneficial_in_empirical_space']]
    latency_certified=[r for r in latency_positive if r['SECOND_GENERALIST_LATENCY_BENEFICIAL']]
    w1=max(windows,key=lambda w:w['W_1G']);w2=max(windows,key=lambda w:w['W_2G'])
    order=lambda w:(-w['R_2G'],-w['W_2G'],int(w['budget'][1:]),tuple(w['tasks']))
    diagnostic=min(positive,key=order) if positive else None
    selected=min(certified,key=order) if certified else None
    fallback=min(latency_certified,key=lambda r:(-r['DeltaJ_2G_seconds'],int(r['budget'][1:]),tuple(r['tasks']))) if latency_certified and not selected else None
    selection={'status':'PRIMARY_FROZEN' if selected or fallback else 'NO_CERTIFIED_SECOND_GENERALIST_REGION_REPLICA_EVIDENCE_REQUIRED',
        'selection_rule':freeze['primary_tie_break'],'selected':selected or fallback,
        'diagnostic_empirical_capacity_candidate':diagnostic,'diagnostic_candidate_is_not_GPU_primary':not bool(selected or fallback),
        'certified_capacity_candidates':certified,'certified_latency_candidates':latency_certified,
        'NO_SECOND_GENERALIST_REGION_CURRENT_POOL':False,
        'why_not_no_region':'Positive conditional windows exist; unmeasured feasible replicas prevent either an absence claim or a physical necessity claim.',
        'GPU_runs':0,'new_GPU_execution_authorized':False}
    save('primary_multihotspot_candidate.json',selection)
    points=[]
    if selected or diagnostic:
        win=selected or diagnostic;s,a,b=win['alpha_S'],win['alpha_1G'],win['alpha_2G']
        heads=next(f['headroom'] for f in fits if f['budget']==win['budget'] and f['set_id']==win['set_id'])
        for label,alpha in [('A0',.8*s),('A1',.98*s),('A2',s+.3*(a-s)),('A3',s+.7*(a-s)),('A4',a+.3*(b-a)),('A5',a+.7*(b-a))]:
            points.append({'point':label,'alpha':alpha,'lambda_rps':rates(bg,heads,alpha),
                'capacity_region':'S-feasible' if alpha<=s else '1G-overflow' if alpha<=a else '2G-necessary-IN-EMPIRICAL-SPACE-ONLY'})
    save('future_hotspot_points.json',{'frozen':bool(selected),'GPU_runs':0,'GPU_execution_authorized':False,
        'status':'FROZEN_CPU_ONLY' if selected else 'NOT_FROZEN_REPLICA_EVIDENCE_REQUIRED',
        'points':points if selected else [],'diagnostic_unfrozen_points':points if not selected else [],
        'context':selected or diagnostic,'no_points_moved_using_GPU_outcomes':True})
    unique=Counter();operational_reject=0
    for line in (OUT/'J_solver_diagnostics.jsonl').open():
        row=json.loads(line);unique[row['solver_status']]+=1
        if row['certificate'] and not row['certificate']['passed']:operational_reject+=1
    latstats=Counter()
    for row in latency:latstats.update(row['enumeration_status_counts'])
    stage=read(OUT/'latency_stage_status.json')
    stage['statistics']=dict(latstats);stage['unique_state_solves']=sum(unique.values())
    stage['completed_contexts_resumed_without_replacing_outcomes']=True
    save('latency_stage_status.json',stage)
    bestlat=max(latency_positive,key=lambda r:r['DeltaJ_2G_seconds']) if latency_positive else None
    gate={'gate':'MULTIHOTSPOT_2G_SUPPORTED' if certified else 'UNDETERMINED_REPLICA_EVIDENCE_REQUIRED',
        'audit_execution_status':'COMPLETE_CPU_ONLY_AND_STOPPED','requested_gate_assignment_complete':bool(certified),
        'conditional_empirical_count_space_gate':'MULTIHOTSPOT_2G_SUPPORTED' if positive else None,
        'certified_S_to_1G_contexts':sum(w['replica_certificate'] and w['W_1G']>0 for w in windows),
        'conditional_S_to_1G_contexts':sum(w['W_1G']>0 for w in windows),
        'conditional_two_G_capacity_contexts':len(positive),'certified_two_G_capacity_contexts':len(certified),
        'certified_two_G_latency_contexts':len(latency_certified),
        'budget_contexts_replica_evidence_incomplete':sum(not w['replica_certificate'] for w in windows),
        'why_neither_1G_ONLY_nor_NOT_SUPPORTED':'B1 supports S->1G; larger budgets contain positive conditional 2G windows and missing replica evidence. Neither universal no-2G nor no-overflow follows.',
        'no_artificial_replica_cap_used_for_claims':True,'GPU_runs':0,'Module9_modified':False,
        'next_step_recommendation':'Resolve replica evidence and real multi-resource accounting before an exhaustion GPU benchmark; no current evidence requires Task-Rich.',
        'final_GPU_primary_frozen':bool(selected or fallback),'formal_A0_A5_frozen':bool(selected)}
    save('module11a_gate.json',gate)
    summary={'contexts':len(windows),'envelopes':len(fits),'J_comparisons':len(latency),'max_W_1G':w1,'max_W_2G':w2,
        'best_uncensored_empirical_latency_advantage':bestlat,
        'observed_two_G_J_advantage_comparisons':sum(r['observed_two_G_J_advantage'] for r in latency),
        'uncensored_empirical_latency_advantage_comparisons':len(latency_positive),
        'uncensored_empirical_latency_advantage_contexts':len({(r['budget'],r['set_id']) for r in latency_positive}),
        'J_comparisons_with_unresolved_potentially_better_states':sum(any(r['unresolved_better_state_count'].values()) for r in latency),
        'unique_frozen_J_solver_outcomes':dict(unique),'unique_operational_route_rejects':operational_reject,
        'latency_enumeration_counts':dict(latstats),
        'S_count_increase_low_to_near_contexts':sum(bool(r['S_count_increases_from_low']) for r in structure),
        'S_identity_change_low_to_near_contexts':sum(r['S_identity_change_from_low'] for r in structure),
        'one_G_plus_S_lower_min_memory_than_exact_two_G_contexts':sum('ONE_G_PLUS_SPECIALISTS_HAS_LOWER_MINIMUM_FEASIBLE_MEMORY_AT_NEAR_BOUNDARY' in r['structural_reasons'] for r in structure),
        'gate':gate,'GPU_runs':0}
    save('summary_metrics.json',summary)
    lines=[
        '# Module 11A — Multi-Hotspot / Second-Generalist Structural Audit',
        '',
        'CPU-only 审计已完成并停止；新 GPU runs = 0。Module 9、模型池、quality/capability、s0、memory、empirical g(x)、worker/runtime 未修改。',
        '',
        '**正式结论：证据不足以认证 2G 必要区域，也不足以宣布当前模型池没有 2G 区域。** gate 记录为 `UNDETERMINED_REPLICA_EVIDENCE_REQUIRED`，而不是强行落入 SUPPORTED / 1G_ONLY / NOT_SUPPORTED。B1 有认证的 S→1G 扩展；B6 有条件性的 2G window，但可容纳未测 Specialist replicas。',
        '',
        '## 范围与容量定义',
        '',
        'Eligible tasks：ASR、Image、LID、OCR、Text；VQA 无 admitted Specialist，不纳入热点组合。机械枚举 10 pairs + 10 triples × 6 budgets = 120 contexts；3 envelopes/context = 360。background 为 frozen mix_03/I1。方向直接复用逐任务 operational headroom，没有结果驱动搜索。',
        '',
        '所有 LP/MILP capacity witness 均满足 admitted path、routing conservation、safe memory 和 D≤0.8g。N_G 计 replica 数，Qwen×2 也是 2G；alpha_S 仅禁止热点→G，并未禁止 background 使用 G，也未人为限制 Specialist replicas。',
        '',
        '缺少 g(x)、M(x) 的 count 未外推。证据支持的 space 为 6912 个 count vectors；该 space 的 optimum 是诊断值，不能将其 evidence bound 当作物理上限。missing-count memory screen 使用已测 prefix + 原单 replica residency，并明确标为 screen，未扩展 calibration table。',
        '',
        '## 用户问题逐项回答',
        '',
        '1. **哪些预算允许 2G？** B4/B5/B6 有已测 count 的 2G residency，且有 rho≤0.8 background witness。最低 admitted 全任务 coverage 的已测 2G safe memory 为 33678.870215 MiB（Phi+Qwen+PaddleOCR；这只是 coverage，不是该最小状态的 operational witness）。B1/B2 在 frozen residency screen 下放不下 2G。B3 的 Phi×2 screen 可放下（最低 30084.146816 MiB），但 Phi×2 的 memory/scaling 没有测量，必须保持 unresolved，不能把它判为物理不可能。',
        '',
        '| Budget | Safe budget MiB | 2G classification | Replica certificate |',
        '|---|---:|---|---|']
    for b in budgets:lines.append(f"| {b['budget']} | {b['memory_budget_mib']:.6f} | {b['status']} | {'PASS' if b['replica_certificate'] else 'EVIDENCE REQUIRED'} |")
    lines.extend(['',
        f"2. **最大 W_1G：{w1['W_1G']:.12f}**，B1 ASR+LID（另有同值 triples）。alpha_S={w1['alpha_S']:.12f}，alpha_1G={w1['alpha_1G']:.12f}。该 B1 window 通过 replica residency screen；共 {gate['certified_S_to_1G_contexts']} 个 B1 contexts 支持 S→1G operational planning 扩展。没有新 GPU stability 验证。",'',
        f"3. **最大 W_2G：{w2['W_2G']:.12f}**，B6 OCR+Text，alpha_1G={w2['alpha_1G']:.12f}，alpha_2G={w2['alpha_2G']:.12f}。这是 measured-count-space 条件值；未测 PaddleOCR/文本 Specialist replicas 可容纳，不能认证真正 Specialist exhaustion。",'',
        f"4. **真正 1G infeasible / 2G feasible？** 已测空间中有 {len(positive)} 个 contexts，全部在 B6；每个 window midpoint 经 fixed LP 验证 2G feasible，且高于全空间 1G MILP upper bound。但物理范围内未测 replicas 的 upper bound 尚不完整，正式 `SECOND_GENERALIST_CAPACITY_NECESSARY=true` 数量为 0。",'',
        f"5. **1G feasible 但 2G 的 J 更低？** 有 {len(latency_positive)} 个未被 solver censoring 影响的条件性比较，跨 {summary['uncensored_empirical_latency_advantage_contexts']} 个 contexts。最高未 censor 的收益为 {bestlat['budget']} {bestlat['set_id']}，alpha={bestlat['alpha']:.12f}：J 从 {1000*bestlat['J_1G_seconds']:.9f} ms 降到 {1000*bestlat['J_2G_seconds']:.9f} ms，DeltaJ={1000*bestlat['DeltaJ_2G_seconds']:.9f} ms。两路线 max rho 为 {bestlat['best_1G']['max_rho']:.9f}/{bestlat['best_2G']['max_rho']:.9f}。这些是 surrogate、非真实 E2E 标签；仍受缺失 replica evidence 限制。",'',
        '6. **没有 2G 是否因为 Specialist 太快/太便宜？** 不能得出这一单一因果结论。B1/B2 首先是 memory；B3 是 G replica evidence；B4/B5 的已测空间中，2G residency 与背景/热点 Specialist capacity 竞争，同样的 alpha 常只能由 1G+S 承载；B6 已出现条件性 2G 扩展。未知 S scaling 仍是核心 proof gap。', '',
        f"7. **Specialist replicas 是否继续压倒 2G？** 低点到近 alpha_1G 的 minimum-memory witness 中，{summary['S_count_increase_low_to_near_contexts']}/120 contexts 增加 Specialist counts，{summary['S_identity_change_low_to_near_contexts']}/120 改变 S identity。{summary['one_G_plus_S_lower_min_memory_than_exact_two_G_contexts']}/120 中同一 workload 的 1G+S minimum feasible memory 比 exact 2G 更低。但 minimum-memory audit 不等于 frozen J 的选择，更不证明所有未测 S replicas 都优于 2G；B6 的正 window 反证了已测空间内的‘总是压倒’。", '',
        '8. **当前六任务池值得进入多热点 GPU 吗？** 有值得继续核验的 CPU 结构，不能宣布六任务池不合适。但当前不应直接作为严格的 Specialist-exhaustion / 2G-necessity GPU benchmark：所有 2G capacity candidates 未通过完整 replica evidence certificate。此次没有 GPU 执行。', '',
        f"9. **冻结哪个 context 和 A0–A5？** 正式 GPU primary 与 A0–A5 均未冻结。仅按预先冻结的 R2/W2/budget/task tuple 规则保存诊断候选：{diagnostic['budget']} {diagnostic['set_id']}，mix_03/I1；alpha_S={diagnostic['alpha_S']:.12f}，alpha_1G={diagnostic['alpha_1G']:.12f}，alpha_2G={diagnostic['alpha_2G']:.12f}，R2={diagnostic['R_2G']:.12f}。下表只是机械计算的未冻结值，不可作为已获认证的 GPU points。", '',
        '| Diagnostic point | Alpha |', '|---|---:|'])
    for p in points:lines.append(f"| {p['point']} | {p['alpha']:.12f} |")
    lines.extend(['',
        '该诊断候选的 capacity maximizing 1G/2G 状态分别为 Qwen+S 与 Phi+Qwen+S；target ASR 的 G fraction 为 0.092793413/0.152418568，Image/LID 保持 S。原 1G selected state 已含 ResNet×2、Whisper-small×2、ConvNeXt、Distil-Whisper、Whisper-turbo，second G 并不是靠人为禁止这些 measured S 扩展得到的。但在 near-boundary memory witness 中，再增加 Distil-Whisper、Whisper-small 或 Whisper-turbo 的 replica memory screen 仍可行，而 g 未测；这正是不能将 conditional window 冻结为 exhaustion benchmark 的原因。', '',
        '10. **Task-Rich 还是真实 multi-resource accounting？** 先处理 replica evidence 与真实 multi-resource accounting。Module10 的 operational certificate 合法状态仍出现 runtime instability，说明规划 proxy 的 capacity 不能自动等价于稳定 runtime capacity；而 Module11 已有潜在 2G 结构，因此尚无证据表明必须加 task。此处只给后续建议，不开始任何 profiling/accounting 实验。', '',
        '## J audit 的可复核边界', '',
        f"冻结 landmark rule 生成 {len(latency)} 个 workload 比较；严格枚举所有 memory/coverage-feasible measured states，使用 task-wise convex relaxation lower bound 安全剪枝。未修改 frozen J objective、初始化、solver 或数值阈值。", '',
        '冻结 solver 实际使用 rho<1 的域，而 operational admission 要求 rho≤0.8。原 solver route 通过独立 60-digit Decimal certificate 后才计入 operationally accepted state；真实超限不当作 floating-point 问题，也不通过改约束/换 route 修复。', '',
        f"唯一 CPU state/workload 检查共 {sum(unique.values())} 个：{dict(unique)}；其中 {operational_reject} 个已收敛 route 超过 operational ceiling。另有一个 conservation exception，记录为 NUMERICAL_SOLVER_EXCEPTION，非 physical infeasibility。没有 random restart、没有改 Module9、没有 GPU 重跑。", '',
        f"{summary['J_comparisons_with_unresolved_potentially_better_states']}/{len(latency)} 个比较有 potential-better numerical/operational censoring；这些不能声称已经取得受约束的全局 J*。CSV 的 J_1G/J_2G 是 best accepted frozen-router outputs，伴随 unresolved count 和完整 witness。53 个 observed positive witness comparisons 中，只有 {len(latency_positive)} 个未受这种 censoring 影响。Benefit 判定只使用现有 numerical certificate interval 是否仍严格正，没有人为性能 threshold。", '',
        '## 验证与停止', '',
        '360 个容量 witnesses 已校验所有约束；105 个 envelope 通过独立逐 deployment LP 枚举交叉检查；18 个重复 MILP 的 deployment、routing、alpha 完全一致；20 个条件性 2G region midpoint 全部验证。completion_verification.json 记录已有 Module10（含 invalid archive）及 Module9/source/runtime 文件哈希的最终复核。', '',
        '该模块执行完成。正式 gate 未被强行分配，不生成虚假的 no-2G 结论，不启动 GPU，不进入 Task-Rich。'])
    (OUT/'module11a_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'gate':gate['gate'],'diagnostic_candidate':diagnostic,'best_J_delta_ms':1000*bestlat['DeltaJ_2G_seconds'],'new_GPU_runs':0}),flush=True)

if __name__=='__main__':main()
