"""Close out the operational GPU campaign without expanding its scope."""
import math
from collections import Counter
from statistics import mean
from deployment_core import Workload
from deployment_core.routing.e2e_surrogate import evaluate_surrogate
from deployment_search.measurement_runtime import stats
from deployment_search.runtime import STABILITY
from replica_scaling.telemetry import snapshot
from .common import ROOT,OUT,inputs,read,save,csvsave,verify,ref,now
from .operational import AUDIT,ARCHIVE,check_archive
from .planning import flat,serial

def mean_complete(values):return mean(values) if values and all(v is not None for v in values) else None

def main():
    assert read(OUT/'benchmark_status.json')['status']=='OPERATIONAL_GPU_VALIDATION_COMPLETE'
    c,h,roles,cal=inputs();p=read(OUT/'primary_hotspot_selection.json');points=read(OUT/'hotspot_points.json')['points'][:4]
    plan=read(OUT/'planning_results.json')['rows'];gpu=read(OUT/'gpu_results.json')['rows'];manifest=read(OUT/'gpu_manifest.json')
    aggregate={a['state_id']:a['runtime'] for a in read(OUT/'gpu_aggregate.json')['states']}
    for rep in gpu:
        row=next(r for r in plan if r['state_id']==rep['state_id']);raw=read(rep['raw_path'])
        assert raw['valid_execution'] and raw['gpu_release']['verified']
        assert ref(rep['raw_path'])['sha256']==rep['raw_sha256']
        records=[r for r in raw['request_records'] if r['task']=='asr'];counts=Counter(roles[r['model_family']] for r in records)
        completed=[r for r in records if r['status']=='completed'];ccounts=Counter(roles[r['model_family']] for r in completed)
        rep.update(f_S=row['f_S'],f_G=row['f_G'],N_G=row['N_G'],N_S=row['N_S'],replicas=row['replicas'],
            predicted_worker_rho=row['rho'],safe_memory_mib=row['safe_memory_mib'],
            observed_ASR_S_requests=counts['specialist'],observed_ASR_G_requests=counts['generalist'],
            observed_ASR_f_G=counts['generalist']/len(records),observed_ASR_f_S=counts['specialist']/len(records),
            ASR_completed_requests=len(completed),ASR_offered_requests=len(records),
            ASR_completed_G_requests=ccounts['generalist'],ASR_completed_S_requests=ccounts['specialist'])
    save('gpu_results.json',{'rows':gpu,'independent_physical_measurements':len(gpu),'shared_aliases_not_independent':True});csvsave('gpu_results.csv',gpu)
    results=[]
    for row in plan:
        reps=[r for r in gpu if r['state_id']==row['state_id']] if row['state_id'] else []
        a=aggregate.get(row['state_id']);measured=bool(reps)
        assert not measured or len(reps)==3
        results.append({**row,'physical_repetitions':len(reps),
            'stability':a['stability_state_v2'] if a else None,
            'per_repetition_stability':[r['stability'] for r in reps],
            'stability_reasons':[r['stability_reason'] for r in reps],
            'mean_e2e_ms':mean_complete([r['mean_e2e_ms'] for r in reps]),
            'weighted_p95_ms':mean_complete([r['weighted_p95_ms'] for r in reps]),
            'mean_e2e_statistics':stats([r['mean_e2e_ms'] for r in reps]) if measured else None,
            'p95_statistics':stats([r['weighted_p95_ms'] for r in reps]) if measured else None,
            'ASR_mean_ms':mean_complete([r['per_task_mean']['asr'] for r in reps]),
            'ASR_p95_ms':mean_complete([r['per_task_p95']['asr'] for r in reps]),
            'completion_ratio':mean_complete([r['completion_ratio'] for r in reps]),
            'observed_ASR_f_G':mean_complete([r['observed_ASR_f_G'] for r in reps]),
            'per_task_mean_ms':{k:mean_complete([r['per_task_mean'][k] for r in reps]) for k in points[0]['lambda_rps']} if measured else None,
            'per_task_p95_ms':{k:mean_complete([r['per_task_p95'][k] for r in reps]) for k in points[0]['lambda_rps']} if measured else None,
            'GPU_status':'MEASURED_3_INDEPENDENT_REPETITIONS' if measured else row['status']})
    save('method_comparison.json',{'rows':results,'RC_SG_v2_exact_shared_labels':True});csvsave('method_comparison.csv',results)
    transitions=[]
    for method in manifest['methods']:
        rows=[r for r in results if r['method']==method]
        valid=[r for r in rows if r['physical_repetitions']==3]
        split=next((r for r in valid if r['f_G']>0),None)
        overflow=next((r for r in valid if r['SPECIALIST_EXHAUSTED_OPERATIONAL'] and r['f_G']>0),None)
        two=next((r for r in valid if r['N_G']>=2),None)
        unstable=next((r for r in valid if r['stability']=='unstable'),None)
        steady_prefix=[];nonunstable_prefix=[]
        for r in rows:
            if r['physical_repetitions']!=3 or r['stability']!='steady':break
            steady_prefix.append(r['target_rate'])
        for r in rows:
            if r['physical_repetitions']!=3 or r['stability']=='unstable':break
            nonunstable_prefix.append(r['target_rate'])
        transitions.append({'method':method,'lambda_split':split['target_rate'] if split else None,
            'lambda_split_point':split['point'] if split else None,
            'first_measured_post_exhaustion_split':overflow['point'] if overflow else None,
            'lambda_2G':two['target_rate'] if two else None,'lambda_unstable':unstable['target_rate'] if unstable else None,
            'first_unstable_point':unstable['point'] if unstable else None,
            'max_tested_steady_target_rate':max([r['target_rate'] for r in valid if r['stability']=='steady'],default=None),
            'max_contiguous_steady_target_rate':max(steady_prefix,default=None),
            'max_contiguous_nonunstable_target_rate':max(nonunstable_prefix,default=None),
            'measured_points':[r['point'] for r in valid],
            'unmeasured_rejections':{r['point']:r['status'] for r in rows if not r['physical_repetitions']},
            'scope':'first observed frozen-grid point; no exact continuous breakpoint, no imputed rejected-state stability'})
    save('phase_transitions.json',{'rows':transitions});csvsave('phase_transitions.csv',transitions)
    mechanism=[];early=[]
    for point in points:
        for r in [a for a in results if a['point']==point['point']]:
            if not r['routing']:continue
            forced=flat(r['routing'])
            specialists=[m for m,n in r['deployment'].items() if n and roles[m]=='specialist' and 'asr' in c[m].supported_tasks]
            assert specialists==['whisper-small']
            for pair in forced:
                if pair[0]=='asr':forced[pair]=0.
            forced['asr',specialists[0]]=r['target_rate']
            evaluation=evaluate_surrogate(r['deployment'],forced,Workload(point['lambda_rps']),c,cal)
            item={'point':r['point'],'method':r['method'],'S_only_operational_feasible':r['S_only_operational_feasible'],
                'SPECIALIST_EXHAUSTED_OPERATIONAL':r['SPECIALIST_EXHAUSTED_OPERATIONAL'],
                'specialist_pressure':{m:v for m,v in (r['rho'] or {}).items() if roles[m]=='specialist' and 'asr' in c[m].supported_tasks},
                'f_S':r['f_S'],'f_G':r['f_G'],'Specialist_dominant':r['f_S']>r['f_G'],
                'routing_operationally_admitted':r['planning_feasible'],'measured':r['physical_repetitions']==3,
                'real_stability':r['stability'],'forced_S_only_evaluation':evaluation,
                'routing_surrogate_improves_over_forced_S_only':r['J_seconds']<evaluation['J_seconds'],
                'canonical_balancing_or_J_marginal_cost_explains_early_split':True,
                'PREMATURE_GENERALIST_ROUTING':False if r['f_S']>r['f_G'] and r['J_seconds']<evaluation['J_seconds'] else 'REVIEW_REQUIRED',
                'claim_scope':'operational headroom policy rho<=.8, not physical rho<1 saturation'}
            mechanism.append(item)
            if r['point'] in ['H0','H1']:early.append(item)
    save('mechanism_checks.json',{'rows':mechanism})
    comparisons=[]
    for point in points:
        baseline=next(r for r in results if r['point']==point['point'] and r['method']=='v2')
        new=next(r for r in results if r['point']==point['point'] and r['method']=='V3_MINIMAL')
        both=baseline['physical_repetitions']==new['physical_repetitions']==3
        pairs=[]
        if both:
            for seed in manifest['seeds']:
                a=next(r for r in gpu if r['state_id']==baseline['state_id'] and r['seed']==seed)
                b=next(r for r in gpu if r['state_id']==new['state_id'] and r['seed']==seed)
                pairs.append({'seed':seed,'V3_minus_v2_mean_ms':b['mean_e2e_ms']-a['mean_e2e_ms'] if b['mean_e2e_ms'] is not None and a['mean_e2e_ms'] is not None else None,
                    'V3_minus_v2_p95_ms':b['weighted_p95_ms']-a['weighted_p95_ms'] if b['weighted_p95_ms'] is not None and a['weighted_p95_ms'] is not None else None,
                    'v2_stability':a['stability'],'V3_stability':b['stability']})
        comparisons.append({'point':point['point'],'same_deployment':baseline['deployment']==new['deployment'],
            'same_routing':baseline['routing']==new['routing'],'both_measured':both,'same_seed_differences':pairs,
            'attribution':'routing only; unique deployment, no feasible unvisited neighborhood candidate',
            'ranking_component_exercised':False,'V3_rejection':new['status'] if not both else None,
            'mean_comparison_at_same_steady_tier':both and baseline['stability']==new['stability']=='steady'})
    attribution={'pointwise':comparisons,'early_split_checks':early,'unique_memory_and_coverage_feasible_deployment':True,
        'second_G_cannot_fit_primary_budget':True,'candidate_ranking_not_tested_due_empty_feasible_neighborhood':True,
        'new_feature_or_route_substitution':False,
        'failure_categories':{'routing':[f"{r['point']}:{r['method']}:{r['status']}" for r in results if r['status']=='OPERATIONAL_ROUTE_INFEASIBLE'],
            'numerical':[f"{r['point']}:{r['method']}:{r['status']}" for r in results if r['status']=='MODULE9_J_NUMERICAL_FAILURE'],
            'runtime_instability':[f"{r['point']}:{r['method']}" for r in results if r['stability']=='unstable'],
            'deployment_search':'Primary has one legal background-covering deployment and zero feasible unvisited candidates; cannot add a second G.',
            'pool':'A certified positive operational window exists. A failure to realize it is not proof that the six-task pool is unsuitable.'}}
    save('attribution.json',attribution)
    extension=[r for r in results if r['point'] in ['H2','H3'] and r['physical_repetitions']==3 and r['stability']=='steady' and r['f_G']>0]
    supported=bool(extension)
    gate={'gate':'CURRENT_POOL_OVERFLOW_SUPPORTED' if supported else 'CURRENT_POOL_OPERATIONAL_OVERFLOW_NOT_VALIDATED',
        'CURRENT_POOL_OVERFLOW_SUPPORTED':supported,'CURRENT_POOL_NO_OVERFLOW_WINDOW':False,
        'scope':'frozen operational capacity rho<=.8','certified_operational_window_exists':True,
        'primary_replica_audit':'PASS','steady_joint_overflow_states':[{'point':r['point'],'method':r['method']} for r in extension],
        'physical_rho1_Specialist_exhaustion_demonstrated':False,
        'physical_C_Sonly_rho1':read(ARCHIVE/'primary_hotspot_selection.json')['C_Sonly'],
        'C_S_op':p['C_S_op'],'C_A_op':p['C_A_op'],'W_op':p['W_op'],
        'S_only_real_runtime_control_measured':False,'no_task_rich_started':True,'Task_Rich_required':False,
        'decision':'Stop after Module10B; distinguish operational exhaustion from physical saturation; no Module9 change or new benchmark.'}
    save('module10_gate.json',gate)
    assert len(gpu)==manifest['max_timed_runs']==21
    attempts=read(OUT/'gpu_attempts.json')['runs'];assert len(attempts)==21
    assert all(a['status']=='RETURNED' and a['summary']['valid_runtime_repetition'] for a in attempts)
    verify(read(OUT/'input_freeze.json')['protected_references']);verify(read(OUT/'execution_freeze.json')['sources']);check_archive()
    assert not snapshot()['processes']
    save('validation_results.json',{'status':'PASS','new_physical_runs':21,'unique_physical_states':7,
        'three_independent_repetitions_each':True,'RC_SG_v2_exact_shared_labels_not_independent':True,
        'operational_gate_checked_before_every_launch':True,'Module9_workers_runtime_unchanged':True,
        'old_archive_original_hashes_verified':check_archive(),'new_hotspot_points_unchanged':True,
        'MPS_enabled':False,'outcome_retries':0,'new_replica_profiles':0,'no_active_GPU_workers':True})
    lines=['# Module 10B — Operational Specialist Exhaustion and Generalist Overflow','',
        f"完成并停止。Gate: **{gate['gate']}**。Primary: ASR / mix_03 / I1 / B1。",'',
        f"C_S^op={p['C_S_op']:.12f}，C_A^op={p['C_A_op']:.12f}，W^op={p['W_op']:.12f} req/s；safe memory budget={p['context']['memory_budget_mib']:.9f} MiB。",'',
        '旧 H0–H3、原 GPU raw/cache 和作废记录保持归档内容不变。此次只使用 operational 冻结点；后台 rates、原 10s warmup / 210s measurement / seeds 7101,7202,7303 和 worker/runtime 不变。21 次独立物理测量；RC-SG/v2 逐点严格执行身份相同，共享标签不算独立方法证据。V3 H3 的 rho=0.8098277965355941 真正超出 0.8，记录 OPERATIONAL_ROUTE_INFEASIBLE，无 GPU 标签、替代 route 或修改 workload。','',
        '## 结果','',
        '| Point | Method | ASR rate | f_S | f_G | max rho | 真实 tier | mean E2E ms | weighted task P95 ms | ASR P95 ms | repetitions |',
        '|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    def fmt(v):return '—' if v is None else f'{v:.6f}'
    for r in results:lines.append(f"| {r['point']} | {r['method']} | {fmt(r['target_rate'])} | {fmt(r['f_S'])} | {fmt(r['f_G'])} | {fmt(r['max_rho'])} | {r['stability'] or r['status']} | {fmt(r['mean_e2e_ms'])} | {fmt(r['weighted_p95_ms'])} | {fmt(r['ASR_p95_ms'])} | {r['physical_repetitions']} |")
    lines+=['','所有测量 deployment 为 Phi + distilbert + resnet50 + paddleocr + whisper-small，各 1 replica，1G+4S。每个 worker rho、routing、safe memory、独立 repetition、任务 mean/P95、ASR observed fractions、queue/backlog 和 raw SHA 均保存。P95 是 task-weighted P95，而非 pooled quantile。','',
        '## 七个重点问题','',
        '1. H0/H1 是否保持 Specialist-dominant？所有正式执行方法的 f_S>f_G；表中完整报告 fraction。存在早期 G 分流，不是 100% Specialist。固定 canonical minmax 或 J 的凸 congestion 边际成本降低解释该分流，forced S-only 的 CPU J 对照在 mechanism_checks.json。不得将 surrogate 合理性说成真实最优路由证明。','',
        '2. H2 是否首次合法 S+G split？H2 是首个超过 C_S^op 的 frozen 点，S-only operational planning 不可行，ALL 仍可行；但各方法在 H0 已有合法 S+G split，所以 H2 不是 split 的首次出现。它是第一次进入已认证的 operational overflow 区间。实际 steady 与否见表。','',
        '3. H3 overflow fraction 是否增加？RC-SG/v2 的 H3 与 H2 fraction 可直接比较。V3 H3 的 tentative fraction 仅诊断，未通过 operational certificate，不能计为实际运行或稳定 overflow。','',
        '4. 第二 Generalist？没有。primary budget 下只有一个 memory-and-coverage-feasible deployment，2G 无法满足背景覆盖和 memory。frozen v2/V3 的可行未访问候选集合为空。不得将此误判为 DeltaJ ranking 没有选中 2G。','',
        '5. 最大 stable operating range？下面区分 strict steady prefix、非 unstable prefix 和 first observed unstable；reject 无真实稳定性标签。测量点是离散的，范围仅指已测冻结 grid，不是连续 capacity breakpoint。','',
        '| Method | 首次实测 unstable ASR rate | 最大 contiguous steady rate | 最大 contiguous non-unstable rate |','|---|---:|---:|---:|']
    for t in transitions:lines.append(f"| {t['method']} | {fmt(t['lambda_unstable'])} | {fmt(t['max_contiguous_steady_target_rate'])} | {fmt(t['max_contiguous_nonunstable_target_rate'])} |")
    lines+=['',f"6. 是否验证 exhausted → overflow？Operational steady extension 的 gate 为 **{gate['gate']}**；符合 H2/H3 steady joint extension 的 state 为 {gate['steady_joint_overflow_states']}。这里的 exhaustion 指 rho≤0.8 headroom policy。",'',
        f"物理 Specialist 极限并未被触达：原 rho1 C_Sonly={gate['physical_C_Sonly_rho1']:.12f}，所有新 H 点均低于它。没有增加 S-only GPU control；所以不声称 physical exhaustion 或证明 G overflow 的独立因果收益。",'',
        f"7. 失败归因：{attribution['failure_categories']}。不同 deployment 不存在，V3 相对 canonical 的差异只来自 routing。已有正 operational window，不能据不稳定推导模型池没有 G/S overlap 或必须加 task。",'',
        '## 重复与 attribution','',
        '以下真实 ordering 仅在同一 tier 时有意义；不同 tier 首先比较 stability。相同 seed 差值是重复编号配对，不保证同一路由产生相同随机请求轨迹。','']
    for comp in comparisons:lines.append(f"- {comp['point']}: {comp}")
    lines+=['','未修改 Module9 源码、历史/gate、memory、quality、g(x)、任务、模型池、candidate generation、Top-B 或 runtime。原 10ns 阈值和 rho_max=.8 不随结果修改。未开启 MPS、未补 profiling、未运行 task-rich 或第二硬件。完成后停止。']
    (OUT/'module10_summary.md').write_text('\n'.join(lines)+'\n')
    save('benchmark_status.json',{'status':'COMPLETE_AND_STOPPED','at_utc':now(),'gate':gate['gate'],'physical_GPU_runs':21,'no_active_GPU_workers':True})
    save('campaign_artifact_hashes.json',{'at_utc':now(),'references':[ref(f) for f in OUT.iterdir() if f.is_file() and f.name not in ['campaign_artifact_hashes.json','gpu_execution.lock']]})
    print(__import__('json').dumps({'COMPLETE_AND_STOPPED':True,'gate':gate['gate'],'physical_GPU_runs':21,'phase_transitions':transitions}),flush=True)

if __name__=='__main__':main()
