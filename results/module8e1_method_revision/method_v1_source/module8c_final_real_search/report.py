"""Read-only analysis; no counterfactual campaign, deployment, or search."""
import csv,collections,statistics,math
from .common import *
from deployment_core.replica_calibration import ReplicaPlanningPolicy
from deployment_core.replicas import replica_statistics,replica_key
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluation
from integer_replica_multiworkload.report import audit_candidates,audit_runtime,csv_write,percent_gain
from candidate_pool_audit.provenance import environment_snapshot

def report():
    freeze=verify_freeze();catalog,hardware,roles,cal=inputs()
    policy=ReplicaPlanningPolicy(catalog,'empirical','empirical',cal)
    inventory=read(P/'candidate_pool_v2_1.json')
    fullroles={m['id']:m['role'] for m in inventory['models'] if m['id'] in inventory['formal_model_ids']}
    def full(x):return {m:x.get(m,0) for m in sorted(fullroles)}
    allowed={(r['task'],r['model']) for r in read(B/'quality_admission_matrix.json')['rows'] if r['quality_admissible']}
    errors=[];rows=[];decisions=[];runtime_states=[];activations={};generalists=[];confirmrows=[];rankreversals=[]
    total_search=total_confirm=rawchecked=0
    objective=NoiseAwareRuntimeObjective(1.,.05)
    for name in freeze['execution_order']:
        d=OUT/name;trace=read(d/'search_trace.json');conf=read(d/'confirmation_results.json')
        protocol=read(d/'protocol.json')
        assert trace['enumeration_calls']==0 and trace['runtime_evaluation_count']<=6 and trace['search_iterations']<=3
        assert file_hash(d/'search_trace.json')==read(d/'post_search_freeze.json')['search_trace_sha256']
        candidate_rows=audit_candidates(name,trace,policy,roles,errors)
        runtime,memory=audit_runtime(d,name,trace,conf,hardware,errors)
        total_search+=runtime['search_physical_attempts'];total_confirm+=conf['physical_attempts']
        assert conf['complete'] and conf['physical_attempts']==4
        assert len(conf['repetitions']['seed'])==len(conf['repetitions']['final'])==2
        assert runtime['valid_unique_runtime_states']<=6 and runtime['search_physical_attempts']==3*runtime['valid_unique_runtime_states']
        cache=read(d/'search_cache/runtime_cache_manifest.json')
        reps=[r['runtime'] for e in cache['configurations'].values() for r in e['repetitions']]
        reps+=[r for rr in conf['repetitions'].values() for r in rr]
        for r in reps:
            raw=read(r['raw_result_path']);rawchecked+=1
            assert r['valid_runtime_repetition'] and raw['valid_execution']
            assert raw['workload_rps']==protocol['workload']['arrival_rates_rps']
            assert raw['measurement_seconds']==210 and raw['warmup_seconds']==10
            assert raw['measurement_window']['duration_seconds']==210
            assert raw['runtime_context_sha256']==read(d/'runtime_context.json')['context_sha256']
            for task,mrates in raw['routing_rps'].items():
                assert math.isclose(sum(mrates.values()),raw['workload_rps'][task],rel_tol=1e-7,abs_tol=1e-12)
                for m,rps in mrates.items():
                    if rps>0:assert (task,m) in allowed and raw['deployment'][m]>0
            for field in ('request_records','warmup_request_records','priming_request_records'):
                for request in raw[field]:assert (request['task'],request['model_family']) in allowed
            assert all(n<=freeze['replica_bounds_full'][m] for m,n in raw['deployment'].items())
            assert len(set(v['pid'] for v in raw['worker_identities'].values()))==sum(raw['deployment'].values())
        for a in candidate_rows:
            a['incumbent']=full(a['incumbent']);a['deployment']=full(a['deployment'])
            a['N_G']=replica_statistics(a['deployment'],fullroles)['N_G']
            a['N_S']=replica_statistics(a['deployment'],fullroles)['N_S']
            a['runtime_feedback_reversal']=bool(a['runtime_evaluated'] and a['delta_z'] is not None and a['delta_z']>0 and a['runtime_better_than_incumbent'] is False)
            a['runtime_preference_overridden_by_other_Top_B']=a['reason']=='another_top_b_candidate_preferred'
        decisions+=candidate_rows
        for iteration in trace['iterations']:
            evaluated=sorted([a for a in iteration['candidates'] if a['runtime'] is not None],key=lambda a:a['rank'])
            if len(evaluated)==2:
                first,second=evaluated
                runtime_prefers_second,reason=objective.compare(ReplicaRuntimeEvaluation(**second['runtime']),
                          ReplicaRuntimeEvaluation(**first['runtime']),second['memory_mib'],first['memory_mib'])
                rankreversals.append({'workload':name,'iteration':iteration['iteration'],
                     'analytical_first':first['mutation']['model'],'analytical_second':second['mutation']['model'],
                     'runtime_prefers_analytical_second':runtime_prefers_second,'objective_reason':reason,
                     'accepted_model':next((a['mutation']['model'] for a in evaluated if a['status']=='ACCEPTED'),None),
                     'scope':'Comparison of actually evaluated Top-B under the unchanged noise-aware runtime objective'})
        for key,entry in cache['configurations'].items():
            rt=read(entry['aggregate']);x=full(entry['identity']['deployment'])
            runtime_states.append({'workload':name,'cache_key':key,'deployment':x,**replica_statistics(x,fullroles),
               'Generalist_identity':[m for m,n in x.items() if n and fullroles[m]=='generalist'],
               'Specialist_identity':[m for m,n in x.items() if n and fullroles[m]=='specialist'],
               'x2_models':[m for m,n in x.items() if n==2],'planning_memory_mib':policy.deployment_memory(entry['identity']['deployment']),
               'stability':rt['stability_state_v2'],'mean_e2e_ms':rt['weighted_mean_e2e'],
               'weighted_task_p95_ms':rt['weighted_p95_e2e'],'valid_repetitions':rt['valid_repetition_count'],
               'runtime':rt,'source':ref(entry['aggregate'])})
        seed,final=full(trace['seed']['deployment']),full(trace['deployment']);a,b=trace['seed_runtime'],trace['runtime']
        ca,cb=conf['aggregates']['seed'],conf['aggregates']['final']
        row={'workload':name,'seed':seed,'final':final,'seed_N_G':replica_statistics(seed,fullroles)['N_G'],
          'seed_N_S':replica_statistics(seed,fullroles)['N_S'],**replica_statistics(final,fullroles),
          'Generalist_identity':[m for m,n in final.items() if n and fullroles[m]=='generalist'],
          'Specialist_identity':[m for m,n in final.items() if n and fullroles[m]=='specialist'],
          'seed_memory_mib':policy.deployment_memory(trace['seed']['deployment']),
          'final_memory_mib':policy.deployment_memory(trace['deployment']),
          'actual_final_idle_mib_mean':statistics.mean(o['actual_idle_mib'] for o in b['memory_observations']),
          'sampled_final_peak_device_mib':max(o['peak_device_mib'] for o in b['memory_observations']),
          'seed_stability':a['stability_state_v2'],'final_stability':b['stability_state_v2'],
          'seed_mean_e2e_ms':a['weighted_mean_e2e'],'final_mean_e2e_ms':b['weighted_mean_e2e'],
          'mean_improvement_percent':percent_gain(a['weighted_mean_e2e'],b['weighted_mean_e2e']),
          'seed_weighted_task_p95_ms':a['weighted_p95_e2e'],'final_weighted_task_p95_ms':b['weighted_p95_e2e'],
          'p95_improvement_percent':percent_gain(a['weighted_p95_e2e'],b['weighted_p95_e2e']),
          'runtime_states':trace['runtime_evaluation_count'],'search_physical_attempts':runtime['search_physical_attempts'],
          'confirmation_physical_attempts':4,'search_status':trace['status'],'enumeration_count':trace['enumeration_calls'],
          'accepted_transitions':sum(c['decision']=='ACCEPTED' for c in candidate_rows),
          'runtime_rollbacks':sum(c['decision']=='ROLLBACK' for c in candidate_rows),
          'runtime_feedback_reversals':sum(c['runtime_feedback_reversal'] for c in candidate_rows),
          'evaluated_x2_models':sorted({m for s in runtime_states if s['workload']==name for m in s['x2_models']}),
          'accepted_x2_models':sorted({m for c in candidate_rows if c['decision']=='ACCEPTED' for m,n in c['deployment'].items() if n==2}),
          'final_x2_models':[m for m,n in final.items() if n==2],
          'confirmation_seed_mean_ms':ca['weighted_mean_e2e'],'confirmation_final_mean_ms':cb['weighted_mean_e2e'],
          'confirmation_mean_improvement_percent':percent_gain(ca['weighted_mean_e2e'],cb['weighted_mean_e2e']),
          'confirmation_seed_weighted_p95_ms':ca['weighted_p95_e2e'],'confirmation_final_weighted_p95_ms':cb['weighted_p95_e2e'],
          'confirmation_p95_improvement_percent':percent_gain(ca['weighted_p95_e2e'],cb['weighted_p95_e2e']),
          'confirmation_seed_stability':ca['stability_state_v2'],'confirmation_final_stability':cb['stability_state_v2'],
          'confirmation_objective_direction_maintained':conf['objective_direction_maintained'],
          'confirmation_objective_reason':conf['objective_reason'],'source_trace':ref(d/'search_trace.json')}
        rows.append(row);confirmrows.append({k:v for k,v in row.items() if k=='workload' or k.startswith('confirmation_')})
        save_json(d/'result_summary.json',row);save_json(d/'memory_validation.json',memory)
        save_json(d/'integer_replica_decisions.json',candidate_rows);save_json(d/'runtime_evaluations.json',runtime)
        activations[name]={'evaluated':row['evaluated_x2_models'],'accepted':row['accepted_x2_models'],'final':row['final_x2_models'],
                           'all_x2_candidates':[c for c in candidate_rows if c['new_count']==2]}
        for m in ('qwen','phi4mm','minicpm'):
            tested=[s for s in runtime_states if s['workload']==name and s['deployment'][m]>0]
            changes=[c for c in candidate_rows if c['candidate']==m]
            generalists.append({'workload':name,'model':m,'present_in_seed':seed[m]>0,'present_in_final':final[m]>0,
              'generated_mutations':len(changes),'runtime_states_containing_model':len(tested),
              'accepted_mutations':sum(c['decision']=='ACCEPTED' for c in changes),'rollback_mutations':sum(c['decision']=='ROLLBACK' for c in changes),
              'states':[{'cache_key':s['cache_key'],'deployment':s['deployment'],'mean_e2e_ms':s['mean_e2e_ms'],
                        'weighted_task_p95_ms':s['weighted_task_p95_ms'],'stability':s['stability'],
                        'vqa_mean':s['runtime']['per_task_mean'].get('visual_qa'),
                        'vqa_p95':s['runtime']['per_task_p95'].get('visual_qa')} for s in tested],
              'mutations':changes,'quality_admissible_VQA':True,
              'interpretation':'Co-deployed addition/removal competition within visited states; no unmeasured single-Generalist winner or global optimum inferred'})
    assert not errors,errors
    assert len(rows)==6 and total_search<=108 and total_confirm==24
    assert environment_snapshot()==read(OUT/'environment_manifest.json')
    signatures=collections.Counter(tuple(sorted(r['final'].items())) for r in rows)
    diversity={'distinct_final_deployments':len(signatures),'all_final_same':len(signatures)==1,
       'final_state_groups':[{'deployment':dict(x),'workloads':[r['workload'] for r in rows if tuple(sorted(r['final'].items()))==x]} for x in signatures],
       'distinct_N_G_values':sorted({r['N_G'] for r in rows}),'distinct_N_S_values':sorted({r['N_S'] for r in rows}),
       'distinct_Generalist_identity_sets':sorted({tuple(r['Generalist_identity']) for r in rows}),
       'distinct_Specialist_identity_sets':sorted({tuple(r['Specialist_identity']) for r in rows}),
       'planning_distinct_seeds':read(B/'workload_diversity.json')['seed_diversity']['distinct_count'],
       'no_VQA_final_N_G':next(r['N_G'] for r in rows if r['workload']=='no_vqa_phase'),
       'no_VQA_N_G_zero_preserved':next(r['N_G']==0 for r in rows if r['workload']=='no_vqa_phase'),
       'planning_vs_runtime_scope':'Planning diversity gate does not imply final deployment diversity; this file reports actual results without retuning workload.'}
    summary={'module':'8C','status':'PASS','workloads':rows,'final_deployment_diversity':diversity,
      'total_search_physical_attempts':total_search,'total_confirmation_physical_attempts':total_confirm,
      'total_valid_campaign_physical_attempts':total_search+total_confirm,
      'failed_premeasurement_physical_attempts':1,'total_physical_attempts':total_search+total_confirm+1,'enumeration_count':0,
      'runtime_feedback_reversals':[c for c in decisions if c['runtime_feedback_reversal']],
      'analytical_vs_runtime_Top_B_preferences':rankreversals,'audit_errors':errors,
      'quality_admissible_routing_paths':21,'raw_repetitions_checked':rawchecked,
      'frozen_inputs_and_algorithms_unchanged':True,'baseline':False,'ablation':False,'dynamic':False,'second_hardware':False,
      'stopped_after_module':'8C','scope':'Six frozen workloads, v2.1, user-selected admission policy, one RTX4090, fixed Top-B/runtime budget; no global optimum or general dominance claim'}
    save_json(OUT/'module8c_summary.json',summary);save_json(OUT/'final_deployment_diversity.json',diversity)
    save_json(OUT/'replica_activation.json',activations);save_json(OUT/'runtime_feedback_reversal.json',
         {'nonimproving_analytical_positive_candidates':summary['runtime_feedback_reversals'],'Top_B_runtime_rank_comparisons':rankreversals})
    save_json(OUT/'generalist_competition.json',generalists);save_json(OUT/'runtime_state_inventory.json',runtime_states)
    save_json(OUT/'confirmation_summary.json',confirmrows)
    csv_write(OUT/'six_workload_summary.csv',rows,list(rows[0]));csv_write(OUT/'integer_replica_decisions.csv',decisions,list(decisions[0]))
    csv_write(OUT/'generalist_competition.csv',generalists,list(generalists[0]))
    save_json(OUT/'audit_validation.json',{'status':'PASS','protected_files_checked':len(freeze['protected_files']),
       'raw_repetitions_checked':rawchecked,'quality_routing_edges_checked':True,'full_12_model_vectors_recorded':True,
       'original_inputs_and_search_algorithm_unchanged':True,'budget_per_workload_max_unique':6,'search_repetitions_per_state':3,
       'confirmation_repetitions_per_seed_and_final':2,'enumeration_count':0,'errors':errors})
    lines=['# Module 8C — Final Real Multi-workload Search','',
       f'**PASS，六组搜索与 independent confirmation 完成。** Search physical attempts={total_search}；confirmation attempts={total_confirm}；另保留一次 premeasurement startup failure；enumeration=0。完成后停止。','',
       '第一轮在 PaddleOCR 启动阶段因 libnvrtc.so.13 搜索路径失败，measurement 未开始、request_records=0、valid states=0、GPU 已释放。完整失败目录保存于 results/module8c_final_real_search_startup_failure_001/。修复仅复用历史 OCR LD_LIBRARY_PATH；新的执行环境及恢复原因在 startup_recovery.json 中冻结。六组有效搜索的 states/repetitions/confirmation 预算未增加，没有丢弃已测结果或更改 frozen 8B inputs。','',
       '严格使用 8B 冻结 pool v2.1、21 条 admitted routing edges、27/27 demand catalog 和六组 λ。PaddleOCR-VL 固定 x=0；JSON/CSV 保存包括此零项的全部 12-type integer vector。未修改 quality/capability/λ/算法。Predictor=None、Top-B=2、每组≤6 unique runtime states，每 state 三 repetitions；seed/final 各二独立 confirmation，不影响搜索决策。沿用 max_iterations=3、patience=1、NoiseAwareRuntimeObjective(1ms,5%)、210s measurement/10s warmup、empirical g/M 与 evidence bounds。','',
       '|Workload|Final deployment|N_G/N_S|Mean E2E ms (seed→final)|Mean gain|Weighted task P95 ms (seed→final)|P95 gain|Stability|States|',
       '|---|---|---|---|---|---|---|---|---|']
    def sparse(x):return ', '.join(f'{m}={n}' for m,n in x.items() if n)
    for r in rows:lines.append(f"|{r['workload']}|{sparse(r['final'])}|{r['N_G']}/{r['N_S']}|{r['seed_mean_e2e_ms']:.3f}→{r['final_mean_e2e_ms']:.3f}|{r['mean_improvement_percent']:.2f}%|{r['seed_weighted_task_p95_ms']:.3f}→{r['final_weighted_task_p95_ms']:.3f}|{r['p95_improvement_percent']:.2f}%|{r['final_stability']}|{r['runtime_states']}/6|")
    lines+=['','## Final deployment diversity','',
       f"实际 distinct final deployments={diversity['distinct_final_deployments']}；N_G values={diversity['distinct_N_G_values']}；N_S values={diversity['distinct_N_S_values']}。",
       'Planning DIVERSIFIED 与实际 final diversity 分别记录。所有重复状态如实保留，不通过修改 workload 或扩大预算追求差异。']
    for group in diversity['final_state_groups']:lines.append(f"- {', '.join(group['workloads'])}: {sparse(group['deployment'])}")
    reasoning=next(r for r in rows if r['workload']=='reasoning_vqa_heavy')
    lines+=['','## Reasoning/VQA-heavy Generalist competition','',
       f"Final Generalists={reasoning['Generalist_identity']}；搜索 seed 为 Phi + 四 Specialists。以下是实际 mutation evidence，新增 Generalist 可能与 Phi 同时驻留，不能等价为单 Generalist 替换胜负。"]
    for g in [g for g in generalists if g['workload']=='reasoning_vqa_heavy']:
        lines.append(f"- {g['model']}: seed={g['present_in_seed']}，final={g['present_in_final']}，containing runtime states={g['runtime_states_containing_model']}，accepted mutations={g['accepted_mutations']}，rollback mutations={g['rollback_mutations']}。")
        for c in g['mutations']:
            if c['runtime_evaluated']:lines.append(f"  - iteration={c['iteration']}，x {c['old_count']}→{c['new_count']}，analytical rank={c['analytical_rank']}，mean={c['real_mean_e2e_ms']:.3f}ms，weighted P95={c['real_weighted_task_p95_ms']:.3f}ms，stability={c['stability']}，{c['decision']} ({c['reason']})。")
    no=next(r for r in rows if r['workload']=='no_vqa_phase')
    lines+=['','## No-VQA and replicas','',
       f"No-VQA: seed N_G={no['seed_N_G']}，final N_G={no['N_G']}；N_G=0 preserved={diversity['no_VQA_N_G_zero_preserved']}。Final={sparse(no['final'])}；search stability={no['final_stability']}，confirmation stability={no['confirmation_final_stability']}。",
       f"Final x=2 activation by workload={json.dumps({r['workload']:r['final_x2_models'] for r in rows},ensure_ascii=False)}。所有 x=2 proposals 的 generated/pruned/evaluated/accepted/rollback 证据见 replica_activation.json，未测候选不推断真实优劣。",'',
       '## Runtime feedback and independent confirmation','',
       f"Analytical positive delta_z 却不改善实际 incumbent 的 reversal={len(summary['runtime_feedback_reversals'])}；实际 Top-B 第二名被 runtime objective 优先的次数={sum(r['runtime_prefers_analytical_second'] for r in rankreversals)}。另一个 Top-B 更好的 rollback 与不改善 incumbent 分开记录。",'',
       '|Workload|Confirmation mean seed→final ms|Mean gain|Confirmation P95 seed→final ms|P95 gain|Objective direction|','|---|---|---|---|---|---|']
    for r in rows:lines.append(f"|{r['workload']}|{r['confirmation_seed_mean_ms']:.3f}→{r['confirmation_final_mean_ms']:.3f}|{r['confirmation_mean_improvement_percent']:.2f}%|{r['confirmation_seed_weighted_p95_ms']:.3f}→{r['confirmation_final_weighted_p95_ms']:.3f}|{r['confirmation_p95_improvement_percent']:.2f}%|{r['confirmation_objective_direction_maintained']} ({r['confirmation_objective_reason']})|")
    lines+=['','Confirmation 即使不维持 objective gain，也如实保留，不回改已冻结 final。seed=final 时仍独立执行 2+2，不缓存复用。',
       '', 'Memory、stability、per-task 与 per-replica metrics、CUDA worker identities 和 telemetry 在 raw 与 memory_validation.json 中保留；sampled peak 不是精确瞬时峰值。VQA/LID 使用已授权语义准入，数值 quality threshold 仍为 UNSPECIFIED。',
       '未启动 baseline、ablation、dynamic、第二硬件实验。结论只覆盖冻结 workload、单平台、证据 bounds 和已访问状态，不声称全局最优或某 Generalist 普遍占优。']
    (OUT/'module8c_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'event':'module8c_report','final_diversity':diversity,'search_attempts':total_search,'confirmation_attempts':total_confirm}),flush=True)

def artifact_manifest():
    save_json(OUT/'artifact_manifest.json',{'module':'8C','artifacts':{str(p.relative_to(OUT)):ref(p) for p in sorted(OUT.rglob('*'))
           if p.is_file() and p.name not in ('artifact_manifest.json','.campaign.lock','.runtime_cache.lock','campaign.log')}})
if __name__=='__main__':report();artifact_manifest()
