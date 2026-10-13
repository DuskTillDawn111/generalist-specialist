"""Descriptive factorial reporting using unchanged frozen lexicographic comparator."""
import csv
import copy
from collections import Counter
from types import SimpleNamespace
from .common import *
from .variants import flags_dict
from deployment_core.replica_calibration import ReplicaPlanningPolicy
from deployment_core.replicas import replica_statistics
from deployment_search.measurement_runtime import stats
from module8e3_heldout_validation.report import compare, candidate_audit
from module8e3_final_audit import normalize_candidates
from deployment_search.runtime import STABILITY

PAIRS=[('M10','M00'),('M01','M00'),('M11','M00'),('M11','M10'),('M11','M01')]


def csvwrite(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def classify_factor(rows, mechanism, valid):
    if not valid:return 'INSUFFICIENT'
    w=sum(r['outcome']=='WIN' for r in rows);l=sum(r['outcome']=='LOSS' for r in rows)
    if l>w or any(r['stability_regression'] or r['severe_mean_regression'] for r in rows):return 'NOT_SUPPORTED'
    if w>=2 and l==0 and mechanism:return 'SUPPORTED'
    if w>l and w>=1:return 'PARTIAL'
    return 'NOT_SUPPORTED'


def report():
    verify_execution();catalog,hardware,roles,cal=inputs();policy=ReplicaPlanningPolicy(catalog,'empirical','empirical',cal)
    workloads=read(F/'heldout_workloads.json')['workloads'];results={m:{} for m in METHODS};traces={m:{} for m in METHODS}
    metrics={m:[] for m in METHODS};candidates={m:{} for m in METHODS}
    for w in workloads:
        name=w['id']
        for m in METHODS:
            source='M1' if m=='M00' else 'M2' if m=='M11' else m
            base=E3 if m in ['M00','M11'] else OUT
            path=base/name/source/'result.json';r=read(path) if path.exists() else None;results[m][name]=r
            folder=('m1_search_traces' if m=='M00' else 'm2_search_traces' if m=='M11' else m.lower()+'_search_traces')
            tp=base/folder/name/'search_complete.json';tr=read(tp) if tp.exists() else None;traces[m][name]=tr
            cs=candidate_audit(name,'M1' if m=='M00' else m,tr,policy);normalize_candidates(cs,{'labels_normalized':0});candidates[m][name]=cs
            if m in NEW:save_json(OUT/folder/name/'audit_trace.json',{'method':m,'workload':name,'candidates':cs,'flags':flags_dict(m)})
            if m in ['M00','M11']:
                old=read(E3/('m1_results.json' if m=='M00' else 'm2_results.json'))
                metric=copy.deepcopy(next(x for x in old['workloads'] if x['workload']==name));metric['method']=m
                metric['reuse_source']=ref(path);metric['new_GPU_runs']=0
            else:
                final=r['confirmations']['final'] if r else None;x=r['deployment'] if r else None
                count=replica_statistics(x,roles) if x else {'N_G':None,'N_S':None}
                mem=final['memory_observations'] if final else []
                metric={'workload':name,'intensity':w['intensity'],'method':m,'status':'COMPLETE' if r else 'INCOMPLETE',
                    'deployment_vector':x,'N_G':count['N_G'],'N_S':count['N_S'],'total_replicas':sum(x.values()) if x else None,
                    'planning_memory_mib':policy.deployment_memory(x) if x else None,'n':2 if final else 0,
                    'weighted_mean_E2E_ms':final['weighted_mean_e2e'] if final else None,'weighted_task_P95_ms':final['weighted_p95_e2e'] if final else None,
                    'completion_ratio':final['completion_ratio'] if final else None,'throughput_rps':final['throughput'] if final else None,
                    'stability':final['stability_state_v2'] if final else None,
                    'peak_VRAM_mib':stats([a.get('peak_owned_mib') for a in mem]),'idle_VRAM_mib':stats([a.get('actual_idle_mib') for a in mem]),
                    'queue_wait':final['per_replica_queue_wait'] if final else None,'worker_latency':final['worker_latency'] if final else None,
                    'final_aggregate':final,'search_aggregate':r['search_runtime'] if r else None,
                    'seed_confirmation':r['confirmations']['seed'] if r else None,
                    'statistics':{'mean_E2E':stats([a['weighted_mean_e2e'] for a in final['repetitions']]) if final else None,
                        'weighted_task_P95':stats([a['weighted_p95_e2e'] for a in final['repetitions']]) if final else None}}
            metric['flags']=flags_dict(m);metrics[m].append(metric)
    acc=read(OUT/'run_accounting.json');sharing=read(OUT/'measurement_sharing_log.json');runs=read(OUT/'physical_measurements.json')['runs']
    donors=old_donors();donors.update({digest(r['identity']):r for r in runs if r.get('valid')})
    by_id={r['physical_raw_id']:r for r in donors.values()}
    checks=[]
    for r in sharing['logical_run_references']:
        d=by_id[r['physical_raw_id']]
        checks.append(r['measurement_identity_hash']==digest(d['identity']) and file_hash(r['raw_path'])==r['raw_sha256']==d['raw_sha256']
            and Path(r['raw_path']).resolve()==Path(d['raw_path']).resolve())
    expected=[(b['workload'],i,s) for b in read(OUT/'execution_order.json')['workloads'] for i,s in enumerate(b['slots'])]
    actual=[(s['workload'],s['slot_index'],s['slot']) for s in read(OUT/'execution_log.json')['slots']]
    complete=all(results[m][w['id']] for m in METHODS for w in workloads)
    confirmation_independent=True;budget_equal=True;barrier=True
    for m in NEW:
        for w in workloads:
            r=results[m][w['id']];tr=traces[m][w['id']]
            if not r:continue
            paths={k:{str(Path(a['raw_result_path']).resolve()) for a in agg['repetitions']} for k,agg in r['confirmations'].items()}
            confirmation_independent &= len(paths['seed'])==len(paths['final'])==2 and not bool(paths['seed']&paths['final'])
            budget_equal &= tr['config']==read(F/'method_comparison_freeze.json')['search_config'] and tr['runtime_evaluation_count']<=6 and tr['search_iterations']<=3
            assert r['logical_search_evaluations']==len([e for e in sharing['logical_evaluations'] if e['method']==m and e['workload']==w['id']])
    for e in sharing['logical_evaluations']:
        if e['role']=='candidate':barrier &= bool(e.get('round_barrier',{}).get(e['method']))
    audit={'complete':complete,'source_and_input_hashes_match':True,'endpoint_regression_equivalence':read(OUT/'regression_equivalence.json')['status']=='PASS',
        'all_raw_reuse_identity_and_hashes_match':all(checks),'execution_order_exact':actual==expected,
        'equal_budget':budget_equal,'confirmation_role_independence':confirmation_independent,'candidate_commitment_firewall':barrier,
        'M00_M11_new_GPU_runs':0,'within_budget':acc['within_budget']}
    valid=all(v for k,v in audit.items() if k!='M00_M11_new_GPU_runs');audit['valid']=valid
    rows=[];outcomes={};mechanisms=[];chains=[];replicas=[]
    for w in workloads:
        name=w['id'];outcomes[name]={}
        for a,b in PAIRS:
            ra,rb=results[a][name],results[b][name]
            fa=ra['confirmations']['final'] if ra else None;fb=rb['confirmations']['final'] if rb else None
            ma=policy.deployment_memory(ra['deployment']) if ra else None;mb=policy.deployment_memory(rb['deployment']) if rb else None
            c=compare(fa,fb,ma,mb);key=a+'_vs_'+b;outcomes[name][key]=c
            c['stability_regression']=bool(fa and fb and STABILITY[fa['stability_state_v2']]<STABILITY[fb['stability_state_v2']])
            c['severe_mean_regression']=bool(fa and fb and fa['stability_state_v2']==fb['stability_state_v2']=='steady' and fb['weighted_mean_e2e']>0 and fa['weighted_mean_e2e']/fb['weighted_mean_e2e']-1>=.2)
            rows.append({'workload':name,'intensity':w['intensity'],'comparison':key,**c,
                **{label+'_'+k:next(m for m in metrics[method] if m['workload']==name).get(k) for label,method in [('A',a),('B',b)] for k in ['weighted_mean_E2E_ms','weighted_task_P95_ms','peak_VRAM_mib','stability','N_G','N_S']}})
        for m in METHODS:
            cs=candidates[m][name];swaps=[c for c in cs if c['targeted_swap']];closures=[c for c in cs if c['replacement_closure']];rc=[c for c in cs if c['candidate_class']=='R']
            beneficial_rank_missed=[]
            for c in swaps:
                matches=[full for full in candidates['M11'][name] if full['runtime_evaluated'] and full['beneficial_to_actual_incumbent'] and full['candidate_deployment']==c['candidate_deployment'] and full['routing']==c['routing']]
                if matches and c['rank'] is not None and c['rank']>2 and not c['runtime_evaluated']:beneficial_rank_missed.append(c)
            mechanisms.append({'workload':name,'intensity':w['intensity'],'method':m,
                'targeted_swaps_generated':len(swaps),'replacement_closures_generated':len(closures),
                'replacements_Top_B':sum(c['Top_B'] for c in swaps),'replacements_evaluated':sum(c['runtime_evaluated'] for c in swaps),
                'replacements_accepted':sum(c['accepted'] for c in swaps),'beneficial_replacement_rank_gt2_missed':len(beneficial_rank_missed),
                'REMOVE_scale_down_generated':sum(c['operation_original'] in ['REMOVE','SCALE_DOWN'] for c in cs),
                'Class_R_Top_B':sum(c['Top_B'] for c in rc),'Class_R_evaluated':sum(c['runtime_evaluated'] for c in rc),
                'Class_R_accepted':sum(c['accepted'] for c in rc),'termination_reason':traces[m][name]['status'] if traces[m][name] else None})
            x2=[c for c in cs if c['contains_x2']]
            replicas.append({'workload':name,'method':m,'generated':len(x2),'Top_B':sum(c['Top_B'] for c in x2),
                'evaluated':sum(c['runtime_evaluated'] for c in x2),'accepted':sum(c['accepted'] for c in x2),'candidate_evidence':x2})
        for full in candidates['M11'][name]:
            if not(full['targeted_swap'] and full['accepted'] and full['beneficial_to_actual_incumbent']):continue
            matches10=[c for c in candidates['M10'][name] if c['candidate_deployment']==full['candidate_deployment'] and c['routing']==full['routing']]
            matches01=[c for c in candidates['M01'][name] if c['candidate_deployment']==full['candidate_deployment'] and c['routing']==full['routing']]
            chain=bool(matches10) and any(c['rank'] is not None and c['rank']>2 and not c['runtime_evaluated'] for c in matches10) and not matches01
            chains.append({'workload':name,'deployment':full['candidate_deployment'],'routing':full['routing'],'M11_iteration':full['iteration'],
                'M11_rank':full['rank'],'M10_matches':[{k:c[k] for k in ['iteration','rank','Top_B','runtime_evaluated','status','analytical_delta_z']} for c in matches10],
                'M01_generated_same_state':bool(matches01),'chain_supported':chain,'final_full_WIN':outcomes[name]['M11_vs_M00']['outcome']=='WIN'})
    counts={a+'_vs_'+b:{k:sum(outcomes[w['id']][a+'_vs_'+b]['outcome']==k for w in workloads) for k in ['WIN','TIE','LOSS']} for a,b in PAIRS}
    if valid:assert counts['M11_vs_M00']=={'WIN':2,'TIE':4,'LOSS':0},'Reused official endpoint result changed'
    factor_rows={key:[r for r in rows if r['comparison'] in contrasts] for key,contrasts in read(OUT/'ablation_spec.json')['claim_policy']['factor_contrasts'].items()}
    chain_names={c['workload'] for c in chains if c['chain_supported'] and c['final_full_WIN']}
    joint={w['id'] for w in workloads if outcomes[w['id']]['M11_vs_M00']['outcome']=='WIN' and all(outcomes[w['id']][m+'_vs_M00']['outcome']!='WIN' for m in ['M10','M01'])}
    r1=classify_factor(factor_rows['R1'],bool(chain_names),valid);r2=classify_factor(factor_rows['R2'],bool(chain_names),valid)
    interaction='INSUFFICIENT' if not valid else 'SUPPORTED' if len(chain_names&joint)>=2 else 'PARTIAL' if joint or chain_names else 'NOT_SUPPORTED'
    full_win={w['id'] for w in workloads if outcomes[w['id']]['M11_vs_M00']['outcome']=='WIN'}
    necessity={factor:bool(valid and full_win) and all(outcomes[n][method+'_vs_M00']['outcome']!='WIN' for n in full_win)
        for factor,method in [('R1','M01'),('R2','M10')]}
    full_regression=[r['workload'] for r in rows if r['comparison'] in ['M11_vs_M10','M11_vs_M01'] and r['outcome']=='LOSS']
    static_ready=valid and not full_regression and (r1=='SUPPORTED' or r2=='SUPPORTED') and interaction in ['SUPPORTED','PARTIAL']
    claims={'status':'PASS' if valid else 'INCOMPLETE_OR_PROTOCOL_INVALID','R1_REPLACEMENT_EFFECT':r1,'R2_DIVERSIFIED_TOPB_EFFECT':r2,
        'R1_R2_INTERACTION':interaction,'comparison_counts':counts,'claim_policy':read(OUT/'ablation_spec.json')['claim_policy'],
        'joint_only_WIN_workloads':sorted(joint),'mechanism_chain_workloads':sorted(chain_names),'mechanism_chains':chains,
        'necessary_for_observed_full_v2_wins_under_frozen_budget':necessity,'necessity_is_not_universal':True,
        'ablation_better_than_full_v2_workloads':sorted(set(full_regression)),'full_v2_regression_vs_ablation':sorted(set(full_regression)),
        'full_v2_regression_vs_v1':counts['M11_vs_M00']['LOSS']>0,'STATIC_ALGORITHM_FREEZE_EVIDENCE_SUFFICIENT':static_ready,
        'MODULE8F_PROTOCOL_DESIGN_READY':static_ready,'dynamic_executed':False,'ROUTING_REVISION_PENDING':True,'protocol_audit':audit,
        'sampling_limit':'Three mixtures, correlated moderate/high levels; final n=2, search n=3. Shared raw is not independent; no significance claim.'}
    factorial=[]
    for w in workloads:
        n=w['id'];means={m:next(r for r in metrics[m] if r['workload']==n)['weighted_mean_E2E_ms'] for m in METHODS}
        allsteady=all(next(r for r in metrics[m] if r['workload']==n)['stability']=='steady' for m in METHODS)
        delta=lambda a,b:means[a]-means[b] if allsteady else None
        factorial.append({'workload':n,'intensity':w['intensity'],'mean_E2E_ms':means,
            'R1_delta_R2_OFF_ms':delta('M10','M00'),'R1_delta_R2_ON_ms':delta('M11','M01'),
            'R2_delta_R1_OFF_ms':delta('M01','M00'),'R2_delta_R1_ON_ms':delta('M11','M10'),
            'interaction_difference_of_differences_ms':(means['M11']-means['M01'])-(means['M10']-means['M00']) if allsteady else None,
            'joint_only_WIN':n in joint,'mechanism_chain':n in chain_names,'primary_comparisons':outcomes[n]})
    for m in METHODS:save_json(OUT/(m.lower()+'_results.json'),{'method':m,'flags':flags_dict(m),'workload_role':'POST-VALIDATION ABLATION SET',
        'workloads':metrics[m],'all_valid_repetitions_retained':True,'reused_endpoint':m in ['M00','M11']})
    csvwrite('ablation_comparison.csv',rows);csvwrite('mechanism_comparison.csv',mechanisms)
    save_json(OUT/'factorial_effects.json',{'descriptive_only':True,'per_workload':factorial,'mechanism_chains':chains,'counts':counts,'classifications':{k:claims[k] for k in ['R1_REPLACEMENT_EFFECT','R2_DIVERSIFIED_TOPB_EFFECT','R1_R2_INTERACTION']}})
    save_json(OUT/'replica_audit.json',{'workloads_methods':replicas,'totals':{k:sum(r[k] for r in replicas) for k in ['generated','Top_B','evaluated','accepted']}})
    save_json(OUT/'ablation_claim_audit.json',claims)
    case=['# Mix_02 mechanism case study','', 'Static post-validation ablation; no revised algorithm or held-out claim after tuning.','']
    for w in workloads:
        if 'mix_02' not in w['id']:continue
        n=w['id'];case += ['## '+n,'']
        for m in METHODS:
            tr=traces[m][n];cs=candidates[m][n]
            target=next(c for c in candidates['M11'][n] if c['replacement_closure'] and c['accepted'])
            match=[c for c in cs if c['candidate_deployment']==target['candidate_deployment'] and c['routing']==target['routing']]
            case.append(f"- {m}: Phi→Qwen exact replacement generated={bool(match)}; records="+json.dumps([{k:c[k] for k in ['iteration','rank','Top_B','runtime_evaluated','accepted','analytical_delta_z','reason']} for c in match])+f"; stop={tr['status'] if tr else 'INCOMPLETE'}")
        case+=['','M00/M01 use only x±1, so cannot generate this two-count replacement in the same neighborhood. M10 generation and exact global rank are shown above; M11 generation and diversified Top-B admission are shown above. An unmeasured candidate is identified as beneficial only by a matching actually measured full-v2 state, with incumbent-path limitations.','']
    (OUT/'mix02_case_study.md').write_text('\n'.join(case))
    totals=read(OUT/'replica_audit.json')['totals']
    texts=['# Module 8E-4 — Minimal revision ablation','',f"Campaign={claims['status']}; R1={r1}; R2={r2}; interaction={interaction}.",'',
        '冻结 v1/v2、六个 workload、canonical routing、quality、candidate pool、objective、budget 与 seeds 均未变。用途标记 POST-VALIDATION ABLATION SET；ROUTING_REVISION_PENDING=true。','',
        '|比较|WIN|TIE|LOSS|','|---|---:|---:|---:|']
    for k,v in counts.items():texts.append(f"|{k}|{v['WIN']}|{v['TIE']}|{v['LOSS']}|")
    texts += ['', '## Required final answers','',
        f"1. 新增 timed runs={acc['new_timed_runs']}；confirmation={acc['confirmation_runs']}；retries={acc['retries']}；M00/M11 新增=0。",
        f"2. M10 vs M00={counts['M10_vs_M00']}。",f"3. M01 vs M00={counts['M01_vs_M00']}。",f"4. M11 vs M00={counts['M11_vs_M00']}，正式结果原样复用。",
        f"5. Replacement-only 是否足以产生 mix_02 improvement：{all(outcomes[w['id']]['M10_vs_M00']['outcome']=='WIN' for w in workloads if 'mix_02' in w['id']) if valid else 'INSUFFICIENT'}。",
        f"6. Diversified-only 有效 primary WIN 数={counts['M01_vs_M00']['WIN']}。",
        f"7. R1 对已观察 full-v2 WIN 必要（仅本集、本预算）={necessity['R1']}。",
        f"8. R2 对已观察 full-v2 WIN 必要（仅本集、本预算）={necessity['R2']}。",
        f"9. 两者协同={interaction}；joint-only workloads={sorted(joint)}。",
        f"10. 没有 diversified 时 ranking 漏掉的有益 replacement：{sum(r['beneficial_replacement_rank_gt2_missed'] for r in mechanisms if r['method']=='M10')} 次；exact mechanism chain={sorted(chain_names)}。",
        f"11. Ablation 比 full-v2 更好 workload={sorted(set(full_regression))}。",f"12. Full-v2 相对 v1 regression={counts['M11_vs_M00']['LOSS']}；相对 ablation={sorted(set(full_regression))}。",
        f"13. x=2 generated/Top-B/evaluated/accepted={totals}；含四方法逐轮 candidate occurrences，不是独立样本数。",
        f"14. R1_REPLACEMENT_EFFECT={r1}；R2_DIVERSIFIED_TOPB_EFFECT={r2}；R1_R2_INTERACTION={interaction}。",
        f"15. 足以冻结当前最终 static algorithm={static_ready}；该 freeze 不包含解决 routing。",
        f"16. 可以另行设计 Module 8F Dynamic Adaptation protocol={static_ready}；本阶段未执行 dynamic。",'',
        f"预算：new logical search evaluations={acc['logical_search_evaluations']}<=72，new unique physical states={acc['unique_physical_states']}，new timed runs={acc['new_timed_runs']}<={acc['frozen_caps']['new_timed_runs']}。",
        f"共享：E3 raw repetition references={acc['reused_E3_repetition_references']}，E4 raw sharing={acc['reused_E4_repetition_references']}；equal-budget={budget_equal}；exact sharing={all(checks)}；confirm independence={confirmation_independent}。",
        '所有有效 repetitions 保留 n/mean/std；复用的 raw 不计额外独立样本。不做显著性或普遍必要性推断。','',
        '本阶段停止；未修改 v1/v2，未执行 method v3 / routing revision / dynamic / resource sweep / second hardware。']
    (OUT/'module8e4_summary.md').write_text('\n'.join(texts)+'\n')
    save_json(OUT/'report_manifest.json',{'artifacts':{p.name:ref(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md'] and p.name!='report_manifest.json'},
        'source_hashes_unchanged':True,'all_valid_repetitions_retained':True,'new_revisions':0})


if __name__=='__main__':report()
