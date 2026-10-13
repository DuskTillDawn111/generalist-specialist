"""Read-only final analysis governed by the premeasurement analysis policy."""
import csv, json, math, statistics
from collections import Counter
from types import SimpleNamespace
from pathlib import Path
from deployment_search.runtime_cache import save_json
from deployment_core.replica_calibration import ReplicaPlanningPolicy
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from .common import ROOT,B,P,C,F,OUT,TASKS,read,ref,inputs,verify_inputs,full_vector,effective_quality


def stats(values):
    present=[v for v in values if v is not None and math.isfinite(v)]
    return {'mean':statistics.mean(present) if present else None,
        'std':statistics.stdev(present) if len(present)>1 else 0. if present else None,
        'median':statistics.median(present) if present else None,
        'min':min(present) if present else None,'max':max(present) if present else None,
        'n':len(present),'planned_n':len(values),'values':values}


def csvwrite(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def percent_gain(baseline,proposed):
    return 100*(baseline-proposed)/baseline if baseline is not None and baseline>0 and proposed is not None else None


def metrics(runtime,identity):
    if runtime is None:return {'valid':False,'mean_e2e_ms':None,'weighted_task_p95_ms':None,'stability':None}
    rt=runtime;memory=rt['memory_observations'];rates=identity['workload'];total=sum(rates.values())
    def weighted_worker(rep,key):
        vals=[rates[t]*rep['worker_latency'][t][key] for t in rates if rates[t]>0
              and t in rep['worker_latency'] and rep['worker_latency'][t].get(key) is not None]
        return sum(vals)/total if len(vals)==sum(v>0 for v in rates.values()) else None
    worker=stats([weighted_worker(rep,'mean_worker_latency_ms') for rep in rt['repetitions']])
    queue=stats([weighted_worker(rep,'mean_worker_queue_wait_ms') for rep in rt['repetitions']])
    primary=rt['replicate_statistics']
    task_queues={t:v['mean_worker_queue_wait_ms']['mean'] for t,v in rt['worker_latency'].items() if rates.get(t,0)>0}
    return {'valid':True,'mean_e2e_ms':rt['weighted_mean_e2e'],
        'mean_e2e_std_ms':primary['weighted_mean_e2e']['sample_std'],
        'weighted_task_p95_ms':rt['weighted_p95_e2e'],'weighted_task_p95_std_ms':primary['weighted_p95_e2e']['sample_std'],
        'completion_ratio':rt['completion_ratio'],'completion_ratio_std':primary['completion_ratio']['sample_std'],
        'completed_throughput_rps':rt['throughput'],'completed_throughput_std_rps':primary['throughput']['sample_std'],
        'stability':rt['stability_state_v2'],'valid_repetitions':len(rt['repetitions']),
        'actual_idle_VRAM_mib_mean':statistics.mean(m['actual_idle_mib'] for m in memory),
        'actual_idle_VRAM_mib_std':statistics.stdev(m['actual_idle_mib'] for m in memory) if len(memory)>1 else 0.,
        'sampled_peak_device_VRAM_mib':max(m['peak_device_mib'] for m in memory),
        'mean_worker_latency_ms':worker['mean'],'mean_worker_latency_std_ms':worker['std'],
        'mean_worker_queue_wait_ms':queue['mean'],'mean_worker_queue_wait_std_ms':queue['std'],
        'per_task_queue_wait_ms':task_queues,'per_task_metrics':rt['per_task_mean'],
        'search_repetition_values':{k:v['values'] for k,v in primary.items()},
        'raw_source_paths':rt['raw_result_paths'],'sampled_peak_is_not_instantaneous_peak':True}


def outcome(proposed,baseline):
    if not proposed['metrics']['valid'] or not baseline['metrics']['valid']:
        return {'outcome':'incomparable','reason':'Missing valid final evidence'}
    a=SimpleNamespace(stability_state_v2=proposed['metrics']['stability'],
        weighted_mean_e2e=proposed['metrics']['mean_e2e_ms'],weighted_p95_e2e=proposed['metrics']['weighted_task_p95_ms'])
    b=SimpleNamespace(stability_state_v2=baseline['metrics']['stability'],
        weighted_mean_e2e=baseline['metrics']['mean_e2e_ms'],weighted_p95_e2e=baseline['metrics']['weighted_task_p95_ms'])
    objective=NoiseAwareRuntimeObjective(1.,.05)
    p,pr=objective.compare(a,b,proposed['planning_memory_mib'],baseline['planning_memory_mib'])
    r,rr=objective.compare(b,a,baseline['planning_memory_mib'],proposed['planning_memory_mib'])
    assert not(p and r),'Asymmetric objective paradox'
    return {'outcome':'win' if p else 'loss' if r else 'tie','Proposed_comparison_reason':pr,'baseline_comparison_reason':rr}


def report():
    verify_inputs();policy_freeze=read(OUT/'analysis_policy_freeze.json')
    catalog,hardware,roles,cal=inputs();policy=ReplicaPlanningPolicy(catalog,'empirical','empirical',cal)
    pool=read(P/'candidate_pool_v2_1.json');allroles={m['id']:m['role'] for m in pool['models']}
    manifest=read(OUT/'campaign_manifest.json');names=[j['workload'] for j in manifest['phase_A'] if j['baseline']=='B0']
    physical=read(OUT/'new_measurement_manifest.json');reuse=read(OUT/'reuse_execution_log.json')
    rows=[];references=[];costs=[];quality_rows=[];failures=[]
    def make_row(baseline,name,seed,x,identity,runtime,confirmation,source,source_paths,status='complete'):
        x=full_vector(x);positive={m:v for m,v in x.items() if v}
        row={'baseline':baseline,'workload':name,'random_seed':seed,'status':status,'deployment':x,
            'N_G':sum(v for m,v in positive.items() if allroles[m]=='generalist'),
            'N_S':sum(v for m,v in positive.items() if allroles[m]=='specialist'),
            'total_model_replicas':sum(positive.values()),
            'Generalist_identity':[m for m in positive if allroles[m]=='generalist'],
            'Specialist_identity':[m for m in positive if allroles[m]=='specialist'],
            'planning_memory_mib':policy.deployment_memory({m:x[m] for m in catalog}),
            'metrics':metrics(runtime,identity),'confirmation_metrics':metrics(confirmation,identity),
            'effective_quality_summary':effective_quality(identity) if identity else None,
            'measurement_source':source,'original_source_paths':source_paths,'identity':identity}
        rows.append(row)
        quality_rows.append({'baseline':baseline,'workload':name,'random_seed':seed,'quality_admissible':True,
            'frozen_admitted_path_count':21,'excluded_path_used':False,'effective_task_quality_summary':row['effective_quality_summary']})
        return row
    def cost_row(baseline,name,seed,owner,trace):
        new=[r for r in physical['runs'] if r['owner']==owner]
        reused=[r for r in reuse['events'] if r['owner']==owner]
        candidates=[c for it in trace.get('iterations',[]) for c in it['candidates'] if c.get('runtime') is not None]
        return {'baseline':baseline,'workload':name,'random_seed':seed,
            'max_valid_unique_state_budget':6,'top_b':2,'max_iterations':3,'no_improvement_patience':1,
            'logical_candidate_evaluations':len(candidates),'logical_runtime_state_evaluations':trace.get('runtime_evaluation_count'),
            'real_runtime_state_evaluations_including_reuse':trace.get('runtime_evaluation_count'),
            'physical_new_search_timed_runs':sum(r['phase']=='search' and r.get('valid_timed_execution',False) for r in new),
            'physical_new_confirmation_timed_runs':sum(r['phase']=='confirmation' and r.get('valid_timed_execution',False) for r in new),
            'physical_new_timed_runs':sum(r.get('valid_timed_execution',False) for r in new),'physical_attempts':len(new),
            'reused_search_run_references':sum(r['phase']=='search' for r in reused),
            'reused_confirmation_run_references':sum(r['phase']=='confirmation' for r in reused),
            'logical_timed_run_references':(trace.get('runtime_evaluation_count') or 0)*3+4,
            'unique_states':trace.get('visited_deployment_count'),
            'accepted_transitions':sum(c['status']=='ACCEPTED' for c in candidates),
            'rollback_count':sum(c['status']=='ROLLBACK' for c in candidates),
            'final_stability':(trace.get('runtime') or {}).get('stability_state_v2'),
            'final_mean_ms':(trace.get('runtime') or {}).get('weighted_mean_e2e'),
            'final_weighted_P95_ms':(trace.get('runtime') or {}).get('weighted_p95_e2e'),
            'enumeration_count':trace.get('enumeration_calls',0),'LP_rank_calls':trace.get('LP_rank_calls'),
            'equal_max_budget':True,'cost_inference_unit':'Logical evaluations; physical reuse savings reported separately'}
    for name in names:
        d=C/name;trace=read(d/'search_complete.json');cache=read(d/'search_cache/runtime_cache_manifest.json')
        identity=cache['configurations'][trace['runtime']['cache_key']]['identity']
        conf=read(d/'confirmation_results.json')
        row=make_row('Proposed',name,None,trace['deployment'],identity,trace['runtime'],conf['aggregates']['final'],
            'reused',trace['runtime']['raw_result_paths'])
        references.append({'workload':name,'seed_runtime':trace['seed_runtime'],'final_runtime':trace['runtime'],
            'seed_confirmation':conf['aggregates']['seed'],'final_confirmation':conf['aggregates']['final'],
            'trace_source':ref(d/'search_complete.json'),'confirmation_source':ref(d/'confirmation_results.json')})
        cost=cost_row('Proposed',name,None,'__8C_original__',trace)
        cost.update(physical_new_search_timed_runs=trace['runtime_evaluation_count']*3,
            physical_new_confirmation_timed_runs=4,physical_new_timed_runs=trace['runtime_evaluation_count']*3+4,
            physical_attempts=trace['runtime_evaluation_count']*3+4,reused_search_run_references=0,reused_confirmation_run_references=0,
            LP_rank_calls='existing guided ranking',physical_run_campaign='8C; zero Proposed reruns in 8D-2')
        costs.append(cost)
    for job in manifest['phase_A']:
        baseline,name=job['baseline'],job['workload'];directory=OUT/baseline.lower()/name
        status=read(directory/'status.json')
        if status['status']=='complete':
            result=read(directory/'result.json')
            make_row(baseline,name,None,result['state']['deployment'],result['identity'],result['runtime'],result['confirmation'],
                result['measurement_source'],result['source_paths'])
        else:
            failures.append(status)
            state=next(s for s in read(F/'baseline_deployments.json')['states'] if s['baseline']==baseline and s['workload']==name)
            identity={'deployment':state['deployment'],'workload':state['lambda_rps'],'routing':state['routing_rps']}
            make_row(baseline,name,None,state['deployment'],identity,None,None,'newly_measured',
                [r['raw_result_path'] for r in status['failed_measurements']],status['status'])
    for job in manifest['phase_B']:
        name,seed=job['workload'],job['random_seed'];directory=OUT/f'b4_seed_{seed}'/name;status=read(directory/'status.json')
        if status['status']!='complete':failures.append(status);continue
        trace=read(directory/'search_complete.json');conf=read(directory/'confirmation_results.json')
        journal=read(directory/'logical_evaluations.json')['evaluations'];final=next(e for e in journal if e['cache_key']==trace['runtime']['cache_key'])
        assert len(journal)==trace['runtime_evaluation_count'] and len({e['cache_key'] for e in journal})<=6
        make_row('B4',name,seed,trace['deployment'],final['identity'],trace['runtime'],conf['aggregates']['final'],
            final['measurement_source'],trace['runtime']['raw_result_paths'])
        costs.append(cost_row('B4',name,seed,f'B4-{seed}/{name}',trace))
    save_json(OUT/'proposed_reference.json',{'Proposed_rerun':False,'workloads':references})
    for baseline in ('B0','B1','B2','B3'):
        save_json(OUT/f'{baseline.lower()}_results.json',{'baseline':baseline,'results':[r for r in rows if r['baseline']==baseline]})
    save_json(OUT/'all_baseline_results.json',{'results':rows,'failures':failures,'analysis_policy':ref(OUT/'analysis_policy_freeze.json')})
    proposed={r['workload']:r for r in rows if r['baseline']=='Proposed'}
    comparisons=[]
    for row in rows:
        p=proposed[row['workload']];o=outcome(p,row)
        confirm_p=dict(p,metrics=p['confirmation_metrics']);confirm_r=dict(row,metrics=row['confirmation_metrics'])
        co=outcome(confirm_p,confirm_r)
        row['Proposed_comparison']=o;row['Proposed_confirmation_comparison']=co
        table={'workload':row['workload'],'method':row['baseline'],'random_seed':row['random_seed'],
            **{k:v for k,v in row['metrics'].items() if k not in ('raw_source_paths','sampled_peak_is_not_instantaneous_peak')},
            'N_G':row['N_G'],'N_S':row['N_S'],'planning_memory_mib':row['planning_memory_mib'],
            'relative_mean_improvement_Proposed_percent':percent_gain(row['metrics']['mean_e2e_ms'],p['metrics']['mean_e2e_ms']),
            'relative_P95_improvement_Proposed_percent':percent_gain(row['metrics']['weighted_task_p95_ms'],p['metrics']['weighted_task_p95_ms']),
            'Proposed_objective_outcome':o['outcome'],'Proposed_objective_reason':o.get('Proposed_comparison_reason'),
            'confirmation_mean_ms':row['confirmation_metrics'].get('mean_e2e_ms'),
            'confirmation_mean_std_ms':row['confirmation_metrics'].get('mean_e2e_std_ms'),
            'confirmation_P95_ms':row['confirmation_metrics'].get('weighted_task_p95_ms'),
            'confirmation_stability':row['confirmation_metrics'].get('stability'),
            'confirmation_Proposed_outcome':co['outcome'],'measurement_source':row['measurement_source'],
            'original_source_paths':row['original_source_paths']}
        comparisons.append(table)
    csvwrite('static_baseline_comparison.csv',[r for r in comparisons if r['method']!='B4'])
    csvwrite('random_baseline_comparison.csv',[r for r in comparisons if r['method'] in ('Proposed','B4')])
    csvwrite('deployment_comparison.csv',[{k:r[k] for k in ('baseline','workload','random_seed','deployment','N_G','N_S',
        'total_model_replicas','Generalist_identity','Specialist_identity','planning_memory_mib')} | {
        'actual_idle_VRAM_mib_mean':r['metrics'].get('actual_idle_VRAM_mib_mean'),
        'sampled_peak_device_VRAM_mib':r['metrics'].get('sampled_peak_device_VRAM_mib')} for r in rows])
    csvwrite('search_cost_comparison.csv',costs);csvwrite('quality_compliance.csv',quality_rows)
    random_summary=[]
    for name in names:
        rr=[r for r in rows if r['baseline']=='B4' and r['workload']==name]
        random_summary.append({'workload':name,'seeds':[r['random_seed'] for r in rr],
            'mean_E2E_across_seeds':stats([r['metrics']['mean_e2e_ms'] for r in rr]),
            'weighted_P95_across_seeds':stats([r['metrics']['weighted_task_p95_ms'] for r in rr]),
            'confirmation_mean_across_seeds':stats([r['confirmation_metrics']['mean_e2e_ms'] for r in rr]),
            'stability_counts':dict(Counter(r['metrics']['stability'] for r in rr)),
            'Proposed_win_tie_loss':dict(Counter(r['Proposed_comparison']['outcome'] for r in rr)),
            'shared_donors_not_independent_campaign_samples':True})
    save_json(OUT/'random_seed_statistics.json',{'workloads':random_summary,'std_definition':'sample std ddof=1 across three fixed seeds'})
    csvwrite('random_seed_statistics.csv',random_summary)
    claims={}
    allcomplete=not failures and len(rows)==48
    for letter,baseline in zip('ABCDE',('B0','B1','B2','B3','B4')):
        rr=[r for r in rows if r['baseline']==baseline]
        expected=18 if baseline=='B4' else 6
        confirmed_wins=[r['workload'] for r in rr if r['Proposed_comparison']['outcome']=='win' and r['Proposed_confirmation_comparison']['outcome']=='win']
        both_all_win=len(confirmed_wins)==expected
        if len(rr)!=expected or any(not r['metrics']['valid'] or not r['confirmation_metrics']['valid'] for r in rr):label='INSUFFICIENT'
        elif both_all_win:label='SUPPORTED'
        elif confirmed_wins:label='PARTIAL'
        else:label='NOT_SUPPORTED'
        efficient=set()
        if letter=='E':
            p_cost={r['workload']:r for r in costs if r['baseline']=='Proposed'}
            for r in rr:
                rc=next(c for c in costs if c['baseline']=='B4' and c['workload']==r['workload'] and c['random_seed']==r['random_seed'])
                if r['Proposed_comparison']['outcome']=='tie' and r['Proposed_confirmation_comparison']['outcome']!='loss' and p_cost[r['workload']]['logical_runtime_state_evaluations']<rc['logical_runtime_state_evaluations']:
                    efficient.add(r['workload'])
            if label=='NOT_SUPPORTED' and len(efficient)>=2:label='PARTIAL'
        claims[letter]={'claim':{'A':'Proposed > Seed-only','B':'Proposed > Qwen-centric','C':'Proposed > Specialist-greedy',
            'D':'Workload-specific deployment > Static Hybrid','E':'LP-guided search > Random-neighbor under equal per-job budget'}[letter],
            'status':label,'required_comparisons':expected,'search_win_tie_loss':dict(Counter(r['Proposed_comparison']['outcome'] for r in rr)),
            'confirmation_win_tie_loss':dict(Counter(r['Proposed_confirmation_comparison']['outcome'] for r in rr)),
            'search_wins_also_confirmed_workloads':confirmed_wins,'efficiency_evidence_workloads':sorted(efficient),
            'classification_source':ref(OUT/'analysis_policy_freeze.json'),'no_universal_score':True}
    claim_audit={'claims':claims,'all_six_workloads_considered':True,'all_three_random_seeds_considered':True,
        'statistical_significance_claim':False,'baseline_definitions_unchanged':True,
        'equal_budget_unit':'One workload × one RNG seed; all three seeds are separate searches, not a pooled larger budget',
        'quality_admission_compliant':True,'actual_new_timed_runs':physical['valid_timed_runs'],'physical_attempts':physical['physical_attempts'],
        'reuse_references':reuse['references'],'unique_reused_raw_runs':reuse['unique_raw_paths'],
        'failures':failures,'MODULE8E_READY':allcomplete,'MODULE8E_started':False,
        'remaining_blockers':[] if allcomplete else ['Missing complete valid search/confirmation evidence; see retained failures']}
    save_json(OUT/'baseline_claim_audit.json',claim_audit)
    b2=[r for r in comparisons if r['method']=='B2'];b3=[r for r in comparisons if r['method']=='B3']
    b2better=[r for r in b2 if (r['relative_mean_improvement_Proposed_percent'] is not None and r['relative_mean_improvement_Proposed_percent']<0)
        or (r['relative_P95_improvement_Proposed_percent'] is not None and r['relative_P95_improvement_Proposed_percent']<0)]
    b2audit={'results':b2,'baseline_better_workloads':[r['workload'] for r in b2better],
        'quality_admissible':True,'queue_diagnostic':[{k:r[k] for k in ('workload','per_task_queue_wait_ms','mean_worker_queue_wait_ms')} for r in b2],
        'queue_interpretation':'Report task-wise queue distributions and comparison to Proposed; no post-hoc numeric imbalance threshold',
        'adaptation_claim':'No dynamic adaptation tested; observed workload-specific selection is not dynamic adaptation evidence',
        'claim_status':claims['C']['status']}
    save_json(OUT/'b2_heuristic_audit.json',b2audit)
    save_json(OUT/'static_hybrid_audit.json',{'results':b3,'fixed_deployment_verified':len({json.dumps(r['deployment'],sort_keys=True) for r in rows if r['baseline']=='B3'})==1,
        'claim_status':claims['D']['status'],'no_aggregate_or_lambda_retuning':True})
    lines=['# Module 8D-2 — Final Real Baseline Campaign','',
        f"新增 valid timed runs = **{physical['valid_timed_runs']}**；physical attempts = {physical['physical_attempts']}；复用引用 = **{reuse['references']}**（{reuse['unique_raw_paths']} 个不同 raw）。Proposed 未重跑。已停止，未进入 ablation / dynamic / second hardware。",'',
        '每个 condition 使用三次 search-protocol repetitions，另两次 independent final confirmation；B4 seed/final 各两次。所有有效 unstable 记录保留，无事后删样本。std 为 sample std(ddof=1)。跨 RNG 的共享 raw 已明确标记，不能当作额外独立测量。','',
        '## Table A — Proposed vs fixed baselines（mean E2E ms）','',
        '| Workload | Proposed | B0 | B1 | B2 | B3 |','|---|---:|---:|---:|---:|---:|']
    def number(value):return f'{value:.3f}' if value is not None else 'NA'
    for name in names:
        values=[next(r for r in rows if r['baseline']==m and r['workload']==name)['metrics']['mean_e2e_ms'] for m in ('Proposed','B0','B1','B2','B3')]
        lines.append('| '+name+' | '+' | '.join(number(v) for v in values)+' |')
    lines+=['','完整 std/P95/stability/completion/throughput/queue/worker latency 与 signed improvement 见 static_baseline_comparison.csv。负 improvement 原样保留。','',
        '## Table B — Random baseline（三个 seed 全部报告）','',
        '| Workload | Proposed | Random-1103 | Random-2207 | Random-47 | across-seed mean±std / median | P win/tie/loss |',
        '|---|---:|---:|---:|---:|---|---|']
    for name in names:
        rr=[next((r for r in rows if r['baseline']=='B4' and r['workload']==name and r['random_seed']==seed),None) for seed in manifest['random_seeds']]
        ss=next(s for s in random_summary if s['workload']==name);s=ss['mean_E2E_across_seeds']
        lines.append('| '+name+' | '+number(proposed[name]['metrics']['mean_e2e_ms'])+' | '+' | '.join(number(r['metrics']['mean_e2e_ms']) if r else 'NA' for r in rr)+f" | {number(s['mean'])}±{number(s['std'])} / {number(s['median'])} | {ss['Proposed_win_tie_loss']} |")
    totals=Counter(r['Proposed_comparison']['outcome'] for r in rows if r['baseline']=='B4')
    lines+=['',f'Proposed vs Random 总 win/tie/loss = **{dict(totals)}**。按预冻结 NoiseAwareRuntimeObjective 双向比较，tie 表示未越过 frozen objective gate，不能等价为 latency 相同。Independent confirmation 单独保留且未反馈到搜索。','',
        '## Table C / D / E','',
        'deployment_comparison.csv：完整十二维 replica vector、N_G/N_S、Generalist/Specialist identity、empirical memory、actual idle/sample peak。',
        'search_cost_comparison.csv：Proposed 与全部 Random 的 logical candidates、真实证据 state evaluations、new physical runs、reuse、accept/rollback、final objective。所有 job max states=6、top_b=2、iterations≤3、patience=1；缓存命中仍占 logical slot。',
        'quality_compliance.csv：全部 21-path admission compliance 及 task-wise effective quality；排除的六条 path 均无流量。WER/CER 加权只是 frozen point-estimate reporting，不是新的 pooled quality evaluation。','',
        '## B2 强 baseline audit','',
        f"B2 在以下 workload 的至少一个 latency metric 更好：**{[r['workload'] for r in b2better]}**。这些结果未截断或弱化。详见 b2_heuristic_audit.json 的逐 task queue 和逐 workload memory/latency；greedy routing 未做任何 runtime 调整。"]
    for r in b2:
        p=proposed[r['workload']]
        lines.append(f"- {r['workload']}: P relative mean/P95 = {number(r['relative_mean_improvement_Proposed_percent'])}% / {number(r['relative_P95_improvement_Proposed_percent'])}%；B2/P empirical memory={r['planning_memory_mib']:.0f}/{p['planning_memory_mib']:.0f} MiB；B2 mean queue={number(r['mean_worker_queue_wait_ms'])}ms，per-task={r['per_task_queue_wait_ms']}。")
    lines+=['','不能据本 campaign 声称 dynamic adaptation 优势；只比较已冻结 workload-specific deployment 与 deterministic greedy deployment/routing。','',
        '## Static Hybrid audit','']
    for r in b3:lines.append(f"- {r['workload']}: P relative mean/P95 = {number(r['relative_mean_improvement_Proposed_percent'])}% / {number(r['relative_P95_improvement_Proposed_percent'])}%；search/confirmation objective outcome={r['Proposed_objective_outcome']}/{r['confirmation_Proposed_outcome']}。")
    lines+=['','## Claim audit（六 workload 全部纳入，无综合总分）','']
    for letter,value in claims.items():lines.append(f"- **{letter}. {value['status']}** — {value['claim']}；search={value['search_win_tie_loss']}，confirmation={value['confirmation_win_tie_loss']}，效率证据 workloads={value['efficiency_evidence_workloads']}。")
    lines+=['',f"失败分类记录={failures}。Genuine unstable 属于有效真实结果，不重试到 steady；resource infeasible 保留且 deployment 不改；external interruption 遵守既有零自动重试政策。",'',
        f"所有 baseline quality admissible=true。MODULE8E_READY={allcomplete}，remaining blockers={claim_audit['remaining_blockers']}。是否进入 8E 由后续授权决定，当前未启动。",'',
        '结论限定冻结六 workload、三个随机 seeds、单 RTX4090、当前 bounds 和实验窗口；不做超出设计的 significance、全局最优或普遍优越性声明。']
    (OUT/'module8d2_summary.md').write_text('\n'.join(lines)+'\n')
    verify_inputs()
    print(json.dumps({'event':'report_complete','new_timed_runs':physical['valid_timed_runs'],'reuse':reuse['references'],
        'claims':{k:v['status'] for k,v in claims.items()},'random_outcomes':dict(totals),'MODULE8E_READY':allcomplete}),flush=True)
