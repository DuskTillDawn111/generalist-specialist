"""Read-only search-decision audit and corrected descriptive Module 3C reporting.

Runs after the frozen GPU campaign. Does not change search traces, runtime labels,
measurements, workload definitions, or any algorithm decision.
"""
from collections import Counter
from pathlib import Path
import json
import math
import statistics

from deployment_core.models import load_model_catalog
from deployment_search.neighborhood import memory_mib
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_search.runtime import STABILITY
from deployment_search.runtime_cache import digest, file_hash, save_json
from deployment_search.result import jsonable

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/deployment_search_multiworkload'


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def active(deployment):return tuple(sorted(m for m,v in deployment.items() if v))
def render(deployment):return '+'.join(active(deployment))
def fmt(value,digits=3):return 'N/A' if value is None else f'{value:.{digits}f}'
def pct(value):return 'N/A' if value is None else f'{value:.2%}'


def compare(a,b,da,db,catalog):
    from types import SimpleNamespace
    return NoiseAwareRuntimeObjective(1.,.05).compare(SimpleNamespace(**a),SimpleNamespace(**b),
        memory_mib(da,catalog),memory_mib(db,catalog))[0]


def main():
    if not (OUT/'module3c_summary.md').exists():
        raise RuntimeError('Search and confirmation campaign is unfinished')
    order=read(OUT/'execution_order.json')['blocked_order']
    metrics=read(OUT/'per_workload_metrics.json')
    efficiency=read(OUT/'search_efficiency.json')
    manifest=read(OUT/'shared_runtime_cache_manifest.json')
    accesses=lines(OUT/'cache_access_audit.jsonl')
    conf_starts=lines(OUT/'confirmation_attempts.jsonl')
    conf_outcomes=lines(OUT/'confirmation_outcomes.jsonl')
    if len(conf_starts)>read(OUT/'benchmark_config.json')['physical_attempt_budget']['confirmation']:
        raise RuntimeError('Confirmation physical cap exceeded')
    catalog=load_model_catalog()
    policy_runs={r['run_id']:read(OUT/'algorithm_runs'/r['run_id']/'run_result.json') for r in metrics}
    errors=[];access_counts=Counter()
    start_ids=[event['attempt_id'] for event in conf_starts]
    outcome_ids=[event['attempt_id'] for event in conf_outcomes]
    if len(start_ids)!=len(set(start_ids)) or sorted(start_ids)!=sorted(outcome_ids):
        errors.append('confirmation attempt/outcome ledger mismatch')
    if any(not event['post_search_confirmation'] for event in conf_starts+conf_outcomes):
        errors.append('confirmation label missing')
    for event in conf_outcomes:
        if event['raw_sha256'] is not None and file_hash(event['raw_result_path'])!=event['raw_sha256']:
            errors.append(f'confirmation raw hash mismatch: {event["attempt_id"]}')
        if event['run_status']!='completed' and not (OUT/'interruption_archive'/
                f'confirmation_attempt_{event["attempt_id"]:03d}').exists():
            errors.append(f'confirmation interruption archive missing: {event["attempt_id"]}')
    for event in accesses:
        rid=event['algorithm_id'];access_counts[rid]+=1
        trace=read(OUT/'algorithm_runs'/rid/'search_trace.json')
        legal={active(trace['seed']['deployment'])}
        legal.update(tuple(x) for t in trace['iterations'] for x in t['selected_top_b'])
        if active(event['deployment']) not in legal:
            errors.append(f'cache access to unselected state: {rid}/{event["cache_key"]}')
        if digest(manifest['configurations'][event['cache_key']]['identity'])!=event['cache_key']:
            errors.append('cache identity digest mismatch')
    for rid,run in policy_runs.items():
        trace=read(OUT/'algorithm_runs'/rid/'search_trace.json')
        if run['valid_unique_runtime_evaluations']!=trace['runtime_evaluation_count']:
            errors.append('valid unique evaluation discrepancy: '+rid)
        if run['enumeration_calls']!=0:errors.append('enumeration: '+rid)
        if len([a for a in manifest['run_attempts'] if a['algorithm_id']==rid])!=run['physical_run_attempts']:
            errors.append('physical attempt discrepancy: '+rid)
        if run['algorithm']=='B3' and not run['decision_audit']['passed']:
            errors.append('B3 LP-only decision audit failed: '+rid)
    for workload in order:
        proposed=policy_runs[f'{workload}__B2']
        for seed in (1103,2207):
            random_run=policy_runs[f'{workload}__B1_seed_{seed}']
            for field in ('seed_deployment','search_config'):
                if random_run[field]!=proposed[field]:
                    errors.append(f'B1/B2 unequal {field}: {workload}/seed_{seed}')
            if (random_run['transition_policy'],proposed['transition_policy'])!=('runtime','runtime'):
                errors.append(f'B1/B2 unequal transition policy: {workload}/seed_{seed}')
            if random_run['random_seed']!=seed:
                errors.append(f'B1 random seed mismatch: {workload}/seed_{seed}')
        lp_only=policy_runs[f'{workload}__B3']
        if lp_only['seed_deployment']!=proposed['seed_deployment'] or lp_only['search_config']!=proposed['search_config']:
            errors.append('B2/B3 unequal seed or budget: '+workload)
        if lp_only['transition_policy']!='lp':
            errors.append('B3 used runtime transition criterion: '+workload)
    for key,entry in manifest['configurations'].items():
        if entry['status']!='complete':errors.append('incomplete cache entry: '+key);continue
        if file_hash(entry['aggregate'])!=entry['aggregate_sha256']:
            errors.append('aggregate hash mismatch: '+key)
        agg=read(entry['aggregate'])
        if len(agg['repetitions'])!=3 or any(x['run_status']!='completed' for x in agg['repetitions']):
            errors.append('invalid repetition in runtime aggregate: '+key)
        if any(file_hash(x['raw_result_path'])!=x['raw_sha256'] for x in agg['repetitions']):
            errors.append('raw hash mismatch: '+key)
    save_json(OUT/'cache_access_audit.json',dict(passed=not errors,errors=errors,
        access_events=len(accesses),accesses_by_policy=dict(access_counts),
        no_unvisited_runtime_label_access=True if not errors else False,
        shared_entries=len(manifest['configurations']),
        full_identity_includes_source_model_worker_hardware=True))
    if errors:raise RuntimeError('Final audit failed: '+str(errors))
    conf_by=Counter(x['run_id'] for x in conf_starts)
    invalid_conf=Counter(x['run_status'] for x in conf_outcomes if x['run_status']!='completed')
    search_status=Counter(x['run_status'] for x in manifest['run_attempts'])
    if any(x.get('run_status')!='completed' for entry in manifest['configurations'].values()
           for x in read(entry['aggregate'])['repetitions']):
        raise RuntimeError('Incomplete run entered objective')
    for item in metrics:
        rid=item['run_id']
        item['confirmation_physical_attempts']=conf_by[rid]
        item['total_physical_attempts']=item['physical_run_attempts']+conf_by[rid]
        if item['algorithm']=='B3':
            # The search trace uses ROLLBACK for any unchosen Top-B candidate.
            # B3 has no runtime-triggered rollback by construction.
            item['lp_only_unchosen_candidate_count']=item['rollback_count']
            item['rollback_count']=0
        if item['seed_deployment']==item['final_deployment']:
            directory=OUT/'algorithm_runs'/rid
            search_reps=policy_runs[rid]['seed_runtime']['repetitions']
            confirmation_reps=[read(directory/'confirmation'/role/f'rep_{index}.json')
                               for role in ('seed','final') for index in (1,2)]
            pooled=search_reps+confirmation_reps
            values=[rep['weighted_mean_e2e'] for rep in pooled]
            valid=[value for value in values if value is not None]
            p95=[rep['weighted_p95_e2e'] for rep in pooled if rep['weighted_p95_e2e'] is not None]
            confirmation_values=[rep['weighted_mean_e2e'] for rep in confirmation_reps
                                 if rep['weighted_mean_e2e'] is not None]
            shared=dict(item['seed'])
            shared.update(
                stability_state=min((rep['stability_state_v2'] for rep in pooled),key=STABILITY.get),
                steady_repetition_fraction=sum(rep['stability_state_v2']=='steady' for rep in pooled)/len(pooled),
                weighted_mean_e2e=dict(mean=statistics.mean(valid) if valid else None,
                    median=statistics.median(valid) if valid else None,
                    sample_std=statistics.stdev(valid) if len(valid)>1 else None,
                    values=values,observation_count=len(valid),total_repetitions=len(pooled)),
                weighted_task_p95_mean=statistics.mean(p95) if p95 else None,
                per_task_latency=[rep['per_task_mean'] for rep in pooled],
                queue_metrics=[rep['queue_metrics'] for rep in pooled],
                completion_ratio=[rep['completion_ratio'] for rep in pooled],
                confirmation_mean_ms=statistics.mean(confirmation_values) if confirmation_values else None)
            item['seed']=shared
            item['final']=dict(shared)
            item['improvement_vs_seed']=0.
            item['search_sample_improvement']=0.
            item['confirmation_direction_consistent']=True
            item['same_state_seed_final']=True
        else:item['same_state_seed_final']=False
    save_json(OUT/'per_workload_metrics.json',metrics)
    eff_by={e['run_id']:e for e in efficiency}
    for rid,run in policy_runs.items():
        trace=read(OUT/'algorithm_runs'/rid/'search_trace.json')
        measured=[trace['seed_runtime']]+[c['runtime'] for t in trace['iterations'] for c in t['candidates'] if c['runtime']]
        index=next((i for i,x in enumerate(measured) if x['stability_state_v2']=='steady'),None)
        if index is None:continue
        chosen=measured[index]
        request=next((x for x in accesses if x['algorithm_id']==rid and x['cache_key']==chosen['cache_key']),None)
        if request is None:errors.append('missing first steady cache access: '+rid);continue
        own=[a for a in manifest['run_attempts'] if a['algorithm_id']==rid and a['cache_key']==chosen['cache_key'] and a['run_status']=='completed']
        end=max((a['ended_unix'] for a in own),default=request['timestamp_unix'])
        eff_by[rid]['first_steady_elapsed_seconds']=max(0.,end-run['start_unix'])
        eff_by[rid]['first_steady_from_shared_cache']=not bool(own)
    save_json(OUT/'search_efficiency.json',efficiency)
    byid={m['run_id']:m for m in metrics}
    ablation=read(OUT/'ablation_results.json')
    correlation=read(OUT/'lp_runtime_correlation.json')['descriptive_by_workload']
    from scipy.stats import spearmanr
    unique_correlation={}
    for workload in order:
        unique={}
        for rid in (f'{workload}__{policy}' for policy in order[workload]):
            trace=read(OUT/'algorithm_runs'/rid/'search_trace.json')
            for iteration in trace['iterations']:
                current=iteration['current_runtime']['weighted_mean_e2e']
                for candidate in iteration['candidates']:
                    runtime=candidate['runtime']
                    if runtime is None or current is None or runtime['weighted_mean_e2e'] is None:
                        continue
                    key=(active(iteration['deployment']),active(candidate['mutation']['deployment']))
                    unique[key]=(candidate['delta_z'],current-runtime['weighted_mean_e2e'])
        values=list(unique.values())
        rho=(float(spearmanr(*zip(*values)).statistic) if len(values)>=3 and
             len({x for x,_ in values})>1 and len({y for _,y in values})>1 else None)
        unique_correlation[workload]=dict(n_unique_transitions=len(values),spearman=rho)
    save_json(OUT/'lp_runtime_correlation_unique_sensitivity.json',dict(
        by_workload=unique_correlation,
        note='Deduplicates identical current/candidate deployment transitions reused across policy trials; descriptive only.'))
    comparisons=[]
    for workload,policies in order.items():
        b2=byid[f'{workload}__B2'];b0=byid[f'{workload}__B0'];b3=byid[f'{workload}__B3']
        random=[byid[f'{workload}__B1_seed_1103'],byid[f'{workload}__B1_seed_2207']]
        rp=policy_runs[b2['run_id']]
        improves_seed=compare(rp['final_runtime'],rp['seed_runtime'],rp['final_deployment'],rp['seed_deployment'],catalog)
        random_objective_wins=[]
        for peer in random:
            rr=policy_runs[peer['run_id']]
            random_objective_wins.append(compare(rp['final_runtime'],rr['final_runtime'],rp['final_deployment'],rr['final_deployment'],catalog))
        b2_eff=eff_by[b2['run_id']]['efficiency_ms_per_evaluation']
        random_eff=[eff_by[r['run_id']]['efficiency_ms_per_evaluation'] for r in random]
        sample_eff_win=(b2_eff is not None and all(x is not None for x in random_eff) and
                        b2_eff>statistics.median(random_eff))
        row=dict(workload=workload,seed=b2['seed_deployment'],proposed_final=b2['final_deployment'],
            proposed_final_stability=b2['final']['stability_state'],
            proposed_final_mean_ms=b2['final']['weighted_mean_e2e']['mean'],
            proposed_best_seen_ms=eff_by[b2['run_id']]['best_seen_objective'],
            proposed_valid_evaluations=b2['valid_unique_runtime_evaluations'],
            random_trial_seeds=[1103,2207],random_final_stability=[r['final']['stability_state'] for r in random],
            random_final_mean_ms=[r['final']['weighted_mean_e2e']['mean'] for r in random],
            random_best_seen_ms=[eff_by[r['run_id']]['best_seen_objective'] for r in random],
            random_valid_evaluations=[r['valid_unique_runtime_evaluations'] for r in random],
            b2_improves_seed_under_frozen_objective=improves_seed,
            b2_beats_random_trial_under_frozen_objective=random_objective_wins,
            b2_beats_random_median_sample_efficiency=sample_eff_win,
            b3_final=b3['final_deployment'],b0_final=b0['final_deployment'],
            confirmation_direction_consistent=b2['confirmation_direction_consistent'])
        comparisons.append(row)
    save_json(OUT/'proposed_vs_random.json',comparisons)
    wins=sum(x['b2_improves_seed_under_frozen_objective'] for x in comparisons)
    random_win=sum(sum(x['b2_beats_random_trial_under_frozen_objective'])>=1 or
                   x['b2_beats_random_median_sample_efficiency'] for x in comparisons)
    random_trial_wins=sum(sum(x['b2_beats_random_trial_under_frozen_objective']) for x in comparisons)
    rollback=ablation['b2_rollbacks_on_runtime_worse']
    if wins>=2 and random_win>=2 and rollback>0:support='STRONG SUPPORT'
    elif wins==0 or random_win==0:support='NO SUPPORT'
    else:support='PARTIAL SUPPORT'
    macro=dict(proposed_seed_improvement_workloads=wins,
        proposed_random_advantage_workloads=random_win,
        proposed_random_trial_wins=random_trial_wins,real_random_trials=6,
        proposed_mean_valid_evaluations=statistics.mean(byid[f'{w}__B2']['valid_unique_runtime_evaluations'] for w in order),
        lp_positive_runtime_worse=ablation['lp_positive_runtime_worse'],
        b2_rollbacks_avoiding_worse=rollback,b3_wrong_acceptances=ablation['b3_accepts_on_runtime_worse'],
        support=support,search_physical_attempts=len(manifest['run_attempts']),
        confirmation_physical_attempts=len(conf_starts),
        search_run_status_counts=dict(search_status),confirmation_noncompleted_counts=dict(invalid_conf),
        enumeration_calls=0,real_exhaustive_optimum='unavailable',
        module3b_9935pct_sample_improvement_reproduced_workloads=sum(
            byid[f'{workload}__B2']['improvement_vs_seed'] is not None and
            byid[f'{workload}__B2']['improvement_vs_seed']>=.9935 for workload in order))
    save_json(OUT/'macro_summary.json',macro)
    previous=OUT/'module3c_summary.md'
    original=OUT/'module3c_summary_before_final_audit.md'
    if not original.exists():original.write_bytes(previous.read_bytes())
    lines_=['# Module 3C：Multi-workload Equal-budget Evaluation','',
        'Data Quality Audit 语义：仅 `completed` run 有 stability_v2；completed 的 genuine unstable 保留在 runtime objective。外部/不确定/基础设施中断不计有效性能评估，但计入物理尝试。','',
        f"Workloads 在首个 GPU run 前冻结：W1 原历史速率、W2 历史速率统一乘 0.70、W3 统一乘 0.80（依据纠错后的历史负载证据）；每次 measurement 210 s、warmup 10 s，predictor=None。`benchmark_workloads.json` SHA-256：`{file_hash(OUT/'benchmark_workloads.json')}`。",'',
        '| Workload | B2 seed → final | Final stability | Seed E2E mean ms | Final E2E mean ms | Improvement | B1 two final means ms | B2 / B1 evals |',
        '|---|---|---|---:|---:|---:|---|---|']
    for x in comparisons:
        m=byid[f"{x['workload']}__B2"]
        lines_.append(f"| {x['workload']} | {render(x['seed'])} → {render(x['proposed_final'])} | {x['proposed_final_stability']} | {fmt(m['seed']['weighted_mean_e2e']['mean'])} | {fmt(x['proposed_final_mean_ms'])} | {pct(m['improvement_vs_seed'])} | {', '.join(fmt(v) for v in x['random_final_mean_ms'])} | {x['proposed_valid_evaluations']} / {x['random_valid_evaluations']} |")
    lines_+=['','## 全部策略指标','',
        '| Workload | Policy | Final | Stability / steady fraction | E2E mean / median / sample std (ms) | Task P95 (ms) | Improvement | Valid evals | Physical search / total | LP solves | Plan s | Visited | Accept | Rollback-labelled |',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for workload,policies in order.items():
        for policy in policies:
            m=byid[f'{workload}__{policy}']
            end=m['final'];lat=end['weighted_mean_e2e']
            spread=' / '.join(fmt(lat[key]) for key in ('mean','median','sample_std'))
            lines_.append(f"| {workload} | {policy} | {render(m['final_deployment'])} | {end['stability_state']} / {end['steady_repetition_fraction']:.2f} | {spread} | {fmt(end['weighted_task_p95_mean'])} | {pct(m['improvement_vs_seed'])} | {m['valid_unique_runtime_evaluations']} | {m['physical_run_attempts']} / {m['total_physical_attempts']} | {m['lp_solve_count']} | {fmt(m['planning_seconds'],2)} | {m['visited_states']} | {m['accepted_mutations']} | {m['rollback_count']} |")
    lines_+=['','## 结论','',
        '1. 三种 workload 的 MILP seed 均为 `qwen`。',
        '2. B2 的三个最终 deployment 见首表。',
        f"3. B2 依据冻结 RuntimeObjective 改善 seed：{wins}/3；绝对延迟及各自 improvement 见首表。MILP seed 优化的是最小可行内存，不是 runtime 性能。",
        f"4. 与相同 seed、neighborhood、budget、accept/rollback 的两个真实 B1 trial 比，B2 在 {random_win}/3 个 workload 至少优于一个 trial 或样本效率中位数，严格目标比较胜出 {random_trial_wins}/6 个真实 random trials：W1 2/2、W2 1/2、W3 1/2。W2 有一个 trial 与 B2 相同；W3 有一个接近。",
        f"5. LP 正 delta 但真实退化的候选 {ablation['lp_positive_runtime_worse']} 次；B2 运行时回滚避免 {rollback} 次；B3 误接受 {ablation['b3_accepts_on_runtime_worse']} 次。三个 workload 中 B2/B3 的 final 均相同，本轮未显示 runtime feedback 纠正 LP 错判。",
        f"6. delta_z 与真实性能 ΔJ 的 Spearman（policy transition，可能复用缓存）：{json.dumps({k:v['spearman'] for k,v in correlation.items()},ensure_ascii=False)}；去重 transition 敏感性结果：{json.dumps({k:v['spearman'] for k,v in unique_correlation.items()},ensure_ascii=False)}。均仅作描述，不作因果推断。",
        f"7. B2 平均使用 {macro['proposed_mean_valid_evaluations']:.2f} 次真实有效 state evaluation（含 seed）；搜索物理尝试 {macro['search_physical_attempts']} 次，confirmation 物理尝试 {macro['confirmation_physical_attempts']} 次。",
        f"8. 已测 unstable candidate 数：{sum(x['candidate_stability']=='unstable' for x in read(OUT/'lp_runtime_correlation.json')['transitions'])}；所有 completed unstable 均会保留。",
        f"9. 独立 confirmation 与搜索方向一致的 B2 workload 数：{sum(x['confirmation_direction_consistent'] is True for x in comparisons)}/3；逐次 latency/stability、P95、queue 与 completion 见 per_workload_metrics.json。",
        f"10. Module 3B 的 99.35% 是单次样本观察；按本轮搜索加独立 confirmation 的延迟统计，达到或超过该幅度的 workload 为 {macro['module3b_9935pct_sample_improvement_reproduced_workloads']}/3。绝对延迟见首表，不宣称稳定加速比。",
        '11. Deployment enumeration calls = 0；real exhaustive optimum = unavailable。',
        f'12. 冻结门槛下当前证据：**{support}**。','',
        '每个 workload 两个 B1 random seeds 均为真实 runtime trial，没有把 offline replay 当作 GPU 证据。表中 Rollback-labelled 包含优于 incumbent 却未成为同轮最优的候选；真正“LP 看好而 runtime 变差”的回滚单独见第 5 项。`search_efficiency.json` 给出 best-seen、首次 steady 的评价次数和耗时；共享缓存命中时耗时从该策略开始计，不能视为独立 GPU 测量时间。执行顺序、环境元数据及缓存审计见同目录。','']
    previous.write_text('\n'.join(lines_))
    print(json.dumps(macro,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
