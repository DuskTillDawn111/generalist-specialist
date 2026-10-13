"""Audit saved decisions, actual memory, provenance and independent confirmation."""
import json
import math
from pathlib import Path
from collections import Counter
from candidate_pool_service_demand.dry_run import load_completed_catalog
from candidate_pool_audit.provenance import environment_snapshot
from deployment_core.replica_calibration import load_replica_calibration, ReplicaPlanningPolicy
from deployment_core.replicas import replica_statistics, replica_key
from deployment_core import Workload
from deployment_search import IntegerGuidedDeploymentSearch, SearchConfig
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_search.runtime_cache import save_json,file_hash,digest
from deployment_search.result import jsonable
from .prepare import OUT,ROOT,read,verify_freeze
from .evaluator import ReplicaRuntimeEvaluation


def report(incomplete=False):
    plan=verify_freeze();protocol=read(OUT/'pilot_protocol.json');context=read(OUT/'runtime_context.json')
    catalog,hardware,roles,records,_=load_completed_catalog();cal=load_replica_calibration(catalog)
    policy=ReplicaPlanningPolicy(catalog,'empirical',calibration=cal);objective=NoiseAwareRuntimeObjective(1.,.05)
    cache_path=OUT/'search_cache/runtime_cache_manifest.json'
    cache=read(cache_path) if cache_path.exists() else {'configurations':{},'run_attempts':[],'cache_hits':0}
    evaluations=[]
    for key,entry in cache['configurations'].items():
        value={'cache_key':key,'identity':entry['identity'],'status':entry['status'],'attempts':entry['repetitions']}
        if entry['status']=='complete':value['runtime_result']=read(entry['aggregate'])
        evaluations.append(value)
    valid=sum(e['status']=='complete' for e in evaluations)
    save_json(OUT/'runtime_evaluations.json',{'valid_unique_runtime_evaluations':valid,'reserved_unique_states':len(evaluations),
        'search_physical_attempts':len(cache['run_attempts']),'unique_cap':6,'physical_cap':18,'cache_hits':cache['cache_hits'],
        'evaluations':evaluations,'confirmation_counted_in_search_budget':False})
    trace_path=OUT/'search_trace.json';trace=read(trace_path) if trace_path.exists() else None
    decisions=[];errors=[];comparisons=[]
    if trace:
        for t in trace['iterations']:
            incumbent=ReplicaRuntimeEvaluation(**t['current_runtime']);memory=policy.deployment_memory(t['deployment']);best=None
            for a in sorted((a for a in t['candidates'] if a['rank'] is not None),key=lambda a:a['rank']):
                if a['runtime'] is None:continue
                rt=ReplicaRuntimeEvaluation(**a['runtime']);better,reason=objective.compare(rt,incumbent,a['memory_mib'],memory)
                if better and (best is None or objective.compare(rt,ReplicaRuntimeEvaluation(**best['runtime']),a['memory_mib'],best['memory_mib'])[0]):best=a
            if not incomplete:
                expected=jsonable(replica_key(best['mutation']['deployment'])) if best else None
                if expected!=t['accepted']:errors.append('accept/rollback replay mismatch iteration '+str(t['iteration']))
            for a in t['candidates']:
                m=a['mutation']['model'];old=t['deployment'][m];new=a['mutation']['deployment'][m]
                kind='scale_down' if new<old else ('generalist_replica_up' if roles[m]=='generalist' else 'specialist_replica_up') if old else ('new_generalist_type' if roles[m]=='generalist' else 'new_specialist_type')
                rt=a['runtime'];better=None;reason=None
                if rt:
                    better,reason=objective.compare(ReplicaRuntimeEvaluation(**rt),incumbent,a['memory_mib'],memory)
                decisions.append({'iteration':t['iteration'],'candidate_type':kind,'model_id':m,'old_count':old,'new_count':new,
                    'deployment':a['mutation']['deployment'],'memory_delta':policy.memory(m,new)-policy.memory(m,old),
                    'empirical_capacity_delta':policy.capacity(m,new)-policy.capacity(m,old),'delta_z':a['delta_z'],
                    'planning_feasible':a['canonical'] is not None,'rank':a['rank'],'Top_B':a['selected_top_b'],
                    'runtime_result':rt,'decision':a['status'],'reason':a['reason'],
                    'objective_better_than_incumbent':better,'replayed_objective_reason':reason})
            # Counterfactual planning only; never run linear GPU serving.
            linear=IntegerGuidedDeploymentSearch().search(Workload(protocol['workload']['arrival_rates_rps']),catalog,hardware,
                SearchConfig(dry_run=True),initial_deployment=t['deployment'],roles=roles,
                capacity_mode='linear',memory_mode='empirical',calibration=cal)
            linear_top=[a.mutation.deployment for a in linear.iterations[0]['candidates'] if a.selected_top_b]
            empirical_top=[a['mutation']['deployment'] for a in t['candidates'] if a['selected_top_b']]
            actual=[a['mutation']['deployment'] for a in t['candidates'] if a['runtime'] is not None]
            comparisons.append({'iteration':t['iteration'],'fixed_incumbent':t['deployment'],'linear_Top_B':linear_top,
                'empirical_Top_B':empirical_top,'actually_evaluated':actual,'selection_changed':linear_top!=empirical_top,
                'scope':'CPU counterfactual at actual incumbent; no linear GPU campaign, no end-to-end linear-search counterfactual claim.'})
        if trace.get('enumeration_calls',0)!=0:errors.append('enumeration invoked')
        if trace['runtime_evaluation_count']>6:errors.append('search call budget')
        if any(t['num_candidates_generated']>2*len(catalog) for t in trace['iterations']):errors.append('one-hop bound')
        if not incomplete and valid!=trace['runtime_evaluation_count']:errors.append('valid/cache/search count mismatch')
    if len(cache['run_attempts'])>18 or len(evaluations)>6:errors.append('physical/reserved budget')
    if len({(a['cache_key'],a['repetition']) for a in cache['run_attempts']})!=len(cache['run_attempts']):errors.append('duplicate search attempt')
    for e in evaluations:
        if digest(e['identity'])!=e['cache_key']:errors.append('cache identity hash mismatch')
        if e['status']=='complete' and len(e['runtime_result']['repetitions'])!=3:errors.append('incomplete aggregate')
    save_json(OUT/'integer_replica_decisions.json',{'audit_passed':not errors,'audit_errors':errors,'decisions':decisions,
        'linear_vs_empirical_at_actual_incumbents':comparisons,'enumeration_calls':trace.get('enumeration_calls',0) if trace else 0})
    all_raw=[]
    for p in OUT.rglob('*.json'):
        if not (p.parent.name=='adapter_smoke' or p.parent.name=='confirmation' or p.parent.parent.name=='raw_runs'):continue
        if p.name=='aggregate.json' or p.name.endswith(('.config.json','.summary.json','.process.json')) or p.name.endswith('_aggregate.json'):continue
        raw=read(p)
        if 'memory_observations' not in raw:continue
        observation=raw['memory_observations'];planned=raw['planning_memory_mib'];actual=observation['actual_idle_mib']
        difference=actual-planned if actual is not None else None
        all_raw.append({'path':str(p),'sha256':file_hash(p),'category':'smoke' if p.parent.name=='adapter_smoke' else 'confirmation' if p.parent.name=='confirmation' else 'search',
            'deployment':raw['deployment'],'execution_status':raw['execution_status'],'runtime_resource_status':raw['runtime_resource_status'],
            'planning_idle_mib':planned,'actual_idle_mib':actual,'idle_difference_mib':difference,
            'idle_relative_difference':difference/planned if difference is not None else None,
            'within_prefrozen_5percent_idle_diagnostic_tolerance':abs(difference)/planned<=.05 if difference is not None else None,
            'peak_owned_mib':observation['peak_owned_mib'],'peak_workload_mib':observation['peak_workload_mib'],
            'peak_device_mib':observation['peak_device_mib'],'gpu_release':raw['gpu_release'],
            'worker_identities':raw['worker_identities'],
            'no_runtime_budget_exceedance':observation['peak_device_mib'] is not None and observation['peak_device_mib']<=hardware.gpu_memory_mib})
    resource_failures=[r for r in all_raw if r['runtime_resource_status']=='RUNTIME_RESOURCE_INFEASIBLE']
    if any(not r['gpu_release']['verified'] for r in all_raw):errors.append('GPU release failure')
    memory={'GPU_capacity_mib':hardware.gpu_memory_mib,'idle_diagnostic_tolerance_relative':.05,
        'note':'5% is a prefrozen diagnostic tolerance, not a safety margin or a peak-memory guarantee. Safety uses the hard 49140 MiB budget.',
        'runs':all_raw,'RUNTIME_RESOURCE_INFEASIBLE':resource_failures,
        'all_measured_idle_within_tolerance':all(r['within_prefrozen_5percent_idle_diagnostic_tolerance'] is True for r in all_raw),
        'all_peaks_within_budget':all(r['no_runtime_budget_exceedance'] for r in all_raw),'all_gpu_releases_verified':all(r['gpu_release']['verified'] for r in all_raw)}
    save_json(OUT/'memory_validation.json',memory)
    before=read(OUT/'before_hashes.json');changed=[name for name,value in before['frozen_files'].items() if file_hash(value['path'])!=value['sha256']]
    native_changed=[name for name,value in before['native_models_and_adapters'].items() if file_hash(ROOT/name)!=value['sha256']]
    environment_unchanged=environment_snapshot()==read(OUT/'environment_manifest.json')['environments']
    if changed or native_changed or not environment_unchanged:errors.append('frozen input/model/environment drift')
    provenance={'frozen_files_checked':len(before['frozen_files']),'changed_frozen_files':changed,
        'native_model_files_checked':len(before['native_models_and_adapters']),'changed_native_files':native_changed,
        'environments_unchanged':environment_unchanged,'pilot_freeze_sha256':file_hash(OUT/'pilot_freeze.json'),
        'implementation_hashes':plan['implementation_hashes']}
    save_json(OUT/'source_validation.json',provenance)
    confirmation=read(OUT/'confirmation_results.json') if (OUT/'confirmation_results.json').exists() else {'complete':False}
    smoke=read(OUT/'replica_adapter_validation.json') if (OUT/'replica_adapter_validation.json').exists() else {'status':'not_started'}
    if not confirmation.get('complete') and not incomplete:errors.append('confirmation incomplete')
    if not (OUT/'confirmation_results.json').exists():save_json(OUT/'confirmation_results.json',confirmation)
    summary(trace,decisions,comparisons,memory,confirmation,smoke,valid,len(cache['run_attempts']),errors,incomplete)
    save_json(OUT/'pilot_validation.json',{'status':'passed' if not errors and not incomplete else 'incomplete' if incomplete else 'failed',
        'audit_errors':errors,'search_valid_unique_runtime_evaluations':valid,'search_physical_attempts':len(cache['run_attempts']),
        'confirmation_physical_attempts':len(read(OUT/'confirmation_attempts.json')) if (OUT/'confirmation_attempts.json').exists() else 0,
        'smoke_physical_attempts':len(read(OUT/'smoke_attempts.json')) if (OUT/'smoke_attempts.json').exists() else 0,
        'enumeration_calls':trace.get('enumeration_calls',0) if trace else 0})
    if errors and not incomplete:raise AssertionError('Pilot audit failed: '+str(errors))
    artifacts=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.md') and p.name!='artifact_manifest.json']
    save_json(OUT/'artifact_manifest.json',{'module':'7C-3','status':'incomplete' if incomplete else 'complete',
        'frozen':not incomplete,'artifacts':{p.name:{'path':str(p),'sha256':file_hash(p)} for p in sorted(artifacts)}})


def summary(trace,decisions,comparisons,memory,confirmation,smoke,valid,attempts,errors,incomplete):
    lines=['# Module 7C-3：Replica-aware Runtime Adapter + Real Integer Search Pilot','',
        'Frozen workload：Module 3C W2_generalist_heavy，rates 未调整；capacity_mode=empirical，predictor=None，Top-B=2。',
        'Family routing 使用原 smooth weighted round-robin，再对该 family 的独立 model#i processes 做 deterministic round-robin。模型 handler、precision、input pool 和 measurement semantics 保持冻结版本。',
        'Evidence bounds：Qwen、Whisper-small、ResNet50≤2，其余 9 types≤1；不外推，不枚举 replica vectors。',
        '',f'Adapter smoke={smoke["status"]}；search valid unique evaluations={valid}/6；search physical repetitions={attempts}/18。',
        f'Audit errors：{errors}。', '']
    if not trace or trace.get('runtime') is None:
        lines+=['Pilot 未完成；所有已有失败/中断记录保留，没有重试、candidate 替换或预算增加。',
            '15 项最终问题中未观察到的值保持 pending，不能宣称有性能改善或可进入多 workload benchmark。']
    else:
        seed=trace['seed']['deployment'];final=trace['deployment'];roles={m:r['role'] for m,r in read(OUT/'runtime_context.json')['models'].items()}
        stats=replica_statistics(final,roles);a=trace['seed_runtime'];b=trace['runtime']
        def gain(key):return (a[key]-b[key])/a[key] if a[key] and b[key] is not None else None
        chosen_two=any(any(n==2 for n in d['deployment'].values()) for d in decisions if d['decision']=='ACCEPTED') or any(n==2 for n in seed.values())
        evaluated_two=[d['deployment'] for d in decisions if d['runtime_result'] and any(n==2 for n in d['deployment'].values())]
        generals=[d for d in decisions if d['candidate_type']=='generalist_replica_up']
        specialists=[d for d in decisions if d['candidate_type'] in ('new_specialist_type','specialist_replica_up')]
        lines += [f'1. Integer MILP seed：{seed}。',f'2. 最终 replica vector：{final}。',
            f'3. N_G={stats["N_G"]}，active Generalist types={stats["active_generalist_types"]}。',
            f'4. N_S={stats["N_S"]}，active Specialist types={stats["active_specialist_types"]}。',
            f'5. Seed/accepted state 是否有 x=2：{chosen_two}；search 中实际 evaluated x=2 states：{evaluated_two}。smoke 的 x=2 只证明 adapter correctness，不能算作算法接受。',
            f'6. Generalist replica-up candidates：{[(d["iteration"],d["model_id"],d["old_count"],d["new_count"],d["rank"],d["decision"]) for d in generals]}。',
            f'7. Specialist candidates：{sorted(set(d["model_id"] for d in specialists))}；逐轮完整 table 见下文和 integer_replica_decisions.json。',
            f'8. Calibration 改变实际 evaluated states 的引导：{any(c["selection_changed"] for c in comparisons)}；CPU linear counterfactual Top-B 在相同实际 incumbent 下比较，未运行 linear GPU campaign，也不宣称其完整 search path/性能已被观察。',
            f'9. Weighted mean E2E：{a["weighted_mean_e2e"]}→{b["weighted_mean_e2e"]} ms，relative gain={gain("weighted_mean_e2e")}；weighted task P95：{a["weighted_p95_e2e"]}→{b["weighted_p95_e2e"]} ms，relative gain={gain("weighted_p95_e2e")}。这不是 pooled system P95。',
            f'10. Actual idle VRAM 全部在冻结 5% diagnostic tolerance 内：{memory["all_measured_idle_within_tolerance"]}；actual peaks≤49140：{memory["all_peaks_within_budget"]}。逐次 deviation、idle、peak 保存在 memory_validation.json；idle 表不保证联合 peak。',
            f'11. OOM/runtime resource infeasible：{len(memory["RUNTIME_RESOURCE_INFEASIBLE"])}；GPU release 全部 verified={memory["all_gpu_releases_verified"]}。',
            f'12. Enumeration calls={trace.get("enumeration_calls",0)}，正式 search 没有调用 oracle。',
            f'13. Runtime evaluations={trace["runtime_evaluation_count"]}/6；valid unique={valid}；confirmation/smoke 独立列账不影响搜索预算或决策。',
            f'14. 独立 2+2 confirmation complete={confirmation.get("complete")}；mean direction={confirmation.get("mean_direction_maintained")}，frozen objective direction={confirmation.get("objective_direction_maintained")}。若 final==seed 则解释为 no deployment change，不把两组噪声当成改善。',
            '15. '+('Pilot 未完成或审计失败，暂不支持进入多 workload benchmark。' if incomplete or errors else
                'Adapter、预算和 decision replay 可用于下一阶段受限 W1/W2/W3 integer validation；仅一个 W2 pilot 不能证明多 workload 稳定性或最优性。'+
                ('独立 confirmation 保持改善方向。' if confirmation.get('objective_direction_maintained') else '独立 confirmation 未确认 frozen-objective 改善，应在扩展 campaign 前审查该结果；不宣称性能收益已确认。')),
            '',f'Search stability：{a["stability_state_v2"]}→{b["stability_state_v2"]}；stop={trace["status"]}。',
            '同 stability 的 mean/P95/memory 判定严格复用原 Module 3 implementation：same-steady 才进行 latency 比较，same-nonsteady 保留 incumbent，latency acceptance 需要 ≥5% mean 改善。失败/不稳定值不被删除。',
            '', '## 每轮 Generalist/Specialist choice','',
            '| Iter | Type | Model / counts | Memory Δ MiB | Empirical Δg | Δz | Rank | Runtime | Decision |',
            '|---|---|---|---:|---:|---:|---:|---|---|']
        for d in decisions:
            runtime=d['runtime_result'];label=f'{runtime["stability_state_v2"]}; mean={runtime["weighted_mean_e2e"]}' if runtime else 'not evaluated'
            lines.append(f'| {d["iteration"]} | {d["candidate_type"]} | {d["model_id"]} {d["old_count"]}→{d["new_count"]} | {d["memory_delta"]} | {d["empirical_capacity_delta"]:.9f} | {d["delta_z"]} | {d["rank"]} | {label} | {d["decision"]} ({d["reason"]}) |')
        lines+=['', '## Independent confirmation','',json.dumps({k:v for k,v in confirmation.items() if k not in ('repetitions','aggregates')},ensure_ascii=False), '']
    lines += ['', 'Raw repetitions、PID/CUDA process identity、model/environment/adapter hashes、family/replica metrics、实际 idle/peak memory、shutdown/release evidence 均保留。',
        'Qwen presence 由唯一 VQA capability 强制；Qwen/Whisper g(2) 来自 ASR，mixed-task 和异构 interference 的泛化仍是 surrogate。',
        '没有 new scaling profiling，没有 replica>evidence bound，没有 dynamic adaptation、full multi-workload integer benchmark 或 final paper benchmark。完成后停止。']
    (OUT/'module7c3_summary.md').write_text('\n'.join(lines)+'\n')
