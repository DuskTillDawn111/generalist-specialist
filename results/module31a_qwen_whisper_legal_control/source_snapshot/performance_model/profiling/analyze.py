"""Descriptive profiling audit; shared features and rows, no fitted predictor."""
from collections import Counter,defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import subprocess
from scipy.stats import spearmanr,pearsonr
from performance_model.dataset import build_rows
from performance_model.features import curve_context
from performance_model.piecewise_baseline import LoadPoint,PiecewiseLoadLatencyModel,same_context
from joint_deploy.measurement.load_generator import SamplePool,periodic_arrivals
from joint_deploy.measurement.router import Router
from joint_deploy.measurement.run_measurement import DEFAULT_POOL
from joint_deploy.run_pure_qwen import percentile
from deployment_core.workload import TASKS
from .plan import OUT,ROOT,save,source,extractor
from .run import assess

METRICS=('mean_e2e_latency_ms','p95_e2e_latency_ms','mean_worker_latency_ms','mean_worker_queue_wait_ms','throughput_rps','within_window_throughput_rps','mean_runtime_latency_ms','mean_schedule_delay_ms')


def stats(values):
    values=[v for v in values if v is not None]
    return dict(n=len(values),values=values,mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values)>1 else None,min=min(values) if values else None,max=max(values) if values else None)


def measured(rows,duration,end):
    successful=[r for r in rows if r['status']=='completed']
    def values(key): return [r[key] for r in successful if r.get(key) is not None]
    mean=lambda v:statistics.mean(v) if v else None
    return dict(mean_e2e_latency_ms=mean(values('e2e_latency_ms')),p95_e2e_latency_ms=percentile(values('e2e_latency_ms'),95),
        mean_worker_latency_ms=mean(values('worker_latency_ms')),mean_worker_queue_wait_ms=mean(values('worker_queue_wait_ms')),
        mean_runtime_latency_ms=mean(values('runtime_latency_ms')),mean_schedule_delay_ms=mean(values('schedule_delay_ms')),
        throughput_rps=len(successful)/duration,within_window_throughput_rps=sum(r['completion_time_ns']<end for r in successful)/duration,
        offered=len(rows),completed=len(successful),incomplete=sum(r['status']=='incomplete' for r in rows),failed=sum(r['status']=='error' for r in rows))


def execution_coverage(plan,runs,state):
    """Distinguish skipped A sweep points from a deferred, subsequently measured B anchor."""
    observed={r['run_id'] for r in runs}
    skipped=set(state.get('skipped_configurations',[]))
    deferred_anchor=state.get('stopped_load_axis_at') is not None and state['stopped_load_axis_at']<.4
    actual_skips=[c['name'] for c in plan['configs'] if c['name'] in skipped and not (c['name']=='A_u_0.40' and deferred_anchor)]
    expected={r['run_id'] for c in plan['configs'] if c['name'] not in actual_skips for r in c['runs']}
    return dict(planned_maximum_runs=sum(len(c['runs']) for c in plan['configs']),
        expected_runs_after_adaptive_stop=len(expected),saved_runs=len(observed),
        skipped_configurations=actual_skips,
        skipped_runs=[r['run_id'] for c in plan['configs'] if c['name'] in actual_skips for r in c['runs']],
        deferred_balanced_anchor='A_u_0.40' if deferred_anchor else None,
        missing_run_ids=sorted(expected-observed),unexpected_run_ids=sorted(observed-expected),
        complete=state.get('status')=='measurement_complete' and observed==expected)


def audit_run(data,spec,config,plan,ex):
    issues=[]; cfg=json.loads(Path(spec['config']).read_text())
    for k,v in cfg.items():
        actual=data.get(k)
        if k=='arrival_process' and isinstance(actual,dict): actual=actual.get('name')
        if actual!=v: issues.append('config mismatch: '+k)
    window=data.get('measurement_window'); records=data.get('request_records',[])
    arrivals=list(periodic_arrivals(cfg['workload_rps'],cfg['measurement_seconds'],cfg['random_seed']))
    if not window or len(arrivals)!=len(records): issues.append('measurement window or arrival count incomplete')
    else:
        router=Router(cfg['routing_rps']); pool=SamplePool(DEFAULT_POOL,[k for k,v in cfg['workload_rps'].items() if v>0],cfg['random_seed'])
        for r,(offset,task) in zip(records,arrivals):
            sample=pool.next(task)
            if (r['task'],r['model'],r['pool_id'])!=(task,router.choose(task),sample['pool_id']): issues.append('routing or sample sequence mismatch')
            if r['scheduled_time_ns']!=window['start_time_ns']+round(offset*1e9): issues.append('scheduled timestamp mismatch')
    features=ex.extract(data['deployment'],data['workload_rps'],data['routing_rps'])
    if not same_context(features,config['expected_features']): issues.append('computed features differ from plan')
    source_info=data.get('source',{})
    if source_info.get('task_pool_hash')!=source(DEFAULT_POOL)['sha256']: issues.append('pool hash mismatch')
    for name,h in source_info.get('harness_hashes',{}).items():
        if source(ROOT/'joint_deploy/measurement'/name)['sha256']!=h: issues.append('harness provenance mismatch')
    counts=Counter((r['task'],r['model']) for r in records)
    insufficient=[dict(task=k,model=m,offered=counts[k,m],completed=sum(r['task']==k and r['model']==m and r['status']=='completed' for r in records))
        for k,row in cfg['routing_rps'].items() for m,v in row.items() if v>0 and sum(r['task']==k and r['model']==m and r['status']=='completed' for r in records)<30]
    assessment=assess(data,plan['measurement_policy'])
    return dict(integrity_passed=not issues,issues=sorted(set(issues)),tail_sample_shortfall=insufficient,
                overloaded_or_unstable=assessment['overloaded_or_unstable'],assessment=assessment)


def analyze(output=OUT,plots=True):
    output=Path(output); plan=json.loads((output/'profiling_plan.json').read_text()); ex=extractor()
    pool={p['pool_id']:p['sample'] for p in (json.loads(l) for l in DEFAULT_POOL.read_text().splitlines())}
    rows=[]; runs=[]; conditions=[]
    for config in plan['configs']:
        group=[]
        for spec in config['runs']:
            path=Path(spec['result'])
            if not path.exists(): continue
            data=json.loads(path.read_text()); audit=audit_run(data,spec,config,plan,ex)
            path_rows=[]
            if data.get('measurement_window'):
                try: path_rows,_=build_rows(path,'module2b',config['experiment_family'],ex)
                except ValueError as error: audit['integrity_passed']=False; audit['issues'].append(str(error))
            rows.extend(path_rows)
            window=data.get('measurement_window') or {}; end=window.get('end_time_ns',0)
            qrecords=[r for r in data.get('request_records',[]) if r['model']=='qwen']
            summary=measured(qrecords,data['measurement_seconds'],end)
            tasks={k:measured([r for r in qrecords if r['task']==k],data['measurement_seconds'],end) for k in TASKS}
            input_audit={k:dict(pool_ids=[r['pool_id'] for r in qrecords if r['task']==k],
                duration_s=stats([pool[r['pool_id']].get('duration_s') for r in qrecords if r['task']==k])) for k in TASKS}
            run=dict(**spec,configuration=config['name'],family=config['experiment_family'],source_sha256=source(path)['sha256'],
                audit=audit,status=data.get('status'),errors=data.get('errors',[]),measurement_window=window,
                system=data.get('system_metrics'),qwen=summary,per_task=tasks,input_sample_audit=input_audit,
                features=data.get('feature_directed_profiling',{}).get('actual_config_features') or ex.extract(data['deployment'],data['workload_rps'],data['routing_rps']))
            runs.append(run); group.append(run)
        conditions.append(dict(name=config['name'],family=config['experiment_family'],features=config['expected_features'],
            u=config['expected_u'],q=config['expected_task_mix'],pressure={k:v['value'] for k,v in config['expected_features']['per_model']['qwen']['colocation_pressure'].items()},
            repetitions=len(group),run_ids=[r['run_id'] for r in group],
            unstable_runs=[r['run_id'] for r in group if r['audit']['overloaded_or_unstable']],
            invalid_runs=[r['run_id'] for r in group if not r['audit']['integrity_passed']],
            tail_shortfalls=[dict(run_id=r['run_id'],paths=r['audit']['tail_sample_shortfall']) for r in group if r['audit']['tail_sample_shortfall']],
            qwen={k:stats([r['qwen'].get(k) for r in group]) for k in METRICS},
            per_task={t:{k:stats([r['per_task'][t].get(k) for r in group]) for k in METRICS} for t in TASKS}))
    byname={c['name']:c for c in conditions}
    a=[c for c in conditions if c['family']=='load_axis' and c['repetitions']]
    def mean(c,k,task=None): return (c['per_task'][task] if task else c['qwen'])[k]['mean']
    adjacent=[]
    for left,right in zip(a,a[1:]):
        for task in [None]+list(TASKS):
            values={k:[mean(c,k,task) for c in (left,right)] for k in METRICS[:4]}
            if any(v is None for v in values['mean_e2e_latency_ms']+values['p95_e2e_latency_ms']+values['mean_worker_queue_wait_ms']): continue
            old,new=values['mean_e2e_latency_ms']; op,np=values['p95_e2e_latency_ms']; oq,nq=values['mean_worker_queue_wait_ms']
            adjacent.append(dict(task=task or 'qwen_request_weighted',u_range=[left['u'],right['u']],mean_factor=new/old,p95_factor=np/op,
                queue_increase_ms=nq-oq,latency_knee_candidate=(new>=2*old or np>=2*op) and nq-oq>=50,
                note='Descriptive adjacent doubling + >=50ms queue increase flag, not a fitted knee or significance test.'))
    cgroup=[c for c in conditions if c['family']=='colocation_axis' and c['repetitions']]
    control=byname['C_no_active_contender']; deltas=[]
    for c in cgroup:
        pairs=[]
        for r in runs:
            if r['configuration']!=c['name']: continue
            ref=next((v for v in runs if v['configuration']==control['name'] and v['seed']==r['seed']),None)
            if ref and ref['qwen']['mean_e2e_latency_ms'] is not None and r['qwen']['mean_e2e_latency_ms'] is not None:
                pairs.append(dict(seed=r['seed'],mean_e2e_delta_ms=r['qwen']['mean_e2e_latency_ms']-ref['qwen']['mean_e2e_latency_ms'],
                    worker_delta_ms=r['qwen']['mean_worker_latency_ms']-ref['qwen']['mean_worker_latency_ms'],
                    queue_delta_ms=r['qwen']['mean_worker_queue_wait_ms']-ref['qwen']['mean_worker_queue_wait_ms'],
                    input_audit={t:dict(same_qwen_pool_sequence=r['input_sample_audit'][t]['pool_ids']==ref['input_sample_audit'][t]['pool_ids'],
                        condition_duration_mean_s=r['input_sample_audit'][t]['duration_s']['mean'],control_duration_mean_s=ref['input_sample_audit'][t]['duration_s']['mean']) for t in TASKS}))
        deltas.append(dict(condition=c['name'],pressure=c['pressure'],paired_values=pairs,
            mean_e2e_delta_ms=stats([p['mean_e2e_delta_ms'] for p in pairs]),worker_delta_ms=stats([p['worker_delta_ms'] for p in pairs]),queue_delta_ms=stats([p['queue_delta_ms'] for p in pairs])))
    associations=[]
    if len(cgroup)==4 and all(mean(c,'mean_e2e_latency_ms') is not None for c in cgroup):
        y=[mean(c,'mean_e2e_latency_ms') for c in cgroup]
        for key in cgroup[0]['pressure']:
            x=[c['pressure'][key] for c in cgroup]
            if None in x or len(set(x))<2 or len(set(y))<2: continue
            associations.append(dict(feature=key,spearman=float(spearmanr(x,y).statistic),pearson=float(pearsonr(x,y).statistic),
                n_conditions=4,note='Exploratory condition-mean association across 4 points; correlated dimensions, no causal or significance claim.'))
        associations.sort(key=lambda v:abs(v['spearman']),reverse=True)
    mix_conditions=[byname['A_u_0.40']]+[c for c in conditions if c['family']=='task_mix_axis']
    mix_sensitivity=[]
    for task in TASKS:
        usable=[c for c in mix_conditions if c['repetitions']==3 and mean(c,'mean_e2e_latency_ms',task) is not None]
        if len(usable)!=4: continue
        low=min(usable,key=lambda c:mean(c,'mean_e2e_latency_ms',task))
        high=max(usable,key=lambda c:mean(c,'mean_e2e_latency_ms',task))
        low_v,high_v=mean(low,'mean_e2e_latency_ms',task),mean(high,'mean_e2e_latency_ms',task)
        signs=[]
        for seed in plan['seeds']:
            lr=next(r for r in runs if r['configuration']==low['name'] and r['seed']==seed)
            hr=next(r for r in runs if r['configuration']==high['name'] and r['seed']==seed)
            signs.append(hr['per_task'][task]['mean_e2e_latency_ms']-lr['per_task'][task]['mean_e2e_latency_ms'])
        mix_sensitivity.append(dict(task=task,lowest_measured_condition=low['name'],highest_measured_condition=high['name'],
            mean_ratio=high_v/low_v,mean_range_ms=high_v-low_v,
            largest_within_condition_sample_std_ms=max(c['per_task'][task]['mean_e2e_latency_ms']['std'] or 0 for c in usable),
            paired_seed_differences_ms=signs,all_three_paired_differences_positive=all(v>0 for v in signs)))
    # Only complete stable A points initialize the declared descriptive baseline.
    eligible=[c for c in a if c['repetitions']==3 and not c['unstable_runs'] and not c['invalid_runs'] and not c['tail_shortfalls']]
    points=[]
    if len(eligible)>=2 and all(same_context(curve_context(c['features'],'qwen'),curve_context(eligible[0]['features'],'qwen')) for c in eligible):
        for c in eligible:
            sources=tuple(r['result'] for r in runs if r['configuration']==c['name'])
            for task in TASKS:
                points.append(LoadPoint('qwen',task,c['u'],curve_context(c['features'],'qwen'),mean(c,'mean_e2e_latency_ms',task),mean(c,'p95_e2e_latency_ms',task),mean(c,'throughput_rps',task),sources))
        baseline=PiecewiseLoadLatencyModel(ex,points)
        for c in eligible:
            f=c['features']; prediction=baseline.predict(f['deployment'],f['workload'],f['routing'])
            if prediction.status!='interpolated_profile_baseline': raise ValueError('Baseline initialization check failed')
    baseline_data=dict(initialized=bool(points),eligible_load_conditions=[c['name'] for c in eligible],
        points=[asdict(p) for p in points],definition='Run-level mean of means/P95; not pooled quantiles. Descriptive interpolation baseline, no accuracy evaluation; no extrapolation.')
    save(output/'piecewise_load_points.json',baseline_data)
    (output/'dataset_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    state=json.loads((output/'execution_state.json').read_text()) if (output/'execution_state.json').exists() else {}
    coverage=execution_coverage(plan,runs,state)
    protected_path=output/'protected_sources.json'
    protected=json.loads(protected_path.read_text()) if protected_path.exists() else {}
    changed=[p for p,h in protected.items() if not (ROOT/p).exists() or source(ROOT/p)['sha256']!=h]
    changed_protocol=[p for p,h in plan['protocol_hashes'].items() if not (ROOT/p).exists() or source(ROOT/p)['sha256']!=h]
    quality=dict(unstable_runs=sum(r['audit']['overloaded_or_unstable'] for r in runs),
        runs_with_incomplete=sum((r['system'] or {}).get('incomplete_requests',0)>0 for r in runs),
        system_incomplete_requests=sum((r['system'] or {}).get('incomplete_requests',0) for r in runs),
        qwen_incomplete_requests=sum(r['qwen']['incomplete'] for r in runs),
        qwen_request_errors=sum(r['qwen']['failed'] for r in runs),
        runs_with_tail_shortfall=sum(bool(r['audit']['tail_sample_shortfall']) for r in runs),
        latency_population='Successful requests only; incomplete requests remain in counts. Failed/censored runs are included and explicitly flagged; their observed P95 is not the full offered-cohort P95.')
    validation=dict(runs_saved=len(runs),dataset_rows=len(rows),integrity_passed=all(r['audit']['integrity_passed'] for r in runs) and not changed and not changed_protocol if runs else None,
        execution_coverage=coverage,data_quality=quality,
        protected_sources=dict(checked=len(protected),changed_paths=changed,unchanged=bool(protected) and not changed),
        frozen_protocol=dict(checked=len(plan['protocol_hashes']),changed_paths=changed_protocol,unchanged=not changed_protocol),
        audits={r['run_id']:r['audit'] for r in runs},source_hashes={r['result']:r['source_sha256'] for r in runs},
        incomplete_or_unstable_runs=[r['run_id'] for r in runs if r['audit']['overloaded_or_unstable']])
    save(output/'dataset_validation.json',validation)
    summary=dict(stage=plan['stage'],status=state.get('status','not_started'),runs_saved=len(runs),configurations_observed=sum(c['repetitions']>0 for c in conditions),
        configurations_with_three_repetitions=sum(c['repetitions']==3 for c in conditions),dataset_rows=len(rows),conditions=conditions,runs=runs,
        execution_state=state,execution_coverage=coverage,data_quality=quality,task_mix_sensitivity=mix_sensitivity,load_axis_adjacent_changes=adjacent,colocation_paired_changes=deltas,resource_pressure_associations=associations,
        piecewise_baseline=baseline_data,aggregation='run-level mean/sample std(ddof=1); pooled within-run Qwen metrics and per-task metrics both saved; no significance claims')
    save(output/'profiling_summary.json',summary)
    markdown(output,summary)
    if plots: subprocess.run(['/root/miniconda3/bin/python',str(Path(__file__).with_name('plot.py')),str(output)],check=True)
    return summary


def markdown(output,s):
    def fmt(stat):
        return '—' if stat['mean'] is None else f"{stat['mean']:.2f} ± {stat['std']:.2f}" if stat['std'] is not None else f"{stat['mean']:.2f} (n=1)"
    coverage=s['execution_coverage']; quality=s['data_quality']
    lines=['# Module 2B: Qwen feature-directed profiling','',
        f"状态 {s['status']}；{s['runs_saved']} runs，{s['configurations_observed']} configurations（{s['configurations_with_three_repetitions']} 完整三重复），{s['dataset_rows']} dataset rows。",
        f"自适应停止后应执行 {coverage['expected_runs_after_adaptive_stop']} 轮；实际未执行配置：{coverage['skipped_configurations']}；覆盖检查完成={coverage['complete']}。",
        f"A 扫描在 u={s['execution_state'].get('stopped_load_axis_at')} 触发停止；deferred balanced anchor={coverage['deferred_balanced_anchor']}。该 anchor 若作为 B 对照补跑，不计为缺失或重复数据。",
        'A: Qwen-only，固定已有六任务比例；B: u=0.4 改变 demand mix（balanced复用A）；C: 固定驻留模型和 Qwen u/q，仅改变竞争路径的流量。',
        '所有 features 由 performance_model/features.py 计算，实际观察到的 cohort routing rates 也单独经相同 extractor 转换。',
        '均值±sample std 描述三轮离散，不是置信区间。worker是handler墙钟；u为standalone service demand，非实测GPU利用率。',
        'C 是已有 offline resource fingerprint 按路由 req/s 加权求和的特征，单位为 fingerprint 单位×req/s；不等同于实测GPU压力或利用率。',
        f"异常/不稳定 {quality['unstable_runs']} 轮；含 incomplete {quality['runs_with_incomplete']} 轮、Qwen incomplete {quality['qwen_incomplete_requests']} 请求；Qwen request error {quality['qwen_request_errors']}；成功样本不足30的轮次 {quality['runs_with_tail_shortfall']}。",
        '**延迟仅统计成功请求。** incomplete 仍保留在分母和原始记录；受 drain 截断的 P95 不是全部 offered 请求的 P95。异常轮次照常纳入描述表，不静默删除。','',
        '| Condition | reps | u_Q | Qwen worker ms | Qwen mean E2E ms | Qwen P95 ms | Queue ms | Qwen window rps | unstable |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for c in s['conditions']:
        keys=('mean_worker_latency_ms','mean_e2e_latency_ms','p95_e2e_latency_ms','mean_worker_queue_wait_ms','within_window_throughput_rps')
        lines.append(f"| {c['name']} | {c['repetitions']} | {c['u']:.4f} | "+' | '.join(fmt(c['qwen'][k]) for k in keys)+f" | {', '.join(c['unstable_runs']) or 'none'} |")
    a=[c for c in s['conditions'] if c['family']=='load_axis' and c['repetitions']]
    lines+=['','## 必须回答的问题','',
        '1. **u 与 latency 是否稳定单调？** 以下为相邻点三轮均值方向；seed级值与逐任务曲线见JSON/图，不能从整体均值推出每任务或每seed单调。']
    for key in ('mean_e2e_latency_ms','p95_e2e_latency_ms','mean_worker_queue_wait_ms','mean_worker_latency_ms'):
        values=[c['qwen'][key]['mean'] for c in a]
        ok=all(v is not None for v in values)
        monotonic=all(b>=a for a,b in zip(values,values[1:])) if ok and len(values)>1 else None
        lines.append(f'   {key}: {values}; nondecreasing={monotonic}。')
    seed_directions=[]
    for seed in sorted({r['seed'] for r in s['runs']}):
        values=[next((r['qwen']['mean_e2e_latency_ms'] for r in s['runs'] if r['configuration']==c['name'] and r['seed']==seed),None) for c in a]
        if len(values)>1 and all(v is not None for v in values): seed_directions.append((seed,all(y>=x for x,y in zip(values,values[1:]))))
    if seed_directions:
        lines.append(f"   各seed整体mean是否单调：{seed_directions}。三轮均值的上升不代表稳定的逐轮单调关系；重复间变化必须一起解释。")
    knees=[v for v in s['load_axis_adjacent_changes'] if v['latency_knee_candidate']]
    knee_ranges=sorted({tuple(v['u_range']) for v in knees if v['task']=='qwen_request_weighted'})
    lines+=['',f'2. **latency knee？** 整体三轮均值的上升候选区间：{knee_ranges}；首次停止点 u={s["execution_state"].get("stopped_load_axis_at")}。候选规则是相邻mean或P95翻倍且queue增加至少50ms，逐任务结果见JSON。重复间差异大时不能据此确定单一可靠knee；不以rho=0.8预设阈值。',
        '', '3. **相同 u 下 mix 差异？** balanced=A_u_0.40，与B三种mix的逐任务mean/P95比较见图B和下表。整体Qwen均值含任务比例变化，需同时看同一task。',
        '', '| Mix | task | mean E2E ms | P95 ms | worker ms | queue ms |','| --- | --- | ---: | ---: | ---: | ---: |']
    for c in s['conditions']:
        if c['name']=='A_u_0.40' or c['family']=='task_mix_axis':
            for task in TASKS:
                lines.append('| '+c['name']+' | '+task+' | '+' | '.join(fmt(c['per_task'][task][k]) for k in ('mean_e2e_latency_ms','p95_e2e_latency_ms','mean_worker_latency_ms','mean_worker_queue_wait_ms'))+' |')
    lines+=['','同task在四种mix间的range（条件按实测端点描述，不是routing/deployment选择）：',
        '| Task | highest/lowest mean ratio | range ms | largest within-condition std ms | positive paired seeds |',
        '| --- | ---: | ---: | ---: | ---: |']
    for v in s['task_mix_sensitivity']:
        lines.append(f"| {v['task']} | {v['mean_ratio']:.3f} | {v['mean_range_ms']:.2f} | {v['largest_within_condition_sample_std_ms']:.2f} | {sum(d>0 for d in v['paired_seed_differences_ms'])}/3 |")
    if s['task_mix_sensitivity']:
        ratios=[v['mean_ratio'] for v in s['task_mix_sensitivity']]
        coherent=sum(v['all_three_paired_differences_positive'] for v in s['task_mix_sensitivity'])
        exceeds=sum(v['mean_range_ms']>v['largest_within_condition_sample_std_ms'] for v in s['task_mix_sensitivity'])
        mix_verdict=f"同task的条件均值相差 {min(ratios):.2f}–{max(ratios):.2f} 倍；{coherent}/{len(ratios)} 个task的端点差在三个配对seed上同向，{exceeds}/{len(ratios)} 个task的条件range大于最大的条件内sample std。"
    else: mix_verdict='三重复尚未齐全，暂不作判断。'
    lines+=['', '3/4. **实测结论与 q 的必要性：** '+mix_verdict,
        'q 应保留为待验证的解释变量；当前的重复间波动、不同mix的测量窗口和输入序列、以及截断样本限制了归因。这些数据不能单独证明 q 已充分解释差异。',
        '', '5. **固定u/q后 C 能否解释额外变化？** 配对seed的C条件减去同驻留环境无活跃竞争者的结果：','',
        '| C condition | mean E2E delta ms | worker delta ms | queue delta ms |','| --- | ---: | ---: | ---: |']
    for d in s['colocation_paired_changes']:
        lines.append('| '+d['condition']+' | '+' | '.join(fmt(d[k]) for k in ('mean_e2e_delta_ms','worker_delta_ms','queue_delta_ms'))+' |')
    for d in s['colocation_paired_changes']:
        if d['condition']=='C_no_active_contender': continue
        values=d['mean_e2e_delta_ms']['values']
        if values: lines.append(f"{d['condition']}：{len(values)} 个配对seed中 {sum(v>0 for v in values)} 个E2E增加；单轮delta={values} ms。")
    active=[d for d in s['colocation_paired_changes'] if d['condition']!='C_no_active_contender' and d['mean_e2e_delta_ms']['n']==3]
    if active:
        increases=[d['condition'] for d in active if all(v>0 for v in d['mean_e2e_delta_ms']['values'])]
        effects=[d['mean_e2e_delta_ms']['mean'] for d in active]
        lines.append(f"已齐全的竞争条件中，配对平均E2E变化范围为 {min(effects):.2f} 至 {max(effects):.2f} ms；三轮均增加的条件：{increases or 'none'}。这只衡量固定u/q下观察到的附加变化，尚未验证 C 的解释充分性或预测能力。")
    lines+=['','C仍存在调度/样本混杂：原harness按task生成arrival后routing。同task加入竞争流量会改变Qwen请求间隔和sample subsequence；JSON保留每seed的样本序列相等检查及音频时长。因此不宣称因果识别。',
        '', '6. **哪些resource dimensions最一致？** 只对四个condition均值计算探索性相关性：','',
        '| Feature | Spearman | Pearson |','| --- | ---: | ---: |']
    for r in s['resource_pressure_associations']: lines.append(f"| {r['feature']} | {r['spearman']:.3f} | {r['pearson']:.3f} |")
    lines+=['','四个点、多个共线维度不能识别真正资源瓶颈，相关性不当作predictor或显著性证据。',
        '', '7. **单独u→L baseline是否不足？** B 的相同u存在上表中的大幅实测差异，单独u无法描述本批数据的全部变化；但重复间波动说明加入q/C也尚未被证明充分。A插值只在固定mix/context内使用，不能外推到不同mix/co-location。',
        f"   本次piecewise baseline初始化：{s['piecewise_baseline']['initialized']}；合格稳定点：{s['piecewise_baseline']['eligible_load_conditions']}。原始异常点仍保留在完整结果中。",
        '', '8. **是否足以开始cross-deployment predictor？** 尚不足以支撑可信的首版跨deployment训练/验证；本批是Qwen探索性数据基础。C只有一个固定驻留环境、三条单一竞争路径及单一压力强度，缺独立deployment holdout、多竞争者组合、输入长度控制和更多压力强度；还需解释重复间漂移与截断。此次不训练。',
        '', '## Data quality and boundaries','',
        '| Condition | Qwen runtime ms | Qwen schedule delay ms |','| --- | ---: | ---: |']
    for c in s['conditions']:
        if c['repetitions']: lines.append('| '+c['name']+' | '+fmt(c['qwen']['mean_runtime_latency_ms'])+' | '+fmt(c['qwen']['mean_schedule_delay_ms'])+' |')
    lines += ['',
        '此表runtime和schedule delay与E2E使用同一成功请求群体，便于分解；原harness的task schedule delay统计包含所有已提交请求，定义保留在raw。主机sender排队也可能放大E2E，不能把全部增长归为GPU worker queue。',
        '原始结果含source hashes、seed、arrival phase、request records、错误/incomplete/drain；低于30成功样本的路径标记tail_sample_shortfall。',
        '实验环境固定OMP/MKL/OPENBLAS=1，三次全新进程。A按负载升序，B/C每个重复内固定seed随机condition顺序；不能排除时间漂移。',
        '没有枚举deployment、没有训练MLP/GBDT等模型、没有guided search/routing optimization。','']
    (output/'profiling_summary.md').write_text('\n'.join(lines))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',type=Path,default=OUT); parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args(); summary=analyze(args.output_dir,not args.no_plots)
    print(json.dumps({k:summary[k] for k in ('status','runs_saved','configurations_observed','dataset_rows')},indent=2))
