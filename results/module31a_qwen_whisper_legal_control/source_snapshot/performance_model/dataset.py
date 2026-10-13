"""Convert Stage 2A/2B-1 real results to service-path rows; audit only, no training."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from deployment_core.models import load_model_catalog
from deployment_core.workload import TASKS
from joint_deploy.run_pure_qwen import percentile
from .features import FeatureExtractor, curve_context
from .piecewise_baseline import same_context
from .resources import ROOT, load_resource_profiles, source

OUT=ROOT/'results/performance_model_foundation'


def input_results():
    sources=[(ROOT/'results/stage2a/smoke_text.json','stage2a','smoke'),
             (ROOT/'results/stage2a/smoke_multitask.json','stage2a','smoke')]
    sources += [(p,'stage2b_1','formal') for p in sorted((ROOT/'results/stage2b/routing_characterization/raw').glob('*.json'))]
    smoke=ROOT/'results/stage2b/routing_characterization/full_deployment_smoke.json'
    if smoke.exists(): sources.append((smoke,'stage2b_1','smoke_gate'))
    return sources


def metric(records,key,quantile=None):
    values=[r[key] for r in records if r.get(key) is not None]
    if not values: return None
    return percentile(values,quantile) if quantile is not None else statistics.mean(values)


def build_rows(path,stage,role,extractor):
    path=Path(path).resolve(); data=json.loads(path.read_text())
    records=data['request_records']; window=data.get('measurement_window')
    if not window: raise ValueError('No measurement window: '+str(path))
    features=extractor.extract(data['deployment'],data['workload_rps'],data['routing_rps'])
    duration=data['measurement_seconds']; end=window['end_time_ns']
    if duration<=0: raise ValueError('Invalid measurement duration')
    errors=[]
    if len({r['request_id'] for r in records})!=len(records): errors.append('duplicate request IDs')
    for r in records:
        if r.get('phase')!='measurement': errors.append('Non-measurement request in formal records')
        if r['task'] not in features['routing'] or features['routing'][r['task']].get(r['model'],0)<=0:
            errors.append('Observed unsupported/nonconfigured routing')
        if not window['start_time_ns']<=r['scheduled_time_ns']<end: errors.append('Request outside cohort')
        if r['status']=='completed':
            if not r['scheduled_time_ns']<=r['actual_submit_time_ns']<=r['completion_time_ns']:
                errors.append('Invalid successful timestamp order')
            computed=(r['completion_time_ns']-r['scheduled_time_ns'])/1e6
            if not math.isclose(computed,r['e2e_latency_ms'],rel_tol=1e-10,abs_tol=1e-9): errors.append('E2E mismatch')
    successful=[r for r in records if r['status']=='completed']
    sm=data['system_metrics']
    if sm['offered_requests']!=len(records) or sm['completed_requests']!=len(successful): errors.append('System counters mismatch')
    for task in TASKS:
        task_rows=[r for r in records if r['task']==task]
        if not task_rows: continue
        completed=[r for r in task_rows if r['status']=='completed']
        original=data['per_task_metrics'][task]
        if original['offered_requests']!=len(task_rows) or original['completed_requests']!=len(completed): errors.append('Task counter mismatch')
        for key,value in [('mean_e2e_latency_ms',metric(completed,'e2e_latency_ms')),('p95_e2e_latency_ms',metric(completed,'e2e_latency_ms',95))]:
            old=original[key]
            if (old is None)!=(value is None) or (value is not None and not math.isclose(old,value,rel_tol=1e-10,abs_tol=1e-9)):
                errors.append('Task latency aggregation mismatch')
    if errors: raise ValueError(str(path)+': '+'; '.join(sorted(set(errors))))
    output=[]
    for task in TASKS:
        for model,rate in features['routing'][task].items():
            if rate<=0: continue
            cohort=[r for r in records if r['task']==task and r['model']==model]
            complete=[r for r in cohort if r['status']=='completed']
            in_window=sum(r['completion_time_ns']<end for r in complete)
            target=features['per_model'][model]
            resource=extractor.resource_profiles.get((task,model),{'values':{},'missing':{'profile':'missing'}})
            output.append(dict(deployment=features['deployment'],workload=features['workload'],routing=features['routing'],
                target_model=model,task=task,standalone_service_demand=target['standalone_service_demand'],
                task_demand_mix=target['task_demand_mix'],task_order=list(TASKS),
                deployment_features=features['deployment_features'],model_features={k:v for k,v in target.items() if k!='colocation_pressure'},
                service_path_resource_features=resource['values'],resource_feature_missing=resource['missing'],
                resource_profile_sources=resource.get('sources',[]),colocation_pressure=target['colocation_pressure'],
                curve_context=curve_context(features,model),
                labels=dict(mean_e2e_latency_ms=metric(complete,'e2e_latency_ms'),p95_e2e_latency_ms=metric(complete,'e2e_latency_ms',95),
                    worker_latency_ms=metric(complete,'worker_latency_ms'),worker_p95_latency_ms=metric(complete,'worker_latency_ms',95),
                    queue_wait_ms=metric(complete,'worker_queue_wait_ms'),queue_wait_p95_ms=metric(complete,'worker_queue_wait_ms',95),
                    runtime_latency_ms=metric(complete,'runtime_latency_ms'),schedule_delay_ms=metric(complete,'schedule_delay_ms'),
                    throughput_rps=len(complete)/duration,within_window_throughput_rps=in_window/duration,
                    offered=len(cohort),completed=len(complete),failed=sum(r['status']=='error' for r in cohort),
                    incomplete=sum(r['status'] not in ('completed','error') for r in cohort),
                    completion_ratio=len(complete)/len(cohort) if cohort else None,
                    completed_during_drain=len(complete)-in_window),
                task_aggregate_labels=data['per_task_metrics'][task],system_labels=sm,
                observation_counts={key:sum(r.get(key) is not None for r in complete) for key in ('worker_latency_ms','worker_queue_wait_ms','e2e_latency_ms')},
                source_result_path=str(path),source_sha256=source(path)['sha256'],run_id=path.stem,seed=data['random_seed'],
                stage=stage,role=role,measurement_seconds=duration,warmup_seconds=data['warmup_seconds'],
                protocol_source=data['source'],initial_arrival_phases=data['arrival_process']['initial_phase_seconds'],
                run_status=data['status'],source_errors=data.get('errors',[]),
                label_scope='task-model service path; task/system aggregates are repeated metadata, never additive across rows'))
    return output,dict(**source(path),stage=stage,role=role,run_id=path.stem,seed=data['random_seed'],
        rows=len(output),requests=len(records),integrity_passed=True,status=data['status'],
        deployed_model_loads={m:p['standalone_service_demand'] for m,p in features['per_model'].items()})


def span(values):
    values=[v for v in values if v is not None]
    return dict(min=min(values) if values else None,max=max(values) if values else None,distinct=len(set(values)))


def audit_curves(rows):
    groups=[]
    for row in rows:
        group=next((g for g in groups if g['target_model']==row['target_model'] and g['task']==row['task']
                    and g['role']==row['role'] and same_context(g['context'],row['curve_context'])),None)
        if group is None:
            group=dict(target_model=row['target_model'],task=row['task'],role=row['role'],context=row['curve_context'],rows=[])
            groups.append(group)
        group['rows'].append(row)
    coverage=[]
    for g in groups:
        xs=sorted({r['standalone_service_demand'] for r in g['rows']})
        coverage.append(dict(target_model=g['target_model'],task=g['task'],role=g['role'],
            distinct_u=len(xs),u_values=xs,rows=len(g['rows']),source_paths=sorted({r['source_result_path'] for r in g['rows']})))
    return dict(controlled_context_groups=coverage,groups_with_two_distinct_loads=sum(g['distinct_u']>=2 for g in coverage),
                sufficient_for_controlled_piecewise_curve=any(g['distinct_u']>=2 and g['role']=='formal' for g in coverage),
                automatic_profile_points_created=0,
                rule='Match task mix, workload composition, deployment memory/count features and pressure/u; same role. Replicate seeds are not new load points.')


def near_load_comparisons(rows):
    groups={}
    for row in rows:
        key=json.dumps({k:row[k] for k in ('deployment','workload','routing','target_model','task','role')},sort_keys=True)
        groups.setdefault(key,[]).append(row)
    groups=list(groups.values()); pairs=[]
    for i,left in enumerate(groups):
        a=left[0]
        for right in groups[i+1:]:
            b=right[0]
            if a['target_model']!=b['target_model'] or a['task']!=b['task']: continue
            ua,ub=a['standalone_service_demand'],b['standalone_service_demand']
            mix_l1=sum(abs(a['task_demand_mix'][k]-b['task_demand_mix'][k]) for k in TASKS)
            if abs(ua-ub)>.05 or abs(ua-ub)/max(ua,ub,1e-30)>.2 or mix_l1<.1: continue
            def summary(group):
                return dict(u=group[0]['standalone_service_demand'],task_demand_mix=group[0]['task_demand_mix'],
                    mean_e2e_latency_ms=metric([r['labels'] for r in group],'mean_e2e_latency_ms'),
                    mean_run_p95_e2e_latency_ms=metric([r['labels'] for r in group],'p95_e2e_latency_ms'),
                    runs=len(group),role=group[0]['role'],paths=[r['source_result_path'] for r in group])
            pairs.append(dict(model=a['target_model'],task=a['task'],mix_l1=mix_l1,left=summary(left),right=summary(right),
                              caveat='Different deployment/protocol/arrival phases may confound difference; not a causal task-mix comparison.'))
    return dict(criteria='Same model/task, |delta u|<=0.05 and relative delta<=20%, mix L1>=0.1; exploratory comparison only',pairs=pairs)


def build_dataset(output=OUT):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    catalog=load_model_catalog(); resources=load_resource_profiles(catalog)
    extractor=FeatureExtractor(catalog,resources)
    rows=[]; sources=[]
    for path,stage,role in input_results():
        part,provenance=build_rows(path,stage,role,extractor)
        rows.extend(part); sources.append(provenance)
    row_path=output/'dataset_rows.jsonl'
    row_path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    keys=sorted({k for p in resources.values() for k in p['values']})
    deployments=list({json.dumps(r['deployment'],sort_keys=True):r['deployment'] for r in rows}.values())
    coverage=dict(rows=len(rows),runs=len(sources),deployments=deployments,tasks=sorted({r['task'] for r in rows}),
        stage_role_counts=dict(Counter(r['stage']+'/'+r['role'] for r in rows)),
        model_load_ranges={m:span([s['deployed_model_loads'][m] for s in sources if m in s['deployed_model_loads']]) for m in sorted(catalog)},
        active_service_path_model_load_ranges={m:span([r['standalone_service_demand'] for r in rows if r['target_model']==m]) for m in sorted(catalog)},
        task_mix_ranges={m:{k:span([r['task_demand_mix'][k] for r in rows if r['target_model']==m]) for k in TASKS} for m in sorted(catalog)},
        resource_features={k:dict(available_paths=sum(p['values'].get(k) is not None for p in resources.values()),
            total_paths=len(resources),missing_paths=[f'{m}/{t}' for (t,m),p in resources.items() if p['values'].get(k) is None],
            available_rows=sum(r['service_path_resource_features'].get(k) is not None for r in rows),
            complete_pressure_rows=sum(r['colocation_pressure'].get(k,{}).get('complete',False) for r in rows)) for k in keys},
        missing_dimensions=['No matched load-intensity sweep at fixed routing proportions/task mix and deployment.',
            'Only one formal deployment; two additional deployments are short Stage 2A smoke runs.',
            'No controlled equal-u/different-mix experiments; no held-out deployment evaluation.',
            'No sustained overload/saturation labels, input length/duration distribution features or latency distribution model.',
            'No per-process resource attribution / SM occupancy / hardware memory-bandwidth counters; NVML and kernel categories do not provide these.',
            'No multi-replica calibration; planning baseline explicitly declines count>1.'],
        curve_audit=audit_curves(rows),near_load_task_mix_audit=near_load_comparisons(rows),
        offline_resource_profiles=[dict(task=k,model=m,**p) for (k,m),p in resources.items()])
    manifest=dict(schema_version=1,rows=len(rows),runs=len(sources),request_records=sum(s['requests'] for s in sources),
        dataset=source(row_path),sources=sources,task_order=list(TASKS),
        row_granularity='one run x positive-routing task-model path; labels recomputed from matching successful measurement requests only',
        label_units='latency ms; throughput requests/s; cohort throughput uses full configured window denominator including completions during bounded drain',
        leakage_rules=['Queue wait/runtime/throughput and task/system measured metrics are labels/metadata, never mandatory planning features.',
            'All rows of a run must remain in the same future split; deployment holdout needed for cross-deployment evaluation.',
            'Do not sum task_aggregate_labels/system_labels repeated across service-path rows.',
            'Smoke is a separate role with small path sample counts; do not treat it as full characterization.'],
        standalone_catalog={m:dict(memory_mib=p.memory_mib,service_time_ms=dict(p.service_time_ms),sources=p.sources) for m,p in catalog.items()},
        trained=False,predictor_accuracy_reported=False,profile_points_created=0)
    for name,data in [('dataset_manifest.json',manifest),('feature_coverage.json',coverage)]:
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_summary(output,manifest,coverage)
    return manifest,coverage


def write_summary(output,manifest,coverage):
    lines=['# Performance Model Foundation / Module 2','',
        f"真实结果：{manifest['runs']} runs，{manifest['request_records']} measurement requests，{manifest['rows']} task-model rows；{len(coverage['deployments'])} 个实际观测 deployment。",
        '包含 Stage 2A 两个 smoke、Stage 2B-1 十五轮正式结果和一个 full-deployment smoke gate。未导入旧 closed-loop pair experiments 作为 E2E labels。',
        '每条 row 保留完整 x/r/lambda、source result 绝对路径、哈希、run ID、seed 和协议。latency 按真实 task-model 请求重新聚合，不复制整个 task 均值。',
        'mean/P95 单位 ms；throughput 为 cohort completions/window seconds，同时保留 within-window throughput、drain count 和 incomplete。',
        '标签只基于成功请求；incomplete/error 与观测数单独保留。跨 run 的 P95 平均不是 pooled P95。','',
        '| Observed deployment | Count vector |','| --- | --- |']
    for i,x in enumerate(coverage['deployments'],1): lines.append(f'| observed {i} (audit label only) | {x} |')
    lines+=['','正式 predictor 不接收 deployment_id，不调用枚举，也不要求全组合 profile。以上列表只统计已观测覆盖。','',
        '| Model | observed u min | observed u max | distinct u |','| --- | ---: | ---: | ---: |']
    for m,s in coverage['model_load_ranges'].items(): lines.append(f"| {m} | {s['min']:.8f} | {s['max']:.8f} | {s['distinct']} |")
    lines+=['','u=Σ r*s0，为 standalone_service_demand，不能称为真实 GPU utilization。任务顺序固定为 '+', '.join(TASKS)+'。',
        '上表包含 deployed-but-idle 模型的 u=0；task-mix 范围仅对 u>0 定义，完整 min/max/distinct 见 feature_coverage.json。','',
        '## 1. 是否足够建立 u → L piecewise curve？','',
        f"受控 context 内至少两个 distinct u 的组数：{coverage['curve_audit']['groups_with_two_distinct_loads']}；正式数据是否足够：{coverage['curve_audit']['sufficient_for_controlled_piecewise_curve']}。",
        '当前不生成 LoadPoint。五个 routing 点虽然带来多个 u，却同时改变 Qwen task mix 和其他模型负载，不能直接串成负载曲线。',
        'PiecewiseLoadLatencyModel 默认返回 insufficient_profile_data；未来只在同 feature context 的实测范围内插值，不外推。多路径任务不能平均 P95，缺少分布时返回 null。','',
        '## 2. 相同/相近 u、不同 task mix 是否出现 latency 差异？','',coverage['near_load_task_mix_audit']['criteria']]
    pairs=coverage['near_load_task_mix_audit']['pairs']
    if not pairs: lines+=['没有满足上述口径的可比较数据，不能判断 task-mix effect。']
    for p in pairs:
        a,b=p['left'],p['right']
        lines += [f"- {p['model']}/{p['task']}: u={a['u']:.5f} vs {b['u']:.5f}，mix L1={p['mix_l1']:.3f}；mean={a['mean_e2e_latency_ms']:.2f} vs {b['mean_e2e_latency_ms']:.2f} ms，平均 run P95={a['mean_run_p95_e2e_latency_ms']:.2f} vs {b['mean_run_p95_e2e_latency_ms']:.2f} ms（{a['runs']} vs {b['runs']} runs）。"]
    lines+=['','这些是描述性差异。smoke 与正式结果的 deployment、warmup、窗口、seed/sample/phase 也不同；不能把差异单独归因于 mix，更不宣称统计显著性。完整成对来源在 feature_coverage.json。',
        '', '## 3. 已有 deployment/resource features','',
        'Stage 1 catalog：五模型实测驻留显存、11 条路径各90样本的 standalone service time。',
        'NVML 离线设备级均值：GPU/memory utilization、power、SM clock、device VRAM、energy/request；Nsight 既有派生表：kernel time/request 和按名称分类的时间比例。',
        '按字段核对已有 NVML summary 和 service identity；Nsight 分类复用现成 CSV 并保留原 report/派生脚本路径，不运行 nsys。',
        '', '| Resource field | Available paths | Missing paths | Complete co-location rows |','| --- | ---: | --- | ---: |']
    for k,v in coverage['resource_features'].items():
        lines.append(f"| {k} | {v['available_paths']}/{v['total_paths']} | {', '.join(v['missing_paths']) or 'none in current 11 paths'} | {v['complete_pressure_rows']}/{manifest['rows']} |")
    lines+=['','现有路径字段完整不等于物理资源画像完整：NVML 约100ms采样、部分窗口只有约1–3秒，设备级值不是进程级；kernel 分类只是名称启发式。',
        'co-location pressure 对每个动态 feature 计算 Σ r*p；单位是对应 fingerprint 单位乘 req/s，不是实际 GPU 利用率或经验证的物理压力。',
        '任一正负载贡献路径缺值时总量为 null，另保留 known_partial_sum 与 missing_paths；零负载贡献为零。','',
        '## 4. 跨 deployment predictor 还缺什么？','']
    lines += ['- '+s for s in coverage['missing_dimensions']]
    lines+=['','Planning predictor 只用 x/r/lambda/offline profiles。实时 queue length、实时 GPU utilization、recent latency 归未来独立 runtime correction，本轮不实现。','',
        '## 5. 下一轮最少采哪些点？','',
        '建议先采一个可解释的最小设计（只是采样建议，本轮不执行）：','',
        '1. x_full 下固定 R0/R50/R100 各自 routing proportions，先取 scale=0.25/0.50/0.75/1.00，各3 seeds；沿用至少约30正式请求/task、有界 drain 和 adaptive overload stop。这是最多12配置/36轮的首轮，不保证所有 routing 都到1.0。',
        '2. 未出现上升区间的 routing 再加1.25/1.50/2.00；发生 overload 即停止更高点。knee 由真机数据观察，不以 rho=0.8 定义。',
        '3. 从稳定低/中负载选至少两个固定 u 的点，构造长 ASR 占比高与短 text/image 占比高的 mix，各3 seeds；保持相同 deployment、相同其他模型负载与测量协议。VQA probe rate 固定，避免混淆自身负载。',
        '4. 至少两个额外可行 deployment，各取相同 workload 和两档稳定负载、3 seeds；例如现有 MILP seed（不含 DistilBERT）和 x_full 配对，其中一个完整 deployment 保留作 holdout。先校验可行性，不枚举全组合，也不按性能选优。',
        '5. 每个采样保留输入长度/音频时长分布、sample IDs、初始 phases、完整请求记录及窗口内吞吐。已有资源 fingerprint 可先复用；新增 service path 时补缺失 offline profile，不猜值。',
        '', '这只是启动跨 deployment 建模的数据维度下界，不代表这些点已足够得到可靠 predictor。训练、accuracy evaluation、guided search、routing optimization 均未执行。','']
    (output/'dataset_summary.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    manifest,coverage=build_dataset(args.output_dir)
    print(json.dumps(dict(rows=manifest['rows'],runs=manifest['runs'],deployments=len(coverage['deployments']),
        controlled_curve_groups=coverage['curve_audit']['groups_with_two_distinct_loads']),indent=2))

if __name__=='__main__':
    main()
