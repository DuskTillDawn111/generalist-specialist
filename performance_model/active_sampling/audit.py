"""Merge real runs, preserve grouping and censoring, audit features without fitting."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
from performance_model.dataset import build_rows
from performance_model.resources import source
from performance_model.piecewise_baseline import same_context
from performance_model.stability import label_stability
from performance_model.profiling.analyze import audit_run
from .common import *
from .grouping import evaluation_configuration_id,protocol_id


def coverage(runs):
    vectors=[r['feature_vector'] for r in runs if r['feature_vector'] is not None]
    keys=sorted({k for v in vectors for k in v})
    return dict(independent_runs=len(runs),configuration_count=len({r['configuration_id'] for r in runs}),
        deployment_count=len({r['deployment_id'] for r in runs}),
        deployments={r['deployment_id']:r['deployment'] for r in runs},
        stability_counts=dict(Counter(r['stability']['category'] for r in runs)),
        feature_ranges={k:dict(min=min(v[k] for v in vectors if v.get(k) is not None),
            max=max(v[k] for v in vectors if v.get(k) is not None),distinct=len({v[k] for v in vectors if v.get(k) is not None}),
            missing=sum(v.get(k) is None for v in vectors)) for k in keys if any(v.get(k) is not None for v in vectors)},
        deployment_groups={did:dict(runs=len(group),configurations=len({r['configuration_id'] for r in group}),
            roles=dict(Counter(r['role'] for r in group)),stability_counts=dict(Counter(r['stability']['category'] for r in group)))
            for did,group in group_by(runs,'deployment_id').items()})


def group_by(rows,key):
    groups=defaultdict(list)
    for row in rows: groups[row[key]].append(row)
    return groups


def audit(output=OUT):
    output=Path(output);plan=json.loads((output/'selected_conditions.json').read_text());pool=json.loads((output/'candidate_pool.json').read_text());ex=extractor()
    sources=existing_sources(); spec_by_path={s['result']:(c,s) for c in plan['conditions'] for s in c['runs']}
    sources += [(Path(p),'active_sampling','active_formal') for p in spec_by_path if Path(p).exists()]
    rows=[];runs=[];errors=[]
    for path,stage,role in sources:
        data=json.loads(path.read_text());f=ex.extract(data['deployment'],data['workload_rps'],data['routing_rps']);issues=[]
        label=label_stability(data,plan['policy']['stability_thresholds']).as_dict();cid=evaluation_configuration_id(data);did=deployment_id(data['deployment'])
        path_rows=[]
        if data.get('measurement_window'):
            try: path_rows,_=build_rows(path,stage,role,ex)
            except ValueError as error: issues.append(str(error))
        if str(path) in spec_by_path:
            condition,spec=spec_by_path[str(path)]
            cfg=json.loads(Path(spec['config']).read_text())
            if source(spec['config'])['sha256']!=spec['config_sha256']: issues.append('config hash changed')
            for k,v in cfg.items():
                actual=data.get(k)
                if k=='arrival_process' and isinstance(actual,dict): actual=actual.get('name')
                if actual!=v: issues.append('config mismatch: '+k)
            if not same_context(f,condition['features']): issues.append('features mismatch')
            if configuration_id(data)!=condition['configuration_id'] or did!=condition['deployment_id']: issues.append('acquisition IDs mismatch')
            # Reuse the existing exact arrival/router/sample-sequence audit when a full window exists.
            if data.get('measurement_window'):
                old_policy={'completion_ratio_threshold':plan['policy']['stability_thresholds']['completion_ratio_min'],
                            'drain_overload_seconds':plan['policy']['stability_thresholds']['drain_unstable_seconds']}
                detail=audit_run(data,spec,{'expected_features':f},{'measurement_policy':old_policy},ex)
                issues.extend(detail['issues'])
        short=[dict(task=r['task'],model=r['target_model'],completed=r['labels']['completed']) for r in path_rows if r['labels']['completed']<plan['policy']['min_requests_per_path']]
        if short: label['stable_region_latency_eligible']=False
        meta=dict(run_id=path.stem,configuration_id=cid,deployment_id=did,seed=data['random_seed'],stage=stage,role=role,
            acquisition_condition_id=configuration_id(data),protocol_id=protocol_id(data),
            source_result_path=str(path.resolve()),source_sha256=source(path)['sha256'],deployment=data['deployment'],
            workload=data['workload_rps'],routing=data['routing_rps'],feature_vector=feature_vector(f) if 'qwen' in f['per_model'] else None,
            features=f,stability=label,tail_sample_shortfalls=short,status=data['status'],system_metrics=data.get('system_metrics'),
            per_task_metrics=data.get('per_task_metrics',{}),protocol_source=data.get('source'),
            integrity_passed=not issues,issues=issues,dataset_rows=len(path_rows),
            expected_category=spec_by_path[str(path)][0]['expected_category'] if str(path) in spec_by_path else None)
        runs.append(meta)
        for row in path_rows:
            row.update(configuration_id=cid,deployment_id=did,acquisition_condition_id=configuration_id(data),protocol_id=protocol_id(data),stability=label,tail_sample_shortfall=row['labels']['completed']<plan['policy']['min_requests_per_path'])
        rows.extend(path_rows);errors.extend(issues)
    if len({r['run_id'] for r in runs})!=len(runs): raise ValueError('Duplicate run IDs; grouping would be ambiguous')
    for group in group_by(runs,'configuration_id').values():
        for rep,r in enumerate(sorted(group,key=lambda r:(r['seed'],r['source_result_path'])),1): r['repetition_id']=rep
    byrun={r['run_id']:r for r in runs}
    for row in rows: row['repetition_id']=byrun[row['run_id']]['repetition_id']
    before=[r for r in runs if r['stage']!='active_sampling'];new=[r for r in runs if r['stage']=='active_sampling']
    old_vectors=[r['feature_vector'] for r in before if r['feature_vector'] is not None];new_vectors=[r['feature_vector'] for r in new if r['feature_vector'] is not None]
    # One vector per distinct x/r/lambda condition, never per task label or repetition.
    candidates={identity(c['feature_vector']):c['feature_vector'] for c in pool['candidate_conditions']}
    def gaps(vectors): return [nearest(v,vectors,pool['distance_space']) for v in candidates.values()]
    bg,ag=gaps(old_vectors),gaps(old_vectors+new_vectors)
    expanded=[k for k,v in coverage(runs)['feature_ranges'].items() if k not in coverage(before)['feature_ranges'] or
        v['min']<coverage(before)['feature_ranges'][k]['min'] or v['max']>coverage(before)['feature_ranges'][k]['max']]
    new_deps=sorted({r['deployment_id'] for r in new}-{r['deployment_id'] for r in before})
    features=dict(before=coverage(before),after=coverage(runs),new_runs=coverage(new),new_deployment_ids=new_deps,
        expanded_feature_ranges=expanded,normalization=pool['distance_space'],
        local_candidate_space_coverage=dict(unique_candidate_vectors=len(candidates),
            mean_nearest_distance_before=statistics.mean(bg),mean_nearest_distance_after=statistics.mean(ag),
            max_nearest_distance_before=max(bg),max_nearest_distance_after=max(ag),
            mean_distance_reduction_fraction=1-statistics.mean(ag)/statistics.mean(bg) if statistics.mean(bg) else 0.,
            scope='Only the declared finite local candidate feature space; not a global deployment-space coverage claim.'),
        selected_distances=[dict(configuration_id=evaluation_configuration_id(c['config_template']),acquisition_condition_id=c['configuration_id'],distance_to_existing=c['nearest_existing_feature_distance'],selection_score=c['selection_score']) for c in plan['conditions']])
    save(output/'feature_coverage_before_after.json',features)
    save(output/'dataset_rows.jsonl.meta.json',dict(rows=len(rows),unit='task-model path label within one run; rows from a run are dependent'))
    (output/'dataset_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    groups={level:{key:[r['run_id'] for r in group] for key,group in group_by(runs,level).items()} for level in ('configuration_id','deployment_id')}
    frozen_changes=[p for p,h in plan['protected_sources'].items() if source(ROOT/p)['sha256']!=h]
    old_changes=[r['path'] for r in pool['prior_sources'] if source(r['path'])['sha256']!=r['sha256']]
    formal=[r for r in runs if r['role'] not in ('smoke','smoke_gate')]
    readiness=dict(formal_deployment_contexts=len({r['deployment_id'] for r in formal}),
        formal_configuration_groups=len({r['configuration_id'] for r in formal}),
        formal_contexts_with_unstable_examples=sorted({r['deployment_id'] for r in formal if r['stability']['category']=='unstable'}),
        formal_contexts_with_strict_stable_latency_examples=sorted({r['deployment_id'] for r in formal if r['stability']['stable_region_latency_eligible']}),
        interpretation='Multiple contexts allow a grouped pilot validation design. Reliable unified cross-deployment training remains unestablished: inspect per-context class support, comparable observation horizons and holdouts; no model trained.')
    manifest=dict(unit='run is an observation; configuration/deployment are evaluation groups, NOT per-task rows',
        original_runs=len(before),new_runs=len(new),total_runs=len(runs),per_task_rows=len(rows),runs=runs,groups=groups,
        grouping_rules=['configuration_id groups canonical x/r/lambda irrespective of measurement duration, seed or protocol; acquisition_condition_id identifies the frozen acquisition condition including duration.',
            'Never split task rows or repetitions of the same configuration across folds.',
            'Cross-deployment evaluation must hold out entire deployment_id groups.',
            'Fit preprocessing and any model only on training groups; do not use full-dataset normalization for evaluation.',
            'Smoke and formal protocols differ; use roles/protocol metadata to define a comparable evaluation cohort.'],
        integrity_passed=not errors and not frozen_changes and not old_changes,issues=errors,
        changed_protected_sources=frozen_changes,changed_prior_results=old_changes,
        expected_new_runs=18,new_run_budget_complete=len(new)==18,validation_readiness=readiness)
    save(output/'dataset_manifest.json',manifest)
    labels=dict(thresholds=plan['policy']['stability_thresholds'],
        definition='stable boolean means no instability criterion. boundary is operationally complete but has backlog/drain warnings; strict stable-region latency eligibility excludes boundary and short samples.',
        before=coverage(before)['stability_counts'],new=coverage(new)['stability_counts'],after=coverage(runs)['stability_counts'],
        new_by_expected={k:dict(Counter(r['stability']['category'] for r in g)) for k,g in group_by(new,'expected_category').items()},
        run_labels={r['run_id']:r['stability'] for r in runs},
        configuration_labels={k:dict(runs=len(g),categories=dict(Counter(r['stability']['category'] for r in g)),
            stable_repetitions=sum(r['stability']['category']=='stable' for r in g)) for k,g in group_by(runs,'configuration_id').items()})
    save(output/'stability_label_summary.json',labels)
    lines=['# Module 2 active profiling audit','',
        f"Selected conditions: 6; saved new runs: {len(new)}/18; prior runs: {len(before)}; total runs: {len(runs)}; task-model rows: {len(rows)}.",
        '**Rows are not independent deployment samples.** run_id/configuration_id/deployment_id/repetition_id are retained in rows and run manifest.',
        '', '| Condition | Mutation | Expected category | Qwen u | seconds | distance to existing | observed categories |','| --- | --- | --- | ---: | ---: | ---: | --- |']
    for c in plan['conditions']:
        g=[r for r in new if r['acquisition_condition_id']==c['configuration_id']]
        lines.append(f"| {c['name']} | {c['mutation']['operation']} {c['mutation']['model']} | {c['expected_category']} | {c['features']['per_model']['qwen']['standalone_service_demand']:.3f} | {c['measurement_seconds']} | {c['nearest_existing_feature_distance']:.4f} | {dict(Counter(r['stability']['category'] for r in g))} |")
    local=features['local_candidate_space_coverage']
    lines += ['',f"Only {pool['local_mutation_count']} immediate mutations were considered; {pool['candidate_condition_count']} bounded feature conditions, {pool['considered_but_not_selected_conditions']} not selected. No recursive expansion or exhaustive deployment enumeration.",
        '', '## Audit answers','',
        f"1. Feature coverage: mean distance of the finite candidate feature space to measured coverage fell {local['mean_nearest_distance_before']:.4f} → {local['mean_nearest_distance_after']:.4f} ({local['mean_distance_reduction_fraction']:.1%}); expanded ranges: {expanded}. This is local feature coverage, not prediction accuracy or global coverage.",
        f"2. Observed new stable/boundary/unstable counts: {labels['new']}; combined counts: {labels['after']}. Expected bands were acquisition priors; labels come only from saved run evidence. No extra runs are added to force a desired label.",
        f"3. New deployment contexts: {len(new_deps)}; deployment count {features['before']['deployment_count']} → {features['after']['deployment_count']}. IDs only support auditing/grouping and are absent from acquisition vectors.",
        f"4. Grouped cross-deployment validation: {readiness['formal_deployment_contexts']} formal deployment contexts and {readiness['formal_configuration_groups']} configuration groups support preparing an exploratory grouped holdout design. Unstable examples occur in {len(readiness['formal_contexts_with_unstable_examples'])} formal contexts; strict all-path stable-latency examples in {len(readiness['formal_contexts_with_strict_stable_latency_examples'])}. These counts do not yet establish reliable unified stability/latency/throughput training: context/load/mix may be confounded, and smoke protocols, short risk windows and input/seed variation require controls. No model was trained.",
        '', '## Quality and limitations','',
        f"Integrity passed: {manifest['integrity_passed']}; changed frozen sources: {frozen_changes}; changed existing results: {old_changes}.",
        f"New runs with <30 successful samples on any positive path: {sum(bool(r['tail_sample_shortfalls']) for r in new)}. P95 is descriptive for these paths, not a reliable tail estimate.",
        'Latencies use successful requests only; incomplete/error records remain. Stable-region latency eligibility excludes warning/unstable runs and short samples. Labels are operational and horizon-dependent, not proof of infinite-horizon queue stability.',
        'The acquisition schedule uses different observation horizons across risk bands. A future stability classifier must address that label-horizon confound rather than treating absence of overload in a short window as proof of stability.',
        'Safety thresholds are serialized in selected_conditions.json, raw results and stability_label_summary.json. Backlog is detected after each bounded original-harness window; this wrapper does not claim live backlog-triggered interruption. Hard host-queue limits, drain timeouts and overall timeouts bound failures. Likely-unstable windows are capped at 180 seconds.',
        'All new features come from the shared offline extractor; u and C are standalone/offline estimates, not measured GPU utilization. Real-time queue evidence is used as a label, not a planning input.',
        'The feature-distance normalization is an acquisition artifact fitted before new measurement; it must not be reused as train/test preprocessing without refitting on training groups.',
        'Completed this budget only. No predictor training, deployment search, routing optimization, forecasting or replica extension.','']
    (output/'summary.md').write_text('\n'.join(lines))
    print(json.dumps({'new_runs':len(new),'total_runs':len(runs),'rows':len(rows),'new_deployments':len(new_deps),'new_labels':labels['new'],'integrity_passed':manifest['integrity_passed']},indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=OUT);args=parser.parse_args();audit(args.output_dir)
