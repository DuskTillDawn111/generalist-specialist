"""CPU-only real bridge intake and strict profiler-domain baseline fitting.

No model loads, GPU queries, timing rescaling, invented runs, or runtime edits.
Execution contexts retain the campaign's complete canonical identity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np

from gsdeploy.forecasting.kernel_baseline import (
    FEATURES, GROUP_FIELDS, MANIFEST_SCHEMA, KernelBaselineEstimator,
    adapt_affine_feature_bindings, adapt_raw_kernel_profiles)
from .kernel_domain_bridge_review_v3 import load_report, request_kernels, compare_sequences
from .native_collect import context_identity, receipt, save
from .profiling import digest
from .smoke_verifier_v3 import check_receipt


WEIGHT_SELECTION = {
    'torch.nn.modules.linear.Linear': 'weight',
    'bitsandbytes.nn.modules.Linear8bitLt': 'weight',
    'auto_gptq.nn_modules.qlinear.qlinear_exllama.QuantLinear': 'qweight',
}


def checked(source):
    if not check_receipt(source):
        raise ValueError('Changed or missing receipt: '+str(source))
    return json.loads(Path(source['path']).read_text())


def capture_identity(report, plan, raw, context):
    run = 'raw-sha256:'+report['raw']['sha256']
    if report.get('capture_run_id') != run:
        raise ValueError('Capture run must equal actual light raw SHA, never a job ID')
    identity = context_identity(context)
    if identity != plan['execution_context_sha256'] or identity != raw['execution_context_sha256']:
        raise ValueError('Canonical complete execution context identity mismatch')
    if context.get('measurement_domain') != 'light-kernel' or context.get('affine_observation') is not False:
        raise ValueError('Light timing domain without affine hooks required')
    return run, identity


def explicit_configuration(record, semantic, family, ordinal, signature, mapping_source,
                           metadata_source, context_source, context):
    name = WEIGHT_SELECTION.get(record.get('implementation'))
    weights = [t for t in record.get('direct_tensors', []) if t.get('name') == name]
    if name is None or len(weights) != 1:
        raise ValueError('Unsupported or ambiguous explicit primary weight tensor')
    tensors = [weights[0], record.get('input', {}), record.get('output', {})]
    if any(not isinstance(t.get('dtype'), str) or not t['dtype'] for t in tensors):
        raise ValueError('Observed weight/input/output dtype evidence required')
    batch = semantic.get('input_batch_size')
    if type(batch) is not int or batch < 1 or context['execution'].get('batch_size') != batch:
        raise ValueError('Semantic batch and frozen execution batch must agree')
    precision = dict(scope='observed_tensor_storage_dtypes_not_kernel_accumulation',
                     weight=tensors[0]['dtype'], input=tensors[1]['dtype'], output=tensors[2]['dtype'],
                     configured_execution=context['worker_bindings'][family]['execution'])
    execution = dict(family=family, module=record['module'], implementation=record['implementation'],
                     operator_local_kernel_ordinal=ordinal,
                     launch_settings={k:v for k,v in signature.items() if k not in ('name','grid')})
    # Grid may change with the explicit workload covariates. Every remaining
    # launch setting, module identity, and local kernel position stays exact.
    return dict(kernel_id='observed-kernel-name-sha256:'+digest(signature['name']),
                precision=json.dumps(precision, sort_keys=True, separators=(',', ':')),
                execution_key='observed-operator-execution-sha256:'+digest(execution),
                batch_size=batch, batch_evidence=metadata_source, weight_tensor_name=name,
                operator_to_kernel_evidence=mapping_source,
                explicit_precision=precision, explicit_execution=execution,
                observer_record_id=record['id'], semantic_record_id=semantic['id'],
                configuration_sources=[metadata_source, context_source, mapping_source])


def isolation_receipt(report, plan, raw, output):
    jobs = {j['id']:j for j in plan['jobs']}
    intervals = []
    for row in raw['request_records']+raw.get('priming_request_records', []):
        w = row['worker_response']
        start, end = w['start_perf_ns'], w['end_perf_ns']
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError('Actual positive worker interval required')
        intervals.append(dict(start_ns=start, end_ns=end, request_id=row['request_id'], alias=row['model']))
        if row in raw['request_records']:
            job=jobs[row['profile_job_id']]
            if list(job['inputs']) != [row['model']] or job['context'] != 'solo-'+row['model']:
                raise ValueError('Explicit one-owner solo job required')
    intervals.sort(key=lambda r:r['start_ns'])
    if any(a['end_ns'] > b['start_ns'] for a,b in zip(intervals, intervals[1:])):
        raise ValueError('Concurrent actual worker intervals invalidate isolated baseline')
    if not raw.get('board_samples') or any(s.get('foreign_pids') for s in raw['board_samples']):
        raise ValueError('Strict owned-process telemetry required')
    path=output/'isolation.json'
    save(path, dict(schema='gsdeploy.kernel_baseline_isolation_review.v3',
        raw=report['raw'], plan=report['plan'], intervals=intervals,
        actual_request_intervals_disjoint=True, explicit_solo_jobs=True,
        sampled_telemetry_foreign_pids_absent=True,
        scope='Four resident owners; only one active request. Telemetry is sampled, not proof against undetected external activity.'))
    return receipt(path)


def sqlite_identity(source):
    with sqlite3.connect(f"file:{Path(source['path']).resolve()}?mode=ro",uri=True) as connection:
        connection.row_factory=sqlite3.Row
        devices=[dict(r) for r in connection.execute('select * from TARGET_INFO_GPU')]
        versions=dict(connection.execute('select name,value from META_DATA_EXPORT'))
    if len(devices)!=1 or not devices[0].get('uuid'):
        raise ValueError('Exact single physical GPU UUID required')
    return 'GPU-'+devices[0]['uuid'].removeprefix('GPU-'), versions


def intake_bridge(bridge_path, output):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    bridge_source=receipt(bridge_path); bridge=checked(bridge_source)
    if bridge.get('schema')!='gsdeploy.kernel_three_domain_bridge_review.v3':
        raise ValueError('Real three-domain bridge review required')
    _verify(bridge['evaluator'])
    domains={}
    for name,key in [('metadata','metadata_review'),('light-kernel','light_review'),('natural','natural_review')]:
        checked(bridge[key]); domains[name]=load_report(bridge[key]['path'],name)
    mr,mp,mraw=domains['metadata']; lr,lp,lraw=domains['light-kernel']
    context=checked(lr['context']); run,context_sha=capture_identity(lr,lp,lraw,context)
    for source in context['sources']+[context['large_weight_stats']]: _verify(source)
    isolation=isolation_receipt(lr,lp,lraw,output)
    key=lambda r:(r['profile_job_id'],r['model'],r['pool_id'])
    metadata={key(r):r for r in mraw['request_records']}; light={key(r):r for r in lraw['request_records']}
    rows=[]; rejected=[]; coverage=[]
    by_alias=defaultdict(list)
    for request in bridge['requests']: by_alias[request['alias']].append(request)
    for alias,requests in by_alias.items():
        mo,lo=mr['owners'][alias],lr['owners'][alias]
        mparsed=checked(mo['parsed']); lparsed=checked(lo['parsed'])
        hardware,versions=sqlite_identity(lo['sqlite'])
        if versions.get('EXPORT_PRODUCT_NAME')!='NVIDIA Nsight Systems' or not versions.get('EXPORT_PRODUCT_VERSION'):
            raise ValueError('Explicit Nsight product/version required')
        for request in requests:
            identity=(request['trial_job_id'],alias,request['pool_id'])
            mrow,lrow=metadata[identity],light[identity]
            groups=checked(request['mapping'])
            ms=request_kernels(mparsed,mo['sqlite']['path'],mrow)
            ls=request_kernels(lparsed,lo['sqlite']['path'],lrow)
            if groups!=compare_sequences(ms,ls):
                raise ValueError('Bridge exact mappings differ from canonical real traces')
            lookup={(i['kernel']['correlation_id'],i['kernel']['nsys_start_ns']):i for i in ls}
            if len(lookup)!=len(ls):raise ValueError('Ambiguous actual light launch identity')
            obs=mrow['worker_response'].get('semantic_forward_observations',{})
            operators={r['id']:r for r in obs.get('affine_operator_observations',{}).get('records',[])}
            semantics={r['id']:r for r in obs.get('records',[])}
            configs={}; measurements=[]; details=[]; local=Counter()
            for group in groups:
                if not request['exact_work_matched'] or group['status']!='EXACT_OBSERVED_SIGNATURE_ORDER_MATCH':
                    rejected.append(dict(alias=alias,request_id=lrow['request_id'],semantic_group=group['semantic_group'],reason='UNKNOWN_GROUP_MAPPING'))
                    continue
                for pair in group['mapped_operator_kernels']:
                    opid=pair['operator_id']; item=lookup[(pair['light_kernel']['correlation_id'],pair['light_kernel']['nsys_start_ns'])]
                    kernel=item['kernel']; config_id=f'{opid}:kernel:{local[opid]}'; ordinal=local[opid];local[opid]+=1
                    if item['group']!=group['semantic_group']:raise ValueError('Semantic group launch mismatch')
                    duration=kernel['nsys_end_ns']-kernel['nsys_start_ns']
                    if pair['light_kernel_duration_ns']!=duration:raise ValueError('Actual unscaled CUPTI duration mismatch')
                    try:
                        config=explicit_configuration(operators[opid],semantics[group['semantic_group']],
                            context['residents'][alias],ordinal,item['signature'],request['mapping'],mr['raw'],lr['context'],context)
                    except (KeyError,ValueError) as exc:
                        rejected.append(dict(alias=alias,request_id=lrow['request_id'],operator_id=opid,reason=str(exc)));continue
                    configs[config_id]=config
                    measurements.append(dict(id=f'{run}:{alias}:{lrow["request_id"]}:{kernel["correlation_id"]}:{kernel["nsys_start_ns"]}',
                                             config_id=config_id,duration_ns=duration))
                    details.append(dict(config_id=config_id,operator_id=opid,semantic_group=group['semantic_group'],
                        launch={k:kernel[k] for k in ('pid','correlation_id','nsys_start_ns','nsys_end_ns','context_id')},signature=item['signature']))
            binding_path=output/('bindings-'+lrow['request_id'].replace(':','-')+'.json')
            save(binding_path,dict(schema='gsdeploy.real_affine_kernel_bindings.v3',metadata_raw=mr['raw'],light_raw=lr['raw'],
                mapping=request['mapping'],context=lr['context'],configurations=configs,launches=details,
                precision_scope='Observed tensor storage and configured quantization; actual accumulator precision not measured',
                weight_selection=WEIGHT_SELECTION))
            binding_source=receipt(binding_path); normalized_configs={}
            for config_id,config in configs.items():
                op=operators[config['observer_record_id']]
                adapted=adapt_affine_feature_bindings(dict(schema='gsdeploy.affine_operator_observations.v2',records=[op]),
                    {op['id']:config},mr['raw'])
                if adapted['rejected_rows']:
                    rejected.extend(adapted['rejected_rows']);continue
                value=adapted['configurations'][op['id']]
                for feature in FEATURES:value['feature_evidence'][feature].append(binding_source)
                normalized_configs[config_id]=value
            manifest=dict(schema=MANIFEST_SCHEMA,run_id=run,split='training',evidence_kind='measured',hardware_id=hardware,
                execution_context_sha256=context_sha,timing_basis='profiled_isolated',
                profiler_id=versions['EXPORT_PRODUCT_NAME']+':'+versions['EXPORT_PRODUCT_VERSION']+':light-kernel',
                profiler_perturbed=True,perturbation_flags=['cuda_cupti_activity','nvtx_semantic_forward_markers','four_resident_owners_solo_requests'],
                isolation_evidence=isolation,measurement_source=lo['sqlite'],configurations=normalized_configs)
            adapted=adapt_raw_kernel_profiles(measurements,manifest);rows.extend(adapted['rows']);rejected.extend(adapted['rejected_rows'])
            coverage.append(dict(alias=alias,request_id=lrow['request_id'],total_light_kernel_count=len(ls),
                accepted_affine_kernel_count=len(adapted['rows']),uncovered_kernel_count=len(ls)-len(adapted['rows']),
                bindings=binding_source,coverage_scope='recognized_leaf_affine_only' if operators else 'UNKNOWN_OUTSIDE_LANGUAGE_OBSERVER'))
    row_path=output/'observations.jsonl'
    with row_path.open('x') as stream:
        for row in rows:stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    result=dict(schema='gsdeploy.real_kernel_baseline_intake.v3',bridge=bridge_source,light_review=bridge['light_review'],
        rows=receipt(row_path),capture_run_id=run,execution_context_sha256=context_sha,coverage=coverage,
        accepted_row_count=len(rows),rejected=rejected,peak_target_observation_count=0,
        unobserved_peak_targets='UNKNOWN_NO_PER_KERNEL_PEAK_MEASUREMENT',independent_capture_count=1,
        GPU_workload_executed=False,natural_timing_validated=False,deployment_qualified=False,
        sources=[receipt(__file__),receipt(Path(__file__).parents[2]/'gsdeploy/forecasting/kernel_baseline.py')])
    save(output/'intake.json',result)
    return result


def _verify(source):
    if not check_receipt(source):raise ValueError('Changed source receipt: '+str(source))


def json_ready(value):
    """Convert nested NumPy fit diagnostics without changing numeric values."""
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:json_ready(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_ready(v) for v in value]
    return value


def fit_intakes(intake_paths, output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    rows=[];runs=set();sources=[]
    for path in intake_paths:
        source=receipt(path);doc=checked(source);sources.append(source)
        if doc.get('schema')!='gsdeploy.real_kernel_baseline_intake.v3':raise ValueError('Real intake required')
        for implementation in doc.get('sources',[]):_verify(implementation)
        lr,lp,lraw=load_report(doc['light_review']['path'],'light-kernel')
        _verify(doc['light_review'])
        actual_run,actual_context=capture_identity(lr,lp,lraw,checked(lr['context']))
        if (doc['capture_run_id'],doc['execution_context_sha256'])!=(actual_run,actual_context):
            raise ValueError('Intake cannot invent capture/context identity')
        if doc['capture_run_id'] in runs:raise ValueError('Repeated capture cannot create independent runs')
        runs.add(doc['capture_run_id']);_verify(doc['rows'])
        for line in Path(doc['rows']['path']).read_text().splitlines():
            row=json.loads(line)
            if row['run_id']!=doc['capture_run_id'] or row['execution_context_sha256']!=doc['execution_context_sha256']:
                raise ValueError('Observation cannot change capture/context identity')
            rows.append(row)
    fit=KernelBaselineEstimator(rows,min_observations=3,min_runs=2)
    groups=[]
    for key,group in fit.groups.items():
        values=np.array([[r['features'][f] for f in FEATURES] for r in group],dtype=float)
        scale=np.ptp(values,axis=0);scale[scale==0]=1
        rank=int(np.linalg.matrix_rank(np.column_stack((np.ones(len(group)),(values-values.mean(axis=0))/scale))))
        models={}
        for target,model in fit.models[key].items():
            models[target]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in model.items()}
        groups.append(dict(group=dict(zip(GROUP_FIELDS,key[:-1])),perturbation_flags=list(key[-1]),
            observation_count=len(group),run_count=len({r['run_id'] for r in group}),
            affine_design_rank=rank,unique_feature_vectors=len({tuple(r['features'][f] for f in FEATURES) for r in group}),models=models))
    known=sum(g['models']['duration_ns']['known'] for g in groups)
    report=dict(schema='gsdeploy.real_kernel_baseline_fit.v3',status='PROFILED_BASELINE_MODELS_AVAILABLE' if known else 'UNKNOWN',
        inputs=sources,training_sha256=fit.training_sha256,observation_count=len(rows),independent_capture_count=len(runs),
        min_observations=3,min_runs=2,group_count=len(groups),known_duration_group_count=known,groups=groups,
        validation_started=False,natural_timing_validated=False,deployment_qualified=False,GPU_workload_executed=False,
        implementation_sources=[receipt(__file__),receipt(Path(__file__).parents[2]/'gsdeploy/forecasting/kernel_baseline.py')],
        interpretation='Training residuals are not confidence bounds. Exact complete contexts and launch settings remain bound; no metadata/light context pooling or natural duration rescaling.')
    report=json_ready(report)
    save(output/'fit.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('intake');p.add_argument('--bridge',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('fit');p.add_argument('--intake',action='append',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    result=intake_bridge(args.bridge,args.output) if args.command=='intake' else fit_intakes(args.intake,args.output)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','accepted_row_count','independent_capture_count','observation_count','group_count','known_duration_group_count')}))


if __name__=='__main__':main()
