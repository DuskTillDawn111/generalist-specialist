"""Small training-capture endpoint first-passage candidate, never SM service.

Prediction at launch API return can supply future observed START/END deadlines
in a separate event model. This module does not alter a reference-work rate.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .idle_owner_selection_audit_v20 import read
from .owner_lifecycle_mark_v21 import receipt

SCHEMA='gsdeploy.launch_cutoff_endpoint_first_passage_candidate.v27'
SAMPLE_MODULUS=32
FEATURES=('intercept','log_own_started_not_ended','log_other_started_not_ended',
    'kernel_already_started','context_prefix_own','context_prefix_other',
    'context_prefix_unknown','log_context_age_ms','log_launch_API_duration_us')
TARGETS=('START_if_not_started','END_if_not_ended')


def features(prefix):
    # Explicit allowlist: kernel endpoint labels/paired future windows cannot
    # enter the design matrix even if passed in an outer dataset record.
    owner=prefix['owner'];counts=prefix['observed_started_not_ended_kernel_count_by_owner']
    context=prefix['context_event_prefix_at_cutoff'];holder=context['owner']
    known=context['status']=='UNIQUE_OPEN_CONTEXT_PREFIX_NOT_SM_SERVICE'
    age=context['context_age_ns'] if known else 0
    duration=prefix['source_launch_API_return_ns']-prefix['source_launch_API_entry_ns']
    if duration<0 or any(type(count) is not int or count<0 for count in counts.values()):
        raise ValueError('Observed nonnegative API duration and prefix activity counts required')
    return np.asarray([1.,math.log1p(counts.get(owner,0)),
        math.log1p(sum(value for alias,value in counts.items() if alias!=owner)),
        float(prefix['already_observed_kernel_start_at_cutoff']),
        float(known and holder==owner),float(known and holder!=owner),float(not known),
        math.log1p(age/1e6),math.log1p(duration/1e3)])


def signature(prefix):
    return (prefix['owner'],prefix['semantic_phase'],prefix['launch_signature'])


def selected(row):
    key=(row['capture']+':'+row['source_node_id']).encode()
    return int(hashlib.sha256(key).hexdigest()[:8],16)%SAMPLE_MODULUS==0


def target(row,kind):
    prefix=row['prefix_features'];labels=row['observation_labels'];cutoff=prefix['cutoff_ns']
    if kind=='START_if_not_started':
        if prefix['already_observed_kernel_start_at_cutoff']:return None
        value=labels['observed_kernel_start_ns']-cutoff
    elif kind=='END_if_not_ended':
        if prefix['already_observed_kernel_end_at_cutoff']:return None
        value=labels['observed_kernel_end_ns']-cutoff
    else:raise ValueError('Declared endpoint target required')
    if type(value) is not int or value<=0:raise ValueError('Strictly future endpoint first passage required')
    return value


def fit(rows,kind):
    groups=defaultdict(list)
    usable=[row for row in rows if target(row,kind) is not None]
    for row in usable:
        groups[signature(row['prefix_features'])].append(math.log1p(target(row,kind)))
    base={key:float(np.mean(values)) for key,values in groups.items()}
    matrix=np.asarray([features(row['prefix_features']) for row in usable])
    labels=np.asarray([math.log1p(target(row,kind))-base[signature(row['prefix_features'])] for row in usable])
    center=matrix.mean(axis=0);scale=matrix.std(axis=0);center[0]=0.;scale[0]=1.;scale[scale<1e-12]=1.
    design=(matrix-center)/scale
    # Fixed normalized least-squares plus L2/N; equivalently X'X + I.
    penalty=np.eye(len(FEATURES));penalty[0,0]=0.
    coefficients=np.linalg.solve(design.T@design+penalty,design.T@labels)
    return dict(target=kind,signature_mean_log1p_ns=[dict(key=list(key),value=value,n=len(groups[key])) for key,value in sorted(base.items())],
        feature_names=list(FEATURES),center=center.tolist(),scale=scale.tolist(),coefficients=coefficients.tolist(),
        feature_minimum=matrix.min(axis=0).tolist(),feature_maximum=matrix.max(axis=0).tolist(),
        training_eligible_n=len(usable),regularization='normalized squared log residual loss plus L2/N, intercept unpenalized',
        fallback_on_unseen_signature=False)


def predict(model,prefix):
    base={tuple(row['key']):row['value'] for row in model['signature_mean_log1p_ns']}
    key=signature(prefix)
    if key not in base:return dict(status='UNKNOWN_LAUNCH_SIGNATURE',predicted_remaining_wall_ns=None)
    row=features(prefix)
    centered=(row-np.asarray(model['center']))/np.asarray(model['scale'])
    adjustment=float(centered@np.asarray(model['coefficients']))
    flags=[model['feature_names'][i] for i in np.flatnonzero((row<np.asarray(model['feature_minimum']))|(row>np.asarray(model['feature_maximum'])))]
    return dict(status='UNCALIBRATED_OBSERVED_ENDPOINT_CANDIDATE',
        predicted_remaining_wall_ns=max(1.,math.expm1(base[key]+adjustment)),
        signature_baseline_remaining_wall_ns=max(1.,math.expm1(base[key])),
        feature_extrapolation=flags,physical_SM_service=False,reference_work_rate=False)


def stats(actual,predicted):
    if not len(actual):return None
    actual,predicted=np.asarray(actual),np.asarray(predicted)
    absolute=np.abs(predicted-actual)
    return dict(n=len(actual),mean_absolute_error_us=float(absolute.mean()/1e3),
        median_absolute_error_us=float(np.median(absolute)/1e3),
        p95_absolute_error_us=float(np.percentile(absolute,95)/1e3),
        median_absolute_relative_error=float(np.median(absolute/actual)),
        mean_absolute_log1p_error=float(np.abs(np.log1p(predicted)-np.log1p(actual)).mean()),
        predicted_sum_ms=float(predicted.sum()/1e6),observed_sum_ms=float(actual.sum()/1e6),
        sums_overlap_across_kernels=True,sums_are_not_request_latency_or_GPU_work=True)


def evaluate(model,rows):
    actual,predicted,baseline=[],[],[];unknown=Counter();strata=defaultdict(lambda:[[],[],[]])
    for row in rows:
        value=target(row,model['target'])
        if value is None:continue
        prefix=row['prefix_features'];estimate=predict(model,prefix)
        if estimate['predicted_remaining_wall_ns'] is None:
            unknown[estimate['status']]+=1;continue
        actual.append(value);predicted.append(estimate['predicted_remaining_wall_ns']);baseline.append(estimate['signature_baseline_remaining_wall_ns'])
        context=prefix['context_event_prefix_at_cutoff']
        relation=('own' if context['owner']==prefix['owner'] else 'other' if context['owner'] else 'unknown')
        for key in ('owner='+prefix['owner'],'phase='+prefix['semantic_phase'],'context='+relation):
            arrays=strata[key];arrays[0].append(value);arrays[1].append(predicted[-1]);arrays[2].append(baseline[-1])
    return dict(requested_eligible=sum(target(row,model['target']) is not None for row in rows),
        unknown_counts=dict(unknown),candidate=stats(actual,predicted),signature_baseline=stats(actual,baseline),
        strata=[dict(stratum=key,candidate=stats(values[0],values[1]),signature_baseline=stats(values[0],values[2])) for key,values in sorted(strata.items())])


def run(evidence_report,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    source=receipt(evidence_report);report=read(evidence_report)
    if report.get('schema')!='gsdeploy.physical_progress_endpoint_evidence.v26' or report.get('training_only') is not True:
        raise ValueError('Training-only v26 endpoint source required')
    refs=[source,report['kernel_rows'],report['launch_vocabulary'],receipt(__file__)]
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Endpoint source changed')
    rows=[];counts=Counter()
    with gzip.open(report['kernel_rows']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);counts['source_rows']+=1
            if selected(row):
                rows.append(row);counts['selected_rows']+=1
                counts['selected_already_STARTED']+=row['prefix_features']['already_observed_kernel_start_at_cutoff']
                counts['selected_already_ENDED']+=row['prefix_features']['already_observed_kernel_end_at_cutoff']
    captures=sorted({row['capture'] for row in rows})
    folds=[]
    for heldout in captures:
        train=[row for row in rows if row['capture']!=heldout];test=[row for row in rows if row['capture']==heldout]
        for kind in TARGETS:
            model=fit(train,kind)
            folds.append(dict(heldout_training_capture=heldout,target=kind,
                training_eligible_n=model['training_eligible_n'],**evaluate(model,test)))
    models=[fit(rows,kind) for kind in TARGETS]
    after=[receipt(ref['path']) for ref in refs]
    if refs!=after:raise ValueError('Endpoint fit source bytes changed')
    output.mkdir(parents=True)
    model=dict(schema=SCHEMA,training_only=True,models=models,implementation=receipt(__file__),sources=refs,
        sample_rule=f'first32bits(SHA256(capture:source_node_id)) mod {SAMPLE_MODULUS} == 0',
        target_domain='Observed future CUDA START/END elapsed wall time at launch API return; includes wait/preemption and unclassified intervals',
        physical_service_fit=False,reference_work_rate=False)
    (output/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    result=dict(schema=SCHEMA,status='SMALL_TRAINING_CAPTURE_HOLDOUT_ENDPOINT_DIAGNOSTIC',counts=dict(counts),
        folds=folds,model=receipt(output/'model.json'),source_receipts_before=refs,source_receipts_after=after,
        event_prefix_clock_is_runtime_observer_available=False,
        limits=['Only two training captures sharing request/input templates; hundreds of thousands of kernels are not independent samples.',
            'CUPTI event timestamp prefixes are reconstructed offline; asynchronous observer delivery latency is not identified.',
            'Launch-cutoff predictions do not update conditional residual survival at later competing events.',
            'An endpoint residual is not continuous SM service or a drop-in reference-work rate.',
            'Static launch descriptor availability before observed kernel completion requires an execution-backend observation contract.'],
        GPU_started=False,validation_labels_used=False,physical_service_fit=False,independent_accuracy_validated=False,
        integrated_into_forecast=False,final_gain_eligible=False)
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence-report',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.evidence_report,args.output)
    print(json.dumps(dict(counts=result['counts'],folds=[{key:value for key,value in row.items() if key!='strata'} for row in result['folds']]),indent=2))
