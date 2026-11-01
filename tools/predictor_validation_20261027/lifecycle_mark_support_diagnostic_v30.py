"""Read-only training-support diagnosis of one frozen simulated UNKNOWN state."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from . import owner_lifecycle_mark_v21 as mark
from .profiling import digest
from .unknown_preflight_archive_v1 import save_synced


def run(model_path, supervision_path, forecast_report_path, output):
    sources=[mark.receipt(p) for p in (model_path,supervision_path,forecast_report_path,__file__,mark.__file__)]
    model=json.loads(Path(model_path).read_text());supervision=json.loads(Path(supervision_path).read_text())
    report=json.loads(Path(forecast_report_path).read_text())
    if model['sources'][0]!=sources[1] or model.get('training_only') is not True:
        raise ValueError('Model-bound training supervision required')
    if supervision.get('training_only') is not True or supervision.get('validation_labels_used') is not False:
        raise ValueError('Training-only labels required')
    if model['model_sha256']!=digest({k:v for k,v in model.items() if k!='model_sha256'}):
        raise ValueError('Frozen model digest changed')
    ref=report['result']['artifact']
    if mark.receipt(ref['path'])!=ref:raise ValueError('Forecast compressed receipt changed')
    with gzip.open(ref['path'],'rb') as stream:payload=stream.read()
    if hashlib.sha256(payload).hexdigest()!=report['result']['uncompressed_sha256'] or len(payload)!=report['result']['uncompressed_bytes']:
        raise ValueError('Forecast decompressed bytes changed')
    forecast=json.loads(payload);scheduler=forecast['scheduler_diagnostic'];oracle=scheduler['oracle']
    if forecast.get('reason')!='UNSUPPORTED_SAME_CONTEXT_REACQUISITION_PREDICTION':
        raise ValueError('Expected frozen same-holder simulation failure required')
    hazard=oracle['latest_cumulative_hazard']['state'];clock=oracle['current_lifecycle_clock_ns']
    if hazard['last_update_ns']!=clock or hazard['lifecycle']!=oracle['current_lifecycle']:
        raise ValueError('Same-cutoff lifecycle and hazard snapshot required')
    states={owner:mark.OwnerLifecycle(**state) for owner,state in oracle['current_lifecycle'].items()}
    query=dict(held_owner=scheduler['held_owner'],context_age_ns=clock-scheduler['restore_start_ns'],
        pending_frontier=hazard['frontier'],previous_held_owner=oracle['decisions'][-1]['held_owner'],
        public_workload=None,pending_age_ns_by_owner=None,pending_reference_work_ns_by_owner=None)
    predicted=mark.probabilities(model,query,states)
    if predicted['active_service_target']!=query['held_owner']:raise ValueError('Same-holder prediction not reproduced')
    rows=supervision['transition_observations'];n=len(rows);fitted=model['fitted']
    target=Counter(row['next_owner'] for row in rows);held=Counter(row['held_owner'] for row in rows)
    if dict(target)!=fitted['label_counts']:raise ValueError('Supervision and model labels differ')
    binary={}
    for i,name in enumerate(model['feature_names']):
        lo,hi=fitted['feature_minimum'][i],fitted['feature_maximum'][i]
        if name!='intercept' and lo in (0.,1.) and hi in (0.,1.) and lo<=hi:
            binary[name]=dict(zero_count=n-round(fitted['center'][i]*n),one_count=round(fitted['center'][i]*n),
                minimum=lo,maximum=hi)
    ocr_rows=[{k:row[k] for k in ('capture','job_id','held_owner','current_restore_start_ns','save_end_ns',
        'context_age_at_save_ns','pending_frontier','next_owner','next_owner_is_pending')}
        for row in rows if row['held_owner']=='ocr#0' or row['next_owner']=='ocr#0']
    by_capture={capture:dict(rows=sum(r['capture']==capture for r in rows),
        target_counts=dict(Counter(r['next_owner'] for r in rows if r['capture']==capture))) for capture in sorted({r['capture'] for r in rows})}
    result=dict(schema='gsdeploy.lifecycle_mark_training_support_diagnostic.v30',status='SPARSE_OCR_AND_UNSEEN_JOINT_STATE_CONFIRMED',
        sources=sources+[ref],training_transitions=n,held_owner_counts=dict(held),target_owner_counts=dict(target),
        same_holder_transition_counts=dict(Counter(r['held_owner'] for r in rows if r['held_owner']==r['next_owner'])),
        by_capture=by_capture,ocr_related_rows=ocr_rows,binary_feature_counts=binary,
        binary_count_method='Frozen all-training fitted unweighted feature center times exact training row count; rounded integer.',
        simulated_failure_clock_ns=clock,simulated_failure_query=query,simulated_failure_lifecycle=oracle['current_lifecycle'],
        deterministic_mark_recomputation=predicted,
        interpretation='OCR has two target labels and two held rows, not zero labels. All OCR pending/ready features are zero in training. Held OCR training rows have empty frontiers at roughly two-second ages; the failed 77,768 ns held-OCR state has two ready text peers and an unseen joint signature. Uncalibrated extrapolation does not establish a supported same-holder transition.',
        guard_removed=False,owner_forced=False,fit_performed=False,validation_labels_used=False,
        forecast_state_is_simulated_not_measured_validation=True,physical_SM_service_identified=False,GPU_started=False)
    if [mark.receipt(s['path']) for s in sources+[ref]]!=sources+[ref]:raise ValueError('Diagnostic source changed')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);save_synced(out/'report.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','supervision','forecast-report','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();r=run(args.model,args.supervision,args.forecast_report,args.output)
    print(json.dumps({k:r[k] for k in ('status','training_transitions','target_owner_counts','simulated_failure_clock_ns')}))
