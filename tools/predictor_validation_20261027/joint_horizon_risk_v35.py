"""Predeclared equal-weight L2 projection of P(clearance > H), no calibration."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
from pathlib import Path

from . import horizon_recovery_risk_v31 as risk
from . import observed_horizon_risk_sidecar_v32 as sidecar
from .idle_owner_selection_audit_v20 import read
from .owner_lifecycle_mark_v21 import receipt

SCHEMA='gsdeploy.joint_horizon_recovery_risk.v35'
POLICY=dict(method='equal_weight_L2_PAVA_nonincreasing',weights=[1,1,1],
    horizons_ns=list(risk.HORIZONS_NS),predeclared_before_external_validation=True,
    missing_horizon_imputation=False,label_dependent_selection=False)
COMMON_FIELDS=('id','run_id','decision_time_ns','original_execution_context_sha256',
    'visible_state_sha256','visible_waiting_count','risk_domain_sha256')


def pava_nonincreasing(values):
    if len(values)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values):
        raise ValueError('Three finite [0,1] scores required')
    blocks=[]
    for index,value in enumerate(values):
        blocks.append([float(value),1,[index]])
        while len(blocks)>1 and blocks[-2][0]/blocks[-2][1]<blocks[-1][0]/blocks[-1][1]:
            right=blocks.pop();left=blocks.pop()
            blocks.append([left[0]+right[0],left[1]+right[1],left[2]+right[2]])
    result=[None]*3
    for total,n,indices in blocks:
        for index in indices:result[index]=total/n
    return result


def join_predictions(predictions):
    if len(predictions)!=3 or {p['horizon_ns'] for p in predictions}!=set(risk.HORIZONS_NS):
        raise ValueError('Exactly one sealed prediction for each predeclared horizon required')
    ordered=sorted(predictions,key=lambda p:p['horizon_ns']);first=ordered[0]
    for prediction in ordered:
        if prediction.get('schema')!=sidecar.SCHEMA:raise ValueError('Exact v32 observer predictions required')
        value=prediction['risk_score']
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1):
            raise ValueError('Finite raw score in [0,1] or explicit UNKNOWN required')
    values=[p['risk_score'] for p in ordered]
    result=dict(schema=SCHEMA,**{key:first[key] for key in COMMON_FIELDS},policy=copy.deepcopy(POLICY),
        raw_scores=values,raw_statuses=[p['status'] for p in ordered],
        model_sha256_by_horizon={str(p['horizon_ns']):p['model_sha256'] for p in ordered},
        projected_scores=None,raw_monotonicity_violation=None,projection_l2_squared=None,
        calibrated_probability=False,probability_validation_established=False,
        six_term_utility_changed=False,rho_changed=False,model_refit=False,
        physical_predictor_qualified=False,independent_accuracy_established=False)
    if any(any(p[key]!=first[key] for key in COMMON_FIELDS) for p in ordered[1:]):
        return dict(result,status='JOINT_UNKNOWN',reason='INCONSISTENT_ORIGINAL_CONTEXT_DOMAIN_CUTOFF_OR_VISIBLE_PREFIX')
    if all(p['status']=='NOT_AT_RISK_CURRENT_BACKLOG_EMPTY' and p['risk_score'] is None for p in ordered):
        return dict(result,status='JOINT_NOT_AT_RISK_CURRENT_BACKLOG_EMPTY')
    if any(p['status']!='UNCALIBRATED_TRAINING_DIAGNOSTIC' or p['risk_score'] is None for p in ordered):
        return dict(result,status='JOINT_UNKNOWN',reason='ALL_THREE_HORIZONS_MUST_HAVE_SUPPORTED_RAW_SCORES')
    if type(first['visible_waiting_count']) is not int or first['visible_waiting_count']<=0:
        raise ValueError('Supported joint risk requires positive observed backlog')
    projected=pava_nonincreasing(values)
    return dict(result,status='UNCALIBRATED_JOINT_HORIZON_DIAGNOSTIC',projected_scores=projected,
        raw_monotonicity_violation=max([0.]+[values[j]-values[i] for i in range(3) for j in range(i+1,3)]),
        projection_l2_squared=sum((a-b)**2 for a,b in zip(values,projected)))


def load_frozen_predictions(path):
    report=read(path)
    if report.get('schema')!=sidecar.SCHEMA or report.get('status')!='SEALED_OBSERVER_RISK_OUTPUT_NO_LABELS_READ' or report.get('future_labels_read') is not False or report.get('model_refit') is not False:
        raise ValueError('Sealed observer prediction-before-label report required')
    if report['source_receipts_before']!=report['source_receipts_after'] or report['complete_static_local_source_closure']!=sidecar.static_source_closure():
        raise ValueError('Frozen observer source closure differs')
    bindings=report['frozen_bindings']
    if set(bindings)!={'model','risk_domain','visible_report','features'}:raise ValueError('Exact frozen observer bindings required')
    refs=[receipt(path),report['predictions'],*report['source_receipts_before']]
    if any(ref not in report['source_receipts_before'] for ref in bindings.values()):raise ValueError('Frozen binding absent from source receipts')
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Frozen prediction source changed')
    model=read(bindings['model']['path']);domain=read(bindings['risk_domain']['path'])
    provider=sidecar.ObservedHorizonRisk(model,domain)
    expected={}
    with gzip.open(bindings['features']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            envelope=json.loads(line)
            if envelope['id'] in expected:raise ValueError('Duplicate observer prefix')
            expected[envelope['id']]=provider.predict_prefix(envelope,cutoff_ns=envelope['decision_time_ns'])
    predictions={}
    with gzip.open(report['predictions']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            prediction=json.loads(line);rid=prediction['id']
            if rid in predictions or prediction!=expected.get(rid):raise ValueError('Prediction differs from exact frozen observer/model replay')
            predictions[rid]=prediction
    if set(expected)!=set(predictions):raise ValueError('Frozen observation/prediction population differs')
    if [receipt(ref['path']) for ref in refs]!=refs:raise ValueError('Frozen prediction source changed during replay')
    return report,model,predictions,refs


def seal(prediction_reports,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if len(prediction_reports)!=3:raise ValueError('Three independently frozen horizon reports required')
    loaded=[load_frozen_predictions(path) for path in prediction_reports]
    if {model['horizon_ns'] for report,model,rows,refs in loaded}!=set(risk.HORIZONS_NS):raise ValueError('Exact three horizon models required')
    first=loaded[0][0]['frozen_bindings']
    for report,model,rows,refs in loaded:
        if any(report['frozen_bindings'][key]!=first[key] for key in ('risk_domain','features','visible_report')):
            raise ValueError('Joint frozen models must share exact domain and observer source')
        if set(rows)!=set(loaded[0][2]):raise ValueError('Three horizon cohorts differ')
    refs=[receipt(__file__),*[ref for report,model,rows,inputs in loaded for ref in inputs]]
    output.mkdir(parents=True);counts={};violations=0;max_violation=0.;n=0
    with gzip.open(output/'joint-sealed-predictions.jsonl.gz','wt',encoding='utf-8') as stream:
        for rid in sorted(loaded[0][2]):
            row=join_predictions([item[2][rid] for item in loaded]);stream.write(json.dumps(row,separators=(',',':'))+'\n')
            n+=1;counts[row['status']]=counts.get(row['status'],0)+1
            if row['raw_monotonicity_violation'] is not None:
                violations+=row['raw_monotonicity_violation']>0
                max_violation=max(max_violation,row['raw_monotonicity_violation'])
    after=[receipt(ref['path']) for ref in refs]
    if refs!=after:raise ValueError('Joint source changed during sealing')
    report=dict(schema=SCHEMA,status='SEALED_JOINT_HORIZON_PREDICTIONS_NO_LABELS_READ',policy=POLICY,
        prediction_reports=[receipt(path) for path in prediction_reports],frozen_visible_report=first['visible_report'],
        predictions=receipt(output/'joint-sealed-predictions.jsonl.gz'),n=n,status_counts=counts,
        raw_monotonicity_violation_n=violations,maximum_raw_monotonicity_violation=max_violation,
        source_receipts_before=refs,source_receipts_after=after,future_labels_read=False,model_refit=False,
        label_dependent_selection=False,calibrated_probability=False,six_term_utility_changed=False,rho_changed=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def evaluate(joint_report_path,label_report_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    joint=read(joint_report_path);labels_report=read(label_report_path)
    if joint.get('schema')!=SCHEMA or joint.get('status')!='SEALED_JOINT_HORIZON_PREDICTIONS_NO_LABELS_READ' or joint.get('policy')!=POLICY or joint.get('future_labels_read') is not False:
        raise ValueError('Predeclared frozen joint report required')
    if joint['frozen_visible_report']!=receipt(label_report_path):raise ValueError('Joint labels differ from frozen observation source')
    if joint['source_receipts_before']!=joint['source_receipts_after']:raise ValueError('Frozen joint boundary changed')
    loaded=[load_frozen_predictions(ref['path']) for ref in joint['prediction_reports']]
    refs=[receipt(joint_report_path),joint['predictions'],receipt(label_report_path),labels_report['labels'],*joint['source_receipts_before']]
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Frozen joint evaluation source changed')
    labels={}
    with gzip.open(labels_report['labels']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);key=row['id'],row['offline_labels']['horizon_ns']
            if key in labels:raise ValueError('Duplicate original horizon label')
            labels[key]=row
    scores={h:[] for h in risk.HORIZONS_NS};seen=set();joint_unknown=0;not_at_risk=0;unknown_targets={h:0 for h in risk.HORIZONS_NS}
    with gzip.open(joint['predictions']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);rid=row['id']
            if rid in seen or row!=join_predictions([item[2][rid] for item in loaded]):raise ValueError('Joint output differs from frozen fixed-policy replay')
            seen.add(rid)
            if any(row['run_id'] in item[1]['provenance']['run_ids'] for item in loaded):raise ValueError('Joint holdout overlaps model training capture')
            for horizon in risk.HORIZONS_NS:
                label=labels[rid,horizon]
                if label['run_id']!=row['run_id'] or label['execution_context_sha256']!=row['original_execution_context_sha256'] or label['decision_time_ns']!=row['decision_time_ns']:
                    raise ValueError('Joint outcome context/cutoff identity differs')
            if row['status']=='JOINT_NOT_AT_RISK_CURRENT_BACKLOG_EMPTY':not_at_risk+=1;continue
            if row['status']=='JOINT_UNKNOWN':joint_unknown+=1;continue
            for index,horizon in enumerate(risk.HORIZONS_NS):
                label=labels[rid,horizon]
                target=risk.binary_target(dict(cutoff_ns=row['decision_time_ns'],waiting_count=row['visible_waiting_count']),label['offline_labels'])
                if target['label'] is None:unknown_targets[horizon]+=1;continue
                scores[horizon].append((target['label'],row['raw_scores'][index],row['projected_scores'][index]))
    if seen!=set(loaded[0][2]) or {(rid,h) for rid in seen for h in risk.HORIZONS_NS}!=set(labels):raise ValueError('Joint original cohort grid differs')
    metrics={str(h):dict(common_support_n=len(cells),unknown_target_n=unknown_targets[h],
        raw_brier_score=None if not cells else sum((a-y)**2 for y,a,b in cells)/len(cells),
        fixed_projection_brier_score=None if not cells else sum((b-y)**2 for y,a,b in cells)/len(cells)) for h,cells in scores.items()}
    after=[receipt(ref['path']) for ref in refs]
    if refs!=after:raise ValueError('Joint evaluation sources changed')
    report=dict(schema=SCHEMA+'.evaluation',status='RAW_AND_PREDECLARED_PROJECTION_CAPTURE_HOLDOUT_DIAGNOSTIC',policy=POLICY,
        per_horizon=metrics,requested_n=len(seen),joint_unknown_n=joint_unknown,not_at_risk_n=not_at_risk,
        raw_monotonicity_violation_n=joint['raw_monotonicity_violation_n'],maximum_raw_monotonicity_violation=joint['maximum_raw_monotonicity_violation'],
        source_receipts_before=refs,source_receipts_after=after,model_refit=False,label_dependent_selection=False,
        calibrated_probability=False,independent_accuracy_established=False,six_term_utility_changed=False,rho_changed=False)
    output.mkdir(parents=True);(output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    pred=sub.add_parser('seal');pred.add_argument('--prediction-report',action='append',required=True);pred.add_argument('--output',required=True)
    evaluation=sub.add_parser('evaluate');evaluation.add_argument('--joint-report',required=True);evaluation.add_argument('--label-report',required=True);evaluation.add_argument('--output',required=True)
    args=parser.parse_args();result=seal(args.prediction_report,args.output) if args.command=='seal' else evaluate(args.joint_report,args.label_report,args.output)
    print(json.dumps(dict(status=result['status']),indent=2))
