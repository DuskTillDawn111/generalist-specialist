"""Independent receipt-causal replay: features -> seal scores -> export labels.

This is a fresh capture validation, not online prediction or a DAG checkpoint.
Prediction opens only the features, readiness qualification and frozen models.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from . import horizon_recovery_risk_v31 as risk
from . import joint_horizon_risk_v35 as joint
from . import observed_horizon_risk_sidecar_v32 as sidecar
from . import ready_risk_holdout_v38 as collector
from . import whole57_visible_risk_adapter_v30 as visible
from .idle_owner_selection_audit_v20 import read
from .owner_lifecycle_mark_v21 import receipt

SCHEMA='gsdeploy.independent_observer_risk_validation.v39'


def checked(ref):
    if receipt(ref['path'])!=ref:raise ValueError('Frozen independent risk source changed')
    return read(ref['path'])


def review(run):
    run=Path(run).resolve();collector.audit(run);raw=read(run/'raw.json');plan=read(run/'plan.json')
    execution=read(run/'execution.json');qualification=read(run/'readiness-only/qualification.json')
    if (raw.get('schema')!=collector.RAW_SCHEMA or raw.get('split')!='risk_validation' or raw.get('training_only') is not False
            or raw.get('status')!='completed' or raw.get('errors') or raw.get('runtime_errors') or raw.get('telemetry_errors')
            or raw.get('final_source_audit')!='PASS' or raw.get('execution_context_sha256')!=plan['execution_context_sha256']):
        raise ValueError('Complete honest independent raw/source/runtime PASS required')
    if execution.get('status')!='COLLECTED_RISK_HOLDOUT_REQUIRES_INDEPENDENT_REVIEW' or execution.get('final_source_audit')!='PASS':
        raise ValueError('Original exclusive watchdog execution must pass')
    if len(raw.get('request_records',[]))!=57 or len(raw.get('priming_request_records',[]))!=12 or any(row.get('status')!='completed' for row in raw['request_records']+raw['priming_request_records']):
        raise ValueError('Complete 57+12 independent cohort required')
    if [r['request_id'] for r in raw['request_records']]!=[r['request_id'] for r in plan['trace']]:raise ValueError('Independent original cohort identities differ')
    if qualification.get('status')!='ACTUAL_READY_EXACT_PASS_MODELS_BOUND_BEFORE_PRIMING' or qualification.get('fit_called') is not False:
        raise ValueError('READY-only qualification before priming required')
    ready=checked(qualification['ready'])
    if ready['ready']!=raw['ready'] or ready['execution_context_sha256']!=plan['execution_context_sha256']:raise ValueError('Actual READY payload changed after qualification')
    return raw,plan,qualification


def export_features(run,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    run=Path(run).resolve();raw,plan,qualification=review(run)
    refs=[receipt(__file__),receipt(run/'raw.json'),receipt(run/'plan.json'),receipt(run/'readiness-only/qualification.json'),*collector.static_source_closure(__file__)]
    begin=raw['phase_start_ns'];end=raw['observation_end_ns'];context=plan['execution_context_sha256'];records=raw['request_records']
    feedback=visible.observed_state.feedback_from_runtime_records(records,context,end)
    registry=visible.public_registry(plan,begin)
    cutoffs=sorted({begin,end,*range(begin,end+1,1_000_000_000),*[r['arrival_ns'] for r in registry if r['arrival_ns']<=end],*[row['received_time_ns'] for row in feedback]})
    state=visible.observed_state.new_observed_state(plan['resource_configuration']['allocations'],context)
    run_id=plan['training_id']+':'+refs[1]['sha256'];output.mkdir(parents=True);n=0
    with gzip.open(output/'prefix-features.jsonl.gz','wt',encoding='utf-8') as stream:
        for cutoff in cutoffs:
            events=[row for row in feedback if row['sequence']>state['last_feedback_sequence'] and row['received_time_ns']<=cutoff]
            state=visible.observed_state.advance_observed_state(state,cutoff,feedback=events,arrivals=visible.arrivals_through(registry,cutoff,state['last_arrival_sequence']))
            prefix=visible.prefix_features(state,registry,cutoff=cutoff,declared_at=begin,
                metadata=visible.received_phase_metadata(records,cutoff),host_pipeline=visible.visible_host_pipeline(records,cutoff))
            row=dict(id=run_id+':'+str(cutoff-begin),run_id=run_id,group_id='independent_same_recipe_content',
                split='independent_risk_validation',kind='measured',domain=plan['domain'],route_id=plan['route_id'],
                execution_context_sha256=context,decision_time_ns=cutoff,prefix_features=prefix)
            stream.write(json.dumps(row,separators=(',',':'))+'\n');n+=1
    after=[receipt(ref['path']) for ref in refs]
    if after!=refs:raise ValueError('Independent source changed during feature export')
    report=dict(schema=SCHEMA+'.features',status='INDEPENDENT_READER_PREFIXES_ONLY_NO_LABELS_EXPORTED',features=receipt(output/'prefix-features.jsonl.gz'),
        readiness_qualification=receipt(run/'readiness-only/qualification.json'),raw_source=receipt(run/'raw.json'),plan=receipt(run/'plan.json'),
        n=n,run_id=run_id,source_receipts_before=refs,source_receipts_after=after,
        independent_validation=True,training_only=False,offline_labels_exported=False,model_refit=False,
        causal_receipt_replay=True,online_prediction_claimed=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def seal(feature_report_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    features=read(feature_report_path)
    if features.get('schema')!=SCHEMA+'.features' or features.get('status')!='INDEPENDENT_READER_PREFIXES_ONLY_NO_LABELS_EXPORTED' or features.get('training_only') is not False:
        raise ValueError('Independent features-only evidence required')
    # Do not open raw_source or plan: only current feature receipts and READY
    # qualification/models/domain are accessed during frozen scoring.
    qualification=checked(features['readiness_qualification']);domain=checked(qualification['expanded_domain'])
    if qualification.get('status')!='ACTUAL_READY_EXACT_PASS_MODELS_BOUND_BEFORE_PRIMING' or qualification.get('fit_called') is not False or qualification.get('numeric_model_parameters_changed') is not False:
        raise ValueError('Exact READY-qualified metadata-only model binding required')
    refs=[receipt(__file__),receipt(feature_report_path),features['features'],features['readiness_qualification'],qualification['expanded_domain'],*collector.static_source_closure(__file__)]
    providers=[]
    for item in qualification['model_bindings']:
        model=checked(item['rebound']);original=checked(item['original'])
        if collector.numeric_model(model)!=collector.numeric_model(original):raise ValueError('Independent model numeric parameters changed')
        providers.append(sidecar.ObservedHorizonRisk(model,domain));refs.extend([item['rebound'],item['original']])
    if len(providers)!=3 or {provider.model['horizon_ns'] for provider in providers}!=set(risk.HORIZONS_NS):raise ValueError('Exactly three frozen models required')
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Frozen prediction source mismatch')
    output.mkdir(parents=True);n=0;counts={};violations=0;maximum=0.
    with gzip.open(features['features']['path'],'rt',encoding='utf-8') as stream,gzip.open(output/'sealed-predictions.jsonl.gz','wt',encoding='utf-8') as dest:
        for line in stream:
            row=json.loads(line)
            if row['split']!='independent_risk_validation':raise ValueError('Holdout cannot be a training capture')
            if any(row['run_id'] in provider.model['provenance']['run_ids'] for provider in providers):raise ValueError('Independent capture overlaps model training')
            singles=[provider.predict_prefix(row,cutoff_ns=row['decision_time_ns']) for provider in providers]
            combined=joint.join_predictions(singles)
            dest.write(json.dumps(dict(id=row['id'],single_horizon=singles,joint_horizon=combined),separators=(',',':'))+'\n')
            n+=1;counts[combined['status']]=counts.get(combined['status'],0)+1
            if combined['raw_monotonicity_violation'] is not None:
                violations+=combined['raw_monotonicity_violation']>0;maximum=max(maximum,combined['raw_monotonicity_violation'])
    after=[receipt(ref['path']) for ref in refs]
    if after!=refs:raise ValueError('Independent frozen score sources changed')
    report=dict(schema=SCHEMA+'.sealed',status='INDEPENDENT_SCORES_SEALED_BEFORE_LABEL_EXPORT',feature_report=receipt(feature_report_path),
        predictions=receipt(output/'sealed-predictions.jsonl.gz'),n=n,joint_status_counts=counts,
        raw_monotonicity_violation_n=violations,maximum_raw_monotonicity_violation=maximum,joint_policy=joint.POLICY,
        source_receipts_before=refs,source_receipts_after=after,raw_service_queue_or_labels_opened=False,
        complete_static_local_source_closure=collector.static_source_closure(__file__),
        model_refit=False,independent_validation=True,training_only=False,probability_calibrated=False,
        six_term_utility_changed=False,rho_changed=False,causal_receipt_replay=True,online_prediction_claimed=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def export_labels(sealed_report_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    sealed=read(sealed_report_path)
    if sealed.get('schema')!=SCHEMA+'.sealed' or sealed.get('status')!='INDEPENDENT_SCORES_SEALED_BEFORE_LABEL_EXPORT':raise ValueError('Scores must already be frozen')
    features=checked(sealed['feature_report']);raw=checked(features['raw_source']);plan=checked(features['plan'])
    refs=[receipt(__file__),receipt(sealed_report_path),sealed['predictions'],sealed['feature_report'],features['features'],features['raw_source'],features['plan']]
    labels=visible.OfflineQueueLabels(visible.public_registry(plan,raw['phase_start_ns']),raw['request_records'],raw['observation_end_ns'])
    output.mkdir(parents=True);n=0
    with gzip.open(features['features']['path'],'rt',encoding='utf-8') as stream,gzip.open(output/'offline-labels.jsonl.gz','wt',encoding='utf-8') as dest:
        for line in stream:
            row=json.loads(line)
            for horizon in risk.HORIZONS_NS:
                label=dict(id=row['id'],run_id=row['run_id'],execution_context_sha256=row['execution_context_sha256'],decision_time_ns=row['decision_time_ns'],offline_labels=labels.outcome(row['decision_time_ns'],horizon))
                dest.write(json.dumps(label,separators=(',',':'))+'\n');n+=1
    after=[receipt(ref['path']) for ref in refs]
    if refs!=after:raise ValueError('Frozen score/raw evidence changed during later label export')
    report=dict(schema=SCHEMA+'.labels',status='OUTCOMES_EXPORTED_AFTER_INDEPENDENT_SCORES_SEALED',sealed_report=receipt(sealed_report_path),
        features_report=sealed['feature_report'],labels=receipt(output/'offline-labels.jsonl.gz'),n=n,source_receipts_before=refs,source_receipts_after=after,model_refit=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def evaluate(sealed_report_path,label_report_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    sealed=read(sealed_report_path);labels_report=read(label_report_path);features=checked(sealed['feature_report'])
    if (sealed.get('schema')!=SCHEMA+'.sealed' or sealed.get('status')!='INDEPENDENT_SCORES_SEALED_BEFORE_LABEL_EXPORT'
            or sealed.get('raw_service_queue_or_labels_opened') is not False or sealed.get('model_refit') is not False or sealed.get('joint_policy')!=joint.POLICY):
        raise ValueError('Strict sealed independent prediction-before-label contract required')
    if labels_report.get('schema')!=SCHEMA+'.labels' or labels_report.get('status')!='OUTCOMES_EXPORTED_AFTER_INDEPENDENT_SCORES_SEALED' or labels_report['sealed_report']!=receipt(sealed_report_path) or labels_report['features_report']!=sealed['feature_report']:
        raise ValueError('Separate later labels must bind exact earlier sealed scores')
    refs=[receipt(__file__),receipt(sealed_report_path),sealed['predictions'],receipt(label_report_path),labels_report['labels'],sealed['feature_report'],features['features'],*sealed['source_receipts_before']]
    if sealed['source_receipts_before']!=sealed['source_receipts_after']:raise ValueError('Frozen scoring source changed')
    if sealed['complete_static_local_source_closure']!=collector.static_source_closure(__file__):raise ValueError('Independent full source closure changed')
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Independent evaluated source changed')
    qualification=checked(features['readiness_qualification']);domain=checked(qualification['expanded_domain'])
    providers=[sidecar.ObservedHorizonRisk(checked(item['rebound']),domain) for item in qualification['model_bindings']]
    prefixes={}
    with gzip.open(features['features']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['id'] in prefixes:raise ValueError('Duplicate original prefix')
            prefixes[row['id']]=row
    labels={}
    with gzip.open(labels_report['labels']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);key=row['id'],row['offline_labels']['horizon_ns']
            if key in labels:raise ValueError('Duplicate horizon outcome')
            labels[key]=row
    singles={h:[] for h in risk.HORIZONS_NS};common={h:[] for h in risk.HORIZONS_NS};unknown={h:0 for h in risk.HORIZONS_NS};seen=set();empty=0;joint_unknown=0
    with gzip.open(sealed['predictions']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            pred=json.loads(line);rid=pred['id'];prefix=prefixes[rid]
            if rid in seen:raise ValueError('Duplicate sealed independent prediction')
            seen.add(rid)
            if prefix['split']!='independent_risk_validation' or any(prefix['run_id'] in provider.model['provenance']['run_ids'] for provider in providers):raise ValueError('Independent holdout overlaps training')
            expected=[provider.predict_prefix(prefix,cutoff_ns=prefix['decision_time_ns']) for provider in providers]
            if pred['single_horizon']!=expected or pred['joint_horizon']!=joint.join_predictions(expected):raise ValueError('Sealed independent scores differ from frozen source/model replay')
            combined=pred['joint_horizon']
            if combined['status']=='JOINT_NOT_AT_RISK_CURRENT_BACKLOG_EMPTY':empty+=1
            elif combined['status']=='JOINT_UNKNOWN':joint_unknown+=1
            by_h={row['horizon_ns']:row for row in expected}
            for index,h in enumerate(risk.HORIZONS_NS):
                label=labels[rid,h]
                if label['run_id']!=prefix['run_id'] or label['execution_context_sha256']!=prefix['execution_context_sha256'] or label['decision_time_ns']!=prefix['decision_time_ns']:raise ValueError('Original outcome context/cutoff mismatch')
                target=risk.binary_target(prefix['prefix_features'],label['offline_labels']);score=by_h[h]['risk_score']
                if target['label'] is None or score is None:unknown[h]+=1;continue
                singles[h].append((target['label'],score))
                if combined['projected_scores'] is not None:common[h].append((target['label'],score,combined['projected_scores'][index]))
    if seen!=set(prefixes) or {(rid,h) for rid in seen for h in risk.HORIZONS_NS}!=set(labels):raise ValueError('Independent whole cohort feature/score/outcome grid differs')
    metrics={str(h):dict(single_scored_n=len(singles[h]),unscored_or_target_unknown_n=unknown[h],
        single_brier=None if not singles[h] else sum((p-y)**2 for y,p in singles[h])/len(singles[h]),
        joint_common_support_n=len(common[h]),raw_common_brier=None if not common[h] else sum((p-y)**2 for y,p,z in common[h])/len(common[h]),
        fixed_projection_common_brier=None if not common[h] else sum((z-y)**2 for y,p,z in common[h])/len(common[h])) for h in risk.HORIZONS_NS}
    def diagnostics(cells):
        positives=[p for y,p in cells if y];negatives=[p for y,p in cells if not y]
        bins=[]
        for index in range(10):
            selected=[(y,p) for y,p in cells if min(9,int(p*10))==index]
            bins.append(dict(lower=index/10,upper=(index+1)/10,n=len(selected),
                mean_score=None if not selected else sum(p for y,p in selected)/len(selected),
                observed_frequency=None if not selected else sum(y for y,p in selected)/len(selected)))
        return dict(rank_auc=None if not positives or not negatives else sum((p>n)+.5*(p==n) for p in positives for n in negatives)/(len(positives)*len(negatives)),reliability_bins=bins)
    for h in risk.HORIZONS_NS:
        model=next(provider.model for provider in providers if provider.model['horizon_ns']==h)
        counts=model['training_class_counts'];rate=counts.get('1',0)/sum(counts.values())
        metrics[str(h)].update(single_diagnostics=diagnostics(singles[h]),
            raw_common_diagnostics=diagnostics([(y,p) for y,p,z in common[h]]),
            fixed_projection_common_diagnostics=diagnostics([(y,z) for y,p,z in common[h]]),
            training_rate_baseline_on_single_support=dict(training_rate=rate,n=len(singles[h]),
                brier_score=None if not singles[h] else sum((rate-y)**2 for y,p in singles[h])/len(singles[h])))
    after=[receipt(ref['path']) for ref in refs]
    if refs!=after:raise ValueError('Independent source changed during evaluation')
    report=dict(schema=SCHEMA+'.evaluation',status='FRESH_EXECUTION_SAME_RECIPE_CONTENT_INDEPENDENT_RISK_EVALUATION',
        n=len(seen),not_at_risk_n=empty,joint_unknown_n=joint_unknown,per_horizon=metrics,joint_policy=joint.POLICY,
        raw_monotonicity_violation_n=sealed['raw_monotonicity_violation_n'],maximum_raw_monotonicity_violation=sealed['maximum_raw_monotonicity_violation'],
        source_receipts_before=refs,source_receipts_after=after,model_refit=False,label_dependent_selection=False,
        calibrated_probability=False,input_generalization_claimed=False,physical_GPU_predictor_qualified=False,
        routing_gain_claimed=False,six_term_utility_changed=False,rho_changed=False,causal_receipt_replay=True,online_prediction_claimed=False)
    output.mkdir(parents=True);(output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    feat=sub.add_parser('features');feat.add_argument('--run',required=True);feat.add_argument('--output',required=True)
    pred=sub.add_parser('seal');pred.add_argument('--feature-report',required=True);pred.add_argument('--output',required=True)
    lab=sub.add_parser('labels');lab.add_argument('--sealed-report',required=True);lab.add_argument('--output',required=True)
    ev=sub.add_parser('evaluate');ev.add_argument('--sealed-report',required=True);ev.add_argument('--label-report',required=True);ev.add_argument('--output',required=True)
    args=parser.parse_args()
    result=export_features(args.run,args.output) if args.command=='features' else seal(args.feature_report,args.output) if args.command=='seal' else export_labels(args.sealed_report,args.output) if args.command=='labels' else evaluate(args.sealed_report,args.label_report,args.output)
    print(json.dumps(dict(status=result['status']),indent=2))
