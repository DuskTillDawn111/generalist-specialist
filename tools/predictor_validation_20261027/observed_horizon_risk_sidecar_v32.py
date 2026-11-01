"""Frozen horizon-risk sidecar for real reader-visible prefixes only.

Prediction never opens labels. QueueDiagnosticEngine's simulated future maxima
and LifecycleDagPhaseModel's internal simulated lifecycle are not observations.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import json
import math
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from . import horizon_recovery_risk_v31 as risk
from .idle_owner_selection_audit_v20 import read
from .owner_lifecycle_mark_v21 import receipt
from .whole57_visible_risk_adapter_v30 import snapshot_at,verify_prefix

SCHEMA='gsdeploy.observed_horizon_risk_sidecar.v32'
ENVELOPE_FIELDS={'id','run_id','group_id','split','kind','domain','route_id',
    'execution_context_sha256','decision_time_ns','prefix_features'}


def static_source_closure():
    """Full local static import closure, parsed without importing GPU/runtime code."""
    project=Path(__file__).resolve().parents[2]
    def resolve(module):
        path=project.joinpath(*module.split('.'))
        for candidate in (path.with_suffix('.py'),path/'__init__.py'):
            if candidate.is_file():return candidate
        return None
    pending=[Path(__file__).resolve()];seen=set()
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path);parts=list(path.relative_to(project).with_suffix('').parts)
        package=parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            modules=[]
            if isinstance(node,ast.Import):modules=[alias.name for alias in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=package[:len(package)-node.level+1] if node.level else []
                if node.module:base+=node.module.split('.')
                modules=['.'.join(base),*['.'.join([*base,alias.name]) for alias in node.names]]
            for module in modules:
                if module.split('.')[0] not in {'gsdeploy','tools'}:continue
                components=module.split('.')
                for length in range(1,len(components)+1):
                    candidate=resolve('.'.join(components[:length]))
                    if candidate is not None and candidate not in seen:pending.append(candidate)
    return [receipt(path) for path in sorted(seen)]


class ObservedHorizonRisk:
    def __init__(self,model,domain,*,allow_cpu_fixture=False):
        risk.check_domain(domain,require_actual=domain.get('kind')!='cpu_fixture')
        if model.get('schema')!=risk.SCHEMA+'.model' or model.get('model_sha256')!=identity({k:v for k,v in model.items() if k!='model_sha256'}):
            raise ValueError('Frozen horizon candidate required')
        if model['risk_domain_sha256']!=domain['risk_domain_sha256']:
            raise ValueError('Model/domain identity mismatch')
        if model['provenance']['kind']=='cpu_fixture' and not allow_cpu_fixture:
            raise ValueError('CPU fixture model is not measured evidence')
        self.model=copy.deepcopy(model);self.domain=copy.deepcopy(domain);self.allow_cpu_fixture=allow_cpu_fixture

    def predict_prefix(self,envelope,*,cutoff_ns):
        risk.check_domain(self.domain,require_actual=self.domain.get('kind')!='cpu_fixture')
        if self.model.get('model_sha256')!=identity({k:v for k,v in self.model.items() if k!='model_sha256'}):
            raise ValueError('Frozen horizon candidate mutated')
        if set(envelope)!=ENVELOPE_FIELDS:
            raise ValueError('Only exact receipt-visible prefix envelope accepted; future labels/forecast forbidden')
        prefix=envelope['prefix_features']
        if prefix.get('schema')!='gsdeploy.whole57_reader_visible_risk_training.v30.prefix' or prefix.get('clock')!='host_perf_counter_ns':
            raise ValueError('Actual reader-visible v30 prefix schema/clock required')
        verify_prefix(prefix)
        if envelope['decision_time_ns']!=cutoff_ns or prefix['cutoff_ns']!=cutoff_ns:
            raise ValueError('Current exact observation cutoff required')
        if prefix['execution_context_sha256']!=envelope['execution_context_sha256']:
            raise ValueError('Original observation context mismatch')
        if prefix['waiting_count']!=len(prefix['waiting_request_ids']) or type(prefix['waiting_count']) is not int or prefix['waiting_count']<0:
            raise ValueError('Consistent explicit current waiting set required')
        result=dict(schema=SCHEMA,id=envelope['id'],run_id=envelope['run_id'],decision_time_ns=cutoff_ns,
            original_execution_context_sha256=envelope['execution_context_sha256'],
            visible_state_sha256=prefix['visible_state_sha256'],horizon_ns=self.model['horizon_ns'],
            visible_waiting_count=prefix['waiting_count'],
            model_sha256=self.model['model_sha256'],risk_domain_sha256=self.domain['risk_domain_sha256'],
            calibrated_probability=False,probability_validation_established=False,
            input_generalization_claimed=False,physical_predictor_qualified=False,
            six_term_utility_changed=False,rho_changed=False,rolling_forecast_implemented=False,
            source='parent_reader_receipt_prefix_not_simulated_queue_diagnostics')
        if prefix['waiting_count']<=0:return dict(result,status='NOT_AT_RISK_CURRENT_BACKLOG_EMPTY',risk_score=None)
        original=envelope['execution_context_sha256']
        if original not in self.domain['member_execution_context_sha256']:
            return dict(result,status='UNKNOWN_EXECUTION_CONTEXT_OUTSIDE_EXPLICIT_RISK_DOMAIN',risk_score=None)
        if envelope['kind']!=self.model['provenance']['kind']:
            raise ValueError('Actual prefix/model evidence kind mismatch')
        aliases=sorted(self.domain['common_execution_projection']['context']['residents'])
        tasks=sorted({r['task'] for r in self.domain['common_execution_projection']['original_input_registry']})
        try:
            names,values=risk.vector(prefix,aliases,tasks)
            row=dict(original_execution_context_sha256=original,risk_domain_sha256=self.domain['risk_domain_sha256'],
                horizon_ns=self.model['horizon_ns'],feature_names=names,feature_values=values.tolist())
            estimate=risk.score_candidate(self.model,row,self.domain,allow_cpu_fixture=self.allow_cpu_fixture)
        except UnknownEvidence as exc:
            estimate=dict(status='UNKNOWN',reason=str(exc),risk_score=None)
        return dict(result,**estimate)

    def predict_runtime_records(self,plan,records,*,phase_start_ns,context_sha256,cutoff_ns,run_id,group_id,kind='measured'):
        # This convenience entry deliberately receives records for causal replay,
        # not queue-diagnostic engine output. snapshot_at filters receipt first.
        prefix=snapshot_at(plan,records,phase_start_ns,context_sha256,cutoff_ns)
        envelope=dict(id=run_id+':'+str(cutoff_ns-phase_start_ns),run_id=run_id,group_id=group_id,
            split='training',kind=kind,domain=plan.get('domain','cpu_fixture'),route_id=plan.get('route_id','fixture'),
            execution_context_sha256=context_sha256,decision_time_ns=cutoff_ns,prefix_features=prefix)
        return self.predict_prefix(envelope,cutoff_ns=cutoff_ns)


def run(model_path,domain_path,visible_report_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    visible=read(visible_report_path)
    if visible.get('training_only') is not True or visible.get('validation_labels_used') is not False:
        raise ValueError('Reviewed real training observer source required')
    # Deliberately no visible['labels'] reference/read/hash: forecast output is
    # sealed before a separate label-side evaluation joins outcomes.
    closure=static_source_closure()
    refs=[receipt(model_path),receipt(domain_path),receipt(visible_report_path),visible['features'],*closure]
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Frozen observation/model source changed')
    provider=ObservedHorizonRisk(read(model_path),read(domain_path))
    output.mkdir(parents=True);counts={};n=0
    with gzip.open(visible['features']['path'],'rt',encoding='utf-8') as stream,gzip.open(output/'sealed-predictions.jsonl.gz','wt',encoding='utf-8') as dest:
        for line in stream:
            envelope=json.loads(line);prediction=provider.predict_prefix(envelope,cutoff_ns=envelope['decision_time_ns'])
            dest.write(json.dumps(prediction,separators=(',',':'))+'\n');n+=1
            counts[prediction['status']]=counts.get(prediction['status'],0)+1
    after=[receipt(ref['path']) for ref in refs]
    if after!=refs:raise ValueError('Source changed during frozen observer risk prediction')
    report=dict(schema=SCHEMA,status='SEALED_OBSERVER_RISK_OUTPUT_NO_LABELS_READ',predictions=receipt(output/'sealed-predictions.jsonl.gz'),
        source_receipts_before=refs,source_receipts_after=after,count=n,status_counts=counts,
        frozen_bindings=dict(model=refs[0],risk_domain=refs[1],visible_report=refs[2],features=refs[3]),
        complete_static_local_source_closure=closure,
        future_labels_read=False,simulated_queue_maxima_used=False,model_refit=False,
        six_term_utility_changed=False,rho_changed=False,rolling_forecast_implemented=False,
        physical_predictor_qualified=False,calibrated_probability=False,independent_risk_accuracy_established=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def evaluate_sealed(prediction_report_path,label_report_path,model_path,output):
    """Separate whole-capture label join; cannot choose/refit the frozen model."""
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    pred_report=read(prediction_report_path);label_report=read(label_report_path);model=read(model_path)
    if (pred_report.get('schema')!=SCHEMA or pred_report.get('status')!='SEALED_OBSERVER_RISK_OUTPUT_NO_LABELS_READ'
            or pred_report.get('future_labels_read') is not False or pred_report.get('model_refit') is not False):
        raise ValueError('Sealed prediction-before-label report required')
    bindings=pred_report['frozen_bindings']
    if set(bindings)!={'model','risk_domain','visible_report','features'} or bindings['model']!=receipt(model_path):
        raise ValueError('Sealed report frozen model identity differs from evaluated model')
    if bindings['visible_report']!=receipt(label_report_path):
        raise ValueError('Outcome source report differs from sealed visible observation report')
    if label_report.get('features')!=bindings['features']:
        raise ValueError('Outcome report features differ from sealed visible prefixes')
    if pred_report['source_receipts_before']!=pred_report['source_receipts_after']:
        raise ValueError('Sealed prediction source boundary changed')
    if any(ref not in pred_report['source_receipts_before'] for ref in bindings.values()):
        raise ValueError('Sealed frozen bindings not in prediction source closure')
    for ref in pred_report['source_receipts_before']:
        if receipt(ref['path'])!=ref:raise ValueError('Frozen prediction source bytes no longer match')
    closure=static_source_closure()
    if pred_report.get('complete_static_local_source_closure')!=closure:
        raise ValueError('Prediction complete static local source closure differs from evaluator')
    domain=read(bindings['risk_domain']['path']);risk.check_domain(domain,require_actual=model['provenance']['kind']!='cpu_fixture')
    if domain['risk_domain_sha256']!=model['risk_domain_sha256']:raise ValueError('Frozen source domain differs from model domain')
    refs=[receipt(prediction_report_path),pred_report['predictions'],receipt(label_report_path),
        label_report['labels'],receipt(model_path),bindings['risk_domain'],*closure]
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Sealed predictions/outcomes changed')
    if model.get('model_sha256')!=identity({k:v for k,v in model.items() if k!='model_sha256'}):raise ValueError('Frozen model mismatch')
    # Re-evaluate only the frozen observation/model path. This is not a fit and
    # does not consult outcomes: it binds scores, statuses and waiting/state
    # fields to their exact source prefixes, including empty-population claims.
    provider=ObservedHorizonRisk(model,domain,allow_cpu_fixture=model['provenance']['kind']=='cpu_fixture')
    expected={}
    with gzip.open(bindings['features']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            envelope=json.loads(line)
            if envelope['id'] in expected:raise ValueError('Duplicate frozen observation prefix')
            expected[envelope['id']]=provider.predict_prefix(envelope,cutoff_ns=envelope['decision_time_ns'])
    horizon=model['horizon_ns'];labels={}
    with gzip.open(label_report['labels']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['offline_labels']['horizon_ns']==horizon:
                if row['id'] in labels:raise ValueError('Duplicate future label')
                labels[row['id']]=row
    scores=[];unknown=0;not_at_risk=0;run_ids=set();classes={};seen=set()
    with gzip.open(pred_report['predictions']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            prediction=json.loads(line);rid=prediction['id']
            if rid in seen:raise ValueError('Duplicate sealed prediction')
            seen.add(rid);run_ids.add(prediction['run_id'])
            if prediction['run_id'] in model['provenance']['run_ids']:raise ValueError('Independent capture overlaps model training')
            if prediction!=expected.get(rid):raise ValueError('Sealed prediction differs from exact frozen observation/model replay')
            if prediction['model_sha256']!=model['model_sha256'] or prediction['horizon_ns']!=horizon:
                raise ValueError('Prediction/frozen model mismatch')
            if prediction['risk_domain_sha256']!=model['risk_domain_sha256'] or prediction['original_execution_context_sha256'] not in domain['member_execution_context_sha256']:
                raise ValueError('Prediction risk domain/original context not bound to model')
            label=labels[rid]
            if label['run_id']!=prediction['run_id'] or label['execution_context_sha256']!=prediction['original_execution_context_sha256']:
                raise ValueError('Prediction/outcome original identity mismatch')
            if label['decision_time_ns']!=prediction['decision_time_ns'] or label['offline_labels']['window_start_ns']!=prediction['decision_time_ns']:
                raise ValueError('Prediction/outcome decision cutoff mismatch')
            value=prediction['risk_score']
            scored_status=prediction['status']=='UNCALIBRATED_TRAINING_DIAGNOSTIC'
            if prediction['status'] not in {'UNCALIBRATED_TRAINING_DIAGNOSTIC','NOT_AT_RISK_CURRENT_BACKLOG_EMPTY',
                    'UNKNOWN_EXECUTION_CONTEXT_OUTSIDE_EXPLICIT_RISK_DOMAIN','UNKNOWN_OUTSIDE_TRAINING_SUPPORT','UNKNOWN'}:
                raise ValueError('Unknown sealed prediction status')
            if scored_status!=(value is not None) or value is not None and (type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1):
                raise ValueError('Prediction risk score/status must be finite [0,1] or explicit unscored')
            if prediction['status']=='NOT_AT_RISK_CURRENT_BACKLOG_EMPTY':not_at_risk+=1;continue
            # Positive population was sealed from prefix by the predictor.
            target=risk.binary_target(dict(cutoff_ns=prediction['decision_time_ns'],waiting_count=1),label['offline_labels'])
            if target['label'] is None or prediction['risk_score'] is None:unknown+=1;continue
            value=prediction['risk_score'];y=target['label'];scores.append((y,value));classes[str(y)]=classes.get(str(y),0)+1
    if seen!=set(labels) or seen!=set(expected):raise ValueError('Sealed cohort differs from whole original outcome/observation grid')
    positives=[p for y,p in scores if y];negatives=[p for y,p in scores if not y]
    auc=None if not positives or not negatives else sum((p>n)+.5*(p==n) for p in positives for n in negatives)/(len(positives)*len(negatives))
    bins=[]
    for index in range(10):
        cells=[(y,p) for y,p in scores if min(9,int(p*10))==index]
        bins.append(dict(lower=index/10,upper=(index+1)/10,n=len(cells),mean_score=None if not cells else sum(p for y,p in cells)/len(cells),observed_frequency=None if not cells else sum(y for y,p in cells)/len(cells)))
    after=[receipt(ref['path']) for ref in refs]
    if refs!=after:raise ValueError('Frozen evaluation evidence changed')
    report=dict(schema=SCHEMA+'.whole_capture_evaluation',status='FROZEN_MODEL_CAPTURE_HOLDOUT_DIAGNOSTIC',run_ids=sorted(run_ids),
        requested_n=len(seen),scored_n=len(scores),unknown_n=unknown,not_at_risk_n=not_at_risk,class_counts=classes,
        brier_score=None if not scores else sum((p-y)**2 for y,p in scores)/len(scores),rank_auc=auc,reliability_bins=bins,
        source_receipts_before=refs,source_receipts_after=after,model_refit=False,validation_label_model_selection=False,
        probability_calibrated=False,input_generalization_claimed=False,physical_predictor_qualified=False,
        six_term_utility_changed=False,rho_changed=False,rolling_forecast_implemented=False)
    output.mkdir(parents=True);(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    predict=sub.add_parser('predict');predict.add_argument('--model',required=True);predict.add_argument('--risk-domain',required=True);predict.add_argument('--visible-report',required=True);predict.add_argument('--output',required=True)
    evaluate=sub.add_parser('evaluate');evaluate.add_argument('--prediction-report',required=True);evaluate.add_argument('--label-report',required=True);evaluate.add_argument('--model',required=True);evaluate.add_argument('--output',required=True)
    args=parser.parse_args()
    result=run(args.model,args.risk_domain,args.visible_report,args.output) if args.command=='predict' else evaluate_sealed(args.prediction_report,args.label_report,args.model,args.output)
    print(json.dumps(dict(status=result['status']),indent=2))
