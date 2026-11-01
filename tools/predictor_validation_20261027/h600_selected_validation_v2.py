"""Prospective selected-estimator H600 validation with explicit numerical-v2 events.

The public execution plan remains byte-for-byte unchanged. Resource estimates
are overlaid on prediction candidates only and bound into the forecast audit.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time

from . import flow_validation, flow_gate, flow_crossfit_report, h600_capacity_v1, h600_resources_v1
from . import run_aware_training
from .duration_variant_fit import _verify_duration_definition
from .ocr_trained_validation import LIMITS, CANDIDATES
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling import digest
from .run_aware_report import predictor_sources as original_predictor_sources
from . import h600_event_predict_v2 as numerical, gamma_h600_v2 as gamma
from . import h600_trained_validation as original_entry
from .duration_selection import evaluate
from .flow_profiles import map_flow_collection

SCHEMA='gsdeploy.independent_H600_selected_numerical_v2_validation_protocol.v1'
AUDIT='gsdeploy.independent_H600_selected_numerical_v2_training_audit.v1'


def predictor_sources():
    return [*original_predictor_sources(),*gamma.sources(),receipt(numerical.__file__),
        receipt(original_entry.__file__),receipt(__file__)]


def audit_selection(path):
    selection=read(path);gamma_report=Path(path).parent/'report.json'
    result=read(gamma_report)
    p=gamma.protocol(checked(result['protocol']))
    baseline=read(checked(p['baseline_report']))
    if (result.get('schema')!='gsdeploy.Gamma_H600_training_CV.v1'
            or result.get('policy')!=gamma.POLICY or result.get('independent_validation') is not False
            or result.get('independent_results_read') is not False or result.get('thresholds_tuned') is not False):
        raise ValueError('Frozen complete Gamma training comparison required')
    if (baseline.get('schema')!='gsdeploy.run_aware_H600_numerical_v2_training_report.v1'
            or baseline.get('run_aware_protocol')!=p['training_protocol']
            or baseline.get('resource_manifest')!=p['resource_manifest']
            or baseline.get('numerical_source')!=receipt(Path(__file__).with_name('h600_event_simulator_v2.py'))
            or baseline.get('entry_source')!=receipt(Path(__file__).with_name('h600_numerical_report_v2.py'))
            or not baseline['created_ns']<p['created_ns']<result['created_ns']):
        raise ValueError('Same prospective numerical-v2 WLS training baseline required')
    units=dict(seeds=[627601,627602],routes=sorted(CANDIDATES))
    expected_units={(seed,cid) for seed in units['seeds'] for cid in units['routes']}
    contracts={};paired=[]
    for name,document in (('WLS',baseline),('Gamma',result)):
        if (document.get('independent_validation') is not False
                or document.get('independent_results_read') is not False
                or document.get('thresholds_tuned') is not False
                or document.get('final_gain_eligible') is not False):
            raise ValueError('Training-only estimator comparison required')
        refs=document['source_receipts']+document.get('predictor_source_receipts',[])
        if not refs:raise ValueError('Complete estimator report source receipts required')
        for ref in refs:checked(ref)
        rows={}
        for row in document['candidates']:
            key=(row['seed'],row['candidate_id'])
            if key in rows or key not in expected_units:
                raise ValueError('Exactly two H600 seeds and four routes required')
            rows[key]=row
            for field in ('heldout_plan','heldout_raw','heldout_scored','model'):
                checked(row[field])
                if row[field] not in refs:raise ValueError('Estimator report omitted a case source receipt')
            plan=read(checked(row['heldout_plan']))
            contract=dict(baseline=plan['baseline_candidate_id'],
                tie_tolerance=plan.get('utility_tie_tolerance',0.),
                candidate_ids=sorted(c['id'] for c in plan['candidates']),tasks=sorted(plan['workload_rps']))
            if (plan['seed']!=row['seed'] or plan.get('split')!='training'
                    or plan.get('window_seconds')!=600. or len(plan['trace'])!=1200
                    or contract['candidate_ids']!=units['routes']
                    or contract['tasks']!=['asr','image','ocr','text']
                    or contracts.setdefault(row['seed'],contract)!=contract
                    or type(row['requested']) is not int or row['requested']!=1200
                    or type(row['known']) is not int or not 0<=row['known']<=1200):
                raise ValueError('Fixed H600 request denominator or comparison contract changed')
        if (set(rows)!=expected_units or document.get('requested')!=9600
                or sum(r['requested'] for r in rows.values())!=9600
                or document.get('known')!=sum(r['known'] for r in rows.values())
                or document.get('complete_prediction_coverage')!=all(r['known']==1200 for r in rows.values())):
            raise ValueError('Complete two-seed four-route 9600-request denominator required')
        if evaluate(document,units,LIMITS,contracts)!=document['original_threshold_assessment']:
            raise ValueError(name+' original-threshold assessment changed')
        paired.append({key:{field:row[field] for field in
            ('heldout_plan','heldout_raw','heldout_scored','observed_U','observed_metrics')}
            for key,row in rows.items()})
    if paired[0]!=paired[1]:
        raise ValueError('Estimators must use identical heldout plans, raw observations and actual metrics')
    def score(r):
        a=r['original_threshold_assessment']
        return [a['decision_errors'],a['temporal_failed_or_unknown'],
            a['normalized_temporal_error_mean'] if a['normalized_temporal_error_mean'] is not None else float('inf')]
    selected='gamma' if score(result)<score(baseline) else 'WLS'
    expected=dict(baseline=p['baseline_report'],gamma_score=score(result),baseline_score=score(baseline),selected=selected,
        independent_validation=False,original_thresholds_passed=(result if selected=='gamma' else baseline)
            ['original_threshold_assessment']['original_training_cv_thresholds_passed'])
    if selection!=expected:raise ValueError('Fixed training-only estimator selection changed')
    return dict(selected=selected,selection=receipt(path),gamma_report=receipt(gamma_report),
        gamma_protocol=result['protocol'],baseline=p['baseline_report'],original_thresholds_passed=selection['original_thresholds_passed'])


def audit_model(path,selection_path):
    selected=audit_selection(selection_path)
    if selected['selected']=='WLS':return original_entry.audit_model(path)
    result=gamma.verify_model(path)
    if (result.get('verified') is not True or result.get('heldout_seed') is not None
            or result.get('training_flow_count')!=16 or result.get('diagnostic_repeats_fitted') is not False
            or result.get('validation_sources_used')!=[]
            or read(path)['gamma_training_evidence']['protocol']!=selected['gamma_protocol']):
        raise ValueError('Strict selected full16 Gamma source-bound numerical fit required')
    return result


def audit_resources(path):
    manifest=h600_resources_v1.verify_manifest(path)
    full=manifest['full'];document=read(checked(full['resources']))
    if (full.get('omitted_seed') is not None or full.get('training_seeds')!=[627601,627602]
            or set(full['candidate_overrides'])!=set(CANDIDATES)
            or document.get('H600_training_evidence',{}).get('validation_sources_used')!=[]):
        raise ValueError('Complete H600 primary-only full resource manifest required')
    return manifest


def audit_protocol(path):
    value=read(path)
    if (value.get('schema')!=SCHEMA or value.get('entry_source')!=receipt(__file__)
            or value.get('protocol_sha256')!=digest({k:v for k,v in value.items() if k!='protocol_sha256'})
            or value.get('validation_thresholds')!=LIMITS or value.get('final_gain_eligible') is not False):
        raise ValueError('Exact source-bound unchanged H600 validation protocol required')
    for ref in value['source_receipts']:checked(ref)
    if audit_selection(checked(value['selection']))!=value['estimator_selection']:
        raise ValueError('Changed prospective selected estimator')
    capacity=h600_capacity_v1.verify_protocol(checked(value['capacity_protocol']))
    if value['target_plan']!=capacity['reserved_target_plan']:
        raise ValueError('Validation must use the original reserved capacity target plan')
    threshold=read(checked(value['threshold_addendum']))
    if (threshold.get('schema')!='gsdeploy.training_cv_selection_threshold_source_addendum.v1'
            or threshold.get('thresholds')!=LIMITS
            or read(checked(threshold['threshold_plan']))['validation_thresholds']!=LIMITS):
        raise ValueError('Original temporal and decision thresholds changed')
    return value


def audit_plan(path,protocol_path):
    protocol=audit_protocol(protocol_path)
    plan=flow_validation.validate_plan(read(checked(protocol['target_plan'])))
    if (receipt(path)!=protocol['target_plan'] or plan.get('phase')!='predictor_validation'
            or plan.get('split')!='validation' or plan.get('seed')!=627701
            or plan.get('window_seconds')!=600. or plan.get('drain_seconds')!=300.
            or plan.get('warmup_seconds')!=0. or len(plan['trace'])!=1200
            or tuple(c['id'] for c in plan['candidates'])!=CANDIDATES):
        raise ValueError('Original reserved seed627701 H600/drain300/warmup0 four-route plan required')
    return plan


def prepare_validation(capacity_protocol_path,threshold_addendum_path,selection_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    capacity=h600_capacity_v1.verify_protocol(capacity_protocol_path)
    selected=audit_selection(selection_path)
    threshold=read(threshold_addendum_path)
    refs=[receipt(capacity_protocol_path),capacity['reserved_target_plan'],receipt(threshold_addendum_path),
        threshold['threshold_plan'],receipt(__file__),receipt(h600_capacity_v1.__file__),
        *[v for v in selected.values() if isinstance(v,dict) and 'path' in v],*predictor_sources(),
        receipt(Path(__file__).with_name('ocr_trained_validation.py'))]
    for ref in refs:checked(ref)
    if threshold.get('thresholds')!=LIMITS:raise ValueError('Original thresholds required')
    value=dict(schema=SCHEMA,created_ns=time.perf_counter_ns(),entry_source=receipt(__file__),
        capacity_protocol=receipt(capacity_protocol_path),target_plan=capacity['reserved_target_plan'],
        selection=receipt(selection_path),estimator_selection=selected,forecast_numerics='selected_completion_event_v2',
        threshold_addendum=receipt(threshold_addendum_path),validation_thresholds=LIMITS,
        source_receipts=refs,independent_outcomes_read=False,GPU_started=False,
        final_gain_eligible=False,target_collection_authorized_by_prepare=False)
    value['protocol_sha256']=digest(value)
    output.mkdir(parents=True,exist_ok=False);save(output/'protocol.json',value)
    audit_plan(value['target_plan']['path'],output/'protocol.json')
    return value


def seal(protocol_path,model_path,resource_manifest_path,quality_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    protocol=audit_protocol(protocol_path);plan=audit_plan(protocol['target_plan']['path'],protocol_path)
    verification=audit_model(model_path,protocol['selection']['path']);resources=audit_resources(resource_manifest_path)
    full=resources['full'];resource_document=read(checked(full['resources']))
    model=read(model_path);quality_document=read(quality_path)
    if (model['execution_context_sha256']!=plan['execution_context_sha256']
            or resource_document['execution_context_sha256']!=plan['execution_context_sha256']
            or quality_document.get('split','training')!='training'):
        raise ValueError('Matching execution identity and training-only quality required')
    sources={ 'plan':protocol['target_plan'],'model':receipt(model_path),
        'resources':full['resources'],'quality':receipt(quality_path)}
    execution_sources=[*plan.get('execution_sources',[]),*[plan[k] for k in
        ('runner_source','context','profile_plan','catalog') if k in plan]]
    if plan.get('prephase_memory_evidence'):execution_sources.append(plan['prephase_memory_evidence']['source'])
    tool_sources=[*predictor_sources(),receipt(__file__),receipt(h600_resources_v1.__file__)]
    refs=[*sources.values(),receipt(protocol_path),receipt(resource_manifest_path),*protocol['source_receipts'],
        *flow_validation._sources(model),*flow_validation._sources(resource_document),*execution_sources,
        *quality_document.get('source_receipts',[]),*resources.get('source_receipts',[]),*tool_sources]
    for ref in refs:checked(ref)
    cutoff=time.perf_counter_ns();predictions=[]
    for candidate in plan['candidates']:
        overlaid=copy.deepcopy(candidate);override=full['candidate_overrides'][candidate['id']]
        for key in ('predicted_startup_joules','predicted_prephase_peak_mib'):overlaid[key]=override[key]
        prediction=numerical.predict(plan,overlaid,model,resource_document,
            quality_document.get('loss_by_family_task',quality_document))
        prediction.update(comparison_scope='Absolute six-term utility with H600 training resource estimates',
            predicted_candidate_resource_estimates={key:overlaid[key] for key in
                ('predicted_startup_joules','predicted_prephase_peak_mib')})
        predictions.append(prediction)
    # Recheck every pre-forecast source; do not seal a mixture of source versions.
    for ref in refs:checked(ref)
    audit_model(model_path,protocol['selection']['path']);audit_resources(resource_manifest_path);audit_plan(sources['plan']['path'],protocol_path)
    value=dict(schema='gsdeploy.sealed_native_flow_predictions.v1',created_ns=time.perf_counter_ns(),
        information_cutoff_ns=cutoff,sources=sources,training_sources=flow_validation._sources(model),
        resource_sources=flow_validation._sources(resource_document),execution_sources=execution_sources,
        quality_sources=quality_document.get('source_receipts',[]),predictor_sources=tool_sources,
        tool_source=receipt(__file__),metric_source=receipt(Path(__file__).with_name('flow_metrics.py')),
        cohort_sha256=flow_validation.cohort_sha(plan),plan_sha256=sources['plan']['sha256'],candidates=predictions,
        scope='Reserved public H600 inputs/routes; source-bound full16 and full H600 resources',
        GPU_started=False,actual_future_trace_used=False,gate_protocol=LIMITS,phase='predictor_validation')
    output.mkdir(parents=True,exist_ok=False);save(output/'predictions.json',value)
    save(output/'training-audit.json',dict(schema=AUDIT,predictions=receipt(output/'predictions.json'),
        validation_protocol=receipt(protocol_path),model=receipt(model_path),resource_manifest=receipt(resource_manifest_path),
        entry_source=receipt(__file__),verifier_source=receipt(__file__),verification=verification,
        estimator_selection=protocol['estimator_selection'],forecast_numerics='selected_completion_event_v2',
        full_resource_estimates=full['candidate_overrides'],source_receipts=refs,
        validation_observations_used_for_fitting=False,capacity_bound=False,final_gain_eligible=False,GPU_started=False))
    return value


def check_sealed(folder):
    folder=Path(folder);audit=read(folder/'training-audit.json')
    if (audit.get('schema')!=AUDIT or audit.get('entry_source')!=receipt(__file__)
            or audit.get('verifier_source')!=receipt(__file__)
            or audit.get('predictions')!=receipt(folder/'predictions.json')):
        raise ValueError('Exact H600 full16 forecast audit required')
    for ref in [*audit['source_receipts'],*[audit[k] for k in
            ('predictions','validation_protocol','model','resource_manifest','entry_source','verifier_source')]]:checked(ref)
    protocol=audit_protocol(audit['validation_protocol']['path'])
    verification=audit_model(audit['model']['path'],protocol['selection']['path']);resources=audit_resources(audit['resource_manifest']['path'])
    sealed=read(folder/'predictions.json')
    if audit.get('estimator_selection')!=protocol['estimator_selection'] or audit.get('forecast_numerics')!='selected_completion_event_v2':
        raise ValueError('Selected estimator or numerical engine audit changed')
    plan=audit_plan(sealed['sources']['plan']['path'],audit['validation_protocol']['path'])
    if (sealed['sources']['plan']!=protocol['target_plan'] or sealed['sources']['model']!=audit['model']
            or sealed['sources']['resources']!=resources['full']['resources'] or sealed.get('gate_protocol')!=LIMITS
            or audit['verification']!=verification or audit['full_resource_estimates']!=resources['full']['candidate_overrides']
            or not protocol['created_ns']<=sealed['information_cutoff_ns']<=sealed['created_ns']):
        raise ValueError('H600 seal differs from its strict model/resource/protocol audit')
    for name,key in (('model','training_sources'),('resources','resource_sources'),('quality','quality_sources')):
        document=read(checked(sealed['sources'][name]))
        if sealed[key]!=flow_validation._sources(document) or document.get('split','training')!='training':
            raise ValueError('H600 sealed source arrays omitted training evidence')
    for ref in [*sealed['sources'].values(),*sealed['training_sources'],*sealed['resource_sources'],
            *sealed['execution_sources'],*sealed['quality_sources'],*sealed['predictor_sources'],sealed['tool_source'],sealed['metric_source']]:checked(ref)
    if len(sealed['candidates'])!=len(CANDIDATES) or {p['candidate_id'] for p in sealed['candidates']}!=set(CANDIDATES):
        raise ValueError('Every declared H600 prediction required')
    for prediction in sealed['candidates']:
        expected={k:resources['full']['candidate_overrides'][prediction['candidate_id']][k] for k in
            ('predicted_startup_joules','predicted_prephase_peak_mib')}
        if (prediction.get('forecast_numerics')!='selected_completion_event_v2'
                or prediction.get('numerics_source')!=receipt(Path(__file__).with_name('h600_event_simulator_v2.py'))
                or prediction.get('original_event_budget')!=100000):
            raise ValueError('Sealed prediction uses a different numerical engine')
        if prediction.get('predicted_candidate_resource_estimates')!=expected:
            raise ValueError('H600 candidate resource overlay differs from full training resource model')
    return audit


def validation_evidence(raw_paths,sealed):
    """Verify immutable scoring and strict native execution before assessment."""
    scored_paths=[];audits=[]
    for path in raw_paths:
        raw,scored_ref=flow_crossfit_report.scored_heldout(receipt(path))
        if (raw.get('plan')!=sealed['sources']['plan']
                or type(raw.get('startup_begin_ns')) is not int or raw['startup_begin_ns']<=sealed['created_ns']):
            raise ValueError('H600 original-plan raw must start after complete forecast/source seal')
        native_ref=raw['native_raw'];mapped=map_flow_collection(checked(native_ref),allow_validation_diagnostics=True)
        if (any(t.get('split')!='validation' or t.get('not_for_training') is not True for t in mapped['traces'])
                or mapped['resource_evidence'].get('split')!='validation'):
            raise ValueError('Independent mapping must remain validation-only diagnostics')
        audits.append(dict(candidate_id=raw['candidate_id'],native_raw=native_ref,scored_raw=scored_ref,
            mapping_report=mapped['mapping_report'],quality_sources=raw['quality_evidence']['source_receipts']))
        scored_paths.append(scored_ref['path'])
    return scored_paths,audits


def report(sealed_dir,raw_paths,output):
    audit=check_sealed(sealed_dir);sealed=read(Path(sealed_dir)/'predictions.json')
    scored_paths,evidence=validation_evidence(raw_paths,sealed)
    value=flow_validation.report(sealed_dir,scored_paths,output);check_sealed(sealed_dir)
    for item in evidence:
        for ref in [item['native_raw'],item['scored_raw'],*item['mapping_report']['sources'],*item['quality_sources']]:checked(ref)
    save(Path(output)/'training-audit.json',dict(schema=AUDIT+'.report',
        sealed_audit=receipt(Path(sealed_dir)/'training-audit.json'),report=receipt(Path(output)/'report.json'),
        entry_source=receipt(__file__),validation_protocol=audit['validation_protocol'],
        strict_native_validation_evidence=evidence))
    return value


def gate(protocol_path,report_path,output):
    if Path(output).exists():raise FileExistsError(output)
    protocol=audit_protocol(protocol_path);audit=read(Path(report_path).parent/'training-audit.json')
    if (audit.get('schema')!=AUDIT+'.report' or audit.get('entry_source')!=receipt(__file__)
            or audit.get('validation_protocol')!=receipt(protocol_path) or audit.get('report')!=receipt(report_path)):
        raise ValueError('H600 report audit must bind exact frozen protocol and report')
    for ref in audit.values():
        if isinstance(ref,dict) and 'path' in ref:checked(ref)
    folder=Path(audit['sealed_audit']['path']).parent;check_sealed(folder)
    value=read(report_path);sealed=read(folder/'predictions.json')
    if value['sealed']!=receipt(folder/'predictions.json') or value['report_source']!=receipt(flow_validation.__file__):
        raise ValueError('Report differs from audited H600 forecast or actual reporter')
    for ref in value['raw_sources']:checked(ref)
    _,evidence=validation_evidence([ref['path'] for ref in value['raw_sources']],sealed)
    if evidence!=audit.get('strict_native_validation_evidence'):
        raise ValueError('H600 gate strict native evidence differs from report audit')
    plan=copy.deepcopy(audit_plan(protocol['target_plan']['path'],protocol_path));plan['validation_thresholds']=LIMITS
    result=flow_gate.assess(plan,value)
    result.update(plan=protocol['target_plan'],validation_protocol=receipt(protocol_path),report=receipt(report_path),
        tool=receipt(__file__),final_gain_eligible=False)
    save(output,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('prepare')
    for key in ('capacity-protocol','threshold-addendum','selection','output'):a.add_argument('--'+key,required=True)
    a=sub.add_parser('seal')
    for key in ('protocol','model','resource-manifest','quality','output'):a.add_argument('--'+key,required=True)
    a=sub.add_parser('report');a.add_argument('--sealed-dir',required=True);a.add_argument('--raw',action='append',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('gate')
    for key in ('protocol','report','output'):a.add_argument('--'+key,required=True)
    args=p.parse_args()
    if args.op=='prepare':value=prepare_validation(args.capacity_protocol,args.threshold_addendum,args.selection,args.output)
    elif args.op=='seal':value=seal(args.protocol,args.model,args.resource_manifest,args.quality,args.output)
    elif args.op=='report':value=report(args.sealed_dir,args.raw,args.output)
    else:value=gate(args.protocol,args.report,args.output)
    print(json.dumps({'status':value.get('status','PREPARED' if args.op=='prepare' else 'SEALED'),'GPU_started':False}))


if __name__=='__main__':main()
