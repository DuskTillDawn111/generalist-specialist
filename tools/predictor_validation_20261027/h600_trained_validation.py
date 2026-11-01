"""Prospective H600 full16 predictor validation on the reserved capacity plan.

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
from .run_aware_report import predictor_sources
from .flow_profiles import map_flow_collection

SCHEMA='gsdeploy.independent_H600_full16_validation_protocol.v1'
AUDIT='gsdeploy.independent_H600_full16_training_audit.v1'


def audit_model(path):
    result=run_aware_training.verify_run_aware_model(path)
    _verify_duration_definition(read(path))
    if (result.get('verified') is not True or result.get('run_aware_model_verified') is not True
            or result.get('variant')!='ocr_handler' or result.get('estimator')!='duration_variants_WLS.v1'
            or result.get('training_flow_count')!=16 or result.get('omitted_flow_id') is not None
            or result.get('omitted_flow_ids')!=[] or result.get('omitted_seed') is not None
            or result.get('validation_sources_used')!=[] or result.get('diagnostic_repeats_fitted') is not False
            or result.get('capacity_evidence_pooled') is not False
            or result.get('direct_and_transitive_sources_verified') is not True):
        raise ValueError('Strict verified full16 OCR handler model without exclusions required')
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


def prepare_validation(capacity_protocol_path,threshold_addendum_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    capacity=h600_capacity_v1.verify_protocol(capacity_protocol_path)
    threshold=read(threshold_addendum_path)
    refs=[receipt(capacity_protocol_path),capacity['reserved_target_plan'],receipt(threshold_addendum_path),
        threshold['threshold_plan'],receipt(__file__),receipt(h600_capacity_v1.__file__),
        receipt(Path(__file__).with_name('ocr_trained_validation.py'))]
    for ref in refs:checked(ref)
    if threshold.get('thresholds')!=LIMITS:raise ValueError('Original thresholds required')
    value=dict(schema=SCHEMA,created_ns=time.perf_counter_ns(),entry_source=receipt(__file__),
        capacity_protocol=receipt(capacity_protocol_path),target_plan=capacity['reserved_target_plan'],
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
    verification=audit_model(model_path);resources=audit_resources(resource_manifest_path)
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
        prediction=flow_crossfit_report.predict(plan,overlaid,model,resource_document,
            quality_document.get('loss_by_family_task',quality_document))
        prediction.update(comparison_scope='Absolute six-term utility with H600 training resource estimates',
            predicted_candidate_resource_estimates={key:overlaid[key] for key in
                ('predicted_startup_joules','predicted_prephase_peak_mib')})
        predictions.append(prediction)
    # Recheck every pre-forecast source; do not seal a mixture of source versions.
    for ref in refs:checked(ref)
    audit_model(model_path);audit_resources(resource_manifest_path);audit_plan(sources['plan']['path'],protocol_path)
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
        entry_source=receipt(__file__),verifier_source=receipt(run_aware_training.__file__),verification=verification,
        full_resource_estimates=full['candidate_overrides'],source_receipts=refs,
        validation_observations_used_for_fitting=False,capacity_bound=False,final_gain_eligible=False,GPU_started=False))
    return value


def check_sealed(folder):
    folder=Path(folder);audit=read(folder/'training-audit.json')
    if (audit.get('schema')!=AUDIT or audit.get('entry_source')!=receipt(__file__)
            or audit.get('verifier_source')!=receipt(run_aware_training.__file__)
            or audit.get('predictions')!=receipt(folder/'predictions.json')):
        raise ValueError('Exact H600 full16 forecast audit required')
    for ref in [*audit['source_receipts'],*[audit[k] for k in
            ('predictions','validation_protocol','model','resource_manifest','entry_source','verifier_source')]]:checked(ref)
    verification=audit_model(audit['model']['path']);resources=audit_resources(audit['resource_manifest']['path'])
    sealed=read(folder/'predictions.json');protocol=audit_protocol(audit['validation_protocol']['path'])
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
    for key in ('capacity-protocol','threshold-addendum','output'):a.add_argument('--'+key,required=True)
    a=sub.add_parser('seal')
    for key in ('protocol','model','resource-manifest','quality','output'):a.add_argument('--'+key,required=True)
    a=sub.add_parser('report');a.add_argument('--sealed-dir',required=True);a.add_argument('--raw',action='append',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('gate')
    for key in ('protocol','report','output'):a.add_argument('--'+key,required=True)
    args=p.parse_args()
    if args.op=='prepare':value=prepare_validation(args.capacity_protocol,args.threshold_addendum,args.output)
    elif args.op=='seal':value=seal(args.protocol,args.model,args.resource_manifest,args.quality,args.output)
    elif args.op=='report':value=report(args.sealed_dir,args.raw,args.output)
    else:value=gate(args.protocol,args.report,args.output)
    print(json.dumps({'status':value.get('status','PREPARED' if args.op=='prepare' else 'SEALED'),'GPU_started':False}))


if __name__=='__main__':main()
