"""New independent OCR-handler model validation entry; no GPU execution.

The frozen sealer/reporter/gate retain their metrics and thresholds. This entry
requires the selected full OCR variant and rechecks its recursive source audit
before and after forecast/report operations. Preparation only writes a new plan.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

from . import flow_validation, flow_gate, flow_protocol
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling_analysis import digest

SEED=627502
CANDIDATES=('shared_balanced','shared_skewed','shared_spill_low','shared_spill_high')
LIMITS={'complete_request_coverage':1.,'complete_candidate_coverage':1.,
    'service_mare_max':.2,'e2e_mare_max':.2,'each_task_p95_absolute_relative_error_max':.2,
    'queue_depth_mean_absolute_error_max_requests':1.,'paired_delta_sign_accuracy_min':1.,
    'candidate_pair_rank_accuracy_min':1.,'require_complete_paired_and_rank_coverage':True,
    'scope':'One prospective periodic known-input pilot; descriptive thresholds, no final-gain confidence claim',
    'post_observation_tuning':False}


def _verifier():
    from . import variant_full_audit
    return variant_full_audit.verify_variant_full_model,receipt(variant_full_audit.__file__)


def audit_model(path):
    verify,source=_verifier();result=verify(path)
    if (result.get('verified') is not True or result.get('variant_full_model_verified') is not True
            or result.get('strict_variant_full_identity_verified') is not True
            or result.get('variant')!='ocr_handler' or result.get('training_flow_count')!=8
            or result.get('omitted_flow_ids')!=[] or result.get('omitted_seed') is not None
            or result.get('validation_sources_used')!=[]):
        raise ValueError('Recursively verified full eight-flow OCR-handler model required')
    # Require the actual full verifier code, not a caller-supplied verifier receipt.
    checked(source)
    return result


def audit_plan(path):
    plan=flow_validation.validate_plan(read(path))
    if (plan.get('phase')!='predictor_validation' or plan.get('split')!='validation'
            or plan.get('seed')!=SEED or plan.get('window_seconds')!=90.
            or len(plan.get('trace',[]))!=181 or tuple(c['id'] for c in plan['candidates'])!=CANDIDATES
            or plan.get('validation_thresholds')!=LIMITS or plan.get('final_confirmation_enabled') is not False
            or plan.get('validation_entry')!=receipt(__file__)):
        raise ValueError('Exact prospective seed627502/90s/181/four-candidate OCR validation contract required')
    source=read(checked(plan['threshold_source_addendum']))
    if source.get('thresholds')!=LIMITS or read(checked(source['threshold_plan']))['validation_thresholds']!=LIMITS:
        raise ValueError('Original prospective validation thresholds changed')
    return plan


def prepare_validation(profile_dir,threshold_addendum_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    addendum=read(threshold_addendum_path)
    if (addendum.get('schema')!='gsdeploy.training_cv_selection_threshold_source_addendum.v1'
            or addendum.get('thresholds')!=LIMITS
            or read(checked(addendum['threshold_plan']))['validation_thresholds']!=LIMITS):
        raise ValueError('Original unchanged threshold-source addendum required')
    plan=flow_protocol.prepare_validation(profile_dir,output,seed=SEED,duration=90.)
    plan_path=output/'plan.json';plan_path.rename(output/'base-validation-plan.json')
    plan.update(kind='prospective_independent_ocr_variant_validation',split='validation',
        created_ns=time.perf_counter_ns(),not_for_training=True,validation_entry=receipt(__file__),
        threshold_source_addendum=receipt(threshold_addendum_path),new_GPU_experiments_started=False,
        model_selection_scope='Training-only three-candidate development; independent outcomes excluded from development')
    plan.pop('plan_sha256');plan['plan_sha256']=digest(plan);save(plan_path,plan)
    audit_plan(plan_path)
    return plan


def seal(plan_path,model_path,resources_path,quality_path,output):
    audit_plan(plan_path);verification=audit_model(model_path);_,verifier_source=_verifier()
    value=flow_validation.seal(plan_path,model_path,resources_path,quality_path,output)
    audit_model(model_path);audit_plan(plan_path)
    save(Path(output)/'training-audit.json',{
        'schema':'gsdeploy.independent_ocr_flow_training_audit.v1',
        'predictions':receipt(Path(output)/'predictions.json'),'model':receipt(model_path),
        'verification':verification,'verifier_source':verifier_source,'entry_source':receipt(__file__),
        'validation_observations_used_for_fitting':False,'GPU_started':False})
    return value


def check_sealed(sealed_dir):
    folder=Path(sealed_dir);audit=read(folder/'training-audit.json')
    if audit.get('schema')!='gsdeploy.independent_ocr_flow_training_audit.v1':
        raise ValueError('Exact OCR full-model training audit required')
    for key in ('predictions','model','verifier_source','entry_source'):checked(audit[key])
    _,verifier_source=_verifier()
    if audit['entry_source']!=receipt(__file__) or audit['verifier_source']!=verifier_source:
        raise ValueError('OCR training audit must bind actual entry and full verifier')
    if audit['predictions']!=receipt(folder/'predictions.json'):
        raise ValueError('OCR training audit binds a different sealed forecast')
    sealed=read(folder/'predictions.json')
    if sealed['sources']['model']!=audit['model']:
        raise ValueError('OCR audit model differs from sealed model')
    audit_plan(checked(sealed['sources']['plan']));audit_model(audit['model']['path'])
    return audit


def report(sealed_dir,raw_paths,output):
    check_sealed(sealed_dir);value=flow_validation.report(sealed_dir,raw_paths,output);check_sealed(sealed_dir)
    save(Path(output)/'training-audit.json',{
        'schema':'gsdeploy.independent_ocr_flow_report_training_audit.v1',
        'sealed_audit':receipt(Path(sealed_dir)/'training-audit.json'),
        'report':receipt(Path(output)/'report.json'),'entry_source':receipt(__file__)})
    return value


def gate(plan_path,report_path,output):
    audit=read(Path(report_path).parent/'training-audit.json')
    if audit.get('schema')!='gsdeploy.independent_ocr_flow_report_training_audit.v1':
        raise ValueError('OCR independent report audit required')
    for key in ('sealed_audit','report','entry_source'):checked(audit[key])
    if audit['report']!=receipt(report_path) or audit['entry_source']!=receipt(__file__):
        raise ValueError('OCR gate audit binds a different report or entry')
    check_sealed(Path(audit['sealed_audit']['path']).parent)
    return flow_gate.run(plan_path,report_path,output)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('prepare')
    for field in ('profile-dir','threshold-addendum','output'):a.add_argument('--'+field,required=True)
    a=sub.add_parser('seal')
    for field in ('plan','model','resources','quality','output'):a.add_argument('--'+field,required=True)
    a=sub.add_parser('report');a.add_argument('--sealed-dir',required=True);a.add_argument('--raw',action='append',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('gate')
    for field in ('plan','report','output'):a.add_argument('--'+field,required=True)
    args=p.parse_args()
    if args.op=='prepare':value=prepare_validation(args.profile_dir,args.threshold_addendum,args.output)
    elif args.op=='seal':value=seal(args.plan,args.model,args.resources,args.quality,args.output)
    elif args.op=='report':value=report(args.sealed_dir,args.raw,args.output)
    else:value=gate(args.plan,args.report,args.output)
    print(json.dumps({'status':value.get('status','PREPARED' if args.op=='prepare' else 'SEALED'),'GPU_started':False}))


if __name__=='__main__':main()
