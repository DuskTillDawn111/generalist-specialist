"""One-shot CPU training completion coordinator; no collection or wait loop.

An absent final continuation report returns WAITING without opening reviews.
Only exact complete twelve-attempt receipts enable the fixed resource and
conditional generated-queue diagnostics. A started output is never resumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import twoFIFO_campaign_continue_v2 as continuation
from . import twoFIFO_generated_wholeblock_evaluation_v1 as generated
from . import twoFIFO_resource_provider_v3 as resources
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_training_completion_coordinator.v1'
COMPLETE='COMPLETE_ALL_TWELVE_NATIVE_ATTEMPTS_REVIEWED'
require=generated.require
receipt=generated.receipt
checked=generated.checked
save=generated.save


def stage_paths(output):
    root=Path(output).resolve()
    return dict(resource_provider_v3=str(root/'resource-provider-v3'),
        generated_wholeblock_evaluation_v1=str(root/'generated-wholeblock-v1'))


def separate_outputs(prepared,execution):
    require(prepared!=execution and prepared not in execution.parents and execution not in prepared.parents,
        'Preparation and execution outputs must be disjoint without nesting')


def source_refs(manifest,design,driver):
    return strict_references([manifest['design'],design['sources'],manifest['continuation_manifest'],
        driver['sources'],manifest['native_accepted_source'],manifest['additional_acceptance_sources'],
        implementation_sources(__file__)])


def prepare(continuation_manifest_path,continuation_report_path,design_path,native_accepted_source_path,
            output,run_output,*,additional_acceptance_paths=()):
    """Bind existing frozen inputs and one future final-report path, CPU only."""
    driver=continuation.audit(continuation_manifest_path)
    design_ref,design=generated.audit_design(design_path)
    require(driver['campaign']==design['campaign'],'Continuation and generated design campaign differ')
    accepted_ref=receipt(native_accepted_source_path);accepted=checked(accepted_ref)
    require(accepted.get('schema')=='gsdeploy.twoFIFO_native_v3_independent_review.v1'
        and accepted.get('status')=='PASS_NATIVE_CONTRACTS_AND_RESOURCE_COVERAGE'
        and accepted.get('training_feature_extraction_eligible') is True
        and accepted.get('validation_labels_read') is False
        and accepted.get('plan') in [item['plan'] for item in driver['attempts']],
        'Accepted native v3 source review for this campaign required')
    destination=Path(run_output).resolve();final_path=Path(continuation_report_path).resolve()
    output=Path(output).resolve()
    separate_outputs(output,destination)
    require(not destination.exists(),'Fresh distinct fixed execution output required')
    require(final_path.name=='report.json' and final_path.parent!=destination
        and destination not in final_path.parents and final_path.parent not in destination.parents,
        'Final continuation report must remain outside coordinator outputs')
    manifest=dict(schema=SCHEMA+'.manifest',status='CPU_PREPARED_SINGLE_CHECK_NO_AUTOMATIC_EXECUTION',
        continuation_manifest=receipt(continuation_manifest_path),continuation_final_report_path=str(final_path),
        design=design_ref,native_accepted_source=accepted_ref,
        additional_acceptance_sources=[receipt(p) for p in additional_acceptance_paths],
        run_output=str(destination),stages=stage_paths(destination),
        absent_final_report_action='WAITING_WITHOUT_REVIEW_OR_LABEL_ACCESS',
        noncomplete_final_report_action='STOP_WITHOUT_FIT',
        existing_output_action='REFUSE_NO_RETRY_OR_OVERWRITE',
        required_attempt_ids=[item['attempt_id'] for item in driver['attempts']],
        GPU_started=False,numerical_fit_run=False,validation_labels_read=False)
    manifest['sources']=source_refs(manifest,design,driver)
    output.mkdir(parents=True,exist_ok=False)
    result=save(output/'manifest.json',manifest);audit(result['path']);return result


def audit(path):
    ref=receipt(path);manifest=checked(ref)
    require(type(manifest) is dict and set(manifest)=={'schema','status','continuation_manifest',
        'continuation_final_report_path','design','native_accepted_source','additional_acceptance_sources',
        'run_output','stages','absent_final_report_action','noncomplete_final_report_action','existing_output_action',
        'required_attempt_ids','GPU_started','numerical_fit_run','validation_labels_read','sources'},
        'Closed completion coordinator manifest required')
    require(manifest['schema']==SCHEMA+'.manifest'
        and manifest['status']=='CPU_PREPARED_SINGLE_CHECK_NO_AUTOMATIC_EXECUTION'
        and manifest['absent_final_report_action']=='WAITING_WITHOUT_REVIEW_OR_LABEL_ACCESS'
        and manifest['noncomplete_final_report_action']=='STOP_WITHOUT_FIT'
        and manifest['existing_output_action']=='REFUSE_NO_RETRY_OR_OVERWRITE'
        and manifest['GPU_started'] is False and manifest['numerical_fit_run'] is False
        and manifest['validation_labels_read'] is False,'Exact one-shot CPU-only scope required')
    for source in manifest['sources']:require(receipt(source['path'])==source,'Coordinator source drift')
    driver=continuation.audit(manifest['continuation_manifest']['path'])
    _,design=generated.audit_design(manifest['design']['path'])
    require(driver['campaign']==design['campaign']
        and manifest['required_attempt_ids']==[item['attempt_id'] for item in driver['attempts']]
        and len(manifest['required_attempt_ids'])==len(set(manifest['required_attempt_ids']))==12,
        'Exact ordered twelve-attempt campaign required')
    require(manifest['stages']==stage_paths(manifest['run_output'])
        and str(Path(manifest['run_output']).resolve())==manifest['run_output']
        and manifest['sources']==source_refs(manifest,design,driver),'Fixed stage paths and frozen source closure required')
    separate_outputs(Path(path).resolve().parent,Path(manifest['run_output']))
    require(receipt(manifest['continuation_manifest']['path'])==manifest['continuation_manifest']
        and receipt(manifest['design']['path'])==manifest['design'],'Bound manifests changed')
    return ref,manifest,driver,design


def complete_reviews(report,manifest,driver):
    """Authenticate completion and review metadata; no raw/response extraction."""
    require(report.get('schema')==continuation.SCHEMA+'.run'
        and report.get('manifest')==manifest['continuation_manifest']
        and report.get('no_retries') is True and report.get('validation_labels_read') is False,
        'Final report must bind the exact reviewed native continuation')
    rows=report.get('completed_attempts')
    require(type(rows) is list and [row.get('attempt_id') for row in rows]==manifest['required_attempt_ids'],
        'Exactly twelve ordered distinct completed attempts required')
    reviews=[];refs=[];plans=[]
    for row,item in zip(rows,driver['attempts']):
        for key in ('execution','training_review','macros'):
            require(type(row.get(key)) is dict and set(row[key])=={'path','sha256'},'Exact durable completion receipt required')
            require(receipt(row[key]['path'])==row[key],'Completed attempt receipt drift')
            refs.append(row[key])
        require(Path(row['execution']['path']).resolve()==Path(item['collection']).resolve()/'execution.json',
            'Execution receipt belongs to another native attempt')
        plan=checked(item['plan']);plans.append(plan)
        continuation.completed_execution(checked(row['execution']),item['plan'],plan)
        review=checked(row['training_review'])
        require(review.get('inputs',{}).get('plan.json')==item['plan'],'Review plan receipt differs from fixed campaign')
        macros=checked(row['macros'])
        require(macros.get('schema')=='gsdeploy.twoFIFO_composable_response_macro_training.v3'
            and macros.get('status')=='COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER'
            and macros.get('mode')=='natural' and macros.get('validation_labels_read') is False
            and all(macros.get(key)==plan[key] for key in ('attempt_id','capture_id','execution_id'))
            and macros.get('whole_attempt_history_used_for_visible_prefix') is True
            and macros.get('case_history_reset') is False
            and macros.get('measured_requests')==sum(case['offered'] for case in plan['cases'])
            and macros.get('priming_rows_excluded_as_labels')==len(plan['priming']),
            'Complete v3 macro receipt must bind exact attempt and whole-block denominator')
        reviews.append((row['training_review'],review))
    status=resources.intake.planned_intake(plans,reviews)
    require(status['complete'] and status['complete_attempts']==12,'All twelve eligible native reviews required before fit')
    return [ref for ref,_ in reviews],strict_references(refs),status


def run(manifest_path):
    """Check once. No sleeps, polling, launch, resume, automatic retry or tuning."""
    manifest_ref,manifest,driver,design=audit(manifest_path)
    output=Path(manifest['run_output']);final=Path(manifest['continuation_final_report_path'])
    require(not output.exists() and all(not Path(p).exists() for p in manifest['stages'].values()),
        'Existing coordinator/stage output: no retry, resume or overwrite')
    base=dict(schema=SCHEMA+'.run',manifest=manifest_ref,GPU_started=False,validation_labels_read=False,
        external_validation_labels_read=False,numerical_fit_run=False,resource_fit_run=False,generated_evaluation_run=False,
        final_gain_eligible=False,main_queue_provider_eligible=False,full_U_resource_qualification=False,
        actual_native_quality_validated=False,quality_from_frozen_training_expectations_not_native_outcome=True)
    if not final.is_file():
        return dict(base,status='WAITING_FOR_FINAL_CONTINUATION_REPORT',reviews_opened=False,
            continuation_final_report_path=str(final))
    final_ref=receipt(final);report=checked(final_ref)
    if report.get('status')!=COMPLETE:
        return dict(base,status='STOPPED_NONCOMPLETE_CONTINUATION_REPORT',reviews_opened=False,
            continuation_report=final_ref,continuation_status=report.get('status'))
    reviews,completion_sources,intake=complete_reviews(report,manifest,driver)
    bound=strict_references([manifest['sources'],manifest_ref,final_ref,completion_sources])
    def frozen():
        for source in bound:require(receipt(source['path'])==source,'Training completion input/source drift')
    frozen()
    output.mkdir(parents=True,exist_ok=False)
    state=dict(base,status='STARTED_NO_RETRY',continuation_report=final_ref,training_reviews=reviews,
        intake=intake,sources=bound,stages=manifest['stages'],reviews_opened=True)
    save(output/'started.json',state)
    # The root output permanently records consumption before either numerical
    # call. A partial stage output can never be retried by this coordinator.
    try:
        frozen()
        state['resource_fit_run']=True
        state['resource_model']=resources.run(design['campaign']['path'],[ref['path'] for ref in reviews],
            manifest['stages']['resource_provider_v3'])
        require(receipt(state['resource_model']['path'])==state['resource_model'],'Resource output receipt drift')
        frozen();save(output/'resource-stage-complete.json',state)
        state['numerical_fit_run']=True;state['generated_evaluation_run']=True
        state['generated_evaluation']=generated.run(manifest['design']['path'],[ref['path'] for ref in reviews],
            manifest['stages']['generated_wholeblock_evaluation_v1'])
        evaluation=checked(state['generated_evaluation'])
        require(evaluation.get('numerical_fit_run') is True
            and evaluation.get('status') in ('COMPLETE_CONDITIONAL_GENERATED_WHOLEBLOCK_TRAINING_EVALUATION',
                'COMPLETE_NATIVE_TRAINING_FIT_UNQUALIFIED_ALL_GENERATED_WORLDS_UNKNOWN'),
            'Generated stage must retain complete native training diagnostic status')
        frozen()
        require(receipt(state['resource_model']['path'])==state['resource_model'],'Resource output changed during generated stage')
        state.update(status='COMPLETE_CPU_TRAINING_DIAGNOSTICS_NOT_PROVIDER_OR_UTILITY_QUALIFICATION',
            generated_evaluation_status=evaluation['status'])
    except BaseException as exc:
        state.update(status='STOPPED_NO_RETRY',error=type(exc).__name__+': '+str(exc))
    return save(output/'report.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--continuation-manifest',required=True)
    p.add_argument('--continuation-report',required=True);p.add_argument('--design',required=True)
    p.add_argument('--native-accepted-source',required=True);p.add_argument('--additional-acceptance',nargs='*',default=[])
    p.add_argument('--output',required=True);p.add_argument('--run-output',required=True)
    p=sub.add_parser('run');p.add_argument('--manifest',required=True)
    a=parser.parse_args()
    result=(prepare(a.continuation_manifest,a.continuation_report,a.design,a.native_accepted_source,a.output,a.run_output,
        additional_acceptance_paths=a.additional_acceptance) if a.op=='prepare' else run(a.manifest))
    print(json.dumps(result))
