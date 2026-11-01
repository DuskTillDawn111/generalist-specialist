"""Explicit diagnostic F8 execution after independent V4 assessment.

Seal is CPU only. Run launches the unchanged native collector in frozen order,
with no retry or replacement. Prediction and capacity failures remain failures.
This entry makes no qualified deployment or positive-benefit certification.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

from . import h600_four_group_diagnostic_v1 as final
from . import h600_admission_pilot_v1 as service
from . import admission_validation_v1 as validation
from . import admission_training_continuation_v1 as continuation
from . import admission_capacity_v1 as capacity
from . import admission_resources_v1 as resources
from . import admission_quality_v1 as quality
from .native_flow import checked,receipt,save
from .probe_validation import read
from .profiling import digest

BASE=final.ROOT/'artifacts/predictor-gpu-20261027'
SCHEMA='gsdeploy.H600_final_F8_diagnostic_execution.v1'


def sources():
    return capacity.unique_refs([receipt(__file__),receipt(continuation.__file__),
        receipt(validation.__file__),receipt(service.__file__),*final.sources()])


def _training(path):
    value=read(path)
    if (Path(path).resolve()!=BASE/'admission-training-continuation-v1/completed.json'
            or value.get('schema')!=continuation.SCHEMA+'.completed'
            or value.get('source')!=receipt(continuation.__file__)
            or value.get('job_count')!=8 or len(value.get('jobs',[]))!=8
            or value.get('service_refitted') is not False
            or value.get('independent_or_final_launched') is not False
            or value.get('final_gain_eligible') is not False):
        raise ValueError('Exact completed finite admission T8 continuation required')
    for ref in value['source_receipts']:checked(ref)
    protocol=continuation.audit_protocol(checked(value['protocol']))
    continuation.audit_root_review(value['protocol']['path'],checked(value['root_review']))
    campaign=final.training.verify(protocol['campaign']['path'])
    if [j['job_id'] for j in value['jobs']]!=[j['job_id'] for j in campaign['jobs']]:
        raise ValueError('All eight ordered source-bound training completions required')
    for key,module in (('capacity_protocol',capacity),('resource_protocol',resources),('quality_protocol',quality)):
        manifest=module.verify_manifest(checked(value['manifests'][key]))
        if manifest['protocol']!=protocol[key]:raise ValueError('Final T8 manifest changed training protocol')
    return value


def _prerequisites(plan_path,validation_report,training_completed):
    if Path(plan_path).resolve()!=BASE/'four-group-diagnostic-v1/plan.json':
        raise ValueError('Exact independently reviewed final plan required')
    plan=final.audit_plan(plan_path)
    training=_training(training_completed)
    if Path(validation_report).resolve()!=BASE/'admission-validation-v1/report/report.json':
        raise ValueError('Exact independent V4 report required')
    verified=validation.verify_report(validation_report)
    if verified['verified'] is not True:raise ValueError('Independent V4 audit must complete')
    report=read(validation_report)
    audit=read(Path(validation_report).parent/'validation-audit.json')
    evidence=audit.get('evidence',[])
    # Report known counts prediction/observation pairs. Actual completion must
    # instead use the native evidence which verify_report just recomputed.
    if (report.get('candidate_coverage')!={'requested':4,'observed':4}
            or len(report.get('candidates',[]))!=4 or len(evidence)!=4
            or {e['candidate_id'] for e in evidence}!=set(validation.capacity.ROUTES)
            or any(e.get('requested')!=1200 or e.get('known')!=1200
                or e.get('quality_audit',{}).get('count')!=1200
                or e.get('quality_audit',{}).get('independently_recomputed') is not True
                or e.get('actual_metrics',{}).get('six_term_comparison_complete') is not True
                for e in evidence)):
        raise ValueError('Complete native V4 cohort and all actual six terms required before F8')
    seal_audit=read(checked(audit['seal_audit']))
    protocol=validation.audit_protocol(checked(seal_audit['protocol']))
    train_protocol=read(checked(training['protocol']))
    if (protocol['campaign']!=train_protocol['campaign']
            or protocol['role_reservation']!=plan['role_reservation']
            or any(seal_audit[k.removesuffix('_protocol')+'_manifest']!=training['manifests'][k]
                for k in ('capacity_protocol','resource_protocol','quality_protocol'))):
        raise ValueError('Final comparison must follow the same training and independent evidence chain')
    gate=read(checked(audit['gate']))
    if gate['status'] not in ('PASS','FAIL','INCOMPLETE'):
        raise ValueError('Original independent prediction gate status required')
    refs=continuation.json_source_closure([sources(),receipt(plan_path),plan,
        receipt(validation_report),report,audit,seal_audit,protocol,
        receipt(training_completed),training])
    return dict(plan=plan,training=training,independent_gate_status=gate['status'],
        independent_gate=audit['gate'],independent_report_verification=verified,
        source_receipts=refs)


def seal(plan_path,validation_report,training_completed,output):
    """Authenticate upstream, then freeze service feasibility before final startup."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    prerequisite=_prerequisites(plan_path,validation_report,training_completed)
    plan=prerequisite['plan'];role=read(checked(plan['role_reservation']))
    model=read(checked(role['service_hypothesis']['model']))
    cutoff=time.perf_counter_ns()
    predictions=[service.predict(plan,c,model) for c in plan['candidates']]
    # This test limits the diagnostic GPU attempt to the original bounded
    # horizon. It does not turn a prediction into physical feasibility proof.
    if any(p['execution_precheck']!='PREDICTED_COMPLETE' or p['known']!=1200 for p in predictions):
        raise ValueError('All eight fixed configurations must predict complete within unchanged bounded drain')
    refs=continuation.json_source_closure([prerequisite['source_receipts'],sources(),role])
    value=dict(schema=SCHEMA+'.seal',created_ns=time.perf_counter_ns(),information_cutoff_ns=cutoff,
        entry_source=receipt(__file__),plan=receipt(plan_path),validation_report=receipt(validation_report),
        training_completed=receipt(training_completed),independent_gate=prerequisite['independent_gate'],
        independent_gate_status=prerequisite['independent_gate_status'],
        independent_report_verification=prerequisite['independent_report_verification'],
        source_receipts=refs,service_predictions=predictions,candidate_order=list(final.ORDER),
        locked_policies=plan['locked_policies'],requests=9600,diagnostic=True,
        service_feasibility_only=True,full_six_term_prediction_known=False,
        no_retries=True,no_seed_replacement=True,not_for_training=True,
        independent_blocks=1,statistical_significance_claim=False,
        final_gain_eligible=False,GPU_started=False)
    value['seal_sha256']=digest(value)
    output.mkdir(parents=True,exist_ok=False);save(output/'seal.json',value)
    return value


def check_sealed(path):
    """Cheap immutable closure check after one numerical prerequisite audit."""
    value=read(path)
    if (value.get('schema')!=SCHEMA+'.seal' or value.get('entry_source')!=receipt(__file__)
            or value.get('seal_sha256')!=digest({k:v for k,v in value.items() if k!='seal_sha256'})
            or value.get('candidate_order')!=list(final.ORDER) or value.get('locked_policies')!=final.LOCKS
            or value.get('requests')!=9600 or value.get('diagnostic') is not True
            or value.get('not_for_training') is not True or value.get('final_gain_eligible') is not False
            or value.get('no_retries') is not True or value.get('no_seed_replacement') is not True):
        raise ValueError('Exact source-bound final diagnostic seal required')
    for ref in value['source_receipts']:checked(ref)
    for k in ('plan','validation_report','training_completed','independent_gate'):checked(value[k])
    if value.get('independent_gate_status')!=read(checked(value['independent_gate'])).get('status'):
        raise ValueError('Final seal changed original independent gate status')
    plan=final.audit_plan(value['plan']['path'])
    if (not plan['created_ns']<value['information_cutoff_ns']<=value['created_ns']
            or tuple(p['candidate_id'] for p in value['service_predictions'])!=final.IDS
            or any(p['execution_precheck']!='PREDICTED_COMPLETE' or p['known']!=1200
                for p in value['service_predictions'])):
        raise ValueError('Final source chronology/full-domain service precheck changed')
    return value


def run(sealed_path,output,*,diagnostic=False):
    if not diagnostic:raise ValueError('Final qualified gate unavailable: explicit diagnostic mode required')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('No final retry, replacement, or output overwrite')
    sealed_ref=receipt(sealed_path);sealed=check_sealed(sealed_path)
    # Once per process, reauthenticate all real numerical/native prerequisites.
    prerequisite=_prerequisites(sealed['plan']['path'],sealed['validation_report']['path'],sealed['training_completed']['path'])
    if (prerequisite['independent_gate']!=sealed['independent_gate']
            or prerequisite['independent_gate_status']!=sealed['independent_gate_status']
            or prerequisite['independent_report_verification']!=sealed['independent_report_verification']):
        raise ValueError('Final independent prerequisite differs from original seal')
    role=read(checked(prerequisite['plan']['role_reservation']))
    required_refs=capacity.unique_refs([*prerequisite['source_receipts'],*sources(),
        prerequisite['plan']['role_reservation'],role['service_hypothesis']['model']])
    sealed_refs={ref['path']:ref for ref in sealed['source_receipts']}
    if any(sealed_refs.get(ref['path'])!=ref for ref in required_refs):
        raise ValueError('Final seal omits authenticated prerequisite source closure')
    model=read(checked(role['service_hypothesis']['model']))
    expected=[service.predict(prerequisite['plan'],c,model) for c in prerequisite['plan']['candidates']]
    if expected!=sealed['service_predictions']:raise ValueError('Final service predictions differ from fresh numerical rollout')
    checked(sealed_ref)
    claim=Path(sealed['plan']['path']).parent/'F8-attempt.json'
    save(claim,dict(schema=SCHEMA+'.attempt',created_ns=time.perf_counter_ns(),
        seal=sealed_ref,plan=sealed['plan'],output=str(output),source=receipt(__file__),
        no_retries=True,diagnostic=True,requested=9600))
    output.mkdir(parents=True,exist_ok=False)
    state=dict(schema=SCHEMA+'.run',source=receipt(__file__),seal=sealed_ref,
        created_ns=time.perf_counter_ns(),candidate_order=list(final.ORDER),attempted=[],completed=[],
        status='STARTED',diagnostic=True,no_retries=True,not_for_training=True,final_gain_eligible=False)
    save(output/'started.json',state)
    error=None
    try:
        for cid in final.ORDER:
            checked(sealed_ref);check_sealed(sealed_path)
            for ref in required_refs:checked(ref)
            state['attempted'].append(cid)
            save(output/(cid+'-attempt.json'),dict(state,attempt_ns=time.perf_counter_ns()))
            final.native_flow.execute(sealed['plan']['path'],cid,output/'runs'/cid)
            audited=final.audit_actual(sealed['plan']['path'],output/'runs'/cid/'raw.json',seal_ns=sealed['created_ns'])
            save(output/(cid+'-actual-audit.json'),audited)
            state['completed'].append(cid)
        state['status']='COMPLETED_DIAGNOSTIC_COLLECTION'
    except Exception as exc:
        state['status']='INCOMPLETE_NO_RETRY';error=type(exc).__name__+':'+str(exc)
        state['error']=error
    checked(sealed_ref);check_sealed(sealed_path)
    for ref in required_refs:checked(ref)
    save(output/'run-state.json',state)
    paths=[output/'runs'/cid/'raw.json' for cid in state['attempted'] if (output/'runs'/cid/'raw.json').exists()]
    report=final.report_actual(sealed['plan']['path'],paths,output/'report',seal_ns=sealed['created_ns'])
    result=dict(schema=SCHEMA+'.completed',created_ns=time.perf_counter_ns(),source=receipt(__file__),
        seal=sealed_ref,run_state=receipt(output/'run-state.json'),report=receipt(output/'report/report.json'),
        attempt=receipt(claim),
        status=report['status'],execution_error=error,known=report['known'],requested=9600,
        final_seal_authenticated=True,independent_gate_status=sealed['independent_gate_status'],
        source_receipts=continuation.json_source_closure([sealed_ref,report,state,sources()]),
        not_for_training=True,statistical_significance_claim=False,final_gain_eligible=False,GPU_started=True)
    save(output/'completed.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    a=sub.add_parser('seal')
    for k in ('plan','validation-report','training-completed','output'):a.add_argument('--'+k,required=True)
    a=sub.add_parser('check');a.add_argument('--sealed',required=True)
    a=sub.add_parser('run');a.add_argument('--sealed',required=True);a.add_argument('--output',required=True)
    a.add_argument('--diagnostic',action='store_true')
    args=parser.parse_args()
    if args.op=='seal':result=seal(args.plan,args.validation_report,args.training_completed,args.output)
    elif args.op=='check':result=check_sealed(args.sealed)
    else:result=run(args.sealed,args.output,diagnostic=args.diagnostic)
    print(json.dumps(dict(status=result.get('status','SEALED'),GPU_started=args.op=='run')))


if __name__=='__main__':main()
