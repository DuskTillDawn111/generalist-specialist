"""One source-bound continuation from CPU full-score readiness to native4.

Preparation and audit are CPU only. Explicit run waits for the original CPU
writer and a separate, real independent budget review; it never creates that
review, fits a model, retries a capture or reads validation labels in advance.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from . import twoFIFO_public181_full_score_continuation_v2 as cpu
from . import twoFIFO_fresh181_native_v4 as native
from . import twoFIFO_fresh181_validation_review_v1 as validation
from . import twoFIFO_fresh181_validation_metrics_v1 as metrics
from . import twoFIFO_fresh181_validation_evaluation_v1 as evaluation
from . import twoFIFO_matching_solo_reference_v1 as solo
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .stage_response_features_v1 import receipt, checked

SCHEMA='gsdeploy.twoFIFO_fresh181_validation_continuation.v1'
MODULE='tools.predictor_validation_20261027.twoFIFO_public181_full_score_continuation_v2'
BUDGET_REVIEW_SCHEMA='gsdeploy.twoFIFO_fresh181_real_budget_independent_readiness_review.v1'
BUDGET_REVIEW_STATUS='PASS_INDEPENDENT_REAL_BUDGET_BOUND_NATIVE4_READINESS_ONLY'
READY='COMPLETE_FULL_SIX_TERM_CPU_BUDGET_READY_FOR_NATIVE4_REVIEW'
CANDIDATES=['shared_balanced','shared_skewed','shared_spill_low','shared_spill_high']
require=cpu.require
verify=cpu.verify
save=cpu.save
ROOT=cpu.resource.ROOT


def _writer_binding(identity,manifest_ref,claim_ref):
    command=identity['command'];cwd=Path(identity['cwd'])
    path=Path(command[5]) if len(command)>5 else Path('/invalid')
    path=path.resolve() if path.is_absolute() else (cwd/path).resolve()
    require(len(command)==7 and command[1:5]==['-m',MODULE,'run','--manifest']
        and command[-1]=='--execute' and path==Path(manifest_ref['path']),
        'Exact original fullscore-v2 CPU writer command required')
    manifest=checked(verify(manifest_ref))
    require(Path(claim_ref['path'])==Path(manifest['output'])/'claim.json'
        and checked(verify(claim_ref))==dict(schema=cpu.SCHEMA+'.claim',manifest=manifest_ref,
            no_retries=True,CPU_only=True,benchmark_invocations_maximum=1),
        'Exact original fullscore-v2 exclusive CPU claim required')
    return dict(identity,manifest=manifest_ref,claim=claim_ref)


def writer_identity(pid,manifest_ref,claim_ref,*,proc_root=Path('/proc')):
    identity,_=cpu._process(pid,proc_root)
    return _writer_binding(identity,manifest_ref,claim_ref)


def _review_refs(refs):
    required=[receipt(m.__file__) for m in (native,native.typed,native.timing,native.expected,native.resource,
        validation,metrics,evaluation,solo)]
    native.response._review_bindings(refs,required)
    documents=[checked(verify(ref)) for ref in refs]
    for source in strict_references(documents):verify(source)
    return documents


def _declaration(cpu_ref,identity,review_refs,budget_review_path,namespace,out,closure,frozen):
    parent=cpu.audit(verify(cpu_ref)['path']);reviews=_review_refs(review_refs)
    basic={k:identity[k] for k in ('pid','command','start_ticks','cwd')}
    require(_writer_binding(basic,cpu_ref,identity['claim'])==identity,
        'Complete pinned original CPU writer identity required')
    require(type(identity['pid']) is int and identity['pid']>0 and identity['start_ticks'].isdigit()
        and Path(identity['cwd']).is_absolute(),'Exact CPU PID, start ticks and cwd required')
    require(type(namespace) is str and re.fullmatch('[a-f0-9]{32}',namespace),
        'Explicit fresh128-bit validation namespace required')
    review_path=Path(budget_review_path).resolve()
    require(review_path.name=='report.json' and review_path!=Path(parent['output'])/'report.json',
        'Separate canonical independent real-budget report path required')
    return dict(schema=SCHEMA,status='CPU_PREPARED_WAITING_EXPLICIT_EXECUTE',
        fullscore_manifest=cpu_ref,fullscore_writer=identity,
        fullscore_final_path=str(Path(parent['output'])/'report.json'),
        independent_source_reviews=review_refs,independent_budget_review_path=str(review_path),
        budget_review_schema=BUDGET_REVIEW_SCHEMA,budget_review_status=BUDGET_REVIEW_STATUS,
        namespace=namespace,output=str(out/'run-v1'),native_preparation_path=str(out/'run-v1/native4'),
        candidates=CANDIDATES,requests_per_candidate=181,lead_seconds=300,
        measurement_seconds=90,drain_seconds=120,prior_READY_settle_seconds=10,
        preflight_free_bytes_required=native.supervisor.PREFLIGHT,
        evaluation_protocol=native.expected.EVALUATION_PROTOCOL,
        implementation_sources=closure,source_freeze=frozen,
        sources=strict_references([cpu_ref,parent,identity['claim'],review_refs,reviews,closure,frozen]),
        poll_seconds=30,no_retries=True,automatic_fit=False,numerical_fit_invoked=False,
        automatic_budget_review=False,validation_labels_read=False,GPU_started=False,
        original_six_term_and_stability_contract_retained=True,
        final_gain_eligible=False)


def prepare(fullscore_manifest,fullscore_pid,independent_reviews,budget_review_path,namespace,output):
    parent_ref=receipt(fullscore_manifest);parent=cpu.audit(fullscore_manifest)
    claim_ref=receipt(Path(parent['output'])/'claim.json')
    identity=writer_identity(fullscore_pid,parent_ref,claim_ref)
    require(not Path(budget_review_path).exists(),
        'Predeclare the independent real-budget review before its publication')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    closure=implementation_sources(__file__);frozen=[]
    for ref in closure:
        target=out/'source-freeze'/Path(ref['path']).relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(ref['path']).read_bytes())
        frozen_ref=receipt(target);require(frozen_ref['sha256']==ref['sha256'],'Continuation source freeze changed')
        frozen.append(dict(original=ref,frozen=frozen_ref))
    value=_declaration(parent_ref,identity,[receipt(p) for p in independent_reviews],budget_review_path,
        namespace,out,closure,frozen)
    for ref in value['sources']:verify(ref)
    result=save(out/'manifest.json',value);audit(result['path']);return result


def audit(path):
    ref=receipt(path);document=checked(verify(ref));require(document.get('schema')==SCHEMA,
        'Exact native4 validation continuation preparation required')
    out=Path(path).resolve().parent;closure=implementation_sources(__file__)
    frozen=[dict(original=r,frozen=receipt(out/'source-freeze'/Path(r['path']).relative_to(ROOT))) for r in closure]
    require(all(r['original']['sha256']==r['frozen']['sha256'] for r in frozen),
        'Full continuation implementation closure freeze required')
    expected=_declaration(document['fullscore_manifest'],document['fullscore_writer'],
        document['independent_source_reviews'],document['independent_budget_review_path'],
        document['namespace'],out,closure,frozen)
    require(document==expected,'Complete fixed native4 continuation declaration must rederive')
    for source in document['sources']:verify(source)
    verify(ref);return document


def _identity_or_dead(manifest):
    pinned=manifest['fullscore_writer']
    try:current=writer_identity(pinned['pid'],manifest['fullscore_manifest'],pinned['claim'])
    except (FileNotFoundError,ProcessLookupError):return None
    require(current==pinned,'Original fullscore-v2 writer PID/start ticks/command changed')
    return current


def _watch(stack,refs):
    refs=strict_references(refs)
    for ref in refs:verify(ref)
    guard=stack.enter_context(SourceChangeGuard(refs));stack.callback(guard.check);guard.check()
    if not hasattr(stack,'native4_source_guards'):stack.native4_source_guards=[]
    stack.native4_source_guards.append(guard)
    return guard


def check_watches(stack):
    for guard in getattr(stack,'native4_source_guards',[]):guard.check()


def wait_fullscore(manifest,stack,check,*,poll_seconds=30,identity_reader=_identity_or_dead,pause=time.sleep):
    while True:
        check();identity=identity_reader(manifest)  # identity is checked even when a final exists
        publication=cpu.publication(manifest['fullscore_final_path'])
        if publication is not None:
            ref,document=publication
            guard=_watch(stack,[ref,document]);guard.check()
            require(document.get('schema')==cpu.SCHEMA+'.run'
                and document.get('manifest')==manifest['fullscore_manifest']
                and document.get('CPU_only') is True and document.get('GPU_started') is False
                and document.get('numerical_fit_invoked') is False
                and document.get('validation_labels_read') is False
                and document.get('CPU_TEST_ONLY') is False,
                'Exact real source-bound canonical fullscore-v2 final required')
            require(document.get('status', '').startswith('COMPLETE_'),'Terminal CPU fullscore final required')
            guard.check();check()
            return ref,document
        require(identity is not None,'Original fullscore writer exited without a complete canonical final')
        pause(poll_seconds)


def qualify_fullscore(manifest,ref,document):
    require(document.get('status')==READY and all(document.get(k) is True for k in
        ('fits_predeclared_budget','budget_qualified','full_score_budget_ready'))
        and document.get('benchmark_invocations')==1,
        'Only complete qualified real full-six-term CPU readiness enables target preparation')
    parent=cpu.audit(manifest['fullscore_manifest']['path'])
    matching=cpu.matching_publication(parent);resource=cpu.resource_publication(parent)
    require(matching is not None and resource is not None
        and not any('terminal_hold' in p for p in (matching,resource)),
        'Exact completed matching15/kernel3/resource8 source chain required')
    inputs={'matching':matching,'resource':resource}
    declared={role:{k:v for k,v in value.items() if k!='documents'} for role,value in inputs.items()}
    require(document.get('inputs')==declared,'CPU final must bind exact canonical complete native training publications')
    eligibility=cpu.qualification(parent,inputs)
    require(eligibility.get('eligible') is True and checked(verify(document['eligibility']))==eligibility
        and Path(document['eligibility']['path'])==Path(parent['output'])/'eligibility.json',
        'Exact independently rederived original fit/resource/support readiness required')
    require(Path(document['budget']['path'])==Path(parent['output'])/'full-six-term-budget/report.json',
        'One canonical actual fullscore budget result required')
    budget=native.timing.audit(document['budget']['path'],matching['fit'],lead_seconds=300,
        resource_manifest_ref=resource['resource_manifest'],resource_review_refs=resource['reviews'])
    require(budget['fits_predeclared_budget'] is True and budget['public_preparation']==parent['public_preparation']
        and budget['full_U_live_pipeline_included'] is True
        and budget['evaluation_protocol']==manifest['evaluation_protocol'],
        'Full-six-term real measured CPU budget and fixed300s protocol required')
    verify(ref);return parent,inputs,budget


def review_required_refs(document):
    return [receipt(m.__file__) for m in (native,native.typed,native.timing,native.expected,native.resource,
        validation,metrics,evaluation,solo)]+[document['budget']]


def accept_budget_review(manifest,fullscore_ref,document,ref):
    review=checked(verify(ref))
    require(Path(ref['path'])==Path(manifest['independent_budget_review_path'])
        and review.get('schema')==BUDGET_REVIEW_SCHEMA and review.get('status')==BUDGET_REVIEW_STATUS
        and review.get('independent_review') is True
        and review.get('fullscore_manifest')==manifest['fullscore_manifest']
        and review.get('fullscore_final')==fullscore_ref and review.get('budget')==document['budget']
        and review.get('fit')==document['inputs']['matching']['fit']
        and review.get('resource_manifest')==document['inputs']['resource']['resource_manifest']
        and review.get('resource_reviews')==document['inputs']['resource']['reviews']
        and review.get('evaluation_protocol')==manifest['evaluation_protocol']
        and review.get('test_summary',{}).get('status')=='PASS'
        and review.get('unresolved_blocking_findings')==[]
        and review.get('GPU_started') is False and review.get('validation_labels_read') is False
        and review.get('numerical_fit_invoked') is False
        and review.get('final_gain_eligible') is False,
        'Separate real independent exact-budget/native4 readiness review required; never fabricate it')
    refs=manifest['independent_source_reviews']+[ref]
    native.response._review_bindings(refs,review_required_refs(document))
    for source in strict_references([ref,review]):verify(source)
    return refs


def wait_budget_review(manifest,fullscore_ref,document,stack,check,*,poll_seconds=30,pause=time.sleep):
    while True:
        check();publication=cpu.publication(manifest['independent_budget_review_path'])
        if publication is not None:
            ref,review=publication;guard=_watch(stack,[ref,review]);check()
            refs=accept_budget_review(manifest,fullscore_ref,document,ref)
            guard.check();check();return refs
        pause(poll_seconds)


def execute_command(manifest_path,item):
    return [sys.executable,'-m','tools.predictor_validation_20261027.twoFIFO_fresh181_native_v4',
        'execute','--manifest',str(Path(manifest_path).resolve()),'--plan',item['plan']['path'],
        '--output',item['collection'],'--execute']


def run_stage(command,log,*,gpu,runner=subprocess.run):
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if not gpu:env['CUDA_VISIBLE_DEVICES']=''
    with Path(log).open('x') as stream:
        process=runner(command,stdout=stream,stderr=subprocess.STDOUT,check=False,env=env)
    require(process.returncode==0,'One-shot stage failed; no automatic retry: '+command[2])


def wait_capacity(path,check,*,poll_seconds=30,pause=time.sleep,disk_usage=shutil.disk_usage):
    """Wait before any claim using the unchanged native supervisor floor."""
    directory=Path(path).resolve()
    while not directory.exists():directory=directory.parent
    while True:
        check();free=disk_usage(directory).free;check()
        if free>=native.supervisor.PREFLIGHT:
            return dict(path=str(directory),actual_free_bytes=free,
                required_free_bytes=native.supervisor.PREFLIGHT,poll_seconds=poll_seconds,
                native_supervisor_recheck_still_required=True)
        pause(poll_seconds)


def _revoke(result,exc):
    previous={k:result.pop(k) for k in ('fullscore','budget_review','native_manifest','reviews','aggregate') if k in result}
    if previous:result['unauthenticated_diagnostics']=dict(previous_outputs=previous,source_authenticated=False,
        usable_for_native_preparation=False,independent_accuracy_qualified=False)
    for row in result.get('candidate_outcomes',[]):
        if row['status'] in ('STARTED','CAPTURE_EXITED_REQUIRES_REVIEW'):row['status']='FAILED_NO_RETRY'
        row.update(source_authenticated=False,empirical_outcome='UNKNOWN_UNAUTHENTICATED')
    result.update(status='COMPLETE_HOLD_SOURCE_OR_NATIVE_VALIDATION_STAGE_FAILURE_NO_RETRY',
        error=type(exc).__name__+': '+str(exc),native_validation_complete=False,
        readiness_qualified=False,independent_accuracy_qualified=False,final_gain_eligible=False,
        aggregate_source_authenticated=False)


def capture_one(root,index,item,target_ref,result,watches,check,run_command,*,poll_seconds=30):
    """An attempt records its fixed-domain ledger even when a stage fails."""
    row=result['candidate_outcomes'][index]
    check();native.require_native_ready(native.audit_plan(item['plan']['path']))
    require(not Path(item['claim']).exists() and not Path(item['collection']).exists(),
        'No started/claimed native4 attempt may be retried')
    capacity=wait_capacity(Path(item['collection']).parent,check,poll_seconds=poll_seconds)
    save(root/('capacity-before-capture-'+str(index)+'.json'),capacity);check()
    command=execute_command(target_ref['path'],item)
    save(root/('capture-'+str(index)+'-started.json'),dict(command=command,plan=item['plan'],no_retries=True))
    row.update(status='STARTED',empirical_outcome='UNKNOWN')
    result['native_capture_invocations']+=1;result['GPU_started']=True
    run_stage(command,root/('capture-'+str(index)+'.log'),gpu=True,runner=run_command);check()
    execution_ref=receipt(Path(item['collection'])/'execution.json')
    execution=checked(verify(execution_ref));native.completed_execution(execution,target_ref,item)
    _watch(watches,[execution_ref,execution,receipt(item['claim'])]);check()
    raw_ref=receipt(Path(item['collection'])/'raw.json')
    row.update(status='CAPTURE_EXITED_REQUIRES_REVIEW',execution=execution_ref,raw=raw_ref,
        actual_request_ledger_absence=None,
        actual_request_ledger=dict(raw=raw_ref,json_pointer='/request_records',
            expected_offered_requests=181,actual_offered_count_verified=False,
            future_labels_read_before_completed_execution=False))
    review_out=root/('review-'+str(index))
    command=[sys.executable,'-m','tools.predictor_validation_20261027.twoFIFO_fresh181_validation_review_v1',
        '--raw',raw_ref['path'],'--manifest',target_ref['path'],
        '--execution',execution_ref['path'],'--output',str(review_out)]
    result['validation_labels_read']=True
    run_stage(command,root/('review-'+str(index)+'.log'),gpu=False,runner=run_command);check()
    review_ref=receipt(review_out/'report.json');review=checked(verify(review_ref))
    _watch(watches,[review_ref,review]);check()
    require(review.get('schema')==validation.SCHEMA and review.get('manifest')==target_ref
        and review.get('plan')==item['plan'] and review.get('execution')==execution_ref,
        'Exact completed native4 independent validation review required')
    result['reviews'].append(review_ref)
    row.update(review=review_ref,status='REVIEW_NOT_ELIGIBLE',source_authenticated=True)
    require(review.get('status')=='PASS_CONDITIONAL_NATIVE_VALIDATION_EVIDENCE_AND_DESCRIPTIVE_ERRORS',
        'Unqualified independent validation evidence; no further native target')
    row['actual_request_ledger']['actual_offered_count_verified']=True
    row.update(status='REVIEWED_CONDITIONAL_EVIDENCE',empirical_outcome='DESCRIPTIVE_ONLY')
    result['completed_candidates'].append(item['candidate_id'])


def run(manifest_path,*,poll_seconds=30,run_command=subprocess.run):
    manifest=audit(manifest_path);ref=receipt(manifest_path)
    root=Path(manifest['output']);root.mkdir(parents=True,exist_ok=False)
    claim=save(root/'claim.json',dict(schema=SCHEMA+'.claim',manifest=ref,no_retries=True,
        native_prepare_invocations_maximum=1,native_capture_invocations_maximum=4))
    result=dict(schema=SCHEMA+'.run',manifest=ref,status='WAITING_EXACT_REAL_CPU_BUDGET_AND_INDEPENDENT_REVIEW',
        native_prepare_invocations=0,native_capture_invocations=0,completed_candidates=[],
        GPU_started=False,numerical_fit_invoked=False,automatic_retry=False,
        validation_labels_read=False,readiness_qualified=False,native_validation_complete=False,
        independent_accuracy_qualified=False,original_stability_evaluated=False,final_gain_eligible=False,
        CPU_TEST_ONLY=run_command is not subprocess.run,candidate_denominator=4,pair_denominator=6,
        offered_per_candidate=181,worlds_per_candidate=12,
        candidate_outcomes=[dict(candidate_id=cid,status='NOT_STARTED',empirical_outcome='UNKNOWN',
            offered_requests=181,plan=None,planned_request_ledger=None,actual_request_ledger=None,
            raw=None,execution=None,review=None,source_authenticated=True) for cid in CANDIDATES])
    try:
        with ExitStack() as watches:
            _watch(watches,[ref,manifest,claim])
            def check():
                check_watches(watches)
            fullscore_ref,document=wait_fullscore(manifest,watches,check,poll_seconds=poll_seconds)
            def check_all():
                check()
            _watch(watches,[fullscore_ref,document]);check_all()
            if document['status']!=READY:
                result.update(status='COMPLETE_HOLD_UPSTREAM_FULLSCORE_NOT_READY_NO_NATIVE_PREPARATION',
                    upstream_status=document['status'],fullscore=fullscore_ref)
            else:
                parent,inputs,budget=qualify_fullscore(manifest,fullscore_ref,document);check_all()
                reviews=wait_budget_review(manifest,fullscore_ref,document,watches,check_all,poll_seconds=poll_seconds)
                _watch(watches,[reviews,[checked(r) for r in reviews],budget]);check_all()
                result.update(fullscore=fullscore_ref,budget_review=reviews[-1],readiness_qualified=True)
                target=Path(manifest['native_preparation_path'])
                require(not target.exists(),'One new native4 preparation only; no overwrite')
                capacity=wait_capacity(target.parent,check_all,poll_seconds=poll_seconds)
                save(root/'capacity-before-native-prepare.json',capacity);check_all()
                save(root/'native-prepare-started.json',dict(fullscore=fullscore_ref,reviews=reviews,no_retries=True))
                result['native_prepare_invocations']=1
                target_ref=native.prepare(parent['public_preparation']['path'],
                    checked(parent['matching_fit_manifest'])['executor_manifest']['path'],
                    inputs['matching']['fit']['path'],manifest['namespace'],target,
                    budget_report_path=document['budget']['path'],
                    resource_manifest_path=inputs['resource']['resource_manifest']['path'],
                    resource_review_paths=[r['path'] for r in inputs['resource']['reviews']],
                    independent_review_paths=[r['path'] for r in reviews],lead_seconds=300)
                target_doc=native.audit(target_ref['path']);_watch(watches,[target_ref,target_doc])
                require(target_doc.get('native_launch_enabled') is True
                    and [a['candidate_id'] for a in target_doc['attempts']]==CANDIDATES,
                    'Exactly the original four source-reviewed enabled native4 targets required')
                result['native_manifest']=target_ref;result['reviews']=[];check_all()
                for row,item in zip(result['candidate_outcomes'],target_doc['attempts']):
                    row.update(plan=item['plan'],planned_request_ledger=dict(plan=item['plan'],
                        json_pointer='/cases/0/arrivals',declared_offered_requests=181),
                        actual_request_ledger_absence='NO_TARGET_CAPTURE_STARTED')
                capture_error=None
                for index,item in enumerate(target_doc['attempts']):
                    try:capture_one(root,index,item,target_ref,result,watches,check_all,run_command,poll_seconds=poll_seconds)
                    except Exception as exc:
                        # A source failure propagates immediately, with no trusted
                        # partial aggregate. A failed native/protocol stage may
                        # still leave genuine earlier independent reviews.
                        check_all();capture_error=type(exc).__name__+': '+str(exc)
                        row=result['candidate_outcomes'][index]
                        if row['status']!='REVIEW_NOT_ELIGIBLE':row['status']='FAILED_NO_RETRY'
                        row.update(error=capture_error,empirical_outcome='UNKNOWN')
                        raw_path=Path(item['collection'])/'raw.json'
                        if raw_path.is_file() and row['actual_request_ledger'] is None:
                            raw_ref=receipt(raw_path);_watch(watches,[raw_ref]);row.update(raw=raw_ref,
                                actual_request_ledger=dict(raw=raw_ref,json_pointer='/request_records',
                                    expected_offered_requests=181,actual_offered_count_verified=False,
                                    failed_capture_diagnostics_only=True))
                        row['actual_request_ledger_absence']='FAILED_CAPTURE_HAS_NO_COMPLETE_AUTHENTICATED_LEDGER' if row['actual_request_ledger'] is None else None
                        break
                aggregate=root/'aggregate'
                command=[sys.executable,'-m','tools.predictor_validation_20261027.twoFIFO_fresh181_validation_evaluation_v1',
                    '--manifest',target_ref['path'],'--reviews',*[r['path'] for r in result['reviews']],
                    '--output',str(aggregate)]
                run_stage(command,root/'aggregate.log',gpu=False,runner=run_command);check_all()
                aggregate_ref=receipt(aggregate/'report.json');aggregate_doc=checked(verify(aggregate_ref))
                _watch(watches,[aggregate_ref,aggregate_doc]);check_all()
                require(aggregate_doc.get('schema')==evaluation.SCHEMA and aggregate_doc.get('manifest')==target_ref
                    and aggregate_doc.get('reviews')==result['reviews']
                    and aggregate_doc.get('candidate_denominator')==4 and aggregate_doc.get('pair_denominator')==6
                    and aggregate_doc.get('offered_per_candidate')==181 and aggregate_doc.get('worlds_per_candidate')==12
                    and aggregate_doc.get('final_gain_eligible') is False,
                    'Exact independently rederived four-candidate/six-pair descriptive aggregate required')
                result.update(status='COMPLETE_FOUR_NATIVE4_VALIDATION_AND_DESCRIPTIVE_EVALUATION_NOT_FINAL_GAIN' if capture_error is None else
                    'COMPLETE_HOLD_INCOMPLETE_NATIVE_VALIDATION_WITH_FIXED_DOMAIN_PARTIAL_AGGREGATE_NO_RETRY',
                    aggregate=aggregate_ref,aggregate_source_authenticated=True,native_validation_complete=capture_error is None,
                    pilot_thresholds_pass=aggregate_doc['all_predeclared_pilot_thresholds_pass'])
                if capture_error is not None:result.update(error=capture_error,readiness_qualified=False)
            check_all();cpu.atomic_save(root/'report.json',result);check_all()
    except Exception as exc:
        _revoke(result,exc);cpu.atomic_save(root/'report.json',result)
    return receipt(root/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    cmd=sub.add_parser('prepare')
    for key in ('fullscore-manifest','budget-review-path','namespace','output'):cmd.add_argument('--'+key,required=True)
    cmd.add_argument('--fullscore-pid',required=True,type=int);cmd.add_argument('--independent-reviews',nargs='+',required=True)
    cmd=sub.add_parser('audit');cmd.add_argument('--manifest',required=True)
    cmd=sub.add_parser('run');cmd.add_argument('--manifest',required=True);cmd.add_argument('--execute',required=True,action='store_true')
    args=parser.parse_args()
    if args.op=='prepare':value=prepare(args.fullscore_manifest,args.fullscore_pid,args.independent_reviews,args.budget_review_path,args.namespace,args.output)
    elif args.op=='audit':value=audit(args.manifest)
    else:value=run(args.manifest)
    print(json.dumps(value))
