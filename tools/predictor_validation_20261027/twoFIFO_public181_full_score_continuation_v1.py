"""Wait for exact existing training writers, then benchmark full six terms once.

CPU only. No fitter, collector, retry, target preparation or GPU command exists
here. Unsupported fits/queries or missing native labels finish in explicit HOLD.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack,nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import twoFIFO_matching_downstream_driver_v1 as matching_driver
from . import twoFIFO_matching_phase12_v1 as response
from . import twoFIFO_matching_prospective_bridge_v1 as law
from . import twoFIFO_public181_after_kernel_v1 as resource_queue
from . import twoFIFO_public181_resource_execute_v1 as resource
from . import twoFIFO_public181_resource_intake_v1 as intake
from . import twoFIFO_public181_expected_score_v2 as expected
from . import twoFIFO_public181_full_score_budget_v1 as budget
from . import twoFIFO_prospective_contract_v1 as public
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import checked,receipt
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard

SCHEMA='gsdeploy.twoFIFO_public181_full_score_training_continuation.v1'
MATCHING_MODULE='tools.predictor_validation_20261027.twoFIFO_matching_downstream_driver_v1'
RESOURCE_MODULE='tools.predictor_validation_20261027.twoFIFO_public181_after_kernel_v1'
require=response.require
verify=response.verify
save=expected.save
publication=resource_queue.kernel.load_publication


def _process(pid,proc_root=Path('/proc')):
    require(type(pid) is int and pid>0,'Positive pinned writer PID required')
    folder=Path(proc_root)/str(pid)
    command=(folder/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    return dict(pid=pid,command=command,start_ticks=(folder/'stat').read_text().rsplit(')',1)[1].split()[19],
        cwd=str((folder/'cwd').resolve())),folder


def _resolve(folder,value):
    path=Path(value)
    return path.resolve() if path.is_absolute() else ((folder/'cwd').resolve()/path).resolve()


def _matching_binding(identity,claim_ref):
    command=identity['command'];claim=checked(verify(claim_ref));cwd=Path(identity['cwd'])
    def resolved(value):return Path(value).resolve() if Path(value).is_absolute() else (cwd/value).resolve()
    require(len(command)==15 and command[1:3]==['-m',MATCHING_MODULE], 'Exact existing matching CPU driver command required')
    pairs=list(zip(command[3::2],command[4::2]));args=dict(pairs)
    require(len(args)==len(pairs)==6 and set(args)=={'--fit-manifest','--generated-design','--resource-design','--queue-claim','--output','--writer-pid'},
        'Unique exact matching CPU driver arguments required')
    require(claim.get('schema')==matching_driver.SCHEMA and claim.get('GPU_started') is False and claim.get('no_retries') is True,
        'Exact nonretrying matching CPU claim required')
    for key in ('fit_manifest','generated_design','resource_design','queue_claim'):
        require(receipt(resolved(args['--'+key.replace('_','-')]))==claim['inputs'][key], 'Matching driver input/claim differs')
    output=resolved(args['--output'])
    require(Path(claim_ref['path']).resolve()==output/'claim.json' and int(args['--writer-pid'])==claim['writer_pid'],
        'Canonical matching driver output and original native writer required')
    require(claim['actions']==matching_driver.stages(args['--fit-manifest'],args['--generated-design'],args['--resource-design'],output),
        'Exact four existing matching CPU actions required')
    return dict(identity,claim=claim_ref,output=str(output))


def matching_identity(pid,claim_ref,*,proc_root=Path('/proc')):
    identity,_=_process(pid,proc_root);return _matching_binding(identity,claim_ref)


def _resource_binding(identity,manifest_ref,claim_ref):
    command=identity['command'];manifest=checked(verify(manifest_ref));path=Path(command[5]) if len(command)>5 else Path('/invalid')
    resolved=path.resolve() if path.is_absolute() else (Path(identity['cwd'])/path).resolve()
    require(len(command)==7 and command[1:5]==['-m',RESOURCE_MODULE,'run','--manifest'] and command[-1]=='--execute'
        and resolved==Path(manifest_ref['path']), 'Exact existing resource queue command required')
    claim=checked(verify(claim_ref))
    require(claim==dict(schema=resource_queue.SCHEMA+'.claim',manifest=manifest_ref,kernel_identity=manifest['kernel_identity'],no_retries=True)
        and Path(claim_ref['path']).resolve()==Path(manifest['output'])/'claim.json', 'Exact resource queue exclusive claim required')
    return dict(identity,manifest=manifest_ref,claim=claim_ref,output=manifest['output'])


def resource_identity(pid,manifest_ref,claim_ref,*,proc_root=Path('/proc')):
    identity,_=_process(pid,proc_root);return _resource_binding(identity,manifest_ref,claim_ref)


def _roles(matching_claim_ref,resource_queue_ref):
    """Authenticate immutable preparations without requiring already-running campaigns idle."""
    claim=checked(verify(matching_claim_ref));fit_ref=claim['inputs']['fit_manifest'];fit=response.audit(verify(fit_ref)['path'])
    require(claim.get('schema')==matching_driver.SCHEMA and claim.get('no_retries') is True and claim.get('GPU_started') is False,
        'Exact source-bound matching CPU driver claim required')
    for ref in strict_references([claim,implementation_sources(matching_driver.__file__)]):verify(ref)
    require(all(ref in claim['sources'] for ref in implementation_sources(matching_driver.__file__)), 'Matching driver implementation not in original claim')
    queue_claim=checked(verify(claim['inputs']['queue_claim']))
    require(queue_claim['matching_manifest']==fit['executor_manifest']
        and str(Path(queue_claim['matching_output'])/'report.json')==fit['expected_continuation_report_path'],
        'Matching CPU source manifest and native queue must agree')
    rq=checked(verify(resource_queue_ref));resource_doc=resource.audit(verify(rq['resource_manifest'])['path'])
    require(rq.get('schema')==resource_queue.SCHEMA and rq.get('status')=='CPU_PREPARED_WAITING_EXPLICIT_EXECUTE'
        and rq.get('no_retries') is True and rq.get('validation_labels_read') is False
        and rq.get('numerical_fit_success_required') is False, 'Exact resource-after-kernel preparation required')
    resource_queue.reviewed(rq['independent_review'],rq['resource_manifest'])
    kernel=checked(verify(rq['kernel_manifest']));campaign=resource_queue.kernel.producer.audit(verify(kernel['campaign'])['path'])
    require(kernel.get('schema')==resource_queue.kernel.SCHEMA and kernel.get('no_retries') is True
        and kernel.get('requires_matching_fit_success') is False and kernel.get('validation_labels_read') is False,
        'Exact native-only kernel continuation required')
    kernel_claim=checked(verify(rq['kernel_claim']))
    require(kernel_claim==dict(manifest=rq['kernel_manifest'],queue_identity=kernel['queue_identity'],no_retries=True)
        and Path(rq['kernel_claim']['path'])==Path(kernel['output'])/'claim.json'
        and rq['kernel_identity']['manifest']==rq['kernel_manifest'] and rq['kernel_identity']['claim']==rq['kernel_claim']
        and rq['kernel_final_path']==str(Path(kernel['output'])/'report.json'), 'Exact kernel writer/claim/final required')
    require(kernel['queue_claim']==claim['inputs']['queue_claim'] and campaign['after_matching_manifest']==fit['executor_manifest']
        and kernel['matching_final_path']==fit['expected_continuation_report_path'], 'Kernel-resource chain belongs to another matching source')
    require(all(ref in rq['sources'] for ref in implementation_sources(resource_queue.__file__))
        and all(ref in kernel['sources'] for ref in implementation_sources(resource_queue.kernel.__file__)),
        'Original resource/kernel continuation closures must remain bound')
    for ref in strict_references([rq,kernel]):verify(ref)
    return claim,fit,rq,resource_doc,kernel,campaign


def _declaration(matching_claim_ref,rq_ref,public_ref,matching_writer,resource_writer,output,closure,frozen):
    claim,fit,rq,resource_doc,kernel,campaign=_roles(matching_claim_ref,rq_ref)
    preparation=public.audit(verify(public_ref)['path']);root=Path(matching_writer['output']);rroot=Path(rq['output'])
    def basic(identity):
        require(type(identity['pid']) is int and identity['pid']>0 and type(identity['start_ticks']) is str
            and identity['start_ticks'].isdigit() and Path(identity['cwd']).is_absolute(), 'Exact pinned process identity required')
        return {key:identity[key] for key in ('pid','command','start_ticks','cwd')}
    require(_matching_binding(basic(matching_writer),matching_claim_ref)==matching_writer
        and _resource_binding(basic(resource_writer),rq_ref,resource_writer['claim'])==resource_writer,
        'Complete pinned command/claim/identity declaration required')
    require(matching_writer['claim']==matching_claim_ref and resource_writer['manifest']==rq_ref
        and resource_writer['claim']==receipt(rroot/'claim.json') and resource_writer['output']==str(rroot),
        'Pinned writer binding differs')
    actions=claim['actions'];require(len(actions)==4 and [a['name'] for a in actions]==['fit','generated','resources','rescore'], 'Original four CPU stages required')
    require(actions[0]['result']==str(root/'fit/report.json'), 'Canonical real matching fit output required')
    sources=strict_references([matching_claim_ref,claim,fit,rq_ref,rq,resource_doc,kernel,campaign,
        resource_writer['claim'],public_ref,preparation,closure,frozen])
    return dict(schema=SCHEMA,status='CPU_PREPARED_EXPLICIT_EXECUTE_REQUIRED',matching_claim=matching_claim_ref,
        matching_writer=matching_writer,matching_fit_manifest=claim['inputs']['fit_manifest'],
        matching_driver_final_path=str(root/'report.json'),matching_fit_complete_path=str(root/'fit-complete.json'),
        matching_fit_path=actions[0]['result'],matching_fit_claim_path=str(Path(claim['inputs']['fit_manifest']['path']).parent/'phase12-fit-claim.json'),
        resource_queue_manifest=rq_ref,resource_writer=resource_writer,resource_final_path=str(rroot/'report.json'),
        resource_native_report_path=str(rroot/'native-resource/report.json'),resource_intake_path=str(rroot/'native-intake/report.json'),
        public_preparation=public_ref,public_contract=preparation['artifacts']['public_contract'],
        expected_resource_attempt_ids=[a['attempt_id'] for a in resource_doc['attempts']],
        evaluation_protocol=expected.EVALUATION_PROTOCOL,lead_seconds=300,poll_seconds=30,
        output=str(Path(output).resolve()/'run-v1'),implementation_sources=closure,source_freeze=frozen,sources=sources,
        no_retries=True,CPU_only=True,GPU_started=False,numerical_fit_invoked=False,validation_labels_read=False,
        benchmark_invocations_maximum=1,automatic_native_target_preparation=False)


def prepare(matching_claim,matching_pid,resource_queue_manifest,resource_pid,public_preparation,output):
    mc=receipt(matching_claim);rq=receipt(resource_queue_manifest);pub=receipt(public_preparation)
    rdoc=checked(verify(rq));rc=receipt(Path(rdoc['output'])/'claim.json')
    mi=matching_identity(matching_pid,mc);ri=resource_identity(resource_pid,rq,rc)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    closure=implementation_sources(__file__);frozen=[]
    for ref in closure:
        path=out/'source-freeze'/Path(ref['path']).relative_to(resource.ROOT);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(Path(ref['path']).read_bytes());fr=receipt(path);require(fr['sha256']==ref['sha256'],'Continuation source changed during freeze')
        frozen.append(dict(original=ref,frozen=fr))
    document=_declaration(mc,rq,pub,mi,ri,out,closure,frozen)
    for ref in document['sources']:verify(ref)
    result=save(out/'manifest.json',document);audit(result['path']);return result


def audit(path):
    ref=receipt(path);doc=checked(verify(ref));require(doc.get('schema')==SCHEMA,'Exact full-score continuation preparation required')
    closure=implementation_sources(__file__);out=Path(path).resolve().parent
    frozen=[dict(original=r,frozen=receipt(out/'source-freeze'/Path(r['path']).relative_to(resource.ROOT))) for r in closure]
    require(all(r['original']['sha256']==r['frozen']['sha256'] for r in frozen),'Complete current-source freeze required')
    expected_doc=_declaration(doc['matching_claim'],doc['resource_queue_manifest'],doc['public_preparation'],
        doc['matching_writer'],doc['resource_writer'],out,closure,frozen)
    require(doc==expected_doc,'Complete fixed source/writer/claim/final declaration must rederive')
    for source in doc['sources']:verify(source)
    verify(ref);return doc


def _identity_or_dead(manifest,role):
    pinned=manifest[role+'_writer']
    try:
        current=matching_identity(pinned['pid'],manifest['matching_claim']) if role=='matching' else resource_identity(
            pinned['pid'],manifest['resource_queue_manifest'],pinned['claim'])
    except (FileNotFoundError,ProcessLookupError):return None
    require(current==pinned,'Pinned '+role+' writer command/PID/start ticks changed')
    return current


def matching_publication(manifest):
    complete=publication(manifest['matching_fit_complete_path']);parent=publication(manifest['matching_driver_final_path'])
    if complete is None:
        if parent is not None:
            claim=checked(verify(manifest['matching_claim']))
            require(parent[1].get('schema')==matching_driver.SCHEMA and parent[1].get('inputs')==claim['inputs'],
                'Foreign matching driver final')
            return dict(terminal_hold='MATCHING_DRIVER_ENDED_WITHOUT_COMPLETED_FIT',artifacts=[parent[0]],documents=[parent[1]])
        return None
    marker_ref,marker=complete;fit_ref=receipt(manifest['matching_fit_path']);fit=checked(verify(fit_ref))
    require(marker==dict(name='fit',result=fit_ref,status='COMPLETE_MATCHING_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS')
        and fit.get('schema')==response.SCHEMA and fit.get('manifest')==manifest['matching_fit_manifest']
        and fit.get('status')==marker['status'] and fit.get('numerical_fit_run') is True
        and fit.get('validation_labels_read') is False and fit.get('GPU_started') is False,
        'Canonical complete real matching fit/marker/source role required')
    claim_ref=receipt(manifest['matching_fit_claim_path']);fit_claim=checked(verify(claim_ref))
    require(fit_claim==dict(manifest=manifest['matching_fit_manifest'],continuation=fit['continuation'],no_retries=True),
        'Exact exclusive original fit claim required; never recreate/retry it')
    return dict(artifacts=[marker_ref,fit_ref,claim_ref],documents=[marker,fit,fit_claim],fit=fit_ref)


def resource_publication(manifest):
    published=publication(manifest['resource_final_path'])
    if published is None:return None
    final_ref,final=published
    require(final.get('schema')==resource_queue.SCHEMA+'.run' and final.get('manifest')==manifest['resource_queue_manifest'],
        'Foreign resource final')
    if final.get('status')!='COMPLETE_EIGHT_NATIVE_RESOURCE_TRAINING_AND_INTAKE_NOT_VALIDATION':
        return dict(terminal_hold='RESOURCE_CHAIN_ENDED_WITHOUT_COMPLETE_INTAKE',artifacts=[final_ref],documents=[final])
    rq=checked(manifest['resource_queue_manifest']);kernel=checked(verify(rq['kernel_manifest']));campaign=checked(verify(kernel['campaign']))
    kernel_final=resource_queue.completed_kernel(rq,kernel,campaign);require(kernel_final is not None,'Missing exact three-kernel final')
    kref,kdoc=kernel_final
    native_ref=receipt(manifest['resource_native_report_path']);native=checked(verify(native_ref))
    intake_ref=receipt(manifest['resource_intake_path']);document=checked(verify(intake_ref))
    require(final.get('kernel_final')==kref and final.get('native_report')==native_ref and final.get('intake')==intake_ref
        and native.get('schema')==resource.CONT_SCHEMA and native.get('manifest')==rq['resource_manifest']
        and native.get('status')=='COMPLETE_EIGHT_NATIVE_RESOURCE_ATTEMPTS_REVIEWED_NO_FIT'
        and native.get('fit_started') is False and native.get('validation_labels_read') is False
        and [r['attempt_id'] for r in native.get('completed_attempts',[])]==manifest['expected_resource_attempt_ids']
        and document.get('schema')==intake.SCHEMA and document.get('manifest')==rq['resource_manifest']
        and document.get('status')=='COMPLETE_EIGHT_NATIVE_RESOURCE_TUPLES_REDERIVED_NO_FIT'
        and document.get('observed_resource_attempts')==8 and document.get('unknown_resource_attempts')==0
        and [r['attempt_id'] for r in document.get('rows',[])]==manifest['expected_resource_attempt_ids']
        and document.get('validation_labels_read') is False and document.get('fit_started') is False,
        'Exact native8 and intake source chain/denominators required')
    reviews=[r['training_review'] for r in native['completed_attempts']]
    require(len(reviews)==8 and len({r['path'] for r in reviews})==8
        and [r['native_review'] for r in document['rows']]==reviews,'Intake must use exactly the same8 native reviews')
    return dict(artifacts=[final_ref,kref,native_ref,intake_ref],documents=[final,kdoc,native,document],
        resource_manifest=rq['resource_manifest'],reviews=reviews,intake=intake_ref)


def wait_inputs(manifest,check_sources,*,poll_seconds=30,identity_reader=_identity_or_dead,pause=time.sleep,publication_watches=None):
    require(type(poll_seconds) in (int,float) and 0<poll_seconds<=60,'Bounded finite polling at most60s required')
    known={};active=[];publications={'matching':matching_publication,'resource':resource_publication}
    with (ExitStack() if publication_watches is None else nullcontext(publication_watches)) as watches:
        while len(known)<2:
            check_sources()
            for guard in active:guard.check()
            for role in ('matching','resource'):
                if role in known:continue
                # Identity check precedes accepting even a complete final. A
                # dead original writer is okay only with its canonical final.
                alive=identity_reader(manifest,role)
                value=publications[role](manifest)
                if value is None:
                    require(alive is not None,role+' writer exited without a complete canonical publication')
                    continue
                refs=strict_references([value['artifacts'],value['documents']])
                guard=watches.enter_context(SourceChangeGuard(refs));watches.callback(guard.check)
                active.append(guard);guard.check();known[role]=value
                if 'terminal_hold' in value:return known
            if len(known)<2:pause(poll_seconds)
        check_sources()
        for guard in active:guard.check()
        # Real run supplies its own stack so publication guards remain active
        # continuously through qualification, command execution and final save.
        for value in known.values():
            for ref in strict_references([value['artifacts'],value['documents']]):verify(ref)
        return known


def qualification(manifest,inputs):
    """Authenticate saved numerics/support without refitting or reading validation."""
    reasons=[value['terminal_hold'] for value in inputs.values() if 'terminal_hold' in value]
    if reasons:return dict(eligible=False,reasons=reasons)
    match=inputs['matching'];resource_input=inputs['resource'];audit_result=response.audit_fit(verify(match['fit']))
    saved=audit_result['report'];require(saved['manifest']==manifest['matching_fit_manifest'],'Foreign current matching fit')
    fitted=saved['fit'];gates=['fitted_root_numerical_qualification','all_leaveout_nuisance_groups_supported',
        'conditional_response_fit_qualified','OCR_entry_support_qualified','response_law_compilation_eligible']
    reasons.extend('MATCHING_FIT_GATE_FALSE:'+key for key in gates if fitted.get(key) is not True)
    native=checked(verify(resource_input['intake']));rows=native['rows']
    for row in rows:
        if row.get('joint_tuple') is None:reasons.append('MISSING_RESOURCE_TUPLE:'+row['attempt_id'])
        quality=row.get('actual_native_quality')
        if quality is None or quality.get('measured_Q') is None:reasons.append('MISSING_ACTUAL_NATIVE_Q:'+row['attempt_id'])
    detail=dict(eligible=False,reasons=reasons,fit=match['fit'],resource_intake=resource_input['intake'],
        original_fit_gates={key:fitted.get(key) for key in gates},fit_numerics_recomputed=False,
        validation_labels_read=False,qualification_is_execution_budget_readiness_only=True)
    if reasons:return detail
    with law.open_law_session(match['fit']['path']) as session:
        payload=session.checked_payload();support=None if payload['law'] is None else law.support(payload['law'],checked(verify(manifest['public_contract'])))
        if payload['law'] is None:reasons.append(payload['unknown_reason'] or 'NO_AUTHENTICATED_LAW')
        elif not support['complete_all_keys_in_every_donor_block']:reasons.append('PUBLIC181_LAW_KEY_SUPPORT_UNKNOWN')
        detail['public_support']=support;session.checked_payload()
    if reasons:return detail
    with expected.open_resource_session(resource_input['resource_manifest']['path'],[r['path'] for r in resource_input['reviews']]) as session:
        rederived=session.checked_payload()['intake']
        require(rederived==native,'Published resource intake must rederive exactly')
        session.checked_payload()
    detail['eligible']=True;return detail


def benchmark_command(manifest,inputs):
    resource_input=inputs['resource'];output=Path(manifest['output'])/'full-six-term-budget'
    command=[sys.executable,'-m','tools.predictor_validation_20261027.twoFIFO_public181_full_score_budget_v1',
        '--fit-report',inputs['matching']['fit']['path'],'--public-preparation',manifest['public_preparation']['path'],
        '--resource-manifest',resource_input['resource_manifest']['path']]
    for ref in resource_input['reviews']:command+=['--resource-review',ref['path']]
    command+=['--output',str(output)]
    return command,output


def run(manifest_path,*,poll_seconds=30,run_command=subprocess.run):
    manifest=audit(manifest_path);ref=receipt(manifest_path);root=Path(manifest['output']);root.mkdir(parents=True,exist_ok=False)
    save(root/'claim.json',dict(schema=SCHEMA+'.claim',manifest=ref,no_retries=True,CPU_only=True,benchmark_invocations_maximum=1))
    result=dict(schema=SCHEMA+'.run',manifest=ref,status='WAITING_EXACT_REAL_TRAINING_INPUTS',CPU_only=True,
        GPU_started=False,numerical_fit_invoked=False,automatic_retry=False,benchmark_invocations=0,
        validation_labels_read=False,independent_prediction_accuracy_qualified=False,final_gain_eligible=False,
        CPU_TEST_ONLY=run_command is not subprocess.run)
    try:
        with SourceChangeGuard(strict_references([ref,manifest])) as guard,ExitStack() as publication_watches:
            inputs=wait_inputs(manifest,guard.check,poll_seconds=poll_seconds,publication_watches=publication_watches);guard.check()
            refs=strict_references([[v['artifacts'],v['documents']] for v in inputs.values()])
            with SourceChangeGuard(refs) as published_guard:
                for source in refs:verify(source)
                eligibility=qualification(manifest,inputs);save(root/'eligibility.json',eligibility)
                guard.check();published_guard.check();result['inputs']={role:{k:v for k,v in value.items() if k!='documents'} for role,value in inputs.items()}
                result['eligibility']=receipt(root/'eligibility.json')
                if not eligibility['eligible']:
                    result.update(status='COMPLETE_HOLD_UNSUPPORTED_OR_UNQUALIFIED_REAL_TRAINING_NO_BENCHMARK',hold_reasons=eligibility['reasons'])
                else:
                    command,output=benchmark_command(manifest,inputs)
                    require(not output.exists(),'One full-score benchmark attempt only; never retry or overwrite')
                    save(root/'benchmark-started.json',dict(command=command,inputs=result['inputs'],no_retries=True))
                    result['benchmark_invocations']=1
                    with (root/'benchmark.log').open('x') as stream:
                        process=run_command(command,stdout=stream,stderr=subprocess.STDOUT,check=False,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                    guard.check();published_guard.check()
                    require(process.returncode==0,'CPU full-score budget command failed; no retry')
                    budget_ref=receipt(output/'report.json')
                    checked_budget=budget.audit(budget_ref['path'],inputs['matching']['fit'],lead_seconds=300,
                        resource_manifest_ref=inputs['resource']['resource_manifest'],resource_review_refs=inputs['resource']['reviews'])
                    require(checked_budget['public_preparation']==manifest['public_preparation'],'Budget public source differs')
                    guard.check();published_guard.check()
                    result.update(status='COMPLETE_FULL_SIX_TERM_CPU_BUDGET_READY_FOR_NATIVE4_REVIEW' if checked_budget['fits_predeclared_budget'] else
                        'COMPLETE_HOLD_FULL_SIX_TERM_CPU_BUDGET_EXCEEDS_FIXED300S',budget=budget_ref,
                        fits_predeclared_budget=checked_budget['fits_predeclared_budget'])
    except Exception as exc:
        result.update(status='COMPLETE_HOLD_SOURCE_OR_CPU_STAGE_FAILURE_NO_RETRY',error=type(exc).__name__+': '+str(exc))
    return save(root/'report.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    cmd=sub.add_parser('prepare')
    for name in ('matching-claim','resource-queue-manifest','public-preparation','output'):cmd.add_argument('--'+name,required=True)
    for name in ('matching-pid','resource-pid'):cmd.add_argument('--'+name,required=True,type=int)
    cmd=sub.add_parser('audit');cmd.add_argument('--manifest',required=True)
    cmd=sub.add_parser('run');cmd.add_argument('--manifest',required=True);cmd.add_argument('--execute',required=True,action='store_true')
    args=parser.parse_args()
    if args.op=='prepare':value=prepare(args.matching_claim,args.matching_pid,args.resource_queue_manifest,args.resource_pid,args.public_preparation,args.output)
    elif args.op=='audit':value=audit(args.manifest)
    else:value=run(args.manifest)
    print(json.dumps(value))
