"""Conditional one-shot V4 continuation; prepare/audit are CPU only.

Waits for the original T8 process and its exact completion, never restarts it.
Only run launches V4 after an explicit source-bound main-thread review.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from . import admission_training_continuation_v1 as tc
from . import admission_validation_v1 as validation
from . import admission_validation_run_v1 as runner
from .native_flow import checked,receipt,save
from .probe_validation import read
from .profiling import digest

BASE=tc.BASE
SCHEMA='gsdeploy.admission_V4_continuation.v1'
T_FOLDER='admission-training-continuation-v1'
V_FOLDER='admission-validation-v1'
FOLDER='admission-validation-continuation-v1'
PID=85446
START_TICKS=602865199
ARGV=['python','-u','-m','tools.predictor_validation_20261027.admission_training_continuation_v1',
      'run','--protocol',f'artifacts/predictor-gpu-20261027/{T_FOLDER}/protocol.json',
      '--review',f'artifacts/predictor-gpu-20261027/{T_FOLDER}/root-review.json']
POLICY=dict(training_pid=PID,training_start_ticks=START_TICKS,training_jobs=8,
    training_requests=9600,independent_seed=628201,candidate_order=list(runner.ORDER),
    independent_runs=4,independent_requests=4800,diagnostic=True,no_retries=True,
    no_seed_replacement=True,service_refit=False,training_launch=False,final_launch=False,
    final_gain_eligible=False)
_PROTOCOL_CACHE=set()


def sources():
    return validation.capacity.unique_refs([receipt(__file__),receipt(tc.__file__),
        receipt(runner.__file__),*validation.sources()])


def _refs_unchecked(value):
    """Extract receipt syntax; source_closure authenticates each unique path."""
    result=[]
    def visit(item):
        if isinstance(item,dict):
            if set(item)=={'path','sha256'}:result.append(item)
            else:
                for child in item.values():visit(child)
        elif isinstance(item,list):
            for child in item:visit(child)
    visit(value);return result


def source_closure(refs):
    """Complete recursive receipts with one hash per unique path.

    Reject conflicting identities before reuse. Frozen execution snapshots also
    bind their original sources, even though original_path is a plain string.
    Callers rehash the returned unique closure across later stage boundaries.
    """
    pending=list(refs);found={}
    while pending:
        ref=pending.pop();key=ref['path']
        if key in found:
            if ref!=found[key]:raise ValueError('Conflicting recursive source identity')
            continue
        path=checked(ref);found[key]=ref
        if path.suffix=='.json':
            value=read(path);pending.extend(_refs_unchecked(value))
            def originals(item):
                if isinstance(item,dict):
                    if 'original_path' in item and isinstance(item.get('frozen'),dict):
                        pending.append(dict(path=str(Path(item['original_path']).resolve()),sha256=item['frozen']['sha256']))
                    for child in item.values():originals(child)
                elif isinstance(item,list):
                    for child in item:originals(child)
            originals(value)
    return list(found.values())


def expected_identity():
    return dict(pid=PID,start_ticks=START_TICKS,argv=ARGV,cwd=str(validation.training.ROOT))


def _paths():
    return dict(training_protocol=str(BASE/T_FOLDER/'protocol.json'),
        training_review=str(BASE/T_FOLDER/'root-review.json'),training_started=str(BASE/T_FOLDER/'started.json'),
        training_completed=str(BASE/T_FOLDER/'completed.json'),validation_protocol=str(BASE/V_FOLDER/'protocol.json'),
        manifests={k:str(BASE/T_FOLDER/(k.removesuffix('_protocol')+'-fit')/'manifest.json') for k in tc.MODULES},
        outputs=dict(sealed=str(BASE/V_FOLDER/'sealed-v1'),execution=str(BASE/V_FOLDER/'execution-v1'),
            report=str(BASE/V_FOLDER/'report')))


def _module_processes():
    found=[]
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():continue
        info=tc.process_identity(int(folder.name))
        if info is None:continue
        argv=info['argv']
        if any(argv[i]=='-m' and argv[i+1]==ARGV[3] for i in range(len(argv)-1)):
            found.append(info)
    return found


def upstream_status(protocol):
    """Original process only; completion file never overrides PID recycling."""
    current=tc.process_identity(PID)
    if any(info['pid']!=PID for info in _module_processes()):
        raise ValueError('Duplicate original T8 continuation process')
    if current is not None and current!=protocol['training_process_identity']:
        raise ValueError('Original T8 PID/start/argv/cwd identity changed or recycled')
    if Path(protocol['paths']['training_completed']).exists():return 'COMPLETED'
    if current is None:raise RuntimeError('Original T8 exited without completed; never restart or retry')
    return 'WAITING'


def prepare(output):
    output=Path(output).resolve()
    if output!=BASE/FOLDER:raise ValueError('Exact fixed V4 continuation output required')
    if output.exists():raise FileExistsError(output)
    paths=_paths();tp,tr=tc.audit_root_review(paths['training_protocol'],paths['training_review'])
    vp=validation.audit_protocol(paths['validation_protocol'])
    if (vp['campaign']!=tp['campaign'] or vp['role_reservation']!=tp['role_reservation']
            or any(vp[k]!=tp[k] for k in tc.MODULES)):
        raise ValueError('V4 and exact T8 must share campaign, roles and three protocols')
    started=read(paths['training_started'])
    if (started.get('source')!=receipt(tc.__file__) or started.get('protocol')!=receipt(paths['training_protocol'])
            or started.get('root_review')!=receipt(paths['training_review']) or started.get('pid')!=PID):
        raise ValueError('Original T8 source-bound started artifact required')
    value=dict(schema=SCHEMA+'.protocol',created_ns=time.perf_counter_ns(),source=receipt(__file__),
        policy=POLICY,paths=paths,training_process_identity=expected_identity(),
        training_protocol=receipt(paths['training_protocol']),training_review=receipt(paths['training_review']),
        training_started=receipt(paths['training_started']),validation_protocol=receipt(paths['validation_protocol']),
        campaign=tp['campaign'],role_reservation=tp['role_reservation'],GPU_started=False,
        training_outcomes_read=False,independent_outcomes_read=False,final_gain_eligible=False)
    upstream_status(value)
    value['source_receipts']=source_closure([*sources(),value['training_protocol'],value['training_review'],
        value['training_started'],value['validation_protocol']])
    value['protocol_sha256']=digest(value);output.mkdir(parents=True,exist_ok=False)
    save(output/'protocol.json',value);audit_protocol(output/'protocol.json');return value


def audit_protocol(path):
    if Path(path).resolve()!=BASE/FOLDER/'protocol.json':raise ValueError('Exact canonical V4 protocol required')
    initial=receipt(path);cache_key=(initial['path'],initial['sha256'])
    value=read(path)
    if (value.get('schema')!=SCHEMA+'.protocol' or value.get('source')!=receipt(__file__)
            or value.get('policy')!=POLICY or value.get('paths')!=_paths()
            or value.get('training_process_identity')!=expected_identity()
            or value.get('GPU_started') is not False or value.get('training_outcomes_read') is not False
            or value.get('independent_outcomes_read') is not False or value.get('final_gain_eligible') is not False
            or value.get('protocol_sha256')!=digest({k:v for k,v in value.items() if k!='protocol_sha256'})):
        raise ValueError('Original V4 policy/process/source/protocol required')
    runner.recheck(value['source_receipts'])
    # A cache hit still hashes the complete immutable transitive closure. The
    # exact protocol hash binds its already-proven completeness, so waiting does
    # not repeatedly parse old raw JSON or reconstruct numerical evidence.
    if cache_key in _PROTOCOL_CACHE:
        checked(initial);return value
    for key in ('training_protocol','training_review','training_started','validation_protocol'):
        if value[key]!=receipt(value['paths'][key]):raise ValueError('Changed exact prerequisite receipt')
    tp,_=tc.audit_root_review(checked(value['training_protocol']),checked(value['training_review']))
    vp=validation.audit_protocol(checked(value['validation_protocol']))
    started=read(checked(value['training_started']))
    if (started.get('source')!=receipt(tc.__file__) or started.get('protocol')!=value['training_protocol']
            or started.get('root_review')!=value['training_review'] or started.get('pid')!=PID):
        raise ValueError('Original T8 source-bound started artifact required')
    if (value['campaign']!=tp['campaign'] or value['role_reservation']!=tp['role_reservation']
            or vp['campaign']!=tp['campaign'] or vp['role_reservation']!=tp['role_reservation']
            or any(vp[k]!=tp[k] for k in tc.MODULES)):
        raise ValueError('Exact common T8/V4 bindings required')
    required=source_closure([*sources(),*[value[k] for k in
        ('training_protocol','training_review','training_started','validation_protocol')]])
    if not {(r['path'],r['sha256']) for r in required}<={(r['path'],r['sha256']) for r in value['source_receipts']}:
        raise ValueError('V4 protocol omitted transitive sources')
    checked(initial);_PROTOCOL_CACHE.add(cache_key)
    return value


def audit_root_review(protocol_path,review_path):
    protocol=audit_protocol(protocol_path);review=read(review_path)
    if (Path(review_path).resolve()!=BASE/FOLDER/'root-review.json'
            or review.get('schema')!=SCHEMA+'.main_thread_review'
            or review.get('decision')!='APPROVED_FOR_CONDITIONAL_V4_ONLY'
            or review.get('reviewer_role')!='main_thread' or review.get('protocol')!=receipt(protocol_path)
            or any(review.get(k)!=protocol[k] for k in ('source','training_protocol','training_review',
                'validation_protocol','campaign','role_reservation','paths','training_process_identity'))
            or type(review.get('created_ns')) is not int or review['created_ns']<=protocol['created_ns']
            or review.get('final_gain_eligible') is not False or review.get('final_launch') is not False
            or not isinstance(review.get('diagnostic_reason'),str) or not review['diagnostic_reason'].strip()):
        raise ValueError('Source-bound main-thread conditional V4-only review required')
    return protocol,review


def wait_upstream(protocol_path,*,poll_seconds=5.):
    if not 0<poll_seconds<=60:raise ValueError('Bounded positive poll interval required')
    while True:
        protocol=audit_protocol(protocol_path);status=upstream_status(protocol)
        if status=='COMPLETED':return
        print(json.dumps(dict(event='WAITING_FOR_ORIGINAL_T8',pid=PID,status=status)),flush=True)
        time.sleep(poll_seconds)


def training_evidence(protocol):
    """One full CPU proof of exact T8 completion and fits, never a service refit."""
    if upstream_status(protocol)!='COMPLETED':raise ValueError('Exact original T8 completed artifact required')
    paths=protocol['paths'];ref=receipt(paths['training_completed']);done=read(checked(ref))
    tp,_=tc.audit_root_review(checked(protocol['training_protocol']),checked(protocol['training_review']))
    if (done.get('schema')!=tc.SCHEMA+'.completed' or done.get('source')!=receipt(tc.__file__)
            or done.get('protocol')!=protocol['training_protocol'] or done.get('root_review')!=protocol['training_review']
            or done.get('job_count')!=8 or done.get('service_refitted') is not False
            or done.get('independent_or_final_launched') is not False or done.get('final_gain_eligible') is not False
            or done.get('GPU_started') is not True):
        raise ValueError('Exact finite source-bound T8 completion required')
    runner.recheck(done['source_receipts'])
    campaign=validation.training.verify(checked(protocol['campaign']));expected=[]
    expected_review=BASE/T_FOLDER/'training-review.json'
    if done.get('training_review')!=receipt(expected_review):raise ValueError('Exact materialized training review required')
    validation.training._review(protocol['campaign']['path'],expected_review)
    for job in campaign['jobs']:
        packet_path=Path(job['output'])/'training-complete.json';packet_ref=receipt(packet_path);packet=read(checked(packet_ref))
        if (packet.get('schema')!=validation.training.SCHEMA+'.completed'
                or packet.get('campaign')!=protocol['campaign'] or packet.get('job')!=job
                or packet.get('review')!=done['training_review'] or packet.get('source')!=receipt(validation.training.__file__)
                or packet.get('request_count')!=1200 or packet.get('controller_sha256')!=job['controller_sha256']
                or packet.get('GPU_started') is not True or packet.get('final_gain_eligible') is not False):
            raise ValueError('All eight original ordered training completion packets required')
        mapped=validation.training.map_training(job,protocol['campaign']['path'])
        if packet['native_raw']!=mapped['native_raw']:raise ValueError('Training completion changed native observation')
        expected.append(dict(job_id=job['job_id'],completion=packet_ref))
    if len(expected)!=8 or done.get('jobs')!=expected:
        raise ValueError('Exact all-eight ordered T8 completion domain required')
    manifests={key:receipt(path) for key,path in paths['manifests'].items()}
    if done.get('manifests')!=manifests:raise ValueError('Exact fixed three T8 fit manifests required')
    for key,module in tc.MODULES.items():
        fit=module.verify_manifest(checked(manifests[key]))
        if fit['protocol']!=tp[key]:raise ValueError('T8 manifest changed mode protocol')
    if done.get('upstream_audit')!=receipt(BASE/T_FOLDER/'upstream-audit.json'):
        raise ValueError('Exact T8 upstream audit required')
    upstream=read(checked(done['upstream_audit']))
    if done.get('upstream_statuses')!={k:upstream[k] for k in ('stage2_gate_status','stage3_status','stage3_conclusion')}:
        raise ValueError('Original upstream FAIL/INCOMPLETE statuses changed')
    refs=source_closure([*protocol['source_receipts'],ref])
    runner.recheck(refs)
    return dict(training_completed=ref,manifests=manifests,upstream_statuses=done['upstream_statuses'],source_receipts=refs)


def _cpu(call,*args):
    env=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    previous={k:os.environ.get(k) for k in env}
    try:os.environ.update(env);return call(*args)
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def run(protocol_path,review_path):
    """One reviewed V4-only invocation; no restart, retry, refit, or final run."""
    protocol,review=audit_root_review(protocol_path,review_path)
    folder=Path(protocol_path).resolve().parent;protocol_ref=receipt(protocol_path);review_ref=receipt(review_path)
    with (folder/'continuation.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('V4 continuation already running; no duplicate') from exc
        save(folder/'started.json',dict(schema=SCHEMA+'.started',source=receipt(__file__),protocol=protocol_ref,
            root_review=review_ref,pid=os.getpid(),created_ns=time.perf_counter_ns(),GPU_started=False,final_gain_eligible=False))
        try:
            wait_upstream(protocol_path)
            checked(protocol_ref);checked(review_ref);protocol,_=audit_root_review(protocol_path,review_path)
            evidence=_cpu(training_evidence,protocol);save(folder/'training-audit.json',evidence)
            refs=source_closure([protocol_ref,review_ref,receipt(folder/'training-audit.json'),*evidence['source_receipts']])
            runner.recheck(refs);outputs=protocol['paths']['outputs'];manifests=evidence['manifests']
            _cpu(validation.seal,protocol['validation_protocol']['path'],manifests['resource_protocol']['path'],
                manifests['quality_protocol']['path'],manifests['capacity_protocol']['path'],outputs['sealed'])
            runner.recheck(refs);checked(protocol_ref);checked(review_ref)
            collection_error=None
            try:runner.run(outputs['sealed'],outputs['execution'],diagnostic=True)
            except Exception as exc:
                collection_error=type(exc).__name__+':'+str(exc)
                if not (Path(outputs['execution'])/'run-state.json').exists():raise
            runner.recheck(refs)
            _cpu(runner.report,outputs['sealed'],outputs['execution'],outputs['report'])
            verification=_cpu(validation.verify_report,Path(outputs['report'])/'report.json')
            if verification.get('verified') is not True:raise ValueError('Independent report recomputation did not verify')
            save(folder/'report-verification.json',verification)
            runner.recheck(refs);checked(protocol_ref);checked(review_ref)
            state=read(Path(outputs['execution'])/'run-state.json');report_folder=Path(outputs['report'])
            report=read(report_folder/'report.json');gate=read(report_folder/'gate.json');audit=read(report_folder/'validation-audit.json')
            value=dict(schema=SCHEMA+'.completed',created_ns=time.perf_counter_ns(),source=receipt(__file__),
                protocol=protocol_ref,root_review=review_ref,training_completed=evidence['training_completed'],
                training_audit=receipt(folder/'training-audit.json'),seal_audit=receipt(Path(outputs['sealed'])/'seal-audit.json'),
                run_state=receipt(Path(outputs['execution'])/'run-state.json'),report=receipt(report_folder/'report.json'),
                gate=receipt(report_folder/'gate.json'),capacity_qualification=receipt(report_folder/'capacity-qualification.json'),
                validation_audit=receipt(report_folder/'validation-audit.json'),report_verification=receipt(folder/'report-verification.json'),
                status='INCOMPLETE_NO_RETRY' if collection_error else 'DIAGNOSTIC_COLLECTED',collection_error=collection_error,
                independent_gate_status=gate['status'],report_status=report['status'],candidate_coverage=report['candidate_coverage'],
                native_requested=4800,native_known=sum(e['known'] for e in audit['evidence']),
                independent_runs_launched=len(state['attempted']),upstream_statuses=evidence['upstream_statuses'],
                service_refitted=False,training_launched=False,final_launched=False,not_for_training=True,
                GPU_started=state['GPU_started'],final_gain_eligible=False)
            value['source_receipts']=source_closure([*refs,*_refs_unchecked(value)])
            runner.recheck(value['source_receipts']);save(folder/'completed.json',value);return value
        except BaseException as exc:
            save(folder/'failed.json',dict(schema=SCHEMA+'.failed',source=receipt(__file__),protocol=protocol_ref,
                root_review=review_ref,error=type(exc).__name__+':'+str(exc),no_retry=True,final_gain_eligible=False))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    a=sub.add_parser('audit');a.add_argument('--protocol',required=True)
    a=sub.add_parser('run');a.add_argument('--protocol',required=True);a.add_argument('--review',required=True)
    args=p.parse_args()
    if args.op=='prepare':value=prepare(args.output)
    elif args.op=='audit':value=audit_protocol(args.protocol)
    else:value=run(args.protocol,args.review)
    print(json.dumps(dict(schema=value['schema'],GPU_started=args.op=='run',final_gain_eligible=False)))


if __name__=='__main__':main()
