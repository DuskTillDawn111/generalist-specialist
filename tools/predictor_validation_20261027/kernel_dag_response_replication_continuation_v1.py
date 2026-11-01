"""Source-bound matching-to-kernel response collection continuation.

CPU preparation creates a new immutable manifest. Explicit run waits for the
pinned matching queue and complete native capture, executes three fresh kernel
lifecycles once, then performs only native review/occupancy/intake/alignment CPU
stages. No fit-success gate or automatic retry is introduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import kernel_dag_response_replication_producer_v1 as producer
from . import kernel_dag_repeated_instance_alignment_v1 as repeated
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .kernel_sequence_audit_v2 import verify
from .native_collect import receipt
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard

SCHEMA='gsdeploy.kernel_dag_response_replication_continuation.v1'
REVIEW_SCHEMA='gsdeploy.kernel_dag_response_replication_producer_independent_review.v1'
REVIEW_STATUS='PASS_CPU_PREPARED_NATIVE_REAUTHENTICATION_INTERFACES_NOT_NATIVE_QUALIFICATION'
QUEUE_SCHEMA='gsdeploy.twoFIFO_matching_after_recovery.v1'
QUEUE_STATUS='COMPLETE_MATCHING_CAPTURE_AFTER_EXACT12_NOT_PREDICTOR_QUALIFICATION'
QUEUE_MODULE='tools.predictor_validation_20261027.twoFIFO_matching_after_recovery_v1'
MODULE='tools.predictor_validation_20261027.kernel_dag_response_replication_continuation_v1'

need=producer.need
read=producer.read
write=producer.write


def continuation_identity(pid,manifest_path,*,proc_root=Path('/proc')):
    need(type(pid) is int and pid>0,'Positive kernel continuation PID required')
    root=Path(proc_root)/str(pid);command=(root/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    need(len(command)==7 and command[1:5]==['-m',MODULE,'run','--manifest'] and command[-1]=='--execute',
         'Exact kernel continuation controller command required')
    path=Path(command[5]);path=path if path.is_absolute() else (root/'cwd').resolve()/path
    need(path.resolve()==Path(manifest_path).resolve(),'Kernel continuation uses foreign manifest')
    return dict(pid=pid,start_ticks=(root/'stat').read_text().rsplit(')',1)[1].split()[19],command=command,
        manifest=receipt(manifest_path))


def queue_identity(pid,claim_ref,final_path,*,proc_root=Path('/proc')):
    need(type(pid) is int and pid>0,'Positive exact matching queue PID required')
    root=Path(proc_root)/str(pid)
    command=(root/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    need(len(command)>=4 and command[1:3]==['-m',QUEUE_MODULE] and command[-1]=='--execute',
         'Exact matching queue controller command required')
    options=command[3:-1]
    need(len(options)%2==0 and len(set(options[::2]))==len(options)//2,'Unambiguous matching queue options required')
    options=dict(zip(options[::2],options[1::2]));cwd=(root/'cwd').resolve()
    def canonical(value):
        path=Path(value);return (path if path.is_absolute() else cwd/path).resolve()
    claim=read(claim_ref)
    need(claim.get('schema')==QUEUE_SCHEMA
         and canonical(options['--output'])==Path(claim_ref['path']).resolve().parent
         and canonical(options['--matching-output'])==Path(final_path).resolve().parent
         and canonical(options['--matching-manifest'])==Path(claim['matching_manifest']['path']).resolve()
         and canonical(options['--fit-manifest'])==Path(claim['fit_manifest']['path']).resolve()
         and canonical(options['--independent-review'])==Path(claim['independent_review']['path']).resolve()
         and Path(claim['matching_output']).resolve()==Path(final_path).resolve().parent,
         'Exact queue claim/output/matching manifest binding required')
    return dict(pid=pid,start_ticks=(root/'stat').read_text().rsplit(')',1)[1].split()[19],
                command=command,claim=claim_ref,final_path=str(Path(final_path).resolve()))


def checked_review(ref,campaign_ref):
    review=read(ref)
    need(review.get('schema')==REVIEW_SCHEMA and review.get('status')==REVIEW_STATUS
         and review.get('campaign')==campaign_ref and review.get('unresolved_blocking_findings')==[]
         and type(review.get('test_summary')) is dict and review['test_summary'].get('status')=='PASS',
         'Pinned root independent CPU producer/native reauthentication review required')
    for source in strict_references([ref,review]):verify(source)
    return review


def idle(campaign):
    for entry in campaign['runs']:
        run=Path(entry['run'])
        need(not any((run/name).exists() for name in ('response-replication-claim.json','execution-start.json',
             'execution.json','raw.json','response-replication-result.json')) and not list(run.glob('trace-unified*')),
             'Only a wholly unstarted three-capture campaign can enter a new continuation')


def prepare(campaign_path,independent_review,queue_pid,queue_claim,matching_final,device_attributes,storage_scenario,output):
    campaign_ref=receipt(campaign_path);campaign=producer.audit(campaign_path);idle(campaign)
    review_ref=receipt(independent_review);review=checked_review(review_ref,campaign_ref)
    claim_ref=receipt(queue_claim);claim=read(claim_ref)
    need(claim.get('matching_manifest')==campaign['after_matching_manifest'],'Queue and kernel producer matching manifests differ')
    identity=queue_identity(queue_pid,claim_ref,matching_final)
    device_ref=receipt(device_attributes);verify(device_ref)
    storage_ref=receipt(storage_scenario);storage=read(storage_ref)
    need(storage.get('schema')=='gsdeploy.kernel_dag_replication_storage_scenario.v1'
         and storage.get('additional_CPU_reserve_bytes')==0 and storage.get('new_chain_peak_scenario_bytes',0)>0,
         'Observed native/export byte scenario required without additional arbitrary CPU reserve')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    sources=strict_references([campaign_ref,campaign,review_ref,review,claim_ref,claim,
        device_ref,storage_ref,storage,implementation_sources(__file__)])
    for ref in sources:verify(ref)
    result=dict(schema=SCHEMA,status='CPU_PREPARED_EXPLICIT_WAIT_EXECUTE_REQUIRED',campaign=campaign_ref,
        independent_review=review_ref,queue_identity=identity,queue_claim=claim_ref,
        queue_final_path=str(Path(queue_claim).resolve().parent/'report.json'),
        matching_final_path=str(Path(matching_final).resolve()),device_attributes=device_ref,
        storage_scenario=storage_ref,
        output=str(output/'run-v1'),sources=sources,no_retries=True,GPU_started=False,fit_started=False,
        requires_matching_fit_success=False,validation_labels_read=False,
        execution_order='ALL_THREE_NATIVE_LIFECYCLES_THEN_CPU_REVIEW_OCCUPANCY_INTAKE_REPEATED_ALIGNMENT')
    write(output/'manifest.json',result);return result


def audit(path):
    manifest=read(receipt(path))
    need(manifest.get('schema')==SCHEMA and manifest.get('status')=='CPU_PREPARED_EXPLICIT_WAIT_EXECUTE_REQUIRED'
         and manifest.get('no_retries') is True and manifest.get('validation_labels_read') is False
         and manifest.get('requires_matching_fit_success') is False,
         'Explicit nonretrying native-only matching continuation required')
    campaign=producer.audit(manifest['campaign']['path']);verify(manifest['campaign']);idle(campaign)
    checked_review(manifest['independent_review'],manifest['campaign']);claim=read(manifest['queue_claim'])
    need(claim.get('matching_manifest')==campaign['after_matching_manifest']
         and manifest['queue_identity']['claim']==manifest['queue_claim']
         and manifest['queue_identity']['final_path']==manifest['matching_final_path']
         and Path(claim['matching_output']).resolve()==Path(manifest['matching_final_path']).resolve().parent
         and Path(manifest['queue_final_path']).resolve()==Path(manifest['queue_claim']['path']).resolve().parent/'report.json',
         'Canonical pinned matching queue/final binding required')
    for ref in manifest['sources']:verify(ref)
    closure=implementation_sources(__file__)
    need(all(ref in manifest['sources'] for ref in closure),'Complete continuation implementation closure required')
    need(all(ref in manifest['sources'] for ref in strict_references([manifest['campaign'],campaign,
         manifest['independent_review'],read(manifest['independent_review']),manifest['queue_claim'],claim,
         manifest['device_attributes'],manifest['storage_scenario'],read(manifest['storage_scenario'])])),
         'Complete campaign/review/queue/device/storage source set required')
    return manifest,campaign


def load_publication(path):
    path=Path(path)
    if not path.is_file():return None
    try:
        payload=path.read_bytes();document=json.loads(payload)
    except (json.JSONDecodeError,UnicodeDecodeError):return None
    ref=dict(path=str(path.resolve()),sha256=hashlib.sha256(payload).hexdigest())
    return (ref,document) if receipt(path)==ref else None


def complete_matching(manifest,campaign):
    native=load_publication(manifest['matching_final_path']);queued=load_publication(manifest['queue_final_path'])
    if native is None or queued is None:return None
    final_ref,final=native;queue_ref,queue=queued
    producer.require_matching_completion(campaign,final_ref)
    need(queue.get('schema')==QUEUE_SCHEMA and queue.get('status')==QUEUE_STATUS
         and queue.get('matching_manifest')==campaign['after_matching_manifest']
         and queue.get('matching_report')==final_ref,'Exact successful matching queue final required')
    return final_ref,queue_ref


def wait_matching(manifest,campaign,check_sources,*,poll_seconds=30,identity_reader=queue_identity,pause=time.sleep):
    need(type(poll_seconds) in (int,float) and 0<poll_seconds<=60,'Bounded polling interval required')
    identity=manifest['queue_identity']
    while True:
        check_sources()
        try:current=identity_reader(identity['pid'],manifest['queue_claim'],manifest['matching_final_path'])
        except (FileNotFoundError,ProcessLookupError):current=None
        if current is not None:need(current==identity,'Matching queue PID/start ticks/command identity changed')
        completed=complete_matching(manifest,campaign)
        if completed is not None:return completed
        need(current is not None,'Matching queue exited before complete native/queue final; stop without retry')
        pause(poll_seconds)


def wait_capacity(manifest,required,check_sources,*,phase,poll_seconds=30,pause=time.sleep,disk_usage=shutil.disk_usage):
    need(type(required) is int and required>0,'Positive evidenced phase storage scenario required')
    while True:
        check_sources();free=disk_usage(Path(manifest['output']).parent).free
        if free>=required:return dict(phase=phase,free_bytes=free,required_scenario_bytes=required,
                                     extra_arbitrary_reserve_bytes=0)
        pause(poll_seconds)


def command(module,*args):
    return [sys.executable,'-m','tools.predictor_validation_20261027.'+module,*map(str,args)]


def cpu_stages(campaign_path,campaign,device_attributes,output):
    actions=[]
    for index,entry in enumerate(campaign['runs']):
        root=Path(output)/('capture-'+str(index));native=root/'native-review';occ=root/'occupancy';intake=root/'intake'
        common=dict(capture_id=entry['capture_id'])
        actions.extend([
            dict(common,name=str(index)+'-native-review',command=command('unified_corun_review_v8_1','review','--run',entry['run'],'--output',native),result=str(native/'report.json'),schema='gsdeploy.unified_corun_strict_review.v8_1',status='PASS'),
            dict(common,name=str(index)+'-occupancy',command=command('occupancy_declaration_audit_v2','--sqlite',native/'trace-unified.sqlite','--device-attributes',device_attributes,'--output',occ),result=str(occ/'audit.json'),schema='gsdeploy.arch_aware_occupancy_declaration_audit.v2',status=None),
            dict(common,name=str(index)+'-intake',command=command('unified_kernel_dag_intake_v1','--review',native/'report.json','--device-attributes',device_attributes,'--occupancy-audit',occ/'audit.json','--output',intake),result=str(intake/'report.json'),schema='gsdeploy.unified_kernel_dag_intake.v1',status='COMPLETE_NATIVE_MAPPING_PATH_DIAGNOSTICS_NO_FIT'),
            dict(common,name=str(index)+'-producer-review',command=command('kernel_dag_response_replication_producer_v1','review','--campaign',campaign_path,'--capture-id',entry['capture_id'],'--native-review',native/'report.json','--output',root/'producer-review.json'),result=str(root/'producer-review.json'),schema=producer.SCHEMA+'.review',status='PASS_CPU_MATRIX_NATIVE_BINDING_ONLY')])
    return actions


def validate_cpu_result(action,result):
    need(result.get('schema')==action['schema'] and (action['status'] is None or result.get('status')==action['status']),
         'CPU stage incomplete/unknown: '+action['name'])
    if action['name'].endswith('-occupancy'):
        need(result.get('sensitivity',{}).get('all_passed') is True and result.get('traces')
             and all(t.get('launch_count')==t.get('known_launch_count') and t.get('unknown_launch_count')==0 for t in result['traces']),
             'Exact full-source occupancy coverage required')
    if action['name'].endswith('-intake'):
        need(result.get('rejected_kernel_count')==0 and result.get('offered_requests')==20,
             'Complete twenty-request unified intake required')


def run(manifest_path,*,poll_seconds=30,run_command=subprocess.run):
    manifest,campaign=audit(manifest_path);root=Path(manifest['output']);root.mkdir(parents=True,exist_ok=False)
    result=dict(schema=SCHEMA+'.run',manifest=receipt(manifest_path),status='WAITING_FOR_PINNED_MATCHING_NATIVE_FINAL',
        completed_captures=[],completed_cpu_stages=[],GPU_started=False,fit_started=False,automatic_retry=False,
        validation_labels_read=False,numerical_qualification=False)
    write(root/'claim.json',dict(manifest=receipt(manifest_path),queue_identity=manifest['queue_identity'],no_retries=True))
    try:
        refs=strict_references([receipt(manifest_path),manifest])
        with SourceChangeGuard(refs) as guard:
            final_ref,queue_ref=wait_matching(manifest,campaign,guard.check,poll_seconds=poll_seconds)
            guard.check();idle(campaign)
            with SourceChangeGuard(strict_references([refs,final_ref,read(final_ref),queue_ref])) as final_guard:
                storage=read(manifest['storage_scenario'])
                required=max(storage['new_chain_peak_scenario_bytes'],
                    storage['original_GPU_collector_floor_bytes']+storage['original_GPU_collector_preflight_scenario_bytes'])
                capacity=wait_capacity(manifest,required,final_guard.check,phase='BEFORE_THREE_NATIVE_CAPTURES',poll_seconds=poll_seconds)
                write(root/'native-capacity.json',capacity)
                for index,entry in enumerate(campaign['runs']):
                    final_guard.check();producer.audit(manifest['campaign']['path'])
                    action=dict(capture_id=entry['capture_id'],command=command('kernel_dag_response_replication_producer_v1','execute',
                        '--campaign',manifest['campaign']['path'],'--capture-id',entry['capture_id'],
                        '--after-receipt',final_ref['path'],'--execute'))
                    write(root/('capture-'+str(index)+'-started.json'),action);result['GPU_started']=True
                    with (root/('capture-'+str(index)+'.log')).open('x') as stream:
                        process=run_command(action['command'],stdout=stream,stderr=subprocess.STDOUT,check=False)
                    final_guard.check();need(process.returncode==0,'Native capture child failed; never retry a started lifecycle')
                    execution_ref=receipt(Path(entry['run'])/'response-replication-result.json');execution=read(execution_ref)
                    need(execution.get('schema')==producer.SCHEMA+'.execution' and execution.get('capture_id')==entry['capture_id']
                         and execution.get('status')=='COLLECTED' and execution.get('outer_source_guard_passed') is True,
                         'Native lifecycle incomplete; preserve prefix and stop')
                    result['completed_captures'].append(dict(capture_id=entry['capture_id'],execution=execution_ref))
                    write(root/('capture-'+str(index)+'-complete.json'),result['completed_captures'][-1])
                # All three native lifecycles finish before expensive CPU exports.
                remaining=storage['new_chain_peak_scenario_bytes']-storage['estimated_native_capture_bytes']
                capacity=wait_capacity(manifest,remaining,final_guard.check,phase='BEFORE_CPU_EXPORTS_AND_SINGLE_TEMPORARY_REBUILD',poll_seconds=poll_seconds)
                write(root/'cpu-capacity.json',capacity)
                for action in cpu_stages(manifest['campaign']['path'],campaign,manifest['device_attributes']['path'],root):
                    final_guard.check();Path(action['result']).parent.parent.mkdir(parents=True,exist_ok=True)
                    need(not Path(action['result']).exists() and
                         (action['name'].endswith('-producer-review') or not Path(action['result']).parent.exists()),
                         'Never reuse a started CPU stage output')
                    write(root/(action['name']+'-started.json'),action)
                    with (root/(action['name']+'.log')).open('x') as stream:
                        process=run_command(action['command'],stdout=stream,stderr=subprocess.STDOUT,check=False,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                    final_guard.check();need(process.returncode==0,'CPU stage failed; no automatic retry: '+action['name'])
                    result_ref=receipt(action['result']);validate_cpu_result(action,read(result_ref))
                    result['completed_cpu_stages'].append(dict(name=action['name'],result=result_ref))
                    write(root/(action['name']+'-complete.json'),result['completed_cpu_stages'][-1])
                entries=[]
                for index,_ in enumerate(campaign['runs']):
                    cap=root/('capture-'+str(index))
                    entries.append(dict(producer_review=receipt(cap/'producer-review.json'),intake_report=receipt(cap/'intake/report.json'),
                        occupancy_audit=receipt(cap/'occupancy/audit.json'),device_attributes=manifest['device_attributes']))
                design_ref=write(root/'repeated-design.json',dict(schema=repeated.SCHEMA+'.design',training_only=True,
                    validation_labels_read=False,captures=entries))
                # CPU adapter reauthenticates all SQL intakes and never fits.
                alignment=repeated.align(design_ref['path'],root/'repeated-alignment')
                final_guard.check()
                result.update(status='COMPLETE_THREE_NATIVE_CAPTURES_AND_CPU_DIAGNOSTICS_NO_FIT',matching_final=final_ref,
                    queue_final=queue_ref,repeated_alignment=receipt(root/'repeated-alignment/report.json'),
                    template_alignment_status=alignment['status'],actual_rank_or_prediction_accuracy_qualified=False)
    except Exception as exc:
        result.update(status='STOPPED_NO_RETRY_PREFIX_PRESERVED',error=type(exc).__name__+': '+str(exc))
    write(root/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare')
    for field in ('campaign','independent-review','queue-claim','matching-final','device-attributes','storage-scenario','output'):p.add_argument('--'+field,required=True)
    p.add_argument('--queue-pid',required=True,type=int)
    p=sub.add_parser('audit');p.add_argument('--manifest',required=True)
    p=sub.add_parser('run');p.add_argument('--manifest',required=True);p.add_argument('--execute',required=True,action='store_true')
    args=parser.parse_args()
    if args.op=='prepare':result=prepare(args.campaign,args.independent_review,args.queue_pid,args.queue_claim,args.matching_final,args.device_attributes,args.storage_scenario,args.output)
    elif args.op=='audit':result=audit(args.manifest)[0]
    else:result=run(args.manifest)
    print(json.dumps({k:v for k,v in result.items() if k in ('schema','status','error','GPU_started','fit_started')}))


if __name__=='__main__':main()
