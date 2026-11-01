"""Fixed-order CPU-prepared continuation; reuse reviewed execution boundaries.

An existing attempt without execution.json is WAITING, never an invitation to
retry. Every invocation writes a fresh exclusive log/output directory.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

from . import twoFIFO_owned_collect_v2 as collector
from . import twoFIFO_composable_macros_v2 as macros
from .twoFIFO_owned_training_review_v2 import review,require
from .stage_response_features_v1 import receipt,checked
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_fixed_campaign_continuation.v1'
COMPLETE='PASS_COMPLETE_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION'


def write_new(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def prepare(campaign_path,output):
    campaign=collector.audit(campaign_path);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    # Existing native outputs remain at their original paths and are only
    # inspected. All new rechecks/macros are saved below each driver run.
    attempts=[]
    for ref in campaign['attempts']:
        plan=checked(ref);parent=Path(ref['path']).parent
        attempts.append(dict(plan=ref,attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],
            collection=str(parent/'collection'),existing_review=str(parent/'native-review01/report.json'),
            existing_macros=str(parent/'macros-v2.json')))
    sources=strict_references([receipt(campaign_path),campaign['attempts'],implementation_sources(__file__)])
    manifest=dict(schema=SCHEMA,status='CPU_PREPARED_EXPLICIT_EXECUTE_REQUIRED',campaign=receipt(campaign_path),
        attempts=attempts,sources=sources,no_retries=True,supervisor_budget=collector.supervisor.BUDGET,
        preflight_free_bytes_required=collector.supervisor.PREFLIGHT,existing_unfinished_action='READ_ONLY_WAITING',
        startup_attempts=12,whole_block_replication_units=3,GPU_started=False,validation_labels_read=False)
    write_new(output/'manifest.json',manifest);audit(output/'manifest.json');return receipt(output/'manifest.json')


def audit(path):
    manifest=checked(receipt(path))
    require(manifest['schema']==SCHEMA and manifest['status']=='CPU_PREPARED_EXPLICIT_EXECUTE_REQUIRED'
        and manifest['no_retries'] is True and manifest['existing_unfinished_action']=='READ_ONLY_WAITING'
        and manifest['supervisor_budget']==collector.supervisor.BUDGET
        and manifest['preflight_free_bytes_required']==collector.supervisor.PREFLIGHT,'Exact original budget and no-retry continuation required')
    campaign=collector.audit(manifest['campaign']['path']);require(checked(manifest['campaign'])==campaign,'Campaign source drift')
    expected=[]
    for ref in campaign['attempts']:
        plan=checked(ref);parent=Path(ref['path']).parent
        expected.append(dict(plan=ref,attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],collection=str(parent/'collection'),
            existing_review=str(parent/'native-review01/report.json'),existing_macros=str(parent/'macros-v2.json')))
    require(manifest['attempts']==expected and len(expected)==12,'Exact twelve planned attempts in immutable order required')
    require(manifest['sources']==strict_references([manifest['campaign'],campaign['attempts'],implementation_sources(__file__)]),
        'Continuation implementation/source closure drift')
    for ref in manifest['sources']:require(receipt(ref['path'])==ref,'Continuation source changed')
    return manifest


def completed_execution(execution,plan_ref,plan):
    require(execution.get('profile_execution')==dict(schema=collector.SCHEMA+'.execution_binding',
        plan=plan_ref,execution_id=plan['execution_id'],attempt_id=plan['attempt_id'])
        and execution.get('budget')==collector.supervisor.BUDGET
        and execution.get('status')=='CHILD_EXITED_REQUIRES_PROFILE_REVIEW'
        and execution.get('child_reaped') is True and execution.get('live_group_pids')==[]
        and execution.get('experiment_lock_held_through_group_exit') is True
        and execution.get('all_owned_descendants_exited_and_reaped') is True
        and execution.get('final_source_audit')=='PASS'
        and execution.get('watchdog',{}).get('status')=='EXITED'
        and execution['watchdog'].get('return_code')==0,'Failed/incomplete original supervisor execution; no retry')


def run(manifest_path,output,*,execute=False):
    """Native execute is off by default; dependencies remain easy to CPU mock."""
    manifest_ref=receipt(manifest_path);manifest=audit(manifest_path)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    # Serialize driver invocations for this fixed campaign without inventing a
    # GPU lease; the original supervisor remains the sole GPU lock/cleanup owner.
    lock=Path(manifest['campaign']['path']).parent/'continuation.lock';log=output/'events.jsonl';finished=[];launches=[]
    with lock.open('a') as guard,log.open('x') as events:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def event(kind,**fields):
            row=dict(event=kind,time_unix_ns=time.time_ns(),**fields);events.write(json.dumps(row,allow_nan=False)+'\n');events.flush()
        def stop(status,**fields):
            result=dict(schema=SCHEMA+'.run',status=status,manifest=manifest_ref,completed_attempts=finished,
                no_retries=True,execute_invocations=len(launches),
                validation_labels_read=False,**fields)
            event('stop',status=status,**fields);write_new(output/'report.json',result);return result
        event('begin',manifest=manifest_ref,execute=execute)
        for item in manifest['attempts']:
            try:
                require(receipt(manifest_path)==manifest_ref,'Continuation manifest changed during run');audit(manifest_path)
                plan=checked(item['plan']);collection=Path(item['collection']);execution_path=collection/'execution.json';launched=False
                if collection.exists() and not execution_path.is_file():
                    return stop('WAITING_EXISTING_ATTEMPT_FINAL_RECEIPT',attempt_id=item['attempt_id'],collection=str(collection))
                if not collection.exists():
                    if not execute:return stop('READY_EXPLICIT_EXECUTE_REQUIRED',attempt_id=item['attempt_id'])
                    free=shutil.disk_usage(collection.parent).free
                    require(free>=collector.supervisor.PREFLIGHT,'Original18GiB filesystem preflight insufficient; stopping without launch')
                    event('launch',attempt_id=item['attempt_id'],free_bytes_before_execute=free)
                    launches.append(item['attempt_id'])
                    collector.execute(item['plan']['path'],collection);launched=True
                    require(execution_path.is_file(),'Execute returned without final supervisor receipt; no retry')
                execution_ref=receipt(execution_path);execution=checked(execution_ref)
                completed_execution(execution,item['plan'],plan)
                check_dir=output/item['attempt_id'];check_dir.mkdir()
                review_ref=review(collection/'raw.json',check_dir/'native-review',execution_path)
                native_review=checked(review_ref)
                require(native_review.get('status')==COMPLETE and native_review.get('training_feature_extraction_eligible') is True
                    and native_review.get('issues')==[],'Native training NOT_ELIGIBLE; stop all following attempts')
                prior_review=Path(item['existing_review'])
                if prior_review.exists():require(checked(receipt(prior_review))==native_review,'Existing native review must rederive exactly')
                macro_doc=macros.extract(review_ref['path'])
                prior_macros=Path(item['existing_macros'])
                if prior_macros.exists():
                    old=checked(receipt(prior_macros))
                    # Macro source receipts differ only by the new, identical
                    # review path. Recheck the original review through extract.
                    require(prior_review.exists() and macros.extract(str(prior_review))==old,'Existing exact macros must rederive')
                    for key in ('requests','macros','measured_requests','capture_id','attempt_id','execution_id'):
                        require(old[key]==macro_doc[key],'Existing macro values differ from driver rederivation')
                write_new(check_dir/'macros-v2.json',macro_doc)
                require(receipt(execution_path)==execution_ref,'Execution receipt changed during recheck')
                finished.append(dict(attempt_id=item['attempt_id'],execution=execution_ref,training_review=review_ref,
                    macros=receipt(check_dir/'macros-v2.json'),launched_by_this_run=launched))
                event('complete',**finished[-1])
            except BaseException as exc:
                return stop('STOPPED_NO_RETRY',attempt_id=item['attempt_id'],error=type(exc).__name__+': '+str(exc))
        return stop('COMPLETE_ALL_TWELVE_NATIVE_ATTEMPTS_REVIEWED')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--campaign',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('run');p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    result=prepare(args.campaign,args.output) if args.op=='prepare' else run(args.manifest,args.output,execute=args.execute)
    print(json.dumps(result))
