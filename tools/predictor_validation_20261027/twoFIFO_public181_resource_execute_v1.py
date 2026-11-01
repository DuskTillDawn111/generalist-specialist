"""New exact-public181 native resource executor; preparation is CPU only.

Explicit derivative of the reviewed matching executor. Eight fresh startups,
settled actual eighteen-request priming, fixed300s pre-cohort budget,90+120
window, original sender/worker FIFO and reviewed monitor/cleanup. No fit.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import uuid

from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.schema import MeasurementConfig
from gsdeploy.forecast_errors import UnknownEvidence
from . import twoFIFO_public181_resource_campaign_v1 as campaign
from . import twoFIFO_matching_execute_v1 as matching
from . import twoFIFO_settled_history_initial_v1 as settled
from .twoFIFO_native_quality_labels_v1 import prepared_references
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import receipt,checked
from .twoFIFO_training_manifest_v1 import require,digest,OWNERS
from .profile_owned_collect_v3 import StartupBoardMonitor,memory_ownership,MONITORING_SCOPE
from .twoFIFO_owned_collect_v3 import (ActivityLedger,atomic_save,initial_rows,history_snapshot,
    wait_until,audit_activity,finish_collection,ACTIVITY_FILE,ACTIVITY_SCHEMA,TIME_DOMAIN)

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='gsdeploy.twoFIFO_public181_resource_owned_training.v1'
RAW_SCHEMA=SCHEMA+'.capture'
CONT_SCHEMA=SCHEMA+'.continuation'
MEASUREMENT_NS=90_000_000_000
DRAIN_NS=120_000_000_000
LEAD_NS=300_000_000_000
SETTLE_NS=10_000_000_000
QUALITY_CATALOG=ROOT/'artifacts/static-study-v6-20261021/inputs/catalog.json'
supervisor=matching.supervisor
_verify=campaign.verify
_write=campaign.save
initial_priming_rows=matching.initial_priming_rows
planned_priming_semantics=matching.planned_priming_semantics


def executable_sources():
    here=Path(__file__)
    return implementation_sources(here,here.with_name('twoFIFO_public181_resource_review_v1.py'),
        here.with_name('twoFIFO_public181_resource_intake_v1.py'),
        here.with_name('semantic_progress_worker_v1.py'))


def _build(campaign_path,output):
    parent_ref=receipt(campaign_path);parent=campaign.audit(campaign_path)
    require(parent['target_precohort_lead_seconds']==300,
            'This executor implements only the explicit300s protocol')
    output=Path(output).resolve();implementation=executable_sources()
    frozen=[dict(original_path=source['path'],frozen=receipt(output/'source-freeze'/Path(source['path']).relative_to(ROOT)))
            for source in implementation]
    require(all(a['sha256']==b['frozen']['sha256'] for a,b in zip(implementation,frozen)),
            'Complete native executor/reviewer/intake source freeze required')
    source_transfer=dict(schema=SCHEMA+'.source_transfer',parent_campaign=parent_ref,
        parent_preparation_protocol_identity=parent['collection_protocol_identity'],
        target_sources=implementation,target_source_freeze=frozen,
        worker_registry=parent['source_transfer']['planned_worker_registry'],
        worker_registry_sha256=parent['source_transfer']['planned_worker_registry_sha256'],
        collection_contract=campaign.COLLECTION,precohort_lead_ns=LEAD_NS,prior_READY_settle_timeout_ns=SETTLE_NS,
        quality_catalog=receipt(QUALITY_CATALOG),
        prior_physical_identity_reused=False,
        native_authentication_basis='EXECUTED_REGISTRY_STDOUT_READY_LIVE_OWNERSHIP_AND_LIFECYCLE_REVIEW',
        preparation_is_native_authentication=False)
    protocol=digest(source_transfer)
    sources=strict_references([parent_ref,parent['sources'],parent['training_attempts'],implementation,frozen,
        source_transfer['quality_catalog']])
    for source in sources:_verify(source)
    reserved={a['execution_id'] for a in parent['training_attempts']+parent['independent_validation_references']}
    attempts=[]
    for base in parent['training_attempts']:
        original=checked(_verify(base['plan']));aid=original['attempt_id']+'-native-v1'
        execution_id=uuid.uuid5(uuid.NAMESPACE_URL,SCHEMA+':'+original['execution_id']+':'+protocol).hex
        require(execution_id not in reserved,'Native execution identity must be fresh')
        reserved.add(execution_id)
        plan=copy.deepcopy(original)
        plan.update(schema=SCHEMA+'.attempt',parent_training_plan=base['plan'],attempt_id=aid,
            quality_catalog=source_transfer['quality_catalog'],
            execution_id=execution_id,source_bindings_schema=SCHEMA+'.source_transfer',
            source_transfer=source_transfer,collection_protocol_identity=protocol,
            parent_collection_protocol_identity=original['collection_protocol_identity'],
            source_freeze=frozen,sources=sources,supervisor_budget=supervisor.BUDGET,
            preflight_free_bytes_required=supervisor.PREFLIGHT,
            execute_entrypoint='tools.predictor_validation_20261027.twoFIFO_public181_resource_execute_v1.collect',
            native_executor_entrypoint='tools.predictor_validation_20261027.twoFIFO_public181_resource_execute_v1.collect',
            collector_timing_policy=dict(no_mutating_checkpoint_during_case=True,
                actual_prefix_and_checkpoint_before_epoch=True),
            training_labels_read=False,validation_labels_read=False,GPU_started=False)
        plan['priming']=[dict(row,request_id=aid+':priming:'+str(i)) for i,row in enumerate(original['priming'])]
        prepared_references(plan,plan['quality_catalog'])
        require(planned_priming_semantics(plan,checked(plan['context']))==plan['priming_semantics'],
                'Native plan must preserve exact original priming semantics')
        plan['execution_context_sha256']=digest(plan)
        folder=output/aid
        attempts.append(dict(plan=plan,path=folder/'plan.json',attempt_id=aid,
            execution_id=execution_id,capture_id=plan['capture_id'],candidate_id=plan['candidate_id'],
            collection=str(folder/'collection'),claim=str(folder/'public181-resource-attempt-claim.json')))
    return dict(parent_ref=parent_ref,parent=parent,source_transfer=source_transfer,
        collection_protocol_identity=protocol,sources=sources,attempts=attempts)


def _manifest(built):
    return dict(schema=SCHEMA+'.manifest',status='CPU_PREPARED_NATIVE_RESOURCE_TRAINING_EXPLICIT_EXECUTE_REQUIRED',
        parent_campaign=built['parent_ref'],attempts=[dict(plan=receipt(a['path']),
            **{k:a[k] for k in ('attempt_id','execution_id','capture_id','candidate_id','collection','claim')})
            for a in built['attempts']],source_transfer=built['source_transfer'],
        collection_protocol_identity=built['collection_protocol_identity'],sources=built['sources'],
        expected_training_attempts=8,training_blocks=built['parent']['training_blocks'],
        independent_validation_references=built['parent']['independent_validation_references'],
        measurement_ns=MEASUREMENT_NS,drain_ns=DRAIN_NS,precohort_lead_ns=LEAD_NS,
        startup_occurrences_per_attempt=1,priming_requests_per_attempt=18,
        training_offered_requests=1448,training_priming_requests=144,
        source_transfer_must_be_checked_from_actual_execution=True,
        native_backend_implemented=True,validation_native_backend_requires_new_preparation=True,
        no_retries=True,automatic_fit=False,numerical_fit_run=False,
        training_labels_read=False,validation_labels_read=False,GPU_started=False,
        main_queue_provider_eligible=False,final_gain_eligible=False)


def prepare(campaign_path,output):
    campaign.audit(campaign_path);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    for source in executable_sources():
        target=output/'source-freeze'/Path(source['path']).relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(source['path']).read_bytes())
        require(receipt(target)['sha256']==source['sha256'],'Source changed during freeze')
    built=_build(campaign_path,output)
    for attempt in built['attempts']:
        attempt['path'].parent.mkdir();_write(attempt['path'],attempt['plan'])
    result=_write(output/'manifest.json',_manifest(built));audit(result['path']);return result


def audit(path):
    manifest=checked(receipt(path))
    require(manifest.get('schema')==SCHEMA+'.manifest','Exact public181 resource executor manifest required')
    _verify(manifest['parent_campaign'])
    built=_build(manifest['parent_campaign']['path'],Path(path).resolve().parent)
    for attempt in built['attempts']:
        require(checked(receipt(attempt['path']))==attempt['plan'],
                'Native resource plan/public calendar/priming/source identity changed')
    require(manifest==_manifest(built),'Complete native resource manifest drift')
    return manifest


def audit_plan(path):
    manifest=audit(Path(path).resolve().parent.parent/'manifest.json')
    require(receipt(path) in [item['plan'] for item in manifest['attempts']], 'Canonical native resource plan required')
    return checked(receipt(path))


def settled_priming(plan,snapshot,epoch):
    rows=snapshot['request_records']
    require([r['request_id'] for r in rows]==[r['request_id'] for r in plan['priming']],
            'Actual full eighteen-request prior history required; no target rows')
    context=checked(plan['context'])
    require(all(snapshot['ready'][owner].get('spec_sha256')==
        context['worker_bindings'][owner.split('#')[0]]['spec_sha256'] for owner in OWNERS),
        'Actual READY worker specification differs from source-bound worker registry')
    from .natural_macro_resources_candidate_v1 import priming_identity
    require(priming_identity(dict(priming_request_records=rows),context)==plan['priming_semantics']['sha256'],
            'Actual successful priming input contents differ')
    producers={owner:plan['execution_id']+':'+owner+':stdout:pid'+str(snapshot['ready'][owner]['pid'])
               for owner in OWNERS}
    return settled.from_snapshot(snapshot,cutoff_ns=snapshot['snapshot_time_ns'],
        forecast_origin_ns=epoch,owner_producers=producers)


def dispatch_case(plan,case,runtime,pool,raw,*,clock_ns,wait_until_ns,checkpoint,monitor,activity):
    """Independent timed enqueue; every CPU copy/write precedes the fixed epoch."""
    with activity.span('precase_history_snapshot',case_id=case['case_id'],phase=monitor.phase):
        settle_begin=clock_ns()
        while True:
            snapshot=history_snapshot(runtime,raw,clock_ns)
            anchor=snapshot['snapshot_time_ns'];epoch=anchor+LEAD_NS
            try:
                seed=settled_priming(plan,snapshot,epoch)
                break
            except UnknownEvidence:
                if clock_ns()-settle_begin>=SETTLE_NS:raise
                monitor.check();wait_until_ns(clock_ns()+10_000_000)
        raw['case_histories'].append(dict(case_id=case['case_id'],prior_history_snapshot=snapshot))
        raw['precohort_training_prefix']=dict(anchor_ns=anchor,epoch_ns=epoch,lead_ns=LEAD_NS,
            settled_seed=seed,source='ACTUAL_EIGHTEEN_PRIMING_AND_READY_PREFIX',
            target_future_labels_used=False,memory_state_inferred_from_seed=False)
        raw['case_preparations'].append(dict(case_id=case['case_id'],
            public_arrivals_sha256=case['public_arrivals_sha256'],public_arrivals=copy.deepcopy(case['arrivals']),
            prior_history_index=0,prepared_before_epoch=True))
    raw['checkpoint_context']=dict(kind='precase',case_id=case['case_id']);checkpoint()
    raw['case_preparations'][-1]['durable_checkpoint_return_ns']=clock_ns()
    require(clock_ns()<epoch,'Pre-cohort snapshot/checkpoint crossed fixed300s budget; cohort not sent')
    wait_until_ns(epoch);monitor.set_phase('measurement')
    end=epoch+MEASUREMENT_NS;deadline=end+DRAIN_NS
    rows=raw['request_records']
    with runtime.lock:
        for row in rows:row.update(scheduled_time_ns=epoch+row['arrival_ns'],status='planned')
        observed=raw['cases'][0]
        observed.update(status='running',case_epoch_ns=epoch,observation_start_ns=epoch,
            measurement_end_ns=end,fixed_drain_deadline_ns=deadline)
        raw['measurement_windows'].append(dict(case_id=case['case_id'],start_ns=epoch,end_ns=end,
            fixed_drain_deadline_ns=deadline));raw['observation_start_ns']=epoch
    for row in sorted(rows,key=lambda r:(r['scheduled_time_ns'],r['arrival_index'])):
        wait_until_ns(row['scheduled_time_ns']);monitor.check()
        with runtime.lock:row['arrival_observed_time_ns']=clock_ns()
        if clock_ns()>=deadline:
            with runtime.lock:row.update(status='incomplete',error='arrival missed fixed drain; offered retained')
        else:
            try:runtime.enqueue(row,pool[row['pool_id']]['sample'])
            except Exception as exc:
                with runtime.lock:row.update(status='error',error=repr(exc))
    wait_until_ns(end);monitor.set_phase('drain')
    runtime.drain(rows,max(0,deadline-clock_ns())/1e9);wait_until_ns(deadline)
    with runtime.lock:
        for row in rows:
            if row['status'] in ('planned','queued','submitted'):
                row.update(status='incomplete',error='fixed drain reached; offered retained')
            row['RESULT_reader_within_fixed_drain']=(type(row.get('completion_time_ns')) is int
                and row['scheduled_time_ns']<=row['completion_time_ns']<=deadline)
        observed.update(status='completed' if all(r['status']=='completed' and r['RESULT_reader_within_fixed_drain']
            for r in rows) else 'incomplete',observation_finalized_ns=clock_ns(),
            fixed_cutoff_overshoot_ns=max(0,clock_ns()-deadline))
    wait_until_ns(deadline+1);monitor.sample();monitor.check()
    raw['checkpoint_context']=dict(kind='postdrain',case_id=case['case_id']);checkpoint()
    return observed


def collect(plan_path,output,*,lease_fd,runtime_factory=DeploymentRuntime,monitor_factory=StartupBoardMonitor,
            clock_ns=time.perf_counter_ns,wait_until_ns=None,ownership_check=memory_ownership):
    supervisor.validate_lease(lease_fd);plan=audit_plan(plan_path);output=Path(output)
    require(output.resolve()==Path(plan_path).resolve().parent/'collection','Canonical native training output required')
    output.mkdir(parents=True,exist_ok=True)
    require(not (output/'raw.json').exists(),'No overwrite/retry of a collection attempt')
    context=checked(plan['context']);pool={r['pool_id']:r for r in (json.loads(l) for l in Path(plan['prepared_pool']['path']).read_text().splitlines())}
    registry=campaign.planned_worker_registry(context)
    runtime=runtime_factory(MeasurementConfig(**plan['runtime_config']),registry);monitor=None
    executed=dict(attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],cases=plan['cases'])
    atomic_save(output/'executed-cases.json',executed)
    injected=(runtime_factory is not DeploymentRuntime or monitor_factory is not StartupBoardMonitor
        or clock_ns is not time.perf_counter_ns or wait_until_ns is not None or ownership_check is not memory_ownership)
    raw=dict(schema=RAW_SCHEMA,collector_schema=RAW_SCHEMA,kind='CPU_TEST_ONLY' if injected else 'measured',
        backend='CPU_TEST_ONLY' if injected else 'native',split='training',
        status='starting',execution_id=plan['execution_id'],execution_context_sha256=plan['execution_context_sha256'],
        capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],plan=receipt(plan_path),context=plan['context'],
        prepared_pool=plan['prepared_pool'],executed_cases=receipt(output/'executed-cases.json'),
        request_records=initial_rows(plan,pool),priming_request_records=initial_priming_rows(plan,pool),
        public_case_calendars=copy.deepcopy(plan['cases']),priming_public_calendar=copy.deepcopy(plan['priming']),
        cases=[dict(case_id=c['case_id'],status='NOT_STARTED',offered=c['offered'],public_arrivals_sha256=c['public_arrivals_sha256'],request_ids=[r['request_id'] for r in c['arrivals']]) for c in plan['cases']],
        case_histories=[],case_preparations=[],measurement_windows=[],board_samples=[],telemetry_errors=[],errors=[],runtime_errors=[],lifecycle_boundaries=[],
        no_mutating_checkpoint_during_case=True,
        checkpoint_spans_file=ACTIVITY_FILE,checkpoint_spans_schema=ACTIVITY_SCHEMA,
        checkpoint_spans_execution_id=plan['execution_id'],checkpoint_time_domain=TIME_DOMAIN,
        checkpoint_context=dict(kind='before_startup',case_id=None),
        monitoring_scope=MONITORING_SCOPE,shutdown_memory_sampled=False,
        owned_peak_scope='sampled_live_declared_launcher_descendants_across_PGID_startup_through_final_drain_before_worker_close_not_instantaneous',
        performance_scope='unprofiled_original_sender_stdin_FIFO_with_semantic_phase_markers',
        source_bindings=plan['sources'],source_freeze=plan['source_freeze'],
        U=None,main_queue_provider_eligible=False,native_CUDA_API_numeric_authentication='UNQUALIFIED',
        training_feature_extraction_eligible=False,all_offered_registered_before_startup=True,
        public181_resource_executor_sources=executable_sources(),
        collection_protocol_identity=plan['collection_protocol_identity'],executed_worker_registry=registry)
    activity=ActivityLedger(output,plan['execution_id'],clock_ns)
    def checkpoint():
        context=raw['checkpoint_context'];now=clock_ns()
        if any(w['start_ns']<=now<=w['fixed_drain_deadline_ns'] for w in raw['measurement_windows']):
            # Interrupted/failed runs still retain their evidence after original
            # cleanup, but can never receive a successful timing-policy claim.
            raw['no_mutating_checkpoint_during_case']=False
        kwargs=dict(case_id=context['case_id'],phase=getattr(monitor,'phase','before_startup'),stage=context['kind'])
        with activity.span('checkpoint',**kwargs):
            with activity.span('checkpoint_locked_snapshot',**kwargs):
                with runtime.lock:
                    if monitor is not None:
                        with monitor._sample_lock:
                            raw['board_samples']=copy.deepcopy(monitor.samples);raw['telemetry_errors']=copy.deepcopy(monitor.errors)
                    raw['runtime_errors']=copy.deepcopy(runtime.errors)
                    snapshot=copy.deepcopy(raw)
            with activity.span('checkpoint_serialization_and_atomic_save',**kwargs):
                atomic_save(output/'raw.json',snapshot)
        activity.persist()
    def wait(target):
        if wait_until_ns is not None:return wait_until_ns(target)
        return wait_until(target,clock_ns,monitor)
    try:
        checkpoint();monitor=monitor_factory(runtime);monitor.sample();monitor.check();monitor.start()
        raw['lifecycle_boundaries'].append(dict(boundary='startup_begin',time_ns=clock_ns()));runtime.start()
        raw['ready_ownership_evidence']={}
        for alias in OWNERS:
            pid=runtime.ready[alias].get('pid');require(type(pid) is int,'Owned live READY pid required')
            evidence=ownership_check({pid:0.},runtime,monitor.collector_pid,monitor.process_group,ownership=monitor.ownership)
            require(evidence['owned_aliases']==[alias] and not evidence['unresolved_nvml_pids'] and not evidence['foreign_pids'],'Uncertified READY owner')
            raw['ready_ownership_evidence'][alias]=evidence['ownership_evidence']
        with runtime.lock:raw['ready']=copy.deepcopy(runtime.ready)
        raw['lifecycle_boundaries'].append(dict(boundary='startup_ready',time_ns=clock_ns()));monitor.set_phase('priming')
        raw['lifecycle_boundaries'].append(dict(boundary='priming_begin',time_ns=clock_ns()))
        for item,row in zip(plan['priming'],raw['priming_request_records']):
            row['scheduled_time_ns']=clock_ns()
            runtime.enqueue(row,pool[item['pool_id']]['sample']);runtime.drain([row],300)
            require(row['status']=='completed','Public181 resource priming failed')
        raw['lifecycle_boundaries'].append(dict(boundary='priming_end',time_ns=clock_ns()))
        raw['checkpoint_context']=dict(kind='postpriming',case_id=None);checkpoint()
        for case in plan['cases']:
            result=dispatch_case(plan,case,runtime,pool,raw,clock_ns=clock_ns,wait_until_ns=wait,
                checkpoint=checkpoint,monitor=monitor,activity=activity)
            raw['all_requests_completed']=result['status']=='completed'
        audit_activity(raw,activity)
        require(raw['no_mutating_checkpoint_during_case'],'Checkpoint crossed a protected case interval')
        raw['status']='completed'
        raw['fixed_resource_observation_complete']=True
        raw['observation_end_ns']=clock_ns()
        raw['lifecycle_boundaries'].append(dict(boundary='measurement_and_drain_end',time_ns=raw['observation_end_ns']))
    except BaseException as exc:
        raw['status']='failed';raw['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        # Shared reviewed cleanup stops monitoring and takes its final sample
        # while workers are live, then closes workers. Final save is atomic.
        finish_collection(runtime,monitor,raw,checkpoint,clock_ns)
    return raw


def execute(plan_path, output, *, manifest_path):
    # Only the receipt-checked snapshot is read before the watch is armed.
    # Every full audit and generated-plan read occurs under this outer guard.
    manifest_ref=receipt(manifest_path);snapshot=checked(manifest_ref)
    watched=strict_references([manifest_ref,snapshot['sources'],snapshot['attempts']])
    with SourceChangeGuard(watched) as source_guard:
        source_guard.check();_verify(manifest_ref)
        manifest=audit(manifest_path)
        require(manifest==snapshot,'Public181 resource manifest snapshot changed before guarded audit')
        plan_ref=receipt(plan_path)
        item=next(a for a in manifest['attempts'] if a['plan']==plan_ref)
        plan=audit_plan(plan_path)
        _verify(plan_ref);require(checked(plan_ref)==plan,'Guarded public181 resource plan read changed')
        require(Path(output).resolve()==Path(item['collection']),'Canonical public181 resource collection path required')
        require(not Path(output).exists(),'Attempt already started; no retry of success/failure/unfinished collection')
        require(not Path(item['claim']).exists(),'Attempt already claimed; no retry even if launch failed')
        supervisor.public_environment_gate();source_guard.check()
        _write(item['claim'],dict(schema=SCHEMA+'.claim',manifest=manifest_ref,plan=plan_ref,
            execution_id=plan['execution_id'],attempt_id=plan['attempt_id'],no_retries=True))
        claim_ref=receipt(item['claim'])
        command=[sys.executable,'-m','tools.predictor_validation_20261027.twoFIFO_public181_resource_execute_v1',
            '_child','--manifest',str(Path(manifest_path).resolve()),'--plan',str(Path(plan_path).resolve()),
            '--output',str(Path(output).resolve()),'--execute']
        # Claim is created only after canonical plan authentication; watch it
        # separately while the outer source/ALL-plan watch remains live.
        with SourceChangeGuard([claim_ref]) as claim_guard:
            def gate():
                source_guard.check();claim_guard.check();_verify(manifest_ref)
                audit(manifest_path);audit_plan(plan_path)
                supervisor.public_environment_gate()
                source_guard.check();claim_guard.check()
            gate()
            result=supervisor.supervise(command,output,
                environment={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                before_launch=gate,after_run=gate,metadata=dict(schema=SCHEMA+'.execution_binding',
                    manifest=manifest_ref,plan=plan_ref,execution_id=plan['execution_id'],attempt_id=plan['attempt_id'],
                    raw_schema=RAW_SCHEMA,claim=claim_ref,downstream_adapter_required=True,automatic_fit=False))
            source_guard.check();claim_guard.check()
            return result


def completed_execution(execution, manifest_ref, item):
    claim_ref=receipt(item['claim'])
    require(checked(claim_ref) == dict(schema=SCHEMA+'.claim',manifest=manifest_ref,
        plan=item['plan'],execution_id=item['execution_id'],attempt_id=item['attempt_id'],no_retries=True),
        'Exact durable no-retry public181 resource claim required')
    require(execution.get('profile_execution') == dict(schema=SCHEMA+'.execution_binding', manifest=manifest_ref,
        plan=item['plan'], execution_id=item['execution_id'], attempt_id=item['attempt_id'],
        raw_schema=RAW_SCHEMA, claim=claim_ref, downstream_adapter_required=True, automatic_fit=False)
        and execution.get('budget') == supervisor.BUDGET
        and execution.get('status') == 'CHILD_EXITED_REQUIRES_PROFILE_REVIEW'
        and execution.get('collection_subprocess_launched') is True
        and type(execution.get('process_group')) is int and execution['process_group'] > 0
        and execution.get('child_reaped') is True and execution.get('live_group_pids') == []
        and execution.get('experiment_lock_held_through_group_exit') is True
        and execution.get('all_owned_descendants_exited_and_reaped') is True
        and execution.get('final_source_audit') == 'PASS'
        and execution.get('watchdog', {}).get('status') == 'EXITED'
        and execution['watchdog'].get('return_code') == 0,
        'Failed/incomplete/old/foreign execution receipt rejected; no retry')
    require(execution.get('command')==[sys.executable,'-m',
        'tools.predictor_validation_20261027.twoFIFO_public181_resource_execute_v1','_child',
        '--manifest',manifest_ref['path'],'--plan',item['plan']['path'],
        '--output',item['collection'],'--execute'], 'Exact owned native child command required')


def run(manifest_path,output,*,execute_enabled=False):
    """Review every complete attempt before the next; never retry or fit."""
    from . import twoFIFO_public181_resource_review_v1 as reviewer
    manifest_ref=receipt(manifest_path);manifest=audit(manifest_path)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    with (Path(manifest_path).resolve().parent/'resource-continuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=dict(schema=CONT_SCHEMA,manifest=manifest_ref,completed_attempts=[],
            execute_invocations=0,no_retries=True,fit_started=False,validation_labels_read=False)
        try:
            for item in manifest['attempts']:
                _verify(manifest_ref);audit(manifest_path)
                result['current_attempt_id']=item['attempt_id']
                collection=Path(item['collection']);final=collection/'execution.json'
                if collection.exists() and not final.is_file():
                    result['status']='WAITING_EXISTING_ATTEMPT_FINAL_RECEIPT';break
                if not collection.exists() and Path(item['claim']).exists():
                    result['status']='HOLD_CLAIMED_ATTEMPT_NO_RETRY';break
                if not collection.exists() and not execute_enabled:
                    result['status']='READY_EXPLICIT_EXECUTE_REQUIRED';break
                if not collection.exists():
                    result['execute_invocations']+=1
                    execute(item['plan']['path'],collection,manifest_path=manifest_path)
                require(final.is_file(),'Native executor returned without final receipt; no retry')
                completed_execution(checked(receipt(final)),manifest_ref,item)
                check=output/item['attempt_id'];check.mkdir()
                reviewed=reviewer.review(collection/'raw.json',check/'native-review',final,
                                         manifest_path=manifest_path)
                doc=checked(reviewed)
                require(doc['status']=='PASS_COMPLETE_PUBLIC181_RESOURCE_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION'
                    and doc['issues']==[] and doc['training_feature_extraction_eligible'] is True,
                    'Native resource review failed; later attempts must not launch')
                result['completed_attempts'].append(dict(attempt_id=item['attempt_id'],
                    execution=receipt(final),training_review=reviewed))
            else:
                result.update(status='COMPLETE_EIGHT_NATIVE_RESOURCE_ATTEMPTS_REVIEWED_NO_FIT',
                    next_gate='TRAINING_ONLY_TUPLE_INTAKE_AND_INDEPENDENT_FRESH_VALIDATION')
        except BaseException as exc:
            result.update(status='STOPPED_NO_RETRY',error=type(exc).__name__+': '+str(exc))
        _write(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--campaign',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('audit');p.add_argument('--manifest',required=True)
    p=sub.add_parser('run');p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true')
    for op in ('execute','_child'):
        p=sub.add_parser(op);p.add_argument('--manifest',required=True);p.add_argument('--plan',required=True)
        p.add_argument('--output',required=True);p.add_argument('--execute',required=True,action='store_true')
    args=parser.parse_args()
    if args.op=='prepare':result=prepare(args.campaign,args.output)
    elif args.op=='audit':result=audit(args.manifest)
    elif args.op=='run':result=run(args.manifest,args.output,execute_enabled=args.execute)
    elif args.op=='execute':result=execute(args.plan,args.output,manifest_path=args.manifest)
    else:
        manifest=audit(args.manifest);plan=audit_plan(args.plan)
        item=next(a for a in manifest['attempts'] if a['plan']==receipt(args.plan))
        require(item['collection']==str(Path(args.output).resolve()), 'Canonical child output required')
        claim=checked(receipt(item['claim']))
        require(claim==dict(schema=SCHEMA+'.claim',manifest=receipt(args.manifest),plan=receipt(args.plan),
            execution_id=plan['execution_id'],attempt_id=plan['attempt_id'],no_retries=True),
            'Exact durable no-retry claim required before native collection')
        result=collect(args.plan,args.output,lease_fd=int(os.environ.get(supervisor.LEASE_FD,'-1')))
    print(json.dumps(result))
    return 1 if result.get('status') in ('failed','FAILED_NO_RETRY','STOPPED_NO_RETRY') else 0


if __name__=='__main__':raise SystemExit(main())
