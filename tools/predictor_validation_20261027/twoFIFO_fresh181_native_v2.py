"""DRAFT source-bound native-v2 target plan; execution remains review HOLD.

CPU preparation/audit only. The separate typed-v2 auditor accepts exactly this
schema/module. No frozen fresh181/typed-v1 global or schema is patched. A later
reviewed collector implementation must keep both preauthenticated law sessions
alive and complete producer, full replay auditor and durable seal within the
predeclared300-second lead, or send zero target requests.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import time
from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.schema import MeasurementConfig
from . import twoFIFO_fresh181_prefix_v1 as prefix
from . import twoFIFO_matching_prospective_bridge_v2 as typed
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard
from .profile_owned_collect_v3 import StartupBoardMonitor,memory_ownership,MONITORING_SCOPE
from .twoFIFO_owned_collect_v3 import (ActivityLedger,atomic_save,initial_rows,history_snapshot,
    wait_until,audit_activity,finish_collection,ACTIVITY_FILE,ACTIVITY_SCHEMA,TIME_DOMAIN)
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from . import twoFIFO_fresh181_execute_v1 as base
from . import twoFIFO_matching_phase12_v1 as response
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import receipt,checked

SCHEMA='gsdeploy.twoFIFO_fresh181_native_target.v2'
RAW_SCHEMA=SCHEMA+'.capture'
OWNERS=base.OWNERS
supervisor=base.supervisor
initial_priming_rows=base.initial_priming_rows
LEAD_SECONDS=300
EXECUTION_HOLD='HOLD_NATIVE_V2_COLLECTOR_AND_TYPED_SEAL_INDEPENDENT_REVIEW_NOT_COMPLETE'
require=response.require
verify=response.verify
save=base.save


def sources():
    from . import twoFIFO_matching_prospective_bridge_v2 as typed
    return strict_references([implementation_sources(__file__),implementation_sources(typed.__file__)])


def _build(public_path,matching_path,fit_ref,namespace,out):
    from . import twoFIFO_matching_prospective_bridge_v2 as typed
    fit_audit=response.audit_fit(verify(fit_ref));fit=fit_audit['report']
    fit_manifest=checked(verify(fit['manifest']))
    require(fit_manifest['executor_manifest']==receipt(matching_path),
        'Native target must bind its exact saved matching source fit')
    built=base._build(public_path,matching_path,namespace,out,lead_seconds=LEAD_SECONDS,
        model_sources=[fit_ref],generator_sources=[receipt(typed.__file__)])
    closure=sources();frozen=[dict(original_path=r['path'],frozen=receipt(out/'source-freeze'/Path(r['path']).relative_to(base.ROOT)))
        for r in closure]
    require(all(r['sha256']==f['frozen']['sha256'] for r,f in zip(closure,frozen)),
        'The complete native-v2 typed collector closure must be frozen')
    bound=strict_references([built['sources'],closure,frozen,fit_ref,fit['sources']])
    for item in built['attempts']:
        plan=item['plan'];plan.update(schema=SCHEMA+'.attempt',sources=bound,source_freeze=frozen,
            execute_entrypoint='tools.predictor_validation_20261027.twoFIFO_fresh181_native_v2.collect',
            native_forecast_transfer_bridge='TYPED_V2_REPORT_REPLAY_REQUIRED_CURRENT_EXECUTION_HOLD',
            native_collector_status=EXECUTION_HOLD,saved_matching_fit=fit_ref,
            target_plan_auditor=dict(module='tools.predictor_validation_20261027.twoFIFO_fresh181_native_v2',
                source=receipt(__file__),schema=SCHEMA+'.attempt'))
        plan['forecast_barrier'].update(lead_seconds=LEAD_SECONDS,
            typed_producer_schema=typed.SCHEMA,typed_producer_source=receipt(typed.__file__),
            typed_report_path=str(Path(item['collection'])/'typed-prospective-report.json'),
            seal_schema=typed.SCHEMA+'.prediction_seal',
            policy='PREAUTHENTICATE_TWO_LAW_SESSIONS_BEFORE_WORKERS_THEN_PREFIX_FULL_REPLAY_TYPED_SEAL_OR_ZERO_TARGET',
            budget=dict(preauthentication_before_worker_startup=True,
                live_prefix_forecast_and_audit_seconds=LEAD_SECONDS,
                live_producer_and_auditor_each_replay_all_worlds=True,
                real_fit_runtime_benchmark_required=True,real_fit_budget_qualification='UNKNOWN_NOT_BENCHMARKED',
                timeout_behavior='STOP_BEFORE_ANY_TARGET_ENQUEUE',default60_seconds_never_assumed_sufficient=True))
        plan.pop('execution_context_sha256');plan['execution_context_sha256']=base.digest(plan)
    built.update(sources=bound,matching_fit=fit_ref,native_v2_sources=closure,native_v2_freeze=frozen)
    return built


def _manifest(built):
    value=base._manifest(built)
    value.update(schema=SCHEMA+'.manifest',status='CPU_PREPARED_NATIVE_V2_DRAFT_REQUIRES_COLLECTOR_AND_TYPED_SEAL_REVIEW',
        saved_matching_fit=built['matching_fit'],native_v2_sources=built['native_v2_sources'],
        native_v2_source_freeze=built['native_v2_freeze'],lead_seconds=LEAD_SECONDS,
        native_forecast_transfer_bridge='TYPED_V2_REPORT_REPLAY_REQUIRED_CURRENT_EXECUTION_HOLD',
        native_collector_status=EXECUTION_HOLD,real_fit_budget_qualified=False,
        native_launch_enabled=False,typed_predictions_sealed=False)
    return value


def prepare(public_path,matching_path,fit_report_path,namespace,output):
    fit_ref=receipt(fit_report_path);response.audit_fit(fit_ref)
    base.public.audit(public_path);producer=base.matching.audit(matching_path)
    parent,_,_=base._context_and_pool(producer)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'context.json').write_bytes(Path(parent['context']['path']).read_bytes())
    for ref in strict_references([base.executable_sources(),sources()]):
        target=out/'source-freeze'/Path(ref['path']).relative_to(base.ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(ref['path']).read_bytes());verify(ref)
    built=_build(public_path,matching_path,fit_ref,namespace,out)
    for item in built['attempts']:save(item['plan_path'],item['plan'])
    result=save(out/'manifest.json',_manifest(built));audit(result['path']);return result


def audit(manifest_path):
    document=checked(receipt(manifest_path))
    require(document.get('schema')==SCHEMA+'.manifest','Explicit source-bound native-v2 manifest required')
    built=_build(document['public_preparation']['path'],document['matching_producer']['path'],
        document['saved_matching_fit'],document['namespace'],Path(manifest_path).resolve().parent)
    for item in built['attempts']:
        require(checked(receipt(item['plan_path']))==item['plan'],'Exact native-v2 source/plan/budget changed')
    require(document==_manifest(built),'Exact native-v2 manifest and all four target plans must rederive')
    for ref in document['sources']:verify(ref)
    return document


def audit_plan(plan_path):
    path=Path(plan_path).resolve();manifest=audit(path.parent.parent/'manifest.json');ref=receipt(path)
    require(ref in [a['plan'] for a in manifest['attempts']],'Canonical native-v2 target plan required')
    plan=checked(ref)
    require(plan.get('schema')==SCHEMA+'.attempt' and plan['target_plan_auditor']==dict(
        module='tools.predictor_validation_20261027.twoFIFO_fresh181_native_v2',source=receipt(__file__),schema=SCHEMA+'.attempt'),
        'Typed-v2 target registry requires its exact source-bound native-v2 auditor')
    return plan


def dispatch_cohort(plan,case,runtime,pool,raw,*,clock_ns,wait_until_ns,checkpoint,monitor,activity,
                    plan_ref,manifest_ref,output,sessions):
    """Publish actual prefix, seal forecasts, then use original explicit FIFO.

    The old dispatch helper fixes10+60 and cannot enforce a forecast barrier;
    this derivative changes only those boundaries and the pre-epoch interface.
    All cohort enqueue operations still use original runtime.enqueue/drain.
    """
    barrier=plan['forecast_barrier']
    with activity.span('precohort_history_snapshot',case_id=case['case_id'],phase=monitor.phase):
        settle_begin=clock_ns()
        while True:
            snapshot=history_snapshot(runtime,raw,clock_ns)
            epoch=clock_ns()+round(barrier['lead_seconds']*1e9)
            try:
                prefix.derive(plan,snapshot,epoch)
                break
            except UnknownEvidence as exc:
                if clock_ns()-settle_begin>=round(barrier['prior_READY_settle_timeout_seconds']*1e9):
                    raw['causal_prefix_status']='UNKNOWN_BUSY_OR_UNPROVED_PRIOR_READY'
                    raw['causal_prefix_error']=str(exc)
                    raw['unsettled_prior_prefix']=save(output/'unsettled-prefix.json',
                        dict(schema=SCHEMA+'.unsettled_prefix',snapshot=snapshot,execution_id=plan['execution_id'],
                            seed=None,reason=str(exc),no_busy_state_coerced_to_settled=True))
                    raise
                monitor.check();wait_until_ns(clock_ns()+10_000_000)
        raw['case_histories'].append(dict(case_id=case['case_id'],prior_history_snapshot=snapshot))
    try:
        with activity.span('precohort_prefix_persistence',case_id=case['case_id'],phase=monitor.phase):
            prefix_ref=prefix.make_prefix(plan_ref,plan,manifest_ref,snapshot,epoch,
                raw['ready_ownership_evidence'],barrier['prefix_path'],backend=raw['backend'])
            gate_ref=prefix.gate_request(prefix_ref,plan_ref,plan,manifest_ref,barrier['gate_request_path'],
                allow_cpu_fixture=raw['backend']=='CPU_TEST_ONLY')
    except Exception as exc:
        raw['causal_prefix_status']='UNKNOWN_NOT_SETTLED_OR_NOT_AUTHENTICATED'
        raw['causal_prefix_error']=type(exc).__name__+': '+str(exc)
        raise
    raw.update(causal_prefix_status='PERSISTED_BEFORE_FIRST_COHORT_SEND',precohort_prefix=prefix_ref,
        prediction_gate=gate_ref,planned_cohort_epoch_ns=epoch)
    producer_session,auditor_session=sessions
    require(producer_session is not auditor_session,
        'Producer and auditor must hold separately authenticated independent law sessions')
    live_begin=clock_ns()
    require(live_begin<epoch,'Prefix persistence exhausted the predeclared300-second live budget')
    report_ref=typed.produce(producer_session,gate_ref['path'],barrier['typed_report_path'],
        allow_cpu_fixture=raw['backend']=='CPU_TEST_ONLY')
    with SourceChangeGuard([report_ref,gate_ref,prefix_ref]) as forecast_guard:
        forecast_guard.check()
        require(clock_ns()<epoch,'Typed producer exceeded the origin; zero targets')
        raw['typed_forecast_report']=report_ref;raw['typed_producer_elapsed_ns']=clock_ns()-live_begin
        # A separately preauthenticated auditor replays every world before sealing.
        # Only the injected CPU clock can admit CPU interface fixtures.
        audit_begin=clock_ns()
        seal_ref=typed.seal(auditor_session,gate_ref['path'],report_ref['path'],barrier['seal_path'],clock_ns=clock_ns,report_reference=report_ref)
        forecast_guard.check();accepted=clock_ns()
        typed.audit_seal_structure(auditor_session,gate_ref,seal_ref,now_ns=accepted,
            allow_cpu_fixture=raw['backend']=='CPU_TEST_ONLY')
        raw['typed_auditor_and_seal_elapsed_ns']=clock_ns()-audit_begin
        require(clock_ns()<epoch,'Prediction authentication crossed the first-arrival origin; no cohort was sent')
        raw.update(prediction_seal=seal_ref,prediction_seal_accepted_ns=accepted,
            predictions_durable_before_cohort=True,forecast_generator_correctness_independently_verified=False)
        raw['case_preparations'].append(dict(case_id=case['case_id'],public_arrivals_sha256=case['public_arrivals_sha256'],
            public_arrivals=copy.deepcopy(case['arrivals']),prior_history_index=0,prepared_before_epoch=True,
            prediction_seal=seal_ref,prefix=prefix_ref))
        raw['checkpoint_context']=dict(kind='precohort_prediction_accepted',case_id=case['case_id']);checkpoint()
        forecast_guard.check()
        require(clock_ns()<epoch,'Pre-cohort durable checkpoint crossed the origin; no cohort was sent')
        # Keep all sealed inputs live under a change/restore watch through drain.
        watched=strict_references([prefix_ref,gate_ref,seal_ref,checked(seal_ref)['typed_report'],
            barrier['model_sources'],barrier['generator_sources']])
        with SourceChangeGuard(watched) as guard:
            forecast_guard.check();guard.check();wait_until_ns(epoch);guard.check();forecast_guard.check();monitor.set_phase('measurement')
            end=epoch+plan['measurement_ns'];deadline=end+plan['drain_ns']
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
                    if row['status'] in ('planned','queued','submitted'):row.update(status='incomplete',error='fixed drain reached; offered retained')
                    row['RESULT_reader_within_fixed_drain']=(type(row.get('completion_time_ns')) is int
                        and row['scheduled_time_ns']<=row['completion_time_ns']<=deadline)
                observed.update(status='completed' if all(r['status']=='completed' and r['RESULT_reader_within_fixed_drain'] for r in rows) else 'incomplete',
                    observation_finalized_ns=clock_ns(),fixed_cutoff_overshoot_ns=max(0,clock_ns()-deadline))
            guard.check()
        wait_until_ns(deadline+1);monitor.sample();monitor.check()
        raw['checkpoint_context']=dict(kind='postdrain',case_id=case['case_id']);checkpoint()
        return observed


def _collect_runtime(plan_path,output,*,lease_fd,runtime_factory=DeploymentRuntime,monitor_factory=StartupBoardMonitor,
            clock_ns=time.perf_counter_ns,wait_until_ns=None,ownership_check=memory_ownership,sessions):
    supervisor.validate_lease(lease_fd);plan=audit_plan(plan_path);output=Path(output)
    injected=(runtime_factory is not DeploymentRuntime or monitor_factory is not StartupBoardMonitor
        or clock_ns is not time.perf_counter_ns or wait_until_ns is not None or ownership_check is not memory_ownership)
    manifest_ref=receipt(Path(plan_path).resolve().parent.parent/'manifest.json')
    require(str(output.resolve())==str(Path(plan['forecast_barrier']['prefix_path']).parent),
        'Canonical fresh181 collection/prefix path required')
    output.mkdir(parents=True,exist_ok=True)
    require(not (output/'raw.json').exists(),'No overwrite/retry of a collection attempt')
    context=checked(plan['context']);pool={r['pool_id']:r for r in (json.loads(l) for l in Path(plan['prepared_pool']['path']).read_text().splitlines())}
    registry={}
    for alias in OWNERS:
        family=alias.split('#')[0];binding=copy.deepcopy(context['worker_bindings'][family])
        binding['command'][1]=str(Path(__file__).with_name('semantic_progress_worker_v1.py'))
        binding['env'].update(context['threads']);binding['env']['PYTHONUNBUFFERED']='1';registry[alias]=binding
    runtime=runtime_factory(MeasurementConfig(**plan['runtime_config']),registry);monitor=None
    executed=dict(attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],cases=plan['cases'])
    atomic_save(output/'executed-cases.json',executed)
    raw=dict(schema=RAW_SCHEMA,collector_schema=RAW_SCHEMA,kind='CPU_TEST_ONLY' if injected else 'measured',
        backend='CPU_TEST_ONLY' if injected else 'native',split='validation',
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
        fresh181_executor_sources=implementation_sources(__file__),
        target_response_labels_collected=False,forecast_target_future_labels_read=False,
        predictions_durable_before_cohort=False,causal_prefix_status='NOT_CAPTURED',
        resource_horizon_support=plan['resource_horizon_support'])
    for row in raw['request_records']:
        row.update(split='validation',validation_view='fresh181',training_role=None)
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
            require(row['status']=='completed','Fresh181 prior priming failed')
        raw['lifecycle_boundaries'].append(dict(boundary='priming_end',time_ns=clock_ns()))
        raw['checkpoint_context']=dict(kind='postpriming',case_id=None);checkpoint()
        for case in plan['cases']:
            result=dispatch_cohort(plan,case,runtime,pool,raw,clock_ns=clock_ns,wait_until_ns=wait,
                checkpoint=checkpoint,monitor=monitor,activity=activity,plan_ref=receipt(plan_path),
                manifest_ref=manifest_ref,output=output,sessions=sessions)
            raw['all_requests_completed']=(result['status']=='completed')
            raw['target_response_labels_collected']=True
        audit_activity(raw,activity)
        require(raw['no_mutating_checkpoint_during_case'],'Checkpoint crossed a protected case interval')
        raw['status']='completed'
        raw['observation_end_ns']=clock_ns()
        raw['lifecycle_boundaries'].append(dict(boundary='measurement_and_drain_end',time_ns=raw['observation_end_ns']))
    except BaseException as exc:
        raw['status']='failed';raw['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        # Shared reviewed cleanup stops monitoring and takes its final sample
        # while workers are live, then closes workers. Final save is atomic.
        finish_collection(runtime,monitor,raw,checkpoint,clock_ns)
    return raw


def collect(plan_path,output,*,lease_fd,**kwargs):
    # Draft remains fail-closed even for direct child invocation. CPU control
    # tests use collect_cpu with explicit injected transport and sessions.
    raise UnknownEvidence(EXECUTION_HOLD)


def collect_cpu(plan_path,output,*,lease_fd,sessions,**kwargs):
    require(kwargs.get('runtime_factory') is not None and kwargs['runtime_factory'] is not DeploymentRuntime
        and kwargs.get('clock_ns') is not None and kwargs['clock_ns'] is not time.perf_counter_ns,
        'Draft CPU collection requires injected transport and clock; cannot launch native workers')
    return _collect_runtime(plan_path,output,lease_fd=lease_fd,sessions=sessions,**kwargs)




def execute(*args,**kwargs):
    raise UnknownEvidence(EXECUTION_HOLD)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    cmd=sub.add_parser('prepare')
    for key in ('public-preparation','matching-producer','fit-report','namespace','output'):cmd.add_argument('--'+key,required=True)
    cmd=sub.add_parser('audit');cmd.add_argument('--manifest',required=True)
    args=parser.parse_args()
    print(prepare(args.public_preparation,args.matching_producer,args.fit_report,args.namespace,args.output)
        if args.op=='prepare' else audit(args.manifest))
