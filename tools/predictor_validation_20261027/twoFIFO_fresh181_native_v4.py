"""Source-bound native-v4 target with explicit fit, timing and review gates.

Only a real saved fit, measured live CPU budget and independent source-bound
review enable supervised execution. Both preauthenticated law sessions remain
alive through producer, full replay auditor and durable seal. Any missed
predeclared deadline releases zero target requests.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import time
from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.schema import MeasurementConfig
from . import twoFIFO_fresh181_prefix_v1 as prefix
from . import twoFIFO_matching_prospective_bridge_v4 as typed
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard
from .profile_owned_collect_v3 import StartupBoardMonitor,memory_ownership,MONITORING_SCOPE
from .twoFIFO_owned_collect_v3 import (ActivityLedger,atomic_save,initial_rows,history_snapshot,
    wait_until,audit_activity,finish_collection,ACTIVITY_FILE,ACTIVITY_SCHEMA,TIME_DOMAIN)
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from . import twoFIFO_fresh181_execute_v1 as base
from . import twoFIFO_matching_phase12_v1 as response
from . import twoFIFO_public181_full_score_budget_v1 as timing
from . import twoFIFO_public181_expected_score_v2 as expected
from . import twoFIFO_public181_resource_execute_v1 as resource
from contextlib import ExitStack
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import receipt,checked

SCHEMA='gsdeploy.twoFIFO_fresh181_native_target.v4'
RAW_SCHEMA=SCHEMA+'.capture'
OWNERS=base.OWNERS
supervisor=base.supervisor
initial_priming_rows=base.initial_priming_rows
LEAD_SECONDS=300
EXECUTION_HOLD='HOLD_MISSING_REAL_SAVED_FIT_BUDGET_OR_SOURCE_BOUND_INDEPENDENT_NATIVE_REVIEW'
require=response.require
verify=response.verify
save=base.save


def sources():
    from . import twoFIFO_matching_prospective_bridge_v4 as typed
    return strict_references([implementation_sources(__file__),implementation_sources(typed.__file__)])


def _build(public_path,matching_path,fit_ref,namespace,out,*,budget_ref,review_refs,lead_seconds,resource_manifest_ref,resource_review_refs):
    from . import twoFIFO_matching_prospective_bridge_v4 as typed
    fit_audit=response.audit_fit(verify(fit_ref));fit=fit_audit['report']
    fit_manifest=checked(verify(fit['manifest']))
    require(fit_manifest['executor_manifest']==receipt(matching_path),
        'Native target must bind its exact saved matching source fit')
    require(type(lead_seconds) is int and lead_seconds==300,'Unified native-v4 protocol requires exactly300 seconds')
    budget=timing.audit(budget_ref['path'],fit_ref,lead_seconds=lead_seconds,
        resource_manifest_ref=resource_manifest_ref,resource_review_refs=resource_review_refs)
    require(budget['public_preparation']==receipt(public_path) and budget.get('full_U_live_pipeline_included') is True
        and budget['resource_manifest']==resource_manifest_ref and budget['resource_reviews']==resource_review_refs
        and budget['evaluation_protocol']==expected.EVALUATION_PROTOCOL,
        'Exact full-six-term saved-law/resource/protocol budget required')
    built=base._build(public_path,matching_path,namespace,out,lead_seconds=min(lead_seconds,300),
        model_sources=[fit_ref,resource_manifest_ref,*resource_review_refs],generator_sources=[receipt(typed.__file__),receipt(expected.__file__)])
    closure=sources();frozen=[dict(original_path=r['path'],frozen=receipt(out/'source-freeze'/Path(r['path']).relative_to(base.ROOT)))
        for r in closure]
    require(all(r['sha256']==f['frozen']['sha256'] for r,f in zip(closure,frozen)),
        'The complete native-v4 typed collector closure must be frozen')
    if review_refs:
        response._review_bindings(review_refs,[receipt(__file__),receipt(typed.__file__),receipt(timing.__file__),receipt(expected.__file__),receipt(resource.__file__),budget_ref])
    ready=bool(review_refs) and budget['fits_predeclared_budget']
    bound=strict_references([built['sources'],closure,frozen,fit_ref,fit['sources'],budget_ref,budget['sources'],review_refs,resource_manifest_ref,resource_review_refs])
    for item in built['attempts']:
        plan=item['plan'];plan.update(schema=SCHEMA+'.attempt',sources=bound,source_freeze=frozen,
            execute_entrypoint='tools.predictor_validation_20261027.twoFIFO_fresh181_native_v4.collect',
            native_forecast_transfer_bridge='TYPED_V4_FULL_REPLAY_AND_DURABLE_SEAL_REQUIRED',
            native_collector_status='SOURCE_REVIEWED_AND_REAL_FIT_BUDGET_BOUND_REQUIRES_EXPLICIT_EXECUTE' if ready else EXECUTION_HOLD,
            saved_matching_fit=fit_ref,real_fit_replay_budget=budget_ref,independent_native_reviews=review_refs,
            expected_score_resources=dict(manifest=resource_manifest_ref,reviews=resource_review_refs),
            evaluation_protocol=expected.EVALUATION_PROTOCOL,shared_dispatch_protocol=expected.shared_dispatch_protocol(),
            resource_horizon_support='EXPECTED_SCORE_V2_NATIVE90_PLUS120_STRUCTURAL_TRANSFER_AND_HCPU_REQUIRED',
            native_launch_enabled=ready,
            target_plan_auditor=dict(module='tools.predictor_validation_20261027.twoFIFO_fresh181_native_v4',
                source=receipt(__file__),schema=SCHEMA+'.attempt'))
        plan['forecast_barrier'].update(lead_seconds=lead_seconds,prior_READY_settle_timeout_seconds=10,
            epoch_anchor='snapshot_time_ns_plus300s',expected_producer_schema=expected.SCHEMA,
            expected_producer_source=receipt(expected.__file__),expected_report_path=str(Path(item['collection'])/'expected-six-term-report.json'),
            typed_producer_schema=typed.SCHEMA,typed_producer_source=receipt(typed.__file__),
            typed_report_path=str(Path(item['collection'])/'typed-prospective-report.json'),
            seal_schema=expected.SCHEMA+'.prediction_seal',
            policy='PREAUTHENTICATE_TWO_LAW_RESOURCE_SESSIONS_BEFORE_WORKERS_THEN_PREFIX_FULL_SIX_TERM_REPLAY_SEAL_OR_ZERO_TARGET',
            budget=dict(preauthentication_before_worker_startup=True,
                live_prefix_forecast_and_audit_seconds=lead_seconds,
                live_producer_and_auditor_each_replay_all_worlds=True,full_U_live_pipeline_included=True,
                real_fit_runtime_benchmark_required=True,real_fit_budget_qualification=budget_ref,
                timeout_behavior='STOP_BEFORE_ANY_TARGET_ENQUEUE',default60_seconds_never_assumed_sufficient=True))
        plan.pop('execution_context_sha256');plan['execution_context_sha256']=base.digest(plan)
    built.update(sources=bound,matching_fit=fit_ref,native_v4_sources=closure,native_v4_freeze=frozen,
        budget_ref=budget_ref,review_refs=review_refs,lead_seconds=lead_seconds,native_launch_enabled=ready,
        real_fit_budget_qualified=budget['fits_predeclared_budget'],resource_manifest_ref=resource_manifest_ref,resource_review_refs=resource_review_refs)
    return built


def _manifest(built):
    value=base._manifest(built)
    value.update(schema=SCHEMA+'.manifest',status='CPU_PREPARED_SOURCE_BOUND_NATIVE_V4_EXPLICIT_EXECUTE_REQUIRED',
        saved_matching_fit=built['matching_fit'],native_v4_sources=built['native_v4_sources'],
        native_v4_source_freeze=built['native_v4_freeze'],lead_seconds=built['lead_seconds'],
        native_forecast_transfer_bridge='TYPED_V4_FULL_REPLAY_AND_DURABLE_SEAL_REQUIRED',
        native_collector_status=EXECUTION_HOLD if not built['native_launch_enabled'] else 'SOURCE_REVIEWED_AND_REAL_FIT_BUDGET_BOUND',
        real_fit_budget_qualified=built['real_fit_budget_qualified'],
        real_fit_replay_budget=built['budget_ref'],independent_native_reviews=built['review_refs'],
        native_launch_enabled=built['native_launch_enabled'],typed_predictions_sealed=False,
        expected_score_resources=dict(manifest=built['resource_manifest_ref'],reviews=built['resource_review_refs']),
        evaluation_protocol=expected.EVALUATION_PROTOCOL,shared_dispatch_protocol=expected.shared_dispatch_protocol(),
        resource_horizon_support='EXPECTED_SCORE_V2_NATIVE90_PLUS120_STRUCTURAL_TRANSFER_AND_HCPU_REQUIRED')
    return value


def prepare(public_path,matching_path,fit_report_path,namespace,output,*,budget_report_path,resource_manifest_path,resource_review_paths,independent_review_paths=(),lead_seconds=300):
    fit_ref=receipt(fit_report_path);response.audit_fit(fit_ref)
    base.public.audit(public_path);producer=base.matching.audit(matching_path)
    parent,_,_=base._context_and_pool(producer)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'context.json').write_bytes(Path(parent['context']['path']).read_bytes())
    for ref in strict_references([base.executable_sources(),sources()]):
        target=out/'source-freeze'/Path(ref['path']).relative_to(base.ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(ref['path']).read_bytes());verify(ref)
    built=_build(public_path,matching_path,fit_ref,namespace,out,budget_ref=receipt(budget_report_path),
        review_refs=[receipt(p) for p in independent_review_paths],lead_seconds=lead_seconds,
        resource_manifest_ref=receipt(resource_manifest_path),resource_review_refs=[receipt(p) for p in resource_review_paths])
    for item in built['attempts']:save(item['plan_path'],item['plan'])
    result=save(out/'manifest.json',_manifest(built));audit(result['path']);return result


def audit(manifest_path):
    document=checked(receipt(manifest_path))
    require(document.get('schema')==SCHEMA+'.manifest','Explicit source-bound native-v4 manifest required')
    built=_build(document['public_preparation']['path'],document['matching_producer']['path'],
        document['saved_matching_fit'],document['namespace'],Path(manifest_path).resolve().parent,
        budget_ref=document['real_fit_replay_budget'],review_refs=document['independent_native_reviews'],lead_seconds=document['lead_seconds'],
        resource_manifest_ref=document['expected_score_resources']['manifest'],resource_review_refs=document['expected_score_resources']['reviews'])
    for item in built['attempts']:
        require(checked(receipt(item['plan_path']))==item['plan'],'Exact native-v4 source/plan/budget changed')
    require(document==_manifest(built),'Exact native-v4 manifest and all four target plans must rederive')
    for ref in document['sources']:verify(ref)
    return document


def audit_plan(plan_path):
    path=Path(plan_path).resolve();manifest=audit(path.parent.parent/'manifest.json');ref=receipt(path)
    require(ref in [a['plan'] for a in manifest['attempts']],'Canonical native-v4 target plan required')
    plan=checked(ref)
    require(plan.get('schema')==SCHEMA+'.attempt' and plan['target_plan_auditor']==dict(
        module='tools.predictor_validation_20261027.twoFIFO_fresh181_native_v4',source=receipt(__file__),schema=SCHEMA+'.attempt'),
        'Typed-v4 target registry requires its exact source-bound native-v4 auditor')
    return plan


def seal_precohort(plan,raw,manifest_ref,plan_ref,output,sessions,clock_ns,activity,
                   forecast_watches,active_guards,monitor):
    """Extend the frozen resource dispatcher's actual precase checkpoint.

    This adds two complete six-term replays inside the same fixed snapshot+300s
    window. The caller still performs its original durable checkpoint and final
    deadline check before it schedules even the first target request.
    """
    barrier=plan['forecast_barrier'];cpu_only=raw['backend']=='CPU_TEST_ONLY'
    require(len(raw['case_histories'])==1 and len(raw['case_preparations'])==1,
        'Exactly one original dispatcher precase snapshot required')
    require(all(r.get('scheduled_time_ns') is None and r.get('enqueued_time_ns') is None
        and r.get('actual_submit_time_ns') is None for r in raw['request_records']),
        'No target may be scheduled or enqueued before the complete forecast seal')
    require(not raw['measurement_windows'] and not raw.get('predictions_durable_before_cohort'),
        'Single forecast barrier before the target cohort required')
    snapshot=raw['case_histories'][0]['prior_history_snapshot']
    actual=raw['precohort_training_prefix'];epoch=actual['epoch_ns']
    require(actual==dict(anchor_ns=snapshot['snapshot_time_ns'],epoch_ns=epoch,lead_ns=resource.LEAD_NS,
        settled_seed=resource.settled_priming(plan,snapshot,epoch),
        source='ACTUAL_EIGHTEEN_PRIMING_AND_READY_PREFIX',target_future_labels_used=False,
        memory_state_inferred_from_seed=False) and epoch==snapshot['snapshot_time_ns']+resource.LEAD_NS,
        'Actual settled priming and snapshot+300s resource protocol required')
    producer_session,auditor_session=sessions
    require(producer_session is not auditor_session and producer_session.queue is not auditor_session.queue,
        'Producer and auditor require separately authenticated complete law/resource sessions')
    with activity.span('precase_expected_forecast',case_id=raw['checkpoint_context']['case_id'],phase=monitor.phase):
        live_begin=clock_ns();require(live_begin<epoch,'No live forecast budget remains; zero targets')
        prefix_ref=prefix.make_prefix(plan_ref,plan,manifest_ref,snapshot,epoch,
            raw['ready_ownership_evidence'],barrier['prefix_path'],backend=raw['backend'])
        gate_ref=prefix.gate_request(prefix_ref,plan_ref,plan,manifest_ref,barrier['gate_request_path'],
            allow_cpu_fixture=cpu_only)
        raw.update(causal_prefix_status='PERSISTED_BEFORE_FIRST_COHORT_SEND',precohort_prefix=prefix_ref,
            prediction_gate=gate_ref,planned_cohort_epoch_ns=epoch)
        require(clock_ns()<epoch,'Prefix persistence exhausted fixed300s budget; zero targets')
        report_ref=expected.produce(producer_session,gate_ref['path'],barrier['expected_report_path'],
            allow_cpu_fixture=cpu_only)
        # Keep the producer's exact bytes watched from return through seal,
        # checkpoint, enqueue/drain and final cleanup, including change/restore.
        guard=forecast_watches.enter_context(SourceChangeGuard([report_ref,gate_ref,prefix_ref]))
        active_guards.append(guard);guard.check()
        raw.update(expected_forecast_report=report_ref,expected_producer_elapsed_ns=clock_ns()-live_begin)
        require(clock_ns()<epoch,'Full six-term producer exceeded origin; zero targets')
        audit_begin=clock_ns()
        # seal performs the second complete replay itself. Never add a third.
        seal_ref=expected.seal(auditor_session,gate_ref['path'],report_ref['path'],barrier['seal_path'],
            clock_ns=clock_ns,report_reference=report_ref)
        guard.check()
        seal_guard=forecast_watches.enter_context(SourceChangeGuard([seal_ref]));active_guards.append(seal_guard)
        accepted=clock_ns()
        expected.audit_seal_structure(auditor_session,gate_ref,seal_ref,now_ns=accepted,allow_cpu_fixture=cpu_only)
        guard.check();seal_guard.check()
        require(clock_ns()<epoch,'Prediction authentication crossed origin; zero targets')
        raw.update(prediction_seal=seal_ref,prediction_seal_accepted_ns=accepted,
            expected_auditor_and_seal_elapsed_ns=clock_ns()-audit_begin,
            predictions_durable_before_cohort=True,forecast_generator_correctness_independently_verified=False,
            live_expected_forecast_passes=2)


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
    registry=resource.campaign.planned_worker_registry(context)
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
    raw.update(executed_worker_registry=registry,shared_dispatch_protocol=plan['shared_dispatch_protocol'],
        resource_training_manifest=plan['expected_score_resources']['manifest'],evaluation_protocol=plan['evaluation_protocol'])
    activity=ActivityLedger(output,plan['execution_id'],clock_ns);forecast_watches=ExitStack();active_guards=[]
    def checkpoint_guard_check(context):
        failed=False
        for guard in active_guards:
            try:guard.check()
            except BaseException as exc:
                if context['kind']!='shutdown':raise
                # Workers are already closed. A failed source watch must not
                # prevent preservation of the failure and all offered rows.
                # This path can never recover successful execution status.
                failed=True;raw['status']='failed'
                message='ShutdownForecastSourceGuardError: '+type(exc).__name__+': '+str(exc)
                if message not in raw['errors']:raw['errors'].append(message)
        return failed
    def checkpoint():
        context=raw['checkpoint_context']
        if context['kind']=='precase':
            seal_precohort(plan,raw,manifest_ref,receipt(plan_path),output,sessions,clock_ns,activity,
                forecast_watches,active_guards,monitor)
        checkpoint_guard_check(context)
        now=clock_ns()
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
        if checkpoint_guard_check(context):
            raw['shutdown_failure_receipt_extra_write']=True
            atomic_save(output/'raw.json',raw)
    def wait(target):
        for guard in active_guards:guard.check()
        result=wait_until_ns(target) if wait_until_ns is not None else wait_until(target,clock_ns,monitor)
        for guard in active_guards:guard.check()
        return result
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
            result=resource.dispatch_case(plan,case,runtime,pool,raw,clock_ns=clock_ns,wait_until_ns=wait,
                checkpoint=checkpoint,monitor=monitor,activity=activity)
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
        try:finish_collection(runtime,monitor,raw,checkpoint,clock_ns)
        finally:forecast_watches.close()
    return raw


def collect(plan_path,output,*,lease_fd,**kwargs):
    supervisor.validate_lease(lease_fd);plan=audit_plan(plan_path);require_native_ready(plan)
    require(not kwargs,'Native collection cannot inject transport, clock or mutable sessions')
    manifest_path=Path(plan_path).resolve().parent.parent/'manifest.json'
    # These two distinct authenticators finish their source/law work before the
    # runtime constructor/startup. Their guards stay active through shutdown.
    with expected.open_target_session(plan['saved_matching_fit']['path'],manifest_path,plan_path) as producer:
        with expected.open_target_session(plan['saved_matching_fit']['path'],manifest_path,plan_path) as auditor:
            return _collect_runtime(plan_path,output,lease_fd=lease_fd,sessions=(producer,auditor))


def collect_cpu(plan_path,output,*,lease_fd,sessions,**kwargs):
    require(kwargs.get('runtime_factory') is not None and kwargs['runtime_factory'] is not DeploymentRuntime
        and kwargs.get('clock_ns') is not None and kwargs['clock_ns'] is not time.perf_counter_ns,
        'CPU collection requires injected transport and clock; cannot launch native workers')
    return _collect_runtime(plan_path,output,lease_fd=lease_fd,sessions=sessions,**kwargs)




def require_native_ready(plan):
    require(plan.get('schema')==SCHEMA+'.attempt','Exact native-v4 target schema required')
    if plan.get('native_launch_enabled') is not True:raise UnknownEvidence(EXECUTION_HOLD)
    response._review_bindings(plan['independent_native_reviews'],
        [receipt(__file__),receipt(typed.__file__),receipt(timing.__file__),receipt(expected.__file__),receipt(resource.__file__),plan['real_fit_replay_budget']])
    budget=timing.audit(plan['real_fit_replay_budget']['path'],plan['saved_matching_fit'],
        lead_seconds=plan['forecast_barrier']['lead_seconds'],resource_manifest_ref=plan['expected_score_resources']['manifest'],
        resource_review_refs=plan['expected_score_resources']['reviews'])
    if not budget['fits_predeclared_budget']:raise UnknownEvidence(EXECUTION_HOLD+':INSUFFICIENT_PREDECLARED_LIVE_SECONDS')
    return plan


def execute(plan_path,output,*,manifest_path):
    manifest_ref=receipt(manifest_path);snapshot=checked(verify(manifest_ref))
    watched=strict_references([manifest_ref,snapshot['sources'],snapshot['attempts']])
    with SourceChangeGuard(watched) as guard:
        guard.check();manifest=audit(manifest_path)
        require(manifest==snapshot,'Guarded native-v4 manifest changed')
        plan_ref=receipt(plan_path);item=next(a for a in manifest['attempts'] if a['plan']==plan_ref)
        plan=audit_plan(plan_path);require_native_ready(plan);verify(plan_ref);guard.check()
        require(str(Path(output).resolve())==item['collection'],'Canonical native-v4 collection required')
        require(not Path(output).exists() and not Path(item['claim']).exists(),'No retry of started/claimed native-v4 capture')
        supervisor.public_environment_gate();guard.check()
        claim_ref=save(item['claim'],dict(schema=SCHEMA+'.claim',manifest=manifest_ref,plan=plan_ref,
            execution_id=plan['execution_id'],attempt_id=plan['attempt_id'],candidate_id=plan['candidate_id'],
            saved_matching_fit=plan['saved_matching_fit'],replay_budget=plan['real_fit_replay_budget'],no_retries=True))
        command=[sys.executable,'-m','tools.predictor_validation_20261027.twoFIFO_fresh181_native_v4','_child',
            '--manifest',str(Path(manifest_path).resolve()),'--plan',str(Path(plan_path).resolve()),
            '--output',str(Path(output).resolve()),'--execute']
        with SourceChangeGuard([claim_ref]) as claim_guard:
            def gate():
                guard.check();claim_guard.check();verify(manifest_ref);audit(manifest_path)
                require_native_ready(audit_plan(plan_path));supervisor.public_environment_gate()
                guard.check();claim_guard.check()
            gate()
            result=supervisor.supervise(command,output,
                environment={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                before_launch=gate,after_run=gate,metadata=dict(schema=SCHEMA+'.execution_binding',
                    manifest=manifest_ref,plan=plan_ref,execution_id=plan['execution_id'],attempt_id=plan['attempt_id'],
                    candidate_id=plan['candidate_id'],raw_schema=RAW_SCHEMA,claim=claim_ref,
                    saved_matching_fit=plan['saved_matching_fit'],replay_budget=plan['real_fit_replay_budget'],
                    expected_producer_schema=expected.SCHEMA,predictions_required_before_cohort=True,automatic_fit=False))
            guard.check();claim_guard.check();return result


def verify_child_claim(manifest_path,plan_path,output):
    manifest=audit(manifest_path);plan=audit_plan(plan_path);require_native_ready(plan);plan_ref=receipt(plan_path)
    item=next(a for a in manifest['attempts'] if a['plan']==plan_ref)
    require(str(Path(output).resolve())==item['collection'],'Canonical native-v4 child output required')
    claim_ref=receipt(item['claim'])
    require(checked(verify(claim_ref))==dict(schema=SCHEMA+'.claim',manifest=receipt(manifest_path),plan=plan_ref,
        execution_id=plan['execution_id'],attempt_id=plan['attempt_id'],candidate_id=plan['candidate_id'],
        saved_matching_fit=plan['saved_matching_fit'],replay_budget=plan['real_fit_replay_budget'],no_retries=True),
        'Exact native-v4 exclusive fit/budget/candidate claim required')
    return plan


def completed_execution(execution,manifest_ref,item):
    """Recheck v4-only completion; an old training/fresh receipt cannot pass."""
    plan=checked(verify(item['plan']));claim_ref=receipt(item['claim'])
    require(checked(verify(claim_ref))==dict(schema=SCHEMA+'.claim',manifest=manifest_ref,plan=item['plan'],
        execution_id=item['execution_id'],attempt_id=item['attempt_id'],candidate_id=item['candidate_id'],
        saved_matching_fit=plan['saved_matching_fit'],replay_budget=plan['real_fit_replay_budget'],no_retries=True),
        'Exact native-v4 exclusive fit/budget completion claim required')
    require(execution.get('profile_execution')==dict(schema=SCHEMA+'.execution_binding',manifest=manifest_ref,
        plan=item['plan'],execution_id=item['execution_id'],attempt_id=item['attempt_id'],candidate_id=item['candidate_id'],
        raw_schema=RAW_SCHEMA,claim=claim_ref,saved_matching_fit=plan['saved_matching_fit'],
        replay_budget=plan['real_fit_replay_budget'],expected_producer_schema=expected.SCHEMA,
        predictions_required_before_cohort=True,automatic_fit=False)
        and execution.get('command')==[sys.executable,'-m',
            'tools.predictor_validation_20261027.twoFIFO_fresh181_native_v4','_child',
            '--manifest',manifest_ref['path'],'--plan',item['plan']['path'],
            '--output',item['collection'],'--execute']
        and execution.get('budget')==supervisor.BUDGET
        and execution.get('status')=='CHILD_EXITED_REQUIRES_PROFILE_REVIEW'
        and execution.get('collection_subprocess_launched') is True
        and type(execution.get('process_group')) is int and execution['process_group']>0
        and execution.get('child_reaped') is True and execution.get('live_group_pids')==[]
        and execution.get('experiment_lock_held_through_group_exit') is True
        and execution.get('all_owned_descendants_exited_and_reaped') is True
        and execution.get('final_source_audit')=='PASS'
        and execution.get('watchdog',{}).get('status')=='EXITED' and execution['watchdog'].get('return_code')==0,
        'Failed/old/foreign/incomplete native-v4 execution rejected; no retry')


def audit_completed_prefix(manifest_path,plan_path,execution_path,output):
    """Bind full typed replay and native ancestry to the completed supervisor.

    No target response labels are opened. This does not substitute for a full
    independent stdout, resource, quality or prediction-accuracy review.
    """
    manifest_ref=receipt(manifest_path);manifest=audit(manifest_path);plan_ref=receipt(plan_path)
    item=next(a for a in manifest['attempts'] if a['plan']==plan_ref);plan=audit_plan(plan_path)
    require(Path(execution_path).resolve()==Path(item['collection'])/'execution.json',
        'Canonical native-v4 completed execution required')
    execution_ref=receipt(execution_path);execution=checked(verify(execution_ref))
    completed_execution(execution,manifest_ref,item);barrier=plan['forecast_barrier']
    gate_ref=receipt(barrier['gate_request_path']);seal_ref=receipt(barrier['seal_path']);seal=checked(verify(seal_ref))
    report_ref=seal['expected_report'];prefix_ref=receipt(barrier['prefix_path'])
    watched=strict_references([manifest_ref,plan_ref,execution_ref,receipt(item['claim']),
        gate_ref,prefix_ref,seal_ref,report_ref,plan['sources']])
    with SourceChangeGuard(watched) as guard:
        guard.check()
        with expected.open_target_session(plan['saved_matching_fit']['path'],manifest_path,plan_path) as session:
            cached,_,document,gate=typed._prefix_and_gate(session.queue,gate_ref)
            require(gate['prefix']==prefix_ref,'Fixed canonical completed prefix receipt required')
            prefix.inspect_native_ownership(document['prefix_snapshot'],document['ready_ownership_evidence'],
                collector_pid=execution['process_group'])
            verify(report_ref);expected.audit_report(session,report_ref['path'],require_native=True)
            verify(report_ref);guard.check()
            expected.audit_seal_structure(session,gate_ref,seal_ref,now_ns=seal['seal_begin_ns'])
            guard.check();source_bindings=watched
            result=save(output,dict(schema=SCHEMA+'.completed_prefix_source_review',
            status='PASS_OWNED_V4_EXECUTION_BOUND_PREFIX_AND_FULL_SIX_TERM_REPLAY_NOT_FULL_PROTOCOL_REVIEW',
            manifest=manifest_ref,plan=plan_ref,execution=execution_ref,claim=receipt(item['claim']),
            prefix=gate['prefix'],gate=gate_ref,prediction_seal=seal_ref,prediction=seal['expected_report'],
            execution_id=plan['execution_id'],candidate_id=plan['candidate_id'],
            prefix_source_bound_to_completed_owned_execution=True,full_typed_replay_rederived=True,
            native_protocol_stdout_independently_reparsed=False,target_raw_or_future_labels_read=False,
            collector_acceptance_clock_independently_reconstructed=False,independent_accuracy_qualified=False,
                resource_horizon_qualified=False,full_U_qualified=False,source_bindings=source_bindings))
            verify(report_ref);guard.check();return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    cmd=sub.add_parser('prepare')
    for key in ('public-preparation','matching-producer','fit-report','namespace','output','budget-report','resource-manifest'):cmd.add_argument('--'+key,required=True)
    cmd.add_argument('--resource-reviews',nargs='+',required=True);cmd.add_argument('--independent-reviews',nargs='*',default=[]);cmd.add_argument('--lead-seconds',type=int,default=300)
    cmd=sub.add_parser('audit');cmd.add_argument('--manifest',required=True)
    cmd=sub.add_parser('audit-prefix')
    for key in ('manifest','plan','execution','output'):cmd.add_argument('--'+key,required=True)
    for op in ('execute','_child'):
        cmd=sub.add_parser(op)
        for key in ('manifest','plan','output'):cmd.add_argument('--'+key,required=True)
        cmd.add_argument('--execute',required=True,action='store_true')
    args=parser.parse_args()
    if args.op=='prepare':result=prepare(args.public_preparation,args.matching_producer,args.fit_report,args.namespace,args.output,
        budget_report_path=args.budget_report,resource_manifest_path=args.resource_manifest,resource_review_paths=args.resource_reviews,independent_review_paths=args.independent_reviews,lead_seconds=args.lead_seconds)
    elif args.op=='audit':result=audit(args.manifest)
    elif args.op=='audit-prefix':result=audit_completed_prefix(args.manifest,args.plan,args.execution,args.output)
    elif args.op=='execute':result=execute(args.plan,args.output,manifest_path=args.manifest)
    else:
        verify_child_claim(args.manifest,args.plan,args.output)
        result=collect(args.plan,args.output,lease_fd=int(os.environ.get(supervisor.LEASE_FD,'-1')))
    print(json.dumps(result))
    sys.exit(1 if result.get('status') in ('failed','FAILED_NO_RETRY','STOPPED_NO_RETRY') else 0)
