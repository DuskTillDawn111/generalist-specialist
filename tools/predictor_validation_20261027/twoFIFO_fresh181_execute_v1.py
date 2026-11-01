"""Fresh four-candidate181 owned collector and strict CPU preparation.

Preparation/audit never collect or fit. The native path preserves the matching
worker backend and public181 workload, adds a durable pre-cohort prediction
barrier, and keeps90+120 resource support explicitly UNKNOWN. No frozen producer
is changed or relabeled; actual execution requires separate --execute.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid
from gsdeploy.forecast_errors import UnknownEvidence

from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.schema import MeasurementConfig
from gsdeploy.runtime.runner import record
from . import twoFIFO_matching_execute_v1 as matching
from . import twoFIFO_prospective_contract_v1 as public
from . import twoFIFO_fresh181_prefix_v1 as prefix
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import receipt,checked
from .twoFIFO_training_manifest_v1 import require,digest,OWNERS
from .profile_owned_collect_v3 import StartupBoardMonitor,memory_ownership,MONITORING_SCOPE
from .twoFIFO_owned_collect_v3 import (ActivityLedger,atomic_save,initial_rows,history_snapshot,
    wait_until,audit_activity,finish_collection,ACTIVITY_FILE,ACTIVITY_SCHEMA,TIME_DOMAIN)

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='gsdeploy.twoFIFO_fresh181_owned_validation.v1'
RAW_SCHEMA=SCHEMA+'.capture'
MEASUREMENT_NS=90_000_000_000
DRAIN_NS=120_000_000_000
supervisor=matching.supervisor
verify=matching._verify
save=prefix.save
NATIVE_BRIDGE_HOLD='HOLD_MISSING_TYPED_MATCHING_LAW_TO_REAL_PREFIX_PROSPECTIVE_FORECAST_BRIDGE'


def require_native_forecast_bridge(plan):
    """No caller receipt/Boolean can replace the missing executable bridge.

    A future implementation must authenticate the matching review-bound law,
    replay its full worlds through the original prospective forecast interface,
    and bind the actual prefix/query/candidate. Source changes then require a
    fresh preparation/review. Current fresh181 work is CPU-only preparation.
    """
    raise UnknownEvidence(NATIVE_BRIDGE_HOLD)


def executable_sources():
    return implementation_sources(__file__,Path(__file__).with_name('semantic_progress_worker_v1.py'))


def runtime_configuration(parent,candidate,contract):
    config=copy.deepcopy(parent['runtime_config'])
    config.update(workload_rps=copy.deepcopy(contract['workload_rps']),
        routing_rps={task:{o:weight*contract['workload_rps'][task] for o,weight in weights.items()}
            for task,weights in candidate['routing_weights'].items()},
        warmup_seconds=contract['warmup_seconds'],measurement_seconds=contract['window_seconds'],
        drain_timeout_seconds=contract['drain_seconds'],
        routing_source='original_frozen181_explicit_routes_no_random_arrival_generation')
    # Runtime still consumes explicit enqueue rows. This field names the
    # retained implementation's supported arrival generator, which is unused.
    config['arrival_process']='deterministic_periodic'
    result=MeasurementConfig(**config)
    result.validate({o:contract['aliases'][o]['supported_tasks'] for o in OWNERS})
    return result


def _context_and_pool(producer):
    plans=[checked(a['plan']) for a in producer['attempts']]
    parent=plans[0];context=checked(parent['context']);pool=matching.supplement.load_pool(parent['prepared_pool'])
    require(all(p['context']==parent['context'] and p['prepared_pool']==parent['prepared_pool']
        and p['runtime_policy']==parent['runtime_policy'] and p['runtime_config']==parent['runtime_config']
        for p in plans),'One authenticated matching backend/pool/policy required')
    semantic=[matching.planned_priming_semantics(p,context) for p in plans]
    require(len({s['sha256'] for s in semantic})==1,'Matching plans must retain one ordered physical priming semantic')
    declarations={r['pool_id']:r for r in context['declared_inputs']}
    require(len(pool)==len(declarations)==22,'Exact matching union22 input domain required')
    for item in pool:
        declared=declarations[item['pool_id']]
        require(item['task_type']==declared['task'] and public.input_identity(item)==declared['exact_input_sha256'],
            'Every matching runtime input must retain exact content')
    return parent,context,pool


def _build(public_path,matching_path,namespace,out,*,lead_seconds,model_sources,generator_sources):
    require(type(namespace) is str and re.fullmatch('[a-f0-9]{32}',namespace),'Explicit fresh128-bit namespace required')
    require(type(lead_seconds) in (int,float) and 1<=lead_seconds<=300,'Predeclared prephase lead in1..300 seconds required')
    require(type(model_sources) is list and type(generator_sources) is list,'Explicit predeclared model/generator receipts required')
    for ref in model_sources+generator_sources:verify(ref)
    pub_ref=receipt(public_path);pub=public.audit(public_path)
    producer_ref=receipt(matching_path);producer=matching.audit(matching_path)
    parent,context,pool=_context_and_pool(producer)
    contract=checked(pub['artifacts']['public_contract'])
    require(len(contract['trace'])==181 and len(contract['candidates'])==4
        and contract['window_seconds']==90 and contract['drain_seconds']==120
        and contract['warmup_seconds']==0 and contract['weights']=={k:1/6 for k in 'QTLPME'},
        'Original four-candidate181 and90+120 equal-six objective required')
    require(set(contract['aliases'])==set(OWNERS) and context['residents']==
        {o:contract['aliases'][o]['family'] for o in OWNERS},'Same physical resident roster required')
    declarations={r['pool_id']:r for r in context['declared_inputs']}
    inputs=public.public_input_manifest(pub['inputs']['public_catalog'],contract,declarations)
    sources=executable_sources()
    frozen=[dict(original_path=r['path'],frozen=receipt(out/'source-freeze'/Path(r['path']).relative_to(ROOT))) for r in sources]
    require(all(x['frozen']['sha256']==r['sha256'] for x,r in zip(frozen,sources)), 'Complete fresh collector source freeze required')
    context_ref=receipt(out/'context.json')
    require(context_ref['sha256']==parent['context']['sha256'] and checked(context_ref)==context,
        'Fresh context bytes must equal authenticated matching context; IDs are plan provenance')
    evidence=strict_references([pub_ref,pub,producer_ref,producer['sources'],context_ref,parent['context'],
        parent['prepared_pool'],sources,frozen,model_sources,generator_sources,
        [receipt(r['sample']['input']) for r in pool if r['task_type']!='text']])
    attempts=[]
    training_ids={a['execution_id'] for a in producer['attempts']}
    for candidate in contract['candidates']:
        cid=candidate['id'];attempt_id='fresh181-'+namespace+'-'+cid
        execution_id=uuid.uuid5(uuid.NAMESPACE_URL,namespace+':'+cid).hex
        require(execution_id not in training_ids,'Fresh execution identity must be disjoint from training')
        folder=out/attempt_id;collection=folder/'collection'
        arrivals=[dict(request_id=r['request_id'],owner=candidate['routes'][r['request_id']],task=r['task'],
            pool_id=r['pool_id'],arrival_ns=r['offset_ns'],arrival_index=i) for i,r in enumerate(contract['trace'])]
        require(all(a['owner'] in OWNERS and a['task'] in contract['aliases'][a['owner']]['supported_tasks']
            and declarations[a['pool_id']]['task']==a['task'] for a in arrivals),'Every fixed route/input must match actual backend support')
        case=dict(case_id='original181',offered=181,arrivals=arrivals,public_arrivals_sha256=digest(arrivals))
        plan=dict(schema=SCHEMA+'.attempt',split='validation',namespace=namespace,candidate_id=cid,
            capture_id=attempt_id,attempt_id=attempt_id,execution_id=execution_id,
            context=context_ref,matching_parent_context=parent['context'],matching_producer=producer_ref,
            prepared_pool=parent['prepared_pool'],public_preparation=pub_ref,
            objective=pub['inputs']['objective'],public_contract=pub['artifacts']['public_contract'],
            historical_request_ids_retained=True,request_identity_scope='execution_id_and_request_id',
            resident_owners=list(OWNERS),runtime_policy=copy.deepcopy(parent['runtime_policy']),
            runtime_config=runtime_configuration(parent,candidate,contract).as_dict(),
            candidate=copy.deepcopy(candidate),cases=[case],measurement_ns=MEASUREMENT_NS,drain_ns=DRAIN_NS,
            priming=[dict(r,request_id=attempt_id+':priming:'+str(i)) for i,r in enumerate(parent['priming'])],
            priming_semantics=matching.planned_priming_semantics(parent,context),
            forecast_barrier=dict(schema=prefix.GATE_SCHEMA,lead_seconds=lead_seconds,poll_seconds=.1,
                prior_READY_settle_timeout_seconds=30,
                prefix_path=str(collection/'precohort-prefix.json'),gate_request_path=str(collection/'prediction-gate.json'),
                seal_path=str(collection/'prediction-seal.json'),model_sources=copy.deepcopy(model_sources),
                generator_sources=copy.deepcopy(generator_sources),
                policy='PERSIST_PREFIX_THEN_PREDICTION_SEAL_BEFORE_FIRST_ARRIVAL_OR_STOP',
                waiting_outside_measurement=True,no_target_future_label_input=True),
            collector_timing_policy=dict(no_full_copy_or_checkpoint_between_epoch_and_fixed_drain=True,
                prefix_and_prediction_persistence_before_epoch=True,waiting_excluded_from_measurement=True),
            sources=evidence,source_freeze=frozen,no_retries=True,automatic_fit=False,
            supervisor_budget=supervisor.BUDGET,preflight_free_bytes_required=supervisor.PREFLIGHT,
            execute_entrypoint='tools.predictor_validation_20261027.twoFIFO_fresh181_execute_v1.collect',
            resource_horizon_support='UNKNOWN_MATCHING10_PLUS60_NOT90_PLUS120',
            native_forecast_transfer_bridge=NATIVE_BRIDGE_HOLD,
            original_six_term_and_stability_contract_retained=True,
            fresh_prefix_is_planned_not_observed=True,predictions_sealed=False,
            main_queue_provider_eligible=False,final_gain_eligible=False,GPU_started=False)
        plan['execution_context_sha256']=digest(plan)
        attempts.append(dict(plan=plan,plan_path=folder/'plan.json',attempt_id=attempt_id,execution_id=execution_id,
            candidate_id=cid,collection=str(collection),claim=str(folder/'fresh181-attempt-claim.json')))
    require(len({a['execution_id'] for a in attempts})==4,'Four fresh candidate execution IDs required')
    return dict(public_ref=pub_ref,producer_ref=producer_ref,namespace=namespace,attempts=attempts,
        sources=evidence,context=context_ref,inputs=inputs,lead_seconds=lead_seconds,
        model_sources=model_sources,generator_sources=generator_sources)


def _manifest(built):
    return dict(schema=SCHEMA+'.manifest',status='CPU_PREPARED_NATIVE_HOLD_MISSING_REVIEW_BOUND_FORECAST_BRIDGE',
        public_preparation=built['public_ref'],matching_producer=built['producer_ref'],namespace=built['namespace'],
        context=built['context'],sources=built['sources'],public_inputs=built['inputs'],lead_seconds=built['lead_seconds'],
        model_sources=built['model_sources'],generator_sources=built['generator_sources'],
        attempts=[dict(plan=receipt(a['plan_path']),**{k:a[k] for k in ('candidate_id','execution_id','attempt_id','collection','claim')}) for a in built['attempts']],
        candidate_count=4,requests_per_candidate=181,measured_offered=724,priming_history_offered=72,
        measurement_ns=MEASUREMENT_NS,drain_ns=DRAIN_NS,no_retries=True,
        objective_and_stability_unchanged=True,historical_request_ids_retained=True,
        request_identity_scope='execution_id_and_request_id',actual_causal_prefix_available=False,
        predictions_sealed=False,automatic_fit=False,numerical_fit_run=False,validation_labels_read=False,GPU_started=False,
        resource_horizon_support='UNKNOWN_MATCHING10_PLUS60_NOT90_PLUS120',
        native_forecast_transfer_bridge=NATIVE_BRIDGE_HOLD,
        main_queue_provider_eligible=False,final_gain_eligible=False)


def prepare(public_path,matching_path,namespace,output,*,lead_seconds=60,model_paths=(),generator_paths=()):
    public.audit(public_path);producer=matching.audit(matching_path);parent,context,pool=_context_and_pool(producer)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'context.json').write_bytes(Path(parent['context']['path']).read_bytes())
    for ref in executable_sources():
        target=out/'source-freeze'/Path(ref['path']).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(ref['path']).read_bytes());require(receipt(target)['sha256']==ref['sha256'],'Source changed during freeze')
    built=_build(public_path,matching_path,namespace,out,lead_seconds=lead_seconds,
        model_sources=[receipt(p) for p in model_paths],generator_sources=[receipt(p) for p in generator_paths])
    for item in built['attempts']:save(item['plan_path'],item['plan'])
    result=save(out/'manifest.json',_manifest(built));audit(result['path']);return result


def audit(path):
    manifest=checked(receipt(path));out=Path(path).resolve().parent
    require(manifest.get('schema')==SCHEMA+'.manifest','Explicit fresh181 manifest required')
    built=_build(manifest['public_preparation']['path'],manifest['matching_producer']['path'],manifest['namespace'],out,
        lead_seconds=manifest['lead_seconds'],model_sources=manifest['model_sources'],generator_sources=manifest['generator_sources'])
    for item in built['attempts']:
        require(checked(receipt(item['plan_path']))==item['plan'],'Exact fresh181 plan/routes/window/priming/source contract changed')
    require(manifest==_manifest(built),'Exact fresh181 manifest/source/objective/eligibility changed')
    for ref in manifest['sources']:verify(ref)
    return manifest


def audit_plan(path):
    path=Path(path).resolve();manifest=audit(path.parent.parent/'manifest.json')
    require(receipt(path) in [a['plan'] for a in manifest['attempts']],'Canonical fresh181 plan required')
    return checked(receipt(path))


def initial_priming_rows(plan,pool):
    rows=matching.initial_priming_rows(plan,pool)
    for row in rows:row.update(split='validation',validation_view='prior_priming_history_not_target_labels')
    return rows


def dispatch_cohort(plan,case,runtime,pool,raw,*,clock_ns,wait_until_ns,checkpoint,monitor,activity,
                    plan_ref,manifest_ref,output,gate_callback=None):
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
    # The callback receives only an immutable public gate receipt. Native CLI
    # uses the canonical external seal file, never an arbitrary shell command.
    if gate_callback is not None:gate_callback(gate_ref)
    while not Path(barrier['seal_path']).is_file() and clock_ns()<epoch:
        monitor.check();wait_until_ns(min(epoch,clock_ns()+round(barrier['poll_seconds']*1e9)))
    require(Path(barrier['seal_path']).is_file(),'Prediction barrier timed out before the first-arrival origin; no cohort was sent')
    seal_ref=receipt(barrier['seal_path']);accepted=clock_ns()
    prefix.audit_prefix(prefix_ref,plan_ref,plan,manifest_ref,allow_cpu_fixture=raw['backend']=='CPU_TEST_ONLY')
    prefix.audit_seal(seal_ref,gate_ref,now_ns=accepted,allow_cpu_fixture=raw['backend']=='CPU_TEST_ONLY')
    require(clock_ns()<epoch,'Prediction authentication crossed the first-arrival origin; no cohort was sent')
    raw.update(prediction_seal=seal_ref,prediction_seal_accepted_ns=accepted,
        predictions_durable_before_cohort=True,forecast_generator_correctness_independently_verified=False)
    raw['case_preparations'].append(dict(case_id=case['case_id'],public_arrivals_sha256=case['public_arrivals_sha256'],
        public_arrivals=copy.deepcopy(case['arrivals']),prior_history_index=0,prepared_before_epoch=True,
        prediction_seal=seal_ref,prefix=prefix_ref))
    raw['checkpoint_context']=dict(kind='precohort_prediction_accepted',case_id=case['case_id']);checkpoint()
    require(clock_ns()<epoch,'Pre-cohort durable checkpoint crossed the origin; no cohort was sent')
    # Keep all sealed inputs live under a change/restore watch through drain.
    watched=strict_references([prefix_ref,gate_ref,seal_ref,checked(seal_ref)['prediction'],
        barrier['model_sources'],barrier['generator_sources']])
    with SourceChangeGuard(watched) as guard:
        guard.check();wait_until_ns(epoch);guard.check();monitor.set_phase('measurement')
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


def collect(plan_path,output,*,lease_fd,runtime_factory=DeploymentRuntime,monitor_factory=StartupBoardMonitor,
            clock_ns=time.perf_counter_ns,wait_until_ns=None,ownership_check=memory_ownership,gate_callback=None):
    supervisor.validate_lease(lease_fd);plan=audit_plan(plan_path);output=Path(output)
    injected=(runtime_factory is not DeploymentRuntime or monitor_factory is not StartupBoardMonitor
        or clock_ns is not time.perf_counter_ns or wait_until_ns is not None or ownership_check is not memory_ownership or gate_callback is not None)
    if not injected:require_native_forecast_bridge(plan)
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
                manifest_ref=manifest_ref,output=output,gate_callback=gate_callback)
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



def execute(plan_path,output,*,manifest_path):
    manifest_ref=receipt(manifest_path);snapshot=checked(manifest_ref)
    watched=strict_references([manifest_ref,snapshot['sources'],snapshot['attempts']])
    with SourceChangeGuard(watched) as guard:
        guard.check();verify(manifest_ref);manifest=audit(manifest_path)
        require(manifest==snapshot,'Guarded fresh181 manifest changed')
        plan_ref=receipt(plan_path);item=next(a for a in manifest['attempts'] if a['plan']==plan_ref)
        plan=audit_plan(plan_path);verify(plan_ref)
        require_native_forecast_bridge(plan)
        require(str(Path(output).resolve())==item['collection'],'Canonical fresh181 collection path required')
        require(not Path(output).exists() and not Path(item['claim']).exists(),'Started/claimed fresh181 attempt cannot be retried')
        supervisor.public_environment_gate();guard.check()
        claim=dict(schema=SCHEMA+'.claim',manifest=manifest_ref,plan=plan_ref,
            execution_id=plan['execution_id'],attempt_id=plan['attempt_id'],candidate_id=plan['candidate_id'],no_retries=True)
        claim_ref=save(item['claim'],claim)
        command=[sys.executable,'-m','tools.predictor_validation_20261027.twoFIFO_fresh181_execute_v1','_child',
            '--manifest',str(Path(manifest_path).resolve()),'--plan',str(Path(plan_path).resolve()),
            '--output',str(Path(output).resolve()),'--execute']
        with SourceChangeGuard([claim_ref]) as claim_guard:
            def gate():
                guard.check();claim_guard.check();verify(manifest_ref);audit(manifest_path);audit_plan(plan_path)
                supervisor.public_environment_gate();guard.check();claim_guard.check()
            gate()
            result=supervisor.supervise(command,output,
                environment={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                before_launch=gate,after_run=gate,metadata=dict(schema=SCHEMA+'.execution_binding',
                    manifest=manifest_ref,plan=plan_ref,execution_id=plan['execution_id'],attempt_id=plan['attempt_id'],
                    candidate_id=plan['candidate_id'],raw_schema=RAW_SCHEMA,claim=claim_ref,
                    predictions_required_before_cohort=True,automatic_fit=False))
            guard.check();claim_guard.check();return result


def completed_execution(execution,manifest_ref,item):
    """Fresh schema only; original training execution cannot impersonate it."""
    claim_ref=receipt(item['claim'])
    require(checked(claim_ref)==dict(schema=SCHEMA+'.claim',manifest=manifest_ref,plan=item['plan'],
        execution_id=item['execution_id'],attempt_id=item['attempt_id'],candidate_id=item['candidate_id'],no_retries=True),
        'Exact exclusive fresh181 claim required')
    require(execution.get('profile_execution')==dict(schema=SCHEMA+'.execution_binding',manifest=manifest_ref,
        plan=item['plan'],execution_id=item['execution_id'],attempt_id=item['attempt_id'],candidate_id=item['candidate_id'],
        raw_schema=RAW_SCHEMA,claim=claim_ref,predictions_required_before_cohort=True,automatic_fit=False)
        and execution.get('budget')==supervisor.BUDGET
        and execution.get('status')=='CHILD_EXITED_REQUIRES_PROFILE_REVIEW'
        and execution.get('collection_subprocess_launched') is True
        and type(execution.get('process_group')) is int and execution['process_group']>0
        and execution.get('child_reaped') is True and execution.get('live_group_pids')==[]
        and execution.get('experiment_lock_held_through_group_exit') is True
        and execution.get('all_owned_descendants_exited_and_reaped') is True
        and execution.get('final_source_audit')=='PASS'
        and execution.get('watchdog',{}).get('status')=='EXITED' and execution['watchdog'].get('return_code')==0,
        'Failed/old/foreign/incomplete fresh181 execution rejected; no retry')


def audit_completed_prefix(manifest_path,plan_path,execution_path,output):
    """Post-run source/order receipt; deliberately reads no target raw labels.

    This is narrower than a native protocol/quality/resource review: it binds
    the source-reviewed owned collector and its prefix/barrier to the completed
    supervised execution. It does not independently reparse worker stdout.
    """
    manifest_ref=receipt(manifest_path);manifest=audit(manifest_path);plan_ref=receipt(plan_path)
    item=next(a for a in manifest['attempts'] if a['plan']==plan_ref);plan=audit_plan(plan_path)
    require(Path(execution_path).resolve()==Path(item['collection'])/'execution.json','Canonical fresh execution receipt required')
    execution_ref=receipt(execution_path);execution=checked(execution_ref);completed_execution(execution,manifest_ref,item)
    barrier=plan['forecast_barrier'];prefix_ref=receipt(barrier['prefix_path']);gate_ref=receipt(barrier['gate_request_path'])
    value=prefix.audit_prefix(prefix_ref,plan_ref,plan,manifest_ref)
    prefix.inspect_native_ownership(value['prefix_snapshot'],value['ready_ownership_evidence'],collector_pid=execution['process_group'])
    gate=prefix.audit_gate(gate_ref);seal_ref=receipt(barrier['seal_path']);seal=checked(seal_ref)
    # The seal itself is a durable pre-origin source receipt. Acceptance time
    # is independently checked by the collector; this narrower review does not
    # reconstruct acceptance from a later target raw.
    prefix.audit_seal(seal_ref,gate_ref,now_ns=seal['seal_begin_ns'])
    return save(output,dict(schema=SCHEMA+'.completed_prefix_source_review',
        status='PASS_OWNED_EXECUTION_BOUND_PREFIX_AND_DECLARED_SEAL_NOT_FULL_PROTOCOL_REVIEW',
        manifest=manifest_ref,plan=plan_ref,execution=execution_ref,claim=receipt(item['claim']),
        prefix=prefix_ref,gate=gate_ref,prediction_seal=seal_ref,prediction=seal['prediction'],
        execution_id=plan['execution_id'],candidate_id=plan['candidate_id'],
        prefix_source_bound_to_completed_owned_execution=True,
        native_protocol_stdout_independently_reparsed=False,target_raw_or_future_labels_read=False,
        collector_acceptance_clock_independently_reconstructed=False,
        independent_accuracy_qualified=False,resource_horizon_qualified=False,full_U_qualified=False,
        source_bindings=strict_references([manifest_ref,plan_ref,execution_ref,prefix_ref,gate_ref,seal_ref,
            seal['prediction'],plan['sources'],gate['model_sources'],gate['forecast_generator_sources']])) )


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    q=sub.add_parser('prepare')
    for arg in ('public-preparation','matching-producer','namespace','output'):q.add_argument('--'+arg,required=True)
    q.add_argument('--lead-seconds',type=float,default=60);q.add_argument('--model-sources',nargs='*',default=[])
    q.add_argument('--generator-sources',nargs='*',default=[])
    q=sub.add_parser('audit');q.add_argument('--manifest',required=True)
    q=sub.add_parser('audit-prefix')
    for arg in ('manifest','plan','execution','output'):q.add_argument('--'+arg,required=True)
    for name in ('execute','_child'):
        q=sub.add_parser(name)
        for arg in ('manifest','plan','output'):q.add_argument('--'+arg,required=True)
        q.add_argument('--execute',required=True,action='store_true')
    a=p.parse_args()
    if a.op=='prepare':result=prepare(a.public_preparation,a.matching_producer,a.namespace,a.output,
        lead_seconds=a.lead_seconds,model_paths=a.model_sources,generator_paths=a.generator_sources)
    elif a.op=='audit':result=audit(a.manifest)
    elif a.op=='audit-prefix':result=audit_completed_prefix(a.manifest,a.plan,a.execution,a.output)
    elif a.op=='execute':result=execute(a.plan,a.output,manifest_path=a.manifest)
    else:
        manifest=audit(a.manifest);plan=audit_plan(a.plan)
        item=next(i for i in manifest['attempts'] if i['plan']==receipt(a.plan))
        require(item['collection']==str(Path(a.output).resolve()),'Canonical child output required')
        claim=checked(receipt(item['claim']))
        require(claim==dict(schema=SCHEMA+'.claim',manifest=receipt(a.manifest),plan=receipt(a.plan),
            execution_id=plan['execution_id'],attempt_id=plan['attempt_id'],candidate_id=plan['candidate_id'],no_retries=True),
            'Exact exclusive fresh181 claim required')
        result=collect(a.plan,a.output,lease_fd=int(os.environ.get(supervisor.LEASE_FD,'-1')))
    print(json.dumps(result))
    return 1 if result.get('status') in ('failed','FAILED_NO_RETRY','STOPPED_NO_RETRY') else 0


if __name__=='__main__':raise SystemExit(main())
