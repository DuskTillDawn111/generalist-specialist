"""Explicit matching-supplement producer; preparation is CPU only.

The collection body preserves reviewed source-rebound dispatch/checkpoint/cleanup.
Only this module authenticates the new plan/schema. No old audit is patched.
A separate invocation and root review are required before GPU collection.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from . import twoFIFO_matching_supplement_v1 as supplement
from . import twoFIFO_owned_collect_v3 as original
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .stage_response_features_v1 import receipt, checked
from .twoFIFO_training_manifest_v1 import require, digest, OWNERS, MEASUREMENT_NS, DRAIN_NS
from .profile_owned_collect_v3 import StartupBoardMonitor, memory_ownership, MONITORING_SCOPE
from .twoFIFO_owned_collect_v3 import (ActivityLedger, atomic_save, initial_rows,
    runtime_config, wait_until, dispatch_case, audit_activity, finish_collection,
    ACTIVITY_FILE, ACTIVITY_SCHEMA, TIME_DOMAIN)

SCHEMA = 'gsdeploy.twoFIFO_matching_owned_training.v1'
RAW_SCHEMA = SCHEMA + '.capture'
CONT_SCHEMA = SCHEMA + '.continuation'
supervisor = original.supervisor


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _verify(ref):
    require(type(ref) is dict and set(ref) == {'path','sha256'} and receipt(ref['path']) == ref,
            'Immutable matching source/evidence receipt changed')


def executable_sources():
    here = Path(__file__)
    return implementation_sources(here, here.with_name('semantic_progress_worker_v1.py'),
        here.with_name('twoFIFO_matching_review_v1.py'),
        here.with_name('twoFIFO_matching_macros_v1.py'),
        here.with_name('twoFIFO_matching_intake_v1.py'))


def planned_priming_semantics(plan,context):
    """Ordered physical input semantics; request IDs remain provenance.

    This is a public-plan projection, never an actual-native-history claim.
    The SHA encoding matches legacy.priming_identity for later reviewed raw.
    """
    declared={r['pool_id']:r for r in context['declared_inputs']}
    require(len(plan['priming']) == 18,'Eighteen ordered priming semantics required')
    rows=[]
    for row in plan['priming']:
        entry=declared[row['pool_id']]
        require(row['owner'] in OWNERS and entry['task']==row['task']
                and entry['exact_input_sha256']==row['exact_input_sha256'],
                'Priming planned content differs from exact context declaration')
        rows.append(dict(model=row['owner'],task=row['task'],pool_id=row['pool_id'],
            exact_input_sha256=row['exact_input_sha256']))
    return dict(ordered_inputs=rows,
        sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        request_ids_are_provenance_not_physical_axes=True,actual_native_history_qualified=False)


def _frozen_sources(out, sources):
    root = Path(__file__).resolve().parents[2]
    return [dict(original_path=r['path'], frozen=receipt(out/'source-freeze'/Path(r['path']).relative_to(root)))
            for r in sources]


def _build(supplement_path, output):
    public_ref = receipt(supplement_path)
    public = supplement.audit(supplement_path)
    out = Path(output).resolve()
    sources = executable_sources()
    freeze = _frozen_sources(out, sources)
    require(len(freeze) == len(sources) and all(a['frozen']['sha256'] == b['sha256']
            for a,b in zip(freeze,sources)), 'Complete current matching executable freeze required')
    context = public['artifacts']['candidate_context']; pool_ref = public['artifacts']['pool']
    pool = supplement.load_pool(pool_ref)
    evidence = strict_references([public_ref, public, context, pool_ref,
        [receipt(r['sample']['input']) for r in pool if r['task_type'] != 'text'],
        sources, freeze])
    attempts = []
    for block in public['blocks']:
        for seed_ref in block['attempts']:
            seed = checked(seed_ref)
            plan = copy.deepcopy(seed)
            plan.update(schema=SCHEMA+'.attempt', context=context, prepared_pool=pool_ref,
                source_freeze=freeze, sources=evidence, matching_supplement=public_ref,
                matching_supplement_plan=seed_ref,
                execute_entrypoint='tools.predictor_validation_20261027.twoFIFO_matching_execute_v1.collect',
                native_CUDA_API_numeric_authentication='UNQUALIFIED',main_queue_provider_eligible=False,
                producer_policy=dict(no_global_admission_cap=True, natural_shared=True,
                    all_block_shards_required=True, priming_history_separate_from_labels=True,
                    resource_scope='startup_through_final_live_sample_before_shutdown',
                    measurement_energy_excludes_drain=True, memory_is_owned_sampled_not_continuous=True))
            plan['priming'] = [dict(row,request_id=plan['attempt_id']+':priming:'+str(i))
                              for i,row in enumerate(seed['priming'])]
            require(len(plan['priming']) == 18, 'Exactly eighteen planned history requests required')
            plan['priming_semantics']=planned_priming_semantics(plan,checked(context))
            plan['execution_context_sha256'] = digest(plan)
            plan_path = out/plan['attempt_id']/'plan.json'
            attempts.append(dict(plan=plan,plan_path=plan_path,attempt_id=plan['attempt_id'],
                execution_id=plan['execution_id'],collection=str(plan_path.parent/'collection'),
                claim=str(plan_path.parent/'matching-attempt-claim.json')))
    require(len(attempts) == 15 and len({a['plan']['capture_id'] for a in attempts}) == 3,
            'Exactly fifteen attempts in three whole training blocks required')
    measured = [r['request_id'] for a in attempts for c in a['plan']['cases'] for r in c['arrivals']]
    primes = [r['request_id'] for a in attempts for r in a['plan']['priming']]
    require(len(measured)==390 and len(primes)==270 and len(set(measured+primes))==660,
            '390 measured and 270 disjoint owned-history request identities required')
    require(all(a['plan']['runtime_config'] == runtime_config().as_dict()
                and a['plan']['measurement_ns'] == MEASUREMENT_NS
                and a['plan']['drain_ns'] == DRAIN_NS
                and a['plan']['supervisor_budget'] == supervisor.BUDGET
                and a['plan']['preflight_free_bytes_required'] == supervisor.PREFLIGHT
                for a in attempts), 'Unchanged runtime/fixed windows/2400s/18GiB required')
    return dict(public_ref=public_ref,sources=evidence,attempts=attempts)


def _manifest(built):
    return dict(schema=CONT_SCHEMA,status='CPU_PREPARED_EXPLICIT_EXECUTE_REQUIRED',
        matching_supplement=built['public_ref'],
        attempts=[dict(plan=receipt(a['plan_path']), **{k:a[k] for k in
            ('attempt_id','execution_id','collection','claim')}) for a in built['attempts']],
        sources=built['sources'],no_retries=True,GPU_started=False,numerical_fit_run=False,
        validation_labels_read=False,automatic_fit=False,raw_schema=RAW_SCHEMA,
        downstream_adapter_status='EXPLICIT_MATCHING_NATIVE_REVIEW_V1',
        supervisor_budget=supervisor.BUDGET,preflight_free_bytes_required=supervisor.PREFLIGHT,
        training_blocks=3,attempt_count=15,case_count=171,measured_offered=390,
        priming_history_offered=270,leaveout_unit='whole_capture_block_all_five_shards',
        fitted_model_qualified=False,kernel_resource_fit=False,main_queue_provider_eligible=False,
        final_gain_eligible=False)


def prepare(supplement_path, output):
    # Authenticate first, so a failed contract never leaves a partial producer.
    supplement.audit(supplement_path)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    for source in executable_sources():
        target=out/'source-freeze'/Path(source['path']).relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(source['path']).read_bytes())
    built=_build(supplement_path,out)
    for item in built['attempts']:
        item['plan_path'].parent.mkdir();_write(item['plan_path'],item['plan'])
    _write(out/'manifest.json',_manifest(built));audit(out/'manifest.json')
    return receipt(out/'manifest.json')


def audit(path):
    manifest=_read(path);out=Path(path).resolve().parent
    require(manifest.get('schema')==CONT_SCHEMA,'Explicit matching producer schema required')
    _verify(manifest['matching_supplement'])
    built=_build(manifest['matching_supplement']['path'],out)
    for item in built['attempts']:
        require(_read(item['plan_path']) == item['plan'], 'Exact matching plan/priming/calendar/source contract changed')
    require(manifest == _manifest(built), 'Exact matching producer manifest/order/budget/eligibility contract changed')
    for source in manifest['sources']:_verify(source)
    return manifest


def audit_plan(path):
    path=Path(path).resolve();manifest=audit(path.parent.parent/'manifest.json')
    require(receipt(path) in [a['plan'] for a in manifest['attempts']],
            'Only canonical matching producer plans may execute')
    return _read(path)


def initial_priming_rows(plan,pool):
    rows=[]
    for index,item in enumerate(plan['priming']):
        row=record('priming',index,item['task'],item['owner'],pool[item['pool_id']],None)
        row.update(request_id=item['request_id'],profile_job_id='priming',capture_id=plan['capture_id'],
            attempt_id=plan['attempt_id'],split='training',validation_view='training',status='not_started')
        rows.append(row)
    return rows


def collect(plan_path,output,*,lease_fd,runtime_factory=DeploymentRuntime,monitor_factory=StartupBoardMonitor,
            clock_ns=time.perf_counter_ns,wait_until_ns=None,ownership_check=memory_ownership):
    supervisor.validate_lease(lease_fd);plan=audit_plan(plan_path);output=Path(output)
    require(not (output/'raw.json').exists(),'No overwrite/retry of a collection attempt')
    context=checked(plan['context']);pool={r['pool_id']:r for r in (json.loads(l) for l in Path(plan['prepared_pool']['path']).read_text().splitlines())}
    registry={}
    for alias in OWNERS:
        family=alias.split('#')[0];binding=copy.deepcopy(context['worker_bindings'][family])
        binding['command'][1]=str(Path(__file__).with_name('semantic_progress_worker_v1.py'))
        binding['env'].update(context['threads']);binding['env']['PYTHONUNBUFFERED']='1';registry[alias]=binding
    runtime=runtime_factory(runtime_config(),registry);monitor=None
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
        matching_executor_sources=implementation_sources(__file__))
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
            require(row['status']=='completed','Matching priming failed')
        raw['lifecycle_boundaries'].append(dict(boundary='priming_end',time_ns=clock_ns()))
        raw['checkpoint_context']=dict(kind='postpriming',case_id=None);checkpoint()
        for case in plan['cases']:
            result=dispatch_case(plan,case,runtime,pool,raw,clock_ns=clock_ns,wait_until_ns=wait,
                checkpoint=checkpoint,monitor=monitor,activity=activity)
            if result['status']!='completed':raise RuntimeError('Incomplete case; subsequent cases retained NOT_STARTED')
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



def execute(plan_path, output, *, manifest_path):
    # Only the receipt-checked snapshot is read before the watch is armed.
    # Every full audit and generated-plan read occurs under this outer guard.
    manifest_ref=receipt(manifest_path);snapshot=checked(manifest_ref)
    watched=strict_references([manifest_ref,snapshot['sources'],snapshot['attempts']])
    with SourceChangeGuard(watched) as source_guard:
        source_guard.check();_verify(manifest_ref)
        manifest=audit(manifest_path)
        require(manifest==snapshot,'Matching manifest snapshot changed before guarded audit')
        plan_ref=receipt(plan_path)
        item=next(a for a in manifest['attempts'] if a['plan']==plan_ref)
        plan=audit_plan(plan_path)
        _verify(plan_ref);require(checked(plan_ref)==plan,'Guarded matching plan read changed')
        require(Path(output).resolve()==Path(item['collection']),'Canonical matching collection path required')
        require(not Path(output).exists(),'Attempt already started; no retry of success/failure/unfinished collection')
        require(not Path(item['claim']).exists(),'Attempt already claimed; no retry even if launch failed')
        supervisor.public_environment_gate();source_guard.check()
        _write(item['claim'],dict(schema=SCHEMA+'.claim',manifest=manifest_ref,plan=plan_ref,
            execution_id=plan['execution_id'],attempt_id=plan['attempt_id'],no_retries=True))
        claim_ref=receipt(item['claim'])
        command=[sys.executable,'-m','tools.predictor_validation_20261027.twoFIFO_matching_execute_v1',
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
        'Exact durable no-retry matching claim required')
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


def run(manifest_path, output, *, execute_enabled=False):
    """Review and rederive macros before every next attempt; never fit."""
    from . import twoFIFO_matching_review_v1 as native_review
    from . import twoFIFO_matching_macros_v1 as macros
    manifest_ref = receipt(manifest_path); manifest = audit(manifest_path)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    lock = Path(manifest_path).resolve().parent/'matching-continuation.lock'
    with lock.open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = dict(schema=CONT_SCHEMA+'.run', manifest=manifest_ref, no_retries=True,
            execute_invocations=0, numerical_fit_run=False, automatic_fit=False, validation_labels_read=False,
            completed_attempts=[])
        try:
            for item in manifest['attempts']:
                _verify(manifest_ref); audit(manifest_path)
                result['attempt_id'] = item['attempt_id']
                collection = Path(item['collection']); final = collection/'execution.json'
                if collection.exists() and not final.is_file():
                    result['status'] = 'WAITING_EXISTING_ATTEMPT_FINAL_RECEIPT'; break
                if not collection.exists() and Path(item['claim']).exists():
                    result['status'] = 'HOLD_CLAIMED_ATTEMPT_NO_RETRY'; break
                if not collection.exists() and not execute_enabled:
                    result['status'] = 'READY_EXPLICIT_EXECUTE_REQUIRED'; break
                if not collection.exists():
                    result['execute_invocations'] += 1
                    execute(item['plan']['path'], collection, manifest_path=manifest_path)
                require(final.is_file(), 'Executor returned without final receipt; no retry')
                execution_ref = receipt(final); completed_execution(checked(execution_ref), manifest_ref, item)
                raw_ref = receipt(collection/'raw.json'); raw = checked(raw_ref)
                require(raw.get('schema') == RAW_SCHEMA and raw.get('collector_schema') == RAW_SCHEMA
                    and raw.get('plan') == item['plan'] and raw.get('execution_id') == item['execution_id'],
                    'Exact new-schema raw binding required; old v3 capture is not rebound evidence')
                check_dir = output/item['attempt_id']; check_dir.mkdir()
                review_ref = native_review.review(raw_ref['path'], check_dir/'native-review', final,
                    manifest_path=manifest_path)
                reviewed = checked(review_ref)
                require(reviewed.get('status') == 'PASS_COMPLETE_MATCHING_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION'
                    and reviewed.get('schema') == native_review.SCHEMA
                    and reviewed.get('matching_manifest') == manifest_ref
                    and reviewed.get('training_feature_extraction_eligible') is True and reviewed.get('issues') == [],
                    'New native matching review NOT_ELIGIBLE; stop all following attempts')
                facts = native_review.resource_facts(raw)
                _write(check_dir/'observed-resources.json', facts)
                require(facts['startup']['energy_joules'] is not None and all(
                    r['measurement_energy_joules'] is not None for r in facts['cases']),
                    'Original power-gap gate: startup and every measurement energy must be observable')
                macro_doc = macros.extract(review_ref['path'])
                require(macro_doc['matching_manifest'] == manifest_ref,
                    'Macro source rebound binding drift')
                _write(check_dir/'macros-v1.json', macro_doc)
                _verify(manifest_ref); audit(manifest_path); _verify(execution_ref); _verify(raw_ref)
                result['completed_attempts'].append(dict(attempt_id=item['attempt_id'], execution=execution_ref,
                    training_review=review_ref, macros=receipt(check_dir/'macros-v1.json'),
                    observed_resources=receipt(check_dir/'observed-resources.json')))
            else:
                result.update(status='COMPLETE_ALL_FIFTEEN_MATCHING_NATIVE_ATTEMPTS_REVIEWED_NO_FIT',
                    next_gate='NEW_SOURCE_BOUND_INTAKE_AND_COORDINATOR_REQUIRED', fit_eligible=False)
        except BaseException as exc:
            result.update(status='STOPPED_NO_RETRY', error=type(exc).__name__+': '+str(exc))
        _write(output/'report.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='op', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--supplement', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('audit'); p.add_argument('--manifest', required=True)
    p = sub.add_parser('run'); p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    p.add_argument('--execute', action='store_true')
    for op in ('execute', '_child'):
        p = sub.add_parser(op); p.add_argument('--manifest', required=True); p.add_argument('--plan', required=True)
        p.add_argument('--output', required=True); p.add_argument('--execute', required=True, action='store_true')
    a = parser.parse_args()
    if a.op == 'prepare': result = prepare(a.supplement, a.output)
    elif a.op == 'audit': result = audit(a.manifest)
    elif a.op == 'run': result = run(a.manifest, a.output, execute_enabled=a.execute)
    elif a.op == 'execute': result = execute(a.plan, a.output, manifest_path=a.manifest)
    else:
        manifest = audit(a.manifest); plan = audit_plan(a.plan)
        require(any(i['plan'] == receipt(a.plan) and i['collection'] == str(Path(a.output).resolve())
            for i in manifest['attempts']), 'Child canonical manifest/plan/output binding required')
        claim = _read(Path(a.plan).parent/'matching-attempt-claim.json')
        require(claim == dict(schema=SCHEMA+'.claim', manifest=receipt(a.manifest), plan=receipt(a.plan),
            execution_id=plan['execution_id'], attempt_id=plan['attempt_id'], no_retries=True), 'Exact exclusive attempt claim required')
        result = collect(a.plan, a.output, lease_fd=int(os.environ.get(supervisor.LEASE_FD, '-1')))
    print(json.dumps(result))
    return 1 if result.get('status') in ('failed', 'FAILED_NO_RETRY', 'STOPPED_NO_RETRY') else 0


if __name__ == '__main__': raise SystemExit(main())
