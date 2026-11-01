"""Single text weight owner plus active G2/OCR on one declared arrival clock.

Prepare and injected CPU tests never load GPU models. Real collection is an
explicit one-attempt operation. These three-owner profiles cannot inherit the
old four-owner startup, memory, capacity g(n), or prediction calibration.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from gsdeploy.runtime.batching import BatchConfiguration
from gsdeploy.runtime.batch_owner import run_arrivals
from . import batch_profile_collect_v2 as v2

ROOT=v2.ROOT
read,save,receipt,checked,digest=v2.read,v2.save,v2.receipt,v2.checked,v2.digest
SCHEMA='gsdeploy.active_background_batch_profile_collection.v3'
TEXT=('text#0','text#1')
BACKGROUND=('generalist_2#0','ocr#0')
PHYSICAL=('shared_text',*BACKGROUND)


def _partition(samples, arrivals, *, measurement_ns=None):
    if not samples or not arrivals:raise ValueError('Nonempty complete sample/cohort declaration required')
    index={s['id']:s for s in samples}
    if len(index)!=len(samples):raise ValueError('Distinct sample identities required')
    ids=set();normalized=[]
    for position,row in enumerate(arrivals):
        if (not isinstance(row,dict) or not isinstance(row.get('request_id'),str)
                or not row['request_id'] or row['request_id'] in ids
                or row.get('sample_id') not in index or row.get('alias') not in (*TEXT,*BACKGROUND)
                or type(row.get('arrival_ns')) is not int or row['arrival_ns']<0
                or measurement_ns is not None and row['arrival_ns']>=measurement_ns):
            raise ValueError('Explicit unique scheduled cohort/owner/input/arrival required')
        sample=index[row['sample_id']];task=sample['task'];alias=row['alias']
        if (row.get('task',task)!=task or task not in ('text','image','asr','ocr')
                or alias in TEXT and task!='text' or alias=='ocr#0' and task!='ocr'):
            raise ValueError('Declared task does not match physical owner capability')
        ids.add(row['request_id'])
        normalized.append(dict(row,task=task,pool_id=row.get('pool_id',sample['id']),arrival_index=position))
    if not {r['alias'] for r in normalized}&set(TEXT):
        raise ValueError('At least one logical text route required')
    if not set(BACKGROUND)<={r['alias'] for r in normalized}:
        raise ValueError('Both G2 and OCR must have declared active requests')
    return index,normalized


def _fingerprint(sample, media_receipts):
    # Labels are not model input. Equal text context/question is still overlap
    # if sample IDs or answer annotations differ. Media bytes are source-bound.
    if sample['task']=='text':return digest(['text',sample['context'],sample['question']])
    media=[]
    for field in ('input','image','image_path','audio','audio_path','path'):
        if isinstance(sample.get(field),str):
            path=Path(sample[field]).resolve()
            if not path.is_file():raise ValueError('Declared media file unavailable: '+str(path))
            ref=receipt(path);media_receipts[str(path)]=ref;media.append(ref['sha256'])
            if field=='input' and sample.get('input_sha256',ref['sha256'])!=ref['sha256']:
                raise ValueError('Declared media hash differs from actual input bytes')
    if not media:raise ValueError('Nontext inputs require source-bound media bytes')
    # Metadata/annotation changes cannot turn identical model media into a
    # disjoint validation input. This is deliberately conservative.
    return digest([sample['task'],sorted(media)])


def prepare(output,spec_path,source_root,samples_path,schedules_path,priming_path,configuration,
            *,measurement_ns,drain_timeout_seconds,supervisor_timeout_seconds,background_registry=None):
    if (type(measurement_ns) is not int or measurement_ns<=0
            or type(drain_timeout_seconds) not in (int,float) or not 0<drain_timeout_seconds<86400):
        raise ValueError('Explicit positive measurement window and bounded drain timeout required')
    if (type(supervisor_timeout_seconds) not in (int,float)
            or not 0<supervisor_timeout_seconds<=86400):
        raise ValueError('Explicit bounded whole-collector supervisor timeout required')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    samples,schedules,priming=map(read,(samples_path,schedules_path,priming_path))
    if any(set(doc)!={'training','validation'} for doc in (samples,schedules,priming)):
        raise ValueError('Explicit training/validation declarations required')
    normalized={};media={};fingerprints={};request_ids={};sample_ids={}
    for split in ('training','validation'):
        if set(priming[split])!={'samples','arrivals'}:raise ValueError('Explicit split priming samples and arrivals required')
        index,rows=_partition(samples[split],schedules[split],measurement_ns=measurement_ns)
        pi,pr=_partition(priming[split]['samples'],priming[split]['arrivals'])
        if any(index[sid]!=pi[sid] for sid in set(index)&set(pi)):
            raise ValueError('Same split sample identity cannot denote different priming/model inputs')
        physical=lambda r:'shared_text' if r['alias'] in TEXT else r['alias']
        measured_paths={(physical(r),r['task']) for r in rows}
        if not measured_paths<={(physical(r),r['task']) for r in pr}:
            raise ValueError('Priming must cover every measured physical owner/task path')
        measured_ids={r['request_id'] for r in rows};prime_ids={r['request_id'] for r in pr}
        if measured_ids&prime_ids:raise ValueError('Priming identities cannot enter measured cohort')
        request_ids[split]=measured_ids|prime_ids
        sample_ids[split]=set(index)|set(pi)
        fingerprints[split]={_fingerprint(s,media) for s in [*samples[split],*priming[split]['samples']]}
        normalized[split]=dict(samples=samples[split],arrivals=rows,
            priming=dict(samples=priming[split]['samples'],arrivals=pr))
    if (request_ids['training']&request_ids['validation'] or sample_ids['training']&sample_ids['validation']
            or fingerprints['training']&fingerprints['validation']):
        raise ValueError('Complete training/validation input or request identity overlap')
    if background_registry is not None:
        registry=read(background_registry)
        for family in ('generalist_2','ocr'):
            entry=registry[family]
            if entry.get('resource_binding',{}).get('mode','shared')!='shared':
                raise ValueError('This mechanism collector declares shared execution, no admission cap')
            command=entry['command']
            if Path(command[1]).name not in ('observed_worker.py','semantic_worker.py'):
                raise ValueError('Background workers must expose START/RESULT/WORKER_READY boundaries')
            if (command.count('--spec')!=1 or Path(command[command.index('--spec')+1]).resolve()!=Path(entry['spec']).resolve()
                    or '--execute' not in command):
                raise ValueError('Background command --spec must match frozen registry spec metadata')
    output.mkdir(parents=True,exist_ok=False)
    projection=output/'text-projection';projection.mkdir()
    save(projection/'samples.json',{s:[x for x in samples[s] if x['task']=='text'] for s in samples})
    save(projection/'arrivals.json',{s:[r for r in normalized[s]['arrivals'] if r['alias'] in TEXT] for s in samples})
    base=v2.prepare(output/'text-owner',spec_path,source_root,projection/'samples.json',
        projection/'arrivals.json',configuration,background_registry=background_registry)
    partitions={}
    for split,part in normalized.items():
        directory=output/split;directory.mkdir()
        for key,value in part.items():save(directory/(key+'.json'),value)
        partitions[split]={key:receipt(directory/(key+'.json')) for key in part}
        partitions[split].update(requested=len(part['arrivals']),
            offered_by_physical_owner={owner:sum(('shared_text' if r['alias'] in TEXT else r['alias'])==owner for r in part['arrivals']) for owner in PHYSICAL})
    deployment=dict(schema=SCHEMA+'.physical_deployment',physical_owners=list(PHYSICAL),
        logical_text_aliases=list(TEXT),logical_text_aliases_share_one_weight_owner=True,
        resident_weight_copies=3,text_weight_copies=1,background_activity='DECLARED_ACTIVE_G2_AND_OCR',
        execution='shared; one FIFO request stream per background owner and real shared text batching',
        baseline_four_owner_profiles_reused=False,baseline_capacity_g_inherited=False,
        startup_identity_inherited=False,memory_identity_inherited=False,energy_identity_inherited=False,
        prediction_profiles='MISSING_REQUIRES_NEW_THREE_PHYSICAL_OWNER_COLLECTION_AND_QUALIFICATION',
        global_request_concurrency_cap=None,common_submatrix_merge=False,prefix_cache_reuse=False)
    save(output/'deployment.json',deployment)
    sources=[receipt(__file__),receipt(spec_path),receipt(samples_path),receipt(schedules_path),receipt(priming_path),*media.values()]
    plan=dict(schema=SCHEMA,status='CPU_PREPARED_GPU_UNVERIFIED',text_owner_plan=receipt(output/'text-owner/plan.json'),
        deployment=receipt(output/'deployment.json'),partitions=partitions,source_receipts=sources,
        measurement_ns=measurement_ns,drain_timeout_seconds=float(drain_timeout_seconds),
        supervisor_timeout_seconds=float(supervisor_timeout_seconds),
        supervisor_scope='Entire startup/priming/measurement/drain/cleanup; external process-group watchdog, not request admission',
        execution_blockers=base['execution_blockers'],
        arrival_clock='One monotonic epoch after all three owners load and explicit split priming completes',
        cohort='Every offered request is preregistered; offsets retained through prepare/submit/parent receipt; no success-only denominator',
        resources='New startup/priming board energy; measurement energy excludes drain; owned sampled VRAM startup-through-drain; no old profile reuse',
        priming_policy='Entire frozen split priming cohort, covering every measured physical-owner/task path; largest-input/startup qualification remains separate',
        host_joint_prediction_profile='MISSING_NEW_ACTIVE_BACKGROUND_HOST/PREPARE/RECEIPT_PROFILE_REQUIRED',
        fit_eligible=False,final_gain_eligible=False,GPU_started=False)
    save(output/'plan.json',plan);return plan


def audit(plan_path):
    plan=read(plan_path)
    if plan.get('schema')!=SCHEMA:raise ValueError('Wrong active-background plan schema')
    for ref in [*plan['source_receipts'],plan['text_owner_plan'],plan['deployment']]:checked(ref)
    v2.audit(checked(plan['text_owner_plan']))
    for part in plan['partitions'].values():
        for key in ('samples','arrivals','priming'):checked(part[key])
    return plan


def background_records(arrivals,epoch_ns,*,phase):
    return [dict(request_id=r['request_id'],sample_id=r['sample_id'],pool_id=r['pool_id'],
        model=r['alias'],task=r['task'],arrival_index=r['arrival_index'],phase=phase,
        scheduled_offset_ns=r['arrival_ns'],scheduled_time_ns=epoch_ns+r['arrival_ns'],
        actual_submit_time_ns=None,completion_time_ns=None,status='planned')
        for r in arrivals if r['alias'] in BACKGROUND]


class BackgroundDispatcher:
    """Preregister every background request, then dispatch independently of text."""
    def __init__(self,runtime,samples,arrivals,epoch_ns,*,clock=time.perf_counter_ns,phase='measurement'):
        self.runtime=runtime;self.samples={s['id']:s for s in samples};self.clock=clock
        self.records=background_records(arrivals,epoch_ns,phase=phase)
        self.cancel=threading.Event();self.error=None;self.thread=None

    def _run(self):
        try:
            for row in sorted(self.records,key=lambda r:(r['scheduled_time_ns'],r['arrival_index'])):
                while self.clock()<row['scheduled_time_ns']:
                    if self.cancel.wait(min(.005,(row['scheduled_time_ns']-self.clock())/1e9)):return
                if self.cancel.is_set():return
                row['arrival_observed_time_ns']=self.clock()
                row['arrival_observation_delay_ns']=row['arrival_observed_time_ns']-row['scheduled_time_ns']
                self.runtime.enqueue(row,copy.deepcopy(self.samples[row['sample_id']]))
        except BaseException as exc:self.error=repr(exc)

    def start(self):
        self.thread=threading.Thread(target=self._run,name='active-background-arrivals',daemon=True)
        self.thread.start()

    def finish(self,deadline_ns):
        self.thread.join(max(0,(deadline_ns-self.clock())/1e9))
        if self.thread.is_alive():
            self.cancel.set();self.thread.join(1)
            raise TimeoutError('Background public arrival dispatcher exceeded fixed drain boundary; thread_exited='+str(not self.thread.is_alive()))
        if self.error:raise RuntimeError(self.error)
        self.runtime.drain(self.records,max(0,(deadline_ns-self.clock())/1e9))


def _wait_ready(runtime,records,deadline_ns,clock):
    # Parent RESULT can precede the worker's next-read boundary. Priming must
    # finish both, preserving the two distinct original observations.
    while True:
        with runtime.lock:
            bad=[r for r in records if r['status'] not in ('completed','queued','submitted','planned')]
            done=all(r['status']=='completed' and r.get('previous_handler_next_request_ready') for r in records)
        if bad:raise RuntimeError('Background request/priming failed: '+repr([r['request_id'] for r in bad]))
        if done:return
        if clock()>=deadline_ns:raise TimeoutError('Missing completed background parent receipt/worker readiness')
        time.sleep(.001)


def resource_summary(samples,*,startup_ns,epoch_ns,measurement_end_ns,drain_end_ns,physical_identity):
    from gsdeploy.energy import integrate_power
    return dict(physical_deployment_sha256=physical_identity,
        startup_energy_joules=integrate_power(samples,startup_ns,epoch_ns),
        startup_includes_all_owner_load_and_priming=True,
        measurement_energy_joules=integrate_power(samples,epoch_ns,measurement_end_ns),
        measurement_energy_excludes_drain=True,
        owned_sampled_peak_startup_through_drain_mib=max((s['owned_used_mib'] for s in samples if startup_ns<=s['time_ns']<=drain_end_ns),default=None),
        memory_safety_factor_applied=False,sampled_memory_can_miss_transients=True,
        capacity_g='MISSING_REQUIRES_NEW_PROFILE',old_four_owner_profiles_reused=False,
        physical_SM_DRAM_L2_demand='UNKNOWN',fit_eligible=False,final_gain_eligible=False)


def _atomic_checkpoint(path,value):
    temporary=Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False))
    temporary.replace(path)


def _live_group_pids(group):
    result=[]
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().rpartition(')')[2].split()
            if int(fields[2])==group and fields[0] not in ('Z','X'):
                result.append(int(path.parent.name))
        except (OSError,ValueError,IndexError):pass
    return sorted(result)


def _stop_group(process,*,grace_seconds=1.):
    signals=[]
    for action in (signal.SIGTERM,signal.SIGKILL):
        if not _live_group_pids(process.pid):break
        try:os.killpg(process.pid,action);signals.append(action.name)
        except ProcessLookupError:break
        until=time.monotonic()+grace_seconds
        while time.monotonic()<until and _live_group_pids(process.pid):time.sleep(.005)
    try:process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:pass
    live=_live_group_pids(process.pid)
    return dict(signals=signals,live_group_pids=live,all_process_group_members_exited=not live,
        returncode=process.poll(),zombie_states_are_exited=True)


def _await_group_exit(group,directory):
    while True:
        try:
            live=_live_group_pids(group)
            if not live:return
            try:
                _atomic_checkpoint(Path(directory)/'supervisor-exit-await.json',dict(
                    status='GPU_LOCK_HELD_AWAITING_PROCESS_GROUP_EXIT',
                    process_group=group,live_group_pids=live,
                    full_cleanup_claimed=False,fit_eligible=False,final_gain_eligible=False))
            except OSError:
                # A full/unavailable evidence filesystem cannot release a
                # live GPU owner. Preserve the ownership guard regardless.
                pass
            os.killpg(group,signal.SIGKILL)
            time.sleep(.25)
        except ProcessLookupError:return
        except KeyboardInterrupt:
            # Interrupt changes the result, not ownership: the caller's
            # experiment lock cannot be released over live model owners.
            continue


def supervise_process(command,directory,timeout_seconds,*,environment=None,grace_seconds=1.,on_start=None):
    """CPU-testable external timeout; child and descendants share a new session."""
    directory=Path(directory);began=time.monotonic();timed_out=False
    process=None;stop=None
    with (directory/'supervised-stdout.log').open('x') as log:
        try:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                env=environment,start_new_session=True)
            if on_start is not None:on_start(process.pid)
            try:process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:timed_out=True
            # A completed parent is insufficient if it abandoned a worker.
            stop=_stop_group(process,grace_seconds=grace_seconds)
        except BaseException:
            try:save(directory/'supervisor-interrupted.json',dict(stop=stop,interrupted=True))
            except OSError:pass
            raise
        finally:
            # Includes interruption inside _stop_group or evidence writing,
            # not only interruption during process.wait.
            if process is not None:
                _await_group_exit(process.pid,directory)
                while process.poll() is None:
                    try:process.wait(timeout=.1)
                    except (KeyboardInterrupt,subprocess.TimeoutExpired):continue
    result=dict(schema=SCHEMA+'.supervisor',command=command,timeout_seconds=timeout_seconds,
        timed_out=timed_out,child_pid=process.pid,process_group=process.pid,
        elapsed_seconds=time.monotonic()-began,**stop)
    save(directory/'supervisor.json',result);return result


def _supervised_collect(plan_path,split):
    plan_path=Path(plan_path).resolve();plan=audit(plan_path);base=v2.audit(checked(plan['text_owner_plan']))
    if plan['execution_blockers']:raise ValueError(';'.join(plan['execution_blockers']))
    if any(os.environ.get(k)!=v for k,v in base['owner_environment'].items()):
        raise ValueError('Declared CUDA/allocator environment required before supervised GPU loading')
    directory=plan_path.parent/split/'collection';directory.mkdir(exist_ok=False)
    save(directory/'supervisor-attempt.json',dict(plan=receipt(plan_path),split=split,
        requested=plan['partitions'][split]['requested'],supervisor_parent_pid=os.getpid(),retry_permitted=False))
    environment=dict(os.environ,GS_BATCH_V3_SUPERVISOR_PARENT=str(os.getpid()))
    command=[sys.executable,'-m','tools.predictor_validation_20261027.batch_profile_collect_v3',
        '_supervised-child','--plan',str(plan_path),'--split',split,'--execute']
    started_groups=[]
    with open(ROOT/'artifacts/.gpu-experiment.lock','a') as gpu_lock:
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            outcome=supervise_process(command,directory,plan['supervisor_timeout_seconds'],environment=environment,
                on_start=started_groups.append)
        finally:
            for group in started_groups:_await_group_exit(group,directory)
        # The supervisor, rather than a potentially killed model process,
        # holds the experiment lock through the verified group-exit boundary.
        # A process stuck in uninterruptible kernel state may not disappear
        # immediately after SIGKILL. Keep the experiment lock rather than
        # authorize another collection or claim cleanup in that state.
        _await_group_exit(outcome['process_group'],directory)
        save(directory/'supervisor-exit-verified.json',dict(process_group=outcome['process_group'],
            live_group_pids=_live_group_pids(outcome['process_group']),all_model_processes_exited=True,
            experiment_lock_held_through_this_observation=True))
    child_report=directory/'report.json'
    result=read(child_report) if child_report.exists() else None
    success=(not outcome['timed_out'] and outcome['returncode']==0
        and outcome['all_process_group_members_exited'] and not outcome['signals'] and result is not None)
    summary=dict(schema=SCHEMA+'.supervised_collection',status='CHILD_COLLECTION_RETURNED' if success else 'SUPERVISOR_STOPPED_OR_FAILED_NOT_QUALIFIED',
        plan=receipt(plan_path),split=split,offered=plan['partitions'][split]['requested'],
        offered_cohort=read(checked(plan['partitions'][split]['arrivals'])),
        priming_cohort=read(checked(plan['partitions'][split]['priming']))['arrivals'],
        supervisor=receipt(directory/'supervisor.json'),child_report=receipt(child_report) if child_report.exists() else None,
        group_exit_proof=receipt(directory/'supervisor-exit-verified.json'),
        last_progress_checkpoint=receipt(directory/'progress.json') if (directory/'progress.json').exists() else None,
        reliable_group_exit_verified=outcome['all_process_group_members_exited'],
        full_cleanup_claimed=success and result.get('clean_shutdown') is True,
        fit_eligible=False,final_gain_eligible=False,retry_permitted=False,GPU_started=True)
    save(directory/'supervised-report.json',summary)
    if result is None:
        save(child_report,dict(schema=SCHEMA+'.collection',status='SUPERVISOR_TIMEOUT_OR_CHILD_FAILURE_NOT_QUALIFIED',
            offered=summary['offered'],offered_cohort=summary['offered_cohort'],all_offered_parent_receipts=False,
            unresolved_request_ids=[r['request_id'] for r in summary['offered_cohort']],
            partial_observations=summary['last_progress_checkpoint'],supervisor=summary['supervisor'],
            no_missing_timing_or_resource_imputed=True,fit_eligible=False,final_gain_eligible=False,
            clean_shutdown=False,GPU_started=True))
    return summary


def collect(plan_path,split,*,execute=False,backend_factory=None,background_runtime_factory=None,
            clock=time.perf_counter_ns):
    if split not in ('training','validation'):raise ValueError('Unknown split')
    if not execute and (backend_factory is None or background_runtime_factory is None):
        raise RuntimeError('GPU collection requires explicit --execute; CPU test requires both injected owners')
    if execute and (backend_factory is not None or background_runtime_factory is not None):
        raise ValueError('CPU doubles cannot be used for GPU collection')
    if execute:return _supervised_collect(plan_path,split)
    return _collect_attempt(plan_path,split,execute=False,backend_factory=backend_factory,
        background_runtime_factory=background_runtime_factory,clock=clock)


def _collect_attempt(plan_path,split,*,execute,backend_factory=None,background_runtime_factory=None,
                     clock=time.perf_counter_ns,_reserved=False):
    plan_path=Path(plan_path).resolve();plan_ref=receipt(plan_path);plan=audit(plan_path)
    base=v2.audit(checked(plan['text_owner_plan']))
    if execute and plan['execution_blockers']:raise ValueError(';'.join(plan['execution_blockers']))
    if execute and any(os.environ.get(k)!=v for k,v in base['owner_environment'].items()):
        raise ValueError('Declared CUDA/allocator environment required before any owner loads')
    directory=plan_path.parent/split/'collection';directory.mkdir(exist_ok=_reserved)
    save(directory/'attempt.json',dict(schema=SCHEMA+'.attempt',plan=plan_ref,split=split,
        execute=execute,CPU_test_only=not execute,retry_permitted=False))
    part=plan['partitions'][split];samples,arrivals,priming=(read(checked(part[k])) for k in ('samples','arrivals','priming'))
    cfg=base['configuration'];config=BatchConfiguration(cfg['max_batch_size'],cfg['max_padded_tokens'],cfg['max_wait_ns'],tuple(cfg['length_buckets']))
    runtime=monitor=gpu_lock=dispatcher=priming_dispatcher=None;startup_ns=epoch_ns=measurement_end=drain_end=None
    text_report=None;priming_report=None;failure=None;shutdown=None;cleanup=[];ready={}
    retained_text_owner=[]
    telemetry=dict(status='UNKNOWN_CPU_TEST_NO_PHYSICAL_TELEMETRY',samples=[])
    text_events=[];priming_text_events=[];checkpoint_errors=[];checkpoint_stop=threading.Event()
    def checkpoint():
        try:
            _atomic_checkpoint(directory/'progress.json',dict(schema=SCHEMA+'.incomplete_progress',
                plan=plan_ref,split=split,offered_cohort=arrivals,priming_cohort=priming['arrivals'],
                clock_ns=clock(),startup_start_ns=startup_ns,epoch_ns=epoch_ns,
                physical_owner_ready=copy.deepcopy(ready),text_events=copy.deepcopy(text_events),
                text_priming_events=copy.deepcopy(priming_text_events),
                background_records=[] if dispatcher is None else copy.deepcopy(dispatcher.records),
                background_priming_records=[] if priming_dispatcher is None else copy.deepcopy(priming_dispatcher.records),
                board_samples=[] if monitor is None else copy.deepcopy(monitor.samples),
                telemetry_errors=[] if monitor is None else list(monitor.errors),
                fit_eligible=False,final_gain_eligible=False,incomplete_checkpoint_not_final_report=True))
        except BaseException as exc:checkpoint_errors.append(repr(exc))
    def checkpoint_loop():
        while not checkpoint_stop.wait(.1):checkpoint()
    checkpoint();checkpoint_thread=threading.Thread(target=checkpoint_loop,name='batch-evidence-checkpoint',daemon=True)
    checkpoint_thread.start()
    try:
        if execute:
            if not _reserved:raise ValueError('Real model loading requires the external supervised process boundary')
            from gsdeploy.runtime.deployment import DeploymentRuntime
            from gsdeploy.runtime.schema import MeasurementConfig
            from .native_collect import BoardMonitor,descendant_or_self
            from gsdeploy.backends.shared_text_batch import load_retained
            registry=read(checked(base['backgrounds']))
            rates={task:sum(r['task']==task and r['alias'] in BACKGROUND for r in arrivals)/(plan['measurement_ns']/1e9) for task in {r['task'] for r in arrivals if r['alias'] in BACKGROUND}}
            routing={task:{owner:sum(r['task']==task and r['alias']==owner for r in arrivals)/(plan['measurement_ns']/1e9) for owner in BACKGROUND} for task in rates}
            runtime=DeploymentRuntime(MeasurementConfig(deployment=dict.fromkeys(BACKGROUND,1),workload_rps=rates,
                routing_rps=routing,warmup_seconds=0,measurement_seconds=plan['measurement_ns']/1e9,
                max_pending_per_model=len(arrivals)+len(priming['arrivals']),startup_timeout_seconds=600),registry)
            class OwnedProxy:
                @property
                def workers(self):return {**runtime.workers,'shared_text':SimpleNamespace(process=SimpleNamespace(pid=os.getpid()))}
                @property
                def ready(self):return {**runtime.ready,'shared_text':dict(pid=os.getpid())}
            monitor=BoardMonitor(OwnedProxy());monitor.sample();monitor.check();monitor.start()
            backend_factory=lambda:load_retained(checked(base['text_spec']),base['source_root'],execute=True,expected_spec_sha256=base['text_spec']['sha256'])
        else:runtime=background_runtime_factory()
        startup_ns=clock();runtime.start()
        if execute:
            pids=[]
            for alias,worker in runtime.workers.items():
                payload=runtime.ready[alias];pid=payload.get('pid');entry=registry[alias]
                if (type(pid) is not int or pid==os.getpid() or pid in pids
                        or not descendant_or_self(pid,worker.process.pid)):
                    raise ValueError('READY background identity is not owned')
                pids.append(pid)
                if payload.get('spec_sha256')!=receipt(entry['spec'])['sha256']:
                    raise ValueError('Background READY spec identity differs from frozen worker binding')
        ready=copy.deepcopy(runtime.ready)
        def load_and_prime():
            nonlocal priming_report,priming_dispatcher
            backend=backend_factory()
            # Keep weights alive after text has drained while G2/OCR or the
            # measurement window continues; residency does not end at last text.
            retained_text_owner.append(backend)
            ready['shared_text']=dict(pid=os.getpid(),backend_identity=copy.deepcopy(backend.identity),
                CPU_TEST_ONLY=not execute,loaded_physical_owner='shared_text')
            # Priming is an explicit separate cohort on the same resident models.
            prime_epoch=clock();prime=BackgroundDispatcher(runtime,priming['samples'],priming['arrivals'],prime_epoch,clock=clock,phase='priming')
            priming_dispatcher=prime
            prime.start()
            try:
                priming_report=run_arrivals([s for s in priming['samples'] if s['task']=='text'],
                    [r for r in priming['arrivals'] if r['alias'] in TEXT],config,lambda:backend,
                    owner='shared_text',parent_receipt=plan_ref,clock=clock,output=directory/'text-priming.json',
                    event_sink=lambda event:priming_text_events.append(event))
                if (priming_report['status']!='COMPLETED' or priming_report['delivered']!=len([r for r in priming['arrivals'] if r['alias'] in TEXT])
                        or any(r['result'].get('success') is not True for r in priming_report['results'])):
                    raise RuntimeError('Unsuccessful text priming prohibits measurement epoch')
                priming_report['existing_loaded_owner_reused']=True
                save(directory/'text-priming-owner-binding.json',dict(runtime=receipt(directory/'text-priming.json'),
                    existing_loaded_owner_reused=True,additional_model_loads=0))
                deadline=clock()+round(plan['drain_timeout_seconds']*1e9)
                prime.finish(deadline);_wait_ready(runtime,prime.records,deadline,clock)
            finally:
                prime.cancel.set()
                if prime.thread:prime.thread.join(1)
                if prime.thread and prime.thread.is_alive():cleanup.append('PRIMING_DISPATCHER_NOT_EXITED')
                save(directory/'background-priming.json',dict(records=copy.deepcopy(prime.records),error=prime.error))
            return backend
        def event_sink(event):
            nonlocal dispatcher,epoch_ns,measurement_end
            text_events.append(event)
            if monitor:monitor.check()
            if event['event']=='OWNER_READY':
                epoch_ns=event['absolute_ns'];measurement_end=epoch_ns+plan['measurement_ns']
                dispatcher=BackgroundDispatcher(runtime,samples,arrivals,epoch_ns,clock=clock)
                dispatcher.start()
            if dispatcher is not None and dispatcher.error:raise RuntimeError(dispatcher.error)
        text_report=run_arrivals([s for s in samples if s['task']=='text'],[r for r in arrivals if r['alias'] in TEXT],
            config,load_and_prime,owner='shared_text',parent_receipt=plan_ref,clock=clock,
            output=directory/'runtime.json',event_sink=event_sink)
        deadline=measurement_end+round(plan['drain_timeout_seconds']*1e9)
        dispatcher.finish(deadline);_wait_ready(runtime,dispatcher.records,deadline,clock)
        while clock()<measurement_end:
            if monitor:monitor.check()
            time.sleep(min(.01,(measurement_end-clock())/1e9))
        drain_end=clock()
        if monitor:monitor.sample();monitor.check()
        audit(plan_path)
    except BaseException as exc:
        failure=repr(exc)
        if text_report is None and hasattr(exc,'report'):text_report=copy.deepcopy(exc.report)
    finally:
        if dispatcher is not None:
            dispatcher.cancel.set()
            if dispatcher.thread:dispatcher.thread.join(1)
            if dispatcher.thread and dispatcher.thread.is_alive():cleanup.append('MEASUREMENT_DISPATCHER_NOT_EXITED')
        if monitor is not None:
            try:monitor.close()
            except BaseException as exc:cleanup.append(repr(exc))
            telemetry=dict(status='SAMPLED_BOARD_ONLY' if monitor.samples and not monitor.errors else 'UNKNOWN',
                samples=monitor.samples,errors=monitor.errors)
        if runtime is not None:
            try:shutdown=runtime.close()
            except BaseException as exc:cleanup.append(repr(exc))
        if gpu_lock is not None:fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()
        checkpoint_stop.set();checkpoint_thread.join(1)
        if checkpoint_thread.is_alive():cleanup.append('CHECKPOINT_THREAD_NOT_EXITED')
        checkpoint()
    bg=[] if dispatcher is None else copy.deepcopy(dispatcher.records)
    save(directory/'background-runtime.json',dict(records=bg,errors=[] if runtime is None else copy.deepcopy(runtime.errors),ready=ready))
    save(directory/'telemetry.json',telemetry);save(directory/'shutdown.json',dict(report=shutdown,errors=cleanup))
    text_rows=[] if text_report is None else text_report['results']
    result_rows=[dict(request_id=r['request_id'],alias=r['alias'],physical_owner='shared_text',task='text',
        scheduled_arrival_ns=r['arrival_ns'],parent_receipt_ns=r['parent_receipt_ns'],
        end_to_end_ns=r['end_to_end_ns'],status='completed',success=r['result'].get('success'),
        original_text_result=r) for r in text_rows]
    for row in bg:
        completed=row.get('completion_time_ns');relative=None if completed is None else completed-epoch_ns
        result_rows.append(dict(request_id=row['request_id'],alias=row['model'],physical_owner=row['model'],task=row['task'],
            scheduled_arrival_ns=row['scheduled_offset_ns'],parent_receipt_ns=relative,
            end_to_end_ns=None if relative is None else relative-row['scheduled_offset_ns'],status=row['status'],
            success=row.get('worker_response',{}).get('success'),original_background_record=row))
    offered_index={r['request_id']:r['arrival_index'] for r in arrivals}
    for row in result_rows:row['offered_arrival_index']=offered_index[row['request_id']]
    result_rows.sort(key=lambda r:r['offered_arrival_index'])
    expected=set(offered_index);actual=[r['request_id'] for r in result_rows]
    receipts_complete=(set(actual)==expected and len(actual)==len(expected)
        and all(r['status']=='completed' and type(r['parent_receipt_ns']) is int
            and r['parent_receipt_ns']>=r['scheduled_arrival_ns'] for r in result_rows))
    within_drain=receipts_complete and all(r['parent_receipt_ns']<=plan['measurement_ns']+round(plan['drain_timeout_seconds']*1e9) for r in result_rows)
    runtime_errors=[] if runtime is None else runtime.errors
    clean_shutdown=not cleanup and isinstance(shutdown,dict) and all(not row.get('forced') and row.get('exit_code')==0 and row.get('bye_received') for row in shutdown.values())
    complete=receipts_complete and within_drain and not failure and not runtime_errors and clean_shutdown
    resources=None
    if None not in (startup_ns,epoch_ns,measurement_end,drain_end):
        resources=resource_summary(telemetry['samples'],startup_ns=startup_ns,epoch_ns=epoch_ns,
            measurement_end_ns=measurement_end,drain_end_ns=drain_end,physical_identity=plan['deployment']['sha256'])
    profiles=directory/'requests';profiles.mkdir()
    actual_by_id={r['request_id']:r for r in result_rows};request_refs={}
    for row in arrivals:
        target=profiles/(digest(row['request_id'])+'.json')
        save(target,dict(schema=SCHEMA+'.request',plan=plan_ref,split=split,
            offered_request=row,actual=actual_by_id.get(row['request_id']),
            unresolved=row['request_id'] not in actual_by_id or actual_by_id[row['request_id']]['status']!='completed',
            fit_eligible=False,physical_deployment=plan['deployment']))
        request_refs[row['request_id']]=receipt(target)
    batch_refs=[]
    batches=directory/'batches';batches.mkdir()
    for index,batch in enumerate([] if text_report is None else text_report['batches']):
        target=batches/(str(index)+'.json')
        save(target,dict(schema=SCHEMA+'.actual_batch',plan=plan_ref,split=split,
            actual_batch=batch,physical_owner='shared_text',active_background_protocol=plan_ref,
            request_receipts=[request_refs[rid] for rid in batch['selection']['members']],
            resource_demand='UNKNOWN',ordered_kernel_plan='UNKNOWN',fit_eligible=False))
        batch_refs.append(receipt(target))
    save(directory/'profile-index.json',dict(schema=SCHEMA+'.index',plan=plan_ref,split=split,
        requests=request_refs,batches=batch_refs,offered=len(arrivals),all_offered_indexed=len(request_refs)==len(arrivals),
        prediction_profiles='MISSING_NEW_QUALIFICATION_REQUIRED',fit_eligible=False,final_gain_eligible=False))
    report=dict(schema=SCHEMA+'.collection',plan=plan_ref,split=split,status=('CPU_TEST_ONLY_COMPLETE' if not execute else 'COLLECTED_REQUIRES_NEW_PROFILE_QUALIFICATION') if complete else 'INCOMPLETE_OR_FAILED_NOT_QUALIFIED',
        CPU_test_only=not execute,GPU_started=execute,offered=len(arrivals),offered_cohort=arrivals,
        parent_receipts=sum(r['status']=='completed' for r in result_rows),all_offered_parent_receipts=receipts_complete,
        within_declared_drain=within_drain,all_outputs_successful=receipts_complete and all(r['success'] is True for r in result_rows),
        results=result_rows,unresolved_request_ids=sorted(expected-{r['request_id'] for r in result_rows if r['status']=='completed' and type(r['parent_receipt_ns']) is int}),failure=failure,
        physical_deployment=plan['deployment'],epoch_ns=epoch_ns,startup_start_ns=startup_ns,
        physical_owner_ready=ready,text_weights_retained_until_after_resource_monitor_close=bool(retained_text_owner),
        measurement_end_ns=measurement_end,drain_end_ns=drain_end,resources=resources,
        text_runtime=receipt(directory/'runtime.json') if (directory/'runtime.json').exists() else None,
        background_runtime=receipt(directory/'background-runtime.json'),telemetry=receipt(directory/'telemetry.json'),
        profile_index=receipt(directory/'profile-index.json'),
        shutdown=receipt(directory/'shutdown.json'),clean_shutdown=clean_shutdown,
        checkpoint=receipt(directory/'progress.json') if (directory/'progress.json').exists() else None,
        checkpoint_errors=checkpoint_errors,
        prediction_profiles_missing=True,old_four_owner_profiles_reused=False,fit_eligible=False,
        final_gain_eligible=False,protocol_sealed=False,retry_permitted=False)
    save(directory/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    q=sub.add_parser('prepare')
    for name in ('output','spec','source-root','samples','schedules','priming'):q.add_argument('--'+name,required=True)
    q.add_argument('--background-registry');q.add_argument('--measurement-ns',type=int,required=True)
    q.add_argument('--drain-timeout-seconds',type=float,required=True)
    q.add_argument('--supervisor-timeout-seconds',type=float,required=True)
    q.add_argument('--max-batch-size',type=int,required=True);q.add_argument('--max-padded-tokens',type=int,required=True)
    q.add_argument('--max-wait-ns',type=int,required=True);q.add_argument('--length-buckets',type=int,nargs='*',default=[])
    q=sub.add_parser('collect');q.add_argument('--plan',required=True)
    q.add_argument('--split',choices=('training','validation'),required=True);q.add_argument('--execute',action='store_true')
    q=sub.add_parser('_supervised-child');q.add_argument('--plan',required=True)
    q.add_argument('--split',choices=('training','validation'),required=True);q.add_argument('--execute',action='store_true',required=True)
    a=p.parse_args()
    if a.op=='prepare':
        result=prepare(a.output,a.spec,a.source_root,a.samples,a.schedules,a.priming,
            BatchConfiguration(a.max_batch_size,a.max_padded_tokens,a.max_wait_ns,tuple(a.length_buckets)),
            measurement_ns=a.measurement_ns,drain_timeout_seconds=a.drain_timeout_seconds,
            supervisor_timeout_seconds=a.supervisor_timeout_seconds,background_registry=a.background_registry)
    elif a.op=='_supervised-child':
        if os.environ.get('GS_BATCH_V3_SUPERVISOR_PARENT')!=str(os.getppid()):
            raise RuntimeError('Only a live external collection supervisor can load this child')
        result=_collect_attempt(a.plan,a.split,execute=True,_reserved=True)
    else:result=collect(a.plan,a.split,execute=a.execute)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
