"""Isolated v3 FIFO collection: no full snapshots in measurement or drain.

prepare/audit are CPU only. execute requires an explicit CLI flag and uses the
unchanged reviewed2400s/18GiB supervisor; this module never imports a model.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
import uuid

from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.schema import MeasurementConfig
from gsdeploy.runtime.runner import record
from . import twoFIFO_training_semantic_preflight_v2 as strict
from .twoFIFO_training_manifest_v1 import digest,require,MEASUREMENT_NS,DRAIN_NS,OWNERS
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .profile_owned_collect_v3 import StartupBoardMonitor,memory_ownership,MONITORING_SCOPE
from . import profile_training_supervisor_v3 as supervisor
from .stage_response_features_v1 import receipt

ROOT=strict.ROOT
SCHEMA='gsdeploy.original_twoFIFO_owned_training_campaign.v3'
RAW_SCHEMA='gsdeploy.original_twoFIFO_owned_training_capture.v3'
ACTIVITY_SCHEMA='gsdeploy.twoFIFO_collector_activity_spans.v3'
ACTIVITY_FILE='checkpoint-spans.json'
TIME_DOMAIN='time.perf_counter_ns'


class ActivityLedger:
    """Complete CPU activity spans in a separately durable canonical sidecar.

    A snapshot cannot contain the end of its own write. The final sidecar is
    written after that write returns; it authenticates completed full-copy and
    serialization spans without rewriting the large raw to add timestamps.
    Tiny ledger persistence also returns before any case epoch is chosen.
    """
    def __init__(self,output,execution_id,clock_ns):
        self.output=Path(output);self.execution_id=execution_id;self.clock_ns=clock_ns;self.spans=[]

    @contextmanager
    def span(self,kind,*,case_id=None,phase=None,stage=None):
        row=dict(span_id=len(self.spans),kind=kind,case_id=case_id,phase=phase,stage=stage,
            time_domain=TIME_DOMAIN,start_ns=self.clock_ns(),end_ns=None)
        self.spans.append(row)
        try:yield row
        finally:row['end_ns']=self.clock_ns()

    def persist(self):
        require(all(type(s['end_ns']) is int for s in self.spans),'Incomplete activity span')
        atomic_save(self.output/ACTIVITY_FILE,dict(schema=ACTIVITY_SCHEMA,execution_id=self.execution_id,
            time_domain=TIME_DOMAIN,checkpoint_spans=self.spans,all_spans_complete=True))


def audit_activity(raw,activity):
    """Fail closed on any completed protected-interval full-copy overlap."""
    for span in activity.spans:
        require(type(span['start_ns']) is int and type(span['end_ns']) is int
            and span['start_ns']<=span['end_ns'],'Incomplete/invalid collector activity span')
        for window in raw['measurement_windows']:
            require(span['end_ns']<window['start_ns'] or span['start_ns']>window['fixed_drain_deadline_ns'],
                'Collector activity crossed measurement/drain: '+str(span['span_id']))


def read(path):return json.loads(Path(path).read_text())


def atomic_save(path,value):
    """Replace a complete fsynced snapshot; never expose half JSON to review."""
    path=Path(path);temporary=path.with_name('.'+path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def wait_until(target,clock_ns,monitor,sleep=time.sleep):
    while True:
        remaining=target-clock_ns()
        if remaining<=0:return
        monitor.check()
        # A monitor query can cross the deadline after the first clock read.
        remaining=target-clock_ns()
        if remaining<=0:return
        sleep(min(.005,remaining/1e9))


def priming_catalog(context):
    selected=[]
    for owner in OWNERS:
        tasks=('text','image','asr') if owner=='generalist_2#0' else (('ocr',) if owner=='ocr#0' else ('text',))
        for task in tasks:
            rows=sorted((r for r in context['declared_inputs'] if r['task']==task),key=lambda r:r['input_size'],reverse=True)[:3]
            require(len(rows)==3,'Original three largest priming inputs required')
            selected.extend(dict(owner=owner,**row) for row in rows)
    require(len(selected)==18,'Original18 priming required')
    return selected


def shard_cases(base,entry):
    blocks=[]
    for index,(natural,ocr) in enumerate(zip(base['capture_blocks'],entry['capture_blocks'])):
        # Reconstruct canonical order before balancing: one core case followed
        # by two OCR cases, then distribute rotated contiguous positions.
        cores=sorted(natural['cases'],key=lambda c:c['case_id']);extras=sorted(ocr['cases'],key=lambda c:c['case_id'])
        ordered=[]
        while cores or extras:
            if cores:ordered.append(cores.pop(0))
            ordered.extend(extras[:2]);extras=extras[2:]
        shift=index*17;ordered=ordered[shift:]+ordered[:shift]
        shards=[ordered[start:start+13] for start in range(0,len(ordered),13)]
        blocks.append(dict(capture_id=natural['capture_id'],shards=shards))
    return blocks


def runtime_config():
    return MeasurementConfig(deployment=dict.fromkeys(OWNERS,1),
        workload_rps=dict(text=1.,ocr=1.,image=.01,asr=.01),
        routing_rps={'text':{'text#0':.5,'text#1':.5},'ocr':{'ocr#0':1.},
            'image':{'generalist_2#0':.01},'asr':{'generalist_2#0':.01}},
        startup_timeout_seconds=600,shutdown_timeout_seconds=30,measurement_seconds=10.,
        drain_timeout_seconds=60.,max_pending_per_model=64,routing_source='controlled_FIFO_training_not_utility')


def prepare(output):
    base,entry=strict.load_frozen();context=read(strict.CONTEXT)
    pool=strict.source_catalog(context)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    # Store only the unchanged declared training catalog, including image/asr
    # needed by original priming. No holdout inputs are selected here.
    pool_path=output/'runtime-inputs.jsonl'
    with pool_path.open('x') as stream:
        for item in context['declared_inputs']:stream.write(json.dumps(pool[item['pool_id']],ensure_ascii=False)+'\n')
    atomic_save(output/'context.json',context)
    sources=implementation_sources(__file__)
    frozen=[]
    for src in sources:
        target=output/'source-freeze'/Path(src['path']).relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(src['path']).read_bytes())
        frozen.append(dict(original_path=src['path'],frozen=receipt(target)))
    attempts=[]
    for block in shard_cases(base,entry):
        for shard_index,cases in enumerate(block['shards']):
            attempt_id=block['capture_id']+'-shard-'+str(shard_index+1)
            plan=dict(schema=SCHEMA+'.attempt',attempt_id=attempt_id,capture_id=block['capture_id'],
                shard_index=shard_index+1,shards_in_block=4,execution_id=uuid.uuid4().hex,
                split='training',cases=cases,measurement_ns=MEASUREMENT_NS,drain_ns=DRAIN_NS,
                resident_owners=list(OWNERS),runtime_policy=base['runtime_policy'],
                priming=priming_catalog(context),runtime_config=runtime_config().as_dict(),
                context=receipt(output/'context.json'),prepared_pool=receipt(pool_path),
                base_manifest=receipt(strict.BASE),OCR_entry_manifest=receipt(strict.ENTRY),
                source_freeze=frozen,sources=strict_references([base['sources'],entry['sources'],sources]),
                no_retries=True,supervisor_budget=supervisor.BUDGET,
                preflight_free_bytes_required=supervisor.PREFLIGHT,
                block_eligible_only_if_all_four_shards_complete=True,
                native_CUDA_API_numeric_authentication='UNQUALIFIED',main_queue_provider_eligible=False)
            plan['collector_timing_policy']=dict(no_mutating_checkpoint_during_case=True,
                protected_interval='case_epoch_through_fixed_drain_deadline',
                public_calendars_durable_before_startup=True,prior_history_durable_before_epoch=True,
                priming_full_snapshot_between_requests=False,activity_schema=ACTIVITY_SCHEMA)
            plan['execution_context_sha256']=digest({k:v for k,v in plan.items() if k!='execution_context_sha256'})
            folder=output/attempt_id;folder.mkdir();atomic_save(folder/'plan.json',plan)
            attempts.append(receipt(folder/'plan.json'))
    campaign=dict(schema=SCHEMA,status='CPU_PREPARED_NOT_COLLECTED',attempts=attempts,
        capture_blocks=3,attempt_count=12,priming_per_attempt=18,total_priming_offered=216,
        measured_offered=546,case_count=150,maximum_cases_per_attempt=13,
        minimum_case_walltime_seconds=70,supervisor_budget=supervisor.BUDGET,
        preflight_free_bytes_required=supervisor.PREFLIGHT,no_retries=True,GPU_started=False)
    atomic_save(output/'campaign.json',campaign);audit(output/'campaign.json')
    return campaign


def audit_plan(path):
    plan=read(path);base,entry=strict.load_frozen()
    require(plan['schema']==SCHEMA+'.attempt' and plan['split']=='training','Exact training attempt required')
    require(plan.get('collector_timing_policy')==dict(no_mutating_checkpoint_during_case=True,
        protected_interval='case_epoch_through_fixed_drain_deadline',
        public_calendars_durable_before_startup=True,prior_history_durable_before_epoch=True,
        priming_full_snapshot_between_requests=False,activity_schema=ACTIVITY_SCHEMA),'Exact v3 checkpoint policy required')
    require(plan['supervisor_budget']==supervisor.BUDGET and plan['preflight_free_bytes_required']==supervisor.PREFLIGHT,
        'Original2400s/18GiB disk budget required')
    require(plan['runtime_config']==runtime_config().as_dict() and plan['runtime_policy']==base['runtime_policy'],
        'Original runtime policy/config required')
    require(plan['base_manifest']==receipt(strict.BASE) and plan['OCR_entry_manifest']==receipt(strict.ENTRY),'Frozen design binding drift')
    require(plan['capture_id'] in strict.CAPTURES and type(plan['shard_index']) is int and 1<=plan['shard_index']<=4,
        'Exact planned capture and one-based shard index required')
    wanted=next(b for b in shard_cases(base,entry) if b['capture_id']==plan['capture_id'])['shards'][plan['shard_index']-1]
    require(plan['cases']==wanted and len(wanted)<=13 and plan['shards_in_block']==4,'Exact balanced shard required')
    require(plan['attempt_id']==plan['capture_id']+'-shard-'+str(plan['shard_index']) and plan['no_retries'] is True,'Attempt ID/no retry drift')
    require(plan['resident_owners']==list(OWNERS) and plan['measurement_ns']==MEASUREMENT_NS and plan['drain_ns']==DRAIN_NS,'Original fixed window/owners required')
    context=strict.checked(plan['context']);require(context==read(strict.CONTEXT),'Frozen context drift')
    require(plan['priming']==priming_catalog(context),'Exact original18 priming required')
    source_pool=strict.source_catalog(context)
    require(receipt(plan['prepared_pool']['path'])==plan['prepared_pool'],'Prepared full priming pool drift')
    rows=[json.loads(l) for l in Path(plan['prepared_pool']['path']).read_text().splitlines()]
    require(len(rows)==16 and {r['pool_id']:r for r in rows}==source_pool,'Exact full training pool required')
    require(plan['sources']==strict_references([base['sources'],entry['sources'],implementation_sources(__file__)]),'Implementation source closure drift')
    for src in plan['sources']:require(receipt(src['path'])==src,'Source changed')
    implementation=implementation_sources(__file__)
    require(len(plan['source_freeze'])==len(implementation)
        and {r['original_path'] for r in plan['source_freeze']}=={r['path'] for r in implementation},'Incomplete/duplicate implementation freeze')
    for src in plan['source_freeze']:
        require(receipt(src['frozen']['path'])==src['frozen'] and receipt(src['original_path'])['sha256']==src['frozen']['sha256'],'Frozen executable differs')
    require(plan['execution_context_sha256']==digest({k:v for k,v in plan.items() if k!='execution_context_sha256'}),'Execution context drift')
    require(isinstance(plan['execution_id'],str) and len(plan['execution_id'])==32,'Independent execution identity required')
    return plan


def audit(path):
    campaign=read(path);require(campaign['schema']==SCHEMA and len(campaign['attempts'])==12,'Exact12-attempt campaign required')
    plans=[audit_plan(ref['path']) for ref in campaign['attempts'] if strict.checked(ref)]
    require(len({p['attempt_id'] for p in plans})==12 and len({p['execution_id'] for p in plans})==12,'Unique attempt/execution IDs required')
    require({p['attempt_id'] for p in plans}=={c+'-shard-'+str(i) for c in strict.CAPTURES for i in (1,2,3,4)},'Missing/extra shard')
    return campaign


def history_snapshot(runtime,raw,clock_ns):
    with runtime.lock:
        # queue.mutex serializes copying with reader puts and collector gets;
        # runtime.lock serializes pending records with sender/collector updates.
        with runtime.events.mutex:events=copy.deepcopy(list(runtime.events.queue))
        return dict(snapshot_time_ns=clock_ns(),request_records=copy.deepcopy(
            raw['priming_request_records']+[r for r in raw['request_records'] if r['status']!='not_started']),
            pending_request_ids=list(runtime.pending),ready=copy.deepcopy(runtime.ready),
            queued_runtime_events=events,runtime_errors=copy.deepcopy(runtime.errors),
            empty_initial_state_assumed=False)


def initial_rows(plan,pool):
    rows=[]
    for case in plan['cases']:
        for arrival in case['arrivals']:
            item=pool[arrival['pool_id']]
            row=record('profile',arrival['arrival_index'],arrival['task'],arrival['owner'],item,None)
            row.update(request_id=arrival['request_id'],case_id=case['case_id'],profile_job_id=case['case_id'],
                capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],arrival_ns=arrival['arrival_ns'],
                split='training',validation_view='training',training_role='natural_FIFO',status='not_started')
            rows.append(row)
    return rows


def finish_collection(runtime,monitor,raw,checkpoint,clock_ns):
    """Use original monitor/runtime cleanup with an atomic locked final save.

    The old collector's final save uses open('x'), incompatible with durable
    checkpoints. Keep its monitor-before-worker-close ordering here.
    """
    try:
        if monitor is not None:
            try:monitor.close();monitor.check()
            except BaseException as exc:
                raw['status']='failed';raw['errors'].append('MonitoringCloseError: '+repr(exc))
            finally:raw['lifecycle_boundaries'].append(dict(boundary='monitoring_end',time_ns=clock_ns()))
    finally:
        raw['lifecycle_boundaries'].append(dict(boundary='shutdown_begin',time_ns=clock_ns()))
        try:runtime.close()
        except BaseException as exc:
            raw['status']='failed';raw['errors'].append('WorkerShutdownError: '+repr(exc))
        finally:
            with runtime.lock:raw['shutdown']=copy.deepcopy(getattr(runtime,'shutdown_report',{}))
            raw['lifecycle_boundaries'].append(dict(boundary='shutdown_end',time_ns=clock_ns()))
            raw['checkpoint_context']=dict(kind='shutdown',case_id=None)
            checkpoint()


def dispatch_case(plan,case,runtime,pool,raw,*,clock_ns,wait_until_ns,checkpoint,monitor,activity=None):
    """All rows pre-registered; independent timed enqueue, fixed70s cutoff."""
    def prepare_case():
        history=history_snapshot(runtime,raw,clock_ns)
        with runtime.lock:
            raw['case_histories'].append(dict(case_id=case['case_id'],prior_history_snapshot=history))
            raw.setdefault('case_preparations',[]).append(dict(case_id=case['case_id'],
                public_arrivals_sha256=case['public_arrivals_sha256'],public_arrivals=copy.deepcopy(case['arrivals']),
                prior_history_index=len(raw['case_histories'])-1,prepared_before_epoch=True))
    if activity is not None:
        with activity.span('precase_history_snapshot',case_id=case['case_id'],phase=monitor.phase):prepare_case()
    else:prepare_case()
    # Persist offered identities/history before choosing the epoch. Full raw
    # snapshots grow with telemetry; disk I/O must not sit between millisecond
    # public arrivals and silently become an arrival scheduler.
    raw['checkpoint_context']=dict(kind='precase',case_id=case['case_id'])
    checkpoint()
    raw['case_preparations'][-1]['durable_checkpoint_return_ns']=clock_ns()
    monitor.set_phase('measurement')
    # Strictly separate the completed persistence from even a coarse/injected
    # clock's epoch. No disk write or history copy follows until drain ends.
    wait_until_ns(clock_ns()+1)
    epoch=clock_ns();end=epoch+MEASUREMENT_NS;deadline=end+DRAIN_NS
    rows=[r for r in raw['request_records'] if r['case_id']==case['case_id']]
    with runtime.lock:
        for row in rows:row.update(scheduled_time_ns=epoch+row['arrival_ns'],status='planned')
        case_record=next(c for c in raw['cases'] if c['case_id']==case['case_id'])
        case_record.update(status='running',case_epoch_ns=epoch,observation_start_ns=epoch,
            measurement_end_ns=end,fixed_drain_deadline_ns=deadline)
        raw['measurement_windows'].append(dict(case_id=case['case_id'],start_ns=epoch,end_ns=end,
            fixed_drain_deadline_ns=deadline))
        raw.setdefault('observation_start_ns',epoch)
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
    runtime.drain(rows,max(0,deadline-clock_ns())/1e9)
    wait_until_ns(deadline)
    with runtime.lock:
        for row in rows:
            if row['status'] in ('planned','queued','submitted'):row.update(status='incomplete',error='fixed drain reached; offered retained')
            row['RESULT_reader_within_fixed_drain']=(row.get('completion_time_ns') is not None and row['scheduled_time_ns']<=row['completion_time_ns']<=deadline)
        case_record.update(status='completed' if all(r['status']=='completed' and r['RESULT_reader_within_fixed_drain'] for r in rows) else 'incomplete',
            observation_finalized_ns=clock_ns(),fixed_cutoff_overshoot_ns=max(0,clock_ns()-deadline))
    wait_until_ns(deadline+1)
    monitor.sample();monitor.check()
    raw['checkpoint_context']=dict(kind='postdrain',case_id=case['case_id']);checkpoint()
    return case_record


def collect(plan_path,output,*,lease_fd,runtime_factory=DeploymentRuntime,monitor_factory=StartupBoardMonitor,
            clock_ns=time.perf_counter_ns,wait_until_ns=None,ownership_check=memory_ownership):
    supervisor.validate_lease(lease_fd);plan=audit_plan(plan_path);output=Path(output)
    require(not (output/'raw.json').exists(),'No overwrite/retry of a collection attempt')
    context=strict.checked(plan['context']);pool={r['pool_id']:r for r in (json.loads(l) for l in Path(plan['prepared_pool']['path']).read_text().splitlines())}
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
        request_records=initial_rows(plan,pool),priming_request_records=[],
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
        training_feature_extraction_eligible=False,all_offered_registered_before_startup=True)
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
        for index,item in enumerate(plan['priming']):
            row=record('priming',index,item['task'],item['owner'],pool[item['pool_id']],clock_ns())
            row.update(request_id=plan['attempt_id']+':priming:'+str(index),profile_job_id='priming',
                split='training',validation_view='training')
            with runtime.lock:raw['priming_request_records'].append(row)
            runtime.enqueue(row,pool[item['pool_id']]['sample']);runtime.drain([row],300)
            require(row['status']=='completed','Original priming failed')
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


def execute(plan_path,output):
    plan_ref=receipt(plan_path);plan=audit_plan(plan_path);supervisor.public_environment_gate()
    attempt=Path(plan_path).parent
    require(Path(output).resolve()==(attempt/'collection').resolve(),'Canonical one-attempt collection path required')
    require(not Path(output).exists(),'Attempt already started; no retries')
    command=[sys.executable,'-m','tools.predictor_validation_20261027.twoFIFO_owned_collect_v3','_child',
        '--plan',str(Path(plan_path).resolve()),'--output',str(Path(output).resolve()),'--execute']
    def gate():
        supervisor.public_environment_gate();strict.checked(plan_ref);audit_plan(plan_path)
    return supervisor.supervise(command,output,environment={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
        before_launch=gate,after_run=gate,metadata=dict(schema=SCHEMA+'.execution_binding',plan=plan_ref,
            execution_id=plan['execution_id'],attempt_id=plan['attempt_id']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True)
    p=sub.add_parser('audit');p.add_argument('--campaign',required=True)
    for op in ('execute','_child'):
        p=sub.add_parser(op);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',required=True,action='store_true')
    args=parser.parse_args()
    if args.op=='prepare':result=prepare(args.output)
    elif args.op=='audit':result=audit(args.campaign)
    elif args.op=='execute':result=execute(args.plan,args.output)
    else:result=collect(args.plan,args.output,lease_fd=int(os.environ.get(supervisor.LEASE_FD,'-1')))
    print(json.dumps(dict(status=result.get('status','CPU_AUDIT_PASS'))))
    return 1 if result.get('status') in ('failed','FAILED_NO_RETRY') else 0


if __name__=='__main__':raise SystemExit(main())
