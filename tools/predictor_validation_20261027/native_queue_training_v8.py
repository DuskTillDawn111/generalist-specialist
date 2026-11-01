"""Prospectively declared native queue-training traces; never validation labels.

Four original owners and twelve priming requests remain resident. The full
arrival/input/route registry and cutoff grid are persisted before collection.
Only the telemetry query/append critical section is serialized; scheduling is
unchanged and no global execution cap is introduced.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter

from gsdeploy.resources import prepare_registry
from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from gsdeploy.runtime.schema import MeasurementConfig
from gsdeploy.tasks import page_ocr_text, score_result
from gsdeploy.quality import sample_totals
from .background_collect import ROOT, context_identity, descendant_or_self, receipt, save
from .startup_collect_v4 import StartupBoardMonitor
from .native_flow import resource_configuration
from .operator_profile_campaign_v2 import audit as source_audit
from .unified_corun_collect_v8 import wait_bounded, group_members, cleanup_group
from .profiling import digest, RESIDENTS

SCHEMA='gsdeploy.native_queue_training_plan.v8'
BASE=ROOT/'artifacts/static-study-v6-20261021/inputs'
DEFAULT_PROFILE=ROOT/'artifacts/predictor-gpu-20261027/resource-full-state-training-v5-20261028-prepared00/capture01/natural'

class BoardMonitor(StartupBoardMonitor):
    def __init__(self,runtime):
        self.query_lock=threading.RLock()
        super().__init__(runtime)
    def sample(self):
        with self.query_lock:
            return super().sample()


def collection_source_paths():
    return [Path(__file__).resolve(), *[Path(__file__).with_name(n) for n in
        ('native_flow.py','startup_collect_v4.py','background_collect.py','flow_metrics.py',
         'operator_profile_campaign_v2.py','unified_corun_collect_v8.py',
         'native_queue_training_review_v8.py','known_input_observed_v5.py',
         'known_input_native_flow_v5.py','smoke_verifier_v3.py','trace_export.py',
         'nvtx_interval_index_v2.py')]]


def make_trace(inputs,pattern):
    if pattern not in ('burst_A','burst_B'):
        raise ValueError('Explicit predeclared queue-training pattern required')
    selected={t:sorted((r for r in inputs if r['task']==t),key=lambda r:r['input_size'])
              for t in ('text','image','asr','ocr')}
    if any(len(v)!=2 for v in selected.values()):
        raise ValueError('Two original exact training contents per task required')
    rows=[]
    def add(at,task,owner,long=False):
        rows.append(dict(offset_ns=round(at*1e9),task=task,
            pool_id=selected[task][int(long)]['pool_id'],alias=owner))
    peers=('text#0','text#1','generalist_2#0')
    stagger=0.0 if pattern=='burst_A' else .12
    for i,a in enumerate(peers):add(.01+i*stagger,'text',a)
    add(.02,'ocr','ocr#0')
    for at,long in ((2.5,True),(2.55,False),(9.,True),(9.05,True)):
        for i,a in enumerate(peers):add(at+i*stagger,'text',a,long)
    add(4.,'image','generalist_2#0')
    add(4.1,'asr','generalist_2#0')
    add(11.,'ocr','ocr#0',True)
    add(11.1,'image','generalist_2#0',True)
    add(11.2,'asr','generalist_2#0',True)
    for i,a in enumerate(peers):add(14.+i*stagger,'text',a)
    rows.sort(key=lambda r:(r['offset_ns'],r['task'],r['alias']))
    return [dict(r,request_id='measurement:'+str(i)) for i,r in enumerate(rows)]


def validate_plan(plan):
    if (plan.get('schema')!=SCHEMA or plan.get('split')!='training'
            or plan.get('training_only') is not True or plan.get('warmup_seconds')!=0.
            or plan.get('arrival_process')!='predeclared_exact_trace'
            or plan.get('plan_sha256')!=digest({k:v for k,v in plan.items() if k!='plan_sha256'})):
        raise ValueError('Immutable prospective training plan required')
    trace=plan['trace'];candidates=plan['candidates']
    if (not trace or len(candidates)!=1 or candidates[0]['resource_configuration']!=resource_configuration()
            or candidates[0]['execution_context_sha256']!=plan['execution_context_sha256']
            or candidates[0]['routes']!={r['request_id']:r['alias'] for r in trace}
            or plan['trace_sha256']!=digest(trace)
            or len({r['request_id'] for r in trace})!=len(trace)
            or trace!=sorted(trace,key=lambda r:(r['offset_ns'],r['task'],r['alias']))):
        raise ValueError('Full fixed FIFO cohort/routes and uncapped sharing required')
    for i,r in enumerate(trace):
        if (set(r)!={'request_id','offset_ns','task','pool_id','alias'}
                or r['request_id']!='measurement:'+str(i)
                or type(r['offset_ns']) is not int or not 0<=r['offset_ns']<round(plan['window_seconds']*1e9)
                or r['task'] not in plan['aliases'][r['alias']]['supported_tasks']):
            raise ValueError('Exact public training arrival identity required')
    if (not plan['cutoff_offsets_ns']
            or any(type(t) is not int or not 0<t<round(plan['window_seconds']*1e9) for t in plan['cutoff_offsets_ns'])
            or plan['cutoff_offsets_ns']!=sorted(set(plan['cutoff_offsets_ns']))):
        raise ValueError('Fixed original-window cutoff grid required')
    if plan['risk_target']!=dict(horizon_ns=2_000_000_000,recovery_threshold_ms=500.):
        raise ValueError('Predeclared diagnostic recovery target required')
    if plan['weights']!={k:1/6 for k in 'QTLPME'}:
        raise ValueError('Original six equal utility weights required')
    return True


def verify_sources(plan):
    validate_plan(plan)
    source_audit(Path(plan['context']['path']).parent)
    for ref in (plan['profile_plan'],plan['context'],plan['catalog'],plan['runner_source'],*plan['collector_sources']):
        checked(ref)
    context=json.loads(checked(plan['context']).read_text())
    if context_identity(context)!=plan['execution_context_sha256']:
        raise ValueError('Training context identity changed')
    profile=json.loads(checked(plan['profile_plan']).read_text())
    if (profile.get('split')!='training' or profile.get('training_only') is not True
            or make_trace(profile['inputs'],plan['group_id'])!=plan['trace']):
        raise ValueError('Training cohort differs from its declared input/pattern recipe')
    counts=Counter(r['task'] for r in plan['trace'])
    if (plan['window_seconds']!=16. or plan['drain_seconds']!=60.
            or plan['workload_rps']!={t:counts[t]/16. for t in ('text','image','asr','ocr')}):
        raise ValueError('Original full-trace offered rates/window/drain required')
    catalog=json.loads(checked(plan['catalog']).read_text())
    if (plan['memory_safe_factor']!=catalog['safe_factor']
            or plan['memory_budget_mib']!=49140 or plan['energy_reference_watts']!=450.
            or plan['startup_amortization_seconds']!=3600.
            or plan['latency_reference_ms']!=dict(text=1000,image=250,asr=3000,ocr=10000)):
        raise ValueError('Original resource and latency scoring constants required')
    for s in context['execution_sources']:
        checked(s['frozen'])
        if receipt(s['original_path'])['sha256']!=s['frozen']['sha256']:
            raise ValueError('Training execution source changed: '+s['original_path'])
    return True


def prepare(profile_dir,output,*,pattern='burst_A',run_id='training01'):
    profile_dir,output=Path(profile_dir).resolve(),Path(output).resolve()
    source_audit(profile_dir)
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Simple exclusive training run ID required')
    profile=json.loads((profile_dir/'plan.json').read_text())
    common=json.loads((profile_dir/'context.json').read_text())
    if profile.get('split')!='training' or profile.get('training_only') is not True:
        raise ValueError('Original training-only input/profile source required')
    sources={Path(s['original_path']).resolve() for s in common['execution_sources']}
    sources.update(collection_source_paths())
    for s in common['execution_sources']:
        if receipt(s['original_path'])['sha256']!=s['frozen']['sha256'] or receipt(s['frozen']['path'])!=s['frozen']:
            raise ValueError('Original training execution source drift')
    trace=make_trace(profile['inputs'],pattern)
    output.mkdir(parents=True,exist_ok=False)
    snapshots=[]
    for i,p in enumerate(sorted(sources)):
        dest=output/'source-snapshot'/str(i)/p.name;dest.parent.mkdir(parents=True)
        shutil.copyfile(p,dest);snapshots.append(dict(original_path=str(p),frozen=receipt(dest)))
    common.update(execution_sources=snapshots,sources=[s['frozen'] for s in snapshots],
        collection_schema=SCHEMA,measurement_domain='natural_predeclared_queue_training_v8',
        queue_training=True,monitor_serialization='RLock complete query and append; original sample midpoints retained',
        parent_training_context=receipt(profile_dir/'context.json'))
    save(output/'context.json',common);ctx=context_identity(common)
    aliases={a:dict(family=f,supported_tasks=['text','image','asr','ocr'] if f=='generalist_2'
        else ['ocr'] if f=='ocr' else ['text']) for a,f in RESIDENTS.items()}
    counts=Counter(r['task'] for r in trace)
    catalog=json.loads((BASE/'catalog.json').read_text())
    candidate=dict(id='shared_declared_training',execution_context_sha256=ctx,
        routes={r['request_id']:r['alias'] for r in trace},resource_configuration=resource_configuration())
    plan=dict(schema=SCHEMA,split='training',training_only=True,run_id=run_id,group_id=pattern,
        execution_context_sha256=ctx,context=receipt(output/'context.json'),
        profile_plan=receipt(profile_dir/'plan.json'),catalog=receipt(BASE/'catalog.json'),
        input_selection=profile['input_selection'],
        inherited_runtime_smoke=profile['inherited_runtime_smoke'],
        runner_source=receipt(__file__),collector_sources=[receipt(p) for p in collection_source_paths()],
        aliases=aliases,window_seconds=16.,drain_seconds=60.,warmup_seconds=0.,
        arrival_process='predeclared_exact_trace',trace=trace,trace_sha256=digest(trace),
        candidates=[candidate],baseline_candidate_id=candidate['id'],
        workload_rps={t:counts[t]/16. for t in ('text','image','asr','ocr')},
        latency_reference_ms=dict(text=1000,image=250,asr=3000,ocr=10000),weights={k:1/6 for k in 'QTLPME'},
        memory_safe_factor=catalog['safe_factor'],memory_budget_mib=49140,energy_reference_watts=450.,
        startup_amortization_seconds=3600.,
        cutoff_offsets_ns=[round(t*1e9) for t in (.05,.15,.35,.75,1.5,2.6,2.9,3.4,4.5,6.,8.,9.1,9.5,10.,11.,12.,14.)],
        risk_target=dict(horizon_ns=2_000_000_000,recovery_threshold_ms=500.),
        scope='TRAINING_ONLY_FULL_DECLARED_ARRIVALS_NOT_INDEPENDENT_PREDICTION_OR_DEPLOYMENT_GAIN',
        evidence_limits=['New collector context; no automatic model transfer qualification',
            'Observed host feedback is not an observed GPU cursor',
            'Short controlled bursts do not establish sustained stability'],GPU_started=False)
    plan['plan_sha256']=digest(plan);save(output/'plan.json',plan);verify_sources(plan)
    return plan

def checked(reference):
    if receipt(reference['path'])!=reference:
        raise ValueError('Changed bound source: '+reference['path'])
    return Path(reference['path'])


def execute(plan_path,candidate_id,output,*,protocol_path=None,predictions_path=None,run_id=None):
    plan_path,output=Path(plan_path).resolve(),Path(output).resolve()
    plan=json.loads(plan_path.read_text())
    if plan.get('schema')!=SCHEMA or digest({k:v for k,v in plan.items() if k!='plan_sha256'})!=plan['plan_sha256']:
        raise ValueError('Flow plan schema/hash mismatch')
    if any(v is not None for v in (protocol_path,predictions_path,run_id)):
        raise ValueError('Training collection cannot be relabeled as validation')
    validate_plan(plan)
    gate_arguments=(protocol_path,predictions_path,run_id)
    if any(v is not None for v in gate_arguments) and not all(v is not None for v in gate_arguments):
        raise ValueError('Validation requires protocol, predictions and run ID together')
    precollection=None
    if protocol_path is not None:
        if plan.get('split')=='training' or plan.get('phase') in ('training','predictor_training'):
            raise ValueError('Training plans cannot be relabeled as independent validation')
        from .prediction_observed_v2 import verify_before_collection
        protocol=json.loads(Path(protocol_path).read_text())
        if any(receipt(p) not in protocol.get('sources',[]) for p in collection_source_paths()):
            raise ValueError('Validation protocol must freeze the actual flow collector source closure')
        precollection=verify_before_collection(protocol_path,predictions_path,plan_path,
            run_id=run_id,candidate_id=candidate_id,output_root=output)
    checked(plan['runner_source'])
    for ref in plan.get('collector_sources',[]):checked(ref)
    checked(plan['catalog'])
    context=json.loads(checked(plan['context']).read_text())
    profile_plan=json.loads(checked(plan['profile_plan']).read_text())
    if context_identity(context)!=plan['execution_context_sha256']:
        raise ValueError('Execution context mismatch')
    for source in context['execution_sources']:
        checked(source['frozen'])
        if receipt(source['original_path'])['sha256']!=source['frozen']['sha256']:
            raise ValueError('Changed worker execution source: '+source['original_path'])
    candidate=next(c for c in plan['candidates'] if c['id']==candidate_id)
    registry={}
    for alias,family in RESIDENTS.items():
        entry=copy.deepcopy(context['worker_bindings'][family])
        entry['command'][1]=str(Path(__file__).with_name(context['worker_variant']))
        entry['env'].update(context['threads']);entry['env']['PYTHONUNBUFFERED']='1'
        registry[alias]=entry
    registry=prepare_registry(registry,candidate['resource_configuration'])
    # READY checks include every route used by the flow or priming, even when
    # the measured text route happens to omit the generalist.
    readiness_routes={'text':{'text#0':.5,'text#1':.49,'generalist_2#0':.01},
                      'image':{'generalist_2#0':1.},'asr':{'generalist_2#0':1.},'ocr':{'ocr#0':1.}}
    config=MeasurementConfig(deployment=dict.fromkeys(RESIDENTS,1),workload_rps=plan['workload_rps'],
        routing_rps=readiness_routes,startup_timeout_seconds=600,shutdown_timeout_seconds=30,
        drain_timeout_seconds=plan['drain_seconds'],measurement_seconds=plan['window_seconds'],
        routing_source='frozen_full_arrival_static_candidate')
    output.mkdir(parents=True,exist_ok=False)
    save(output/'invocation.json',dict(plan=receipt(plan_path),candidate_id=candidate_id,runner=receipt(__file__)))
    runtime=DeploymentRuntime(config,registry)
    monitor=None
    raw=dict(schema='gsdeploy.native_flow_collection.v1',kind='measured',backend='native',status='starting',
        split='training',training_only=True,queue_training_schema=SCHEMA,run_id=plan['run_id'],group_id=plan['group_id'],
        execution_context_sha256=plan['execution_context_sha256'],plan=receipt(plan_path),context=plan['context'],
        candidate_id=candidate_id,trace_sha256=plan['trace_sha256'],resource_configuration=candidate['resource_configuration'],
        resident_aliases=list(RESIDENTS),replica_to_family=RESIDENTS,
        request_records=[],priming_request_records=[],warmup_request_records=[],errors=[],
        effective_objective={k:plan[k] for k in ('workload_rps','latency_reference_ms','weights','memory_safe_factor',
            'memory_budget_mib','energy_reference_watts','startup_amortization_seconds')})
    if precollection is not None:
        raw.update(split='validation',run_id=run_id,precollection=receipt(precollection['receipt_path']))
    pool={r['pool_id']:r for r in (json.loads(line) for line in (BASE/'runtime-inputs.jsonl').read_text().splitlines())}
    lock=open(ROOT/'artifacts/.gpu-experiment.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        verify_sources(plan)
        raw['collection_started_unix_ns']=time.time_ns()
        save(output/'execution-start.json',dict(plan=receipt(plan_path),runner=receipt(__file__),
            started_unix_ns=raw['collection_started_unix_ns'],split='training',training_only=True,
            full_trace_predeclared=True,original_measurement_window_seconds=plan['window_seconds']))
        if precollection is not None:
            # Another experiment can hold the GPU lock for minutes. Recheck
            # frozen identities after that wait, before this collection starts.
            from .prediction_observed_v2 import _bindings
            for key in ('protocol','predictions','plan','adapter_source','scorer_source'):
                checked(precollection[key])
            checked(raw['precollection'])
            _bindings(protocol_path,predictions_path,plan_path,run_id)
            # Record the real collection boundary before any board sampling or
            # worker startup, after waiting for the exclusive GPU lock. Never
            # reconstruct this timestamp from completed output metadata.
            raw['collection_started_unix_ns']=time.time_ns()
            if raw['collection_started_unix_ns']<=precollection['checked_unix_ns']:
                raise RuntimeError('Collection must start after prospective prediction check')
        monitor=BoardMonitor(runtime)
        monitor.collection_budget={**context['collection_budget'],'path':str(output)}
        monitor.phase='startup'
        monitor.sample();monitor.check();raw['startup_begin_ns']=time.perf_counter_ns();monitor.start()
        runtime.start();raw['startup_end_ns']=time.perf_counter_ns()
        for alias,worker in runtime.workers.items():
            pid=runtime.ready[alias].get('pid')
            if type(pid) is not int or not descendant_or_self(pid,worker.process.pid):
                raise RuntimeError('Unowned READY PID: '+alias)
        raw['ready']=copy.deepcopy(runtime.ready)
        monitor.sample();monitor.check()
        print(json.dumps(dict(event='ready',candidate=candidate_id)),flush=True)
        monitor.phase='priming'
        for alias,family in RESIDENTS.items():
            tasks=['text','image','asr'] if family=='generalist_2' else ['ocr' if family=='ocr' else 'text']
            for task in tasks:
                selected=sorted((r for r in profile_plan['inputs'] if r['task']==task),
                                key=lambda r:r['input_size'],reverse=True)[:3]
                for item in selected:
                    row=record('priming',len(raw['priming_request_records']),task,alias,pool[item['pool_id']],time.perf_counter_ns())
                    raw['priming_request_records'].append(row);runtime.enqueue(row,pool[item['pool_id']]['sample'])
                    runtime.drain([row],300)
                    monitor.check()
                    if row['status']!='completed':
                        raise RuntimeError('Priming failed')
        def wait(target):
            while time.perf_counter_ns()<target:
                monitor.check();time.sleep(min(.005,max(0,(target-time.perf_counter_ns())/1e9)))
        # Warmup is separately offered/drained, never inserted into measurement.
        monitor.phase='warmup'
        warmup_start=time.perf_counter_ns()
        for index,item in enumerate(plan['trace']):
            if item['offset_ns']>=plan['warmup_seconds']*1e9:
                break
            target=warmup_start+item['offset_ns'];wait(target)
            row=record('warmup',index,item['task'],candidate['routes'][item['request_id']],pool[item['pool_id']],target)
            raw['warmup_request_records'].append(row);runtime.enqueue(row,pool[item['pool_id']]['sample'])
        wait(warmup_start+round(plan['warmup_seconds']*1e9));runtime.drain(raw['warmup_request_records'],plan['drain_seconds'])
        if any(r['status']!='completed' for r in raw['warmup_request_records']):
            raise RuntimeError('Warmup failed')
        monitor.sample();monitor.check()
        start=time.perf_counter_ns();deadline=start+round(plan['window_seconds']*1e9)
        monitor.phase='measurement'
        raw.update(phase_start_ns=start,deadline_ns=deadline)
        for i,item in enumerate(plan['trace']):
            row=record('measurement',i,item['task'],candidate['routes'][item['request_id']],pool[item['pool_id']],start+item['offset_ns'])
            if row['request_id']!=item['request_id']:
                raise ValueError('Record request identity mismatch')
            row.update(split='training',validation_view='TRAINING_ONLY_NOT_VALIDATION')
            raw['request_records'].append(row)
        for i,(item,row) in enumerate(zip(plan['trace'],raw['request_records'])):
            wait(row['scheduled_time_ns']);runtime.enqueue(row,pool[item['pool_id']]['sample'])
            if i%50==0:
                print(json.dumps(dict(event='flow_progress',offered=i+1,total=len(plan['trace']))),flush=True)
        wait(deadline);raw['arrival_loop_end_ns']=time.perf_counter_ns()
        monitor.phase='drain'
        runtime.drain(raw['request_records'],plan['drain_seconds'])
        raw['observation_end_ns']=time.perf_counter_ns();monitor.sample();monitor.check()
        raw['status']='completed'
    except BaseException as exc:
        raw['status']='failed';raw['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        if monitor:
            try:monitor.close()
            except BaseException as exc:
                raw['status']='failed';raw['errors'].append('Monitor cleanup: '+repr(exc))
            raw['board_samples']=monitor.samples;raw['telemetry_errors']=monitor.errors
        try:runtime.close()
        except BaseException as exc:
            raw['status']='failed';raw['errors'].append('Worker cleanup: '+repr(exc))
        raw['shutdown']=runtime.shutdown_report;raw['runtime_errors']=runtime.errors
        try:raw['admission_snapshot_after']=runtime.admission.snapshot() if runtime.admission else None
        except BaseException as exc:
            raw['status']='failed';raw['errors'].append('Admission cleanup snapshot: '+repr(exc))
        try:verify_sources(plan)
        except (ValueError,OSError) as exc:
            raw['status']='failed';raw['errors'].append('Source drift: '+str(exc))
        raw['collection_complete']=raw['status']=='completed'
        raw['qualification_issues']=[]
        if any(r['status']!='completed' for r in raw['request_records']):
            raw['qualification_issues'].append('INCOMPLETE_OR_ERROR_REQUESTS')
        if runtime.errors or raw.get('telemetry_errors'):
            raw['qualification_issues'].append('RUNTIME_OR_TELEMETRY_ERRORS')
        if any(s.get('forced') or s.get('exit_code')!=0 or not s.get('bye_received') for s in runtime.shutdown_report.values()):
            raw['qualification_issues'].append('UNCLEAN_SHUTDOWN')
        try:save(output/'raw.json',raw)
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    # Quality scoring is offline. Original runtime samples/catalog can contain
    # references; downstream prediction adapters must project public fields and
    # reader-visible timing only, never pass these full documents to a model.
    catalog=json.loads((BASE/'catalog.json').read_text());totals={};quality_errors=[]
    for row in raw['request_records']:
        if row['status']!='completed':
            continue
        try:
            result=copy.deepcopy(row['worker_response'])
            if row['task']=='ocr':
                result['prediction']=page_ocr_text(result['prediction'])
            reference=catalog['quality']['references'][row['task']][row['pool_id']]
            numerator,denominator=sample_totals(row['task'],score_result(reference,result))
            totals[row['request_id']]=dict(task=row['task'],numerator=numerator,denominator=denominator)
        except (KeyError,TypeError,ValueError) as exc:
            quality_errors.append(dict(request_id=row['request_id'],error=str(exc)))
    evidence=dict(kind='measured_per_request',totals_by_request=totals,errors=quality_errors,
        source_receipts=[receipt(BASE/'catalog.json'),receipt(__file__),receipt(ROOT/'gsdeploy/tasks.py'),receipt(ROOT/'gsdeploy/quality.py')])
    # Keep the originally saved native raw immutable; annotated raw binds it.
    annotated={**raw,'quality_evidence':evidence,'native_raw':receipt(output/'raw.json')}
    save(output/'scored-raw.json',annotated)
    print(json.dumps(dict(status=raw['status'],candidate=candidate_id,requests=len(raw['request_records']),
                          quality_count=len(totals),errors=raw['errors'],output=str(output/'scored-raw.json'))),flush=True)
    return annotated


def execute_supervised(plan_path, output):
    """Continuous time/disk guard over loading, priming, measurement and drain."""
    plan_path,output=Path(plan_path).resolve(),Path(output).resolve()
    plan=json.loads(plan_path.read_text());verify_sources(plan)
    budget=dict(free_disk_reserve_bytes=12*2**30,disk_abort_margin_bytes=2**30,
        incremental_storage_scenario_bytes=6*2**30,total_timeout_seconds=1200.,
        watchdog_interval_seconds=.2,cleanup_grace_seconds=8.)
    if shutil.disk_usage(output.parent).free<18*2**30:
        raise ValueError('Training disk reserve plus declared storage scenario unavailable')
    output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','tools.predictor_validation_20261027.native_queue_training_v8',
             'cohort','--plan',str(plan_path),'--output',str(output/'collection')]
    execution=dict(schema='gsdeploy.native_queue_training_execution.v8',plan=receipt(plan_path),
        runner=receipt(__file__),command=command,budget=budget,split='training',training_only=True,
        started_unix_ns=time.time_ns(),status='FAILED_NO_RETRY',GPU_started=False,
        watchdog_scope='Entire owned process group; GPU lock held only by cohort child')
    save(output/'execution-start.json',execution)
    process=None
    try:
        with (output/'stdout.log').open('x') as stdout,(output/'stderr.log').open('x') as stderr:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
            execution['GPU_started']=True;execution['process_group']=process.pid
            execution['watchdog']=wait_bounded(process,output,budget)
        verify_sources(plan)
        path=output/'collection/raw.json'
        raw=json.loads(path.read_text()) if path.exists() else None
        if (execution['watchdog']['status']=='EXITED' and execution['watchdog']['return_code']==0
                and raw is not None and raw.get('status')=='completed'
                and raw.get('collection_complete') is True and not raw.get('qualification_issues')):
            execution['status']='COLLECTED_TRAINING_ONLY'
        execution['raw']=receipt(path) if path.exists() else None
    except BaseException as exc:
        execution['error']=repr(exc)
    finally:
        try:
            if process is not None and group_members(process.pid):
                execution['exception_cleanup']=cleanup_group(process,budget['cleanup_grace_seconds'])
                execution['status']='FAILED_NO_RETRY'
        except BaseException as exc:
            execution['cleanup_error']=repr(exc);execution['status']='FAILED_NO_RETRY'
        try:
            verify_sources(plan)
            execution['final_source_audit']='PASS'
        except BaseException as exc:
            execution['final_source_audit']='FAIL';execution['source_error']=repr(exc)
            execution['status']='FAILED_NO_RETRY'
        execution['ended_unix_ns']=time.time_ns()
        save(output/'execution.json',execution)
    return execution



def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--profile-dir',default=str(DEFAULT_PROFILE));p.add_argument('--output',required=True)
    p.add_argument('--pattern',choices=('burst_A','burst_B'),default='burst_A');p.add_argument('--run-id',required=True)
    p=sub.add_parser('audit');p.add_argument('--plan',required=True)
    p=sub.add_parser('execute');p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true',required=True)
    p=sub.add_parser('cohort');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.op=='prepare':
        plan=prepare(args.profile_dir,args.output,pattern=args.pattern,run_id=args.run_id)
        print(json.dumps(dict(status='CPU_PREPARED_TRAINING_ONLY',requests=len(plan['trace']),plan=receipt(Path(args.output)/'plan.json'))))
    elif args.op=='audit':
        verify_sources(json.loads(Path(args.plan).read_text()));print(json.dumps(dict(status='CPU_AUDIT_PASS')))
    elif args.op=='cohort':
        raw=execute(args.plan,'shared_declared_training',args.output)
        return 0 if raw.get('status')=='completed' else 1
    else:
        result=execute_supervised(args.plan,args.output)
        print(json.dumps({k:v for k,v in result.items() if k not in ('watchdog',)},indent=2))
        return 0 if result['status']=='COLLECTED_TRAINING_ONLY' else 1
    return 0

if __name__=='__main__':raise SystemExit(main())
