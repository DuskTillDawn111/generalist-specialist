"""Training-only whole-57 host/lifecycle capture, with an explicit GPU entry.

prepare/audit are CPU-only. The arrival and four-route recipes are declared here;
no validation plan, request IDs, outcomes or fitted predictor is read. Execution
uses the existing native worker, FIFO dispatch, serialized identity telemetry,
single-session profiler and owned-process watchdog.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from gsdeploy.resources import prepare_registry
from gsdeploy.routing import Router
from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from gsdeploy.runtime.schema import MeasurementConfig
from . import unified_corun_collect_v8_1 as existing
from .native_collect import ROOT,context_identity,receipt,save
from .native_flow import resource_configuration
from .operator_profile_campaign_v2 import audit as source_audit
from .profiling import RESIDENTS,digest

PLAN_SCHEMA='gsdeploy.whole57_joint_host_lifecycle_training_plan.v23'
RAW_SCHEMA='gsdeploy.whole57_joint_host_lifecycle_training_raw.v23'
LIGHT='whole57-joint-light-v23'
NATURAL='whole57-joint-natural-v23'
DEFAULT_SOURCE=ROOT/'artifacts/predictor-gpu-20261027/resource-full-state-training-v5-20261028-prepared00/capture01/natural'
BUDGET=dict(free_disk_reserve_bytes=12*2**30,disk_abort_margin_bytes=2**30,
    incremental_storage_scenario_bytes=6*2**30,total_timeout_seconds=2400.,
    watchdog_interval_seconds=.2,cleanup_grace_seconds=8.)
RECIPE=dict(schema='gsdeploy.predeclared_whole57_training_recipe.v23',
    window_seconds=30.,drain_seconds=120.,warmup_seconds=0.,rates=dict(text=1.85,ocr=.05),
    phases=[dict(start=0.,end=10.,multiplier=.5),dict(start=10.,end=20.,multiplier=2.),dict(start=20.,end=30.,multiplier=.5)],
    task_thresholds=dict(text=0.,ocr=.73),
    routes=[dict(id='balanced',fraction=.5,spill=0.),dict(id='skewed',fraction=.25,spill=0.),
            dict(id='spill_low',fraction=.5,spill=1/6),dict(id='spill_high',fraction=.5,spill=1/3)],
    route_source='PREDECLARED_TRAINING_RECIPE_NOT_SELECTED_FROM_FAILED_VALIDATION',
    input_rule='Alternate sorted exact training pool IDs independently within each task',
    admission='uncapped_shared',batch_size=1,weight_reuse=False)


def read(path):
    with Path(path).open() as stream:return json.load(stream)


def checked(ref):
    if receipt(ref['path'])!=ref:raise ValueError('Changed training source: '+ref['path'])
    return read(ref['path'])


def make_trace(inputs,training_id):
    if not re.fullmatch(r'whole57-training-[A-Za-z0-9_-]+',training_id):
        raise ValueError('Fresh whole57-training-* identity required')
    pools={task:sorted({r['pool_id'] for r in inputs if r['task']==task}) for task in ('text','ocr')}
    if any(len(rows)!=2 for rows in pools.values()):raise ValueError('Two exact training inputs per measured task required')
    rows=[]
    for task,rate in RECIPE['rates'].items():
        index=0;threshold=RECIPE['task_thresholds'][task]
        while threshold<rate*RECIPE['window_seconds']:
            accumulated=0.
            for phase in RECIPE['phases']:
                offered=(phase['end']-phase['start'])*rate*phase['multiplier']
                if threshold<accumulated+offered:
                    offset=phase['start']+(threshold-accumulated)/(rate*phase['multiplier']);break
                accumulated+=offered
            rows.append(dict(offset_ns=round(offset*1e9),task=task,pool_id=pools[task][index%2]))
            index+=1;threshold+=1.
    rows.sort(key=lambda r:(r['offset_ns'],r['task']))
    for index,row in enumerate(rows):row['request_id']=training_id+':req'+str(index).zfill(3)
    if len(rows)!=57:raise ValueError('Complete 57-request training recipe required')
    return rows


def routes_for(trace,route_id):
    selected=next((r for r in RECIPE['routes'] if r['id']==route_id),None)
    if selected is None:raise ValueError('One of four predeclared training routes required')
    fraction,spill=selected['fraction'],selected['spill']
    text={'text#0':(1-spill)*fraction,'text#1':(1-spill)*(1-fraction)}
    if spill:text['generalist_2#0']=spill
    weights=dict(text=text,ocr={'ocr#0':1.});router=Router(weights)
    return {r['request_id']:router.choose(r['task']) for r in trace},weights


def collection_sources():
    return [Path(__file__).resolve(),*[Path(__file__).with_name(name) for name in (
        'unified_corun_collect_v8_1.py','unified_corun_collect_v8.py','native_flow.py',
        'semantic_worker.py','semantic_observer.py','operator_profile_campaign_v2.py',
        'trace_export.py','nvtx_interval_index_v2.py','session_clock_audit_v2.py',
        'unified_corun_review_v8_1.py','smoke_verifier_v3.py','whole57_joint_review_v23.py',
        'joint_host_scenarios_v13.py','startup_telemetry_review_v4_1.py')]]


def audit(run):
    run=Path(run).resolve();plan=read(run/'plan.json');context=read(run/'context.json')
    if (plan.get('schema')!=PLAN_SCHEMA or plan.get('split')!='training' or plan.get('training_only') is not True
            or plan.get('recipe')!=RECIPE or plan.get('domain') not in (LIGHT,NATURAL)
            or plan.get('plan_sha256')!=digest({k:v for k,v in plan.items() if k!='plan_sha256'})):
        raise ValueError('Exact predeclared training-only plan required')
    profile=checked(plan['training_profile_plan']);checked(plan['training_profile_context'])
    if profile.get('split')!='training' or profile.get('training_only') is not True:
        raise ValueError('Audited training-only source profile required')
    if profile['inputs']!=plan['inputs'] or make_trace(plan['inputs'],plan['training_id'])!=plan['trace']:
        raise ValueError('Whole original training recipe/inputs required')
    routes,weights=routes_for(plan['trace'],plan['route_id'])
    if (plan['routes']!=routes or plan['routing_weights']!=weights or plan['trace_sha256']!=digest(plan['trace'])
            or plan['resource_configuration']!=resource_configuration() or plan['weights']!={k:1/6 for k in 'QTLPME'}):
        raise ValueError('Original uncapped four-owner route and six equal weights required')
    if (context_identity(context)!=plan['execution_context_sha256'] or context['measurement_domain']!=plan['domain']
            or context['residents']!=RESIDENTS or context['execution']['physical_owners']!=4
            or context['execution']['batch_size']!=1 or context['execution']['weight_reuse'] is not False
            or context['worker_variant']!='semantic_worker.py' or context['child_profilers'] is not False
            or context['capture_mode']!='whole_lifetime' or context['collection_schema']!=RAW_SCHEMA
            or context['collection_budget']!=BUDGET
            or context['whole57_training_recipe_sha256']!=digest(RECIPE)
            or context['whole57_training_route_id']!=plan['route_id']
            or context['controller_python']!=sys.executable
            or context['profiler']!=('single-nsys-process-tree-cuda-nvtx-gpuctxsw' if plan['domain']==LIGHT else 'none-natural-with-semantic-markers')):
        raise ValueError('Exact current native shared-residency context required')
    for path in collection_sources():
        if not any(s['original_path']==str(path.resolve()) and s['frozen']['sha256']==receipt(path)['sha256'] for s in context['execution_sources']):
            raise ValueError('Actual collector/reviewer source omitted: '+str(path))
    for source in context['execution_sources']:
        if receipt(source['frozen']['path'])!=source['frozen'] or receipt(source['original_path'])['sha256']!=source['frozen']['sha256']:
            raise ValueError('Frozen collection source drift: '+source['original_path'])
    # Keep the established source/input/resource and checkpoint-stat audits.
    source_audit(run)
    existing.priming_items(plan);existing.build_registry(context)
    return dict(status='CPU_TRAINING_PREPARATION_AUDIT_PASS',requests=57,priming_requests=12,
        domain=plan['domain'],route_id=plan['route_id'],GPU_started=False)


def prepare(output,source_run=DEFAULT_SOURCE,*,training_id='whole57-training-r01'):
    source_run=Path(source_run).resolve();source_audit(source_run)
    profile=read(source_run/'plan.json');common=read(source_run/'context.json')
    if profile.get('split')!='training' or profile.get('training_only') is not True:
        raise ValueError('Only audited training-profile inputs may be read')
    trace=make_trace(profile['inputs'],training_id)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    originals={Path(s['original_path']).resolve() for s in common['execution_sources']};originals.update(collection_sources())
    snapshots=[]
    for index,path in enumerate(sorted(originals)):
        target=out/'source-snapshot'/str(index)/path.name;target.parent.mkdir(parents=True)
        shutil.copyfile(path,target);snapshots.append(dict(original_path=str(path),frozen=receipt(target)))
    runs=[]
    for domain in (LIGHT,NATURAL):
        for route in RECIPE['routes']:
            run=out/domain/route['id'];run.mkdir(parents=True)
            context=copy.deepcopy(common)
            context.update(execution_sources=snapshots,sources=[r['frozen'] for r in snapshots],
                measurement_domain=domain,collection_schema=RAW_SCHEMA,worker_variant='semantic_worker.py',
                child_profilers=False,capture_mode='whole_lifetime',execution_nonce=uuid.uuid4().hex,
                controller_python=sys.executable,nsys=shutil.which('nsys'),
                profiler='single-nsys-process-tree-cuda-nvtx-gpuctxsw' if domain==LIGHT else 'none-natural-with-semantic-markers',
                whole57_training_recipe_sha256=digest(RECIPE),whole57_training_route_id=route['id'],
                monitor_serialization='RLock covers complete sample, identity events and phase boundary sampling',
                collection_budget=copy.deepcopy(BUDGET))
            if context['nsys'] is None:raise ValueError('Nsight CLI required for paired CPU preparation')
            routes,weights=routes_for(trace,route['id'])
            plan=dict(schema=PLAN_SCHEMA,split='training',training_only=True,training_id=training_id,
                domain=domain,route_id=route['id'],recipe=copy.deepcopy(RECIPE),
                execution_context_sha256=context_identity(context),inputs=copy.deepcopy(profile['inputs']),
                trace=trace,trace_sha256=digest(trace),routes=routes,routing_weights=weights,
                resource_configuration=resource_configuration(),weights={k:1/6 for k in 'QTLPME'},
                training_profile_plan=receipt(source_run/'plan.json'),training_profile_context=receipt(source_run/'context.json'),
                input_selection=profile['input_selection'],inherited_runtime_smoke=profile['inherited_runtime_smoke'],
                source_identity_limits=['New instrumented training context, not validation or deployment gain',
                    'Same-execution host and native Nsight clocks retained; calibrated error is unknown',
                    'Natural semantic/host checkpoints are not first CUDA releases',
                    'Repeated owner requests remain one complete cohort; existing v17 one-owner bank cannot consume it without a new contract'],
                GPU_started=False)
            plan['plan_sha256']=digest(plan);save(run/'context.json',context);save(run/'plan.json',plan)
            runs.append(dict(run=str(run),plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),audit=audit(run)))
    result=dict(schema='gsdeploy.whole57_joint_training_campaign.v23',status='CPU_PREPARED_NOT_EXECUTED',
        runs=runs,requests_per_run=57,priming_requests_per_run=12,training_only=True,recipe=RECIPE,
        source_profile=receipt(source_run/'plan.json'),source_context=receipt(source_run/'context.json'),
        validation_plan_or_outcomes_read=False,validation_started=False,GPU_started=False)
    save(out/'campaign.json',result);return result


def clock_anchor(label):
    before=time.perf_counter_ns();unix=time.time_ns();after=time.perf_counter_ns()
    return dict(label=label,perf_before_ns=before,unix_ns=unix,perf_after_ns=after,
        host_clock='time.perf_counter_ns',unix_clock='time.time_ns',simultaneous_sample_claimed=False)


def register_and_offer(raw,runtime,plan,pool,monitor,*,start_ns,wait=None):
    """Register all 57 before first enqueue; sender pipe blocking cannot drop arrivals."""
    if wait is None:
        def wait(target):
            while time.perf_counter_ns()<target:
                monitor.check();time.sleep(min(.005,max(0,(target-time.perf_counter_ns())/1e9)))
    rows=[]
    for index,item in enumerate(plan['trace']):
        row=record('training',index,item['task'],plan['routes'][item['request_id']],pool[item['pool_id']],start_ns+item['offset_ns'])
        row.update(request_id=item['request_id'],split='training',validation_view='TRAINING_ONLY_NOT_VALIDATION')
        rows.append(row)
    raw['request_records']=rows;raw['phase_start_ns']=start_ns
    raw['observation_start_ns']=start_ns
    raw['deadline_ns']=start_ns+round(RECIPE['window_seconds']*1e9)
    for item,row in zip(plan['trace'],rows):
        wait(row['scheduled_time_ns']);runtime.enqueue(row,pool[item['pool_id']]['sample'])
    wait(raw['deadline_ns']);raw['arrival_loop_end_ns']=time.perf_counter_ns()
    return rows


def cohort(run):
    run=Path(run).resolve();audit(run);plan=read(run/'plan.json');context=read(run/'context.json')
    if os.environ.get(existing.NONCE_ENV)!=context['execution_nonce']:
        raise ValueError('Only the declared exclusive controller may launch this cohort')
    if (run/'raw.json').exists():raise FileExistsError('Training run already used')
    raw=dict(schema=RAW_SCHEMA,status='starting',split='training',training_only=True,
        plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),
        execution_context_sha256=plan['execution_context_sha256'],domain=plan['domain'],route_id=plan['route_id'],
        planned_public_cohort=copy.deepcopy(plan['trace']),request_records=[],priming_request_records=[],
        lifecycle_boundaries=[],clock_anchors=[clock_anchor('before_startup')],errors=[],
        controller_identity=existing.process_identity(os.getpid()),
        clock_contract=dict(host='Original host perf_counter_ns observations in this execution',
            CUDA='One native Nsight session for light only; retain session origin and causal clock brackets',
            per_owner_median_map_used_as_common_clock=False,calibrated_cross_clock_error_bound_ns=None),
        U=None,original_stability_evaluated=False,claim='TRAINING_OBSERVATIONS_NOT_INDEPENDENT_VALIDATION')
    runtime=None;monitor=None
    pool_path=next(s['original_path'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl'))
    allowed={r['pool_id'] for r in plan['inputs']}
    pool={r['pool_id']:r for r in map(json.loads,Path(pool_path).read_text().splitlines()) if r['pool_id'] in allowed}
    if set(pool)!=allowed:raise ValueError('Complete approved training runtime inputs required')
    try:
        registry=prepare_registry(existing.build_registry(context),plan['resource_configuration'])
        config=MeasurementConfig(deployment=dict.fromkeys(RESIDENTS,1),workload_rps=RECIPE['rates'],
            warmup_seconds=0.,
            routing_rps={'text':{'text#0':.49,'text#1':.49,'generalist_2#0':.02},'ocr':{'ocr#0':1.},
                'image':{'generalist_2#0':1.},'asr':{'generalist_2#0':1.}},
            startup_timeout_seconds=600,shutdown_timeout_seconds=30,drain_timeout_seconds=RECIPE['drain_seconds'],
            measurement_seconds=RECIPE['window_seconds'],routing_source='predeclared_training_recipe')
        monitor=existing.IdentityMonitor(SimpleNamespace(workers={},ready={}))
        monitor.collection_budget={**context['collection_budget'],'path':str(run)}
        monitor.phase='pre_startup';monitor.sample();monitor.check();monitor.start()
        existing.boundary(raw,monitor,'startup_begin')
        runtime=DeploymentRuntime(config,registry);monitor.runtime=runtime;runtime.start()
        for alias,worker in runtime.workers.items():
            pid=runtime.ready[alias].get('pid')
            if type(pid) is not int or not existing.descendant_or_self(pid,worker.process.pid):
                raise RuntimeError('Unowned READY PID: '+alias)
        raw['ready']=copy.deepcopy(runtime.ready)
        raw['live_owned_processes']={a:existing.process_identity(v['pid']) for a,v in runtime.ready.items()}
        existing.boundary(raw,monitor,'startup_end');existing.boundary(raw,monitor,'priming_begin')
        for alias,item in existing.priming_items(plan):
            row=record(plan['training_id']+':priming',len(raw['priming_request_records']),item['task'],alias,pool[item['pool_id']],time.perf_counter_ns())
            raw['priming_request_records'].append(row);runtime.enqueue(row,pool[item['pool_id']]['sample']);runtime.drain([row],300)
            monitor.check()
            if row['status']!='completed':raise RuntimeError('Original training priming failed')
        existing.boundary(raw,monitor,'priming_end');raw['clock_anchors'].append(clock_anchor('after_priming'))
        existing.boundary(raw,monitor,'active_begin')
        register_and_offer(raw,runtime,plan,pool,monitor,start_ns=time.perf_counter_ns())
        monitor.phase='drain';runtime.drain(raw['request_records'],RECIPE['drain_seconds'])
        raw['observation_end_ns']=time.perf_counter_ns();raw['clock_anchors'].append(clock_anchor('after_drain'))
        existing.boundary(raw,monitor,'active_end')
        if any(r['status']!='completed' for r in raw['request_records']):raise RuntimeError('Incomplete full training cohort')
        raw['status']='completed'
    except BaseException as exc:raw['status']='failed';raw['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        if runtime is not None:
            if monitor is not None:
                with monitor._sample_lock:
                    monitor.phase='shutdown'
                    raw['lifecycle_boundaries'].append(dict(boundary='shutdown_begin',time_ns=time.perf_counter_ns(),clock='host_perf_counter_ns'))
            try:runtime.close()
            except BaseException as exc:raw['status']='failed';raw['errors'].append('Runtime cleanup: '+repr(exc))
            raw['shutdown']=runtime.shutdown_report;raw['runtime_errors']=runtime.errors
            raw['lifecycle_boundaries'].append(dict(boundary='shutdown_end',time_ns=time.perf_counter_ns(),clock='host_perf_counter_ns'))
        if monitor is not None:
            try:monitor.close()
            except BaseException as exc:raw['status']='failed';raw['errors'].append('Monitor cleanup: '+repr(exc))
            raw['board_samples']=monitor.samples;raw['telemetry_errors']=monitor.errors;raw['live_identity_events']=monitor.identity_events
        raw['clock_anchors'].append(clock_anchor('after_shutdown'))
        try:audit(run);raw['final_source_audit']='PASS'
        except Exception as exc:raw['status']='failed';raw['final_source_audit']='FAIL';raw['errors'].append(str(exc))
        save(run/'raw.json',raw)
    return raw


def outer_command(run,context,domain):
    command=[context['controller_python'],'-m','tools.predictor_validation_20261027.whole57_joint_training_v23','cohort','--run',str(run)]
    if domain==NATURAL:return command
    if domain!=LIGHT:raise ValueError('Declared training domain required')
    return [context['nsys'],'profile','--trace=cuda,nvtx','--cuda-trace-scope=process-tree','--gpuctxsw=true',
        '--sample=none','--cpuctxsw=none','--trace-fork-before-exec=false','--wait=all','--capture-range=none',
        '--cuda-memory-usage=false','--force-overwrite=false','--output='+str(Path(run)/'trace-whole57'),*command]


def execute(run):
    run=Path(run).resolve();audit(run);context=read(run/'context.json');plan=read(run/'plan.json')
    if any((run/n).exists() for n in ('execution-start.json','execution.json','raw.json')) or list(run.glob('trace-whole57*')):
        raise FileExistsError('Exclusive training execution already used; no retries')
    budget=context['collection_budget'];command=outer_command(run,context,plan['domain'])
    with (ROOT/'artifacts/.gpu-experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);audit(run)
        if shutil.disk_usage(run).free<budget['free_disk_reserve_bytes']+budget['incremental_storage_scenario_bytes']:
            raise ValueError('Disk reserve plus declared trace budget unavailable')
        intent=dict(command=command,plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),started_unix_ns=time.time_ns(),training_only=True)
        save(run/'execution-start.json',intent)
        execution=dict(schema='gsdeploy.whole57_joint_training_execution.v23',status='FAILED_NO_RETRY',
            **intent,execution_start=receipt(run/'execution-start.json'),GPU_started=False);process=None
        try:
            with (run/'stdout.log').open('x') as stdout,(run/'stderr.log').open('x') as stderr:
                process=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True,
                    env={**os.environ,existing.NONCE_ENV:context['execution_nonce'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
                execution.update(GPU_started=True,process_group=process.pid,launched_process=existing.process_identity(process.pid))
                execution['watchdog']=existing.wait_bounded(process,run,budget)
            audit(run);raw=read(run/'raw.json') if (run/'raw.json').exists() else None
            if raw is not None:execution['raw']=receipt(run/'raw.json')
            if (execution['watchdog']['status']=='EXITED' and execution['watchdog']['return_code']==0
                    and raw is not None and raw['status']=='completed' and len(raw['request_records'])==57):
                execution['status']='COLLECTED_TRAINING_REQUIRES_INDEPENDENT_REVIEW'
        except BaseException as exc:execution['error']=repr(exc)
        finally:
            try:
                if process is not None and existing.group_members(process.pid):
                    execution['exception_cleanup']=existing.cleanup_group(process,budget['cleanup_grace_seconds']);execution['status']='FAILED_NO_RETRY'
            except BaseException as exc:
                execution['status']='FAILED_NO_RETRY';execution['cleanup_error']=repr(exc)
            try:audit(run);execution['final_source_audit']='PASS'
            except Exception as exc:execution['status']='FAILED_NO_RETRY';execution['final_source_audit']='FAIL';execution['source_error']=str(exc)
            execution['ended_unix_ns']=time.time_ns();save(run/'execution.json',execution);fcntl.flock(lock,fcntl.LOCK_UN)
    return execution


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True);p.add_argument('--source-run',default=str(DEFAULT_SOURCE));p.add_argument('--training-id',default='whole57-training-r01')
    for name in ('audit','cohort','execute'):
        p=sub.add_parser(name);p.add_argument('--run',required=True)
        if name=='execute':p.add_argument('--execute',action='store_true',required=True)
    args=parser.parse_args()
    result=prepare(args.output,args.source_run,training_id=args.training_id) if args.op=='prepare' else audit(args.run) if args.op=='audit' else cohort(args.run) if args.op=='cohort' else execute(args.run)
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','request_records','planned_public_cohort','priming_request_records','board_samples','live_identity_events')},indent=2))
    return 1 if result.get('status') in ('failed','FAILED_NO_RETRY') else 0


if __name__=='__main__':raise SystemExit(main())
