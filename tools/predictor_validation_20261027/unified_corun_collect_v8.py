"""Exclusive four-owner training cohort, one outer Nsight session, no child profiler.

Preparation and tests are CPU-only. GPU collection requires the explicit execute
CLI. Scheduler ownership remains UNKNOWN unless a subsequent review proves it.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from gsdeploy.runtime.schema import MeasurementConfig
from .native_collect import ROOT, receipt, save, context_identity
from .operator_profile_campaign_v2 import audit as source_audit
from .profiling import RESIDENTS, digest
from .startup_collect_v4 import StartupBoardMonitor, boundary, descendant_or_self

PLAN_SCHEMA = 'gsdeploy.unified_corun_training_plan.v8'
RAW_SCHEMA = 'gsdeploy.unified_corun_training_raw.v8'
LIGHT = 'unified-light'
NATURAL = 'natural'
NONCE_ENV = 'GS_UNIFIED_CORUN_EXECUTION_NONCE'
DEFAULT_SOURCE = ROOT/'artifacts/predictor-gpu-20261027/small-startup-training-patch-v4-20261028-prepared01/shared-SSG/light-kernel'
DEFAULT_PARAMETERS = ROOT/'artifacts/predictor-corun-training-evidence-plan-v8-20261028-final01/parameters.json'


def validate_jobs(jobs):
    if not jobs or len(jobs)>24 or sum(len(j['inputs']) for j in jobs)>50:
        raise ValueError('Bounded nonempty training cohort required')
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate job IDs')
    for j in jobs:
        owners=set(j['inputs'])
        if (not owners or not owners<=set(RESIDENTS) or owners!=set(j['tasks'])
                or owners!=set(j['start_offsets_ms']) or j.get('split')!='training'
                or j.get('validation_view')!='TRAINING_ONLY_NOT_VALIDATION'
                or any(type(v) not in (int,float) or not 0<=v<=1000 for v in j['start_offsets_ms'].values())):
            raise ValueError('Exact owner/task/offset training job required')
        if any(task not in ('text','image','asr','ocr') for task in j['tasks'].values()):
            raise ValueError('Unknown task')


def priming_items(plan):
    result=[]
    for alias,family in RESIDENTS.items():
        tasks=['text','image','asr'] if family=='generalist_2' else ['ocr' if family=='ocr' else 'text']
        for task in tasks:
            selected=sorted((x for x in plan['inputs'] if x['task']==task),key=lambda x:x['input_size'],reverse=True)[:3]
            result.extend((alias,item) for item in selected)
    if len(result)!=12:
        raise ValueError('Exactly twelve original owner-bound priming requests required')
    return result


def build_registry(context):
    registry={}
    for alias,family in RESIDENTS.items():
        entry=copy.deepcopy(context['worker_bindings'][family])
        command=entry['command']
        if len(command)<2 or any(Path(str(v)).name=='nsys' for v in command):
            raise ValueError('Plain native child command required')
        command[1]=str(Path(__file__).with_name('semantic_worker.py').resolve())
        entry['env'].update(context['threads'])
        entry['env'].update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        entry['env'][NONCE_ENV]=context['execution_nonce']
        registry[alias]=entry
    return registry


def audit(run):
    run=Path(run).resolve()
    result=source_audit(run)
    plan=json.loads((run/'plan.json').read_text());context=json.loads((run/'context.json').read_text())
    if (plan.get('schema')!=PLAN_SCHEMA or plan.get('split')!='training' or plan.get('training_only') is not True
            or plan.get('domain') not in (LIGHT,NATURAL) or context.get('measurement_domain')!=plan['domain']
            or context.get('collection_schema')!=RAW_SCHEMA or context.get('worker_variant')!='semantic_worker.py'
            or context.get('child_profilers') is not False or context.get('capture_mode')!='whole_lifetime'
            or context.get('affine_observation') is not False):
        raise ValueError('Honest exclusive unified training domain/context required')
    if (context.get('profiler') != ('single-nsys-process-tree-cuda-nvtx-gpuctxsw' if plan['domain']==LIGHT else 'none-natural-with-semantic-markers')
            or plan.get('paired_jobs_sha256')!=digest(plan['jobs'])
            or plan.get('plan_sha256')!=digest({k:v for k,v in plan.items() if k!='plan_sha256'})):
        raise ValueError('Plan/profiler identity changed')
    validate_jobs(plan['jobs']);priming_items(plan);build_registry(context)
    original=json.loads(Path(plan['original_plan']['path']).read_text())
    if receipt(plan['original_plan']['path'])!=plan['original_plan'] or original['inputs']!=plan['inputs']:
        raise ValueError('Original priming input identity changed')
    if receipt(plan['parameters']['path'])!=plan['parameters']:
        raise ValueError('Declared training parameter receipt changed')
    budget=context['collection_budget']
    if (budget['free_disk_reserve_bytes']<12*2**30 or not 0<budget['watchdog_interval_seconds']<=.5
            or not 0<budget['total_timeout_seconds']<=2400 or budget['disk_abort_margin_bytes']<2**30
            or budget['incremental_storage_scenario_bytes']<=budget['disk_abort_margin_bytes']):
        raise ValueError('Required disk floor and continuous watchdog interval')
    return {**result,'schema':PLAN_SCHEMA,'domain':plan['domain'],'priming_requests':12}


def prepare(output,source_run=DEFAULT_SOURCE,parameters=DEFAULT_PARAMETERS):
    output=Path(output).resolve();source_run=Path(source_run).resolve();parameters=Path(parameters).resolve()
    source_audit(source_run)
    original=json.loads((source_run/'plan.json').read_text());common=json.loads((source_run/'context.json').read_text())
    params=json.loads(parameters.read_text())
    jobs=copy.deepcopy(next(s['jobs'] for s in params['stages'] if s['id']=='B'))
    validate_jobs(jobs)
    if set(a for j in jobs for a in j['inputs'])!=set(RESIDENTS):
        raise ValueError('First capability cohort must measure every resident owner')
    output.mkdir(parents=True,exist_ok=False)
    originals={Path(s['original_path']).resolve() for s in common['execution_sources']}
    originals.update(Path(__file__).with_name(name).resolve() for name in (
        'unified_corun_collect_v8.py','unified_corun_review_v8.py','paired_training_work_review_v4.py',
        'startup_training_collection_review_v4_1.py','startup_telemetry_review_v4_1.py','smoke_verifier_v3.py','session_clock_audit_v2.py'))
    sources=[]
    for index,path in enumerate(sorted(originals)):
        frozen=output/'source-snapshot'/str(index)/path.name;frozen.parent.mkdir(parents=True)
        shutil.copyfile(path,frozen);sources.append(dict(original_path=str(path),frozen=receipt(frozen)))
    common.update(execution_sources=sources,sources=[s['frozen'] for s in sources],
        collection_schema=RAW_SCHEMA,worker_variant='semantic_worker.py',affine_observation=False,
        child_profilers=False,capture_mode='whole_lifetime',execution_nonce=uuid.uuid4().hex,
        GPU_context_scheduling='UNKNOWN until reliable PID/context provenance is verified',
        collection_budget=dict(free_disk_reserve_bytes=12*2**30,disk_abort_margin_bytes=2**30,
            incremental_storage_scenario_bytes=6*2**30,total_timeout_seconds=1200,
            watchdog_interval_seconds=.2,cleanup_grace_seconds=8),
        telemetry_scope='startup_before_runtime_construction_through_shutdown',
        unified_session_scope='whole lifetime: startup, all twelve priming requests, active jobs, shutdown; no child range controls',
        controller_python=sys.executable,nsys=shutil.which('nsys'))
    if common['nsys'] is None:raise ValueError('Nsight CLI required for declared paired preparation')
    runs={}
    for domain in (LIGHT,NATURAL):
        run=output/domain;run.mkdir()
        context=copy.deepcopy(common);context.update(measurement_domain=domain,
            profiler='single-nsys-process-tree-cuda-nvtx-gpuctxsw' if domain==LIGHT else 'none-natural-with-semantic-markers')
        plan=dict(schema=PLAN_SCHEMA,execution_context_sha256=context_identity(context),resident_aliases=RESIDENTS,
            inputs=original['inputs'],jobs=jobs,split='training',training_only=True,domain=domain,
            paired_jobs_sha256=digest(jobs),input_selection=original['input_selection'],
            inherited_runtime_smoke=original['inherited_runtime_smoke'],original_plan=receipt(source_run/'plan.json'),
            parameters=receipt(parameters),no_retries=True,GPU_started=False,
            claim='UNIFIED_TRAINING_MECHANISM_NOT_VALIDATION_OR_DEPLOYMENT_GAIN')
        plan['plan_sha256']=digest(plan)
        save(run/'context.json',context);save(run/'plan.json',plan)
        runs[domain]=dict(run=str(run),plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),audit=audit(run))
    result=dict(schema='gsdeploy.unified_corun_campaign.v8',status='CPU_PREPARED',runs=runs,
        jobs_per_domain=len(jobs),requests_per_domain=sum(len(j['inputs']) for j in jobs),priming_requests_per_domain=12,
        GPU_started=False,fit_performed=False,validation_started=False,
        execution_order=[LIGHT,NATURAL],storage_scenario_is_upper_bound=False,
        minimum_free_bytes=12*2**30,incremental_storage_scenario_bytes=6*2**30,
        scheduler_ownership='UNKNOWN',same_sm_concurrency='UNKNOWN')
    save(output/'campaign.json',result);return result


def process_identity(pid,proc_root=Path('/proc')):
    result=dict(pid=pid,observed_perf_ns=time.perf_counter_ns())
    try:
        path=Path(proc_root)/str(pid);status=(path/'status').read_text()
        fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status.splitlines() if ':' in line}
        stat=(path/'stat').read_text().rsplit(')',1)[1].split()
        env=(path/'environ').read_bytes().split(b'\0')
        nonce=next((x.split(b'=',1)[1].decode() for x in env if x.startswith((NONCE_ENV+'=').encode())),None)
        result.update(ppid=int(fields['PPid']),state=fields['State'].split()[0],nspid=[int(x) for x in fields.get('NSpid','').split()],
            start_ticks=int(stat[19]),process_group=int(stat[2]),session=int(stat[3]),
            pid_namespace_inode=os.stat(path/'ns/pid').st_ino,execution_nonce=nonce,
            namespace_interpretation='NSpid as exposed by this proc mount; host scheduler namespace not assumed')
    except (OSError,ValueError,IndexError,KeyError) as exc:result['error']=repr(exc)
    return result


def smi_pid_view():
    try:
        command=['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']
        r=subprocess.run(command,capture_output=True,text=True,timeout=2)
        return dict(time_ns=time.perf_counter_ns(),command=command,return_code=r.returncode,stdout=r.stdout,stderr=r.stderr,
            pid_namespace='UNKNOWN; do not assume host PIDs')
    except (OSError,subprocess.TimeoutExpired) as exc:return dict(time_ns=time.perf_counter_ns(),error=repr(exc),pid_namespace='UNKNOWN')


class IdentityMonitor(StartupBoardMonitor):
    def __init__(self,runtime):
        super().__init__(runtime);self.identity_events=[];self.last_fingerprint=None

    def sample(self):
        row=super().sample()
        if row is None:return row
        workers=list(self.runtime.workers.items());ready=dict(self.runtime.ready)
        fingerprint=tuple((a,w.process.pid if w.process else None,ready.get(a,{}).get('pid')) for a,w in workers)
        if fingerprint!=self.last_fingerprint:
            self.last_fingerprint=fingerprint
            self.identity_events.append(dict(time_ns=row['time_ns'],phase=self.phase,
                nvml_processes_mib=row['processes_mib'],nvml_pid_namespace='OBSERVED_VALUES_NO_NAMESPACE_ASSUMPTION',
                nvidia_smi=smi_pid_view(),ready=copy.deepcopy(ready),
                launchers={a:process_identity(w.process.pid) for a,w in workers if w.process},
                ready_processes={a:process_identity(v['pid']) for a,v in ready.items()},
                bridge_status='UNKNOWN: raw views retained; no arithmetic PID or context-ID join'))
        return row


def cohort(output):
    output=Path(output).resolve();audit(output)
    plan=json.loads((output/'plan.json').read_text());context=json.loads((output/'context.json').read_text())
    if os.environ.get(NONCE_ENV)!=context['execution_nonce']:
        raise ValueError('Cohort must be launched by its declared controller')
    if (output/'raw.json').exists():raise FileExistsError('Exclusive cohort already used')
    traced=plan['domain']==LIGHT;jobs=plan['jobs']
    save(output/'executed-jobs.json',dict(trace=traced,single_session=traced,capture_mode='whole_lifetime',
        child_profilers=False,profile_controls_sent=False,jobs=jobs))
    registry=build_registry(context)
    config=MeasurementConfig(deployment=dict.fromkeys(RESIDENTS,1),workload_rps=dict(text=1.,ocr=1.,image=.01,asr=.01),
        routing_rps={'text':{'text#0':.5,'text#1':.5},'ocr':{'ocr#0':1.},'image':{'generalist_2#0':.01},'asr':{'generalist_2#0':.01}},
        startup_timeout_seconds=600,shutdown_timeout_seconds=30,routing_source='controlled_profile_not_utility',measurement_seconds=1.)
    raw=dict(schema=RAW_SCHEMA,kind='measured',backend='native',status='starting',lifecycle_boundaries=[],
        execution_context_sha256=plan['execution_context_sha256'],plan=receipt(output/'plan.json'),context=receipt(output/'context.json'),
        executed_jobs=receipt(output/'executed-jobs.json'),request_records=[],priming_request_records=[],batches=[],errors=[],
        performance_scope='single_session_profiler_perturbed' if traced else 'unprofiled_with_phase_markers',
        measurement_domain=plan['domain'],child_commands={a:e['command'] for a,e in registry.items()},
        controller_nonce=context['execution_nonce'],cohort_identity=process_identity(os.getpid()),
        child_profilers=False,profile_controls_sent=False,owned_scheduler_attribution='UNKNOWN',
        U=None,original_stability_evaluated=False,claim='TRAINING_MECHANISM_NOT_DEPLOYMENT_UTILITY')
    pool={r['pool_id']:r for r in map(json.loads,(ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl').read_text().splitlines())}
    runtime=None;monitor=None
    try:
        monitor=IdentityMonitor(SimpleNamespace(workers={},ready={}))
        monitor.collection_budget={**context['collection_budget'],'path':str(output)}
        monitor.phase='pre_startup';monitor.sample();monitor.check()
        boundary(raw,monitor,'startup_begin');monitor.start()
        runtime=DeploymentRuntime(config,registry);monitor.runtime=runtime;runtime.start()
        for alias,worker in runtime.workers.items():
            pid=runtime.ready[alias].get('pid')
            if type(pid) is not int or not descendant_or_self(pid,worker.process.pid):raise RuntimeError('Unowned READY: '+alias)
        raw['ready']=copy.deepcopy(runtime.ready)
        raw['live_owned_processes']={a:process_identity(v['pid']) for a,v in runtime.ready.items()}
        boundary(raw,monitor,'startup_end');boundary(raw,monitor,'priming_begin')
        for alias,item in priming_items(plan):
            r=record('priming',len(raw['priming_request_records']),item['task'],alias,pool[item['pool_id']],time.perf_counter_ns())
            raw['priming_request_records'].append(r);runtime.enqueue(r,pool[item['pool_id']]['sample']);runtime.drain([r],300)
            if r['status']!='completed':raise RuntimeError('Original priming failed')
        boundary(raw,monitor,'priming_end');boundary(raw,monitor,'active_begin')
        raw['observation_start_ns']=time.perf_counter_ns();time.sleep(.3)
        for index,job in enumerate(jobs):
            monitor.check();start=time.perf_counter_ns();rows=[]
            for alias,offset in sorted(job['start_offsets_ms'].items(),key=lambda kv:kv[1]):
                target=start+round(offset*1e6)
                while time.perf_counter_ns()<target:time.sleep(min(.005,max(0,(target-time.perf_counter_ns())/1e9)))
                item=pool[job['inputs'][alias]]
                if item['task_type']!=job['tasks'][alias]:raise ValueError('Task/input mismatch')
                r=record('profile',len(raw['request_records']),item['task_type'],alias,item,target)
                r.update(profile_job_id=job['id'],split=job['split'],validation_view=job['validation_view'])
                rows.append(r);raw['request_records'].append(r);runtime.enqueue(r,item['sample'])
            runtime.drain(rows,300)
            raw['batches'].append(dict(job_id=job['id'],start_ns=start,end_ns=time.perf_counter_ns(),request_ids=[r['request_id'] for r in rows]))
            if any(r['status']!='completed' for r in rows):raise RuntimeError('Measured job failed: '+job['id'])
            print(json.dumps(dict(event='unified_progress',jobs=index+1,total=len(jobs),requests=len(raw['request_records']))),flush=True)
        time.sleep(.3);raw['observation_end_ns']=time.perf_counter_ns();boundary(raw,monitor,'active_end')
        raw['status']='completed'
    except BaseException as exc:
        raw['status']='failed';raw['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        if runtime is not None:
            if monitor is not None:monitor.phase='shutdown'
            raw['lifecycle_boundaries'].append(dict(boundary='shutdown_begin',time_ns=time.perf_counter_ns(),clock='host_perf_counter_ns'))
            try:runtime.close()
            except BaseException as exc:raw['errors'].append('Cleanup: '+repr(exc));raw['status']='failed'
            raw['shutdown']=runtime.shutdown_report;raw['runtime_errors']=runtime.errors
            raw['lifecycle_boundaries'].append(dict(boundary='shutdown_end',time_ns=time.perf_counter_ns(),clock='host_perf_counter_ns'))
        if monitor is not None:
            try:monitor.close()
            except BaseException as exc:raw['errors'].append('Monitor close: '+repr(exc));raw['status']='failed'
            raw['board_samples']=monitor.samples;raw['telemetry_errors']=monitor.errors
            raw['live_identity_events']=monitor.identity_events
        save(output/'raw.json',raw)
    return raw


def outer_command(run,context,domain):
    command=[context['controller_python'],'-m','tools.predictor_validation_20261027.unified_corun_collect_v8','cohort','--run',str(run)]
    if domain==NATURAL:return command
    if domain!=LIGHT:raise ValueError('Unknown profiler domain')
    return [context['nsys'],'profile','--trace=cuda,nvtx','--cuda-trace-scope=process-tree','--gpuctxsw=true',
        '--sample=none','--cpuctxsw=none','--trace-fork-before-exec=false','--wait=all','--capture-range=none',
        '--cuda-memory-usage=false','--force-overwrite=false','--output='+str(Path(run)/'trace-unified'),*command]


def group_members(pgid):
    members=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==pgid and fields[0]!='Z':members.append(int(path.name))
        except (OSError,ValueError,IndexError):pass
    return sorted(members)


def cleanup_group(process,grace_seconds=8):
    events=[]
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGKILL):
        members=group_members(process.pid)
        if not members:break
        try:os.killpg(process.pid,sig)
        except ProcessLookupError:break
        events.append(dict(signal=sig.name,owned_process_group=process.pid,members=members,time_ns=time.perf_counter_ns()))
        deadline=time.monotonic()+(0.5 if sig==signal.SIGKILL else grace_seconds)
        while group_members(process.pid) and time.monotonic()<deadline:
            process.poll();time.sleep(.05)
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:pass
    return dict(events=events,live_survivors=group_members(process.pid),scope='only launched new process session/group; zombies excluded from live survivors')


def wait_bounded(process,run,budget):
    begin=time.monotonic();initial_free=shutil.disk_usage(run).free;samples=[];reason=None
    while True:
        free=shutil.disk_usage(run).free;elapsed=time.monotonic()-begin
        samples.append(dict(elapsed_seconds=elapsed,free_bytes=free))
        if free<budget['free_disk_reserve_bytes']+budget['disk_abort_margin_bytes']:reason='DISK_MARGIN'
        elif initial_free-free>budget['incremental_storage_scenario_bytes']:reason='STORAGE_SCENARIO_EXCEEDED'
        elif elapsed>budget['total_timeout_seconds']:reason='TIMEOUT'
        if reason is not None:return dict(status=reason,samples=samples,cleanup=cleanup_group(process,budget['cleanup_grace_seconds']))
        code=process.poll()
        if code is not None:
            survivors=group_members(process.pid)
            return dict(status='EXITED' if not survivors else 'LIVE_DESCENDANTS_AFTER_EXIT',return_code=code,samples=samples,
                cleanup=cleanup_group(process,budget['cleanup_grace_seconds']) if survivors else dict(events=[],live_survivors=[]))
        time.sleep(budget['watchdog_interval_seconds'])


def execute(run):
    run=Path(run).resolve();audit(run)
    if any((run/name).exists() for name in ('execution-start.json','execution.json','raw.json')) or list(run.glob('trace-unified*')):
        raise FileExistsError('Exclusive run already used; no retries')
    context=json.loads((run/'context.json').read_text());plan=json.loads((run/'plan.json').read_text());budget=context['collection_budget']
    command=outer_command(run,context,plan['domain'])
    with (ROOT/'artifacts/.gpu-experiment.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        audit(run)
        if shutil.disk_usage(run).free<budget['free_disk_reserve_bytes']+budget['incremental_storage_scenario_bytes']:
            raise ValueError('Disk reserve plus declared startup-trace scenario unavailable')
        save(run/'execution-start.json',dict(command=command,plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),
            started_unix_ns=time.time_ns(),controller=process_identity(os.getpid()),no_retries=True))
        process=None;result=dict(schema='gsdeploy.unified_corun_execution.v8',status='FAILED_NO_RETRY',command=command,
            plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),started_unix_ns=time.time_ns(),GPU_started=False,
            execution_start=receipt(run/'execution-start.json'),lock_owner='outer executor only',
            domain=plan['domain'],watchdog_interval_seconds=budget['watchdog_interval_seconds'])
        try:
            with (run/'stdout.log').open('x') as stdout,(run/'stderr.log').open('x') as stderr:
                process=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True,
                    env={**os.environ,NONCE_ENV:context['execution_nonce'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
                result.update(GPU_started=True,launched_process=process_identity(process.pid),process_group=process.pid)
                result['watchdog']=wait_bounded(process,run,budget)
            raw=json.loads((run/'raw.json').read_text()) if (run/'raw.json').exists() else None
            audit(run)
            if (result['watchdog']['status']=='EXITED' and result['watchdog']['return_code']==0
                    and raw is not None and raw.get('status')=='completed'):
                result['status']='COLLECTED'
            result['raw']=receipt(run/'raw.json') if raw is not None else None
        except BaseException as exc:
            result['error']=repr(exc)
        finally:
            if process is not None and group_members(process.pid):
                result['exception_cleanup']=cleanup_group(process,budget['cleanup_grace_seconds'])
                result['status']='FAILED_NO_RETRY'
            result['ended_unix_ns']=time.time_ns();result['free_bytes_at_end']=shutil.disk_usage(run).free
            save(run/'execution.json',result)
            fcntl.flock(handle,fcntl.LOCK_UN)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True);p.add_argument('--source-run',default=str(DEFAULT_SOURCE));p.add_argument('--parameters',default=str(DEFAULT_PARAMETERS))
    for name in ('audit','cohort','execute'):
        p=sub.add_parser(name);p.add_argument('--run',required=True)
        if name=='execute':p.add_argument('--execute',action='store_true',required=True)
    args=parser.parse_args()
    result=prepare(args.output,args.source_run,args.parameters) if args.op=='prepare' else audit(args.run) if args.op=='audit' else execute(args.run) if args.op=='execute' else cohort(args.run)
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','request_records','priming_request_records','board_samples','live_identity_events')},indent=2))
    return 0 if result.get('status') not in ('failed','FAILED_NO_RETRY') else 1


if __name__=='__main__':raise SystemExit(main())
