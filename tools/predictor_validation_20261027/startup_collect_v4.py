"""Startup-aware fixed-resident collection under a new source/context identity.

Active profiling uses the inherited per-worker Nsight path. System context-switch
profiling is a separate probe until a unified single-session runner is verified.
"""
from __future__ import annotations
import copy
import ctypes as C
import fcntl
import json
import shutil
from pathlib import Path
import time
from types import SimpleNamespace

from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from gsdeploy.runtime.schema import MeasurementConfig
from . import background_collect as base
from .background_collect import ROOT,RESIDENTS,receipt,save,context_identity,descendant_or_self


class MemoryInfo(C.Structure):
    _fields_=[('total',C.c_ulonglong),('free',C.c_ulonglong),('used',C.c_ulonglong)]


def owned_processes(values,workers,ready):
    """Resolve each actual NVML PID against its owned launcher, even before READY."""
    owned={};foreign=[]
    for pid in values:
        aliases=[alias for alias,worker in workers if worker.process is not None and
                 (pid==ready.get(alias,{}).get('pid') or descendant_or_self(pid,worker.process.pid))]
        if len(aliases)>1:raise ValueError('Ambiguous actual process owner')
        if aliases:owned[pid]=aliases[0]
        else:foreign.append(pid)
    return owned,sorted(foreign)


class StartupBoardMonitor(base.BoardMonitor):
    def __init__(self,runtime,interval=.02):
        super().__init__(runtime,interval)
        self.phase='before_startup'
        self.collection_budget=None
        self.collection_begin=time.monotonic()

    def sample(self):
        try:
            begin=time.perf_counter_ns();power=C.c_uint()
            self.check_rc(self.lib.nvmlDeviceGetPowerUsage(self.handle,C.byref(power)))
            memory=MemoryInfo();self.check_rc(self.lib.nvmlDeviceGetMemoryInfo(self.handle,C.byref(memory)))
            processes=(base.ProcessInfo*128)();count=C.c_uint(128)
            self.check_rc(self.lib.nvmlDeviceGetComputeRunningProcesses_v2(self.handle,C.byref(count),processes))
            values={int(p.pid):int(p.used)/2**20 for p in processes[:count.value]}
            workers=list(self.runtime.workers.items());ready=dict(self.runtime.ready)
            owned,foreign=owned_processes(values,workers,ready)
            end=time.perf_counter_ns()
            row=dict(time_ns=(begin+end)//2,query_duration_ns=end-begin,phase=self.phase,
                watts=power.value/1000,board_used_mib=memory.used/2**20,board_total_mib=memory.total/2**20,
                board_free_mib=memory.free/2**20,processes_mib=values,owned_pid_aliases=owned,
                owned_used_mib=sum(values[pid] for pid in owned),owned_aliases=sorted(set(owned.values())),
                owner_loading_states={alias:('READY' if alias in ready else 'LAUNCHED_WAITING_READY' if worker.process is not None else 'NOT_LAUNCHED') for alias,worker in workers},
                expected_owner_aliases=list(RESIDENTS),foreign_pids=foreign,
                peak_scope='NVML sampled process/board usage only; absent process not an allocated-memory-zero or instantaneous-peak claim')
            self.samples.append(row);return row
        except Exception as exc:self.errors.append(repr(exc))

    def check(self):
        super().check()
        if self.collection_budget is not None:
            budget=self.collection_budget
            if shutil.disk_usage(budget['path']).free<budget['free_disk_reserve_bytes']:
                raise RuntimeError('STARTUP_COLLECTION_DISK_RESERVE_EXCEEDED')
            if time.monotonic()-self.collection_begin>budget['total_timeout_seconds']:
                raise RuntimeError('STARTUP_COLLECTION_TOTAL_TIME_BUDGET_EXCEEDED')


def boundary(raw,monitor,name):
    event=dict(boundary=name,time_ns=time.perf_counter_ns(),unix_ns=time.time_ns(),clock='host_perf_counter_ns')
    raw['lifecycle_boundaries'].append(event);monitor.phase=name
    monitor.sample();monitor.check();return event


def collect(output, *, limit=None, trace=False, split=None, contexts=None):
    output = Path(output).resolve()
    if (output/'raw.json').exists() or any(output.glob('trace-*.nsys-rep')):
        raise FileExistsError('Collection output already exists; prepare a new run')
    context = json.loads((output/'context.json').read_text())
    plan = json.loads((output/'plan.json').read_text())
    for src in context['sources']:
        if receipt(src['path']) != src:
            raise ValueError('Frozen source changed: '+src['path'])
    for src in context.get('execution_sources',[]):
        if receipt(src['original_path'])['sha256'] != src['frozen']['sha256']:
            raise ValueError('Executable source changed: '+src['original_path'])
    if context_identity(context) != plan['execution_context_sha256']:
        raise ValueError('Context mismatch')
    jobs=[j for j in plan['jobs'] if (split is None or j['split']==split)
          and (contexts is None or j['context'] in contexts)]
    jobs=jobs[:limit] if limit else jobs
    if not jobs:
        raise ValueError('No matching profile jobs')
    save(output/'executed-jobs.json',dict(split=split,contexts=contexts,limit=limit,trace=trace,jobs=jobs))
    aliases = dict(RESIDENTS)
    registry = {}
    for alias, family in aliases.items():
        entry = copy.deepcopy(context['worker_bindings'][family])
        cmd = entry['command']
        cmd[1] = str(Path(__file__).with_name(context.get('worker_variant','observed_worker.py')).resolve())
        entry['env'].update(context['threads'])
        entry['env']['PYTHONUNBUFFERED'] = '1'
        if trace:
            # Start capture only after every resident has loaded and primed.
            entry['command'] = ['nsys','profile','--trace=cuda,nvtx','--sample=none',
                '--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop',
                '--cuda-memory-usage=false','--force-overwrite=false',
                '--output='+str(output/('trace-'+alias.replace('#','-'))), *cmd]
        registry[alias] = entry
    workload = dict(text=1., ocr=1.,image=.01,asr=.01)
    config = MeasurementConfig(deployment=dict.fromkeys(aliases,1), workload_rps=workload,
        routing_rps={'text':{'text#0':.5,'text#1':.5}, 'ocr':{'ocr#0':1.},
                     'image':{'generalist_2#0':.01},'asr':{'generalist_2#0':.01}},
        startup_timeout_seconds=600, shutdown_timeout_seconds=30,
        routing_source='controlled_profile_not_utility', measurement_seconds=1.)
    runtime = None
    monitor = None
    raw = dict(schema='gsdeploy.startup_native_phase_collection.v4', kind='measured', backend='native',
        status='starting', lifecycle_boundaries=[], execution_context_sha256=plan['execution_context_sha256'],
        plan=receipt(output/'plan.json'), context=receipt(output/'context.json'),
        executed_jobs=receipt(output/'executed-jobs.json'),
        request_records=[], priming_request_records=[], batches=[], errors=[],
        performance_scope='profiler_perturbed' if trace else 'unprofiled_with_phase_markers',
        U=None, original_stability_evaluated=False, claim='CONTROLLED_PROFILES_NOT_END_TO_END_UTILITY')
    lock = open(ROOT/'artifacts/.gpu-experiment.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    pool = {r['pool_id']:r for r in (json.loads(l) for l in (ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl').read_text().splitlines())}
    try:
        monitor=StartupBoardMonitor(SimpleNamespace(workers={},ready={}))
        monitor.collection_budget={**context['collection_budget'],'path':str(output)}
        monitor.phase='pre_startup';monitor.sample();monitor.check()
        boundary(raw,monitor,'startup_begin')
        monitor.start()
        runtime=DeploymentRuntime(config,registry);monitor.runtime=runtime
        runtime.start()
        for alias,worker in runtime.workers.items():
            pid=runtime.ready[alias].get('pid')
            if type(pid) is not int or not descendant_or_self(pid,worker.process.pid):
                raise RuntimeError('READY pid is not an owned live descendant: '+alias)
        raw['ready'] = copy.deepcopy(runtime.ready)
        print(json.dumps(dict(event='all_residents_ready', ready=runtime.ready)), flush=True)
        boundary(raw,monitor,'startup_end')
        boundary(raw,monitor,'priming_begin')
        # Public holdout inputs may be warmed, but their measurements are excluded
        # from fitting. This is a parameter-fit holdout, not an unseen-execution claim.
        for alias, family in aliases.items():
            tasks = ['text','image','asr'] if family=='generalist_2' else ['ocr' if family=='ocr' else 'text']
            selected = [item for task in tasks for item in sorted((x for x in plan['inputs'] if x['task']==task),
                        key=lambda x:x['input_size'], reverse=True)[:3]]
            for item in selected:
                task=item['task']
                r = record('priming', len(raw['priming_request_records']), task, alias, pool[item['pool_id']], time.perf_counter_ns())
                raw['priming_request_records'].append(r)
                runtime.enqueue(r, pool[item['pool_id']]['sample'])
                runtime.drain([r],300)
                if r['status'] != 'completed':
                    raise RuntimeError('Priming failed: '+str(r))
        boundary(raw,monitor,'priming_end')
        if trace:
            # bypass result collector for control by registering a normal pending row
            for alias in aliases:
                rid = 'profile_start:'+alias
                r = {'request_id':rid,'model':alias,'task':'__profile_start__','status':'submitted'}
                runtime.pending[rid] = r
                runtime.workers[alias].send(dict(op='profile_start',request_id=rid))
                runtime.drain([r],30)
                if r['status'] != 'completed':
                    raise RuntimeError('Profiler start failed')
        monitor.sample(); monitor.check()
        boundary(raw,monitor,'active_begin')
        raw['observation_start_ns'] = time.perf_counter_ns()
        time.sleep(.3)
        for index, job in enumerate(jobs):
            monitor.check()
            start = time.perf_counter_ns()
            rows=[]
            for alias, offset in sorted(job['start_offsets_ms'].items(), key=lambda kv:kv[1]):
                target = start+round(offset*1e6)
                while time.perf_counter_ns() < target:
                    time.sleep(min(.005,max(0,(target-time.perf_counter_ns())/1e9)))
                item = pool[job['inputs'][alias]]
                default_task='ocr' if alias=='ocr#0' else 'text'
                if item['task_type'] != job.get('tasks',{}).get(alias,default_task):
                    raise ValueError('Planned task does not match public input: '+alias)
                r = record('profile',len(raw['request_records']),item['task_type'],alias,item,target)
                r.update(profile_job_id=job['id'],split=job['split'],validation_view=job['validation_view'])
                raw['request_records'].append(r); rows.append(r)
                runtime.enqueue(r,item['sample'])
            runtime.drain(rows,300)
            raw['batches'].append(dict(job_id=job['id'],start_ns=start,end_ns=time.perf_counter_ns(),
                                      request_ids=[r['request_id'] for r in rows]))
            if any(r['status']!='completed' for r in rows):
                raise RuntimeError('Profile job failed: '+job['id'])
            if index%8==0 or index+1==len(jobs):
                print(json.dumps(dict(event='profile_progress',jobs=index+1,total=len(jobs),
                                      requests=len(raw['request_records']))),flush=True)
                # Intermediate observations survive an interrupted long collection.
                with (output/'progress.json').open('w') as f:
                    json.dump(dict(completed_jobs=index+1,total_jobs=len(jobs)),f)
        time.sleep(.3)
        raw['observation_end_ns'] = time.perf_counter_ns()
        boundary(raw,monitor,'active_end')
        monitor.sample(); monitor.check()
        raw['status'] = 'completed'
    except BaseException as exc:
        raw['status']='failed'
        raw['errors'].append(type(exc).__name__+': '+str(exc))
        print(json.dumps(dict(event='failed',error=raw['errors'][-1])),flush=True)
    finally:
        if runtime is not None:
            if monitor is not None:
                monitor.phase='shutdown'
                raw['lifecycle_boundaries'].append(dict(boundary='shutdown_begin',time_ns=time.perf_counter_ns(),clock='host_perf_counter_ns'))
            runtime.close()
            raw['shutdown']=runtime.shutdown_report
            raw['runtime_errors']=runtime.errors
            raw['lifecycle_boundaries'].append(dict(boundary='shutdown_end',time_ns=time.perf_counter_ns(),clock='host_perf_counter_ns'))
        if monitor is not None:
            monitor.close()
            raw['board_samples']=monitor.samples
            raw['telemetry_errors']=monitor.errors
        save(output/'raw.json',raw)
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()
    print(json.dumps(dict(status=raw['status'],output=str(output/'raw.json'),requests=len(raw['request_records']))),flush=True)
    return raw

