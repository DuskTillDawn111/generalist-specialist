"""Isolated four-owner collector with externally held experiment lease.

Collection body preserves background_collect v7 execution semantics, with its
internal lock replaced by a validated live supervisor lease. Old source intact.
"""
from __future__ import annotations
import copy
import ctypes as C
import json
import os
from pathlib import Path
import time
import threading

from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from gsdeploy.runtime.schema import MeasurementConfig
from .background_collect import (ROOT, BoardMonitor, ProcessInfo, context_identity, receipt,
                                 save, descendant_or_self)
from .profiling import RESIDENTS
from .profile_training_supervisor_v3 import validate_lease

MONITORING_SCOPE = 'startup_through_final_drain_sample_before_worker_close'

def _proc_identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid, state=fields[0], ppid=int(fields[1]),
        process_group=int(fields[2]), session=int(fields[3]), start_ticks=int(fields[19]))


class LiveCollectorOwnership:
    """Pin real live Popen children; certify ancestry across sessions/groups.

    Prior observations never substitute for current ancestry. A reused PID,
    vanished edge or reparented worker is not a certified owner. PGID is retained
    as evidence only: nsys legitimately creates another session/process group.
    """
    def __init__(self, collector_pid):
        self.collector = _proc_identity(collector_pid)
        self.launchers = {}

    def chain(self, pid):
        chain, seen = [], set()
        while pid > 0 and pid not in seen:
            seen.add(pid)
            row = _proc_identity(pid)
            if row['state'] == 'Z':
                raise ValueError('Zombie cannot certify live GPU ownership')
            chain.append(row)
            if pid == self.collector['pid']:
                if row['start_ticks'] != self.collector['start_ticks']:
                    raise ValueError('Collector PID identity changed')
                break
            pid = row['ppid']
        # Detect concurrent reparent/exit/reuse rather than accepting a stitched
        # chain assembled from different process identities.
        for row in chain:
            current = _proc_identity(row['pid'])
            if (current['start_ticks'],current['ppid']) != (row['start_ticks'],row['ppid']) or current['state']=='Z':
                raise ValueError('Unstable live process ancestry')
        return chain

    def launcher(self, process):
        # poll uses this Popen's original waitable child. If it died before the
        # first observation, a recycled /proc PID must never become its identity.
        if process.poll() is not None:
            raise ValueError('Declared launcher exited')
        row = _proc_identity(process.pid)
        if process.poll() is not None or row['state']=='Z':
            raise ValueError('Declared launcher exited while identity was pinned')
        key = (process.pid,row['start_ticks'])
        previous = self.launchers.get(id(process))
        if previous is not None and previous['identity'] != key:
            raise ValueError('Declared launcher PID identity changed')
        # Keep the Popen alive too: Python id reuse cannot recycle this binding.
        self.launchers[id(process)] = dict(process=process,identity=key)
        return key


def memory_ownership(values, runtime, collector_pid, process_group, *, ownership=None):
    """Resolve only current declared-launcher descendants, including pre-READY."""
    ownership = ownership if ownership is not None else LiveCollectorOwnership(collector_pid)
    owned, foreign, unknown, aliases, evidence, reasons = {}, [], [], set(), [], []
    for pid, mib in values.items():
        try:
            if ownership.collector['pid'] != collector_pid:
                raise ValueError('Collector tracker identity mismatch')
            chain = ownership.chain(pid)
            by_pid = {r['pid']:r for r in chain}
            if collector_pid not in by_pid:
                foreign.append(pid); continue
            candidates = []
            for name, worker in runtime.workers.copy().items():
                process = getattr(worker,'process',None)
                if process is None or process.pid not in by_pid:
                    continue
                identity = ownership.launcher(process)
                if identity != (process.pid,by_pid[process.pid]['start_ticks']):
                    raise ValueError('Launcher changed across ancestry observation')
                candidates.append((name,identity))
            if len(candidates) != 1:
                foreign.append(pid); continue
            alias, identity = candidates[0]
            # Recheck after Popen pinning, which may itself race with exit.
            final = ownership.chain(pid)
            if [(r['pid'],r['start_ticks'],r['ppid']) for r in chain] != [
                    (r['pid'],r['start_ticks'],r['ppid']) for r in final]:
                raise ValueError('Ancestry changed during ownership validation')
            owned[pid] = mib; aliases.add(alias)
            evidence.append(dict(pid=pid,start_ticks=chain[0]['start_ticks'],
                process_group=chain[0]['process_group'],collector_pid=collector_pid,
                collector_start_ticks=ownership.collector['start_ticks'],
                collector_process_group=process_group,alias=alias,
                launcher_pid=identity[0],launcher_start_ticks=identity[1],
                ancestry=chain,process_group_equality_required=False,
                owner_scope='live_declared_launcher_descendant_with_stable_pid_start_ticks'))
        except (OSError,ValueError,IndexError,AttributeError) as exc:
            unknown.append(pid); reasons.append(dict(pid=pid,reason=str(exc)))
    return dict(owned_used_mib=sum(owned.values()), owned_processes_mib=owned,
        owned_aliases=sorted(aliases),foreign_pids=sorted(foreign),
        unresolved_nvml_pids=sorted(unknown),ownership_evidence=evidence,
        unresolved_ownership_reasons=reasons)


class StartupBoardMonitor(BoardMonitor):
    """Sample owned descendants before load through final drain, before close."""
    def __init__(self,runtime,interval=.02):
        self._sample_lock=threading.RLock()
        self._close_lock=threading.Lock();self._closed=False
        self.collector_pid=os.getpid();self.process_group=os.getpgrp();self.phase='startup'
        self.ownership=LiveCollectorOwnership(self.collector_pid)
        super().__init__(runtime,interval)

    def sample(self):
        with self._sample_lock:
            return self._sample()

    def _sample(self):
        try:
            a=time.perf_counter_ns();power=C.c_uint()
            self.check_rc(self.lib.nvmlDeviceGetPowerUsage(self.handle,C.byref(power)))
            processes=(ProcessInfo*128)();n=C.c_uint(128)
            self.check_rc(self.lib.nvmlDeviceGetComputeRunningProcesses_v2(self.handle,C.byref(n),processes))
            values={int(p.pid):int(p.used)/2**20 for p in processes[:n.value]}
            ownership=memory_ownership(values,self.runtime,self.collector_pid,self.process_group,
                ownership=self.ownership)
            b=time.perf_counter_ns()
            row=dict(time_ns=(a+b)//2,query_duration_ns=b-a,watts=power.value/1000,
                processes_mib=values,phase=self.phase,**ownership)
            self.samples.append(row)
            if ownership['unresolved_nvml_pids']:
                self.errors.append('UNKNOWN_NVML_PID_OWNERSHIP:'+str(ownership['unresolved_nvml_pids']))
            return row
        except Exception as exc:
            self.errors.append(repr(exc))

    def check(self):
        with self._sample_lock:
            return super().check()

    def set_phase(self,phase):
        with self._sample_lock:
            self.phase=phase

    def close(self):
        """Stop/join before the last live-worker query; never sample shutdown.

        Joining without the sample lock lets an in-flight query finish. The
        outer supervisor retains the original wall-time/cleanup bound if NVML
        cannot return. Do not detach a still-running sampler after a timeout.
        """
        with self._close_lock:
            if self._closed:
                return
            self.stop.set()
            thread=getattr(self,'thread',None)
            if thread is not None:
                thread.join()
            try:
                self.set_phase('final_drain')
                self.sample()
            finally:
                try:
                    self.check_rc(self.lib.nvmlShutdown())
                finally:
                    self._closed=True


def finish_collection(runtime, monitor, raw, output):
    """Persist every telemetry error and close workers after monitoring stops."""
    try:
        if monitor is not None:
            try:
                monitor.close()
                monitor.check()
            except BaseException as exc:
                raw['status']='failed'
                raw['errors'].append('MonitoringCloseError: '+type(exc).__name__+': '+str(exc))
            finally:
                raw['board_samples']=monitor.samples
                raw['telemetry_errors']=monitor.errors
                raw['lifecycle_boundaries'].append(dict(boundary='monitoring_end',time_ns=time.perf_counter_ns()))
    finally:
        raw['lifecycle_boundaries'].append(dict(boundary='shutdown_begin',time_ns=time.perf_counter_ns()))
        try:
            runtime.close()
            raw['shutdown']=runtime.shutdown_report
            raw['runtime_errors']=runtime.errors
        except BaseException as exc:
            raw['status']='failed'
            raw['errors'].append('WorkerShutdownError: '+type(exc).__name__+': '+str(exc))
            raw['shutdown']=getattr(runtime,'shutdown_report',{})
            raw['runtime_errors']=getattr(runtime,'errors',[])
        finally:
            raw['lifecycle_boundaries'].append(dict(boundary='shutdown_end',time_ns=time.perf_counter_ns()))
            save(output/'raw.json',raw)


def collect(output, *, trace, lease_fd):
    validate_lease(lease_fd)
    from .paired_profile_campaign_v6 import audit
    declared = audit(output)
    if type(trace) is not bool or trace != (declared['mode'] == 'kernel'):
        raise ValueError('Collector mode must exactly match frozen paired plan')
    limit = split = contexts = None
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
    runtime = DeploymentRuntime(config, registry)
    monitor = None
    raw = dict(schema='gsdeploy.native_phase_collection.v1',
        collector_schema='gsdeploy.profile_owned_collection.v3', kind='measured', backend='native',
        status='starting', execution_context_sha256=plan['execution_context_sha256'],
        plan=receipt(output/'plan.json'), context=receipt(output/'context.json'),
        executed_jobs=receipt(output/'executed-jobs.json'),
        request_records=[], priming_request_records=[], batches=[], errors=[],
        performance_scope='profiler_perturbed' if trace else 'unprofiled_with_phase_markers',
        lifecycle_boundaries=[],monitoring_scope=MONITORING_SCOPE,
        shutdown_memory_sampled=False,
        owned_peak_scope='sampled_live_declared_launcher_descendants_across_PGID_startup_through_final_drain_before_worker_close_not_instantaneous',
        U=None, original_stability_evaluated=False, claim='CONTROLLED_PROFILES_NOT_END_TO_END_UTILITY')
    pool = {r['pool_id']:r for r in (json.loads(l) for l in (ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl').read_text().splitlines())}
    try:
        monitor = StartupBoardMonitor(runtime)
        monitor.sample();monitor.check();monitor.start()
        raw['lifecycle_boundaries'].append(dict(boundary='startup_begin',time_ns=time.perf_counter_ns()))
        runtime.start()
        raw['ready_ownership_evidence'] = {}
        for alias,worker in runtime.workers.items():
            pid=runtime.ready[alias].get('pid')
            if type(pid) is not int:
                raise RuntimeError('READY pid is not an owned live descendant: '+alias)
            ready_owner=memory_ownership({pid:0.},runtime,monitor.collector_pid,monitor.process_group,
                ownership=monitor.ownership)
            if ready_owner['owned_aliases'] != [alias] or ready_owner['unresolved_nvml_pids'] or ready_owner['foreign_pids']:
                raise RuntimeError('READY pid lacks stable declared-launcher ancestry: '+alias)
            raw['ready_ownership_evidence'][alias]=ready_owner['ownership_evidence']
        raw['ready'] = copy.deepcopy(runtime.ready)
        print(json.dumps(dict(event='all_residents_ready', ready=runtime.ready)), flush=True)
        raw['lifecycle_boundaries'].append(dict(boundary='startup_ready',time_ns=time.perf_counter_ns()))
        monitor.set_phase('priming')
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
        raw['lifecycle_boundaries'].append(dict(boundary='priming_end',time_ns=time.perf_counter_ns()))
        monitor.set_phase('measurement')
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
        raw['lifecycle_boundaries'].append(dict(boundary='measurement_and_drain_end',time_ns=raw['observation_end_ns']))
        monitor.sample(); monitor.check()
        raw['status'] = 'completed'
    except BaseException as exc:
        raw['status']='failed'
        raw['errors'].append(type(exc).__name__+': '+str(exc))
        print(json.dumps(dict(event='failed',error=raw['errors'][-1])),flush=True)
    finally:
        finish_collection(runtime,monitor,raw,output)
    print(json.dumps(dict(status=raw['status'],output=str(output/'raw.json'),requests=len(raw['request_records']))),flush=True)
    return raw
