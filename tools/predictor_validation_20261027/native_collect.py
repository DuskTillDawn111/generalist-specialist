"""Predeclared, fixed-residency native profiling; no utility claim from batches."""
from __future__ import annotations
import argparse
import copy
import ctypes as C
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from gsdeploy.runtime.schema import MeasurementConfig
from .profiling import RESIDENTS, make_plan, digest


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def context_identity(context):
    value={k:v for k,v in context.items() if k not in {'sources','execution_sources'}}
    value['source_content']=[dict(original_path=s['original_path'],sha256=s['frozen']['sha256'])
                             for s in context['execution_sources']]
    return digest(value)


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def descendant_or_self(pid, launcher_pid):
    """Verify READY identity against the process tree we actually launched."""
    seen=set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            status=Path(f'/proc/{pid}/status').read_text()
        except (OSError,ValueError):
            return False
        if pid == launcher_pid:
            return True
        parents=[line.split()[1] for line in status.splitlines() if line.startswith('PPid:')]
        if len(parents)!=1:
            return False
        pid=int(parents[0])
    return False


class ProcessInfo(C.Structure):
    _fields_ = [('pid', C.c_uint), ('used', C.c_ulonglong),
                ('gpu_instance', C.c_uint), ('compute_instance', C.c_uint)]


class BoardMonitor:
    """NVML board power and per-owned-process VRAM in one common host clock."""
    def __init__(self, runtime, interval=.02):
        self.runtime, self.interval = runtime, interval
        self.samples, self.errors = [], []
        self.stop = threading.Event()
        self.lib = C.CDLL('libnvidia-ml.so.1')
        self.check_rc(self.lib.nvmlInit_v2())
        self.handle = C.c_void_p()
        self.check_rc(self.lib.nvmlDeviceGetHandleByIndex_v2(0, C.byref(self.handle)))

    @staticmethod
    def check_rc(rc):
        if rc:
            raise RuntimeError('NVML error ' + str(rc))

    def sample(self):
        try:
            a = time.perf_counter_ns()
            power = C.c_uint()
            self.check_rc(self.lib.nvmlDeviceGetPowerUsage(self.handle, C.byref(power)))
            processes = (ProcessInfo * 128)()
            n = C.c_uint(128)
            self.check_rc(self.lib.nvmlDeviceGetComputeRunningProcesses_v2(self.handle, C.byref(n), processes))
            values = {int(p.pid): int(p.used)/2**20 for p in processes[:n.value]}
            owned = {self.runtime.ready.get(name, {}).get('pid', w.process.pid): name
                     for name,w in self.runtime.workers.items() if w.process}
            b = time.perf_counter_ns()
            row = dict(time_ns=(a+b)//2, query_duration_ns=b-a, watts=power.value/1000,
                       processes_mib=values, owned_used_mib=sum(v for p,v in values.items() if p in owned),
                       owned_aliases=sorted(name for p,name in owned.items() if p in values),
                       foreign_pids=sorted(p for p in values if p not in owned))
            self.samples.append(row)
            return row
        except Exception as exc:
            self.errors.append(repr(exc))

    def check(self):
        if self.errors:
            raise RuntimeError('Telemetry failed: ' + self.errors[-1])
        if self.samples:
            row = self.samples[-1]
            if row['foreign_pids'] or row['owned_used_mib'] > 49140:
                raise RuntimeError('Foreign GPU process or GPU memory budget exceeded')

    def start(self):
        def loop():
            while not self.stop.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        self.sample()
        self.lib.nvmlShutdown()


def prepare(output, *, repetitions=3, input_count=4, offsets=(0,50,200), semantic=False):
    if type(input_count) is not int or not 4 <= input_count <= 32:
        raise ValueError('input_count must be an integer in [4,32]')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = ROOT/'artifacts/static-study-v6-20261021/inputs'
    inputs = [json.loads(line) for line in (base/'runtime-inputs.jsonl').read_text().splitlines()]
    workers = json.loads((base/'worker-registry.json').read_text())
    selected = []
    for task in ('text', 'ocr'):
        rows = []
        for row in inputs:
            if row['task_type'] != task:
                continue
            sample = row['sample']
            size = len(sample['context']) + len(sample['question']) if task == 'text' else os.path.getsize(sample['input'])
            rows.append(dict(task=task, pool_id=row['pool_id'], input_size=size))
        rows.sort(key=lambda r: (r['input_size'], r['pool_id']))
        selected += [rows[round(i*(len(rows)-1)/(input_count-1))] for i in range(input_count)]
    # Bind all loaded Python sources, frozen worker specs, input pool and settings.
    paths = {Path(__file__).resolve(), Path(__file__).with_name('observed_worker.py'),
             base/'runtime-inputs.jsonl', base/'worker-registry.json',
             ROOT/'gsdeploy/runtime/deployment.py', ROOT/'gsdeploy/runtime/transport.py'}
    paths.update((ROOT/'gsdeploy').rglob('*.py'))
    paths.update(Path(__file__).parent.glob('*.py'))
    for family in set(RESIDENTS.values()):
        entry = workers[family]
        paths.add(Path(entry['spec']))
        cmd = entry['command']
        source = Path(cmd[cmd.index('--source-root')+1])
        paths.update(source.rglob('*.py'))
        spec = json.loads(Path(entry['spec']).read_text())
        if spec.get('adapter_source'):
            paths.add(Path(spec['adapter_source']))
    execution_sources=[]
    for index,path in enumerate(sorted(paths)):
        frozen=output/'source-snapshot'/str(index)/path.name
        frozen.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,frozen)
        execution_sources.append(dict(original_path=str(path),frozen=receipt(frozen)))
    context = dict(residents=RESIDENTS, sources=[s['frozen'] for s in execution_sources],
                   execution_sources=execution_sources,
                   worker_bindings={f:workers[f] for f in sorted(set(RESIDENTS.values()))},
                   threads={k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                   execution=dict(batch_size=1, backend='retained-native-observed-v1', weight_reuse=False),
                   counters='UNKNOWN:ERR_NVGPUCTRPERM', device='RTX4090-49140MiB',
                   worker_variant='semantic_worker.py' if semantic else 'observed_worker.py',
                   phase_scope='native inference phase includes host and transfers; CUPTI association separate',
                   profiler='none-natural-timing')
    context_sha = context_identity(context)
    plan = make_plan(selected, context_sha, repetitions=repetitions, offsets_ms=offsets)
    save(output/'context.json', context)
    save(output/'plan.json', plan)
    print(json.dumps(dict(status='PREPARED', output=str(output), jobs=len(plan['jobs']), context_sha256=context_sha)), flush=True)
    return output


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
    workload = dict(text=1., ocr=1.)
    config = MeasurementConfig(deployment=dict.fromkeys(aliases,1), workload_rps=workload,
        routing_rps={'text':{'text#0':.5,'text#1':.5}, 'ocr':{'ocr#0':1.}},
        startup_timeout_seconds=600, shutdown_timeout_seconds=30,
        routing_source='controlled_profile_not_utility', measurement_seconds=1.)
    runtime = DeploymentRuntime(config, registry)
    monitor = None
    raw = dict(schema='gsdeploy.native_phase_collection.v1', kind='measured', backend='native',
        status='starting', execution_context_sha256=plan['execution_context_sha256'],
        plan=receipt(output/'plan.json'), context=receipt(output/'context.json'),
        executed_jobs=receipt(output/'executed-jobs.json'),
        request_records=[], priming_request_records=[], batches=[], errors=[],
        performance_scope='profiler_perturbed' if trace else 'unprofiled_with_phase_markers',
        U=None, original_stability_evaluated=False, claim='CONTROLLED_PROFILES_NOT_END_TO_END_UTILITY')
    lock = open(ROOT/'artifacts/.gpu-experiment.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    pool = {r['pool_id']:r for r in (json.loads(l) for l in (ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl').read_text().splitlines())}
    try:
        runtime.start()
        for alias,worker in runtime.workers.items():
            pid=runtime.ready[alias].get('pid')
            if type(pid) is not int or not descendant_or_self(pid,worker.process.pid):
                raise RuntimeError('READY pid is not an owned live descendant: '+alias)
        raw['ready'] = copy.deepcopy(runtime.ready)
        print(json.dumps(dict(event='all_residents_ready', ready=runtime.ready)), flush=True)
        monitor = BoardMonitor(runtime)
        monitor.start()
        # Prime largest three selected inputs per alias while all residents stay loaded.
        for alias, family in aliases.items():
            task = 'ocr' if family == 'ocr' else 'text'
            selected = sorted((x for x in plan['inputs'] if x['task']==task), key=lambda x:x['input_size'], reverse=True)[:3]
            for item in selected:
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
        monitor.sample(); monitor.check()
        raw['status'] = 'completed'
    except BaseException as exc:
        raw['status']='failed'
        raw['errors'].append(type(exc).__name__+': '+str(exc))
        print(json.dumps(dict(event='failed',error=raw['errors'][-1])),flush=True)
    finally:
        if monitor:
            monitor.close()
            raw['board_samples']=monitor.samples
            raw['telemetry_errors']=monitor.errors
        runtime.close()
        raw['shutdown']=runtime.shutdown_report
        raw['runtime_errors']=runtime.errors
        save(output/'raw.json',raw)
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()
    print(json.dumps(dict(status=raw['status'],output=str(output/'raw.json'),requests=len(raw['request_records']))),flush=True)
    return raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--output',required=True); a.add_argument('--repetitions',type=int,default=3)
    a.add_argument('--input-count',type=int,default=4)
    a.add_argument('--semantic',action='store_true')
    a=sub.add_parser('execute');a.add_argument('--output',required=True);a.add_argument('--limit',type=int)
    a.add_argument('--trace',action='store_true')
    a.add_argument('--split',choices=['training','validation']);a.add_argument('--contexts',nargs='+')
    args=p.parse_args()
    if args.op=='prepare':prepare(args.output,repetitions=args.repetitions,input_count=args.input_count,semantic=args.semantic)
    else:collect(args.output,limit=args.limit,trace=args.trace,split=args.split,contexts=args.contexts)


if __name__=='__main__':main()
