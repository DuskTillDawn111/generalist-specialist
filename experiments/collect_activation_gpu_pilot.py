"""Bounded native collector pilot. Host phases are not pure GPU time or utility."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from gsdeploy.runtime.schema import MeasurementConfig
from gsdeploy.trace import make_trace
from tools.predictor_validation_20261027.native_collect import BoardMonitor, descendant_or_self


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def emit(**fields):
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def rebind_checkpoint(spec, destination):
    """Reverify bytes after relocation, then bind current stat in a new spec."""
    spec = copy.deepcopy(spec)
    checkpoint = spec['resources']['checkpoint']
    receipts = []
    for entry in checkpoint.get('files', []) if isinstance(checkpoint, dict) else []:
        path = Path(checkpoint['path'])/entry['name']
        before = path.stat()
        if before.st_size != entry['size']:
            raise ValueError('Checkpoint size mismatch: '+str(path))
        expected = entry.get('sha256') or entry.get('git_blob')
        if not expected:
            raise ValueError('No pinned content digest: '+str(path))
        h = hashlib.sha256() if entry.get('sha256') else hashlib.sha1()
        if not entry.get('sha256'):
            h.update(f'blob {before.st_size}\0'.encode())
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b''):
                h.update(block)
        after = path.stat()
        stable = lambda s: (s.st_ino, s.st_dev, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if stable(before) != stable(after) or h.hexdigest() != expected:
            raise ValueError('Checkpoint pinned digest mismatch: '+str(path))
        entry['stat'] = dict(inode=after.st_ino, device=after.st_dev, size=after.st_size,
                             mtime_ns=after.st_mtime_ns, ctime_ns=after.st_ctime_ns)
        receipts.append(dict(path=str(path), digest=expected,
                             algorithm='sha256' if entry.get('sha256') else 'git_blob_sha1'))
    # OCR has a separately pinned adapter/checkpoint contract; no invented stat.
    save(destination, spec)
    return receipts


def collect(output, *, budget_seconds=300, arrival_seconds=30, seed=6110401, profiled=False,
            complete_phases=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = ROOT/'artifacts/static-study-v6-20261021/inputs'
    pool = [json.loads(line) for line in (base/'runtime-inputs.jsonl').read_text().splitlines()]
    workers = json.loads((base/'worker-registry.json').read_text())
    aliases = {'generalist_2#0': 'generalist_2', 'image#0': 'image', 'ocr#0': 'ocr'}
    registry = {}
    observed = ROOT/'tools/predictor_validation_20261027'/('complete_observed_worker.py'
                 if complete_phases else 'observed_worker.py')
    sources = {Path(__file__), observed, base/'runtime-inputs.jsonl',
               ROOT/'gsdeploy/runtime/deployment.py', ROOT/'gsdeploy/runtime/transport.py',
               ROOT/'tools/predictor_validation_20261027/native_collect.py'}
    sources.add(ROOT/'tools/predictor_validation_20261027/observed_worker.py')
    checkpoint_receipts = []
    for alias, family in aliases.items():
        entry = copy.deepcopy(workers[family])
        entry['command'][1] = str(observed)
        entry['env'].update(dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                 OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1'))
        registry[alias] = entry
        spec = json.loads(Path(entry['spec']).read_text())
        sources.add(Path(entry['spec']))
        rebound = output/(family+'-worker.json')
        checkpoint_receipts.extend(rebind_checkpoint(spec, rebound))
        entry['spec'] = str(rebound)
        entry['spec_sha256'] = receipt(rebound)['sha256']
        command = entry['command']
        command[command.index('--spec')+1] = str(rebound)
        sources.update(Path(command[command.index('--source-root')+1]).rglob('*.py'))
        if spec['resources'].get('protocol'):
            sources.add(Path(spec['resources']['protocol']))
        sources.add(rebound)
        if spec.get('adapter_source'):
            sources.add(Path(spec['adapter_source']))
    save(output/'checkpoint-verification.json', checkpoint_receipts)
    # Input-size ordering only selects a small development pilot; it is not a
    # claim of semantic difficulty. Independent confirmation is reserved.
    def size(row):
        sample = row['sample']
        if row['task_type'] == 'text':
            return len(sample['context']) + len(sample['question'])
        return Path(sample['input']).stat().st_size
    selected = {}
    for task in ('text', 'image', 'ocr', 'asr'):
        rows = sorted((r for r in pool if r['task_type'] == task), key=lambda r: (size(r), r['pool_id']))
        selected[task] = [rows[round(q*(len(rows)-1))] for q in (.25, .5, .75)]
    jobs = []
    # Same input and loaded roster in solo and co-run: changes actual activity.
    for task in selected:
        jobs.append(dict(id='solo-g2-'+task, requests=[['generalist_2#0', task, 1]]))
    for task in ('image', 'ocr'):
        jobs.append(dict(id='solo-s-'+task, requests=[[task+'#0', task, 1]]))
    for task in selected:
        jobs.append(dict(id='corun-g2-'+task, requests=[['generalist_2#0', task, 1],
            ['image#0', 'image', 1], ['ocr#0', 'ocr', 1]]))
    rates = dict.fromkeys(selected, .1)
    trace = make_trace(rates, arrival_seconds, seed, 'poisson')
    plan = dict(schema='gsdeploy.activation_gpu_pilot_plan.v1', split='development_pilot',
        hardware_policy='same_GPU_model; UUID_is_provenance_only', residents=aliases,
        gpu_budget_seconds=budget_seconds, arrival_seconds=arrival_seconds, seed=seed,
        selected_inputs={t:[dict(pool_id=r['pool_id'], size_proxy=size(r)) for r in rows]
                         for t, rows in selected.items()}, jobs=jobs, arrival_trace=trace,
        specialist_fractions=dict(image=.5, ocr=.5), input_size_is_semantic_difficulty=False,
        timing_scope='profiler_perturbed' if profiled else 'natural_host_phase_markers',
        phase_contract='continuous_native_call_spans' if complete_phases else 'legacy_partial_native_call_spans',
        utility=None, qualification='COLLECTOR_PILOT_NOT_PREDICTOR_TRAINING_OR_CONFIRMATION')
    save(output/'plan.json', plan)
    save(output/'worker-registry.json', registry)
    snapshot = []
    for index, src in enumerate(sorted(sources)):
        dest = output/'source-snapshot'/str(index)/src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        snapshot.append(dict(source=receipt(src), snapshot=receipt(dest)))
    save(output/'sources.json', snapshot)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total',
                                  '--format=csv,noheader,nounits'], text=True).strip()
    config = MeasurementConfig(deployment=dict.fromkeys(aliases, 1), workload_rps=rates,
        routing_rps={'text':{'generalist_2#0':.1}, 'asr':{'generalist_2#0':.1},
                     'image':{'generalist_2#0':.05, 'image#0':.05},
                     'ocr':{'generalist_2#0':.05, 'ocr#0':.05}},
        startup_timeout_seconds=budget_seconds-30, shutdown_timeout_seconds=15,
        measurement_seconds=arrival_seconds, routing_source='controlled_development_pilot')
    runtime = DeploymentRuntime(config, registry)
    raw = dict(schema='gsdeploy.activation_gpu_pilot.v1', kind='measured', backend='native',
        status='starting', gpu=gpu, plan=receipt(output/'plan.json'), U=None,
        clock='host_perf_counter_ns', performance_scope=plan['timing_scope'],
        stage_scope='native calls include Python/transfers/CUDA; no pure GPU stage claim',
        request_records=[], batches=[], errors=[])
    monitor = None
    lock = (ROOT/'artifacts/.gpu-experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def alarm(_signum, _frame):
        raise TimeoutError('Predeclared pilot wall time budget exhausted')
    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, budget_seconds-20)
    try:
        monitor = BoardMonitor(runtime)
        monitor.sample(); monitor.check(); monitor.start()
        raw['startup_begin_time_ns'] = time.perf_counter_ns()
        emit(event='startup', gpu=gpu, output=str(output))
        runtime.start()
        for alias, worker in runtime.workers.items():
            if not descendant_or_self(runtime.ready[alias]['pid'], worker.process.pid):
                raise RuntimeError('READY process is not owned: '+alias)
        raw['ready'] = copy.deepcopy(runtime.ready)
        raw['all_ready_time_ns'] = time.perf_counter_ns()
        emit(event='all_ready', elapsed_seconds=(raw['all_ready_time_ns']-raw['startup_begin_time_ns'])/1e9)

        def submit(alias, task, item, scheduled, group):
            r = record('pilot', len(raw['request_records']), task, alias, item, scheduled)
            r['profile_group'] = group
            raw['request_records'].append(r)
            runtime.enqueue(r, item['sample'])
            return r

        def drain(rows):
            runtime.drain(rows, 90)
            end = time.monotonic()+3
            while any('previous_handler_next_request_ready' not in r for r in rows) and time.monotonic()<end:
                time.sleep(.005)
            if any(r['status'] != 'completed' or 'previous_handler_next_request_ready' not in r for r in rows):
                raise RuntimeError('Request failure or missing READY: '+str([(r['request_id'],r['status'],r.get('error')) for r in rows]))
            monitor.check()

        # Priming is separate from paired measurements and uses a different input.
        for alias, tasks in [('generalist_2#0', list(selected)), ('image#0',['image']), ('ocr#0',['ocr'])]:
            for task in tasks:
                drain([submit(alias,task,selected[task][0],time.perf_counter_ns(),'priming')])
                emit(event='primed', alias=alias, task=task)
        for job in jobs:
            start = time.perf_counter_ns()
            rows = [submit(alias,task,selected[task][i],start,job['id']) for alias,task,i in job['requests']]
            drain(rows)
            raw['batches'].append(dict(id=job['id'], start_ns=start, end_ns=time.perf_counter_ns(),
                                      request_ids=[r['request_id'] for r in rows]))
            emit(event='job_complete', job=job['id'], handler_ms=[r['worker_latency_ms'] for r in rows])
        raw['measurement_start_time_ns'] = start = time.perf_counter_ns()
        counts = dict.fromkeys(selected,0)
        for offset, task in trace:
            target = start+round(offset*1e9)
            while time.perf_counter_ns() < target:
                monitor.check()
                time.sleep(min(.01, max(0, (target-time.perf_counter_ns())/1e9)))
            n = counts[task]; counts[task] += 1
            alias = task+'#0' if task in ('image','ocr') and n%2==0 else 'generalist_2#0'
            submit(alias,task,selected[task][n%3],target,'arrival_window')
        while time.perf_counter_ns() < start+round(arrival_seconds*1e9):
            monitor.check(); time.sleep(.01)
        raw['measurement_end_time_ns'] = time.perf_counter_ns()
        drain(raw['request_records'])
        raw['drain_end_time_ns'] = time.perf_counter_ns()
        monitor.sample(); monitor.check()
        raw['status'] = 'completed'
    except BaseException as exc:
        raw['status'] = 'failed'
        raw['errors'].append(type(exc).__name__+': '+str(exc))
        emit(event='failed', error=raw['errors'][-1])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        runtime.close()
        raw['shutdown'] = runtime.shutdown_report
        raw['runtime_errors'] = runtime.errors
        if monitor:
            monitor.close()
            raw['board_samples'] = monitor.samples
            raw['telemetry_errors'] = monitor.errors
        raw['end_time_ns'] = time.perf_counter_ns()
        save(output/'raw.json', raw)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    emit(status=raw['status'], requests=len(raw['request_records']), output=str(output/'raw.json'))
    return 0 if raw['status']=='completed' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--budget-seconds', type=float, default=300)
    p.add_argument('--profiled', action='store_true', help='Declare an external process-tree profiler')
    p.add_argument('--complete-phases', action='store_true')
    args = p.parse_args()
    if args.budget_seconds < 60:
        p.error('budget must allow startup and cleanup (>=60 seconds)')
    raise SystemExit(collect(args.output, budget_seconds=args.budget_seconds, profiled=args.profiled,
                            complete_phases=args.complete_phases))
