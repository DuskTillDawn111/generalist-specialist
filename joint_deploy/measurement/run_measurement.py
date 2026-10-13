"""Execute a supplied deployment/workload/routing. No search, selection or prediction."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .deployment_runtime import DeploymentRuntime
from .load_generator import SamplePool, periodic_arrivals, phases_for
from .metrics import annotate_records, summarize
from .router import Router
from .schema import MeasurementConfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / 'benchmark/profiles/model_service_profile_6task.json'
DEFAULT_POOL = ROOT / 'benchmark/task_pools/mixed_formal_1200_6task.jsonl'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_info(profile, pool, registry):
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True)
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                             text=True, capture_output=True, timeout=10)
        gpu_info = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu_info = str(exc)
    return {
        'git_commit': commit.stdout.strip() if commit.returncode == 0 else None,
        'git_commit_note': None if commit.returncode == 0 else 'workspace is not a Git checkout',
        'model_profile': str(profile), 'model_profile_hash': sha256(profile),
        'task_pool': str(pool), 'task_pool_hash': sha256(pool), 'gpu_snapshot': gpu_info,
        'worker_protocol_hash': sha256(ROOT / 'joint_deploy/worker_protocol.py'),
        'harness_hashes': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'workers': {m: {'python': str(c['python']), 'script': str(c['script']),
                        'script_hash': sha256(c['script'])} for m, c in registry.items()},
        'environment': {key: os.environ.get(key) for key in
                        ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
    }


def new_record(phase, index, task, model, item, scheduled):
    return {'request_id': f'{phase}:{index}', 'phase': phase, 'arrival_index': index,
            'task': task, 'model': model, 'pool_id': item['pool_id'],
            'sample_id': item['sample'].get('id'), 'scheduled_time_ns': scheduled,
            'actual_submit_time_ns': None, 'completion_time_ns': None,
            'worker_latency_ms': None, 'worker_queue_wait_ms': None, 'status': 'planned'}


def wait_until(deadline_ns):
    while True:
        remaining = (deadline_ns - time.perf_counter_ns()) / 1e9
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.1))


def execute_window(runtime, config, pool_path, phase, duration, records):
    # Reset routing and sample order for measurement so warmup length cannot bias its sequence.
    pool = SamplePool(pool_path, [k for k, v in config.workload_rps.items() if v > 0], config.random_seed)
    router = Router(config.routing_rps)
    start_ns, unix_ns = time.perf_counter_ns(), time.time_ns()
    end_ns = start_ns + round(duration * 1e9)
    window = {'start_time_ns': start_ns, 'start_unix_ns': unix_ns, 'end_time_ns': end_ns,
              'duration_seconds': duration, 'cohort': 'requests scheduled in [start, end)',
              'completion_count_definition': 'successful cohort completions up to the drain deadline'}
    for index, (offset, task) in enumerate(periodic_arrivals(config.workload_rps, duration, config.random_seed)):
        item = pool.next(task)
        record = new_record(phase, index, task, router.choose(task), item, start_ns + round(offset * 1e9))
        records.append(record)
        wait_until(record['scheduled_time_ns'])
        runtime.enqueue(record, item['sample'])
    wait_until(end_ns)
    window['arrival_loop_end_time_ns'] = time.perf_counter_ns()
    runtime.drain(records, config.drain_timeout_seconds)
    window['drain_end_time_ns'] = time.perf_counter_ns()
    window['drain_seconds'] = (window['drain_end_time_ns'] - end_ns) / 1e9
    return window


def run_measurement(config, output, pool_path=DEFAULT_POOL, profile_path=DEFAULT_PROFILE, registry=None):
    output, pool_path, profile_path = Path(output), Path(pool_path).resolve(), Path(profile_path).resolve()
    metadata = json.loads(profile_path.read_text())
    config.validate({m: d['supported_tasks'] for m, d in metadata['models'].items()})
    # Validate sample availability before launching processes.
    tasks = [k for k, v in config.workload_rps.items() if v > 0]
    samples = SamplePool(pool_path, tasks, config.random_seed)
    runtime = DeploymentRuntime(config, registry)
    source = source_info(profile_path, pool_path, {m: c for m, c in runtime.registry.items()
                                                if config.deployment.get(m, 0) > 0})
    result = {'schema_version': 1, **config.as_dict(), 'source': source,
              'arrival_process': {'name': config.arrival_process, 'random_seed': config.random_seed,
                                  'initial_phase_seconds': phases_for(config.workload_rps, config.random_seed)},
              'clock': {'request_times': 'host perf_counter_ns; same clock for scheduled/submit/completion',
                        'submit_definition': 'immediately before write to worker stdin; write end also saved',
                        'worker_times': 'worker time.time_ns; queue_wait is wall-clock based, not pure GPU time'},
              'warmup': {}, 'measurement_window': None, 'models': {}, 'per_task_metrics': {},
              'system_metrics': None, 'request_records': [], 'warmup_request_records': [],
              'priming_request_records': [], 'status': 'starting', 'errors': []}
    try:
        print('Starting workers...', flush=True)
        runtime.start()
        print('All configured workers READY.', flush=True)
        if config.warmup_seconds > 0:
            # Execute each positive service path at least once, even at tiny routing rates.
            priming = result['priming_request_records']
            for task, route in sorted(config.routing_rps.items()):
                for model, rate in sorted(route.items()):
                    if rate <= 0:
                        continue
                    item = samples.next(task)
                    record = new_record('priming', len(priming), task, model, item, time.perf_counter_ns())
                    priming.append(record)
                    runtime.enqueue(record, item['sample'])
            runtime.drain(priming, config.drain_timeout_seconds)
            if any(r['status'] != 'completed' for r in priming):
                raise RuntimeError('Service-path priming failed or timed out')
            print('Priming complete; starting periodic warmup.', flush=True)
            result['warmup'] = execute_window(runtime, config, pool_path, 'warmup', config.warmup_seconds,
                                              result['warmup_request_records'])
            if any(r['status'] != 'completed' for r in result['warmup_request_records']):
                raise RuntimeError('Warmup failed or timed out; measurement was not started')
        print('Starting measurement window.', flush=True)
        result['measurement_window'] = execute_window(runtime, config, pool_path, 'measurement',
                                                       config.measurement_seconds, result['request_records'])
        result['status'] = ('completed' if all(r['status'] == 'completed' for r in result['request_records'])
                            else 'completed_with_failures')
    except (Exception, KeyboardInterrupt) as exc:
        result['status'] = 'failed'
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        try:
            result['shutdown'] = runtime.close()
        except Exception as exc:
            result['errors'].append(f'shutdown error: {exc}')
            result['status'] = 'failed'
        result['errors'].extend(runtime.errors)
        if runtime.errors or any(s['exit_code'] != 0 or s['forced'] or not s['bye_received']
                                 for s in result.get('shutdown', {}).values()):
            result['status'] = 'failed'
        for key in ('request_records', 'warmup_request_records', 'priming_request_records'):
            for record in result[key]:
                if record['status'] in ('planned', 'queued', 'submitted'):
                    record.update(status='incomplete', error='run terminated before a response')
            annotate_records(result[key])
        if result['measurement_window']:
            per_task, system, models = summarize(result['request_records'], config, result['measurement_window'])
            result.update(per_task_metrics=per_task, system_metrics=system, models=models)
        for model in runtime.workers:
            result['models'].setdefault(model, {})['ready_payload'] = runtime.ready.get(model)
        result['warmup']['status_counts'] = dict(Counter(r['status'] for r in result['warmup_request_records']))
        result['warmup']['configured_seconds'] = config.warmup_seconds
        result['warmup']['priming_excluded_from_measurement'] = True
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pool', type=Path, default=DEFAULT_POOL)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args(argv)
    try:
        config = MeasurementConfig(**json.loads(args.config.read_text()))
        result = run_measurement(config, args.output, args.pool, args.profile)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({'status': result['status'], 'system_metrics': result['system_metrics'],
                      'errors': result['errors'], 'output': str(args.output.resolve())}, indent=2))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
