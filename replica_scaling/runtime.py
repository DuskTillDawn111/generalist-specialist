"""Scoped replica experiment atop unchanged single-instance worker transport.

Each transport ID has count=1; family count lives explicitly in experiment schema.
Two IDs launch two separate frozen model scripts, never two threads in one model.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
import threading
import time
from types import SimpleNamespace
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from joint_deploy.measurement.load_generator import SamplePool, periodic_arrivals
from joint_deploy.measurement.metrics import annotate_records, summarize, percentile
from joint_deploy.measurement.router import Router
from joint_deploy.measurement.run_measurement import new_record
from joint_deploy.measurement.schema import MeasurementConfig
from performance_model.local_calibration.stability_v2 import reconstruct, classify
from .common import OUT, frozen_plan, write, sha
from .telemetry import Monitor, snapshot, release


def replica_config(model, count, rate, seed, plan):
    if type(count) is not int or count not in (1, 2):
        raise ValueError('Only 1-vs-2 independent replicas are authorized')
    ids = [f'{model["model_id"]}#{i}' for i in range(count)]
    task = model['task']
    config = MeasurementConfig(deployment={i: 1 for i in ids}, workload_rps={task: rate},
        routing_rps={task: {i: rate / count for i in ids}}, random_seed=seed,
        warmup_seconds=plan['warmup']['seconds'], measurement_seconds=plan['measurement_seconds'],
        drain_timeout_seconds=plan['drain_timeout_seconds'], startup_timeout_seconds=plan['startup_timeout_seconds'],
        shutdown_timeout_seconds=plan['shutdown_timeout_seconds'], max_pending_per_model=plan['max_pending_per_replica'],
        routing_source='frozen_7C1_equal_round_robin')
    config.validate({i: [task] for i in ids})
    registry = {i: dict(model['launcher']) for i in ids}
    return config, registry


def backlog_guard(records, now, s0_ms, thresholds):
    outstanding = [r for r in records if r['status'] in ('queued', 'submitted')]
    n = len(outstanding)
    oldest = max(((now - r['scheduled_time_ns']) / 1e9 for r in outstanding), default=0.)
    debt = n * s0_ms / 1000
    reasons = []
    if n >= thresholds['max_outstanding_requests']:
        reasons.append('max_outstanding_requests')
    if debt >= thresholds['max_outstanding_baseline_service_seconds']:
        reasons.append('max_outstanding_baseline_service_seconds')
    if oldest >= thresholds['max_oldest_outstanding_age_seconds']:
        reasons.append('max_oldest_outstanding_age_seconds')
    return dict(triggered=bool(reasons), reasons=reasons, outstanding=n,
                oldest_seconds=oldest, baseline_service_seconds=debt, time_ns=now)


def execute(runtime, config, pool_path, phase, duration, records, s0_ms, guard, monitor=None):
    pool = SamplePool(pool_path, list(config.workload_rps), config.random_seed)
    router = Router(config.routing_rps)
    start, unix = time.perf_counter_ns(), time.time_ns()
    planned_end = start + round(duration * 1e9)
    next_check = start
    stop = None
    def wait(deadline):
        nonlocal next_check, stop
        while True:
            now = time.perf_counter_ns()
            if guard is not None and now >= next_check:
                with runtime.lock:
                    evidence = backlog_guard(records, now, s0_ms, guard)
                next_check = now + 100_000_000
                if evidence['triggered']:
                    stop = dict(kind='protective_backlog_stop', **evidence)
                    return False
                if monitor and any(s['foreign_pids'] for s in monitor.samples[-2:]):
                    stop = dict(kind='external_gpu_process_interference', time_ns=now,
                                evidence=monitor.samples[-2:])
                    return False
                if any(w.process.poll() is not None for w in runtime.workers.values()):
                    stop = dict(kind='worker_exit_during_measurement', time_ns=now)
                    return False
            if now >= deadline:
                return True
            time.sleep(min((deadline - now) / 1e9, .05))
    for index, (offset, task) in enumerate(periodic_arrivals(config.workload_rps, duration, config.random_seed)):
        scheduled = start + round(offset * 1e9)
        if not wait(scheduled):
            break
        item = pool.next(task)
        record = new_record(phase, index, task, router.choose(task), item, scheduled)
        records.append(record)
        runtime.enqueue(record, item['sample'])
    if stop is None:
        wait(planned_end)
    end = min(planned_end, stop['time_ns']) if stop else planned_end
    arrival_end = time.perf_counter_ns()
    if monitor:
        monitor.phase = 'drain' if phase == 'measurement' else 'warmup_drain'
    runtime.drain(records, config.drain_timeout_seconds)
    drain_end = time.perf_counter_ns()
    return dict(start_time_ns=start, start_unix_ns=unix, end_time_ns=end,
        planned_end_time_ns=planned_end, duration_seconds=(end - start) / 1e9,
        planned_duration_seconds=duration, arrival_loop_end_time_ns=arrival_end,
        drain_end_time_ns=drain_end, drain_seconds=(drain_end - end) / 1e9,
        stop=stop, cohort='Requests scheduled in the actually observed [start,end) window; stopped windows are not full-budget repetitions.')


def scalar_metrics(result, config):
    rows = result['request_records']
    window = result['measurement_window']
    if not window:
        return None
    completed = [r for r in rows if r['status'] == 'completed']
    start, end = window['start_time_ns'], window['end_time_ns']
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    extract = lambda key: [r[key] for r in completed if r.get(key) is not None]
    system = result['system_metrics']
    per_replica = {}
    for worker_id in config.deployment:
        offered = [r for r in rows if r['model'] == worker_id]
        done = [r for r in offered if r['status'] == 'completed']
        within = [r for r in done if r['completion_time_ns'] <= end]
        per_replica[worker_id] = dict(offered_requests=len(offered), completed_requests=len(done),
            within_window_completions=len(within), completed_throughput_rps=len(within) / window['duration_seconds'],
            cohort_completion_throughput_rps=len(done) / window['duration_seconds'],
            completion_ratio=len(done) / len(offered) if offered else None,
            pid=result['worker_identities'][worker_id]['pid'])
    return dict(offered_load_rps=sum(config.workload_rps.values()), actual_offered_load_rps=len(rows) / window['duration_seconds'],
        completed_throughput_rps=sum(r['completion_time_ns'] <= end for r in completed) / window['duration_seconds'],
        final_half_window_throughput_rps=sum(r['completion_time_ns'] > (start + end) / 2 and r['completion_time_ns'] <= end for r in completed) / (window['duration_seconds'] / 2),
        completion_ratio=system['completion_ratio'], completed_requests=len(completed), offered_requests=len(rows),
        within_window_completion_ratio=sum(r['completion_time_ns'] <= end for r in completed) / len(rows) if rows else None,
        cohort_throughput_including_drain_rps=len(completed) / ((window['drain_end_time_ns'] - start) / 1e9),
        mean_e2e_ms=mean(extract('e2e_latency_ms')), p95_e2e_ms=percentile(extract('e2e_latency_ms'), 95),
        mean_queue_wait_ms=mean(extract('worker_queue_wait_ms')), p95_queue_wait_ms=percentile(extract('worker_queue_wait_ms'), 95),
        mean_worker_latency_ms=mean(extract('worker_latency_ms')), p95_worker_latency_ms=percentile(extract('worker_latency_ms'), 95),
        mean_schedule_delay_ms=mean(extract('schedule_delay_ms')), per_replica=per_replica)


def run(plan, slot, output):
    model = next(m for m in plan['models'] if m['model_id'] == slot['model_id'])
    config, registry = replica_config(model, slot['replica_count'], slot['offered_load_rps'], slot['random_seed'], plan)
    baseline = snapshot()
    if baseline['processes'] or baseline['used_mib'] > 128 or baseline['uuid'] != plan['gpu']['uuid']:
        raise RuntimeError('Standalone isolation gate failed before launch')
    runtime = DeploymentRuntime(config, registry)
    monitor = Monitor(runtime, plan['telemetry_interval_seconds'])
    result = dict(schema_version=1, **config.as_dict(), slot=slot,
        family_deployment={model['model_id']: slot['replica_count']}, replica_to_family={i: model['model_id'] for i in config.deployment},
        frozen_plan_sha256=sha(OUT / 'replica_scaling_plan.json'),
        provenance=dict(model_manifest_sha256=model['model_manifest_sha256'], source_revision=model['source_revision'],
            adapter_sha256=model['adapter']['sha256'], environment_hash=model['environment_hash'],
            executable_hash=model['executable_hash'], precision=model['precision'], pool_hash=plan['input_pool']['sha256']),
        baseline_gpu=baseline, clock='Host perf_counter_ns for arrival/submit/response; frozen worker wall-clock timers for worker/queue; no cross-clock subtraction.',
        status='starting', execution_status='not_started', errors=[], request_records=[], priming_request_records=[], warmup_request_records=[],
        worker_identities={}, warmup={}, measurement_window=None, models={}, system_metrics=None)
    monitor.start()
    try:
        runtime.start()
        result['worker_identities'] = {worker_id: dict(worker_id=worker_id, family=model['model_id'],
            pid=worker.process.pid, ready_payload=runtime.ready[worker_id], cuda_context_identity='NVML compute-process PID; no CUDA context API exposed by frozen worker',
            model_manifest_sha256=model['model_manifest_sha256'], source_revision=model['source_revision'],
            environment_hash=model['environment_hash'], adapter_sha256=model['adapter']['sha256']) for worker_id, worker in runtime.workers.items()}
        if len({w['pid'] for w in result['worker_identities'].values()}) != slot['replica_count']:
            raise RuntimeError('Independent worker PID gate failed')
        result['idle_after_load'] = monitor.idle('idle_after_load')
        priming = result['priming_request_records']
        pool = SamplePool(plan['input_pool']['path'], [model['task']], slot['random_seed'])
        for worker_id in config.deployment:
            item = pool.next(model['task'])
            record = new_record('priming', len(priming), model['task'], worker_id, item, time.perf_counter_ns())
            priming.append(record)
            runtime.enqueue(record, item['sample'])
        runtime.drain(priming, config.drain_timeout_seconds)
        if any(r['status'] != 'completed' for r in priming):
            raise RuntimeError('Replica priming failed')
        warm_rate = model['loads']['low']
        warm_config = replace(config, workload_rps={model['task']: warm_rate}, routing_rps={model['task']: {i: warm_rate / slot['replica_count'] for i in config.deployment}})
        monitor.phase = 'warmup'
        result['warmup'] = execute(runtime, warm_config, plan['input_pool']['path'], 'warmup', config.warmup_seconds,
                                   result['warmup_request_records'], model['service_demand_ms'], None, monitor)
        if any(r['status'] != 'completed' for r in result['warmup_request_records']):
            raise RuntimeError('Warmup failed')
        result['idle_before_workload'] = monitor.idle('idle_before_workload')
        monitor.phase = 'measurement'
        result['measurement_window'] = execute(runtime, config, plan['input_pool']['path'], 'measurement',
            config.measurement_seconds, result['request_records'], model['service_demand_ms'], plan['protective_stop'], monitor)
        stop = result['measurement_window']['stop']
        result['execution_status'] = stop['kind'] if stop else 'completed'
        result['status'] = 'completed' if all(r['status'] == 'completed' for r in result['request_records']) else 'completed_with_failures'
        if stop:
            result['status'] = 'failed'
            result['errors'].append(stop['kind'])
    except KeyboardInterrupt:
        result['status'] = 'failed'
        result['execution_status'] = 'external_interruption'
        result['errors'].append('Observed KeyboardInterrupt in experiment harness')
    except Exception as exc:
        result['status'] = 'failed'
        result['execution_status'] = 'worker_or_harness_failure'
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        monitor.phase = 'shutdown'
        try:
            result['shutdown'] = runtime.close()
        except Exception as exc:
            result['errors'].append(f'shutdown: {exc}')
        monitor.close()
        result['errors'].extend(runtime.errors)
        result['telemetry'] = dict(interval_seconds=plan['telemetry_interval_seconds'], samples=monitor.samples, errors=monitor.errors,
                                  peak_note='Sampled NVML high-water mark, not an exact allocation peak; memory utilization is bandwidth activity, not VRAM occupancy.')
        if runtime.errors or any(v['exit_code'] != 0 or v['forced'] or not v['bye_received'] for v in result.get('shutdown', {}).values()):
            result['status'] = 'failed'
        for key in ('request_records', 'priming_request_records', 'warmup_request_records'):
            for row in result[key]:
                if row['status'] in ('planned', 'queued', 'submitted'):
                    row.update(status='incomplete', error='run terminated before response')
            annotate_records(result[key])
        if result['measurement_window']:
            actual_config = replace(config, measurement_seconds=result['measurement_window']['duration_seconds'])
            per_task, system, models = summarize(result['request_records'], actual_config, result['measurement_window'])
            result.update(per_task_metrics=per_task, system_metrics=system, models=models)
            result['metrics'] = scalar_metrics(result, actual_config)
            aliases = {i: SimpleNamespace(service_time_ms={model['task']: model['service_demand_ms']}) for i in config.deployment}
            evidence = reconstruct(result, aliases, plan['stability_thresholds'])
            result['stability_evidence'] = evidence
            result['stability'] = classify(evidence, plan['stability_thresholds'])
        result['gpu_release'] = release(baseline, [w.process.pid for w in runtime.workers.values() if w.process])
        result['valid_full_repetition'] = bool(result['execution_status'] == 'completed' and result['measurement_window'] and
            len(result['worker_identities']) == slot['replica_count'] and result['gpu_release']['verified'] and
            not monitor.errors and not any(s['foreign_pids'] for s in monitor.samples))
        result['warmup_status_counts'] = dict(Counter(r['status'] for r in result['warmup_request_records']))
        write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    plan = frozen_plan()
    slot = next(s for s in plan['schedule'] if s['run_id'] == args.run_id)
    output = OUT / 'raw_runs' / (slot['run_id'] + '.json')
    if output.exists():
        raise RuntimeError('Refuse to overwrite raw run')
    result = run(plan, slot, output)
    print(json.dumps(dict(output=str(output), run_id=slot['run_id'], execution=result['execution_status'],
        stability=result.get('stability', {}).get('stability_state_v2'), metrics=result.get('metrics'),
        valid=result['valid_full_repetition'], release=result['gpu_release']['verified'])), flush=True)
    return 0 if result['gpu_release']['verified'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
