"""Causal completion feedback for the shared-GPU dispatcher.

The runtime still owns the unrevealed arrival/input cohort. A dispatcher receives
only current queue state and an allowlist of completions already collected.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import time

from tools.spillover_capacity_20261022.exposure import frozen_api
from tools.queue_refined_20261023.runtime_window import queue_snapshot


def feedback(row):
    """No answers, references, tokens, or future worker events cross this API."""
    return dict(request_id=row['request_id'], model=row['model'], task=row['task'],
                pool_id=row['pool_id'], enqueued_ns=row['enqueued_time_ns'],
                actual_submit_ns=row.get('actual_submit_time_ns'),
                completion_time_ns=row['completion_time_ns'],
                worker_service_ms=row.get('worker_response', {}).get('service_ms'))


def choose_and_enqueue(runtime, dispatcher, row, sample):
    with runtime.lock:
        now = time.perf_counter_ns()
        snapshot = queue_snapshot(runtime, dispatcher.aliases, now, row['phase'])
        seen = getattr(dispatcher, '_host_completed_ids', set())
        observed = []
        for previous in runtime.pending.values():
            if (previous.get('phase') == row['phase'] and previous.get('status') == 'completed'
                    and previous['request_id'] not in seen):
                event = feedback(previous)
                if event['completion_time_ns'] > now:
                    raise ValueError('Future completion feedback')
                observed.append(event)
                seen.add(previous['request_id'])
        dispatcher._host_completed_ids = seen
        dispatcher.observe_completed(observed, now)
        dispatcher.set_arrival(row['request_id'], row['scheduled_time_ns'])
        decision = dispatcher.choose(row['task'], row['pool_id'], now, snapshot)
        row['routing_decision'] = dict(decision_time_ns=now, snapshot=snapshot,
                                     decision=decision, newly_observed_completions=observed,
                                     decision_compute_ns=time.perf_counter_ns() - now)
        if decision.get('alias') is None:
            raise RuntimeError('NO_EXECUTABLE_COUPLED_POLICY_CANDIDATE')
        row['model'] = decision['alias']
        runtime.enqueue(row, sample)
    return decision


def window(runtime, config, pool_path, phase, duration, records, monitor, dispatcher, metadata=None):
    if records:
        raise ValueError('Fresh phase-local records required')
    api = frozen_api()
    runner = importlib.import_module(api['trace'].__package__ + '.runtime.runner')
    pool = api['workload'].SamplePool(pool_path, list(config.workload_rps), config.random_seed)
    trace = api['trace'].make_trace(config.workload_rps, duration, config.random_seed, config.arrival_process)
    payloads = []
    for index, (offset, task) in enumerate(trace):
        item = pool.next(task)
        row = runner.record(phase, index, task, None, item, 0)
        row['scheduled_offset_seconds'] = offset
        records.append(row)
        payloads.append(item['sample'])
    start = time.perf_counter_ns()
    end = start + round(duration * 1e9)
    metadata = {} if metadata is None else metadata
    metadata.update(start_time_ns=start, start_unix_ns=time.time_ns(), end_time_ns=end,
                    duration_seconds=duration, offered_requests=len(trace),
                    trace_sha256=api['trace'].trace_digest(trace),
                    routing_semantics='causal_shared_gpu_feedback_v1',
                    static_SWRR_certificate_applicable=False)
    dispatcher.start_phase(start, end)
    for row in records:
        row['scheduled_time_ns'] = start + round(row['scheduled_offset_seconds'] * 1e9)
    for row, sample in zip(records, payloads):
        runner.wait_until(row['scheduled_time_ns'], monitor)
        choose_and_enqueue(runtime, dispatcher, row, sample)
    runner.wait_until(end, monitor)
    metadata['arrival_loop_end_time_ns'] = time.perf_counter_ns()
    runtime.drain(records, config.drain_timeout_seconds)
    metadata['drain_end_time_ns'] = time.perf_counter_ns()
    metadata['drain_seconds'] = (metadata['drain_end_time_ns'] - end) / 1e9
    metadata['dispatch_decisions_sha256'] = hashlib.sha256(json.dumps([
        dict(request_id=r['request_id'], model=r['model'], routing_decision=r['routing_decision'])
        for r in records], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return metadata
