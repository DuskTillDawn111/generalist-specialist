"""Keep causal current-stream evidence beside the existing host observer.

This consumes parent-visible PHASE observations without calling them phase
boundaries or turning a stream event into an executable GPU progress cursor.
It accepts a complete, persistent runtime registry, not a sliding window.
"""
from __future__ import annotations

import copy

from gsdeploy.forecasting.observed_state import (
    advance_observed_state, feedback_from_runtime_records, new_observed_state,
)
from gsdeploy.resources import identity

SCHEMA = 'gsdeploy.visible_stream_event_ledger.v1'
PROBE = 'gsdeploy.stream_event_forward_observation.v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ns(value):
    return type(value) is int and value >= 0


def seal(value):
    value['state_sha256'] = identity({k: v for k, v in value.items() if k != 'state_sha256'})
    return value


def new_state(aliases, execution_context_sha256, *, frozen_cutoff_ns=None):
    host = new_observed_state(aliases, execution_context_sha256,
        mode='online' if frozen_cutoff_ns is None else 'frozen',
        frozen_cutoff_ns=frozen_cutoff_ns)
    return seal(dict(schema=SCHEMA, host=host, visible_device_events=[],
        observed_at_ns=None, kernel_cursor=None, main_lifecycle_resume_eligible=False))


def visible_feeds(records, context, cutoff_ns):
    """Do not examine any event body before its parent receipt becomes visible."""
    host_records, device = [], []
    seen = set()
    for row in records:
        rid, alias = row['request_id'], row['model']
        require(rid not in seen, 'Persistent registry requires unique request IDs')
        seen.add(rid)
        safe = dict(request_id=rid, model=alias, execution_events=[])
        for event in row.get('execution_events', []):
            received = event.get('received_time_ns')
            require(ns(received), 'Parent receipt required before reading event body')
            if received > cutoff_ns:
                continue
            require(event.get('request_id') == rid, 'Event/request identity mismatch')
            if event.get('event') == 'observation':
                require(event.get('kind') == 'device_event_query', 'Unsupported visible observation')
                device.append(dict(alias=alias, **copy.deepcopy(event)))
            else:
                safe['execution_events'].append(copy.deepcopy(event))
        received = row.get('response_reader_event_enqueue_time_ns')
        if received is not None:
            require(ns(received), 'RESULT parent receipt must be an integer')
            if received <= cutoff_ns:
                safe['response_reader_event_enqueue_time_ns'] = received
                safe['worker_response'] = {'end_perf_ns': row.get('worker_response', {}).get('end_perf_ns')}
        ready = row.get('previous_handler_next_request_ready')
        if ready is not None:
            received = ready.get('received_time_ns')
            require(ns(received), 'Readiness parent receipt must be an integer')
            if received <= cutoff_ns:
                safe['previous_handler_next_request_ready'] = copy.deepcopy(ready)
        host_records.append(safe)
    device.sort(key=lambda x: (x['received_time_ns'], x['request_id'], x.get('observation_sequence', 0)))
    return feedback_from_runtime_records(host_records, context, cutoff_ns), device


def validate_events(events, host):
    sequences, clocks, streams, completed = {}, {}, {}, set()
    for event in events:
        rid, alias = event['request_id'], event['alias']
        require(alias in host['aliases'], 'Unknown device observation owner')
        lo, hi, received = (event.get(k) for k in
            ('host_query_begin_ns', 'host_query_end_ns', 'received_time_ns'))
        require(all(ns(t) for t in (lo, hi, received)) and lo <= hi <= received,
            'Device query host/parent chronology invalid')
        require(event.get('perf_ns') == hi and event.get('schema') == PROBE,
            'Device query schema/clock identity mismatch')
        sequence = event.get('observation_sequence')
        require(type(sequence) is int and sequence == sequences.get(rid, 0) + 1,
            'Per-request device sequence gap/duplicate')
        require(lo >= clocks.get(rid, 0), 'Per-request device query clock moved backward')
        sequences[rid], clocks[rid] = sequence, hi
        phase = event.get('phase_id')
        require(isinstance(phase, str) and phase and event.get('segment_id') == phase,
            'Exact semantic phase identity required')
        observed = host['requests'].get(rid, {})
        boundary = observed.get('phases', {}).get(phase, {})
        require(observed.get('alias') == alias and boundary.get('kind') == 'gpu'
            and ns(boundary.get('end_ns')) and boundary['end_ns'] <= lo,
            'Device query lacks a visible matching host forward boundary')
        event_id = event.get('event_id')
        require(event_id == rid + ':' + phase + ':stream_end', 'Exact stream event identity required')
        require(all(ns(event.get(k)) for k in ('stream_id', 'device_index', 'forward_index')),
            'Explicit stream/device/forward identities required')
        key = rid, event_id
        binding = tuple(event[k] for k in ('alias', 'phase_id', 'stream_id', 'device_index', 'forward_index'))
        require(key not in streams or streams[key] == binding, 'Stream event identity changed')
        streams[key] = binding
        require(key not in completed, 'Completed event cannot be queried again')
        require(event.get('evidence_scope') == 'recorded_current_stream_only'
            and event.get('all_request_GPU_work_complete') is False
            and all(k in event and event[k] is None for k in
                ('in_flight_remaining_fraction', 'completion_time_ns', 'physical_SM_service_ns')),
            'Stream evidence cannot supply global/fractional/SM progress')
        done = event.get('end_event_complete')
        require(type(done) is bool, 'Device completion query must be boolean')
        elapsed = event.get('recorded_stream_elapsed_ms')
        if done:
            import math
            require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
                'Completed event interval must be finite and nonnegative')
            completed.add(key)
        else:
            require(elapsed is None, 'Incomplete event has no measured elapsed interval')


def advance(previous, records, cutoff_ns):
    require(previous.get('schema') == SCHEMA and previous.get('state_sha256') ==
        identity({k: v for k, v in previous.items() if k != 'state_sha256'}), 'Mutated ledger checkpoint')
    require(ns(cutoff_ns) and (previous['observed_at_ns'] is None or
        cutoff_ns >= previous['observed_at_ns']), 'Backward observation cutoff')
    state = copy.deepcopy(previous)
    host = state['host']
    feedback, events = visible_feeds(records, host['execution_context_sha256'], cutoff_ns)
    require(len(feedback) >= host['last_feedback_sequence'], 'Persistent host registry lost events')
    # Comparing already consumed facts also catches corrected/tampered history;
    # old host state alone does not retain the original feedback feed.
    prior_feedback = state.get('host_feedback_prefix', [])
    require(feedback[:len(prior_feedback)] == prior_feedback, 'Visible host history changed')
    prior_device = state['visible_device_events']
    require(events[:len(prior_device)] == prior_device, 'Visible device history changed or lost')
    host = advance_observed_state(host, cutoff_ns,
        feedback=feedback[host['last_feedback_sequence']:])
    validate_events(events, host)
    state.update(host=host, host_feedback_prefix=feedback, visible_device_events=events,
        observed_at_ns=cutoff_ns, kernel_cursor=None, main_lifecycle_resume_eligible=False)
    return seal(state)
