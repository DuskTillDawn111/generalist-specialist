"""Collector-processed native prefix and guarded v5_2 forecast boundary.

This adapter does not manufacture an engine checkpoint from a native snapshot.
Busy handlers retain a set of possible current states. Only an independently
checkable settled boundary can enter the existing engine's native seed API.
"""
from __future__ import annotations

import copy

from gsdeploy.forecast_errors import UnknownEvidence
from . import twoFIFO_settled_history_initial_v1 as settled

SCHEMA = 'gsdeploy.twoFIFO_native_busy_state.v1'
MISSING = (
    'ACTIVE_PROFILE_NODE_AND_REMAINING_WORK_SUPPORT',
    'ATOMIC_SENDER_QUEUE_AND_WORKER_START_ACKNOWLEDGEMENT',
    'READER_COLLECTOR_INFLIGHT_MESSAGE_WATERMARK',
    'CROSS_STREAM_DEPENDENCIES_AND_DYNAMIC_EOS_SUPPORT',
)
QUERY_SCHEMA = 'gsdeploy.stream_event_forward_observation.v1'
BOUNDARIES = frozenset(('START', 'RESULT', 'WORKER_READY'))


def need(ok, reason):
    if not ok:
        raise UnknownEvidence('UNKNOWN_NATIVE_BUSY_STATE:' + reason)


def ns(value):
    return type(value) is int and value >= 0


def _seal(value):
    value['state_sha256'] = settled.digest(value)
    return value


def _processed(event, cut, *, occurrence='perf_ns'):
    # Deliberately read only this visibility scalar before touching the body.
    at = event.get('collector_processing_time_ns')
    need(ns(at), 'missing_collector_processing_clock')
    if at > cut:
        return None
    clocks = [event.get(k) for k in (occurrence, 'received_time_ns',
        'collector_dequeued_time_ns', 'collector_processing_time_ns')]
    need(all(ns(t) for t in clocks) and clocks == sorted(clocks), 'station_chronology')
    return copy.deepcopy(event)


def from_snapshot(snapshot, *, public_cohort, cutoff_ns, owner_producers,
                  execution_context_sha256, forecast_origin_ns=None, previous=None):
    """Consume collector snapshot schema, retaining the entire public calendar.

    ``public_cohort`` uses the reference engine request schema. Native records
    may additionally contain prior history. Runtime ``pending_request_ids`` is
    a persistent record registry, never a count of unfinished GPU requests.
    Reader-queued envelopes remain opaque until collector processing. The
    collector may also hold a dequeued message outside its queue: absence of a
    queued envelope is therefore never evidence that protocol stations are idle.
    """
    cut = cutoff_ns
    origin = cut if forecast_origin_ns is None else forecast_origin_ns
    need(ns(cut) and ns(origin) and cut <= origin and
         ns(snapshot.get('snapshot_time_ns')) and cut <= snapshot['snapshot_time_ns'], 'cutoff')
    need(type(owner_producers) is dict and owner_producers and
         len(set(owner_producers.values())) == len(owner_producers), 'owner_producers')
    need(type(execution_context_sha256) is str and len(execution_context_sha256) == 64, 'context')
    need(snapshot.get('empty_initial_state_assumed') is False, 'unproved_initial_state')
    roster = snapshot.get('ready', {})
    need(set(roster) == set(owner_producers) and all(type(r.get('pid')) is int and r['pid'] > 0
         for r in roster.values()), 'worker_lifetimes')
    public = copy.deepcopy(public_cohort)
    fields = {'request_id', 'owner', 'task', 'pool_id', 'arrival_ns'}
    need(type(public) is list and public and all(type(r) is dict and set(r) == fields and
         r['owner'] in roster and ns(r['arrival_ns']) and all(type(r[k]) is str and r[k]
         for k in fields - {'arrival_ns'}) for r in public), 'complete_public_calendar')
    need(len({r['request_id'] for r in public}) == len(public), 'duplicate_public_id')
    rows = snapshot.get('request_records')
    need(type(rows) is list, 'record_registry')
    ids = [r.get('request_id') for r in rows]
    need(all(type(r) is str and r for r in ids) and len(set(ids)) == len(ids), 'duplicate_native_id')
    need(set(snapshot.get('pending_request_ids', [])) == set(ids), 'persistent_registry')
    stations, sender, requests = [], {}, {}
    all_record_events_processed = True
    for row in rows:
        rid, owner = row['request_id'], row['model']
        sending = {}
        for key in ('scheduled_time_ns', 'enqueued_time_ns', 'actual_submit_time_ns',
                    'submit_write_completed_time_ns'):
            t = row.get(key)
            if t is not None:
                need(ns(t), 'sender_clock')
                if t <= cut:
                    sending[key] = t
        # A later persistent registry contains rows which did not exist at the
        # historical cutoff. The public schedule alone cannot reveal when a
        # row entered runtime.pending. Admit it only after an actual parent
        # sender action or a collector-processed message is known by cutoff.
        event_processing = [e.get('collector_processing_time_ns') for e in row.get('execution_events', [])]
        result_processing = row.get('response_collector_processing_time_ns')
        ready_record = row.get('previous_handler_next_request_ready')
        ready_processing = None if ready_record is None else ready_record.get('collector_processing_time_ns')
        row_known = any(k != 'scheduled_time_ns' for k in sending) or any(
            ns(t) and t <= cut for t in [*event_processing, result_processing, ready_processing])
        if not row_known:
            all_record_events_processed = False
            continue
        need(owner in roster, 'foreign_owner')
        known = list(sending.values())
        need(known == sorted(known), 'sender_chronology')
        sender[rid] = sending
        visible = []
        for event in row.get('execution_events', []):
            safe = _processed(event, cut)
            if safe is None:
                all_record_events_processed = False
                continue
            need(safe.get('request_id') == rid, 'event_request_identity')
            # Native semantic PHASE and device-query emitters do not carry a
            # PID. Their owner is the runtime row's stdout transport binding.
            # An explicit PID, if present, must still match that lifetime.
            need('pid' not in safe or safe['pid'] == roster[owner]['pid'], 'event_worker_identity')
            if safe.get('kind') == 'device_event_query':
                need(safe.get('schema') == QUERY_SCHEMA and safe.get('event') in
                     (None, 'observation'), 'query_schema')
                kind = 'QUERY'
            elif safe.get('event') in ('start', 'end'):
                need(type(safe.get('kind')) is str and safe['kind'] and
                     type(safe.get('segment_id')) is str and safe['segment_id'], 'phase_identity')
                kind = 'PHASE'
            else:
                # START is its own native payload, not any unclassified event
                # which happens to omit an `event` member.
                need(safe.get('pid') == roster[owner]['pid'] and 'event' not in safe and
                     'kind' not in safe and 'segment_id' not in safe and 'schema' not in safe,
                     'unrecognized_event_not_a_START')
                kind = 'START'
            visible.append((kind, safe))
        at = row.get('response_collector_processing_time_ns')
        if at is not None:
            need(ns(at), 'result_processing_clock')
            if at <= cut:
                # Only processed RESULT permits access to its response body.
                response = row.get('worker_response', {})
                safe = _processed(dict(perf_ns=response.get('end_perf_ns'),
                    received_time_ns=row.get('response_reader_event_enqueue_time_ns'),
                    collector_dequeued_time_ns=row.get('response_collector_dequeued_time_ns'),
                    collector_processing_time_ns=at), cut)
                safe['success'] = response.get('success')
                visible.append(('RESULT', safe))
        ready = row.get('previous_handler_next_request_ready')
        if ready is not None:
            safe = _processed(ready, cut)
            if safe is not None:
                need(safe.get('request_id') == rid and safe.get('pid') == roster[owner]['pid'], 'ready_identity')
                visible.append(('WORKER_READY', safe))
        facts = {kind: e for kind, e in visible if kind in BOUNDARIES}
        need(len(facts) == sum(kind in BOUNDARIES for kind, _ in visible), 'duplicate_boundary')
        if 'WORKER_READY' in facts:
            need({'START', 'RESULT'} <= set(facts), 'READY_without_processed_predecessors')
        if 'RESULT' in facts:
            need('START' in facts, 'RESULT_without_processed_START')
        ordered = [facts[k] for k in ('START', 'RESULT', 'WORKER_READY') if k in facts]
        clock_keys = ('perf_ns', 'received_time_ns', 'collector_dequeued_time_ns', 'collector_processing_time_ns')
        need(all(all(a[k] <= b[k] for k in clock_keys) for a,b in zip(ordered,ordered[1:])),
             'handler_boundary_chronology')
        for index, (kind, event) in enumerate(visible):
            stations.append(dict(request_id=rid, owner=owner, kind=kind,
                request_message_ordinal=index, payload=event, **{k:event[k] for k in
                ('perf_ns', 'received_time_ns', 'collector_dequeued_time_ns', 'collector_processing_time_ns')}))
        if 'WORKER_READY' in facts:
            support = ['HANDLER_RELEASED_AT_OBSERVED_READY']
        elif 'RESULT' in facts:
            support = ['HANDLER_HANDOFF_OR_RELEASE_NOTICE_PENDING']
        elif 'START' in facts:
            support = ['GPU_OR_HOST_WORK_ACTIVE', 'HANDLER_FINISHED_NOTICE_PENDING']
        else:
            support = ['NOT_STARTED', 'STARTED_NOTICE_PENDING', 'FINISHED_NOTICES_PENDING']
        requests[rid] = dict(owner=owner, sender=sending, observed_boundaries=facts,
            possible_current_states=support, exact_remaining_work=None,
            missing_START_proves_waiting=False, host_elapsed_is_completed_work=False)
    stations.sort(key=lambda e: (e['collector_processing_time_ns'], e['received_time_ns'],
        e['request_id'], e['request_message_ordinal']))
    offered = []
    for row in public:
        fact = requests.get(row['request_id'])
        if fact is not None:
            need(fact['owner'] == row['owner'] and
                 fact['sender'].get('scheduled_time_ns', row['arrival_ns']) == row['arrival_ns'], 'public_native_binding')
        else:
            fact = dict(owner=row['owner'], sender={}, observed_boundaries={},
                possible_current_states=['FUTURE_PUBLIC_ARRIVAL'] if row['arrival_ns'] > cut else
                    ['NOT_ENQUEUED', 'NOT_STARTED', 'STARTED_NOTICE_PENDING', 'FINISHED_NOTICES_PENDING'],
                exact_remaining_work=None, missing_START_proves_waiting=False,
                host_elapsed_is_completed_work=False)
        offered.append(dict(**row, native_state=fact,
            public_future_arrival=row['arrival_ns'] > cut,
            status='UNKNOWN' if row['request_id'] not in requests else 'OBSERVED_PREFIX_WITH_STATE_SUPPORT'))
    binding = dict(public_cohort=public, owner_producers=owner_producers,
                   execution_context_sha256=execution_context_sha256,
                   worker_lifetimes={o: dict(pid=r['pid'], spec_sha256=r.get('spec_sha256')) for o, r in roster.items()})
    if previous is not None:
        old = {k:v for k,v in previous.items() if k != 'state_sha256'}
        need(previous.get('state_sha256') == settled.digest(old) and previous.get('schema') == SCHEMA,
             'mutated_previous_state')
        need(cut >= previous['cutoff_ns'] and previous['binding'] == binding, 'changed_binding_or_backward_cut')
        before = [e for e in stations if e['collector_processing_time_ns'] <= previous['cutoff_ns']]
        need(before == previous['collector_processed_prefix'], 'changed_processed_prefix')
        for rid, old_sender in previous['sender_prefix'].items():
            need(rid in sender and all(sender[rid].get(k) == v for k,v in old_sender.items()), 'changed_sender_prefix')
        for rid, now_sender in sender.items():
            old_sender = previous['sender_prefix'].get(rid, {})
            need(all(k in old_sender for k,v in now_sender.items() if
                k != 'scheduled_time_ns' and v <= previous['cutoff_ns']), 'inserted_old_sender_fact')
    seed, reason = None, None
    # Native busy residuals are not accepted as an explicit latent checkpoint.
    # This is the existing proven entry point, not a new busy point estimate.
    try:
        need(snapshot['snapshot_time_ns'] == cut, 'historical_replay_cannot_certify_settled_snapshot')
        need(not set(ids) & {r['request_id'] for r in public}, 'current_cohort_in_native_history')
        need(all_record_events_processed and snapshot.get('queued_runtime_events') == [] and
             snapshot.get('runtime_errors') == [], 'unprocessed_or_failed_native_station')
        need(all({'START', 'RESULT', 'WORKER_READY'} <= set(r['observed_boundaries'])
             and len(r['sender']) == 4 for r in requests.values()), 'native_active_or_pending')
        seed = settled.from_snapshot(snapshot, cutoff_ns=cut, forecast_origin_ns=origin,
                                     owner_producers=owner_producers)
    except UnknownEvidence as exc:
        reason = str(exc)
    result = dict(schema=SCHEMA, cutoff_ns=cut, forecast_origin_ns=origin, binding=binding,
        offered_requests=len(public), full_offered_cohort=offered, sender_prefix=sender,
        native_request_states=requests, collector_processed_prefix=stations,
        reader_queued_envelope_count=len(snapshot.get('queued_runtime_events', []))
            if snapshot['snapshot_time_ns'] == cut else None,
        queued_message_bodies_read=False, dequeued_unprocessed_message_possible=True,
        protocol_idle_inferred_from_empty_queue=False,
        settled_initial_state=seed, continuation_state=None,
        reference_entry_eligible=seed is not None, busy_numeric_resume_eligible=False,
        blocked_reason=reason, required_additional_evidence=list(MISSING) if seed is None else [],
        native_source_authenticated=False, GPU_started=False, validation_labels_read=False)
    return _seal(result)


def forecast(state, **kwargs):
    """Actually call v5_2 on the supported boundary; fail closed otherwise."""
    need(state.get('state_sha256') == settled.digest({k:v for k,v in state.items()
         if k != 'state_sha256'}), 'mutated_state')
    need(kwargs.get('requests') == state['binding']['public_cohort'], 'forecast_cohort_changed')
    need(kwargs.get('window_start_ns') == state['forecast_origin_ns'], 'forecast_origin_changed')
    need(not any(k in kwargs for k in ('settled_initial_state', 'continuation_state',
        'initial_sender_available_ns', 'initial_worker_read_release_ns', 'initial_protocol_state')),
        'caller_cannot_override_native_state')
    need(state['reference_entry_eligible'] and state['settled_initial_state'] is not None,
         'busy_state_support_not_identified:' + str(state['blocked_reason']))
    from . import actual_worker_twoFIFO_reference_v5_2 as engine
    return engine.simulate(**kwargs, settled_initial_state=state['settled_initial_state'])
