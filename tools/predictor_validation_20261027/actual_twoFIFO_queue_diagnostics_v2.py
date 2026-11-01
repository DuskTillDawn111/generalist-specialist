"""Public-bound diagnostics of generated v5 worlds, never runtime risk features.

Work is the sum of macro response-reference coordinates. It is neither solo
wall service nor SM activity/resource demand. Work for not-yet-executed requests
requires an explicit public-key/arrival-ordinal supply; exposure cannot identify
it. All event clocks below describe the supplied CPU world, not observations.
"""
from collections import defaultdict
from fractions import Fraction

from .actual_worker_twoFIFO_query_adapter_v5 import clock, query_bound_view
from .macro_response_queue_reference_v1 import encode

SCHEMA = 'gsdeploy.actual_twoFIFO_queue_diagnostics.v2'
WORK_UNITS = 'sum_of_macro_response_reference_coordinate_ns_not_solo_wall_service'


def _decode(value):
    if type(value) is dict:
        if set(value) == {'numerator', 'denominator'}:
            return clock(value)
        return {key: _decode(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_decode(item) for item in value]
    return value


def _envelopes(rows, costs, scales):
    ordered = sorted(rows, key=lambda row: row['arrival_ns'])
    result = []
    for scale in scales:
        scale = clock(scale)
        if not scale:
            raise ValueError('Strictly positive envelope scale required')
        left = 0; count_max = 0; work_max = Fraction(0); work = Fraction(0)
        known = all(costs[row['request_id']] is not None for row in ordered)
        for right, row in enumerate(ordered):
            if known:
                work += costs[row['request_id']]
            while ordered[left]['arrival_ns'] <= row['arrival_ns'] - scale:
                if known:
                    work -= costs[ordered[left]['request_id']]
                left += 1
            count_max = max(count_max, right - left + 1)
            work_max = max(work_max, work)
        result.append(dict(window_ns=scale, max_arrival_count=count_max,
            max_offered_response_reference_work_ns=work_max if known else None,
            work_cost_missing_requests=sum(costs[row['request_id']] is None for row in ordered)))
    return result


def diagnose(reference, public_requests, *, window_start_ns, window_end_ns,
             drain_end_ns, scales_ns=(100_000_000, 1_000_000_000, 10_000_000_000),
             work_by_public_key_ordinal=None):
    """Explain the entire public arrival trace and events through world cutoff.

    Optional work keys are (owner, task, pool_id, public arrival ordinal), never
    historical request IDs. Missing costs remain UNKNOWN, including future work.
    Public binding certifies query equality only, not native execution fidelity.
    """
    view = query_bound_view(reference, public_requests, window_start_ns=window_start_ns,
        window_end_ns=window_end_ns, drain_end_ns=drain_end_ns)
    world = _decode(view)
    rows = world['public_offered_requests']; by_id = {row['request_id']: row for row in rows}
    start, end = world['original_window']; cutoff = world['cutoff_ns']
    drain = world['original_drain_end_ns']
    supplied = {} if work_by_public_key_ordinal is None else dict(work_by_public_key_ordinal)
    key_by_id = {rid: (row['owner'], row['task'], row['pool_id'],
                      world['public_key_arrival_ordinals'][rid]) for rid, row in by_id.items()}
    if set(supplied) - set(key_by_id.values()):
        raise ValueError('Work supply contains foreign public key/arrival ordinal')
    costs = {rid: None if supplied.get(key) is None else clock(supplied[key])
             for rid, key in key_by_id.items()}
    submits = world['parent_dispatches']; starts = world['starts']
    ready = world['READY_occurrences']; receipts = world['RESULT_receipts']
    result_occurrences = {event['request_id']: event['occurrence_ns'] for event in world['trace']
                          if event['kind'] == 'latent_occurrence' and event['action'] == 'result'}
    schedule = world['transport_schedule']
    # Dict iteration preserves generated sender sequence, including equal-time
    # submit ties. Arrival order / lexical request ID is not the worker FIFO.
    submit_order = {rid: order for order, rid in enumerate(submits)}
    schedule_order = {rid: order for order, rid in enumerate(schedule)}
    exposures = defaultdict(list)
    cuts = {start, cutoff}
    for row in rows:
        cuts.add(row['arrival_ns'])
    for values in (submits, starts, ready, receipts, result_occurrences):
        cuts.update(values.values())
    for send in schedule.values():
        cuts.update(send[key] for key in ('enqueued_ns', 'parent_submit_ns',
            'parent_write_record_ns', 'stdin_line_available_ns'))
    for exposure in world['response_exposure']:
        exposures[exposure['request_id']].append(exposure)
        cuts.update((exposure['left_ns'], exposure['right_ns']))
    cuts = sorted(at for at in cuts if start <= at <= cutoff)

    def consumed(rid, at):
        return sum((max(Fraction(0), min(at, e['right_ns']) - e['left_ns']) * e['rate']
                    for e in exposures[rid]), Fraction(0))

    for rid, full in costs.items():
        used = consumed(rid, cutoff)
        if full is not None and (used > full or (rid in ready and used != full)):
            raise ValueError('Supplied full reference work conflicts with generated exposure')

    def occurred(events, rid, at):
        return rid in events and events[rid] <= at

    def snapshot(at):
        lifecycle = {key: [] for key in ('scheduled_not_arrived', 'arrived_not_submitted',
            'submitted_waiting_START', 'running_START_to_READY', 'worker_released')}
        delivery = {key: [] for key in ('RESULT_not_occurred', 'RESULT_in_transit', 'RESULT_reader_received')}
        waiting = []; pending = defaultdict(list); running = {}; sender_waiting = defaultdict(list)
        for rid, row in by_id.items():
            if row['arrival_ns'] > at:
                state = 'scheduled_not_arrived'
            elif not occurred(submits, rid, at):
                state = 'arrived_not_submitted'
                if schedule[rid]['enqueued_ns'] <= at:
                    sender_waiting[row['owner']].append(rid)
            elif not occurred(starts, rid, at):
                state = 'submitted_waiting_START'; pending[row['owner']].append(rid)
            elif not occurred(ready, rid, at):
                state = 'running_START_to_READY'; running[row['owner']] = rid
            else:
                state = 'worker_released'
            lifecycle[state].append(rid)
            if state in ('arrived_not_submitted', 'submitted_waiting_START'):
                waiting.append(rid)
            delivery['RESULT_reader_received' if occurred(receipts, rid, at) else
                     'RESULT_in_transit' if occurred(result_occurrences, rid, at) else
                     'RESULT_not_occurred'].append(rid)
        heads = []; convoy = 0
        for owner, members in sorted(pending.items()):
            members.sort(key=lambda rid: submit_order[rid])
            head = members[0]
            behind_handler = len(members) if owner in running else 0
            behind_waiting_head = len(members) - 1 if owner not in running else 0
            convoy += behind_handler + behind_waiting_head
            heads.append(dict(owner=owner, head_request_id=head, FIFO_request_ids=members,
                head_submit_wait_ns=at-submits[head], head_arrival_age_ns=at-by_id[head]['arrival_ns'],
                running_predecessor_request_id=running.get(owner),
                pending_behind_running_handler=behind_handler,
                pending_behind_waiting_head=behind_waiting_head,
                resource_admission_feasibility=None))
        sender_heads = []
        for owner, members in sorted(sender_waiting.items()):
            # The transport schedule itself was inserted in enqueue sequence.
            members.sort(key=lambda rid: schedule_order[rid])
            head = members[0]
            sender_heads.append(dict(owner=owner, head_request_id=head,
                enqueued_waiting_request_ids=members, head_enqueue_wait_ns=at-schedule[head]['enqueued_ns']))
        arrived = [rid for rid in by_id if by_id[rid]['arrival_ns'] <= at]
        unfinished = [rid for rid in by_id if not occurred(receipts, rid, at)]
        def residual(ids):
            if any(costs[rid] is None for rid in ids):
                return None
            return sum((costs[rid]-consumed(rid, at) for rid in ids), Fraction(0))
        return dict(at_ns=at, worker_lifecycle=lifecycle, RESULT_delivery=delivery,
            lifecycle_count={key: len(ids) for key, ids in lifecycle.items()},
            RESULT_delivery_count={key: len(ids) for key, ids in delivery.items()},
            offered_requests=len(rows), arrived_requests=len(arrived),
            scheduled_but_not_START_request_ids=lifecycle['scheduled_not_arrived'] + waiting,
            arrived_waiting_START_request_ids=waiting, RESULT_unfinished_request_ids=unfinished,
            arrived_RESULT_unfinished_request_ids=[rid for rid in unfinished if rid in arrived],
            worker_FIFO_heads=heads, parent_sender_FIFO_heads=sender_heads,
            FIFO_convoy_pending_requests=convoy,
            arrived_remaining_response_reference_work_ns=residual(arrived),
            complete_cohort_remaining_response_reference_work_ns=residual(list(by_id)),
            arrived_not_START_reference_work_ns=residual(waiting),
            max_waiting_arrival_age_ns=max((at-by_id[rid]['arrival_ns'] for rid in waiting), default=Fraction(0)),
            sm_demand=None, dram_demand=None, l2_demand=None, power_watts=None, memory_bytes=None)

    states = [snapshot(at) for at in cuts]
    slices = [dict(left_ns=a['at_ns'], right_ns=b['at_ns'], duration_ns=b['at_ns']-a['at_ns'],
                   state=a) for a, b in zip(states, states[1:])]

    def episodes(field):
        result = []; opened = None
        for state in states:
            busy = bool(state[field])
            if busy and opened is None:
                opened = state['at_ns']
            elif not busy and opened is not None:
                result.append(dict(backlog_start_ns=opened, clear_ns=state['at_ns'],
                    recovery_ns=state['at_ns']-opened, right_censored=False,
                    left_censored=opened == start))
                opened = None
        if opened is not None:
            result.append(dict(backlog_start_ns=opened, clear_ns=None, recovery_ns=None,
                observed_duration_ns=cutoff-opened, right_censored=True, left_censored=opened == start))
        return result

    endpoint = states[-1]
    latencies = [receipts[rid]-by_id[rid]['arrival_ns'] for rid in by_id if rid in receipts]
    complete = len(latencies) == len(rows)
    p95 = None
    if complete and latencies:
        ordered = sorted(latencies); position = Fraction(95, 100)*(len(ordered)-1)
        lower = position.numerator // position.denominator
        p95 = ordered[lower] + (ordered[min(lower+1, len(ordered)-1)]-ordered[lower])*(position-lower)
    # If simulation stops early, neither future completion nor original drain
    # recovery is inferred from the transport schedule or current active rates.
    original_drain = dict(status='SIMULATED_THROUGH_ORIGINAL_DRAIN' if cutoff == drain else
        'UNKNOWN_SIMULATION_STOPPED_BEFORE_ORIGINAL_DRAIN',
        arrived_waiting_START_request_ids=endpoint['arrived_waiting_START_request_ids'] if cutoff == drain else None,
        RESULT_unfinished_request_ids=endpoint['RESULT_unfinished_request_ids'] if cutoff == drain else None,
        worker_unreleased_request_ids=[rid for group in ('arrived_not_submitted','submitted_waiting_START',
            'running_START_to_READY') for rid in endpoint['worker_lifecycle'][group]] if cutoff == drain else None)
    return encode(dict(schema=SCHEMA, source_schema=world['schema'],
        status='GENERATED_WORLD_CPU_DIAGNOSTICS_ONLY',
        query_fingerprint_sha256=world['query_fingerprint_sha256'],
        public_query_identity_verified=True, public_offered_requests=len(rows),
        original_window=[start, end], original_drain_end_ns=drain, cutoff_ns=cutoff,
        work_units=WORK_UNITS, work_source='explicit_public_key_arrival_ordinal_supply',
        unknown_work_public_keys=[list(key_by_id[rid]) for rid in by_id if costs[rid] is None],
        full_public_arrival_envelopes=_envelopes(rows, costs, scales_ns),
        event_slices=slices, cutoff_state=endpoint,
        worker_lifecycle_and_RESULT_delivery_are_separate_axes=True,
        queue_recovery_episodes=episodes('arrived_waiting_START_request_ids'),
        RESULT_backlog_recovery_episodes=episodes('arrived_RESULT_unfinished_request_ids'),
        RESULT_backlog_includes_not_yet_arrived_public_requests=False,
        max_arrived_waiting_START=max(len(state['arrived_waiting_START_request_ids']) for state in states),
        max_waiting_arrival_age_ns=max([state['max_waiting_arrival_age_ns'] for state in states] +
            [s['right_ns']-by_id[rid]['arrival_ns'] for s in slices
             for rid in s['state']['arrived_waiting_START_request_ids']]),
        max_worker_FIFO_head_submit_wait_ns=max([Fraction(0)] +
            [s['right_ns']-submits[h['head_request_id']] for s in slices
             for h in s['state']['worker_FIFO_heads']]),
        FIFO_convoy_pending_request_ns=sum((s['duration_ns']*s['state']['FIFO_convoy_pending_requests']
                                            for s in slices), Fraction(0)),
        FIFO_convoy_is_generated_serial_predecessor_wait_not_resource_HOL=True,
        resource_feasible_HOL_blocked_ns=None, original_drain=original_drain,
        reader_E2E_completed_requests=len(latencies), reader_E2E_offered_requests=len(rows),
        reader_E2E_P95_ns=p95,
        reader_E2E_P95_definition='linear_empirical_percentile_over_complete_public_cohort_exact_unrounded_clocks',
        native_current_visible_risk_features=None,
        native_observation_assimilated=False, native_assimilation_eligible=False,
        main_queue_provider_eligible=False, resource_provider_qualified=False,
        transport_decomposition_fitted=False, busy_transport_transfer_qualified=False,
        independent_prediction_accuracy_validated=False, calibrated_long_term_risk=False,
        stability_rule_changed=False, stability_qualified=False,
        validation_labels_read=False, GPU_started=False))
