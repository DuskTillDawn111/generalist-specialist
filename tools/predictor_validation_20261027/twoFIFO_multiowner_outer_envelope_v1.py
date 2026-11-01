"""Sound, possibly loose reader-latency bounds for finite full-world rates.

This outer relaxation deliberately forgets cross-owner rate/event dependence.
It never selects a hidden pipe time or asserts that extrema are attainable.
Every actual trajectory of the declared model is inside the envelope; empirical
rate transfer and physical GPU correctness are separate, unqualified hypotheses.
"""
from collections import defaultdict
from fractions import Fraction

from gsdeploy.forecast_errors import UnknownEvidence
from .macro_response_queue_reference_v1 import rational, validate_plan
from .protocol_timing_slack_v3 import checked as checked_slacks
from .twoFIFO_paired_transport_v3 import source_constraints, sender_schedule
from .twoFIFO_transport_partial_identification_v2 import project_target
from .twoFIFO_single_owner_transport_intervals_v1 import p95

SCHEMA = 'gsdeploy.twoFIFO_multiowner_outer_envelope.v1'
CERTIFICATE = dict(schema=SCHEMA+'.caller_subdomain_declaration',
    all_visible_and_future_owners_in_public_roster=True,
    initial_visible_peers_empty=True, initial_protocol_pending_empty=True,
    per_owner_single_stdout_lifetime=True,
    rates_depend_only_on_owner_macro_phase_and_visible_peer_counts=True,
    fixed_work_and_nonnegative_protocol_slacks_within_draw=True,
    no_external_reader_or_sender_backpressure=True)


def need(condition, reason):
    if not condition:
        raise UnknownEvidence('UNKNOWN_MULTIOWNER_OUTER_ENVELOPE:'+reason)


def _prepare(requests, plans, rates, timing, transport, sender_initial,
             worker_initial, protocol, certificate):
    need(certificate == CERTIFICATE, 'EXPLICIT_CLOSED_WORLD_DECLARATION_REQUIRED')
    owners = {r['owner'] for r in requests}
    need(owners and set(sender_initial) == set(worker_initial) == owners,
         'EXACT_OWNER_STATIONS_REQUIRED')
    need(type(protocol) is dict and set(protocol) == {
        'reader_free', 'collector_free', 'last_collector_arrival', 'pending_messages'}
        and protocol['pending_messages'] in ([], ()), 'UNSETTLED_PROTOCOL')
    for at in [*sender_initial.values(), *worker_initial.values(),
               *protocol['reader_free'].values(), protocol['collector_free'],
               protocol['last_collector_arrival']]:
        rational(at)
    counts = defaultdict(int)
    for r in requests:
        counts[(r['owner'], r['task'], r['pool_id'])] += 1
    ordinal = defaultdict(int); rows = defaultdict(list); producers = {}
    keys = set(); extrema = {}
    for position, r in sorted(enumerate(requests), key=lambda p: (p[1]['arrival_ns'], p[0])):
        key = r['owner'], r['task'], r['pool_id']; index = ordinal[key]; ordinal[key] += 1
        need(key in plans, 'MISSING_PUBLIC_PLAN')
        supplied = plans[key]
        need(type(supplied) is not tuple or len(supplied) == counts[key], 'PLAN_SLOT_DENOMINATOR')
        plan = supplied[index] if type(supplied) is tuple else supplied
        validate_plan(plan)
        need(plan.dispatch_to_worker_start_ns == 0, 'LEGACY_DISPATCH_DELAY')
        drawkey = (*key, index); keys.add(drawkey)
        need(drawkey in transport, 'MISSING_PAIRED_TRANSPORT')
        draw = transport[drawkey]; source_constraints(draw)
        producer = plan.ready_producer_id
        need(producers.get(r['owner'], producer) == producer, 'CHANGED_OWNER_STDOUT_LIFETIME')
        producers[r['owner']] = producer
        elapsed = [Fraction(0), Fraction(0)]; messages = []
        for macro in plan.macros:
            ratekey = r['owner'], macro.phase
            if ratekey not in extrema:
                grid = [(ratekey[0], ratekey[1], n, same)
                        for n in range(len(owners)) for same in range(n+1)]
                need(all(k in rates for k in grid), 'INCOMPLETE_FINITE_RATE_GRID:'+repr(ratekey))
                values = [rational(rates[k], positive=True) for k in grid]
                extrema[ratekey] = min(values), max(values)
            slow, fast = extrema[ratekey]
            for mark in macro.marks:
                need(mark.producer_id == producer, 'MARK_ON_EXTERNAL_PRODUCER')
                need(plan in timing and mark.identity in timing[plan], 'MISSING_FIXED_PROTOCOL_SLACK')
                slack = checked_slacks(timing[plan][mark.identity])
                messages.append(dict(identity=mark.identity, action=mark.action,
                    offset=[elapsed[0]+mark.work_ns/fast, elapsed[1]+mark.work_ns/slow],
                    reader_slack_ns=slack['reader_slack_ns']))
            elapsed = [elapsed[0]+macro.reference_work_ns/fast,
                       elapsed[1]+macro.reference_work_ns/slow]
        need(plan in timing and 'worker_READY' in timing[plan], 'MISSING_READY_PROTOCOL_SLACK')
        slack = checked_slacks(timing[plan]['worker_READY'])
        messages.append(dict(identity='worker_READY', action='ready', offset=list(elapsed),
                             reader_slack_ns=slack['reader_slack_ns']))
        rows[r['owner']].append(dict(public=r, position=position, draw=draw,
            duration_bounds=elapsed, messages=messages, producer=producer))
    need(set(transport) == keys, 'TRANSPORT_OFFFERED_DENOMINATOR')
    need(len(set(producers.values())) == len(owners), 'SHARED_STDOUT_PRODUCER')
    need(set(protocol['reader_free']) <= set(producers.values()), 'EXTERNAL_READER')
    for owner, values in rows.items():
        values.sort(key=lambda row: (row['public']['arrival_ns']+
                    row['draw'].enqueue_after_arrival_ns, row['position']))
        previous = rational(sender_initial[owner])
        for row in values:
            row['send'] = sender_schedule(row['public']['arrival_ns'], previous, row['draw'])
            previous = row['send']['parent_write_record_ns']
    return rows, extrema


def evaluate(requests, plans, rates, *, protocol_slacks, transport_draws,
             domain_certificate, initial_sender_available_ns,
             initial_worker_read_release_ns, initial_protocol_state,
             window_start_ns, window_end_ns, drain_end_ns):
    """Bound the whole original cohort without enumerating latent endpoints.

    For every macro, all possible count-state rates must be present and positive.
    Finite rate extrema bound work duration even when collector processing moves
    the visible-state transitions. Producer FIFO reader times are monotone in
    occurrence times and independent of collector completion in this model.
    Collector processing times themselves are intentionally not bounded here.
    """
    start, end, drain = map(rational, (window_start_ns, window_end_ns, drain_end_ns))
    if not start < end <= drain:
        raise ValueError('Original positive window and fixed drain required')
    requests = [dict(r) for r in requests]; ids = []
    for r in requests:
        if (set(r) != {'request_id', 'owner', 'task', 'pool_id', 'arrival_ns'}
            or any(type(r[k]) is not str or not r[k] for k in ('request_id', 'owner', 'task', 'pool_id'))
            or r['request_id'] in ids or not start <= rational(r['arrival_ns']) < end):
            raise ValueError('Exact distinct complete public cohort required')
        ids.append(r['request_id'])
    base = dict(schema=SCHEMA, offered_requests=len(ids), offered_request_ids=ids,
        all_offered_requests_retained=True, envelope_is_sharp=False,
        extrema_simultaneously_attainable_claimed=False, latent_probability_assigned=False,
        latent_point_selected=False, physical_rate_bounds_certified=False,
        resource_bounds=None, full_U_bounds=None, collector_processing_bounds=None,
        main_queue_provider_eligible=False, final_gain_eligible=False,
        validation_labels_read=False, GPU_started=False)
    try:
        rows, extrema = _prepare(requests, plans, rates, protocol_slacks, transport_draws,
            initial_sender_available_ns, initial_worker_read_release_ns,
            initial_protocol_state, domain_certificate)
    except UnknownEvidence as exc:
        return dict(base, status='UNKNOWN_OUTSIDE_FINITE_FULL_WORLD_DOMAIN', reason=str(exc),
            requests=[dict(request_id=rid, reader_E2E_ns_bounds=None,
                fixed_drain_incomplete_indicator_bounds=[0, 1]) for rid in ids],
            complete_cohort_P95_ns_bounds=None)
    bounded = {}
    for owner, values in rows.items():
        ready = [rational(initial_worker_read_release_ns[owner])]*2
        reader = [rational(initial_protocol_state['reader_free'].get(values[0]['producer'], start))]*2
        for row in values:
            s = row['send']['parent_submit_ns']; source = row['send']['paired_source_constraints']
            # START(R,u) is monotone in R. Optimize the paired feasible u set
            # separately at the two bounding R values; never assume peer-rate
            # trajectories at those extrema are jointly attainable.
            beginnings = [project_target(source, s, R)['START_ns_bounds'][i]
                          for i, R in enumerate(ready)]
            ready = [beginnings[i]+row['duration_bounds'][i] for i in (0, 1)]
            receipt = None
            for message in row['messages']:
                reader = [max(reader[i], beginnings[i]+message['offset'][i])+
                          message['reader_slack_ns'] for i in (0, 1)]
                if message['action'] == 'result':
                    receipt = list(reader)
            r = row['public']; a, b = receipt
            bounded[r['request_id']] = dict(request_id=r['request_id'],
                START_ns_bounds=beginnings, READY_ns_bounds=ready,
                RESULT_reader_ns_bounds=receipt,
                reader_E2E_ns_bounds=[a-r['arrival_ns'], b-r['arrival_ns']],
                fixed_drain_incomplete_indicator_bounds=[int(a > drain), int(b > drain)])
    output = [bounded[rid] for rid in ids]
    quantiles = [p95([row['reader_E2E_ns_bounds'][i] for row in output]) for i in (0, 1)]
    always = all(row['RESULT_reader_ns_bounds'][1] <= drain for row in output)
    return dict(base, status='CONSERVATIVE_FINITE_RATE_FULL_COHORT_OUTER_ENVELOPE',
        requests=output, rate_extrema=[dict(owner=k[0], phase=k[1], min_rate=v[0], max_rate=v[1])
                                      for k, v in sorted(extrema.items())],
        untruncated_model_complete_cohort_P95_ns_bounds=quantiles,
        complete_cohort_P95_ns_bounds=quantiles if always else None,
        original_fixed_drain_P95_defined_for_all_feasible_worlds=always,
        fixed_drain_incomplete_fraction_bounds=[Fraction(sum(row['fixed_drain_incomplete_indicator_bounds'][i]
            for row in output), len(output)) for i in (0, 1)],
        P95_not_inferred_from_completed_subset=True,
        empirical_rate_transfer_still_requires_independent_validation=True)
