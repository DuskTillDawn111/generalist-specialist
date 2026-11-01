"""Exact rational CPU composition reference for explicitly supplied joint worlds.

This is neither a trained joint law nor native assimilation. A caller supplies
public-input templates, response-clock marks and their JOINT delivery delays.
Actual macro transitions and processed notifications use separate event queues.
No physical context, SM clock, global admission cap or utility model is used.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
import heapq

from gsdeploy.forecast_errors import UnknownEvidence

SCHEMA = 'gsdeploy.macro_response_queue_finite_reference.v1'
ACTIONS = {'phase', 'clear', 'query', 'result', 'diagnostic'}


def rational(value, *, positive=False):
    if type(value) not in (int, Fraction) or value < 0 or (positive and value == 0):
        raise ValueError('Nonnegative exact rational clock required')
    return Fraction(value)


@dataclass(frozen=True)
class Mark:
    identity: str
    work_ns: Fraction
    processing_delay_ns: Fraction
    action: str
    producer_id: str
    phase: str | None = None
    # Query evidence stays attached to its original event/stream identity.
    event_id: str | None = None
    stream_id: str | None = None
    query_complete: bool | None = None
    # CLEAR must name its scope; a stream/query is never request-wide END.
    clear_scope: str | None = None
    # Reader receipt is the E2E endpoint; processing controls peer visibility.
    reader_delay_ns: Fraction | None = None


@dataclass(frozen=True)
class Macro:
    identity: str
    phase: str
    reference_work_ns: Fraction
    marks: tuple[Mark, ...]


@dataclass(frozen=True)
class RequestPlan:
    macros: tuple[Macro, ...]
    ready_processing_delay_ns: Fraction
    ready_producer_id: str
    dispatch_to_worker_start_ns: Fraction = Fraction(0)
    ready_reader_delay_ns: Fraction | None = None


def validate_plan(plan):
    if type(plan) is not RequestPlan or not plan.macros or type(plan.macros) is not tuple:
        raise ValueError('Explicit nonempty immutable joint request template required')
    rational(plan.ready_processing_delay_ns)
    rational(plan.dispatch_to_worker_start_ns)
    if plan.ready_reader_delay_ns is not None:
        if rational(plan.ready_reader_delay_ns) > plan.ready_processing_delay_ns:
            raise ValueError('READY reader receipt cannot follow processing')
    if not plan.ready_producer_id:
        raise ValueError('Explicit READY producer identity required')
    ids = set(); results = 0; macro_ids = set()
    for macro in plan.macros:
        if (type(macro) is not Macro or not macro.identity or not macro.phase
                or macro.identity in macro_ids or type(macro.marks) is not tuple):
            raise ValueError('Distinct immutable response macro identity required')
        macro_ids.add(macro.identity)
        work = rational(macro.reference_work_ns, positive=True)
        previous = Fraction(0)
        for mark in macro.marks:
            if (type(mark) is not Mark or not mark.identity or mark.identity in ids
                    or mark.action not in ACTIONS or not mark.producer_id):
                raise ValueError('Distinct explicit joint mark identity and producer required')
            ids.add(mark.identity)
            at = rational(mark.work_ns); rational(mark.processing_delay_ns)
            if mark.reader_delay_ns is not None:
                if rational(mark.reader_delay_ns) > mark.processing_delay_ns:
                    raise ValueError('Reader receipt cannot follow collector processing')
            if not previous <= at <= work:
                raise ValueError('Joint response-clock marks must be ordered inside their macro')
            previous = at
            if mark.action == 'phase' and not mark.phase:
                raise ValueError('Phase notification requires its explicit semantic phase')
            if mark.action == 'query' and (not mark.event_id or not mark.stream_id
                    or type(mark.query_complete) is not bool):
                raise ValueError('Query requires original stream/event identity and outcome')
            if mark.action == 'clear' and mark.clear_scope not in ('macro', 'request'):
                raise ValueError('CLEAR requires explicit macro or request terminal scope')
            results += mark.action == 'result'
    if results != 1:
        raise ValueError('Exactly one independently placed RESULT mark required')


def encode(value):
    """Keep exact rational ns in JSON without presenting them as native stamps."""
    if isinstance(value, Fraction):
        return {'numerator': value.numerator, 'denominator': value.denominator}
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def simulate(requests, plans, rates, *, window_start_ns, window_end_ns,
             drain_end_ns, cutoff_ns=None, producer_fifo=()):
    """Compose one explicit CPU world; never inspect heldout execution traces.

    plans are indexed ONLY by public (owner, task, pool_id), never request/cohort
    ID. A value is a shared RequestPlan or an immutable tuple of prechosen joint
    draws consumed by that public key's dispatch ordinal. This permits a shared
    capture latent plus distinct request draws without historical ID lookup.
    rates[(own_phase, visible_peer_count, same_phase_count)] are explicit
    positive rational response rates. Their scientific provenance is a separate
    requirement. Delays are already jointly chosen with marks/reference work;
    this routine does not independently draw or calibrate any marginal law.

    The dispatch hypothesis is explicit: a worker takes its FIFO head after
    READY processing. READY may precede RESULT receipt. This hypothesis needs a
    runtime-specific audit before any native provider can use this reference.
    """
    start, end, drain = map(rational, (window_start_ns, window_end_ns, drain_end_ns))
    if not start < end <= drain:
        raise ValueError('Original measurement and drain window required')
    cutoff = drain if cutoff_ns is None else rational(cutoff_ns)
    if not start <= cutoff <= drain:
        raise ValueError('Cutoff must be inside original fixed drain')
    cohort = {}; arrivals = []; pending = defaultdict(deque); owners = set()
    offered_per_key = defaultdict(int)
    for i, row in enumerate(requests):
        if (type(row) is not dict or set(row) != {'request_id', 'owner', 'task', 'pool_id', 'arrival_ns'}
                or not all(type(row[k]) is str and row[k] for k in ('request_id','owner','task','pool_id'))
                or row['request_id'] in cohort):
            raise ValueError('Complete distinct public cohort identities required')
        arrival = rational(row['arrival_ns'])
        if not start <= arrival < end:
            raise ValueError('Every offered request must be in original arrival window')
        key = row['owner'], row['task'], row['pool_id']
        if key not in plans:
            raise UnknownEvidence('UNKNOWN_PUBLIC_INPUT_RESPONSE_TEMPLATE')
        supplied = plans[key]
        if type(supplied) is tuple:
            if not supplied:
                raise UnknownEvidence('UNKNOWN_EMPTY_PUBLIC_INPUT_JOINT_DRAWS')
            for plan in supplied:
                validate_plan(plan)
        else:
            validate_plan(supplied)
        offered_per_key[key] += 1
        cohort[row['request_id']] = dict(row)
        arrivals.append((arrival, i, row['request_id'])); owners.add(row['owner'])
    heapq.heapify(arrivals)
    for key,count in offered_per_key.items():
        if type(plans[key]) is tuple and len(plans[key]) != count:
            raise UnknownEvidence('EXPLICIT_ONE_JOINT_DRAW_PER_OFFERED_PUBLIC_KEY_ORDINAL_REQUIRED')
    checked_rates = {}
    for key, rate in rates.items():
        if (type(key) is not tuple or len(key) != 3 or type(key[0]) is not str
                or any(type(x) is not int for x in key[1:]) or not 0 <= key[2] <= key[1]):
            raise ValueError('Explicit phase and visible-evidence count rate key required')
        checked_rates[key] = rational(rate, positive=True)
    fifo = frozenset(producer_fifo)
    # Templates and lookup rates are snapshotted before any execution mutation.
    plans = dict(plans)
    active = {}; occupied = {}; visible = {}; notifications = []
    worker_starts = []; receipt_events = []
    starts = {}; dispatches = {}; ready_occurrences = {}; receipts = {}; result_processed = {}
    trace = []; exposure = []
    sequence = 0; producer_next = defaultdict(int); producer_last_available = {}
    maximum_parallel = 0; clock = start; phase_watermarks = {}
    owner_issued = defaultdict(int); processed_issuance_watermarks = {}
    draw_cursors = defaultdict(int); consumed_draws = []
    producer_last_reader = {}

    def notify(owner, request_id, action, delay, producer, identity, phase=None, mark=None,
               macro_id=None,reader_delay_override=None):
        nonlocal sequence
        available = clock + rational(delay)
        reader = (available if mark is None or mark.reader_delay_ns is None
            else clock+rational(mark.reader_delay_ns))
        if reader_delay_override is not None:
            reader=clock+rational(reader_delay_override)
        producer_sequence = producer_next[producer]; producer_next[producer] += 1
        issuance = owner_issued[owner]; owner_issued[owner] += 1
        if producer in fifo and available < producer_last_available.get(producer, start):
            raise UnknownEvidence('JOINT_WORLD_VIOLATES_EXPLICIT_PRODUCER_FIFO')
        if producer in fifo and reader < producer_last_reader.get(producer,start):
            raise UnknownEvidence('JOINT_WORLD_VIOLATES_EXPLICIT_PRODUCER_READER_FIFO')
        producer_last_available[producer] = available
        producer_last_reader[producer] = reader
        sequence += 1
        body = dict(owner=owner, request_id=request_id, action=action, phase=phase,
            identity=identity, occurrence_ns=clock, producer_id=producer,
            producer_sequence=producer_sequence,macro_id=macro_id,
            clear_scope=None if mark is None else mark.clear_scope,
            owner_issuance_ordinal=issuance,reader_receipt_ns=reader)
        if mark and mark.action == 'query':
            body.update(event_id=mark.event_id, stream_id=mark.stream_id,
                query_complete=mark.query_complete, all_GPU_complete=None)
        heapq.heappush(notifications, (available, sequence, body))
        if action == 'result':
            heapq.heappush(receipt_events,(reader,sequence,request_id))
        trace.append(dict(kind='latent_occurrence', **body))

    def dispatch():
        nonlocal maximum_parallel
        for owner in sorted(owners):
            if owner in occupied or not pending[owner]:
                continue
            request_id = pending[owner].popleft(); row = cohort[request_id]
            key = owner, row['task'], row['pool_id']; ordinal = draw_cursors[key]
            supplied = plans[key]
            plan = supplied[ordinal] if type(supplied) is tuple else supplied
            draw_cursors[key] += 1
            consumed_draws.append(dict(owner=owner,task=row['task'],pool_id=row['pool_id'],
                public_key_dispatch_ordinal=ordinal,request_id=request_id,
                draw_mode='per_ordinal_prechosen_joint_draw' if type(supplied) is tuple else 'explicit_shared_template'))
            occupied[owner] = request_id; dispatches[request_id] = clock
            heapq.heappush(worker_starts,(clock+plan.dispatch_to_worker_start_ns,owner,request_id,plan))
            trace.append(dict(kind='parent_dispatch', owner=owner, request_id=request_id, at_ns=clock))
        maximum_parallel = max(maximum_parallel, len(occupied))

    def rate_for(owner, state):
        phase = state['plan'].macros[state['macro_index']].phase
        peer = [value['phase'] for alias,value in visible.items() if alias != owner]
        key = phase, len(peer), sum(value == phase for value in peer)
        if key not in checked_rates:
            raise UnknownEvidence('UNKNOWN_VISIBLE_MACRO_RESPONSE_RATE:' + repr(key))
        return checked_rates[key], key

    while True:
        # Drain all equal-time occurrences/notifications/dispatch cascades before
        # integrating the next strictly positive interval. Old pending notices
        # survive actual macro transitions and even the next request START.
        changed = True
        while changed:
            changed = False
            while arrivals and arrivals[0][0] <= clock:
                _, _, request_id = heapq.heappop(arrivals)
                pending[cohort[request_id]['owner']].append(request_id); changed = True
            while worker_starts and worker_starts[0][0] <= clock:
                at,owner,request_id,plan=heapq.heappop(worker_starts)
                active[owner]=dict(request_id=request_id,plan=plan,macro_index=0,
                    age=Fraction(0),next_mark=0)
                starts[request_id]=at
                trace.append(dict(kind='request_start',owner=owner,request_id=request_id,at_ns=at))
                changed=True
            for owner in sorted(tuple(active)):
                state = active[owner]; plan = state['plan']
                macro = plan.macros[state['macro_index']]
                while (state['next_mark'] < len(macro.marks)
                        and macro.marks[state['next_mark']].work_ns <= state['age']):
                    mark = macro.marks[state['next_mark']]; state['next_mark'] += 1
                    notify(owner,state['request_id'],mark.action,mark.processing_delay_ns,
                        mark.producer_id,mark.identity,mark.phase,mark,macro.identity); changed = True
                if state['age'] == macro.reference_work_ns:
                    trace.append(dict(kind='macro_complete',owner=owner,request_id=state['request_id'],
                        macro_id=macro.identity,at_ns=clock))
                    state['macro_index'] += 1; state['age'] = Fraction(0); state['next_mark'] = 0
                    if state['macro_index'] == len(plan.macros):
                        ready_occurrences[state['request_id']] = clock
                        notify(owner,state['request_id'],'ready',plan.ready_processing_delay_ns,
                            plan.ready_producer_id,'worker_READY',
                            reader_delay_override=plan.ready_reader_delay_ns)
                        del active[owner]
                    changed = True
            while receipt_events and receipt_events[0][0] <= clock:
                received,_,request_id=heapq.heappop(receipt_events)
                if request_id in receipts:
                    raise ValueError('Duplicate generated RESULT reader receipt')
                receipts[request_id]=received
                trace.append(dict(kind='RESULT_reader_receipt',request_id=request_id,at_ns=received))
                changed=True
            while notifications and notifications[0][0] <= clock:
                available, _, body = heapq.heappop(notifications)
                owner, request_id, action = body['owner'],body['request_id'],body['action']
                trace.append(dict(kind='processed_notification',available_ns=available,**body))
                if action == 'phase':
                    # Keep the watermark even after CLEAR/READY: an old BEGIN
                    # must not resurrect an already closed handler or macro.
                    if body['owner_issuance_ordinal'] < processed_issuance_watermarks.get(owner, -1):
                        raise UnknownEvidence('UNQUALIFIED_CROSS_PRODUCER_STALE_PHASE_POLICY')
                    phase_watermarks[owner] = body['occurrence_ns']
                    processed_issuance_watermarks[owner] = body['owner_issuance_ordinal']
                    visible[owner] = dict(request_id=request_id,phase=body['phase'],
                        macro_id=body['macro_id'],occurrence_ns=body['occurrence_ns'],
                        owner_issuance_ordinal=body['owner_issuance_ordinal'])
                elif action == 'clear':
                    phase_watermarks[owner] = max(phase_watermarks.get(owner,start),body['occurrence_ns'])
                    processed_issuance_watermarks[owner] = max(
                        processed_issuance_watermarks.get(owner,-1),body['owner_issuance_ordinal'])
                    prior = visible.get(owner,{})
                    if (body['clear_scope'] == 'request' and prior.get('request_id') == request_id
                            and prior['owner_issuance_ordinal'] > body['owner_issuance_ordinal']):
                        raise UnknownEvidence('REQUEST_TERMINAL_PRECEDES_LATER_PHASE_ISSUANCE')
                    if (prior.get('request_id') == request_id and
                            (body['clear_scope'] == 'request' or prior.get('macro_id') == body['macro_id'])):
                        visible.pop(owner)
                elif action == 'result':
                    if request_id in result_processed:
                        raise ValueError('Duplicate generated RESULT processing')
                    result_processed[request_id] = available
                elif action == 'ready':
                    if occupied.get(owner) != request_id or owner in active:
                        raise ValueError('READY must follow its complete actual handler')
                    occupied.pop(owner)
                    phase_watermarks[owner] = max(phase_watermarks.get(owner,start),body['occurrence_ns'])
                    processed_issuance_watermarks[owner] = max(
                        processed_issuance_watermarks.get(owner,-1),body['owner_issuance_ordinal'])
                    if visible.get(owner,{}).get('request_id') == request_id:
                        visible.pop(owner)
                # Query never clears a response macro or changes its phase.
                changed = True
            before = len(dispatches); dispatch(); changed |= len(dispatches) != before
        if clock == cutoff:
            break
        next_clock = cutoff
        if arrivals:
            next_clock = min(next_clock,arrivals[0][0])
        if notifications:
            next_clock = min(next_clock,notifications[0][0])
        if worker_starts:
            next_clock=min(next_clock,worker_starts[0][0])
        if receipt_events:
            next_clock=min(next_clock,receipt_events[0][0])
        current_rates = {}
        for owner,state in active.items():
            rate,key = rate_for(owner,state); current_rates[owner] = rate,key
            macro = state['plan'].macros[state['macro_index']]
            target = (macro.marks[state['next_mark']].work_ns
                if state['next_mark'] < len(macro.marks) else macro.reference_work_ns)
            next_clock = min(next_clock,clock+(target-state['age'])/rate)
        if next_clock <= clock:
            raise ValueError('Nonadvancing exact response clock')
        for owner,(rate,key) in current_rates.items():
            state = active[owner]; previous = state['age']
            state['age'] += (next_clock-clock)*rate
            exposure.append(dict(owner=owner,request_id=state['request_id'],
                macro_id=state['plan'].macros[state['macro_index']].identity,
                left_ns=clock,right_ns=next_clock,rate=rate,evidence_key=key,
                previous_virtual_age_ns=previous,new_virtual_age_ns=state['age']))
        clock = next_clock
    latencies = defaultdict(list)
    for request_id,received in receipts.items():
        row = cohort[request_id]
        latencies[row['task']].append(received-row['arrival_ns'])
    return encode(dict(schema=SCHEMA,status='EXPLICIT_JOINT_WORLD_CPU_COMPOSITION_ONLY',
        clock_domain='exact_rational_ns_reference_not_native_integer_timestamps',
        visibility_endpoint='parent_processing',dispatch_contract='per_owner_FIFO_after_READY_processing',
        dispatch_contract_matches_original_sender=False,
        original_window=[start,end],original_drain_end_ns=drain,cutoff_ns=cutoff,
        offered_requests=len(cohort),parent_dispatches=dispatches,starts=starts,RESULT_receipts=receipts,
        E2E_endpoint='scheduled_arrival_to_RESULT_reader_receipt',RESULT_processed=result_processed,
        READY_occurrences=ready_occurrences,task_E2E_latency_arrays_ns=dict(latencies),
        unfinished_request_ids=[key for key in cohort if key not in receipts],
        future_public_arrivals=[dict(arrival_ns=a,request_id=r) for a,_,r in sorted(arrivals)],
        pending_worker_starts=[dict(at_ns=at,owner=owner,request_id=r) for at,owner,r,_ in sorted(worker_starts)],
        pending_RESULT_reader_receipts=[dict(at_ns=at,request_id=r) for at,_,r in sorted(receipt_events)],
        per_owner_FIFO={owner:list(rows) for owner,rows in sorted(pending.items())},
        occupied_owners=occupied,active_response_states={owner:dict(request_id=s['request_id'],
            macro_id=s['plan'].macros[s['macro_index']].identity,virtual_age_ns=s['age'])
            for owner,s in active.items()},visible_macro_evidence=visible,
        processed_phase_occurrence_watermarks=phase_watermarks,
        processed_owner_issuance_watermarks=processed_issuance_watermarks,
        pending_notifications=[dict(available_ns=at,body=body) for at,_,body in sorted(notifications)],
        trace=trace,response_exposure=exposure,maximum_parallel_handlers=maximum_parallel,
        consumed_public_key_joint_draws=consumed_draws,
        global_admission_cap=None,context_lifetime_is_response_service=False,
        query_success_ends_macro=False,response_and_notifications_jointly_supplied=True,
        internal_work_template_qualified=False,joint_law_fitted=False,
        owner_issuance_order='explicit_joint_world_generated_not_certified_native_sequence',
        native_assimilation_eligible=False,main_queue_provider_eligible=False,
        six_item_resource_predictions='UNKNOWN_NO_RESOURCE_PROVIDER',
        independent_prediction_accuracy_validated=False,validation_labels_read=False,GPU_started=False))
