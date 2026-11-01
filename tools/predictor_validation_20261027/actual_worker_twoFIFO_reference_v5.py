"""Isolated owner/phase CPU two-FIFO reference with exact continuation.

Checkpoints contain generated/supplied latent worlds, not inferred native state.
All transport components remain explicit. No missing state is silently zero.
"""
import copy
from dataclasses import asdict
import hashlib
import json
from collections import defaultdict, deque
from fractions import Fraction
import heapq
from gsdeploy.forecast_errors import UnknownEvidence
from .macro_response_queue_reference_v1 import rational, validate_plan, encode
from .protocol_timing_slack_v3 import TimingStations, checked as checked_slacks
from .worker_stdin_fifo_contract_v1 import sender_fifo, worker_fifo_START, TransportDraw
SCHEMA='gsdeploy.actual_worker_twoFIFO_cpu_reference.v5'
STATE_SCHEMA='gsdeploy.actual_worker_twoFIFO_explicit_continuation.v5'
STATE_FIELDS=frozenset(('arrivals','pending','active','occupied','visible','notifications',
    'worker_starts','receipt_events','protocol_readers','sender_events','sender_available',
    'worker_release','starts','dispatches','ready_occurrences','receipts','result_processed',
    'trace','exposure','sequence','producer_next','maximum_parallel','maximum_actual_parallel',
    'phase_watermarks','owner_issued','processed_issuance_watermarks','consumed_draws',
    'producer_last_reader','stations','original_initial_protocol_state',
    'original_initial_sender_state','original_initial_worker_state'))


def decode(value):
    """Decode only exact reference rationals, never coerce floats to clocks."""
    if isinstance(value,dict):
        if set(value)=={'numerator','denominator'}:
            if (type(value['numerator']) is not int or type(value['denominator']) is not int
                    or value['denominator']<=0):raise ValueError('Invalid exact continuation rational')
            return Fraction(value['numerator'],value['denominator'])
        return {k:decode(v) for k,v in value.items()}
    if isinstance(value,list):return [decode(v) for v in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(encode(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _coalesce(rows):
    """Remove arbitrary pause/integration cuts without changing work or clocks."""
    grouped=defaultdict(list)
    for row in rows:grouped[row['owner']].append(dict(row))
    result=[]
    identity=('request_id','macro_id','actual_phase','task','pool_id',
              'public_key_arrival_ordinal','rate','evidence_key')
    for values in grouped.values():
        merged=[]
        for row in values:
            prior=merged[-1] if merged else None
            if (prior is not None and all(prior[k]==row[k] for k in identity)
                    and prior['right_ns']==row['left_ns']
                    and prior['new_virtual_age_ns']==row['previous_virtual_age_ns']):
                prior.update(right_ns=row['right_ns'],new_virtual_age_ns=row['new_virtual_age_ns'])
            else:merged.append(row)
        result.extend(merged)
    return sorted(result,key=lambda r:(r['left_ns'],r['owner'],r['request_id'],r['macro_id']))


def _check_checkpoint(state,fingerprint,start,cutoff,selected_plans,owners,owner_producers,cohort):
    """Require a complete causal-prefix world; this does not authenticate native facts."""
    if (type(state) is not dict or set(state)!={'schema','at_ns','input_sha256','payload','qualification'}
            or state.get('schema')!=STATE_SCHEMA or state.get('input_sha256')!=fingerprint
            or state.get('qualification')!='EXPLICIT_LATENT_CPU_WORLD_NOT_NATIVE_ASSIMILATION'):
        raise UnknownEvidence('UNKNOWN_COMPLETE_BOUND_EXPLICIT_CONTINUATION_STATE')
    at=rational(state['at_ns']);payload=state['payload']
    if not start<=at<=cutoff:raise UnknownEvidence('UNKNOWN_INITIAL_STATE_FROM_FUTURE_OR_OUTSIDE_WORLD')
    if type(payload) is not dict or set(payload)!=STATE_FIELDS:
        raise UnknownEvidence('UNKNOWN_MISSING_CONTINUATION_COMPONENTS')
    maps=('pending','active','occupied','visible','sender_available','worker_release','starts','dispatches',
          'ready_occurrences','receipts','result_processed','producer_next','phase_watermarks','owner_issued',
          'processed_issuance_watermarks','producer_last_reader','stations')
    arrays=('arrivals','notifications','worker_starts','receipt_events','protocol_readers','sender_events',
            'trace','exposure','consumed_draws')
    if (any(type(payload[k]) is not dict for k in maps)
            or any(type(payload[k]) is not list for k in arrays)
            or any(type(payload[k]) is not int or payload[k]<0 for k in
                   ('sequence','maximum_parallel','maximum_actual_parallel'))):
        raise UnknownEvidence('UNKNOWN_CONTINUATION_COMPONENT_TYPES')
    for key in ('producer_next','owner_issued','processed_issuance_watermarks'):
        if any(type(value) is not int or value<0 for value in payload[key].values()):
            raise UnknownEvidence('UNKNOWN_CONTINUATION_PRODUCER_COUNTERS')
    for key in ('sender_available','worker_release'):
        if set(payload[key])!=owners:raise UnknownEvidence('UNKNOWN_CONTINUATION_OWNER_FIFO_STATE')
        for value in payload[key].values():rational(value)
    if set(payload['stations'])!={'reader_free','collector_free','last_collector_arrival'}:
        raise UnknownEvidence('UNKNOWN_CONTINUATION_PROTOCOL_STATIONS')
    for value in [*payload['stations']['reader_free'].values(),payload['stations']['collector_free'],
                  payload['stations']['last_collector_arrival']]:rational(value)
    for owner,state_row in payload['active'].items():
        rid=state_row.get('request_id');plan=selected_plans.get(rid)
        if (set(state_row)!={'request_id','macro_index','age','next_mark'} or owner not in owners
                or plan is None or payload['occupied'].get(owner)!=rid
                or type(state_row['macro_index']) is not int or not 0<=state_row['macro_index']<len(plan.macros)):
            raise UnknownEvidence('UNKNOWN_ACTIVE_CONTINUATION_COORDINATES')
        macro=plan.macros[state_row['macro_index']];age=rational(state_row['age'])
        index=state_row['next_mark']
        if (type(index) is not int or not 0<=index<=len(macro.marks)
                or age>=macro.reference_work_ns
                or (index and macro.marks[index-1].work_ns>age)
                or (index<len(macro.marks) and macro.marks[index].work_ns<=age)):
            raise UnknownEvidence('UNKNOWN_ACTIVE_VIRTUAL_AGE_OR_MARK_CURSOR')
    for key in ('starts','dispatches','ready_occurrences','receipts','result_processed','phase_watermarks'):
        if any(rational(value)>at for value in payload[key].values()):
            raise UnknownEvidence('UNKNOWN_INITIAL_STATE_CONTAINS_FUTURE_COMPLETED_FACT')
    for row in payload['visible'].values():
        if rational(row['occurrence_ns'])>at:raise UnknownEvidence('UNKNOWN_FUTURE_VISIBLE_PHASE')
    issued=defaultdict(int);owner_counts=defaultdict(int);last_reader={}
    for row in payload['trace']:
        # Future scheduled START/reader/dequeue/processing is a latent queue
        # prediction only. Its triggering occurrence must already be issued.
        kind=row['kind']
        trigger=(row.get('available_ns') if kind=='processed_notification' else
                 row.get('reader_ns') if kind=='protocol_reader_available' else
                 row.get('occurrence_ns',row.get('at_ns')))
        if trigger is not None and rational(trigger)>at:
            raise UnknownEvidence('UNKNOWN_INITIAL_TRACE_FROM_FUTURE')
        if kind=='latent_occurrence':
            producer=row['producer_id'];owner=row['owner']
            if (owner_producers.get(owner)!=producer or row['producer_sequence']!=issued[producer]
                    or row['owner_issuance_ordinal']!=owner_counts[owner]):
                raise UnknownEvidence('UNKNOWN_CONTINUATION_SOURCE_PRODUCER_SEQUENCE')
            issued[producer]+=1;owner_counts[owner]+=1;last_reader[producer]=row['reader_receipt_ns']
    if (dict(issued)!=payload['producer_next'] or dict(owner_counts)!=payload['owner_issued']
            or last_reader!=payload['producer_last_reader'] or sum(issued.values())!=payload['sequence']):
        raise UnknownEvidence('UNKNOWN_CONTINUATION_PRODUCER_LIFETIME_STATE')
    if any(value>=payload['owner_issued'].get(owner,0) for owner,value in payload['processed_issuance_watermarks'].items()):
        raise UnknownEvidence('UNKNOWN_UNISSUED_PROCESSED_WATERMARK')
    reserved={owner:rid for _,owner,rid in payload['worker_starts']}
    if (set(reserved)&set(payload['active']) or
            {**reserved,**{owner:r['request_id'] for owner,r in payload['active'].items()}}!=payload['occupied']):
        raise UnknownEvidence('UNKNOWN_CONTINUATION_WORKER_RESERVATION')
    seen=set(payload['occupied'].values())
    for owner,rows in payload['pending'].items():
        if owner not in owners or type(rows) is not list:raise UnknownEvidence('UNKNOWN_CONTINUATION_OWNER_QUEUE')
        for rid in rows:
            if rid in seen or rid not in cohort or cohort[rid]['owner']!=owner:
                raise UnknownEvidence('UNKNOWN_CONTINUATION_DUPLICATE_OR_FOREIGN_FIFO_HEAD')
            seen.add(rid)
    for row in payload['exposure']:
        if rational(row['right_ns'])>at:raise UnknownEvidence('UNKNOWN_INITIAL_EXPOSURE_FROM_FUTURE')
    for queue,arity in (('notifications',3),('protocol_readers',4)):
        for item in payload[queue]:
            if len(item)!=arity or rational(item[0])<=at:
                raise UnknownEvidence('UNKNOWN_NONADVANCING_PENDING_PROTOCOL_STATE')
            body=item[2]
            if (rational(body['occurrence_ns'])>at or owner_producers.get(body['owner'])!=body['producer_id']
                    or payload['producer_next'].get(body['producer_id'],0)<=body['producer_sequence']
                    or payload['owner_issued'].get(body['owner'],0)<=body['owner_issuance_ordinal']):
                raise UnknownEvidence('UNKNOWN_UNISSUED_OR_FUTURE_PENDING_PROTOCOL_MESSAGE')
    for key in ('arrivals','sender_events','worker_starts','receipt_events'):
        if any(rational(row[0])<=at for row in payload[key]):
            raise UnknownEvidence('UNKNOWN_NONADVANCING_CONTINUATION_EVENT')
    return at,payload

def simulate(requests, plans, rates, *, window_start_ns, window_end_ns,
             drain_end_ns, cutoff_ns=None, producer_fifo=(), protocol_slacks=None,
             transport_draws=None, initial_sender_available_ns=None,
             initial_worker_read_release_ns=None, initial_protocol_state=None,
             continuation_state=None):
    """Compose public-key/ordinal supplied draws without historical ID lookup.

    Transport draws use (owner,task,pool_id,arrival ordinal within that key).
    Plans use the same key and ordinal. Worker release is actual READY plus the
    next draw's explicit continuation; parent processing controls visibility.
    A fresh boundary needs every original explicit initial station. A resumed
    boundary needs the complete checkpoint emitted below; incomplete/future
    native snapshots are not accepted as already inferred latent state.
    """
    if transport_draws is None or (continuation_state is None and
            (initial_sender_available_ns is None or initial_worker_read_release_ns is None)):
        raise UnknownEvidence('UNKNOWN_EXPLICIT_TRANSPORT_DRAWS_OR_INITIAL_FIFO_STATE')
    if initial_protocol_state is None and continuation_state is None:
        raise UnknownEvidence('UNKNOWN_EXPLICIT_INITIAL_PROTOCOL_STATIONS')
    if protocol_slacks is None:
        raise ValueError("Explicit joint protocol-slack mapping required")
    if continuation_state is not None:
        continuation_state=decode(copy.deepcopy(continuation_state))
        candidate=continuation_state.get('payload',{}) if isinstance(continuation_state,dict) else {}
        if not set(STATE_FIELDS)<=set(candidate):
            raise UnknownEvidence('UNKNOWN_MISSING_CONTINUATION_COMPONENTS')
        supplied=(initial_sender_available_ns,initial_worker_read_release_ns,initial_protocol_state)
        original=(candidate['original_initial_sender_state'],candidate['original_initial_worker_state'],
                  candidate['original_initial_protocol_state'])
        if any(value is not None and _digest(value)!=_digest(bound) for value,bound in zip(supplied,original)):
            raise UnknownEvidence('UNKNOWN_CONTINUATION_ORIGINAL_INITIAL_STATE_BINDING')
        initial_sender_available_ns,initial_worker_read_release_ns,initial_protocol_state=original
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
        if (type(key) is not tuple or len(key) != 4 or any(type(x) is not str or not x for x in key[:2])
                or any(type(x) is not int for x in key[2:]) or not 0 <= key[3] <= key[2]):
            raise ValueError('Explicit owner, semantic phase and visible-evidence count rate key required')
        checked_rates[key] = rational(rate, positive=True)
    fifo = frozenset(producer_fifo)
    # Templates and lookup rates are snapshotted before any execution mutation.
    plans = dict(plans)
    active = {}; occupied = {}; visible = {}; notifications = []
    worker_starts = []; receipt_events = []; protocol_readers = []
    if set(initial_sender_available_ns)!=owners or set(initial_worker_read_release_ns)!=owners:
        raise ValueError('Explicit initial FIFO state required for every offered owner')
    if set(initial_protocol_state)!={'reader_free','collector_free','last_collector_arrival','pending_messages'} or initial_protocol_state['pending_messages'] not in ((),[]):
        raise UnknownEvidence('EXPLICIT_INITIAL_PROTOCOL_STATIONS_AND_EMPTY_PENDING_REQUIRED')
    stations = TimingStations(start)
    stations.reader_free={key:rational(value) for key,value in initial_protocol_state['reader_free'].items()}
    stations.collector_free=rational(initial_protocol_state['collector_free'])
    stations.last_collector_arrival=rational(initial_protocol_state['last_collector_arrival'])
    if min([stations.collector_free,stations.last_collector_arrival,*stations.reader_free.values()])<start:
        raise ValueError('Initial protocol station clocks cannot precede world start')
    worker_release={owner:rational(at) for owner,at in initial_worker_read_release_ns.items()}
    sender_available={owner:rational(at) for owner,at in initial_sender_available_ns.items()}
    sender_schedule={}; selected_plans={}; public_ordinals={}; sender_events=[]
    owner_producers={}
    ordered=[]; offered_ordinals=defaultdict(int)
    for arrival,index,request_id in sorted(arrivals):
        row=cohort[request_id];key=row['owner'],row['task'],row['pool_id']
        ordinal=offered_ordinals[key];offered_ordinals[key]+=1;draw_key=(*key,ordinal)
        if draw_key not in transport_draws or type(transport_draws[draw_key]) is not TransportDraw:
            raise UnknownEvidence('UNKNOWN_EXPLICIT_PUBLIC_KEY_ORDINAL_TRANSPORT_DRAW')
        draw=transport_draws[draw_key]
        enqueue=arrival+rational(draw.enqueue_after_arrival_ns)
        ordered.append((enqueue,index,request_id,draw))
        supplied=plans[key];plan=supplied[ordinal] if type(supplied) is tuple else supplied
        if plan.dispatch_to_worker_start_ns!=0:
            raise UnknownEvidence('PLAN_DISPATCH_DELAY_CONFLICTS_WITH_EXPLICIT_TRANSPORT_DECOMPOSITION')
        producer=plan.ready_producer_id
        if (any(mark.producer_id!=producer for macro in plan.macros for mark in macro.marks)
                or owner_producers.get(row['owner'],producer)!=producer):
            raise UnknownEvidence('SOURCE_WORKER_REQUIRES_ONE_STDOUT_PRODUCER_PER_OWNER_LIFETIME')
        owner_producers[row['owner']]=producer
        selected_plans[request_id]=plan;public_ordinals[request_id]=ordinal
    if len(set(owner_producers.values()))!=len(owner_producers):
        raise UnknownEvidence('DISTINCT_WORKER_OWNERS_REQUIRE_DISTINCT_STDOUT_PRODUCERS')
    fifo=frozenset(owner_producers.values())|fifo
    for sender_sequence,(enqueue,index,request_id,draw) in enumerate(sorted(ordered)):
        row=cohort[request_id];owner=row['owner']
        send=sender_fifo(row['arrival_ns'],sender_available[owner],draw)
        sender_available[owner]=send['parent_write_record_ns'];sender_schedule[request_id]=send
        heapq.heappush(sender_events,(send['parent_submit_ns'],sender_sequence,request_id))
    transport_keys={(*((r['owner'],r['task'],r['pool_id'])),public_ordinals[rid]) for rid,r in cohort.items()}
    if set(transport_draws)!=transport_keys:
        raise ValueError('Exactly one transport draw per offered public-key ordinal required')
    starts = {}; dispatches = {}; ready_occurrences = {}; receipts = {}; result_processed = {}
    trace = []; exposure = []
    sequence = 0; producer_next = defaultdict(int); producer_last_available = {}
    maximum_parallel = 0; maximum_actual_parallel=0; clock = start; phase_watermarks = {}
    owner_issued = defaultdict(int); processed_issuance_watermarks = {}
    draw_cursors = defaultdict(int); consumed_draws = []
    producer_last_reader = {}

    # Fingerprint includes the immutable supplied world law/calendar; no labels
    # or mutable future response state are inputs. Continuation cannot change a
    # future draw, semantic rate, producer binding or original initial station.
    fingerprint=_digest(dict(requests=requests,plans=[dict(key=list(key),plans=[asdict(p) for p in
        (value if type(value) is tuple else (value,))]) for key,value in sorted(plans.items())],
        rates=[dict(key=list(key),value=value) for key,value in sorted(checked_rates.items())],
        protocol_slacks=[dict(request_id=rid,slacks=protocol_slacks.get(plan)) for rid,plan in sorted(selected_plans.items())],
        transport=[dict(key=list(key),draw=asdict(value)) for key,value in sorted(transport_draws.items())],
        original_initial_sender=initial_sender_available_ns,original_initial_worker=initial_worker_read_release_ns,
        original_initial_protocol=initial_protocol_state,producer_fifo=sorted(fifo),window=[start,end,drain]))
    if continuation_state is not None:
        clock,saved=_check_checkpoint(continuation_state,fingerprint,start,cutoff,selected_plans,owners,owner_producers,cohort)
        arrivals=[tuple(row) for row in saved['arrivals']]
        pending=defaultdict(deque,{owner:deque(rows) for owner,rows in saved['pending'].items()})
        active={owner:dict(row,plan=selected_plans[row['request_id']]) for owner,row in saved['active'].items()}
        occupied=saved['occupied'];visible=saved['visible'];notifications=[tuple(row) for row in saved['notifications']]
        worker_starts=[(*row,selected_plans[row[2]]) for row in saved['worker_starts']]
        receipt_events=[tuple(row) for row in saved['receipt_events']]
        protocol_readers=[tuple(row) for row in saved['protocol_readers']]
        sender_events=[tuple(row) for row in saved['sender_events']]
        sender_available=saved['sender_available'];worker_release=saved['worker_release']
        starts=saved['starts'];dispatches=saved['dispatches'];ready_occurrences=saved['ready_occurrences']
        receipts=saved['receipts'];result_processed=saved['result_processed']
        trace=saved['trace'];exposure=saved['exposure'];sequence=saved['sequence']
        producer_next=defaultdict(int,saved['producer_next']);maximum_parallel=saved['maximum_parallel']
        maximum_actual_parallel=saved['maximum_actual_parallel'];phase_watermarks=saved['phase_watermarks']
        owner_issued=defaultdict(int,saved['owner_issued']);processed_issuance_watermarks=saved['processed_issuance_watermarks']
        consumed_draws=saved['consumed_draws'];producer_last_reader=saved['producer_last_reader']
        stations.reader_free=saved['stations']['reader_free'];stations.collector_free=saved['stations']['collector_free']
        stations.last_collector_arrival=saved['stations']['last_collector_arrival']
        # Exposure rows are JSON arrays after restoration, but fresh identity
        # keys are tuples; normalize once so pause cuts coalesce identically.
        for row in exposure:row['evidence_key']=tuple(row['evidence_key'])
        for queue in (arrivals,notifications,worker_starts,receipt_events,protocol_readers,sender_events):heapq.heapify(queue)

    def notify(owner, request_id, action, delay, producer, identity, phase=None, mark=None,
               macro_id=None,reader_delay_override=None):
        nonlocal sequence
        plan=active[owner]['plan']
        if plan not in protocol_slacks or identity not in protocol_slacks[plan]:
            raise UnknownEvidence('UNKNOWN_JOINT_PROTOCOL_SLACK_MARK')
        slacks=checked_slacks(protocol_slacks[plan][identity])
        reader=stations.issue(clock,producer,slacks)
        producer_sequence = producer_next[producer]; producer_next[producer] += 1
        issuance = owner_issued[owner]; owner_issued[owner] += 1
        if producer in fifo and reader < producer_last_reader.get(producer,start):
            raise UnknownEvidence('PROTOCOL_READER_RECURRENCE_VIOLATES_FIFO')
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
        heapq.heappush(protocol_readers, (reader, sequence, body, slacks))
        if action == 'result':
            heapq.heappush(receipt_events,(reader,sequence,request_id))
        trace.append(dict(kind='latent_occurrence', **body))

    def dispatch():
        nonlocal maximum_parallel
        for owner in sorted(owners):
            if owner in occupied or not pending[owner]:
                continue
            request_id=pending[owner].popleft();row=cohort[request_id]
            plan=selected_plans[request_id];ordinal=public_ordinals[request_id]
            occupied[owner]=request_id
            at=worker_fifo_START(worker_release[owner],sender_schedule[request_id])
            if at<clock:
                raise ValueError('Worker START recurrence cannot precede current causal boundary')
            heapq.heappush(worker_starts,(at,owner,request_id,plan))
            consumed_draws.append(dict(owner=owner,task=row['task'],pool_id=row['pool_id'],
                public_key_arrival_ordinal=ordinal,request_id=request_id,
                draw_mode='explicit_joint_response_and_transport_draw'))
            trace.append(dict(kind='worker_head_reserved',owner=owner,request_id=request_id,
                              at_ns=clock,scheduled_START_ns=at))
        maximum_parallel=max(maximum_parallel,len(occupied))

    def rate_for(owner, state):
        phase = state['plan'].macros[state['macro_index']].phase
        peer = [value['phase'] for alias,value in visible.items() if alias != owner]
        key = owner, phase, len(peer), sum(value == phase for value in peer)
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
                trace.append(dict(kind='scheduled_arrival',request_id=request_id,at_ns=clock)); changed = True
            while sender_events and sender_events[0][0] <= clock:
                at,_,request_id=heapq.heappop(sender_events)
                owner=cohort[request_id]['owner'];pending[owner].append(request_id)
                dispatches[request_id]=at
                trace.append(dict(kind='parent_submit',owner=owner,request_id=request_id,at_ns=at))
                changed=True
            while worker_starts and worker_starts[0][0] <= clock:
                at,owner,request_id,plan=heapq.heappop(worker_starts)
                active[owner]=dict(request_id=request_id,plan=plan,macro_index=0,
                    age=Fraction(0),next_mark=0)
                starts[request_id]=at
                maximum_actual_parallel=max(maximum_actual_parallel,len(active))
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
                        worker_release[owner]=clock
                        if occupied.get(owner)!=state['request_id']:
                            raise ValueError('Actual READY must release its own worker handler')
                        occupied.pop(owner)
                        del active[owner]
                    changed = True
            while receipt_events and receipt_events[0][0] <= clock:
                received,_,request_id=heapq.heappop(receipt_events)
                if request_id in receipts:
                    raise ValueError('Duplicate generated RESULT reader receipt')
                receipts[request_id]=received
                trace.append(dict(kind='RESULT_reader_receipt',request_id=request_id,at_ns=received))
                changed=True
            while protocol_readers and protocol_readers[0][0] <= clock:
                reader,seq,body,slacks=heapq.heappop(protocol_readers)
                dequeue,processing=stations.accept_reader(reader,slacks)
                body.update(collector_dequeued_ns=dequeue)
                heapq.heappush(notifications,(processing,seq,body))
                trace.append(dict(kind='protocol_reader_available',reader_ns=reader,
                    dequeue_ns=dequeue,processing_ns=processing,**body))
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
                    phase_watermarks[owner] = max(phase_watermarks.get(owner,start),body['occurrence_ns'])
                    processed_issuance_watermarks[owner] = max(
                        processed_issuance_watermarks.get(owner,-1),body['owner_issuance_ordinal'])
                    if visible.get(owner,{}).get('request_id') == request_id:
                        visible.pop(owner)
                # Query never clears a response macro or changes its phase.
                changed = True
            before=len(consumed_draws);dispatch();changed|=len(consumed_draws)!=before
        if clock == cutoff:
            break
        next_clock = cutoff
        if arrivals:
            next_clock = min(next_clock,arrivals[0][0])
        if sender_events:
            next_clock=min(next_clock,sender_events[0][0])
        if notifications:
            next_clock = min(next_clock,notifications[0][0])
        if protocol_readers:
            next_clock = min(next_clock,protocol_readers[0][0])
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
                actual_phase=state['plan'].macros[state['macro_index']].phase,
                task=cohort[state['request_id']]['task'],pool_id=cohort[state['request_id']]['pool_id'],
                public_key_arrival_ordinal=public_ordinals[state['request_id']],
                left_ns=clock,right_ns=next_clock,rate=rate,evidence_key=key,
                previous_virtual_age_ns=previous,new_virtual_age_ns=state['age']))
        clock = next_clock
    latencies = defaultdict(list)
    for request_id,received in receipts.items():
        row = cohort[request_id]
        latencies[row['task']].append(received-row['arrival_ns'])
    exposure=_coalesce(exposure)
    checkpoint=dict(schema=STATE_SCHEMA,at_ns=clock,input_sha256=fingerprint,
        qualification='EXPLICIT_LATENT_CPU_WORLD_NOT_NATIVE_ASSIMILATION',payload=dict(
            arrivals=sorted(arrivals),pending={owner:list(rows) for owner,rows in pending.items()},
            active={owner:{key:value for key,value in state.items() if key!='plan'} for owner,state in active.items()},
            occupied=occupied,visible=visible,notifications=sorted(notifications),
            worker_starts=[(at,owner,rid) for at,owner,rid,_ in sorted(worker_starts)],
            receipt_events=sorted(receipt_events),protocol_readers=sorted(protocol_readers),
            sender_events=sorted(sender_events),sender_available=sender_available,worker_release=worker_release,
            starts=starts,dispatches=dispatches,ready_occurrences=ready_occurrences,receipts=receipts,
            result_processed=result_processed,trace=trace,exposure=exposure,sequence=sequence,
            producer_next=dict(producer_next),maximum_parallel=maximum_parallel,
            maximum_actual_parallel=maximum_actual_parallel,phase_watermarks=phase_watermarks,
            owner_issued=dict(owner_issued),processed_issuance_watermarks=processed_issuance_watermarks,
            consumed_draws=consumed_draws,producer_last_reader=producer_last_reader,
            stations=dict(reader_free=stations.reader_free,collector_free=stations.collector_free,
                last_collector_arrival=stations.last_collector_arrival),
            original_initial_protocol_state=initial_protocol_state,
            original_initial_sender_state=initial_sender_available_ns,
            original_initial_worker_state=initial_worker_read_release_ns))
    if set(checkpoint['payload'])!=STATE_FIELDS:raise AssertionError('Incomplete continuation implementation')
    return encode(dict(schema=SCHEMA,status='EXPLICIT_JOINT_WORLD_CPU_COMPOSITION_ONLY',
        continuation_state=checkpoint,
        rate_interface='owner_semantic_phase_processed_peer_count_same_semantic_phase_count',
        continuation_is_explicit_latent_world_not_native_snapshot=True,
        native_history_assimilation_implemented=False,
        clock_domain='exact_rational_ns_reference_not_native_integer_timestamps',
        visibility_endpoint='parent_processing',dispatch_contract='parent_sender_FIFO_and_actual_worker_stdin_FIFO',
        dispatch_contract_matches_original_sender=True,
        source_FIFO_order_matched_under_explicit_transport_and_initial_state=True,
        READY_processing_gates_worker_START=False,parent_write_record_gates_worker_START=False,
        transport_decomposition_fitted=False,busy_transport_transfer_qualified=False,
        public_offered_requests=[dict(row) for row in requests],
        public_key_arrival_ordinals=public_ordinals,transport_schedule=sender_schedule,
        supplied_transport_draws=[dict(owner=k[0],task=k[1],pool_id=k[2],public_key_arrival_ordinal=k[3],
            **{field:getattr(draw,field) for field in draw.__dataclass_fields__})
            for k,draw in sorted(transport_draws.items())],
        initial_protocol_state=initial_protocol_state,
        worker_actual_release_ns=worker_release,
        pending_parent_sender=[dict(request_id=rid,owner=cohort[rid]['owner'],**send)
            for rid,send in sender_schedule.items() if send['parent_submit_ns']>cutoff],
        pending_parent_enqueue=[rid for rid,send in sender_schedule.items()
            if send['enqueued_ns']>cutoff],
        enqueued_waiting_parent_sender=[rid for rid,send in sender_schedule.items()
            if send['enqueued_ns']<=cutoff<send['parent_submit_ns']],
        parent_sends_not_yet_recorded=[rid for rid,send in sender_schedule.items()
            if send['parent_submit_ns']<=cutoff<send['parent_write_record_ns']],
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
            macro_id=s['plan'].macros[s['macro_index']].identity,
            actual_phase=s['plan'].macros[s['macro_index']].phase,
            task=cohort[s['request_id']]['task'],pool_id=cohort[s['request_id']]['pool_id'],
            public_key_arrival_ordinal=public_ordinals[s['request_id']],virtual_age_ns=s['age'])
            for owner,s in active.items()},visible_macro_evidence=visible,
        processed_phase_occurrence_watermarks=phase_watermarks,
        processed_owner_issuance_watermarks=processed_issuance_watermarks,
        pending_notifications=[dict(available_ns=at,body=body) for at,_,body in sorted(notifications)],
        pending_protocol_reader_events=[dict(reader_ns=at,body=body) for at,_,body,_ in sorted(protocol_readers)],
        protocol_reader_free=stations.reader_free,protocol_collector_free=stations.collector_free,
        trace=trace,response_exposure=exposure,maximum_parallel_reserved_or_active_handlers=maximum_parallel,
        maximum_parallel_handlers=maximum_actual_parallel,
        consumed_public_key_joint_draws=consumed_draws,
        global_admission_cap=None,context_lifetime_is_response_service=False,
        query_success_ends_macro=False,response_and_notifications_jointly_supplied=True,
        internal_work_template_qualified=False,joint_law_fitted=False,
        owner_issuance_order='explicit_joint_world_generated_not_certified_native_sequence',
        protocol_timing_model='per_worker_reader_and_shared_serial_collector_slacks',
        protocol_slacks_are_identified_CPU_service=False,
        stdout_backpressure_mechanism_identified=False,
        worker_macro_rates_use_parent_visible_covariates_not_physical_SM_state=True,
        parent_collector_order_assumption='reader_availability_order_with_generated_sequence_ties',
        native_assimilation_eligible=False,main_queue_provider_eligible=False,
        six_item_resource_predictions='UNKNOWN_NO_RESOURCE_PROVIDER',
        independent_prediction_accuracy_validated=False,validation_labels_read=False,GPU_started=False))


def resource_timeline_view(reference):
    """Explicit structural adapter; preserves v5 origin and qualification flags.

    The resource timeline consumes actual response intervals only. No native
    dispatch, resource provider, or physical SM interpretation is certified.
    """
    from .macro_response_queue_reference_v1 import SCHEMA as interval_schema
    if reference.get('schema')!=SCHEMA:
        raise ValueError('Explicit v5 two-FIFO world required')
    view=copy.deepcopy(reference)
    for exposure in view['response_exposure']:
        key=exposure['evidence_key']
        if (len(key)!=4 or key[0]!=exposure['owner'] or key[1]!=exposure['actual_phase']):
            raise ValueError('Exact v5 owner/semantic phase rate key required')
        exposure['owner_phase_rate_key']=list(key)
        exposure['evidence_key']=list(key[1:])
    view.update(schema=interval_schema,origin_schema=SCHEMA,
        adapter='v5_actual_response_intervals_for_v1_resource_timeline',
        schema_adapter_confers_native_qualification=False)
    return view
