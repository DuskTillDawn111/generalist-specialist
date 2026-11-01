"""Exact paired-transport bounds in an explicitly certified single-owner world.

No latent distribution is fitted or selected as a prediction. Two extremizing
feasible schedules certify the componentwise attainable envelope, including
the actual producer reader FIFO and shared collector max-plus recurrences.
Outside the declared subdomain every offered request remains UNKNOWN.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math

from gsdeploy.forecast_errors import UnknownEvidence
from .macro_response_queue_reference_v1 import rational,validate_plan,encode
from .twoFIFO_paired_transport_v3 import PairedTransportConstraints,source_constraints,sender_schedule
from .protocol_timing_slack_v3 import checked as checked_slacks
from .macro_six_term_bridge_v1 import whole_score

SCHEMA='gsdeploy.twoFIFO_single_owner_transport_intervals.v1'
CERTIFICATE=dict(schema=SCHEMA+'.caller_subdomain_declaration',single_owner_full_world=True,
    no_external_visible_or_future_producers=True,initial_visible_peers_empty=True,
    transport_set='cartesian_product_of_per_request_paired_feasible_sets',
    response_work_and_protocol_slacks_fixed_within_draw=True)


def need(condition,reason):
    if not condition:raise UnknownEvidence('UNKNOWN_SINGLE_OWNER_INTERVAL_SUBDOMAIN:'+reason)


def p95(values):
    values=sorted(values);position=Fraction(95,100)*(len(values)-1)
    lo=position.numerator//position.denominator;hi=math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def _templates(requests,plans,rates,timing,transport,certificate,initial_sender,initial_worker,protocol,start,end):
    need(certificate==CERTIFICATE,'explicit_exact_subdomain_declaration_required')
    owners={r['owner'] for r in requests};need(len(owners)==1,'multiple_or_zero_owners')
    owner=next(iter(owners))
    need(set(initial_sender)==set(initial_worker)==owners,'exact_initial_owner_stations')
    need(type(protocol) is dict and set(protocol)=={'reader_free','collector_free','last_collector_arrival','pending_messages'}
        and protocol['pending_messages'] in ([],()),'unsettled_initial_protocol')
    for at in [*initial_sender.values(),*initial_worker.values(),*protocol['reader_free'].values(),
               protocol['collector_free'],protocol['last_collector_arrival']]:
        need(rational(at)>=start,'use_explicit_initial_clocks_at_or_after_origin')
    ordinals=defaultdict(int);rows=[];producers=set();offered_keys=set()
    ordered=sorted(enumerate(requests),key=lambda pair:(pair[1]['arrival_ns'],pair[0]))
    counts=defaultdict(int)
    for _,r in ordered:counts[(r['owner'],r['task'],r['pool_id'])]+=1
    for _,r in ordered:
        key=r['owner'],r['task'],r['pool_id'];ordinal=ordinals[key];ordinals[key]+=1
        need(key in plans,'missing_joint_plan')
        supplied=plans[key]
        need(type(supplied) is not tuple or len(supplied)==counts[key],'joint_plan_slot_denominator')
        plan=supplied[ordinal] if type(supplied) is tuple else supplied;validate_plan(plan)
        need(plan.dispatch_to_worker_start_ns==0,'nonzero_legacy_dispatch_delay')
        drawkey=(*key,ordinal);offered_keys.add(drawkey)
        need(drawkey in transport and type(transport[drawkey]) is PairedTransportConstraints,'paired_transport_required')
        draw=transport[drawkey];source=source_constraints(draw)
        producer=plan.ready_producer_id;producers.add(producer)
        work=Fraction(0);messages=[]
        for macro in plan.macros:
            ratekey=(owner,macro.phase,0,0)
            need(ratekey in rates,'missing_no_peer_rate')
            rate=rational(rates[ratekey],positive=True)
            for mark in macro.marks:
                need(mark.producer_id==producer,'multiple_stdout_producers')
                need(plan in timing and mark.identity in timing[plan],'missing_joint_protocol_slack')
                messages.append(dict(identity=mark.identity,action=mark.action,offset_ns=work+mark.work_ns/rate,
                    slacks=checked_slacks(timing[plan][mark.identity])))
            work+=macro.reference_work_ns/rate
        need(plan in timing and 'worker_READY' in timing[plan],'missing_READY_protocol_slack')
        messages.append(dict(identity='worker_READY',action='ready',offset_ns=work,
            slacks=checked_slacks(timing[plan]['worker_READY'])))
        rows.append(dict(public=dict(r),public_key_ordinal=ordinal,plan=plan,draw=draw,source=source,
            producer=producer,messages=messages,handler_duration_ns=work))
    need(set(transport)==offered_keys,'transport_offered_denominator')
    need(len(producers)==1,'owner_producer_lifetime_changed')
    producer=next(iter(producers))
    need(set(protocol['reader_free'])<=producers,'external_protocol_reader')
    need(protocol['last_collector_arrival']<=protocol['reader_free'].get(producer,start),
        'initial_collector_reader_order')
    # Public-key ordinals choose templates before enqueue order. Sender/worker
    # order is the same stable (enqueue,input-position) order as frozen v5_2.
    rows.sort(key=lambda row:(row['public']['arrival_ns']+row['draw'].enqueue_after_arrival_ns,
        next(i for i,r in enumerate(requests) if r['request_id']==row['public']['request_id'])))
    previous=rational(initial_sender[owner])
    for row in rows:
        row['send']=sender_schedule(row['public']['arrival_ns'],previous,row['draw'])
        previous=row['send']['parent_write_record_ns']
    return owner,producer,rows


def _extreme(rows,initial_READY,protocol,producer,start,upper):
    ready=rational(initial_READY);reader=rational(protocol['reader_free'].get(producer,start))
    collector=rational(protocol['collector_free']);result={};messages=[]
    for row in rows:
        s=row['send']['parent_submit_ns'];source=row['source'];q=source['previous_READY_minus_submit_ns']
        H=source['observed_submit_to_START_ns'];limit=min(source['submit_to_write_record_ns'],H)
        # For fixed R, max(R-s,u)-max(q,u) is monotone in u. Its
        # direction changes at R-s=q; endpoints give both exact extrema.
        candidates=[(s+max(ready-s,u)+H-max(q,u),u) for u in (Fraction(0),limit)]
        B,u=(max(candidates) if upper else min(candidates))
        ready=B+row['handler_duration_ns'];rid=row['public']['request_id'];receipt=None;processed=None
        for message in row['messages']:
            occurrence=B+message['offset_ns'];slack=message['slacks']
            reader=max(occurrence,reader)+slack['reader_slack_ns']
            dequeue=max(reader,collector)+slack['collector_dequeue_slack_ns']
            collector=dequeue+slack['collector_processing_slack_ns']
            messages.append(dict(request_id=rid,identity=message['identity'],action=message['action'],
                occurrence_ns=occurrence,reader_ns=reader,dequeue_ns=dequeue,processing_ns=collector))
            if message['action']=='result':receipt=reader;processed=collector
        result[rid]=dict(START_ns=B,READY_ns=ready,RESULT_reader_ns=receipt,RESULT_processed_ns=processed,
            extremizing_u_ns=u,paired_continuation_ns=H-max(q,u),
            sender_submit_ns=s,parent_write_record_ns=row['send']['parent_write_record_ns'])
    return dict(requests=result,protocol_messages=messages,
        witness_is_feasible_extremizer_not_predictive_latent_draw=True)


def evaluate(requests,plans,rates,*,protocol_slacks,transport_draws,domain_certificate,
             initial_sender_available_ns,initial_worker_read_release_ns,initial_protocol_state,
             window_start_ns,window_end_ns,drain_end_ns,scoring_plan=None,quality=None):
    """Full-cohort sharp bounds for one declared no-peer owner; no native IO.

    Protocol queues may accumulate arbitrary fixed nonnegative FIFO slacks.
    Their processing does not gate actual READY, sender submit or no-peer rates.
    Latent u coordinates are independent feasible-set coordinates, never random.
    """
    start,end,drain=map(rational,(window_start_ns,window_end_ns,drain_end_ns))
    if not start<end<=drain:raise ValueError('Original positive window and fixed drain required')
    requests=[dict(r) for r in requests];ids=[]
    for r in requests:
        if (set(r)!={'request_id','owner','task','pool_id','arrival_ns'}
            or any(type(r[k]) is not str or not r[k] for k in ('request_id','owner','task','pool_id'))
            or r['request_id'] in ids or not start<=rational(r['arrival_ns'])<end):
            raise ValueError('Exact distinct complete public cohort required')
        ids.append(r['request_id'])
    base=dict(schema=SCHEMA,offered_requests=len(requests),offered_request_ids=ids,
        latent_probability_law_identified=False,latent_probability_assigned=False,
        latent_point_used_as_forecast=False,endpoint_enumeration_used_as_multiowner_envelope=False,
        main_queue_provider_eligible=False,final_gain_eligible=False,full_U_bounds=None,
        full_U_unknown_reason='NO_QUALIFIED_RESOURCE_BOUNDS_OR_VERIFIED_RESOURCE_CANCELLATION',
        actual_native_quality_validated=False,validation_labels_read=False,GPU_started=False)
    try:
        owner,producer,rows=_templates(requests,plans,rates,protocol_slacks,transport_draws,domain_certificate,
            initial_sender_available_ns,initial_worker_read_release_ns,initial_protocol_state,start,end)
    except UnknownEvidence as exc:
        return dict(base,status='UNKNOWN_OUTSIDE_CERTIFIED_SINGLE_OWNER_SUBDOMAIN',reason=str(exc),
            requests=[dict(request_id=rid,START_ns_bounds=None,RESULT_reader_ns_bounds=None,reader_E2E_ns_bounds=None,
                fixed_drain_incomplete_indicator_bounds=[0,1]) for rid in ids],complete_cohort_P95_ns_bounds=None,
            all_offered_requests_retained=True)
    low=_extreme(rows,initial_worker_read_release_ns[owner],initial_protocol_state,producer,start,False)
    high=_extreme(rows,initial_worker_read_release_ns[owner],initial_protocol_state,producer,start,True)
    output=[]
    for r in requests:
        rid=r['request_id'];lo=low['requests'][rid];hi=high['requests'][rid]
        a,b=lo['RESULT_reader_ns'],hi['RESULT_reader_ns']
        output.append(dict(request_id=rid,START_ns_bounds=[lo['START_ns'],hi['START_ns']],
            READY_ns_bounds=[lo['READY_ns'],hi['READY_ns']],RESULT_reader_ns_bounds=[a,b],
            RESULT_processed_ns_bounds=[lo['RESULT_processed_ns'],hi['RESULT_processed_ns']],
            reader_E2E_ns_bounds=[a-r['arrival_ns'],b-r['arrival_ns']],
            completed_by_fixed_drain=('ALWAYS' if b<=drain else 'NEVER' if a>drain else 'DEPENDS_ON_FEASIBLE_TRANSPORT'),
            fixed_drain_incomplete_indicator_bounds=[int(a>drain),int(b>drain)]))
    always_complete=all(row['RESULT_reader_ns_bounds'][1]<=drain for row in output)
    may_complete=all(row['RESULT_reader_ns_bounds'][0]<=drain for row in output)
    result=dict(base,status='EXACT_SINGLE_OWNER_PAIRED_TRANSPORT_COHORT_ENVELOPE',requests=output,
        all_offered_requests_retained=True,domain_certificate=domain_certificate,
        lower_witness=low,upper_witness=high,simultaneously_attainable_componentwise_extrema=True,
        untruncated_model_complete_cohort_P95_ns_bounds=[p95([row['reader_E2E_ns_bounds'][i] for row in output]) for i in (0,1)],
        complete_cohort_P95_ns_bounds=([p95([row['reader_E2E_ns_bounds'][i] for row in output]) for i in (0,1)] if always_complete else None),
        original_fixed_drain_P95_defined_for_all_feasible_worlds=always_complete,
        original_fixed_drain_P95_defined_for_some_feasible_world=may_complete,
        fixed_drain_incomplete_fraction_bounds=[Fraction(sum(row['fixed_drain_incomplete_indicator_bounds'][i] for row in output),len(output)) for i in (0,1)],
        P95_not_inferred_from_completed_subset=True,resource_bounds_inferred=False)
    if (scoring_plan is None)!=(quality is None):raise ValueError('Score plan and frozen quality must be supplied together')
    if scoring_plan is not None:
        scores=[]
        for witness in (low,high):
            receipts={rid:encode(row['RESULT_reader_ns']) for rid,row in witness['requests'].items() if row['RESULT_reader_ns']<=drain}
            frame=dict(original_window=encode([start,end]),original_drain_end_ns=encode(drain),cutoff_ns=encode(drain),
                RESULT_receipts=receipts,unfinished_request_ids=[rid for rid in ids if rid not in receipts],offered_requests=len(ids))
            scores.append(whole_score(frame,requests,scoring_plan,quality)['score'])
        result.update(original_score_at_componentwise_extrema=scores,
            original_QTLP_term_bounds={k:[scores[0]['terms'][k],scores[1]['terms'][k]]
                if scores[0]['terms'][k] is not None and scores[1]['terms'][k] is not None else None for k in 'QTLP'},
            quality_from_frozen_training_expectations_not_native_outcome=True)
    return result
