"""Causal settled-prefix station seed; never infer active native response state.

Pure snapshot inspection is diagnostic only. Authenticated extraction accepts
only the new unperturbed v3 collector review and independently rederives it.
"""
import copy
import hashlib
import json
import tempfile
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from .stage_response_features_v1 import checked, receipt

SCHEMA = 'gsdeploy.twoFIFO_native_settled_history_initial.v1'
SEQUENCE_SCOPE = 'tracked_request_messages_since_capture_start_excluding_startup_READY_relative_offset'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def need(value, reason):
    if not value: raise UnknownEvidence('UNKNOWN_NATIVE_SETTLED_PREFIX:' + reason)


def clock(value):
    need(type(value) is int and value >= 0, 'integer_clock')
    return value


def from_snapshot(snapshot, *, cutoff_ns, forecast_origin_ns, owner_producers):
    """Use this immutable prefix alone; never fill from final request records.

    The startup stdout sequence is not captured. We expose an explicit relative
    tracked-message offset, rather than pretending it is a native wire counter.
    Every worker must have a processed handler READY in the supplied prefix.
    """
    snapshot=copy.deepcopy(snapshot);cut=clock(cutoff_ns);origin=clock(forecast_origin_ns)
    need(type(snapshot) is dict and clock(snapshot.get('snapshot_time_ns')) == cut <= origin, 'cutoff_origin_binding')
    need(snapshot.get('queued_runtime_events') == [] and snapshot.get('runtime_errors') == []
         and snapshot.get('empty_initial_state_assumed') is False, 'unsettled_or_unproved_empty')
    need(type(owner_producers) is dict and owner_producers and all(type(x) is str and x for
        x in [*owner_producers,*owner_producers.values()]) and len(set(owner_producers.values()))==len(owner_producers), 'producer_lifetimes')
    roster=snapshot.get('ready');rows=snapshot.get('request_records')
    need(type(roster) is dict and set(roster)==set(owner_producers) and type(rows) is list and rows, 'complete_owner_roster_and_history')
    ids=[r.get('request_id') for r in rows]
    need(all(type(x) is str and x for x in ids) and len(set(ids))==len(ids), 'distinct_registry')
    need(type(snapshot.get('pending_request_ids')) is list and len(snapshot['pending_request_ids'])==len(ids)
         and set(snapshot['pending_request_ids'])==set(ids), 'registry_not_unfinished_queue')
    messages={o:[] for o in roster};sends={o:[] for o in roster};readies={o:[] for o in roster}
    lifetime={}
    for owner,r in roster.items():
        need(type(r.get('pid')) is int and r['pid']>0 and r.get('clock')=='time.perf_counter_ns'
             and type(r.get('spec_sha256')) is str and len(r['spec_sha256'])==64, 'owned_worker_lifetime')
        lifetime[owner]=dict(pid=r['pid'],spec_sha256=r['spec_sha256'])
    def message(owner, event, perf_key='perf_ns'):
        need(type(event) is dict, 'message_present')
        clocks=[clock(event.get(k)) for k in (perf_key,'received_time_ns','collector_dequeued_time_ns','collector_processing_time_ns')]
        need(clocks==sorted(clocks) and clocks[-1]<=cut, 'all_four_message_clocks_at_or_before_cutoff')
        if 'pid' in event: need(event['pid']==roster[owner]['pid'], 'worker_pid_changed')
        messages[owner].append(tuple(clocks))
        return clocks
    for row in rows:
        owner=row.get('model');need(owner in roster, 'foreign_owner')
        need(row.get('status')=='completed' and row.get('worker_response',{}).get('success') is True, 'unfinished_or_failed_request')
        scheduled,enqueue,submit,write=[clock(row.get(k)) for k in
            ('scheduled_time_ns','enqueued_time_ns','actual_submit_time_ns','submit_write_completed_time_ns')]
        need(scheduled<=enqueue<=submit<=write<=cut, 'future_arrival_or_sender_write')
        events=row.get('execution_events');need(type(events) is list and events, 'missing_START_and_phase_history')
        event_clocks=[message(owner,e) for e in events]
        begin=clock(row.get('observed_worker_start_time_ns'))
        need(begin==clock(row['worker_response'].get('start_perf_ns')) and begin==event_clocks[0][0]
             and submit<=begin and events[0].get('pid')==roster[owner]['pid'], 'START_lifetime_binding')
        result=dict(perf_ns=row['worker_response'].get('end_perf_ns'),
            received_time_ns=row.get('response_reader_event_enqueue_time_ns'),
            collector_dequeued_time_ns=row.get('response_collector_dequeued_time_ns'),
            collector_processing_time_ns=row.get('response_collector_processing_time_ns'))
        rc=message(owner,result);ready=row.get('previous_handler_next_request_ready');qc=message(owner,ready)
        need(ready.get('pid')==roster[owner]['pid'] and ready.get('request_id')==row['request_id']
             and ready.get('clock')=='host_perf_counter_ns' and ready.get('boundary')=='previous_handler_next_request_ready'
             and ready.get('scope')=='after_response_flush_before_next_read' and row.get('completion_time_ns')==rc[1]
             and begin<=rc[0] and all(all(a<=b for a,b in zip(ec,rc)) for ec in event_clocks)
             and all(a<=b for a,b in zip(rc,qc)), 'RESULT_READY_chronology')
        sends[owner].append((submit,write));readies[owner].append((begin,qc[0]))
    for owner in roster:
        need(messages[owner] and readies[owner], 'no_handler_READY_dominating_missing_startup_clocks')
        ordered=sorted(messages[owner],key=lambda x:x[0])
        need(all(all(a<=b for a,b in zip(left,right)) for left,right in zip(ordered,ordered[1:])), 'stdout_FIFO_chronology')
        handlers=sorted(readies[owner]);need(all(left[1]<=right[0] for left,right in zip(handlers,handlers[1:])), 'overlapping_owner_handlers')
        sender=sorted(sends[owner]);need(all(left[1]<=right[0] for left,right in zip(sender,sender[1:])), 'sender_FIFO_chronology')
    counts={owner_producers[o]:len(messages[o]) for o in roster}
    owner_counts={o:len(messages[o]) for o in roster}
    latest_reader={owner_producers[o]:max(m[1] for m in messages[o]) for o in roster}
    processed=[m[3] for values in messages.values() for m in values]
    reader_arrivals=[m[1] for values in messages.values() for m in values]
    return dict(schema=SCHEMA,status='SETTLED_NATIVE_PREFIX_CONDITIONAL_REFERENCE_SEED_ONLY',
        cutoff_ns=cut,forecast_origin_ns=origin,snapshot_sha256=digest(snapshot),
        owner_producers=dict(owner_producers),worker_lifetimes=lifetime,
        initial_sender_available_ns={o:max(x[1] for x in sends[o]) for o in roster},
        initial_worker_read_release_ns={o:max(x[1] for x in readies[o]) for o in roster},
        initial_protocol_state=dict(reader_free=latest_reader,collector_free=max(processed),
            last_collector_arrival=max(reader_arrivals),pending_messages=[]),
        producer_next=counts,owner_issued=owner_counts,producer_last_reader=latest_reader,
        phase_watermarks={o:max(x[1] for x in readies[o]) for o in roster},
        processed_issuance_watermarks={o:n-1 for o,n in owner_counts.items()},sequence=sum(counts.values()),
        sequence_scope=SEQUENCE_SCOPE,missing_startup_sequence_assumption_explicit=True,
        prior_request_ids=ids,current_cohort_contains_prior_labels=False,
        prior_pending_request_ids_mean_record_registry=True,native_source_authenticated=False,
        trained_or_prediction_validated=False)


def extract(training_review, *, case_id, forecast_origin_ns, owner_producers):
    """Authenticate unperturbed v3 training review before any snapshot access."""
    from . import twoFIFO_owned_training_review_v3 as review_module
    ref=receipt(training_review);review=checked(ref)
    need(review.get('schema')==review_module.SCHEMA and review.get('validation_labels_read') is False
         and review.get('status')=='PASS_COMPLETE_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION'
         and review.get('training_feature_extraction_eligible') is True and review.get('issues')==[], 'clean_v3_training_review')
    raw_ref=review['raw'];raw=checked(raw_ref)
    with tempfile.TemporaryDirectory(prefix='settled-prefix-review-') as tmp:
        result_ref=review_module.review(raw_ref['path'],Path(tmp)/'review',None if review.get('execution') is None else review['execution']['path'])
        need(checked(result_ref)==review, 'independent_exact_review_rederivation')
    matches=[h for h in raw['case_histories'] if h['case_id']==case_id]
    need(len(matches)==1, 'unique_captured_case_history')
    snapshot=matches[0]['prior_history_snapshot']
    result=from_snapshot(snapshot,cutoff_ns=snapshot['snapshot_time_ns'],
        forecast_origin_ns=forecast_origin_ns,owner_producers=owner_producers)
    result.update(native_source_authenticated=True,sources=[ref,raw_ref],case_id=case_id)
    return result


def checked_seed(seed, *, origin_ns, owner_producers):
    """Validate an explicitly supplied seed; authentication is a separate gate."""
    need(type(seed) is dict and seed.get('schema')==SCHEMA and seed.get('status')==
         'SETTLED_NATIVE_PREFIX_CONDITIONAL_REFERENCE_SEED_ONLY', 'seed_schema')
    seed=copy.deepcopy(seed)
    need(seed.get('forecast_origin_ns')==origin_ns and clock(seed.get('cutoff_ns'))<=origin_ns,
         'seed_origin_cutoff')
    mapping=seed.get('owner_producers',{})
    need(mapping and len(set(mapping.values()))==len(mapping) and all(mapping.get(o)==p for o,p in owner_producers.items()), 'seed_producer_lifetime_binding')
    need(seed.get('sequence_scope')==SEQUENCE_SCOPE and seed.get('missing_startup_sequence_assumption_explicit') is True
         and seed.get('current_cohort_contains_prior_labels') is False, 'seed_relative_sequence_assumption')
    owners=set(mapping);producers=set(mapping.values());cut=seed['cutoff_ns']
    for key in ('initial_sender_available_ns','initial_worker_read_release_ns','phase_watermarks'):
        need(set(seed.get(key,{}))==owners and all(clock(v)<=cut for v in seed[key].values()), 'seed_owner_clock')
    need(set(seed.get('worker_lifetimes',{}))==owners and all(type(v.get('pid')) is int and v['pid']>0
        and type(v.get('spec_sha256')) is str and len(v['spec_sha256'])==64 for v in seed['worker_lifetimes'].values()), 'seed_owned_lifetime')
    ps=seed.get('initial_protocol_state',{})
    need(set(ps)=={'reader_free','collector_free','last_collector_arrival','pending_messages'} and ps['pending_messages']==[]
        and set(ps['reader_free'])==producers and all(clock(v)<=cut for v in [*ps['reader_free'].values(),ps['collector_free'],ps['last_collector_arrival']]), 'seed_protocol_station')
    need(set(seed.get('producer_next',{}))==producers and set(seed.get('owner_issued',{}))==owners
        and set(seed.get('producer_last_reader',{}))==producers and set(seed.get('processed_issuance_watermarks',{}))==owners, 'seed_counter_domains')
    for owner,producer in mapping.items():
        n=seed['producer_next'][producer]
        need(type(n) is int and n>0 and seed['owner_issued'][owner]==n
            and seed['processed_issuance_watermarks'][owner]==n-1
            and seed['producer_last_reader'][producer]==ps['reader_free'][producer], 'seed_tracked_counter_consistency')
    need(type(seed.get('prior_request_ids')) is list and seed['prior_request_ids']
         and all(type(r) is str and r for r in seed['prior_request_ids'])
         and len(set(seed['prior_request_ids']))==len(seed['prior_request_ids']), 'seed_prior_identity_registry')
    need(seed.get('sequence')==sum(seed['producer_next'].values()), 'seed_global_counter')
    return seed
