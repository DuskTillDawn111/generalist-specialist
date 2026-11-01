"""Train-only forward-to-next-forward clocks, including asynchronous tails.

These are disjoint response intervals, not effective GPU service. Internal host
END and event queries remain separate delayed observations. No fitted provider
or future milestone replay is exported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .stage_host_partition_v1 import extract as host_extract, require, availability
from .stage_response_features_v1 import checked, receipt, ordered_events, training_roles
from .stage_response_training_v2 import causal_path

SCHEMA='gsdeploy.composable_response_macro_training.v1'


def visible_macro_prefix(events,cutoff,owner,phase):
    """Last processed forward remains in its response tail after stream query.

    Query success is not macro END: selection and other GPU work can follow.
    The next BEGIN changes that owner's phase; native END closes the macro.
    """
    active={}
    for e in events:
        if e['collector_processing_time_ns']>cutoff:continue
        edge,segment=e.get('event'),e.get('segment_id')
        if edge=='start' and segment in ('native_inference','native_ocr'):
            active[e['owner']]=dict(owner=e['owner'],request_id=e['request_id'],
                phase='entry' if segment=='native_inference' else 'ocr_inference',
                phase_id=segment,available_ns=e['collector_processing_time_ns'])
        elif edge=='start' and e.get('semantic_phase') in ('prefill','decode'):
            active[e['owner']]=dict(owner=e['owner'],request_id=e['request_id'],
                phase=e['semantic_phase'],phase_id=segment,available_ns=e['collector_processing_time_ns'])
        elif edge=='end' and segment in ('native_inference','native_ocr'):
            prior=active.get(e['owner'])
            require(prior is not None and prior['request_id']==e['request_id'],
                'Native response END requires its visible owner/request prefix')
            active.pop(e['owner'])
    peers=[value for alias,value in sorted(active.items()) if alias!=owner]
    return dict(cutoff_processing_ns=cutoff,
        peer_visible_macro_owner_count=len(peers),
        peer_same_visible_macro_phase_owner_count=sum(p['phase']==phase for p in peers),
        peer_macros=peers,query_success_ends_macro=False,
        visible_macro_is_actual_SM_activity=False,
        macro_absence_or_presence_certifies_physical_completion=False)


def macro_covariate_path(events,start,end,owner,phase):
    cuts=sorted({start,*[e['collector_processing_time_ns'] for e in events
        if start<e['collector_processing_time_ns']<end]})
    path=[dict(left_ns=at,right_ns=cuts[i+1] if i+1<len(cuts) else end,
        state=visible_macro_prefix(events,at,owner,phase),
        each_state_uses_only_its_processed_prefix=True) for i,at in enumerate(cuts)]
    for piece in path:piece['duration_ns']=piece['right_ns']-piece['left_ns']
    require(sum(x['duration_ns'] for x in path)==end-start,'Complete macro response exposure required')
    return path


def request_macros(row, events, capture_sha, mode):
    begin=row['worker_response']['start_perf_ns']
    ready=row['previous_handler_next_request_ready']['perf_ns']
    own=row['execution_events']
    language=[e for e in own if e.get('event')=='start'
        and e.get('semantic_phase') in ('prefill','decode')]
    ocr=[e for e in own if e.get('event')=='start' and e.get('segment_id')=='native_ocr']
    require(bool(language) != bool(ocr), 'One explicit language-forward or coarse OCR execution required')
    starts=language or ocr
    parent='native_inference' if language else 'native_ocr'
    ends=[e for e in own if e.get('event')=='end' and e.get('segment_id')==parent]
    require(len(ends)==1, 'Unique complete inclusive native END required')
    native_end=ends[0]['perf_ns']
    require(begin <= starts[0]['perf_ns'] < native_end <= ready,
        'Ordered handler/first forward/native END/READY required')
    boundaries=[begin,*[e['perf_ns'] for e in starts],native_end,ready]
    require(boundaries==sorted(boundaries) and len({e['segment_id'] for e in starts})==len(starts),
        'Unique ordered macro boundary identity required')
    pieces=[]
    for index,(left,right) in enumerate(zip(boundaries,boundaries[1:])):
        if right==left:continue
        if index==0:
            ident,phase,kind='entry_to_first_forward','entry','host_entry_response'
            metadata=None
        elif index==len(boundaries)-2:
            ident,phase,kind='native_end_to_worker_ready','ready_tail','host_ready_response'
            metadata=None
        else:
            metadata=starts[index-1]
            ident=metadata['segment_id']
            phase=metadata.get('semantic_phase') or 'ocr_inference'
            kind='language_forward_macro' if language else 'ocr_macro'
        # Events can be processed after this physical macro has ended. Retain
        # both clocks and never force notification inside its response interval.
        milestones=[dict(event=e.get('event'),kind=e.get('kind'),phase_id=e.get('phase_id',e.get('segment_id')),
            clocks=availability(e),end_event_complete=e.get('end_event_complete'),
            runtime_outcome_known_at_start=False,raw_event_ordinal=ordinal,
            raw_event_pointer=dict(capture_sha256=capture_sha,request_id=row['request_id'],
                execution_event_ordinal=ordinal),raw_event=dict(e))
            for ordinal,e in enumerate(own) if left<=e['perf_ns']<right
                and e.get('event') in ('start','end','observation')]
        for milestone in milestones:
            milestone['delay_from_macro_begin_ns']=milestone['clocks']['occurrence_ns']-left
        path=macro_covariate_path(events,left,right,row['model'],phase)
        shape=None if metadata is None else {k:metadata.get(k) for k in
            ('input_sequence_length','input_batch_size','forward_index')}
        pieces.append(dict(request_id=row['request_id'],owner=row['model'],task=row['task'],pool_id=row['pool_id'],
            profile_job_id=row['profile_job_id'],capture_sha256=capture_sha,
            macro_id=ident,semantic_phase=phase,kind=kind,left_ns=left,right_ns=right,
            response_wall_ns=right-left,response_domain=mode+'_forward_and_async_tail_wall',
            forward_shape=shape,BEGIN_available_ns=metadata['collector_processing_time_ns'] if metadata else None,
            causal_covariate_path=path,internal_training_milestones=milestones,
            prior_stream_pending_evidence_diagnostic_path=causal_path(events,left,right,row['model'],phase),
            includes_previous_forward_device_tail_and_selection=(kind=='language_forward_macro'),
            additional_stream_elapsed_or_sync_wait_may_be_added=False,
            true_SM_service_identified=False,milestone_outcomes_are_future_training_labels=True,
            prospective_shape_template_qualification_required=True,
            runtime_future_milestones_must_be_generated_not_replayed=True))
    require(sum(p['response_wall_ns'] for p in pieces)==ready-begin,
        'Macro responses must cover worker occupancy exactly once')
    return dict(request_id=row['request_id'],owner=row['model'],pool_id=row['pool_id'],
        profile_job_id=row['profile_job_id'],worker_start_to_ready_ns=ready-begin,macros=pieces,
        physical_completion_and_notification_availability_distinct=True)


def extract(training_review):
    host=host_extract(training_review)
    review_ref=receipt(training_review);review=checked(review_ref)
    raw_ref=review['inputs']['raw.json'];raw=checked(raw_ref)
    plan,_,jobs=training_roles(raw)
    result=[]
    for jid in jobs:
        rows=[r for r in raw['request_records'] if r['profile_job_id']==jid]
        events=ordered_events(rows)
        result.extend(request_macros(row,events,raw_ref['sha256'],plan['mode']) for row in rows)
    sources=[review_ref,raw_ref,receipt(__file__),
        receipt(Path(__file__).with_name('stage_response_training_v2.py')),
        receipt(Path(__file__).with_name('stage_response_features_v1.py')),
        receipt(Path(__file__).with_name('stage_host_partition_v1.py'))]
    require(all(receipt(r['path'])==r for r in sources),'Sources changed during macro extraction')
    return dict(schema=SCHEMA,status='DISJOINT_TRAINING_RESPONSE_MACROS_NOT_FITTED_PROVIDER',
        mode=plan['mode'],sources=sources,requests=result,
        measured_requests=len(result),macros=sum(len(x['macros']) for x in result),
        exact_worker_clock_reconstruction=True,worker_clock_reference=host['sources'],
        complete_cohort_trajectory_lookup_used=False,physical_context_as_SM_service=False,
        fitted_model_qualified=False,main_queue_provider_eligible=False,
        GPU_started=False,validation_labels_read=False,final_gain_eligible=False,
        limitations=['Entry, each forward plus asynchronous tail, and READY tail are mutually disjoint response channels.',
            'A last forward macro may contain additional model-specific work, including G2 TTS-bound scanning.',
            'Joint response/notification law and future work templates must be trained and independently validated.',
            'Public future arrivals may be known; actual future milestone times and outcomes are not public inputs.',
            'Current-stream queries do not identify all-stream completion or effective SM service.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-review',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=extract(args.training_review)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],requests=result['measured_requests'],macros=result['macros'])))
