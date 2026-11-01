"""Explicit complete-or-censored stdout review for fixed public181 resources.

Censoring is an observed outcome, not a successful response. Every present
message is checked; absent RESULT/READY and unfinished phases remain explicit.
No partial device progress or missing quality value is inferred.
"""
from __future__ import annotations
import copy
from .twoFIFO_training_manifest_v1 import require,OWNERS
from .stage_response_features_v1 import stamp
from .device_progress_smoke_raw_verify_v3 import reparse_outputs,validate_request,AVAILABILITY

SCHEMA='gsdeploy.twoFIFO_public181_partial_protocol.v1'


def inspect_prefix(row,ready,*,shutdown_end_ns):
    rid=row['request_id'];events=row.get('execution_events',[])
    require(row.get('status') in ('completed','incomplete','error','queued','submitted','planned'),
            'Explicit retained target outcome required')
    scheduled=row.get('scheduled_time_ns');require(stamp(scheduled),'Actual public arrival required')
    enqueue=row.get('enqueued_time_ns');submit=row.get('actual_submit_time_ns');write=row.get('submit_write_completed_time_ns')
    if enqueue is not None:require(stamp(enqueue) and scheduled<=enqueue<=shutdown_end_ns,'Ordered enqueue prefix required')
    if submit is not None:require(stamp(enqueue) and stamp(submit) and enqueue<=submit<=shutdown_end_ns,'Ordered sender prefix required')
    if write is not None:require(stamp(submit) and stamp(write) and submit<=write<=shutdown_end_ns,'Ordered write prefix required')
    starts=[];active={};closed=set();last=None
    for event in events:
        times=[event.get(k) for k in ('perf_ns',*AVAILABILITY)]
        require(event.get('request_id')==rid and all(stamp(t) for t in times)
            and times==sorted(times) and times[-1]<=shutdown_end_ns
            and (last is None or all(a<=b for a,b in zip(last,times))),
            'Partial stdout request/four-clock FIFO prefix required')
        last=times
        if event.get('event') is None:
            require(not starts and not active and not closed and event is events[0]
                and event.get('pid')==ready.get('pid') and stamp(submit) and submit<=times[0]
                and row.get('observed_worker_start_time_ns')==times[0], 'Exact partial handler START required')
            starts.append(event);continue
        require(len(starts)==1,'Partial PHASE cannot precede START')
        kind=event.get('event');key=(event.get('kind'),event.get('segment_id'))
        if kind=='start':
            require(key not in active and key not in closed,'Partial duplicate phase START')
            active[key]=times[0]
        elif kind=='end':
            require(key in active and active[key]<=times[0],'Partial phase END without START')
            del active[key];closed.add(key)
        elif kind=='observation':
            require(row['model']!='ocr#0' and event.get('kind')=='device_event_query'
                and event.get('schema')=='gsdeploy.stream_event_forward_observation.v1'
                and type(event.get('end_event_complete')) is bool
                and event.get('all_request_GPU_work_complete') is False
                and all(event.get(k) is None for k in ('in_flight_remaining_fraction','completion_time_ns','physical_SM_service_ns')),
                'Limited partial device query identity/scope required')
        else:raise ValueError('Unknown partial stdout PHASE edge')
    require(bool(starts)==(row.get('observed_worker_start_time_ns') is not None),
            'Partial START field cannot be fabricated or omitted')
    response=row.get('worker_response');late=row.get('late_response_after_drain')
    require(not(response and late),'Duplicate complete and late stdout outcomes')
    payload=response or late
    reader=row.get('response_reader_event_enqueue_time_ns')
    if payload is not None:
        require(type(payload) is dict and payload.get('request_id')==rid,'Partial RESULT/ERROR request identity required')
        if payload.get('success') is True:
            require(payload.get('task')==row['task'] and payload.get('task_type')==row['task'],
                    'Partial successful RESULT task identity required')
            require(len(starts)==1 and payload.get('start_perf_ns')==starts[0]['perf_ns']
                and stamp(payload.get('end_perf_ns')) and payload['start_perf_ns']<=payload['end_perf_ns'],
                'Partial successful RESULT handler bracket required')
        else:require(type(payload.get('error')) is str or payload.get('success') is False,
                     'Explicit failed stdout outcome required')
        times=[row.get('response_'+suffix) for suffix in
            ('reader_event_enqueue_time_ns','collector_dequeued_time_ns','collector_processing_time_ns')]
        require(all(stamp(t) for t in times) and times==sorted(times) and times[-1]<=shutdown_end_ns
            and (last is None or all(a<=b for a,b in zip(last[1:],times))),
            'Partial RESULT/ERROR availability must follow stdout prefix')
        if payload.get('end_perf_ns') is not None:
            require(stamp(payload['end_perf_ns']) and payload['end_perf_ns']<=times[0],
                    'Partial RESULT occurrence must precede reader')
        require((row.get('completion_time_ns')==reader if response else
            row.get('completion_time_ns') is None and row.get('late_response_time_ns')==reader),
            'Partial response versus post-drain response identity required')
    else:
        require(all(row.get(k) is None for k in ('completion_time_ns','response_reader_event_enqueue_time_ns',
            'response_collector_dequeued_time_ns','response_collector_processing_time_ns')),
            'Missing RESULT cannot supply response clocks')
    nxt=row.get('previous_handler_next_request_ready')
    if nxt is not None:
        times=[nxt.get(k) for k in ('perf_ns',*AVAILABILITY)]
        require(payload is not None and nxt.get('request_id')==rid and nxt.get('pid')==ready.get('pid')
            and nxt.get('clock')=='host_perf_counter_ns' and nxt.get('boundary')=='previous_handler_next_request_ready'
            and nxt.get('scope')=='after_response_flush_before_next_read' and all(stamp(t) for t in times)
            and times==sorted(times) and times[-1]<=shutdown_end_ns
            and reader<=times[1] and (last is None or all(a<=b for a,b in zip(last,times))),
            'Exact present partial READY chain required')
    return dict(request_id=rid,observed_messages=len(events)+int(payload is not None)+int(nxt is not None),
        status='OBSERVED_PARTIAL_OR_COMPLETE_PREFIX_NOT_A_SUCCESS_LABEL',
        missing_RESULT=payload is None,missing_READY=nxt is None,unfinished_phase_keys=[list(k) for k in active],
        partial_device_progress_qualified=False,missing_quality_imputed=False)


def inspect_protocol(raw):
    rows=raw['priming_request_records']+raw['request_records']
    end=next(b['time_ns'] for b in raw['lifecycle_boundaries'] if b['boundary']=='shutdown_end')
    prefixes=[inspect_prefix(row,raw['ready'][row['model']],shutdown_end_ns=end) for row in rows]
    # Rebuild every present successful original output, including a RESULT
    # observed after fixed drain. This temporary projection neither invents
    # outputs nor drops partial rows from the separately checked denominator.
    view=copy.copy(raw);view['priming_request_records']=[];view['request_records']=[]
    projected={}
    for row in rows:
        payload=row.get('worker_response') or row.get('late_response_after_drain')
        if payload is not None and payload.get('success') is True:
            item=dict(row,worker_response=payload);projected[row['request_id']]=item
            view['request_records'].append(item)
    protocol=reparse_outputs(view);issues=list(protocol['issues']);details=[]
    for row,prefix in zip(rows,prefixes):
        if row['request_id'] in projected and row.get('previous_handler_next_request_ready') is not None:
            # The complete validator requires successful completed status and a
            # completion clock; a late response has an explicit temporary view.
            item=projected[row['request_id']]
            item.update(status='completed',completion_time_ns=row.get('completion_time_ns') or row.get('late_response_time_ns'))
            found,detail=validate_request(item,raw['ready'][row['model']],protocol['policies'].get(row['model']))
            issues.extend(found);details.append(dict(prefix=prefix,complete_protocol=detail))
        else:details.append(dict(prefix=prefix,complete_protocol=None))
    require(all(not p['missing_RESULT'] and not p['missing_READY'] for p in prefixes[:18]),
            'All original eighteen priming RESULT and READY messages required')
    return dict(schema=SCHEMA,issues=issues,requests=details,sources=protocol['sources'],
        original_output_reparse=protocol,all_offered_and_priming_prefixes=len(prefixes),
        partial_observations=sum(d['complete_protocol'] is None for d in details),
        absent_outcomes_are_successes=False,resource_observation_can_outlive_request_success=True)
