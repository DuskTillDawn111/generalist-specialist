"""Training-only natural reader-visible START-to-RESULT completion exposure.

Reader handler activity means START has been received and RESULT has not yet
been received. It is not physical worker activity, CUDA readiness or service.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import copy
import gzip
import json
from pathlib import Path

from .implementation_source_closure_v27 import implementation_sources,strict_references
from .native_collect import context_identity
from .owner_lifecycle_mark_v21 import IntervalIndex,receipt
from .unknown_preflight_archive_v1 import save_synced

OWNERS=('generalist_2#0','ocr#0','text#0','text#1')
SCHEMA='gsdeploy.natural_reader_handler_completion_risk.v45'
STATUS='NATURAL_READER_HANDLER_COMPLETION_RISK_EXTRACTED_NO_FIT'
FEATURE_FIELDS={'alias','task','pool_id','attained_reader_handler_age_ns','active_handler_by_owner'}


def checked(ref):
    if set(ref)!={'path','sha256'} or receipt(ref['path'])!=ref:raise ValueError('Exact existing source reference required')
    return json.loads(Path(ref['path']).read_text())


def validate_features(value):
    if (set(value)!=FEATURE_FIELDS or value['alias'] not in OWNERS
            or not isinstance(value['task'],str) or not value['task']
            or not isinstance(value['pool_id'],str) or not value['pool_id']
            or type(value['attained_reader_handler_age_ns']) is not int
            or value['attained_reader_handler_age_ns']<0
            or set(value['active_handler_by_owner'])!=set(OWNERS)
            or any(type(v) is not bool for v in value['active_handler_by_owner'].values())):
        raise ValueError('Only complete reader-visible identity, attained age and boolean mask features allowed')
    return value


def reader_interval(record):
    starts=[e for e in record['execution_events'] if 'event' not in e]
    if len(starts)!=1:raise ValueError('Exactly one actually received START required, without worker-clock fallback')
    origin=starts[0]['received_time_ns'];terminal=record['response_reader_event_enqueue_time_ns']
    if (type(origin) is not int or type(terminal) is not int or not 0<=origin<terminal
            or record['completion_time_ns']!=terminal):
        raise ValueError('Positive exact reader START-to-RESULT interval required')
    return origin,terminal


def split_request(record,cohort,capture,indexes,cuts):
    if cohort not in ('measured','priming'):raise ValueError('Explicit original cohort identity required')
    lo,hi=reader_interval(record)
    boundaries=[lo,*cuts[bisect_right(cuts,lo):bisect_left(cuts,hi)],hi]
    for left,right in zip(boundaries,boundaries[1:]):
        features=validate_features(dict(alias=record['model'],task=record['task'],pool_id=record['pool_id'],
            attained_reader_handler_age_ns=left-lo,
            active_handler_by_owner={o:indexes[o].contains(left) for o in OWNERS}))
        yield dict(capture=capture,request_id=record['request_id'],cohort=cohort,
            origin_reader_START_ns=lo,start_ns=left,end_ns=right,age_entry_ns=left-lo,
            age_exit_ns=right-lo,completion_observed=right==hi,features=features)


def clocks(event):
    fields=('perf_ns','received_time_ns','collector_dequeued_time_ns','collector_processing_time_ns')
    if any(type(event.get(k)) is not int or event[k]<0 for k in fields):
        raise ValueError('All actual worker/reader/collector clocks required, no invented delivery')
    if not all(event[a]<=event[b] for a,b in zip(fields,fields[1:])):
        raise ValueError('Actual event/delivery/collector clocks must preserve causality')
    return dict(worker_perf_ns=event['perf_ns'],reader_received_ns=event['received_time_ns'],
        collector_dequeued_ns=event['collector_dequeued_time_ns'],collector_processing_ns=event['collector_processing_time_ns'])


def extract(review_reference,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    review=checked(review_reference)
    if (review.get('schema')!='gsdeploy.whole57_joint_training_review.v23' or review.get('status')!='PASS'
            or review.get('issues')!=[] or review.get('trace') is not None or review.get('requests')!=57
            or review.get('training_only') is not True):
        raise ValueError('Complete original natural no-profiler 57-request training review required')
    evidence=review['evidence'];raw=checked(evidence['raw.json']);plan=checked(evidence['plan.json']);context=checked(evidence['context.json'])
    if (raw.get('training_only') is not True or raw['domain']!='whole57-joint-natural-v23'
            or plan['domain']!=raw['domain'] or context['measurement_domain']!=raw['domain']
            or raw['route_id']!=plan['route_id'] or context['whole57_training_route_id']!=raw['route_id']
            or raw['execution_context_sha256']!=plan['execution_context_sha256']
            or context_identity(context)!=plan['execution_context_sha256']
            or raw['plan']!=evidence['plan.json'] or raw['context']!=evidence['context.json']
            or set(raw['ready'])!=set(OWNERS) or raw['clock_contract']['per_owner_median_map_used_as_common_clock'] is not False
            or len(raw['request_records'])!=57 or len(raw['priming_request_records'])!=12):
        raise ValueError('Original raw, route, context, common-host clock, READY and full57+12 identity must agree')
    capture=evidence['raw.json']['sha256'];records=[('priming',r) for r in raw['priming_request_records']]+[('measured',r) for r in raw['request_records']]
    if len({r['request_id'] for _,r in records})!=69:raise ValueError('All original measured and priming request identities must be unique')
    public={r['request_id']:r for r in raw['planned_public_cohort']}
    if set(public)!={r['request_id'] for r in raw['request_records']}:raise ValueError('Entire offered measured cohort must be preserved')
    spans=defaultdict(list)
    for cohort,row in records:
        if row['status']!='completed' or row['model'] not in OWNERS:raise ValueError('Original complete request and owner required')
        if cohort=='measured':
            planned=public[row['request_id']]
            if ((row['task'],row['pool_id'],row['scheduled_time_ns']-raw['phase_start_ns'])!=
                    (planned['task'],planned['pool_id'],planned['offset_ns'])):raise ValueError('Offered cohort public identities/arrivals changed')
        spans[row['model']].append(reader_interval(row))
    for values in spans.values():
        ordered=sorted(values)
        if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):
            raise ValueError('Same-owner reader START/RESULT chronology is not serial; boolean mask would lose request count')
    indexes={owner:IntervalIndex(spans[owner]) for owner in OWNERS}
    cuts=sorted({t for values in spans.values() for span in values for t in span})
    implementations=implementation_sources(__file__)
    sources=strict_references([review_reference,list(evidence.values()),review['host_lifecycle'],
        [s['frozen'] for s in context['execution_sources']],implementations])
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Natural source dependency changed')
    out.mkdir(parents=True,exist_ok=False);risk_path=out/'reader-completion-risk.jsonl.gz'
    request_path=out/'request-boundaries-and-labels.jsonl.gz';events_path=out/'received-events.jsonl.gz'
    counts=Counter();phase_missing=[];phase_pairs=0;event_rows=[];first_measurement=min(reader_interval(r)[0] for r in raw['request_records'])
    priming_active_at_first_measurement=[]
    with gzip.open(risk_path,'wt') as risk_file,gzip.open(request_path,'wt') as request_file:
        for cohort,row in records:
            lo,hi=reader_interval(row);owner=row['model'];worker=row['worker_response'];ready=row['previous_handler_next_request_ready']
            if (worker['request_id']!=row['request_id'] or ready['request_id']!=row['request_id']
                    or ready['pid']!=raw['ready'][owner]['pid'] or ready['clock']!='host_perf_counter_ns'
                    or ready['boundary']!='previous_handler_next_request_ready'
                    or ready['scope']!='after_response_flush_before_next_read'
                    or not worker['start_perf_ns']<worker['end_perf_ns']<=ready['perf_ns']):
                raise ValueError('Worker END and next-ready are separate source-bound host labels')
            if cohort=='priming' and lo<=first_measurement<hi:priming_active_at_first_measurement.append(row['request_id'])
            identity=dict(capture=capture,request_id=row['request_id'],cohort=cohort,alias=owner,task=row['task'],pool_id=row['pool_id'])
            for risk in split_request(row,cohort,capture,indexes,cuts):
                risk_file.write(json.dumps(risk,separators=(',',':'))+'\n')
                counts[cohort+'_risk_intervals']+=1;counts[cohort+'_exposure_ns']+=risk['end_ns']-risk['start_ns']
                counts[cohort+'_completion_events']+=risk['completion_observed']
            counts[cohort+'_requests']+=1
            labels=dict(identity,scheduled_time_ns=row['scheduled_time_ns'],enqueued_time_ns=row['enqueued_time_ns'],
                dispatch_begin_ns=row['actual_submit_time_ns'],dispatch_write_completed_ns=row['submit_write_completed_time_ns'],
                origin_reader_START_ns=lo,RESULT_reader_received_ns=hi,
                worker_handler_start_perf_ns=worker['start_perf_ns'],worker_handler_end_perf_ns=worker['end_perf_ns'],
                worker_next_ready_perf_ns=ready['perf_ns'],worker_next_ready_reader_received_ns=ready['received_time_ns'],
                worker_response_labels=copy.deepcopy(worker),worker_next_ready_record=copy.deepcopy(ready))
            request_file.write(json.dumps(labels,separators=(',',':'))+'\n')
            pairs=defaultdict(dict)
            for ordinal,event in enumerate(row['execution_events']):
                observed=clocks(event);kind='START' if 'event' not in event else 'PHASE_'+event['event'].upper()
                if kind not in ('START','PHASE_START','PHASE_END'):raise ValueError('Original declared execution event required')
                if not worker['start_perf_ns']<=event['perf_ns']<=worker['end_perf_ns']:raise ValueError('Python wall events must remain inside handler labels')
                event_rows.append(dict(identity,event_kind=kind,source_event_ordinal=ordinal,**observed,payload=copy.deepcopy(event)))
                if 'event' in event:
                    pair=pairs[event['segment_id']]
                    if event['event'] in pair:raise ValueError('Duplicate phase boundary cannot be silently paired')
                    pair[event['event']]=event['perf_ns']
            for segment,pair in pairs.items():
                if set(pair)!={'start','end'}:phase_missing.append(dict(request_id=row['request_id'],segment_id=segment,observed=pair));continue
                if pair['end']<pair['start']:raise ValueError('Negative Python phase wall interval')
                phase_pairs+=1
            result_event=dict(perf_ns=worker['end_perf_ns'],received_time_ns=hi,
                collector_dequeued_time_ns=row['response_collector_dequeued_time_ns'],collector_processing_time_ns=row['response_collector_processing_time_ns'])
            event_rows.append(dict(identity,event_kind='RESULT',**clocks(result_event),payload=copy.deepcopy(worker)))
            event_rows.append(dict(identity,event_kind='WORKER_READY',**clocks(ready),payload=copy.deepcopy(ready)))
    with gzip.open(events_path,'wt') as event_file:
        for event in sorted(event_rows,key=lambda r:(r['reader_received_ns'],r['request_id'],r['event_kind'])):
            event_file.write(json.dumps(event,separators=(',',':'))+'\n')
    if (counts['measured_requests']!=57 or counts['priming_requests']!=12
            or counts['measured_completion_events']!=57 or counts['priming_completion_events']!=12):
        raise ValueError('Every original request must contribute exactly one reader RESULT target')
    save_synced(out/'actual-ready.json',dict(capture=capture,source_raw=evidence['raw.json'],ready=copy.deepcopy(raw['ready'])))
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Natural source dependency changed during extraction')
    result=dict(schema=SCHEMA,status=STATUS,training_only=True,validation_labels_used=False,fit_performed=False,
        capture=capture,route=raw['route_id'],execution_context_sha256=raw['execution_context_sha256'],
        source_review=review_reference,source_raw=evidence['raw.json'],source_plan=evidence['plan.json'],source_context=evidence['context.json'],
        sources=sources,implementations=implementations,actual_ready=receipt(out/'actual-ready.json'),
        risk_chronology=receipt(risk_path),request_boundaries=receipt(request_path),received_events=receipt(events_path),counts=dict(counts),
        reader_history_window_ns=[min(cuts),max(cuts)],original_measurement_observation_window_ns=[raw['observation_start_ns'],raw['observation_end_ns']],
        phase_pairs=phase_pairs,missing_phase_pairs=phase_missing,
        priming_requests_active_at_first_measured_START=priming_active_at_first_measurement,
        feature_fields=sorted(FEATURE_FIELDS),risk_origin='Actually received START execution event, received_time_ns.',
        risk_terminal='RESULT reader event enqueue, exactly original completion_time_ns.',
        mask_definition='All original measured+priming reader START-to-RESULT half-open intervals; never worker perf handler mask.',
        interval_cuts='Only actually received START/RESULT boundaries across all owners; phase/READY are retained independently and never risk cuts.',
        prefix_visibility='An event is visible to parent reader only at reader_received_ns; collector-driven state requires its separate collector_processing_ns.',
        priming_model_use='Preserved as separate cohort/history. Root first fit selects measured risk only; held-out priming is never training.',
        future_END_or_RESULT_used_before_observation=False,worker_ready_equals_RESULT_assumed=False,
        Python_phase_END_equals_CUDA_END_claimed=False,physical_context_gate_used=False,
        physical_worker_activity_identified=False,SM_service_identified=False,
        native_clock_projection_used=False,phase_risk_fitted=False,GPU_started=False,
        independent_prediction_accuracy_validated=False)
    save_synced(out/'report.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review',required=True)
    parser.add_argument('--review-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=extract(dict(path=str(Path(args.review).absolute()),sha256=args.review_sha256),args.output)
    print(json.dumps(dict(status=result['status'],counts=result['counts'],report=receipt(Path(args.output)/'report.json'))))
