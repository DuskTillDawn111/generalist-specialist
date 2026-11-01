"""New joint timing-slack law; preserves the failed v2 fixed-delay evidence.

START plus every measured PHASE/RESULT/WORKER_READY participates in one source
decomposition. Source exact reconstruction is not protocol mechanism identity.
"""
import copy
import hashlib
import json

from . import joint_request_response_transport_candidate_v2 as work
from . import local_joint_macro_candidate_v1 as local
from .protocol_timing_slack_v3 import derive,checked

SCHEMA='gsdeploy.joint_request_protocol_timing_training_candidate.v3'


def source_messages(raw):
    rows=[]
    for request in raw['request_records']:
        owner,rid=request['model'],request['request_id']
        for ordinal,event in enumerate(request['execution_events']):
            rows.append(dict(identity=(owner,rid,'event:'+str(ordinal)),producer=owner,
                occurrence_ns=event['perf_ns'],reader_ns=event['received_time_ns'],
                dequeue_ns=event['collector_dequeued_time_ns'],processing_ns=event['collector_processing_time_ns']))
        rows.append(dict(identity=(owner,rid,'RESULT'),producer=owner,
            occurrence_ns=request['worker_response']['end_perf_ns'],reader_ns=request['response_reader_event_enqueue_time_ns'],
            dequeue_ns=request['response_collector_dequeued_time_ns'],processing_ns=request['response_collector_processing_time_ns']))
        ready=request['previous_handler_next_request_ready']
        rows.append(dict(identity=(owner,rid,'worker_READY'),producer=owner,
            occurrence_ns=ready['perf_ns'],reader_ns=ready['received_time_ns'],
            dequeue_ns=ready['collector_dequeued_time_ns'],processing_ns=ready['collector_processing_time_ns']))
    return rows


def build_law(documents,fit):
    law=work.build_law(documents,fit);slacks={};starts={};decompositions=[]
    for doc in documents:
        review=local.checked(doc['sources'][0]);raw_ref=review['inputs']['raw.json'];raw=local.checked(raw_ref)
        messages=source_messages(raw);derived=derive(messages)
        expected_count=sum(len(r['execution_events'])+2 for r in raw['request_records'])
        work.require(len(derived)==len(messages)==expected_count,
            'Independent raw event counts plus one RESULT and READY per request must match')
        work.require(expected_count==1538,'This qualified training domain contains 1538 measured messages')
        for row in derived:slacks[(raw_ref['sha256'],*row['identity'])]=row['protocol_slacks']
        for row in raw['request_records']:
            events=[(i,e) for i,e in enumerate(row['execution_events']) if e.get('event') is None]
            work.require(len(events)==1 and events[0][1]['perf_ns']==row['observed_worker_start_time_ns'],
                'One exact worker START message required')
            starts[(raw_ref['sha256'],row['request_id'])]=events[0]
        decompositions.append(dict(raw=raw_ref,messages=len(messages),all_original_clocks_reconstructed=True,
            measured_requests=len(raw['request_records']),
            all_message_sequence_including_control_and_background_authenticated=False))
    provenance=law['provenance_not_used_for_lookup']
    work.require(len(provenance)==len(law['vectors']),'Every request needs its own source provenance')
    new_provenance=[]
    for vector,source_row in zip(law['vectors'],provenance):
        work.require(vector['vector_id']==source_row['vector_id'],'Ordered request/provenance pairing changed')
        source=copy.deepcopy(source_row);capture=vector['training_capture_sha256']
        owner=vector['public_key'][0];rid=source['request_id'];ordinal,event=starts[capture,rid]
        first=vector['nodes'][0]
        # START is a real protocol message, not a zero-work phantom omitted from
        # the reader/collector queue. Its occurrence is exactly handler BEGIN.
        first['marks'].insert(0,dict(identity='event:'+str(ordinal),reference_work_at=work.fraction_record(0),
            **work.lags(event['perf_ns'],event['received_time_ns'],event['collector_dequeued_time_ns'],event['collector_processing_time_ns']),
            action='diagnostic',phase=None,source_macro_id='worker_START',occurrence_macro_id=first['macro_id'],
            query_complete=None,query_scope=None,event_reference_phase_id=None,clear_scope=None))
        used=set()
        for node in vector['nodes']:
            for mark in node['marks']:
                key=(capture,owner,rid,mark['identity'])
                work.require(key in slacks,'A required mark lacks source timing decomposition')
                mark['protocol_slacks']=slacks[key];used.add(key)
        ready_key=(capture,owner,rid,'worker_READY');vector['ready_protocol_slacks']=slacks[ready_key];used.add(ready_key)
        expected={k for k in slacks if k[:3]==(capture,owner,rid)}
        work.require(used==expected,'Every source message must appear exactly once in the request joint vector')
        vector['vector_id']=hashlib.sha256(json.dumps({k:v for k,v in vector.items() if k!='vector_id'},sort_keys=True).encode()).hexdigest()
        source['vector_id']=vector['vector_id'];new_provenance.append(source)
    law.update(schema=SCHEMA,status='JOINT_REQUEST_WORK_AND_PROTOCOL_TIMING_SLACK_TRANSPORT_TRAINING_ONLY',
        provenance_not_used_for_lookup=new_provenance,source_timing_decompositions=decompositions,
        START_message_in_entry_work_zero=True,all_measured_protocol_messages_included=True,
        per_worker_reader_station=True,shared_serial_parent_collector_station=True,
        global_collector_order_is_reader_availability_assumption=True,
        protocol_slacks_are_identified_CPU_service=False,
        original_absolute_notification_delays_held_fixed=False,
        source_clock_reconstruction_does_not_qualify_transport=True,
        omitted_control_priming_and_background_parent_events_remain_limitation=True)
    return law


def to_reference(vector,*,public_dispatch_ordinal):
    plan=work.to_reference_plan(vector,public_dispatch_ordinal=public_dispatch_ordinal)
    timings={}
    for node in vector['nodes']:
        for mark in node['marks']:
            work.require(mark['identity'] not in timings,'Distinct generated message identity required')
            timings[mark['identity']]=checked(mark['protocol_slacks'])
    timings['worker_READY']=checked(vector['ready_protocol_slacks'])
    return plan,timings
