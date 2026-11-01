"""Exact CPU sender/stdin FIFO adapter and source-bound training lag audit.

Parent write-completion stamps do not gate worker START. READY processing does
not gate an already queued worker request. Unobserved transport components are
explicit supplied-world inputs, not a fitted idle handoff or native queue law.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path

from .macro_joint_response_fit_v1 import verified_document
from .stage_response_features_v1 import checked, receipt
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA='gsdeploy.worker_stdin_FIFO_CPU_contract_audit.v1'


def require(condition,message):
    if not condition:raise ValueError(message)


def ns(value):
    require(type(value) in (int,Fraction) and value>=0,'Explicit nonnegative rational clock required')
    return Fraction(value)


@dataclass(frozen=True)
class TransportDraw:
    enqueue_after_arrival_ns: Fraction
    sender_turn_extra_ns: Fraction
    submit_to_write_record_ns: Fraction
    submit_to_stdin_line_available_ns: Fraction
    worker_release_to_START_extra_ns: Fraction


def sender_fifo(arrival_ns,previous_write_record_ns,draw):
    """One per-owner sender; never wait for a worker RESULT or READY notice."""
    require(type(draw) is TransportDraw,'Explicit joint transport decomposition required; missing components are UNKNOWN')
    arrival,previous=ns(arrival_ns),ns(previous_write_record_ns)
    values=[ns(getattr(draw,key)) for key in draw.__dataclass_fields__]
    enqueue_extra,pickup_extra,write_extra,line_extra,worker_extra=values
    require(line_extra<=write_extra,'A complete stdin line must be written before the parent records send completion')
    enqueued=arrival+enqueue_extra;submit=max(enqueued,previous)+pickup_extra
    return dict(enqueued_ns=enqueued,parent_submit_ns=submit,
        parent_write_record_ns=submit+write_extra,stdin_line_available_ns=submit+line_extra,
        worker_release_to_START_extra_ns=worker_extra)


def worker_fifo_START(previous_READY_occurrence_ns,send):
    """Release from the previous actual handler, not parent notice processing.

    READY perf is sampled before its print/flush. The explicitly supplied extra
    includes remaining READY emit, next read, parse, task check and pre-START
    preparation. It is not identified separately in existing donor raws.
    """
    release=max(ns(previous_READY_occurrence_ns),ns(send['stdin_line_available_ns']))
    return release+ns(send['worker_release_to_START_extra_ns'])


def idle_donor_decomposition_bounds(submit,write_record,start,previous_READY):
    """Feasible components only; never select one unobserved decomposition."""
    submit,write_record,start,previous=map(ns,(submit,write_record,start,previous_READY))
    require(submit<=start and submit<=write_record and previous<=start,'Ordered observed send/START/READY clocks required')
    lower=submit;upper=min(write_record,start)
    return dict(stdin_line_available_ns_bounds=[lower,upper],
        worker_release_to_START_extra_ns_bounds=[start-max(previous,upper),start-max(previous,lower)],
        uniquely_identified=lower==upper,
        previous_handler_READY_precedes_submit=previous<=submit,
        exact_worker_stdin_waiting_idle_observed=False)


def compose_explicit_world(requests,draws,handlers,*,initial_worker_read_release_ns,initial_sender_available_ns):
    """Small adapter oracle; all durations/delays explicitly supplied by ordinal.

    Response durations here are fixed inputs only. The real macro reference can
    call sender_fifo at public enqueue and worker_fifo_START on generated READY
    occurrence while dynamically generating its own response clocks.
    """
    sender={owner:ns(at) for owner,at in initial_sender_available_ns.items()}
    worker={owner:ns(at) for owner,at in initial_worker_read_release_ns.items()}
    ordinals=defaultdict(int);rows=[];ids=set();last_notices={};last_enqueue={}
    for _,index,row in sorted((ns(r['arrival_ns']),i,r) for i,r in enumerate(requests)):
        require(set(row)=={'request_id','owner','task','pool_id','arrival_ns'}
            and row['request_id'] not in ids,'Distinct complete public requests required')
        ids.add(row['request_id']);owner=row['owner'];ordinal=ordinals[owner];ordinals[owner]+=1
        key=(owner,row['task'],row['pool_id'],ordinal)
        require(key in draws and key in handlers,'UNKNOWN_EXPLICIT_PUBLIC_KEY_ORDINAL_TRANSPORT_OR_RESPONSE_DRAW')
        send=sender_fifo(row['arrival_ns'],sender[owner],draws[key]);sender[owner]=send['parent_write_record_ns']
        require(send['enqueued_ns']>=last_enqueue.get(owner,Fraction(0)),
            'UNKNOWN_TRANSPORT_DRAWS_REORDER_PUBLIC_PARENT_ENQUEUE_FIFO')
        last_enqueue[owner]=send['enqueued_ns']
        start=worker_fifo_START(worker[owner],send);handler=handlers[key]
        result_mark=start+ns(handler['START_to_RESULT_mark_ns'])
        ready=start+ns(handler['START_to_READY_occurrence_ns'])
        require(result_mark<=ready,'RESULT timestamp must precede post-flush READY boundary')
        result_reader=result_mark+ns(handler['RESULT_mark_to_reader_ns'])
        result_processed=result_reader+ns(handler['RESULT_reader_to_processing_ns'])
        ready_reader=ready+ns(handler['READY_occurrence_to_reader_ns'])
        ready_processed=ready_reader+ns(handler['READY_reader_to_processing_ns'])
        require(result_reader<=ready_reader and result_processed<=ready_processed,
            'UNKNOWN_JOINT_WORLD_VIOLATES_SAME_PRODUCER_RESULT_READY_FIFO')
        if owner in last_notices:
            previous_reader,previous_processed=last_notices[owner]
            require(previous_reader<=result_reader and previous_processed<=result_processed,
                'UNKNOWN_JOINT_WORLD_VIOLATES_CROSS_REQUEST_STDOUT_FIFO')
        last_notices[owner]=ready_reader,ready_processed;worker[owner]=ready
        rows.append(dict(request_id=row['request_id'],owner=owner,public_owner_ordinal=ordinal,
            arrival_ns=ns(row['arrival_ns']),**send,worker_START_ns=start,
            READY_occurrence_ns=ready,READY_reader_ns=ready_reader,READY_processed_ns=ready_processed,
            RESULT_mark_ns=result_mark,RESULT_reader_ns=result_reader,RESULT_processed_ns=result_processed))
    return dict(rows=rows,worker_release_endpoint='READY_occurrence_plus_explicit_worker_continuation',
        sender_release_endpoint='previous_parent_write_record',
        READY_processing_gates_queued_worker_START=False,parent_write_record_gates_worker_START=False,
        transport_law_fitted=False,response_model_fitted=False,main_queue_provider_eligible=False,
        validation_labels_read=False,GPU_started=False)


def encode(value):
    if isinstance(value,Fraction):return int(value) if value.denominator==1 else dict(numerator=value.numerator,denominator=value.denominator)
    if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [encode(v) for v in value]
    return value


def audit(paths):
    root=Path(__file__).resolve().parents[2];captures=[];sources=[]
    roles={name:str(root/path) for name,path in dict(
        parent_deployment='gsdeploy/runtime/deployment.py',parent_transport='gsdeploy/runtime/transport.py',
        generic_protocol='gsdeploy/runtime/protocol.py',baseline_observed_worker='tools/predictor_validation_20261027/observed_worker.py',
        actual_semantic_worker='tools/predictor_validation_20261027/semantic_progress_worker_v1.py',
        active_collector='tools/predictor_validation_20261027/profile_owned_collect_v3.py',
        admission='gsdeploy/runtime/admission.py').items()}
    for path in paths:
        macro,closure=verified_document(path);sources.extend(closure)
        review=checked(macro['sources'][0]);raw_ref=review['inputs']['raw.json'];raw=checked(raw_ref)
        context=checked(raw['context']);source_rows={r['original_path']:r['frozen'] for r in context['execution_sources']}
        binding={role:source_rows[original] for role,original in roles.items()}
        for ref in binding.values():require(receipt(ref['path'])==ref,'Frozen queue source changed')
        # The actual collector copies these bindings into aliases unchanged.
        require(all(v.get('resource_binding',{}).get('mode')!='request_admission'
            for v in context['worker_bindings'].values()),'Admission-enabled execution needs a separate sender contract')
        require(context['worker_variant']=='semantic_progress_worker_v1.py','Exact actual worker variant required')
        deployment=Path(binding['parent_deployment']['path']).read_text()
        transport=Path(binding['parent_transport']['path']).read_text()
        worker_source=Path(binding['actual_semantic_worker']['path']).read_text()
        require('self.workers[model].send(' in deployment and 'for line in sys.stdin:' in worker_source
            and 'self.process.stdin.flush()' in transport,'Expected frozen send/stdin source contract required')
        prior={};records=[]
        all_rows=sorted([*raw['priming_request_records'],*raw['request_records']],key=lambda r:r['actual_submit_time_ns'])
        for row in all_rows:
            owner=row['model'];previous=prior.get(owner)
            submit=row['actual_submit_time_ns'];write=row['submit_write_completed_time_ns'];start=row['observed_worker_start_time_ns']
            require(row['scheduled_time_ns']<=row['enqueued_time_ns']<=submit<=write
                and submit<=start==row['worker_response']['start_perf_ns'],'Exact monotone sender/worker clocks required')
            if previous:
                require(previous['submit_write_completed_time_ns']<=submit
                    and previous['enqueued_time_ns']<=row['enqueued_time_ns']
                    and previous['previous_handler_next_request_ready']['perf_ns']<=start,
                    'Per-owner sender/stdin FIFO violated')
            if row['phase']=='profile':
                require(previous is not None,'Measured donor must bind previous same-worker request, including priming')
                ready=previous['previous_handler_next_request_ready']
                record=dict(capture_sha256=raw_ref['sha256'],request_id=row['request_id'],owner=owner,
                    previous_request_id=previous['request_id'],scheduled_ns=row['scheduled_time_ns'],
                    enqueue_ns=row['enqueued_time_ns'],submit_ns=submit,write_record_ns=write,worker_START_ns=start,
                    previous_READY_occurrence_ns=ready['perf_ns'],
                    previous_READY_processing_ns=ready['collector_processing_time_ns'],
                    observed_submit_to_START_ns=start-submit,
                    previous_handler_busy_at_submit=submit<ready['perf_ns'],
                    worker_START_precedes_parent_write_record=start<write,
                    START_precedes_previous_READY_processing=start<ready['collector_processing_time_ns'],
                    prior_READY_to_submit_ns=submit-ready['perf_ns'],
                    scheduled_to_enqueue_ns=row['enqueued_time_ns']-row['scheduled_time_ns'],
                    enqueue_to_submit_ns=submit-row['enqueued_time_ns'],
                    submit_to_write_record_ns=write-submit,
                    component_bounds=idle_donor_decomposition_bounds(submit,write,start,ready['perf_ns']),
                    pure_idle_handoff_qualified=False,pipe_line_and_worker_pre_START_components_identified=False)
                records.append(record)
            prior[owner]=row
        captures.append(dict(capture_sha256=raw_ref['sha256'],source_binding=binding,measured_donors=len(records),
            priming_rows_used_only_to_establish_predecessors=len(raw['priming_request_records']),
            busy_at_submit=sum(r['previous_handler_busy_at_submit'] for r in records),
            START_before_write_record=sum(r['worker_START_precedes_parent_write_record'] for r in records),
            START_before_previous_READY_processing=sum(r['START_precedes_previous_READY_processing'] for r in records),
            minimum_prior_READY_to_submit_ns=min(r['prior_READY_to_submit_ns'] for r in records),
            submit_to_START_range_ns=[min(r['observed_submit_to_START_ns'] for r in records),max(r['observed_submit_to_START_ns'] for r in records)],
            records=records))
        sources.extend(binding.values())
    require(len(captures)==3 and len({c['capture_sha256'] for c in captures})==3,'Three distinct natural captures required')
    closure=strict_references([sources,implementation_sources(__file__)])
    require(all(receipt(r['path'])==r for r in closure),'Queue source changed during audit')
    return encode(dict(schema=SCHEMA,status='SOURCE_BOUND_FIFO_CONTRACT_WITH_IDLE_HANDLER_COMPOSITE_DONORS_ONLY',
        sources=closure,captures=captures,measured_donors=sum(c['measured_donors'] for c in captures),
        all_donors_prior_handler_READY_before_submit=all(c['busy_at_submit']==0 for c in captures),
        exact_worker_stdin_idle_authenticated=False,standalone_pure_handoff_identified=False,
        READY_processing_gates_queued_worker_START=False,
        parent_write_record_is_worker_line_availability_timestamp=False,
        busy_pipe_backpressure_and_worker_queue_delay_law='UNKNOWN_NO_BUSY_DONORS',
        parent_enqueue_sender_scheduling_serialization_pipe_and_pre_START_decomposition='UNKNOWN_NOT_SEPARATELY_OBSERVED',
        admission_configuration='NONE_IN_THESE_CAPTURES_NOT_A_NEW_POLICY',
        native_CUDA_API_numeric_authentication='UNQUALIFIED',main_queue_provider_eligible=False,
        validation_labels_read=False,GPU_started=False,
        limitations=[
            'Previous READY perf timestamp precedes its emit/flush, so no prior handler overlap does not prove worker is waiting in stdin.',
            'Observed submit-to-START is an idle-handler composite including serialization, pipe/read/parse and pre-START preparation.',
            'Parent write record can follow worker START; using it as a worker gate contradicts observed rows.',
            'No busy submitted donor identifies pipe backpressure or FIFO waiting under a changed offered schedule.',
            'CPU adapter requires explicit latent line availability and worker continuation; no missing component is silently zero.',
            'Fixed response durations in the adapter oracle are supplied worlds, not a fitted response model or native execution replay.']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-macros',nargs='+',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=audit(args.training_macros)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],donors=result['measured_donors'])))
