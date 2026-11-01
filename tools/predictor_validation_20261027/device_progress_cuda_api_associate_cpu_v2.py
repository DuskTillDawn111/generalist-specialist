"""Conditional native EventRecord/Query/ElapsedTime association, CPU only.

Read existing SQLite via the frozen v1 producer. No CUDA calls, profiler export,
clock fitting, event-handle recovery or full-request completion inference.
"""
import argparse
import json
from pathlib import Path

from .native_collect import receipt,save
from .device_progress_cuda_api_probe_cpu_v1 import read_sqlite,record_containment

# CUDA driver_types.h cudaErrorNotReady and cuda.h CUDA_ERROR_NOT_READY.
NOT_READY=600


def require(condition,message):
    if not condition:raise ValueError(message)


def ns(value):return type(value) is int and value>=0


def expected_operations(row):
    rid=row['request_id'];response=row['worker_response'];doc=response['semantic_forward_observations'];progress=doc['device_progress']
    require(doc['request_id']==rid,'Semantic request identity mismatch')
    phases={phase['id']:phase for phase in doc['records']};records=progress['records'];queries=progress['queries'];primary=[]
    require(len(phases)==len(doc['records'])==len(records) and len({r['phase_id'] for r in records})==len(records),'Exact forward/record coverage required')
    for record in records:
        phase=record['phase_id'];require(phase in phases and record['event_id']==rid+':'+phase+':stream_end','Raw record identity mismatch')
        for edge in ('begin','end'):
            bracket=record[edge+'_record_host_bracket_ns']
            require(isinstance(bracket,list) and len(bracket)==2 and all(ns(x) for x in bracket) and bracket[0]<=bracket[1],'Invalid host EventRecord bracket')
            primary.append(dict(operation='Record',phase_id=phase,edge=edge,host_bracket_ns=bracket,expected_return=0,nvtx_label=phases[phase]['nvtx_name']))
    require([q['observation_sequence'] for q in queries]==list(range(1,len(queries)+1)),'Per-request query sequence required')
    completed=set()
    for query in queries:
        phase=query['phase_id'];done=query['end_event_complete'];lo,hi=query['host_query_begin_ns'],query['host_query_end_ns']
        require(query['request_id']==rid and phase in phases and phase not in completed and type(done) is bool,'Raw query identity/completion mismatch')
        require(all(ns(x) for x in (lo,hi)) and lo<=hi,'Invalid host EventQuery bracket')
        primary.append(dict(operation='Query',phase_id=phase,host_bracket_ns=[lo,hi],expected_return=0 if done else NOT_READY,
            observation_sequence=query['observation_sequence'],done=done))
        if done:completed.add(phase)
    primary.sort(key=lambda item:(item['host_bracket_ns'][0],item['host_bracket_ns'][1]))
    require(all(a['host_bracket_ns'][1]<=b['host_bracket_ns'][0] for a,b in zip(primary,primary[1:])),'Overlapping logical probe API brackets')
    operations=[]
    for index,item in enumerate(primary):
        operations.append(item)
        if item['operation']=='Query' and item['done']:
            # elapsed_time executes after the captured query-end timestamp and
            # before the next source-level API operation (or handler end).
            upper=primary[index+1]['host_bracket_ns'][0] if index+1<len(primary) else response['end_perf_ns']
            require(ns(upper) and upper>=item['host_bracket_ns'][1],'ElapsedTime source-order bracket invalid')
            operations.append(dict(operation='ElapsedTime',phase_id=item['phase_id'],observation_sequence=item['observation_sequence'],
                host_bracket_ns=[item['host_bracket_ns'][1],upper],expected_return=0,
                bracket_basis='source_order_after_query_end_before_next_probe_API_or_handler_end'))
    return operations


def associate_request(row,pid,native):
    containment=record_containment(row,pid,native);offset=containment['conditional_clock'];expected=expected_operations(row);layers={}
    for table in native['native_api_tables']:
        issues=[];bindings=[];used=set();lower,upper=offset['lower_ns'],offset['upper_ns']
        calls=[(index,call) for index,call in enumerate(native['event_apis']) if call['table']==table and call['pid']==pid and call['global_tid']==offset['global_tid']]
        native_ranges={item['label']:item for item in offset['matched_ranges']}
        if lower>upper:issues.append('EMPTY_CONDITIONAL_CONSTANT_OFFSET_INTERVAL')
        for logical in expected:
            candidates=[]
            for index,call in calls:
                if index in used or call['operation']!=logical['operation']:continue
                lo,hi=call['native_start_ns'],call['native_end_ns']
                if not all(ns(x) for x in (lo,hi)) or lo>hi:continue
                feasible_lower=max(lower,logical['host_bracket_ns'][0]-lo);feasible_upper=min(upper,logical['host_bracket_ns'][1]-hi)
                if feasible_lower>feasible_upper:continue
                if logical['operation']=='Record':
                    phase=native_ranges[logical['nvtx_label']]
                    if not phase['native_start_ns']<=lo<=hi<=phase['native_end_ns']:continue
                candidates.append((index,call,feasible_lower,feasible_upper))
            if len(candidates)!=1:
                issues.append('MISSING_OR_AMBIGUOUS_NATIVE_CALL:'+logical['operation']+':'+logical['phase_id']);continue
            index,call,lower,upper=candidates[0];used.add(index)
            if type(call['return_value']) is not int or call['return_value']!=logical['expected_return']:
                issues.append('NATIVE_RETURN_MISMATCH:'+logical['operation']+':'+logical['phase_id'])
            if bindings and call['native_start_ns']<bindings[-1]['native_call']['native_end_ns']:
                issues.append('NATIVE_OPERATION_SEQUENCE_OVERLAP_OR_REVERSAL')
            bindings.append(dict(logical=logical,native_call=call,source_native_index=index))
        # Every native event API that could be inside this handler must be
        # accounted for. Extra internal APIs make this strict association
        # ambiguous; they are retained as a negative, not silently discarded.
        begin,end=row['worker_response']['start_perf_ns'],row['worker_response']['end_perf_ns']
        extras=[]
        if offset['nonempty']:
            for index,call in calls:
                if index in used:continue
                if call['native_end_ns']+offset['upper_ns']>=begin and call['native_start_ns']+offset['lower_ns']<=end:extras.append(call)
        if extras:issues.append('UNASSOCIATED_NATIVE_EVENT_APIS_IN_POSSIBLE_HANDLER_WINDOW')
        layers[table]=dict(conditional_association_consistent=not issues and len(bindings)==len(expected),issues=issues,
            common_offset_after_API_constraints_ns=[lower,upper],bindings=bindings,extra_native_calls=extras)
    complete=[table for table,value in layers.items() if value['conditional_association_consistent']]
    return dict(schema='gsdeploy.native_event_API_conditional_association.v2',request_id=row['request_id'],pid=pid,
        conditional_clock=offset,record_containment=containment,expected_operations=expected,layers=layers,
        conditional_query_elapsed_association_consistent=bool(complete),consistent_API_layers=complete,
        query_and_elapsed_association_authenticated=False,independent_clock_alignment_authenticated=False,
        event_handle_identity_authenticated=False,stream_handle_identity_authenticated=False,reported_elapsed_value_authenticated=False,
        all_request_GPU_work_complete=False,kernel_cursor=None,main_lifecycle_resume_eligible=False,
        limits=['A consistent association depends on equal clock rate and constant offset over this request; dual NVTX brackets do not independently establish a drift bound.',
            'Native API names/return codes authenticate observed call presence in their own timeline; missing event/stream arguments and output float prevent independent event-handle or elapsed-value authentication.',
            'Runtime and driver layers are alternatives, never added together as separate logical probes. Exact raw/source/SQLite execution provenance requires its separate review.'])


def produce(raw_path,sqlite_map_path,output):
    raw_ref=receipt(raw_path);raw=json.loads(Path(raw_path).read_text());plan_ref=raw['plan']
    require(receipt(plan_ref['path'])==plan_ref,'Plan receipt changed');plan=json.loads(Path(plan_ref['path']).read_text())
    require(plan['split']=='smoke' and plan['mode']=='kernel' and raw['performance_scope']=='profiler_perturbed','Measured kernel smoke only; validation excluded')
    mapping_ref=receipt(sqlite_map_path);mapping=json.loads(Path(sqlite_map_path).read_text());reports=[];sql_refs=[]
    for row in raw['request_records']:
        if row['model']=='ocr#0':continue
        require(row['split']=='smoke','Every measured role must be smoke')
        native=read_sqlite(mapping[row['model']]);sql_refs.append(native['sqlite']);reports.append(associate_request(row,raw['ready'][row['model']]['pid'],native))
    require(receipt(raw_path)==raw_ref and receipt(sqlite_map_path)==mapping_ref,'Inputs changed')
    out=Path(output);out.mkdir(exist_ok=False);report=dict(schema='gsdeploy.native_event_API_association_CPU_report.v2',
        status='CONDITIONAL_ASSOCIATION_REVIEW_ONLY_NOT_COMPLETE_SMOKE_AUTHENTICATION',raw=raw_ref,sqlite_map=mapping_ref,sqlites=sql_refs,requests=reports,
        all_requests_conditionally_consistent=bool(reports) and all(r['conditional_query_elapsed_association_consistent'] for r in reports),
        sources=[receipt(__file__),receipt(Path(__file__).with_name('device_progress_cuda_api_probe_cpu_v1.py'))],GPU_started=False,validation_results_read=False)
    save(out/'report.json',report);return receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw',required=True);parser.add_argument('--sqlite-map',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(produce(args.raw,args.sqlite_map,args.output)))
