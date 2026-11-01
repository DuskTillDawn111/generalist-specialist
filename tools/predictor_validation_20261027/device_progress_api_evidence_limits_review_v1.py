"""Preserve actual native API evidence and unresolved export limitations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

from .native_collect import receipt,save
from .trace_export import linux_pid
from .device_progress_smoke_raw_verify_v3 import checked


def review(run,association,output):
    run=Path(run).resolve();raw_ref=receipt(run/'raw.json');raw=checked(raw_ref);strict_ref=receipt(run/'smoke-report.json');strict=checked(strict_ref)
    association_ref=receipt(association);associations=checked(association_ref)
    if strict['raw']!=raw_ref or associations['raw']!=raw_ref:raise ValueError('Actual raw bindings disagree')
    owners=[];sources=[]
    for report in associations['requests']:
        row=next(r for r in raw['request_records'] if r['request_id']==report['request_id']);alias=row['model'];pid=raw['ready'][alias]['pid']
        export=strict['traces'][alias]['export'];sql_ref=export['output'];sources.extend([sql_ref,export['source']])
        if export['returncode']!=0 or receipt(sql_ref['path'])!=sql_ref or receipt(export['source']['path'])!=export['source']:raise ValueError('Exact existing export chain required')
        c=sqlite3.connect('file:'+sql_ref['path']+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
        try:
            tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
            metadata=[dict(r) for r in c.execute('select * from META_DATA_EXPORT')]
            apis=[dict(r) for r in c.execute('select s.value as name,count(*) as count,group_concat(distinct a.returnValue) as distinct_return_values from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on s.id=a.nameId group by s.value')]
            variants=[a for a in apis if any(x in a['name'].casefold() for x in ('elapsed','getexporttable','getprocaddress'))]
            native_events=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_CUDA_EVENT')]
            diagnostics=[dict(r) for r in c.execute('select * from DIAGNOSTIC_EVENT')]
            native_schema=[dict(r) for r in c.execute('pragma table_info(CUPTI_ACTIVITY_KIND_CUDA_EVENT)')]
        finally:c.close()
        by_correlation={}
        for event in native_events:by_correlation.setdefault((linux_pid(event['globalPid']),event['correlationId']),[]).append(event)
        records=[];queries=[];elapsed=[]
        for layer_name,layer in report['layers'].items():
            for binding in layer['bindings']:
                logical,call=binding['logical'],binding['native_call']
                if logical['operation']=='Record':
                    matches=by_correlation.get((pid,call['native']['correlationId']),[])
                    records.append(dict(layer=layer_name,phase_id=logical['phase_id'],edge=logical['edge'],native_call=call,native_event_rows=matches,
                        exact_one_PID_scoped_record_correlation=len(matches)==1,
                        raw_event_handle_independently_bound=False,raw_stream_pointer_to_native_stream_id_independently_bound=False))
                elif logical['operation']=='Query':
                    q=next(q for q in row['worker_response']['semantic_forward_observations']['device_progress']['queries'] if q['observation_sequence']==logical['observation_sequence'])
                    queries.append(dict(layer=layer_name,sequence=q['observation_sequence'],phase_id=q['phase_id'],raw_complete=q['end_event_complete'],
                        host_query_bracket_ns=logical['host_bracket_ns'],native_query_interval_ns=[call['native_start_ns'],call['native_end_ns']],
                        native_return_value=call['return_value'],raw_expected_semantic_return_value=logical['expected_return'],
                        return_field_agrees_with_raw=call['return_value']==logical['expected_return'],independent_boolean_authentication=False,
                        conditional_mapping_common_offset_ns=layer['common_offset_after_API_constraints_ns']))
                else:elapsed.append(binding)
        owners.append(dict(alias=alias,pid=pid,sqlite=sql_ref,nsys_export_metadata=metadata,runtime_API_counts=apis,
            driver_API_table_present='CUPTI_ACTIVITY_KIND_DRIVER' in tables,elapsed_or_dispatch_API_name_variants_observed=variants,
            raw_forward_count=len(row['worker_response']['semantic_forward_observations']['records']),expected_EventRecord_calls=10,expected_EventQuery_calls=10,expected_EventElapsedTime_calls=5,
            observed_bound_Record_calls=len(records),observed_bound_Query_calls=len(queries),observed_bound_ElapsedTime_calls=len(elapsed),
            conditional_clock=report['conditional_clock'],records=records,queries=queries,
            native_event_table_schema=native_schema,native_event_rows=native_events,
            native_event_timestamps_all_zero=bool(native_events) and all(e['timestamp']==0 for e in native_events),
            raw_query_boolean_counts=dict(Counter(str(q['end_event_complete']) for q in row['worker_response']['semantic_forward_observations']['device_progress']['queries'])),
            return_value_disagreements=sum(not q['return_field_agrees_with_raw'] for q in queries),
            dropped_or_error_diagnostic_messages=[d for d in diagnostics if any(x in d['text'].casefold() for x in ('dropped','lost','error'))],
            record_correlations=all(r['exact_one_PID_scoped_record_correlation'] for r in records)))
    docbase=Path('/opt/nvidia/nsight-systems/2025.6.3');docs=[receipt(docbase/'documentation/AnalysisGuide/index.html'),receipt(docbase/'documentation/UserGuide/index.html'),receipt(docbase/'target-linux-x64/reports/cuda_api_trace.py'),receipt(docbase/'target-linux-x64/exporter/export_schema_version_notes.txt')]
    result=dict(schema='gsdeploy.actual_native_event_API_evidence_limits.v1',status='PARTIAL_NATIVE_CALL_EVIDENCE_NUMERICAL_AUTHENTICATION_UNQUALIFIED',
        raw=raw_ref,strict_smoke=strict_ref,conditional_association=association_ref,owners=owners,
        findings=dict(native_Record_and_Query_calls_present=True,per_forward_native_Record_containment=True,
            query_boolean_return_field_semantics='UNRESOLVED_CURRENT_CAPTURE_EXPORT',native_EventElapsedTime_calls_observed=False,
            elapsed_value_and_event_object_identity='UNKNOWN',native_device_completion_timestamps='UNKNOWN_TABLE_TIMESTAMPS_ZERO',
            all_request_GPU_work_complete=False,kernel_cursor=None,main_lifecycle_resume_eligible=False),
        installed_tool_documentation=[dict(evidence='Nsight report maps runtime.returnValue to Result; AnalysisGuide labels it function return. Neither local source qualifies the observed all-zero field for not-ready queries.',sources=[docs[0],docs[2]]),
            dict(evidence='profile --help: --cuda-trace-all-apis defaults false and skips some APIs; --cuda-event-trace defaults false and enables device-side completion only when enabled. Current frozen command enables neither.',sources=[docs[1]]),
            dict(evidence='Missing ElapsedTime names, including suffix variants and driver/dispatch alternatives, are a capture-coverage limitation. No dynamic field-reliability probe was run.',sources=docs)],
        limits=['Current all-zero exported Query return values disagree with fifteen raw false reports. This mismatch prevents independent boolean authentication; it does not by itself establish fabricated raw output.',
            'Every request has ten matched native Records and ten matched native Queries. Five expected ElapsedTime calls per request are absent from exported API tables; global trace missing rate is unidentifiable.',
            'Native event IDs/streams can be correlated to Records in this trace, but API argument handles do not independently bind them to raw Python event IDs or the raw stream pointer.',
            'A nonempty conditional constant-offset interval permits source-order consistency checks; it is not an independent drift or clock-alignment proof.',
            'These API/numeric limitations do not invalidate separately reviewed host-response/phase/telemetry collection provenance, nor qualify fitting or progress fractions.'],
        sources=[receipt(__file__),*sources,*docs],GPU_started=False,validation_results_read=False)
    for ref in [raw_ref,strict_ref,association_ref,*sources]:
        if receipt(ref['path'])!=ref:raise ValueError('Evidence changed')
    out=Path(output);out.mkdir(exist_ok=False);save(out/'report.json',result);return receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--association',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(review(args.run,args.association,args.output)))
