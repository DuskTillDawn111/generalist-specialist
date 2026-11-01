"""Read-only whole57 native endpoint capability inventory; no forecast fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from .idle_owner_selection_audit_v20 import read
from .owner_lifecycle_mark_v21 import receipt


def inventory(connection):
    result={}
    for name, in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        quoted='"'+name.replace('"','""')+'"'
        result[name]=[row[1] for row in connection.execute('PRAGMA table_info('+quoted+')')]
    return result


def run(review_path,output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    review=read(review_path)
    if review.get('status')!='PASS' or review.get('training_only') is not True:
        raise ValueError('Passed training-only whole57 review required')
    trace=review['trace']; refs=[receipt(review_path),trace['sqlite'],trace['native_first_api_kernel'],
        trace['session_clock'],trace['session_metadata'],review['host_lifecycle'],review['evidence']['raw.json'],receipt(__file__)]
    for ref in refs:
        if receipt(ref['path'])!=ref: raise ValueError('Changed whole57 source')
    path=Path(trace['sqlite']['path'])
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as connection:
        tables=inventory(connection)
    raw=read(review['evidence']['raw.json']['path'])
    source_map={
        'kernel_endpoints_and_launch_descriptor':dict(table='CUPTI_ACTIVITY_KIND_KERNEL',required_columns=['start','end','correlationId','deviceId','contextId','streamId']),
        'launch_API_intervals':dict(table='CUPTI_ACTIVITY_KIND_RUNTIME',required_columns=['start','end','correlationId','globalTid','nameId']),
        'request_phase_identity':dict(table='NVTX_EVENTS',required_columns=['start','end']),
        'memory_transfers':dict(table='CUPTI_ACTIVITY_KIND_MEMCPY',required_columns=['start','end','correlationId']),
        'device_memset':dict(table='CUPTI_ACTIVITY_KIND_MEMSET',required_columns=['start','end','correlationId']),
        'native_synchronization':dict(table='CUPTI_ACTIVITY_KIND_SYNCHRONIZATION',required_columns=['start','end','eventId','eventSyncId','streamId']),
        'native_CUDA_event_markers':dict(table='CUPTI_ACTIVITY_KIND_CUDA_EVENT',required_columns=['timestamp','eventId','eventSyncId','streamId']),
        'context_events':dict(table='GPU_CONTEXT_SWITCH_EVENTS',required_columns=['timestamp','tag','contextId','globalPid','gpuId'])}
    for entry in source_map.values():
        entry['table_present']=entry['table'] in tables
        entry['missing_columns']=[column for column in entry['required_columns'] if column not in tables.get(entry['table'],[])]
        entry['semantic_completeness_certified']=False
    after=[receipt(ref['path']) for ref in refs]
    if after!=refs: raise ValueError('Whole57 bytes changed during schema inventory')
    result=dict(schema='gsdeploy.whole57_endpoint_observation_contract.v29',training_only=True,
        status='SCHEMA_INVENTORY_NOT_NATIVE_DAG_CERTIFICATION',request_count=review['requests'],
        source_receipts_before=refs,source_receipts_after=after,tables=tables,native_sources=source_map,
        raw_field_presence={name:name in raw for name in ('phase_start_ns','deadline_ns','arrival_loop_end_ns','observation_end_ns','lifecycle_boundaries','clock_anchors','ready')},
        raw_observation_end_ns=raw.get('observation_end_ns'),
        endpoint_right_censor_rule='Only same native-clock certified observation end may censor native GPU endpoints; raw perf_counter observation_end is not automatically native time.',
        minimal_endpoint_fields=['capture_sha256','request_id','owner_pid','launch_thread','correlation_id','stream_id','native_API_entry_ns','native_API_return_ns','native_kernel_START_ns','native_kernel_END_ns','endpoint_observed_or_right_censored','native_observation_end_ns','event_timestamp_clock','observer_delivery_time_or_UNKNOWN','dependency_certification_or_UNKNOWN'],
        unresolved=dict(native_DAG_dependency_ready=None,native_observation_end_ns=None,
            observer_delivery_timestamps=None,complete_event_loss_bound=None,
            calibrated_cross_clock_error_bound_ns=review.get('calibrated_cross_clock_error_bound_ns'),
            runtime_launch_descriptor_availability=None),
        GPU_started=False,model_fitted=False,validation_labels_used=False,
        limitations=['Presence of table/columns does not prove full stream/event dependency semantics or completeness.',
            'Native first-api artifact covers first endpoints; full kernel chain must be streamed from SQLite rather than loading the multi-GB parsed JSON.',
            'No observer-delivery instrumentation was collected in this whole57 run.',
            'Do not replace uncertain common-clock mappings with per-owner medians as exact clock.'])
    output.mkdir(parents=True);(output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.review,args.output)
    print(json.dumps(dict(status=result['status'],native_sources=result['native_sources'],unresolved=result['unresolved']),indent=2))
