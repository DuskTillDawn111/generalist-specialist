"""Read existing Nsight SQLite event APIs; never export or launch a trace.

This producer preserves native API rows and exact NVTX ranges for a separate
association review. It does not authenticate a raw's JSON by itself.
"""
import argparse
import json
from pathlib import Path
import re
import sqlite3

from .native_collect import receipt,save
from .trace_export import linux_pid

TABLES=('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER')
API=re.compile(r'^(?:cuda|cu)Event(?P<operation>Record(?:WithFlags)?|Query|ElapsedTime)(?:_v[0-9]+)?$')


def read_sqlite(path):
    ref=receipt(path);connection=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row
    try:
        tables={row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        if not {'StringIds','NVTX_EVENTS'}<=tables:raise ValueError('Native StringIds/NVTX tables required')
        strings={row['id']:row['value'] for row in connection.execute('select id,value from StringIds')}
        schemas={};events=[];ranges=[]
        for table in TABLES:
            if table not in tables:continue
            schemas[table]=[dict(row) for row in connection.execute('pragma table_info('+table+')')]
            for row in connection.execute('select * from '+table+' order by start,end'):
                native=dict(row);name=strings.get(row['nameId'],'');match=API.fullmatch(name)
                if not match:continue
                operation=match.group('operation');operation='Record' if operation.startswith('Record') else operation
                events.append(dict(table=table,name=name,operation=operation,pid=linux_pid(row['globalTid']),global_tid=row['globalTid'],
                    native_start_ns=row['start'],native_end_ns=row['end'],return_value=row['returnValue'],native=native))
        for row in connection.execute('select * from NVTX_EVENTS where end is not null order by start,end'):
            label=row['text'] or strings.get(row['textId'],'')
            if not label.startswith('GS_STAGE:'):continue
            ranges.append(dict(label=label,pid=linux_pid(row['globalTid']),global_tid=row['globalTid'],native_start_ns=row['start'],native_end_ns=row['end']))
    finally:connection.close()
    if receipt(path)!=ref:raise ValueError('SQLite changed during read')
    return dict(sqlite=ref,native_api_tables=schemas,event_apis=events,nvtx_ranges=ranges)


def conditional_nvtx_offset(row,pid,native_ranges):
    """perf=native+delta, conditional on equal rate and constant local offset."""
    document=row['worker_response']['semantic_forward_observations'];matches=[];lower=[];upper=[]
    for bracket in document['device_progress']['nvtx_host_brackets']:
        found=[r for r in native_ranges if r['pid']==pid and r['label']==bracket['name']]
        if len(found)!=1:raise ValueError('One native NVTX range per exact measured semantic forward required')
        native=found[0]
        for before,after,edge in [('push_before_ns','push_after_ns','native_start_ns'),('pop_before_ns','pop_after_ns','native_end_ns')]:
            lo=bracket[before]-native[edge];hi=bracket[after]-native[edge]
            if lo>hi:raise ValueError('Malformed dual-sided NVTX host bracket')
            lower.append(lo);upper.append(hi)
        matches.append(native)
    if not matches or len({m['global_tid'] for m in matches})!=1:raise ValueError('One measured worker thread required for operation association')
    lo,hi=max(lower),min(upper)
    return dict(lower_ns=lo,upper_ns=hi,nonempty=lo<=hi,conditional_on='equal_clock_rate_and_constant_offset_over_this_request',
        independent_clock_drift_bound=False,global_tid=matches[0]['global_tid'],matched_ranges=matches)


def record_containment(row,pid,native):
    """Authenticate actual per-forward EventRecord calls by native containment.

    API tables omit event handles and output float values, so this alone cannot
    authenticate raw event IDs, stream handles or the reported elapsed number.
    """
    offset=conditional_nvtx_offset(row,pid,native['nvtx_ranges']);layers={}
    for table in native['native_api_tables']:
        records=[]
        for phase in offset['matched_ranges']:
            found=[a for a in native['event_apis'] if a['table']==table and a['operation']=='Record' and a['global_tid']==phase['global_tid']
                and phase['native_start_ns']<=a['native_start_ns']<=a['native_end_ns']<=phase['native_end_ns']]
            records.append(dict(nvtx_label=phase['label'],native_record_calls=found,exact_two_successful_calls=len(found)==2 and all(a['return_value']==0 for a in found)))
        layers[table]=records
    return dict(conditional_clock=offset,record_containment_by_API_layer=layers,
        per_forward_record_calls_authenticated=any(all(p['exact_two_successful_calls'] for p in rows) for rows in layers.values()),
        query_and_elapsed_association_authenticated=False,event_handle_identity_authenticated=False,stream_handle_identity_authenticated=False,reported_elapsed_value_authenticated=False)


def produce(path,output):
    document=read_sqlite(path);document.update(schema='gsdeploy.native_event_API_CPU_producer.v1',status='NATIVE_ROWS_EXTRACTED_NOT_RAW_AUTHENTICATED',
        producer=receipt(__file__),GPU_started=False,validation_results_read=False,
        limits=['Runtime and driver layers may represent nested calls; do not count both layers as separate logical probes.',
            'Exact measured request/PID/TID/NVTX association and host query brackets are required before calling a raw probe authenticated.',
            'ElapsedTime runs after the query-end host timestamp. Query brackets do not enclose ElapsedTime.',
            'API rows normally omit event/stream argument handles and the returned elapsed float. Those identities/values remain source-bound rather than independently measured by these tables.'])
    out=Path(output);out.mkdir(exist_ok=False);save(out/'report.json',document);return receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sqlite',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(produce(args.sqlite,args.output)))
