"""Read-only CUDA stream/event DAG prototype; no timestamp-based cross edges.

Event identities join (process,context,eventId,eventSyncId), not a reused event
handle alone. Stream ordering uses observed host API issue order. The timestamp
zero in Nsight CUDA_EVENT records is never treated as event completion time.
Host synchronization and default-stream implicit dependencies are reported as
open obligations, so this partial DAG is not yet a deployable scheduler.
"""
from __future__ import annotations

import argparse
import collections
import heapq
import json
from pathlib import Path
import sqlite3

from .native_collect import receipt
from .trace_export import linux_pid


def build_dag(activities,events,waits):
    nodes={};streams=collections.defaultdict(list);edges=set();unresolved=[]
    event_lookup={}
    def add(node):
        if node['id'] in nodes:raise ValueError('Duplicate CUDA DAG node identity')
        nodes[node['id']]=node
        streams[tuple(node['stream_key'])].append(node)
    for activity in activities:add(activity)
    for event in events:
        add(event)
        k=tuple(event['event_identity'])
        if k in event_lookup:raise ValueError('Ambiguous reused event sync identity')
        event_lookup[k]=event['id']
    for wait in waits:
        add(wait)
        source=event_lookup.get(tuple(wait['event_identity']))
        if source is None:unresolved.append(dict(wait=wait['id'],reason='NO_EXACT_EVENT_RECORD_IDENTITY'))
        else:edges.add((source,wait['id'],'event_record_to_wait'))
    for members in streams.values():
        members.sort(key=lambda n:(n['issue_start_ns'],n['id']))
        for before,after in zip(members,members[1:]):
            if (before['issue_start_ns']==after['issue_start_ns'] or
                    before.get('issue_thread')!=after.get('issue_thread') and
                    before.get('issue_end_ns',before['issue_start_ns'])>after['issue_start_ns']):
                unresolved.append(dict(nodes=[before['id'],after['id']],reason='AMBIGUOUS_CONCURRENT_STREAM_ISSUE_ORDER'))
            else:edges.add((before['id'],after['id'],'stream_fifo'))
    incoming={n:0 for n in nodes};outgoing=collections.defaultdict(list)
    for before,after,_ in edges:incoming[after]+=1;outgoing[before].append(after)
    queue=[n for n,count in incoming.items() if count==0];heapq.heapify(queue)
    finish={n:0 for n in nodes};visited=0
    while queue:
        n=heapq.heappop(queue);visited+=1;finish[n]+=nodes[n].get('duration_ns',0)
        for successor in outgoing[n]:
            finish[successor]=max(finish[successor],finish[n]);incoming[successor]-=1
            if incoming[successor]==0:heapq.heappush(queue,successor)
    cyclic=visited!=len(nodes)
    return dict(nodes=list(nodes.values()),edges=[dict(before=a,after=b,kind=k) for a,b,k in sorted(edges)],
        unresolved=unresolved,cyclic=cyclic,
        partial_dag_gpu_activity_critical_path_ns=None if cyclic else max(finish.values(),default=0),
        critical_path_scope='Lower-bound diagnostic with zero event/wait nodes; launch supply and host sync not completed',
        stream_count=len(streams),timestamp_order_cross_stream_edges_invented=False,
        deployment_qualified=False,physically_validated=False)


def audit_sqlite(path):
    path=Path(path).resolve();source=receipt(path)
    connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);connection.row_factory=sqlite3.Row
    try:
        runtime={}
        for row in connection.execute('select r.*,s.value as api_name from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id'):
            key=(linux_pid(row['globalTid']),row['correlationId'])
            if key in runtime:raise ValueError('Ambiguous process/correlation CUDA runtime identity')
            runtime[key]=dict(row)
        tables={r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
        activities=[];missing=[]
        for table,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','gpu'),('CUPTI_ACTIVITY_KIND_MEMCPY','copy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
            if table not in tables:continue
            for row in connection.execute('select rowid,* from '+table):
                pid=linux_pid(row['globalPid']);launch=runtime.get((pid,row['correlationId']))
                if launch is None:
                    missing.append(dict(table=table,rowid=row['rowid'],reason='MISSING_RUNTIME_ISSUE'));continue
                activities.append(dict(id=kind+':'+str(row['rowid']),kind=kind,
                    stream_key=[pid,row['deviceId'],row['contextId'],row['streamId']],
                    issue_start_ns=launch['start'],issue_end_ns=launch['end'],issue_thread=launch['globalTid'],
                    duration_ns=row['end']-row['start'],
                    observed_start_ns=row['start'],observed_end_ns=row['end'],
                    correlation_id=row['correlationId'],api_name=launch['api_name'],
                    memory_metadata={k:row[k] for k in ('bytes','copyKind','srcKind','dstKind','memKind') if k in row.keys()}))
        events=[];waits=[];sync_inventory=collections.Counter();host_sync=[]
        if 'CUPTI_ACTIVITY_KIND_CUDA_EVENT' in tables:
            for row in connection.execute('select rowid,* from CUPTI_ACTIVITY_KIND_CUDA_EVENT'):
                pid=linux_pid(row['globalPid']);launch=runtime.get((pid,row['correlationId']))
                if launch is None:
                    missing.append(dict(table='CUDA_EVENT',rowid=row['rowid'],reason='MISSING_RUNTIME_ISSUE'));continue
                events.append(dict(id='event:'+str(row['rowid']),kind='event_record',
                    stream_key=[pid,row['deviceId'],row['contextId'],row['streamId']],
                    issue_start_ns=launch['start'],issue_end_ns=launch['end'],issue_thread=launch['globalTid'],duration_ns=0,
                    event_identity=[pid,row['contextId'],row['eventId'],row['eventSyncId']],
                    raw_event_timestamp=row['timestamp'],raw_timestamp_used_as_completion=False))
        if 'CUPTI_ACTIVITY_KIND_SYNCHRONIZATION' in tables:
            for row in connection.execute('select rowid,* from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION'):
                pid=linux_pid(row['globalPid']);launch=runtime.get((pid,row['correlationId']))
                name=launch['api_name'] if launch else 'UNKNOWN_RUNTIME';sync_inventory[name]+=1
                if name.startswith(('cudaStreamWaitEvent','cuStreamWaitEvent')):
                    waits.append(dict(id='wait:'+str(row['rowid']),kind='event_wait',
                        stream_key=[pid,row['deviceId'],row['contextId'],row['streamId']],
                        issue_start_ns=launch['start'],issue_end_ns=launch['end'],issue_thread=launch['globalTid'],duration_ns=0,
                        event_identity=[pid,row['contextId'],row['eventId'],row['eventSyncId']]))
                else:host_sync.append(dict(rowid=row['rowid'],api_name=name,correlation_id=row['correlationId']))
        dag=build_dag(activities,events,waits)
    finally:connection.close()
    if receipt(path)!=source:raise ValueError('SQLite changed during dependency reconstruction')
    return dict(schema='gsdeploy.cuda_stream_event_dag_prototype.v2',source=source,
        evaluator=receipt(__file__),scope='Whole trace, including initialization/warmup; not request training observations',
        activity_counts=dict(collections.Counter(n['kind'] for n in activities)),event_record_count=len(events),
        event_wait_count=len(waits),sync_api_inventory=dict(sync_inventory),
        host_sync_obligations=host_sync,missing_runtime=missing,
        remaining_obligations=['Host blocking synchronization dependencies','Default-stream mode semantics',
            'Request-stage source attribution for copy/memset/event nodes','Concurrent intra-owner runtime cursor',
            'Host-launch supply model','Cross-context scheduling and copy DRAM interference'],dag=dag,
        GPU_started=False,training_observations_created=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sqlite',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit_sqlite(args.sqlite)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'dag.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(activities=result['activity_counts'],events=result['event_record_count'],
        waits=result['event_wait_count'],unresolved=len(result['dag']['unresolved']),cyclic=result['dag']['cyclic'])))


if __name__=='__main__':main()
