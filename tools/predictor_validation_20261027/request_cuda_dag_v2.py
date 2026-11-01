"""Source-bound request attribution and blocking host synchronization for CUDA DAGs."""
from __future__ import annotations

import argparse
import bisect
import collections
import json
from pathlib import Path
import sqlite3

from .cuda_dependency_dag_v2 import audit_sqlite
from .kernel_sequence_audit_v2 import load_collection
from .native_collect import receipt
from .trace_export import linux_pid


def topological(nodes,edges):
    incoming={n['id']:0 for n in nodes};outgoing=collections.defaultdict(list)
    for e in edges:
        incoming[e['after']]+=1;outgoing[e['before']].append(e['after'])
    pending=collections.deque(n for n,count in incoming.items() if count==0);visited=0
    while pending:
        n=pending.popleft();visited+=1
        for successor in outgoing[n]:
            incoming[successor]-=1
            if incoming[successor]==0:pending.append(successor)
    return visited==len(nodes)


def request_plan(request_id,request,trace,full_dag,sqlite_path):
    row=request['row'];pid=request['kernels'][0]['pid']
    regions=[r for r in trace['nvtx_ranges'] if r.get('stage') and r['stage'][0]==request_id and r['pid']==pid]
    def inside(stamp):return any(r['nsys_start_ns']<=stamp<=r['nsys_end_ns'] for r in regions)
    selected=[dict(n) for n in full_dag['dag']['nodes'] if n['stream_key'][0]==pid and inside(n['issue_start_ns'])]
    nodes={n['id']:n for n in selected};ids=set(nodes)
    expected={(k['correlation_id'],k['nsys_start_ns'],k['nsys_end_ns']) for k in request['kernels']}
    observed={(n['correlation_id'],n['observed_start_ns'],n['observed_end_ns']) for n in selected if n['kind']=='gpu'}
    if expected!=observed:raise ValueError('Request GPU DAG does not exactly match canonical owned kernel denominator')
    canonical={(k['correlation_id'],k['nsys_start_ns'],k['nsys_end_ns']):k for k in request['kernels']}
    for node in selected:
        if node['kind']=='gpu':
            node['kernel']=canonical[node['correlation_id'],node['observed_start_ns'],node['observed_end_ns']]
    edges=[e for e in full_dag['dag']['edges'] if e['before'] in ids and e['after'] in ids]
    external=[e for e in full_dag['dag']['edges'] if e['before'] not in ids and e['after'] in ids]
    original_nodes={n['id']:n for n in full_dag['dag']['nodes']}
    external_pending=[e for e in external if original_nodes[e['before']]['issue_start_ns']>=min(r['nsys_start_ns'] for r in regions)]
    if external_pending:raise ValueError('Request DAG has unresolved external concurrent dependencies')
    clock=request['clock_mapping']
    if not clock or clock.get('valid') is not True:raise ValueError('Handler-to-trace mapping required')
    start=row['worker_response']['start_perf_ns']-clock['offset_ns']
    handler_ns=row['worker_response']['end_perf_ns']-row['worker_response']['start_perf_ns']
    for node in selected:
        node['release_ns']=node['issue_start_ns']-start
        if node['release_ns']<0:raise ValueError('Mapped CUDA issue precedes handler start')
    connection=sqlite3.connect(Path(sqlite_path).resolve().as_uri()+'?mode=ro',uri=True);connection.row_factory=sqlite3.Row
    try:
        runtime={(linux_pid(r['globalTid']),r['correlationId']):dict(r)
            for r in connection.execute('select r.*,s.value as api_name from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id')}
        tables={r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
        syncs=list(connection.execute('select rowid,* from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION')) if 'CUPTI_ACTIVITY_KIND_SYNCHRONIZATION' in tables else []
    finally:connection.close()
    by_stream=collections.defaultdict(list);by_thread_stream=collections.defaultdict(list);event_nodes={}
    for node in selected:
        by_stream[tuple(node['stream_key'])].append(node)
        by_thread_stream[node['issue_thread'],tuple(node['stream_key'])].append(node)
        if node['kind']=='event_record':event_nodes[tuple(node['event_identity'])]=node
    for group in (by_stream,by_thread_stream):
        for key,members in group.items():
            members.sort(key=lambda n:n['issue_start_ns'])
            group[key]=(members,[n['issue_start_ns'] for n in members])
    sync_count=0;unresolved=[]
    for sync in syncs:
        launch=runtime.get((linux_pid(sync['globalPid']),sync['correlationId']))
        if not launch or linux_pid(sync['globalPid'])!=pid or not inside(launch['start']):continue
        name=launch['api_name']
        if name.startswith(('cudaStreamWaitEvent','cuStreamWaitEvent')):continue
        sync_id='host-sync:'+str(sync['rowid'])
        scopes=[key for key in by_stream if key[0]==pid and key[2]==sync['contextId']]
        if name.startswith(('cudaStreamSynchronize','cuStreamSynchronize')):
            scopes=[key for key in scopes if key[3]==sync['streamId']]
        elif name.startswith(('cudaDeviceSynchronize','cuCtxSynchronize')):pass
        elif name.startswith(('cudaEventSynchronize','cuEventSynchronize')):
            event=event_nodes.get((pid,sync['contextId'],sync['eventId'],sync['eventSyncId']))
            if event is None or event['issue_start_ns']>=launch['start']:
                unresolved.append(dict(sync=sync_id,reason='NO_EXACT_HOST_EVENT_SYNC_RECORD'));continue
            scopes=[]
        else:
            unresolved.append(dict(sync=sync_id,api_name=name,reason='UNMODELED_HOST_SYNC_API'));continue
        latest={}
        for key in scopes:
            members,stamps=by_stream[key];index=bisect.bisect_left(stamps,launch['start'])-1
            if index>=0:latest[key]=members[index]
        if name.startswith(('cudaEventSynchronize','cuEventSynchronize')):latest['event']=event
        node=dict(id=sync_id,kind='host_sync',duration_ns=0,release_ns=launch['start']-start,
                  api_name=name,issue_start_ns=launch['start'],issue_end_ns=launch['end'],issue_thread=launch['globalTid'])
        nodes[sync_id]=node;sync_count+=1
        for predecessor in latest.values():edges.append(dict(before=predecessor['id'],after=sync_id,kind='blocking_host_sync_wait'))
        following={}
        for (thread,key),(members,stamps) in by_thread_stream.items():
            if thread!=launch['globalTid']:continue
            index=bisect.bisect_left(stamps,launch['end'])
            if index<len(members):following[key]=members[index]
        for successor in following.values():edges.append(dict(before=sync_id,after=successor['id'],kind='host_sync_return_before_next_issue'))
    sync_threads=collections.defaultdict(list)
    for node in nodes.values():
        if node['kind']=='host_sync':sync_threads[node['issue_thread']].append(node)
    for members in sync_threads.values():
        members.sort(key=lambda n:n['issue_start_ns'])
        for before,after in zip(members,members[1:]):
            edges.append(dict(before=before['id'],after=after['id'],kind='host_blocking_call_order'))
    outgoing={e['before'] for e in edges}
    nodes['handler-end']=dict(id='handler-end',kind='handler_end',duration_ns=0,release_ns=handler_ns)
    for n in list(nodes):
        if n!='handler-end' and n not in outgoing:edges.append(dict(before=n,after='handler-end',kind='handler_waits_for_native_work'))
    edges=[dict(before=a,after=b,kind=k) for a,b,k in sorted({(e['before'],e['after'],e['kind']) for e in edges})]
    cyclic=not topological(list(nodes.values()),edges)
    return dict(request_id=request_id,alias=row['model'],family=None,task=row['task'],pool_id=row['pool_id'],
        nodes=list(nodes.values()),edges=edges,unresolved=unresolved,cyclic=cyclic,
        kernel_count=len(expected),activity_counts=dict(collections.Counter(n['kind'] for n in nodes.values())),
        host_sync_nodes=sync_count,pre_request_external_dependencies_assumed_completed=len(external),
        split=row['split'],source_sqlite=receipt(sqlite_path),clock_mapping=clock,
        timing_scope='Profiled issue release/duration replay; handler endpoint retained as terminal anchor',
        default_stream_semantics='UNMEASURED_NO_IMPLICIT_LEGACY_DEFAULT_STREAM_EDGES_ADDED',
        natural_timing_validated=False,deployment_qualified=False)


def convert(raw_path,parsed_paths):
    code_sources=[receipt(__file__),receipt(Path(__file__).with_name('cuda_dependency_dag_v2.py')),
                  receipt(Path(__file__).with_name('kernel_sequence_audit_v2.py'))]
    raw,plan,context,requests,audit=load_collection(raw_path,parsed_paths)
    if (plan.get('split')!='training' or plan.get('domain')!='light-kernel'
            or any(len(job['inputs'])!=1 or job.get('split')!='training' for job in plan['jobs'])):
        raise ValueError('Only real light-kernel training allowed in executable request DAG conversion')
    profiles=[]
    for path in parsed_paths:
        trace=json.loads(Path(path).read_text());sql=next(s['path'] for s in trace['sources'] if Path(s['path']).suffix=='.sqlite')
        dag=audit_sqlite(sql)
        for rid,request in requests.items():
            if not any(s['path']==sql for s in request['source_traces']):continue
            profile=request_plan(rid,request,trace,dag,sql);profile['family']=context['residents'][profile['alias']]
            profiles.append(profile)
    if len(profiles)!=len(requests):raise ValueError('DAG request denominator incomplete')
    if any(receipt(s['path'])!=s for s in code_sources):raise ValueError('DAG conversion source changed during execution')
    return dict(schema='gsdeploy.request_cuda_dag_profiles.v2',execution_context_sha256=raw['execution_context_sha256'],
        profiles=profiles,denominator=len(requests),training_only=True,
        sources=[receipt(raw_path),*[receipt(p) for p in parsed_paths],*code_sources],
        GPU_started=False,physical_accuracy_validated=False,unseen_input_supported=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',required=True);parser.add_argument('--parsed',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=convert(args.raw,args.parsed);args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'profiles.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps([dict(request=p['request_id'],kernel_count=p['kernel_count'],host_sync=p['host_sync_nodes'],
                           unresolved=len(p['unresolved']),cyclic=p['cyclic']) for p in result['profiles']]))


if __name__=='__main__':main()
