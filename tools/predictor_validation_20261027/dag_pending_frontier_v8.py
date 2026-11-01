"""Source-DAG-certified GPU pending frontiers in a single common clock.

All recorded stream/event predecessors must have completed before a queued
kernel is counted ready. Activity completion uses CUPTI end. Virtual event/wait
completion is bounded only by an observed started descendant, never timestamp
zero. Already-started unfinished kernels remain a separate valid witness.
Default-stream implicit ordering remains the original explicit DAG hypothesis.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter,defaultdict,deque
import json
import math
from pathlib import Path
import sqlite3

from .context_envelope_diagnostic_v8 import IntervalUnion,context_intervals,controlled_creation
from .cuda_dependency_dag_v2 import build_dag
from .native_collect import receipt,save
from .trace_export import linux_pid
from .unified_corun_review_v8_1 import load_review,LIGHT


def contains(index,stamp):
    i=bisect_right(index.starts,stamp)-1
    return i>=0 and stamp<index.ends[i]


def completion_upper_bounds(nodes,edges):
    """Compute safe virtual-node completion bounds from started descendants."""
    incoming={n:0 for n in nodes};outgoing=defaultdict(list);predecessors=defaultdict(list)
    for edge in edges:
        a,b=edge['before'],edge['after'];incoming[b]+=1;outgoing[a].append(b);predecessors[b].append(a)
    queue=deque(n for n,v in incoming.items() if v==0);order=[]
    while queue:
        n=queue.popleft();order.append(n)
        for after in outgoing[n]:
            incoming[after]-=1
            if incoming[after]==0:queue.append(after)
    if len(order)!=len(nodes):raise ValueError('Cyclic recorded stream/event DAG')
    bounds={}
    for n in reversed(order):
        node=nodes[n]
        if node['kind'] in ('gpu','copy','memset'):
            bounds[n]=node['observed_end_ns']
        else:
            candidates=[]
            for child in outgoing[n]:
                value=nodes[child].get('observed_start_ns',bounds[child])
                if value is not None:candidates.append(value)
            bounds[n]=min(candidates) if candidates else None
    return bounds,predecessors,order


def analyze(review_path,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    report,plan,raw=load_review(review_path,LIGHT);sql=report['trace']['sqlite'];parsed_ref=report['trace']['parsed']
    parsed=json.loads(Path(parsed_ref['path']).read_text());rows={r['request_id']:r for r in raw['request_records']}
    sources=[receipt(review_path),*report['evidence'].values(),sql,parsed_ref,receipt(__file__),receipt(Path(__file__).with_name('cuda_dependency_dag_v2.py'))]
    pid_owner={v['pid']:a for a,v in raw['ready'].items()};phases=defaultdict(list);windows=defaultdict(list)
    for p in parsed['phases']:
        if p['request_id'] in rows:phases[p['request_id']].append(p)
    for rid,values in phases.items():
        row=rows[rid];pid=raw['ready'][row['model']]['pid']
        windows[pid].append((min(p['nsys_start_ns'] for p in values),max(p['nsys_end_ns'] for p in values),rid))
    windows={pid:sorted(v) for pid,v in windows.items()};starts={pid:[a for a,b,rid in v] for pid,v in windows.items()}
    def request_for(pid,start,end):
        if pid not in windows:return None
        i=bisect_right(starts[pid],start)-1
        if i>=0:
            a,b,rid=windows[pid][i]
            if a<=start and end<=b:return rid
        return None
    c=sqlite3.connect('file:'+sql['path']+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    context_events,brackets,pairing_issues=context_intervals(c)
    mapping,attribution=controlled_creation(c,raw,context_events,brackets)
    strings={r['id']:r['value'] for r in c.execute('select id,value from StringIds')};runtime={};sync_returns=defaultdict(list)
    for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
        pid=linux_pid(r['globalTid']);rid=request_for(pid,r['start'],r['end'])
        if rid is None:continue
        key=pid,r['correlationId']
        if key in runtime:raise ValueError('Ambiguous original process/correlation runtime identity')
        runtime[key]=dict(r,api_name=strings[r['nameId']],request_id=rid)
        if runtime[key]['api_name'].startswith(('cudaDeviceSynchronize','cuCtxSynchronize')):
            sync_returns[rid].append((r['start'],r['end']))
    activities=[];event_nodes=[];waits=[];missing=[]
    for table,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','gpu'),('CUPTI_ACTIVITY_KIND_MEMCPY','copy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
        for r in c.execute('select rowid,* from '+table):
            pid=linux_pid(r['globalPid']);api=runtime.get((pid,r['correlationId']))
            if api is None:continue
            activities.append(dict(id=kind+':'+str(r['rowid']),kind=kind,request_id=api['request_id'],
                stream_key=[pid,r['deviceId'],r['contextId'],r['streamId']],
                issue_start_ns=api['start'],issue_end_ns=api['end'],issue_thread=api['globalTid'],duration_ns=r['end']-r['start'],
                observed_start_ns=r['start'],observed_end_ns=r['end'],correlation_id=r['correlationId']))
    for r in c.execute('select rowid,* from CUPTI_ACTIVITY_KIND_CUDA_EVENT'):
        pid=linux_pid(r['globalPid']);api=runtime.get((pid,r['correlationId']))
        if api is None:continue
        event_nodes.append(dict(id='event:'+str(r['rowid']),kind='event_record',request_id=api['request_id'],
            stream_key=[pid,r['deviceId'],r['contextId'],r['streamId']],issue_start_ns=api['start'],issue_end_ns=api['end'],issue_thread=api['globalTid'],duration_ns=0,
            event_identity=[pid,r['contextId'],r['eventId'],r['eventSyncId']],raw_timestamp_used_as_completion=False))
    for r in c.execute('select rowid,* from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION'):
        pid=linux_pid(r['globalPid']);api=runtime.get((pid,r['correlationId']))
        if api is None:continue
        if api['api_name'].startswith(('cudaStreamWaitEvent','cuStreamWaitEvent')):
            waits.append(dict(id='wait:'+str(r['rowid']),kind='event_wait',request_id=api['request_id'],
                stream_key=[pid,r['deviceId'],r['contextId'],r['streamId']],issue_start_ns=api['start'],issue_end_ns=api['end'],issue_thread=api['globalTid'],duration_ns=0,
                event_identity=[pid,r['contextId'],r['eventId'],r['eventSyncId']]))
    c.close();dag=build_dag(activities,event_nodes,waits)
    if dag['cyclic']:raise ValueError('Cyclic original recorded dependency DAG')
    nodes={n['id']:n for n in dag['nodes']};completion,preds,order=completion_upper_bounds(nodes,dag['edges'])
    unknown=set()
    for problem in dag['unresolved']:
        unknown.update(problem.get('nodes',[]))
        if 'wait' in problem:unknown.add(problem['wait'])
    for n in order:
        if any(p in unknown for p in preds[n]):unknown.add(n)
    # Source kernels are exact canonical denominator and API semantic labels.
    canonical={};semantic_ranges=defaultdict(list)
    for rid,row in rows.items():
        labels={v['id']:v['semantic_phase'] for v in row['worker_response'].get('semantic_forward_observations',{}).get('records',[])}
        for p in phases[rid]:
            if p['segment_id'] in labels:semantic_ranges[rid].append((p['nsys_start_ns'],p['nsys_end_ns'],labels[p['segment_id']]))
            for k in p['kernels']:
                key=k['pid'],k['correlation_id'],k['nsys_start_ns'],k['nsys_end_ns']
                if key in canonical:raise ValueError('Duplicate canonical measured GPU kernel')
                canonical[key]=k
    observed_keys={(n['stream_key'][0],n['correlation_id'],n['observed_start_ns'],n['observed_end_ns']) for n in activities if n['kind']=='gpu'}
    if observed_keys!=set(canonical):raise ValueError('Source DAG and canonical GPU denominators differ')
    pending=defaultdict(list);byphase=defaultdict(lambda:defaultdict(list));counts=Counter();kernel_records=[]
    for n in activities:
        if n['kind']!='gpu':continue
        rid=n['request_id'];row=rows[rid];start=n['observed_start_ns'];end=n['observed_end_ns'];key=n['id']
        matched=[label for a,b,label in semantic_ranges[rid] if a<=n['issue_start_ns'] and n['issue_end_ns']<=b]
        if len(matched)>1:raise ValueError('Ambiguous API semantic phase')
        phase=matched[0] if matched else ('ocr_inference' if row['task']=='ocr' else 'outside_language_forwards')
        bounds=[completion[p] for p in preds[key]]
        root_sync=[b for a,b in sync_returns[rid] if b<=n['issue_start_ns']]
        # For a root in the filtered request graph, an observed whole-context
        # sync return proves prior same-owner activity drained. Otherwise its
        # outside predecessor obligation remains UNKNOWN.
        root_ok=bool(preds[key]) or bool(root_sync)
        known=key not in unknown and root_ok and all(b is not None for b in bounds)
        ready=max([n['issue_end_ns'],*bounds]) if known else None
        witness=min(start,ready) if ready is not None else start
        pending[rid].append((witness,end));byphase[rid][phase].append((witness,end))
        counts['kernels']+=1;counts['recorded_DAG_dependencies_known']+=known
        counts['queued_kernel_before_start_certified']+=ready is not None and ready<start
        counts['queued_pending_ns']+=max(0,start-ready) if ready is not None else 0
        counts['unfinished_started_only']+=ready is None or ready>=start
        kernel_records.append(dict(node_id=key,request_id=rid,phase=phase,issue_end_ns=n['issue_end_ns'],
            certified_ready_bound_ns=ready,observed_start_ns=start,observed_end_ns=end,
            recorded_dependency_count=len(preds[key]),recorded_dependencies_known=known))
    indexes={rid:IntervalUnion(v) for rid,v in pending.items()};phase_indexes={rid:{p:IntervalUnion(v) for p,v in g.items()} for rid,g in byphase.items()}
    byjob=defaultdict(list)
    for row in rows.values():byjob[row['profile_job_id']].append(row)
    jobs={j['id']:j for j in plan['jobs']};samples=[];censored=Counter();handoffs=[];initial=[]
    all_brackets=sorted((a,b,alias) for alias,key in mapping.items() for a,b in brackets[key]);bracket_starts=[a for a,b,alias in all_brackets]
    issue_times=sorted(problem['timestamp'] for problem in pairing_issues)
    for jobid,requests in byjob.items():
        members={r['model']:r for r in requests};request_bounds={r['model']:(min(a for a,b in ranges),max(b for a,b in ranges)) for r in requests for ranges in [[(p['nsys_start_ns'],p['nsys_end_ns']) for p in phases[r['request_id']]]]}
        def state_at(t):
            state=[]
            for alias,row in sorted(members.items()):
                rid=row['request_id']
                if contains(indexes[rid],t):
                    active=sorted(p for p,index in phase_indexes[rid].items() if contains(index,t))
                    state.append([alias,row['task'],row['pool_id'],active])
            return state
        first=min(indexes[r['request_id']].starts[0] for r in requests)
        context_at=[alias for a,b,alias in all_brackets if a<=first<b]
        if len(context_at)==1:initial.append(dict(job_id=jobid,job_name=jobs[jobid]['name'],clock_ns=first,frontier=state_at(first),observed_context=context_at[0],context_already_held_before_frontier=True))
        for alias,row in members.items():
            rid=row['request_id'];lo,hi=request_bounds[alias]
            for a,b in brackets[mapping[alias]]:
                if b<=lo or a>=hi:continue
                if a<lo or b>hi:censored['cross_request_boundary']+=1;continue
                active=[p for p,r in members.items() if indexes[r['request_id']].overlap(a,b)==b-a]
                own_full=alias in active;peers=sorted(p for p in active if p!=alias)
                static={p:sorted(label for label,index in phase_indexes[members[p]['request_id']].items() if index.overlap(a,b)==b-a) for p in active}
                stable=own_full and bool(peers) and all(len(v)==1 for v in static.values())
                reason='complete_stable_recorded_DAG_backlog' if stable else 'queue_empty_or_phase_state_changed'
                frontier=[[p,members[p]['task'],members[p]['pool_id'],static[p]] for p in sorted(active)]
                samples.append(dict(job_id=jobid,job_name=jobs[jobid]['name'],alias=alias,task=row['task'],pool_id=row['pool_id'],
                    envelope_start_ns=a,envelope_end_ns=b,envelope_length_ns=b-a,
                    phase=static.get(alias,[None])[0] if len(static.get(alias,[]))==1 else 'UNRESOLVED_PHASE',
                    stable_frontier=frontier,continuous_unfinished_kernel_peers=peers,
                    eligible_for_conditional_surrogate_budget=stable,reason=reason))
                if not stable:censored[reason]+=1
                # A gap qualifies only if current and next owner retain known
                # pending GPU work throughout and no unresolved switch event
                # occurs in the gap. It is still a context-availability gap.
                i=bisect_right(bracket_starts,b-1)
                if i>=len(all_brackets):continue
                na,nb,next_owner=all_brackets[i]
                if next_owner not in members or next_owner==alias or na<b:continue
                unknown_gap=bisect_right(issue_times,na)-bisect_right(issue_times,b-1)>0
                next_rid=members[next_owner]['request_id']
                if (not unknown_gap and na<=min(hi,request_bounds[next_owner][1])
                        and indexes[rid].overlap(b,na)==na-b and indexes[next_rid].overlap(b,na)==na-b):
                    handoffs.append(dict(job_id=jobid,clock_ns=b,current_owner=alias,next_owner=next_owner,
                        gap_end_ns=na,availability_gap_ns=na-b,frontier=state_at(b),
                        global_idle_excluded_by_pending_work=True,physical_switch_cost_identified=False))
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Recorded-DAG pending source changed')
    result=dict(schema='gsdeploy.recorded_DAG_pending_frontier.v8',status='SOURCE_DAG_PENDING_FRONTIER_EXTRACTED',
        sources=sources,controlled_temporal_attribution=attribution,pairing_issues=pairing_issues,
        recorded_DAG_unresolved=dag['unresolved'],counts=dict(counts),samples=samples,handoff_samples=handoffs,initial_context_observations=initial,
        complete_stable_backlog_samples=sum(s['eligible_for_conditional_surrogate_budget'] for s in samples),
        censored_reasons=dict(censored),native_event_timestamp_zero_used=False,
        virtual_completion_basis='upper bound from observed started DAG descendant',
        host_blocking_return_basis='actual launch API completed after same-thread blocking synchronization return',
        default_stream_implicit_dependencies_identified=False,
        default_stream_assumption='same original no implicit legacy-default-stream edges hypothesis; not fully identified hardware eligibility',
        all_native_dependencies_identified=False,missing_switch_events_filled=False,
        full_stateful_policy_fitted=False,physical_SM_service_identified=False,GPU_started=False,validation_labels_used=False)
    save(out/'kernel-ready-observations.json',dict(schema='gsdeploy.recorded_DAG_kernel_ready_observations.v8',sources=sources,rows=kernel_records))
    result['kernel_ready_observations']=receipt(out/'kernel-ready-observations.json');save(out/'report.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=analyze(a.review,a.output)
    print(json.dumps(dict(status=r['status'],counts=r['counts'],complete_stable_backlog_samples=r['complete_stable_backlog_samples'],handoff_samples=len(r['handoff_samples']),initial_context_observations=len(r['initial_context_observations']),censored_reasons=r['censored_reasons'])))


if __name__=='__main__':main()
