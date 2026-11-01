"""Retain every native scheduler context and audit process/kernel evidence.

Temporal first appearance and kernel endpoint correspondence are descriptive
evidence, not a native CUDA/driver ID or OS namespace identity join.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import json
from pathlib import Path
import sqlite3

from .background_collect import receipt,save
from .context_envelope_diagnostic_v8 import context_intervals,IntervalUnion
from .trace_export import linux_pid
from .unified_corun_review_v8_1 import load_review,LIGHT


def analyze(review_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    report,plan,raw=load_review(review_path,LIGHT)
    source_paths=[Path(__file__),Path(__file__).with_name('context_envelope_diagnostic_v8.py'),
                  Path(__file__).with_name('trace_export.py')]
    before=[receipt(p) for p in source_paths]
    for p in source_paths:
        dest=output/'sources'/p.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    sql=report['trace']['sqlite'];sources=[receipt(review_path),sql,*report['evidence'].values()]
    c=sqlite3.connect('file:'+sql['path']+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    events,brackets,pairing_issues=context_intervals(c)
    keys=sorted(events);indexes={k:IntervalUnion(brackets[k]) for k in keys}
    sessions=[dict(r) for r in c.execute('select * from TARGET_INFO_SESSION_START_TIME')]
    if len(sessions)!=1:raise ValueError('Exact common session clock required')
    base=sessions[0]['systemClockNs'];aliases=set(raw['ready'])
    startup=[e for e in raw['live_identity_events'] if e['phase']=='startup_begin' and set(e['ready'])!=aliases]
    all_ready=next(e for e in raw['live_identity_events'] if set(e['ready'])==aliases)
    previous=set();windows=[];primary={}
    for i,event in enumerate(startup):
        current=set(event['launchers']);new=current-previous
        if len(new)!=1 or set(event['ready'])!=previous:
            raise ValueError('Observed sequential owner startup required')
        alias=next(iter(new));begin=event['time_ns'];end=(startup[i+1] if i+1<len(startup) else all_ready)['time_ns']
        candidates=[k for k in keys if begin<base+events[k][0]['timestamp']<end]
        if len(candidates)==1:primary[alias]=candidates[0]
        windows.append(dict(alias=alias,begin_perf_ns=begin,end_perf_ns=end,
            first_appearance_contexts=[list(k) for k in candidates],
            evidence_kind='FIRST_SCHEDULER_APPEARANCE_NOT_NATIVE_CREATE',
            worker_pid=raw['ready'][alias]['pid'],owned_process=raw['live_owned_processes'][alias]))
        previous=current
    owner_by_pid={v['pid']:a for a,v in raw['ready'].items()}
    native_contexts=[dict(r) for r in c.execute('select * from TARGET_INFO_CUDA_CONTEXT_INFO')]
    processes=[dict(r) for r in c.execute('select * from PROCESSES')]
    table_names={r[0] for r in c.execute('select name from sqlite_master where type="table"')}
    context_apis=[dict(r) for r in c.execute("select s.value function,count(*) count from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where lower(s.value) like '%ctx%' or lower(s.value) like '%context%' or lower(s.value) like '%reset%' group by s.value")]
    tags={r['id']:r['name'] for r in c.execute('select * from ENUM_GPU_CTX_SWITCH')}
    counts_by_context={k:Counter() for k in keys};unique_by_context={k:Counter() for k in keys}
    endpoints_by_owner=Counter();uncovered=Counter();overlap=Counter();kernel_groups=[]
    for r in c.execute('select globalPid,deviceId,contextId,count(*) count,min(start) first_ns,max(end) last_ns from CUPTI_ACTIVITY_KIND_KERNEL group by globalPid,deviceId,contextId'):
        kernel_groups.append(dict(r,decoded_pid=linux_pid(r['globalPid'])))
    for row in c.execute('select globalPid,start,end from CUPTI_ACTIVITY_KIND_KERNEL'):
        alias=owner_by_pid.get(linux_pid(row['globalPid']),'UNOWNED_KERNEL_PROCESS')
        for t in (row['start'],row['end']):
            endpoints_by_owner[alias]+=1;hits=[]
            for key in keys:
                index=indexes[key];at=bisect_right(index.starts,t)-1
                if at>=0 and t<=index.ends[at]:hits.append(key);counts_by_context[key][alias]+=1
            if len(hits)==1:unique_by_context[hits[0]][alias]+=1
            elif not hits:uncovered[alias]+=1
            else:overlap[alias]+=1
    # Keep priming and measurement range membership independently of native IDs.
    strings={r['id']:r['value'] for r in c.execute('select id,value from StringIds')}
    ranges=defaultdict(list)
    for row in c.execute('select globalTid,start,end,text,textId from NVTX_EVENTS where end is not null'):
        text=row['text'] or strings.get(row['textId'],'')
        if text.startswith('GS_STAGE:'):ranges[linux_pid(row['globalTid'])].append((row['start'],row['end'],text))
    request_checks=[]
    for row in raw['priming_request_records']+raw['request_records']:
        alias=row['model'];prefix='GS_STAGE:'+row['request_id']+':'
        selected=[(a,b) for a,b,text in ranges[raw['ready'][alias]['pid']] if text.startswith(prefix)]
        request_checks.append(dict(alias=alias,request_id=row['request_id'],
            phase='priming' if row in raw['priming_request_records'] else 'measurement',
            nvtx_range_count=len(selected),nvtx_extent_ns=[min(a for a,b in selected),max(b for a,b in selected)] if selected else None))
    context_rows=[]
    for key in keys:
        gp=key[3];rows=events[key];mapped=[a for a,k in primary.items() if k==key]
        same_process=[a for a,k in primary.items() if k[3]==gp]
        context_rows.append(dict(context=list(key),switch_count=len(rows),
            tag_counts=dict(Counter(tags[r['tag']] for r in rows)),
            first_switch=rows[0],last_switch=rows[-1],paired_bracket_count=len(brackets[key]),
            first_appearance_owner_candidates=mapped,
            same_scheduler_globalPid_as_primary_owner=same_process,
            scheduler_decoded_pid=linux_pid(gp),
            scheduler_globalPid_in_PROCESSES=[r for r in processes if r['globalPid']==gp],
            scheduler_contextId_in_CUDA_metadata=[r for r in native_contexts if r['contextId']==key[2]],
            kernel_endpoint_hits_by_owner=dict(counts_by_context[key]),
            unique_kernel_endpoint_hits_by_owner=dict(unique_by_context[key]),
            direct_native_context_identity='UNKNOWN',native_creation_timestamp=None,native_destruction_timestamp=None,
            retained=True))
    c.close()
    after=[receipt(p) for p in source_paths]
    for reference in sources:
        if receipt(reference['path'])!=reference:raise ValueError('Audit source changed')
    if before!=after:raise ValueError('Audit producer changed')
    result=dict(schema='gsdeploy.multimodal_all_context_attribution_audit.v10',
        status='DESCRIPTIVE_ALL_CONTEXT_AUDIT_COMPLETE_NATIVE_IDENTITY_UNKNOWN',
        sources=sources,source_before=before,source_after=after,source_closure_unchanged=True,
        strict_review_status=report['status'],controlled_first_appearance_windows=windows,
        context_count=len(keys),all_contexts_retained=True,contexts=context_rows,
        CUDA_context_metadata=native_contexts,kernel_process_context_groups=kernel_groups,
        kernel_endpoint_totals=dict(endpoints_by_owner),uncovered_kernel_endpoints=dict(uncovered),
        multiply_covered_kernel_endpoints=dict(overlap),request_range_checks=request_checks,
        pairing_issues=pairing_issues,
        native_creation_destruction_evidence=dict(
            driver_activity_table_present='CUPTI_ACTIVITY_KIND_DRIVER' in table_names,
            context_activity_table_present='CUPTI_ACTIVITY_KIND_CONTEXT' in table_names,
            runtime_context_API_groups=context_apis,
            observed_switch_tags=sorted({tags[r['tag']] for rows in events.values() for r in rows}),
            status='UNKNOWN_NO_NATIVE_CREATION_DESTRUCTION_JOIN',
            note='RESTORE_START/SAVE_END are scheduling events, not CUDA context creation/destruction. First/last switch cannot substitute for lifetime endpoints.'),
        limitations=['CUDA activity contextId and scheduler contextId are different namespaces; no numeric equality join',
            'Scheduler globalPid and CUPTI/PROCESSES globalPid views do not establish an OS namespace bridge',
            'Same scheduler globalPid as an independently supported primary is process-level evidence only; extras retain separate identities',
            'Kernel endpoint envelope correspondence is temporal evidence, not a direct native context join or SM service proof',
            'Strict and paired PASS do not qualify extra-context attribution'],
        native_context_attribution_qualified=False,physical_SM_service_identified=False,
        fit_performed=False,validation_labels_used=False,GPU_started=False)
    save(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result=analyze(args.review,args.output)
    print(json.dumps(dict(status=result['status'],context_count=result['context_count'],all_contexts_retained=True)))


if __name__=='__main__':main()
