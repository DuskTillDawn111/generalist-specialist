"""Check opaque scheduler-process-group owner hypotheses without native-ID joins."""
from bisect import bisect_right
from collections import Counter
from itertools import combinations
import argparse
import json
from pathlib import Path
import sqlite3

from .context_envelope_diagnostic_v8 import IntervalUnion,context_intervals
from .multi_native_composite_v5 import checked
from .native_collect import receipt,save
from .trace_export import linux_pid


def analyze(audit_path,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    auditref=receipt(audit_path);audit=checked(auditref)
    sqlref=next(r for r in audit['sources'] if r['path'].endswith('.sqlite'))
    rawref=next(r for r in audit['sources'] if r['path'].endswith('/raw.json'))
    raw=checked(rawref)
    if receipt(sqlref['path'])!=sqlref:raise ValueError('Source SQLite changed')
    c=sqlite3.connect('file:'+sqlref['path']+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    events,brackets,issues=context_intervals(c)
    idx={k:IntervalUnion(v) for k,v in brackets.items()}
    mapping={tuple(r['context']):r['same_scheduler_globalPid_as_primary_owner'] for r in audit['contexts']}
    if any(len(v)!=1 for v in mapping.values()):raise ValueError('Unique scheduler-group candidate required')
    owner={k:v[0] for k,v in mapping.items()};pidowner={v['pid']:a for a,v in raw['ready'].items()}
    pairwise=[]
    for a,b in combinations(sorted(events),2):
        overlap=idx[a].intersection(idx[b])
        pairwise.append(dict(left=list(a),right=list(b),same_candidate_owner=owner[a]==owner[b],
                             positive_overlap_ns=overlap.prefix[-1],positive_overlap_intervals=overlap.intervals))
    counts=Counter();mismatches=[];uncovered=[];priming=Counter()
    request_checks=audit['request_range_checks']
    for kernel in c.execute('select globalPid,deviceId,contextId,streamId,correlationId,start,end from CUPTI_ACTIVITY_KIND_KERNEL'):
        alias=pidowner.get(linux_pid(kernel['globalPid']),'UNOWNED');scope='outside_request_NVTX';rid=None
        scopes=[r for r in request_checks if r['alias']==alias and r['nvtx_extent_ns'] and r['nvtx_extent_ns'][0]<=kernel['start'] and kernel['end']<=r['nvtx_extent_ns'][1]]
        if len(scopes)>1:raise ValueError('Ambiguous request scope')
        if scopes:scope=scopes[0]['phase'];rid=scopes[0]['request_id']
        for endpoint in ('start','end'):
            t=kernel[endpoint];hits=[]
            for key,index in idx.items():
                at=bisect_right(index.starts,t)-1
                if at>=0 and t<=index.ends[at]:hits.append((key,index.intervals[at]))
            counts['all_endpoints']+=1
            if len(hits)==1:
                key,window=hits[0]
                if owner[key]==alias:
                    counts['candidate_owner_matching_endpoints']+=1
                    if scope=='priming':priming[alias]+=1
                else:
                    near=[]
                    for ownkey,index in idx.items():
                        if owner[ownkey]!=alias:continue
                        j=bisect_right(index.starts,t)-1
                        for p in (j,j+1):
                            if 0<=p<len(index.intervals):
                                a,b=index.intervals[p];near.append(dict(context=list(ownkey),envelope=[a,b],distance_ns=max(a-t,t-b,0)))
                    mismatches.append(dict(kernel=dict(kernel),endpoint=endpoint,timestamp=t,request_id=rid,scope=scope,
                                           actual_kernel_owner=alias,opaque_context=list(key),candidate_context_owner=owner[key],
                                           envelope=list(window),offset_after_restore_ns=t-window[0],
                                           offset_before_save_ns=window[1]-t,nearest_own_envelopes=sorted(near,key=lambda r:r['distance_ns'])[:2]))
            elif not hits:
                counts['uncovered_endpoints']+=1
                # Keep all unmatched endpoints; no tolerance / clipping / owner reassignment.
                uncovered.append(dict(kernel=dict(kernel),endpoint=endpoint,timestamp=t,owner=alias,scope=scope,request_id=rid))
            else:
                counts['multiply_covered_endpoints']+=1
    c.close()
    sources=[receipt(__file__),auditref,sqlref,rawref]
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Group hypothesis source changed')
    result=dict(status='OPAQUE_SCHEDULER_PROCESS_GROUP_HYPOTHESIS_AUDITED',sources=sources,
                opaque_contexts=[dict(context=list(k),candidate_owner=owner[k]) for k in sorted(owner)],
                all_contexts_retained=True,pairwise_envelope_checks=pairwise,
                all_paired_envelopes_mutually_exclusive=not any(r['positive_overlap_ns'] for r in pairwise),
                counts=dict(counts),cross_owner_endpoints=mismatches,uncovered_endpoints=uncovered,
                priming_candidate_owner_matching_endpoints=dict(priming),pairing_issues=issues,
                candidate_group_mapping_usable_as_explicit_statistical_hypothesis=True,
                strict_all_endpoint_owner_attribution_pass=not mismatches and not uncovered and not counts['multiply_covered_endpoints'],
                direct_native_ID_join=False,physical_SM_service=False,GPU_started=False,
                explanation='Original four-context mapping used controlled temporal startup and exact independent priming, not native ID equality. Same scheduler process group extends this as an explicit transitive hypothesis; mismatches and unresolved envelopes remain model support limitations, never relabelled or silently removed.')
    save(out/'report.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=analyze(a.audit,a.output);print(json.dumps(dict(counts=r['counts'],cross_owner=len(r['cross_owner_endpoints']),mutually_exclusive=r['all_paired_envelopes_mutually_exclusive'])))
