"""Separate startup, real priming and measured endpoint correspondence."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import json
from pathlib import Path
import sqlite3

from .background_collect import receipt,save
from .context_envelope_diagnostic_v8 import context_intervals,IntervalUnion
from .multi_native_composite_v5 import checked
from .trace_export import linux_pid


def analyze(audit_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    audit_ref=receipt(audit_path);audit=checked(audit_ref)
    sql_ref=next(r for r in audit['sources'] if r['path'].endswith('.sqlite'))
    if receipt(sql_ref['path'])!=sql_ref:raise ValueError('Changed original SQLite')
    raw_ref=next(r for r in audit['sources'] if r['path'].endswith('/raw.json'));raw=checked(raw_ref)
    source_ref=receipt(__file__);(output/'source.py').write_bytes(Path(__file__).read_bytes())
    by_owner=defaultdict(list)
    for row in audit['request_range_checks']:
        if not row['nvtx_extent_ns']:raise ValueError('Original request NVTX extent required')
        by_owner[row['alias']].append(row)
    starts={}
    for alias,rows in by_owner.items():
        rows.sort(key=lambda r:r['nvtx_extent_ns'][0]);starts[alias]=[r['nvtx_extent_ns'][0] for r in rows]
        if any(a['nvtx_extent_ns'][1]>b['nvtx_extent_ns'][0] for a,b in zip(rows,rows[1:])):
            raise ValueError('Owner request scopes overlap')
    c=sqlite3.connect('file:'+sql_ref['path']+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    events,brackets,issues=context_intervals(c);keys=sorted(events);indexes={k:IntervalUnion(brackets[k]) for k in keys}
    owners={r['pid']:a for a,r in raw['ready'].items()};cells=Counter();totals=Counter();uncovered=Counter();multiply=Counter()
    for row in c.execute('select globalPid,start,end from CUPTI_ACTIVITY_KIND_KERNEL'):
        alias=owners.get(linux_pid(row['globalPid']),'UNKNOWN_OWNER');scope='outside_original_request_NVTX';rid=None
        if alias in starts:
            at=bisect_right(starts[alias],row['start'])-1
            if at>=0 and row['end']<=by_owner[alias][at]['nvtx_extent_ns'][1]:
                request=by_owner[alias][at];scope=request['phase'];rid=request['request_id']
        totals[alias,scope,rid]+=2
        for t in (row['start'],row['end']):
            hits=[]
            for key in keys:
                idx=indexes[key];at=bisect_right(idx.starts,t)-1
                if at>=0 and t<=idx.ends[at]:hits.append(key)
            if not hits:uncovered[alias,scope,rid]+=1
            if len(hits)>1:multiply[alias,scope,rid]+=1
            for key in hits:cells[key,alias,scope,rid]+=1
    c.close()
    records=[dict(context=list(key),alias=alias,scope=scope,request_id=rid,endpoints=n)
        for (key,alias,scope,rid),n in cells.items()]
    total_rows=[dict(alias=a,scope=s,request_id=rid,endpoints=n,uncovered_endpoints=uncovered[a,s,rid],
        multiply_covered_endpoints=multiply[a,s,rid]) for (a,s,rid),n in totals.items()]
    for ref in (source_ref,audit_ref,sql_ref,raw_ref):
        if receipt(ref['path'])!=ref:raise ValueError('Scoped audit source changed')
    result=dict(schema='gsdeploy.multimodal_context_scoped_kernel_endpoint_audit.v10',
        status='DESCRIPTIVE_SCOPED_ENDPOINT_AUDIT_COMPLETE_NATIVE_IDENTITY_UNKNOWN',
        sources=[source_ref,audit_ref,sql_ref,raw_ref],source_closure_unchanged=True,
        retained_context_count=len(keys),all_contexts_retained=True,
        context_request_endpoint_cells=records,request_endpoint_totals=total_rows,pairing_issues=issues,
        conclusion='Extra contexts have separate retained envelope/kernel correspondence. This does not invent missing native creation/destruction or an OS namespace join.',
        native_context_attribution_qualified=False,physical_SM_service_identified=False,
        fit_performed=False,validation_labels_used=False,GPU_started=False)
    save(output/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();r=analyze(args.audit,args.output);print(json.dumps(dict(status=r['status'],contexts=r['retained_context_count'])))


if __name__=='__main__':main()
