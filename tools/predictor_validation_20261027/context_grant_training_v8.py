"""Extract conservative complete/backlogged context-envelope training samples.

A kernel already started but not ended is sufficient observed unfinished GPU
work. A pending launch alone is not proof of eligible work. Consequently many
envelopes are censored and unsupported states remain UNKNOWN. Envelope length
is not physical quantum and SAVE-to-RESTORE gap is not automatically switch cost.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import json
from pathlib import Path
import sqlite3
import statistics

from .context_envelope_diagnostic_v8 import IntervalUnion,context_intervals,controlled_creation
from .native_collect import receipt,save
from .trace_export import linux_pid
from .unified_corun_review_v8_1 import load_review,LIGHT


def stats(values):
    v=sorted(values)
    return dict(count=len(v),min=min(v) if v else None,median=statistics.median(v) if v else None,
                p10=v[int((len(v)-1)*.1)] if v else None,p90=v[int((len(v)-1)*.9)] if v else None,
                max=max(v) if v else None)


def extract(review_path,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    report,plan,raw=load_review(review_path,LIGHT)
    sql=report['trace']['sqlite'];parsed_ref=report['trace']['parsed']
    parsed=json.loads(Path(parsed_ref['path']).read_text())
    sources=[receipt(review_path),*report['evidence'].values(),sql,parsed_ref,receipt(__file__)]
    c=sqlite3.connect('file:'+sql['path']+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    events,brackets,issues=context_intervals(c);mapping,attribution=controlled_creation(c,raw,events,brackets)
    runtime={(linux_pid(r['globalTid']),r['correlationId']):dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')}
    c.close();by_request={r['request_id']:r for r in raw['request_records']};byjob=defaultdict(list)
    ranges=defaultdict(list);kernels=defaultdict(list);kernel_phase_ranges=defaultdict(lambda:defaultdict(list))
    forward_ranges=defaultdict(list)
    semantic_ids={rid:{v['id']:v['semantic_phase'] for v in row['worker_response'].get('semantic_forward_observations',{}).get('records',[])}
                  for rid,row in by_request.items()}
    for phase in parsed['phases']:
        rid=phase['request_id']
        if rid in semantic_ids and phase['segment_id'] in semantic_ids[rid]:forward_ranges[rid].append(phase)
    for p in parsed['phases']:
        if p['request_id'] not in by_request:continue
        row=by_request[p['request_id']];alias=row['model']
        ranges[row['request_id']].append((p['nsys_start_ns'],p['nsys_end_ns']))
        for k in p['kernels']:
            api=runtime[raw['ready'][alias]['pid'],k['correlation_id']]
            kernels[row['request_id']].append((k['nsys_start_ns'],k['nsys_end_ns'],api['start'],api['end']))
            matches=[p for p in forward_ranges[row['request_id']] if p['nsys_start_ns']<=api['start'] and api['end']<=p['nsys_end_ns']]
            if len(matches)>1:raise ValueError('Ambiguous kernel API semantic membership')
            category=semantic_ids[row['request_id']][matches[0]['segment_id']] if matches else ('ocr_inference' if row['task']=='ocr' else 'outside_language_forwards')
            kernel_phase_ranges[row['request_id']][category].append((k['nsys_start_ns'],k['nsys_end_ns']))
    for row in raw['request_records']:
        byjob[row['profile_job_id']].append(row)
    jobs={j['id']:j for j in plan['jobs']};samples=[];censored=Counter();grouped=defaultdict(list)
    for jobid,rows in byjob.items():
        job=jobs[jobid];bounds={r['model']:(min(a for a,b in ranges[r['request_id']]),max(b for a,b in ranges[r['request_id']])) for r in rows}
        owner_rows={r['model']:r for r in rows}
        observed={r['model']:IntervalUnion((a,b) for a,b,_,_ in kernels[r['request_id']]) for r in rows}
        phase_observed={r['model']:{category:IntervalUnion(values) for category,values in kernel_phase_ranges[r['request_id']].items()} for r in rows}
        submitted={r['model']:IntervalUnion((min(lo,b),b) for a,b,lo,hi in kernels[r['request_id']]) for r in rows}
        # submitted spans are a separate descriptive queue proxy; not an
        # eligibility witness and never counted as fully-backlogged support.
        for alias,row in owner_rows.items():
            lo,hi=bounds[alias]
            for a,b in brackets[mapping[alias]]:
                if b<=lo or a>=hi:continue
                if a<lo or b>hi:
                    censored['envelope_crosses_request_boundary']+=1;continue
                peers=[p for p in owner_rows if p!=alias and bounds[p][0]<b and bounds[p][1]>a]
                if not peers:
                    censored['no_active_peer_request']+=1;continue
                own_full=observed[alias].overlap(a,b)==b-a
                eligible_peers=[p for p in peers if observed[p].overlap(a,b)==b-a]
                submitted_peers=[p for p in peers if submitted[p].overlap(a,b)==b-a]
                reason='complete_observed_unfinished_kernel_backlog' if own_full and eligible_peers else 'insufficient_continuous_kernel_backlog'
                # Label unfinished kernels by their launch API's static NVTX
                # membership. An asynchronous kernel may outlive Python phase
                # wall; phase-wall containment is not the criterion.
                categories={name for name,index in phase_observed[alias].items() if index.overlap(a,b)==b-a}
                category=next(iter(categories)) if len(categories)==1 else 'UNRESOLVED_PHASE'
                item=dict(job_id=jobid,job_name=job['name'],alias=alias,task=row['task'],pool_id=row['pool_id'],phase=category,
                    envelope_start_ns=a,envelope_end_ns=b,envelope_length_ns=b-a,
                    own_kernel_lifetime_union_covers_envelope=own_full,active_peer_requests=sorted(peers),
                    continuous_unfinished_kernel_peers=sorted(eligible_peers),
                    submitted_launch_proxy_peers=sorted(submitted_peers),
                    eligible_for_conditional_surrogate_budget=reason.startswith('complete') and category!='UNRESOLVED_PHASE',reason=reason)
                samples.append(item)
                if item['eligible_for_conditional_surrogate_budget'] and category!='UNRESOLVED_PHASE':
                    grouped[alias,row['task'],row['pool_id'],category,tuple(sorted(eligible_peers))].append(b-a)
                else:censored[reason]+=1
    groups=[dict(alias=k[0],task=k[1],pool_id=k[2],phase=k[3],continuous_peer_set=list(k[4]),envelope_length_ns=stats(v)) for k,v in grouped.items()]
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Training grant source changed')
    result=dict(schema='gsdeploy.context_envelope_training_samples.v8',status='CONSERVATIVE_TRAINING_SAMPLE_EXTRACTION',
        sources=sources,controlled_temporal_attribution=attribution,pairing_issues=issues,
        samples=samples,conditional_groups=groups,censored_reasons=dict(censored),
        complete_backlog_sample_count=sum(s['eligible_for_conditional_surrogate_budget'] for s in samples),
        submitted_launch_is_eligible_backlog=False,missing_context_events_filled=False,
        idle_gap_fitted_as_switch_cost=False,physical_quantum_identified=False,
        stateful_policy_fitted=False,GPU_started=False,validation_labels_used=False)
    save(out/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=extract(a.review,a.output)
    print(json.dumps({k:r[k] for k in ('status','complete_backlog_sample_count','conditional_groups','censored_reasons')}))


if __name__=='__main__':main()
