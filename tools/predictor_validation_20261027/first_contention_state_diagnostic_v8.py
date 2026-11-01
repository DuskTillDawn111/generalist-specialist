"""Compare real first-contention history to the frozen simulated SS-short state.

No scheduler or host parameter is fitted. Rejected observed entries remain in
the report, including cases where actual held context differs from prior ready
owner. An exact phase frontier is not assumed to identify context age.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import json
from pathlib import Path
import sqlite3

from .context_envelope_diagnostic_v8 import IntervalUnion,context_intervals
from .dag_host_release_v7 import checked
from .native_collect import receipt,save


def observed(report_path,job_name):
    ref=receipt(report_path);d=checked(ref)
    source=lambda suffix:next(s for s in d['sources'] if s['path'].endswith(suffix))
    raw=checked(source('/raw.json'));plan=checked(source('/plan.json'))
    job=next(j for j in plan['jobs'] if j['name']==job_name)
    rows={r['request_id']:r for r in raw['request_records'] if r['profile_job_id']==job['id']}
    obs=checked(d['kernel_ready_observations'])['rows'];obs=[r for r in obs if r['request_id'] in rows]
    sql=source('.sqlite')
    with sqlite3.connect(Path(sql['path']).as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row;_,brackets,issues=context_intervals(c)
    owner_context={w['alias']:tuple(w['context']) for w in d['controlled_temporal_attribution']['creation_windows']}
    envelopes=sorted((a,b,owner) for owner,k in owner_context.items() for a,b in brackets[k])
    envelope_starts=[a for a,b,p in envelopes]
    intervals=defaultdict(list)
    for row in obs:
        ready=row['certified_ready_bound_ns'];start=row['observed_start_ns']
        intervals[row['request_id'],row['phase']].append((min(ready,start) if ready is not None else start,row['observed_end_ns']))
    indexes={k:IntervalUnion(v) for k,v in intervals.items()}
    cuts=sorted({t for ix in indexes.values() for a,b in ix.intervals for t in (a,b)})
    origin=cuts[0]
    def state_at(t):
        grouped=defaultdict(list)
        for (rid,phase),ix in indexes.items():
            i=bisect_right(ix.starts,t)-1
            if i>=0 and t<ix.ends[i]:grouped[rid].append(phase)
        return [[rows[rid]['model'],rows[rid]['task'],rows[rid]['pool_id'],sorted(phases)] for rid,phases in sorted(grouped.items(),key=lambda kv:rows[kv[0]]['model'])]
    entries=[];outside_spans=[];first_spans=[];previous=[];initial=True
    for a,b in zip(cuts,cuts[1:]):
        state=state_at(a);owners={r[0] for r in state};old_owners={r[0] for r in previous}
        if a-origin<5_000_000:
            first_spans.append(dict(start_offset_ns=a-origin,end_offset_ns=b-origin,frontier=state))
        if len(owners)>=2 and all(r[3]==['outside_language_forwards'] for r in state):
            outside_spans.append(dict(start_offset_ns=a-origin,end_offset_ns=b-origin))
        if len(owners)>=2 and len(old_owners)<2:
            i=bisect_right(envelope_starts,a)-1
            held=envelopes[i] if i>=0 and a<envelopes[i][1] else None
            accepted=(held is not None and held[2] in owners and (not old_owners or held[2] in old_owners))
            reason='no_paired_context_envelope' if held is None else 'held_owner_not_in_ready_frontier' if held[2] not in owners else 'held_context_differs_from_previous_ready_owner' if old_owners and held[2] not in old_owners else None
            entries.append(dict(offset_ns=a-origin,frontier=state,previous_frontier=previous,
                observed_held_owner=held[2] if held else None,
                observed_context_envelope_age_ns=a-held[0] if held else None,
                observed_residual_to_envelope_end_ns=held[1]-a if held else None,
                original_fitter_admits_entry_sample=accepted,original_fitter_rejection_reason=reason,
                original_fitter_reason='initial' if initial else 'contention_entry'))
        if state:initial=False
        previous=state
    phases=[]
    for rid,row in rows.items():
        selected=[r for r in obs if r['request_id']==rid]
        first_by_phase={phase:min(r['observed_start_ns'] for r in selected if r['phase']==phase)-origin for phase in {r['phase'] for r in selected}}
        phases.append(dict(request_id=rid,owner=row['model'],first_GPU_start_offset_ns=min(r['observed_start_ns'] for r in selected)-origin,
            first_pending_bound_offset_ns=min(min(r['certified_ready_bound_ns'],r['observed_start_ns']) if r['certified_ready_bound_ns'] is not None else r['observed_start_ns'] for r in selected)-origin,
            first_GPU_by_phase_offset_ns=first_by_phase,
            handler_start_perf_ns=row['worker_response']['start_perf_ns'],
            actual_submit_perf_ns=row['actual_submit_time_ns'],
            first_semantic_from_handler_offset_ns=row['worker_response']['semantic_forward_observations']['records'][0]['start_ns']-row['worker_response']['start_perf_ns']))
    sources=[ref,*d['sources'],d['kernel_ready_observations']]
    if any(receipt(s['path'])!=s for s in sources):raise ValueError('First-contention source changed')
    return dict(source_report=ref,sources=sources,job_id=job['id'],job_name=job_name,request_phases=phases,
        ready_timeline_origin_trace_ns=origin,first_five_ms_frontier_spans=first_spans,
        all_contention_entry_observations=entries,
        outside_outside_span_count=len(outside_spans),outside_outside_total_ns=sum(r['end_offset_ns']-r['start_offset_ns'] for r in outside_spans),
        outside_outside_spans=outside_spans,
        entry_rejection_counts=dict(Counter(r['original_fitter_rejection_reason'] for r in entries if not r['original_fitter_admits_entry_sample'])),
        context_pairing_issues=issues,default_stream_dependency_identified=False)


def analyze(report_paths,simulation_path,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    sim_ref=receipt(simulation_path);sim=checked(sim_ref)
    actual=[observed(p,'SS-short') for p in report_paths]
    frozen=[r for r in sim['checks'] if r['job_name']=='SS-short']
    sources=[receipt(__file__),sim_ref,receipt(Path(__file__).with_name('dag_availability_complete_fit_v8.py')),
             receipt(Path(__file__).with_name('trained_dag_availability_v8.py'))]
    for r in actual:sources.extend(r['sources'])
    sources=list({r['path']:r for r in sources}.values())
    if any(receipt(r['path'])!=r for r in sources):raise ValueError('First-contention evidence changed')
    result=dict(schema='gsdeploy.first_contention_frontier_history_audit.v8',sources=sources,
        observed_captures=actual,frozen_simulation_checks=frozen,
        conclusions=['Observed early host/API request order varies by independent run; pooled solo prefix can reverse the co-run first owner',
            'Outside-language label combines early preparation kernels with late generation/postprocessing kernels',
            'Observed context holder need not equal the previous owner with pending GPU work; original entry fitter censors that case',
            'Phase/current-owner policy state omits context-envelope age and pending-work amount, so Markov sufficiency is unproven',
            'A residual to envelope end is a censored availability observation, not a globally fixed hardware quantum'],
        host_model_correctness_established=False,phase_frontier_Markov_sufficiency_established=False,
        new_fit_performed=False,validation_labels_used=False,GPU_started=False)
    save(out/'report.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',action='append',required=True)
    p.add_argument('--simulation',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=analyze(a.report,a.simulation,a.output)
    print(json.dumps(dict(report=receipt(Path(a.output)/'report.json'),captures=len(r['observed_captures']))))

if __name__=='__main__':main()
