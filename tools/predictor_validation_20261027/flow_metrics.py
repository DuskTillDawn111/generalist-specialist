"""CPU diagnostic six-term arithmetic for native flow parent records.

This helper cannot grant stability, source identity, arrival cohort or physical
eligibility. Actual parent completion timestamps define E2E and throughput.
Startup board energy includes load/priming/warmup up to measurement start.
"""
from __future__ import annotations

import math
import statistics

from gsdeploy.energy import integrate_power,energy_norm
from gsdeploy.metrics import percentile


def _number(value,*,positive=False):
    return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0)


def score_flow(raw,plan):
    if raw.get('schema')!='gsdeploy.native_flow_collection.v1':raise ValueError('Explicit native flow raw schema required')
    rates=plan['workload_rps'];refs=plan['latency_reference_ms'];weights=plan['weights']
    if not rates or set(refs)!=set(rates) or any(not _number(v,positive=True) for v in [*rates.values(),*refs.values()]):
        raise ValueError('Positive offered rates and full latency references required')
    if set(weights)!=set('QTLPME') or any(not _number(v) for v in weights.values()) or abs(sum(weights.values())-1)>1e-12:
        raise ValueError('Complete normalized six-term weights required')
    safe=plan['memory_safe_factor'];budget=plan['memory_budget_mib'];watts=plan['energy_reference_watts'];amortization=plan['startup_amortization_seconds']
    if not all(_number(v,positive=True) for v in (safe,budget,watts,amortization)) or safe<1:raise ValueError('Original positive resource/reference constants required')
    start,end,drain=raw['phase_start_ns'],raw['deadline_ns'],raw['observation_end_ns']
    startup=raw['startup_begin_ns']
    if any(type(v) is not int for v in (start,end,drain,startup)) or not startup<=start<end<=drain:raise ValueError('Actual startup/measurement/drain chronology required')
    horizon=(end-start)/1e9;records=raw['request_records'];seen=set();by_task={t:[] for t in rates};reasons=[]
    for row in records:
        rid=row['request_id'];task=row['task'];scheduled=row['scheduled_time_ns']
        if rid in seen or task not in rates or type(scheduled) is not int or not start<=scheduled<end:raise ValueError('Unique measurement-cohort request and offered task required')
        seen.add(rid);by_task[task].append(row)
        if row['status']=='completed':
            completed=row.get('completion_time_ns')
            if type(completed) is not int or not scheduled<=completed<=drain:raise ValueError('Actual parent completion within observed drain required')
    completed=[r for r in records if r['status']=='completed'];within=sum(r['completion_time_ns']<=end for r in completed)
    terms={'Q':None,'T':1-within/len(records) if records else None,'L':None,'P':None,'M':None,'E':None}
    if not records:reasons.append('EMPTY_MEASUREMENT_COHORT')
    stats={};complete_tasks=True;total=sum(rates.values())
    for task,rows in by_task.items():
        done=[r for r in rows if r['status']=='completed'];latencies=[(r['completion_time_ns']-r['scheduled_time_ns'])/1e6 for r in done]
        stats[task]={'offered':len(rows),'completed':len(done),'completed_within_window':sum(r['completion_time_ns']<=end for r in done),
                     'mean_e2e_ms':statistics.mean(latencies) if latencies else None,'p95_e2e_ms':percentile(latencies,95),
                     'latency_population':'Actual parent completed measurement cohort, including bounded drain completions'}
        if not latencies:complete_tasks=False;reasons.append('MISSING_TASK_LATENCY:'+task)
    if complete_tasks:
        terms['L']=sum(rates[t]/total*stats[t]['mean_e2e_ms']/refs[t] for t in rates)
        terms['P']=sum(rates[t]/total*stats[t]['p95_e2e_ms']/refs[t] for t in rates)
    # Existing runtime reports diagnostics from successful completed requests,
    # while qualification separately requires completion_ratio==1.
    if len(completed)!=len(records):reasons.append('INCOMPLETE_COHORT_LATENCIES_DIAGNOSTIC_ONLY')
    quality=raw.get('quality_evidence',{}).get('totals_by_request',{})
    totals={t:{'numerator':0.,'denominator':0.,'requests':0} for t in rates};qknown=True
    for row in records:
        score=quality.get(row['request_id'])
        if row['status']!='completed' or not isinstance(score,dict) or score.get('task')!=row['task']:
            qknown=False;continue
        numerator,denominator=score.get('numerator'),score.get('denominator')
        if not _number(numerator) or not _number(denominator,positive=True):qknown=False;continue
        if row['task'] in ('text','image') and numerator>denominator:raise ValueError('Text/image loss totals must be <=1')
        target=totals[row['task']];target['numerator']+=numerator;target['denominator']+=denominator;target['requests']+=1
    if qknown and all(v['denominator']>0 for v in totals.values()):terms['Q']=sum(rates[t]/total*v['numerator']/v['denominator'] for t,v in totals.items())
    else:reasons.append('MISSING_ACTUAL_QUALITY_TOTALS')
    samples=raw['board_samples'];memory=[];foreign=[];missing=[];invalid_memory=False
    times=[s.get('time_ns') for s in samples]
    if any(type(t) is not int for t in times) or any(a>=b for a,b in zip(times,times[1:])):
        raise ValueError('Strictly increasing actual telemetry timestamps required')
    valid_power=all(_number(s.get('watts')) for s in samples)
    expected=set(raw.get('resident_aliases',plan.get('resident_aliases',[])))
    # Parent may provide replica_to_family instead of a duplicate alias list.
    if not expected:expected=set(raw.get('replica_to_family',{}))
    for sample in samples:
        if sample.get('foreign_pids'):foreign.extend(sample['foreign_pids'])
        if not _number(sample.get('owned_used_mib')):
            invalid_memory=True
            continue
        if expected and set(sample.get('owned_aliases',[]))!=expected:
            # During loading residents may be absent; raw owned evidence is
            # still legitimate, but full resident coverage after ready required.
            if sample['time_ns']>=raw.get('startup_end_ns',start):missing.append(sample['time_ns'])
        memory.append(sample['owned_used_mib'])
    if foreign:reasons.append('FOREIGN_GPU_PROCESSES')
    if missing:reasons.append('MISSING_OWNED_RESIDENT_TELEMETRY')
    if invalid_memory:reasons.append('INVALID_OR_MISSING_RAW_OWNED_MEMORY_SAMPLE')
    if memory and not invalid_memory and not foreign and not missing and expected:terms['M']=safe*max(memory)/budget
    else:reasons.append('UNKNOWN_RAW_OWNED_PEAK_MEMORY')
    run_j=integrate_power(samples,start,end) if valid_power else None
    startup_j=integrate_power(samples,startup,start) if valid_power else None
    energy={'kind':'measured' if raw.get('kind')=='measured' and raw.get('backend')=='native' else 'synthetic_fixture',
            'scope':'whole_gpu_board','coverage_valid':run_j is not None and startup_j is not None and not foreign,
            'run_joules':run_j,'startup_joules':startup_j}
    terms['E']=energy_norm(energy,horizon,watts,amortization)
    if terms['E'] is None:reasons.append('UNKNOWN_BOARD_ENERGY_COVERAGE')
    complete=all(v is not None for v in terms.values());diagnostic=-sum(weights[t]*v for t,v in terms.items() if v is not None)
    return {'schema':'gsdeploy.native_flow_six_term_diagnostics.v1','terms':terms,
        'diagnostic_U':diagnostic if complete else None,'diagnostic_partial_U':diagnostic,'utility_complete':complete,
        'per_task':stats,'quality_totals_by_task':totals,'offered':len(records),'completed':len(completed),
        'completed_within_window':within,'completion_ratio':len(completed)/len(records) if records else None,
        'memory':{'raw_owned_sampled_peak_mib':max(memory) if memory and not invalid_memory else None,'safe_factor':safe,
                  'safe_peak_mib':safe*max(memory) if memory and not invalid_memory else None,'safe_factor_application_count':1,
                  'budget_mib':budget,'budget_satisfied':terms['M']<=1 if terms['M'] is not None else None,
                  'capacity_bound':False,'scope':'All owned worker sampled peak including idle residents and startup/prime/warmup/drain samples'},
        'energy':{**energy,'startup_scope':'startup_begin through measurement phase_start, including ready/prime/warmup; startup_end ready marker does not truncate energy',
                  'run_scope':'measurement phase_start..deadline; bounded drain energy excluded','startup_amortization_seconds':amortization},
        'unknown_or_ineligible_reasons':sorted(set(reasons)),'eligible':None,
        'eligibility_scope':'Caller must separately establish source/cohort/quality identity, original stability and native resource protocol; this helper grants no eligibility',
        'GPU_started':False}
