"""Describe per-call thread CPU, system CPU and elapsed time without subtracting stalls.

All formal semantic events and bands are retained. Outer thread brackets cover
existing observer overhead as well as the call; CPU and elapsed clocks have
separate envelopes. This is diagnosis, not a new prediction or acceptance rule.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.activation_completion_bands import completion_band_layout
from experiments.analyze_activation_host_diagnostic import validate_host_bracket,validate_trace,overlap_ms
from experiments.analyze_activation_semantic import summarize
from gsdeploy.forecasting.native_training import validate_capture


def describe_event(event):
    validate_host_bracket(event);b=event['passive_host_bracket']
    delta={k:b['usage_after'][k]-v for k,v in b['usage_before'].items()}
    wall=(b['wall_end_ns']-b['wall_start_ns'])/1e6
    cpu=(b['cpu_end_ns']-b['cpu_start_ns'])/1e6
    host=(event['host_end_ns']-event['host_start_ns'])/1e6
    return dict(component=event['component'],ordinal=event['ordinal'],thread_id=event['thread_id'],
        native_thread_id=b['native_thread_id'],host_start_ns=event['host_start_ns'],host_end_ns=event['host_end_ns'],
        outer_wall_start_ns=b['wall_start_ns'],outer_wall_end_ns=b['wall_end_ns'],
        semantic_host_ms=host,device_interval_ms=event['device_elapsed_ms'],
        outer_wall_ms=wall,thread_cpu_ms=cpu,outer_wall_minus_thread_cpu_ms=wall-cpu,
        outer_minus_semantic_host_ms=wall-host,usage_delta=delta,
        usage_user_ms=1000*delta['user_cpu_seconds'],usage_system_ms=1000*delta['system_cpu_seconds'])


def analyze(captures):
    sources=[];observations=[];bands=[];event_groups=defaultdict(list);band_groups=defaultdict(list)
    for path in captures:
        bundle=load_bundle(path.resolve());raw=bundle['raw'];plan=bundle['plan']
        validate_capture(bundle,allowed_splits=('development_host_diagnostic',))
        if plan['host_trace_mode']!='on' or plan['gc_trace_mode']!='on' or plan.get('not_for_training') is not True:
            raise ValueError('Explicit development host+GC trace arm required')
        sources.append(bundle['source'])
        for row in raw['request_records']:
            w=row['worker_response'];family=raw['resident_aliases'][row['model']];trace=w['passive_gc_trace']
            validate_trace(trace,w['start_perf_ns'],w['end_perf_ns'],raw['ready'][row['model']]['passive_gc_policy'])
            common=dict(capture=str(path.resolve()),request_id=row['request_id'],job=row['profile_job_id'],
                family=family,task=row['task'],pool_id=row['pool_id'])
            indices=[]
            for event in w['semantic_phase_events']:
                value=dict(common,**describe_event(event),
                    gc_overlap_ms=overlap_ms(trace['events'],event['host_start_ns'],event['host_end_ns']))
                index=len(observations);indices.append(index);observations.append(value)
                event_groups[family,row['task'],row['pool_id'],event['component'],event['ordinal']].append(index)
            for span in completion_band_layout(row,family):
                if span['kind']!='gpu':continue
                a,b=span['start_ns'],span['end_ns']
                selected=[i for i in indices if observations[i]['host_start_ns']<b and observations[i]['host_end_ns']>a]
                value=dict(common,band=span['native_band_role'],start_ns=a,end_ns=b,duration_ms=(b-a)/1e6,
                    gc_overlap_ms=overlap_ms(trace['events'],a,b),semantic_event_indices=selected,
                    scope='overlapping semantic host intervals; outer brackets can cross band boundaries, never sum nested CPU intervals')
                index=len(bands);bands.append(value);band_groups[family,row['task'],row['pool_id'],span['native_band_role']].append(index)
    repeats=[]
    for key,indices in sorted(event_groups.items()):
        values=[observations[i] for i in indices]
        if len(values)<2:continue
        repeats.append(dict(family=key[0],task=key[1],pool_id=key[2],component=key[3],ordinal=key[4],
            observation_indices=indices,outer_wall_ms=summarize([v['outer_wall_ms'] for v in values]),
            system_cpu_ms=summarize([v['usage_system_ms'] for v in values]),
            maximum_outer_wall_spread_ms=max(v['outer_wall_ms'] for v in values)-min(v['outer_wall_ms'] for v in values)))
    band_repeats=[]
    for key,indices in sorted(band_groups.items()):
        if len(indices)<2:continue
        values=[bands[i]['duration_ms'] for i in indices];lo,hi=min(values),max(values)
        band_repeats.append(dict(family=key[0],task=key[1],pool_id=key[2],band=key[3],n=len(indices),
            band_indices=indices,raw_constant_minimax=(hi-lo)/(hi+lo),minimum_ms=lo,maximum_ms=hi))
    groups=defaultdict(list)
    for value in observations:groups[value['family']+'/'+value['task']+'/'+value['component']].append(value)
    summary={k:dict(n=len(v),outer_wall_ms=summarize([x['outer_wall_ms'] for x in v]),
        thread_cpu_ms=summarize([x['thread_cpu_ms'] for x in v]),
        system_cpu_ms=summarize([x['usage_system_ms'] for x in v]),
        wall_minus_thread_cpu_ms=summarize([x['outer_wall_minus_thread_cpu_ms'] for x in v])) for k,v in sorted(groups.items())}
    return dict(schema='gsdeploy.host_bracket_diagnostic.v1',sources=sources,observations=observations,bands=bands,
        repeated_events=repeats,repeated_bands=band_repeats,path_summary=summary,
        physical_predictor_qualified=False,formal_utility_qualified=False,
        limitations='Observed thread/system CPU is descriptive. Wall minus CPU mixes GPU synchronization, blocking and descheduling. rusage encloses a slightly wider interval than CPU/wall clocks. No causal NUMA or CUDA API attribution. No latency subtraction. Raw minimax lacks root-order correction and is not a qualification gate.',
        analysis_sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'experiments/analyze_activation_host_diagnostic.py',ROOT/'experiments/activation_completion_bands.py']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',action='append',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.capture)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(captures=len(result['sources']),events=len(result['observations']),bands=len(result['bands']),
        repeated_band_groups=len(result['repeated_bands']),physical_predictor_qualified=False)))
