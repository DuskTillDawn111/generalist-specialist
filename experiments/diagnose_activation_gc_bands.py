"""All repeated completion bands, with GC overlap retained as an observed covariate.

No GC duration is subtracted from measured service or qualification errors.
This diagnostic never trains a predictor on after-the-fact GC observations.
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
from experiments.analyze_activation_gc_diagnostic import overlap_ms
from experiments.analyze_activation_semantic import summarize
from gsdeploy.forecasting.native_training import validate_capture


def analyze(base):
    report=json.loads((base/'description.json').read_text());groups=defaultdict(list);by_path=defaultdict(list)
    for name in ['control-round1','trace-round1','trace-round2','control-round2']:
        bundle=load_bundle(base/name);validate_capture(bundle,allowed_splits=('development_gc_diagnostic',))
        if bundle['source']!=report['sources'][name]:raise ValueError('Diagnostic raw source differs from observer analysis')
        for row in bundle['raw']['request_records']:
            family=bundle['raw']['resident_aliases'][row['model']];response=row['worker_response']
            trace=response.get('passive_gc_trace');mode='trace' if trace else 'control'
            for span in completion_band_layout(row,family):
                if span['kind']!='gpu':continue
                a,b=span['start_ns'],span['end_ns'];role=span['native_band_role']
                item=dict(capture=name,request_id=row['request_id'],job=row['profile_job_id'],
                    duration_ms=(b-a)/1e6,start_ns=a,end_ns=b,
                    gc_overlap_ms=overlap_ms(trace['events'],a,b) if trace else None,
                    gc_events=[e for e in trace['events'] if e['start_ns']<b and e['end_ns']>a] if trace else None)
                groups[mode,family,row['task'],role,row['pool_id']].append(item)
                by_path[mode,family,row['task'],role].append(item)
    profiles=[]
    for key,rows in sorted(groups.items()):
        values=[r['duration_ms'] for r in rows];low,high=min(values),max(values)
        profiles.append(dict(mode=key[0],family=key[1],task=key[2],band=key[3],pool_id=key[4],n=len(rows),
            raw_constant_minimax_residual=(high-low)/(high+low),minimum_ms=low,maximum_ms=high,
            observations=rows))
    summaries=[]
    for key,rows in sorted(by_path.items()):
        summaries.append(dict(mode=key[0],family=key[1],task=key[2],band=key[3],
            duration_ms=summarize([r['duration_ms'] for r in rows]),
            gc_overlap_ms=summarize([r['gc_overlap_ms'] for r in rows if r['gc_overlap_ms'] is not None]),
            gc_affected_observations=sum((r['gc_overlap_ms'] or 0)>0 for r in rows)))
    return dict(schema='gsdeploy.passive_gc_band_diagnostic.v1',sources=report['sources'],
        observer_repeat_check_passed=report['observer_repeat_check_passed'],observer_issues=report['issues'],
        profiles=profiles,path_summaries=summaries,
        interpretation='Raw constant minimax is a descriptive repeated-input range, without order correction; not a qualification gate or impossibility claim for state-dependent predictors. GC overlap is post-hoc correlation, never an ex-ante prediction input. All bands and samples retained; no GC subtraction.',
        physical_predictor_qualified=False,formal_utility_qualified=False,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.base)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(profiles=len(result['profiles']),observer_repeat_check_passed=result['observer_repeat_check_passed'],
        raw_constant_residual_over_point2=sum(p['raw_constant_minimax_residual']>.2 for p in result['profiles']))))
