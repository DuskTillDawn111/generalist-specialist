"""Training-only repeat limits and asynchronous host/CUDA boundary diagnostics."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.activation_semantic_bands import band_layout
from experiments.analyze_activation_semantic import output_identity, summarize
from gsdeploy.forecasting.native_training import validate_capture


def analyze(bundles):
    if len(bundles)!=2 or len({validate_capture(b) for b in bundles})!=1:
        raise ValueError('Two complete training solo repeats of one context required')
    if any(b['plan']['mode']!='solo' for b in bundles):raise ValueError('Solo repeats required')
    keyed=[]
    for b in bundles:
        rows={}
        for row in b['raw']['request_records']:
            family=b['raw']['resident_aliases'][row['model']];key=(family,row['task'],row['pool_id'])
            if key in rows:raise ValueError('Unique complete input profile required')
            bands=band_layout(row,family)
            events=row['worker_response']['semantic_phase_events']
            component='ocr_detection' if family=='ocr' else 'language' if family=='generalist_2' else 'image_features' if family=='image' else 'model_forward'
            selected=[e for e in events if e['component']==component]
            if not selected:raise ValueError('Expected component missing')
            first=min(selected,key=lambda e:e['host_start_ns'])
            rows[key]=dict(handler_ms=row['worker_response']['worker_latency_ms'],
                bands_ms={s['id']:(s['end_ns']-s['start_ns'])/1e6 for s in bands if s['kind']=='gpu'},
                first_component_host_ms=(first['host_end_ns']-first['host_start_ns'])/1e6,
                first_component_CUDA_event_ms=first['device_elapsed_ms'],
                first_component=component,output=output_identity(row['worker_response']))
        keyed.append(rows)
    if set(keyed[0])!=set(keyed[1]):raise ValueError('Complete input profiles differ')
    contrasts=[];groups=defaultdict(list)
    for key,left in sorted(keyed[0].items()):
        right=keyed[1][key]
        if left['output']!=right['output']:raise ValueError('Output changed across training repeats')
        if set(left['bands_ms'])!=set(right['bands_ms']):raise ValueError('Band layout changed')
        measures={'handler':(left['handler_ms'],right['handler_ms']),
            **{band:(left['bands_ms'][band],right['bands_ms'][band]) for band in left['bands_ms']},
            'first_component_host':(left['first_component_host_ms'],right['first_component_host_ms']),
            'first_component_CUDA_event':(left['first_component_CUDA_event_ms'],right['first_component_CUDA_event_ms'])}
        for measure,(a,b) in measures.items():
            if min(a,b)<=0:raise ValueError('Positive measured durations required')
            # For a single D, minimizing max(|a/D-1|,|b/D-1|) yields D=(a+b)/2.
            error=abs(b-a)/(a+b)
            row=dict(family=key[0],task=key[1],pool_id=key[2],measure=measure,
                run1_ms=a,run2_ms=b,max_min_ratio=max(a,b)/min(a,b),
                constant_baseline_best_possible_max_relative_error=error,
                constant_baseline_cannot_fit_both_at_20_percent=error>.2,
                run1_predict_run2_relative_error=abs(b-a)/b,
                run2_predict_run1_relative_error=abs(a-b)/a)
            contrasts.append(row);groups[key[0]+'/'+key[1]+':'+measure].append(row)
    return dict(schema='gsdeploy.training_boundary_stability.v1',sources=[b['source'] for b in bundles],
        context=bundles[0]['raw']['execution_context_sha256'],contrasts=contrasts,
        groups={k:dict(profiles=len(rows),maximum_ratio=max(r['max_min_ratio'] for r in rows),
            cannot_fit_constant_at_20_percent=sum(r['constant_baseline_cannot_fit_both_at_20_percent'] for r in rows),
            run1_predict_run2=summarize([r['run1_predict_run2_relative_error'] for r in rows]),
            run2_predict_run1=summarize([r['run2_predict_run1_relative_error'] for r in rows])) for k,rows in groups.items()},
        physical_predictor_qualified=False,formal_utility_qualified=False,
        limitations=['constant-baseline obstruction applies only without changing state features; not an impossibility result for richer predictors',
            'two training runs cannot certify held-out accuracy or concurrency behavior',
            'host return boundary and CUDA event stream interval are different clocks/scopes; no absolute clock alignment inferred',
            'CUDA events include scheduling gaps; no kernel-only resource or hardware-counter claim',
            'coarser native-call stability motivates a model comparison; it cannot certify that coarser phases predict co-run interference'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',action='append',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    r=analyze([load_bundle(path.resolve()) for path in a.capture]);r['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(a.output),groups=r['groups'])))
