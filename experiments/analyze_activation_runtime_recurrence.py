"""Describe repeated-input timing within runtime contrasts, without fitting.

All preceding priming, solo and co-run requests count toward owner/input
ordinals. These observations cannot establish a cache or kernel mechanism.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.activation_completion_bands import completion_band_layout
from experiments.analyze_activation_semantic import summarize
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_training import validate_capture


def analyze(base):
    runs=[]
    for name in ('strict-round1','scoped-round1','scoped-round2','strict-round2'):
        bundle=load_bundle(base/name);validate_capture(bundle,allowed_splits=('development_runtime_scope_diagnostic',))
        raw=bundle['raw'];counts=defaultdict(int);solo=defaultdict(list)
        for row in raw['priming_request_records']+raw['request_records']:
            key=row['model'],row['task'],row['pool_id']
            prior=counts[key];counts[key]+=1
            if not row.get('profile_job_id','').startswith('solo:'):continue
            family=raw['resident_aliases'][row['model']]
            solo[key].append(dict(request_id=row['request_id'],earlier_same_input_requests=prior,
                handler_ms=row['worker_response']['worker_latency_ms'],
                bands_ms={s['native_band_role']:(s['end_ns']-s['start_ns'])/1e6
                          for s in completion_band_layout(row,family) if s['kind']=='gpu'}))
        profiles=[];paths=defaultdict(list)
        for key,rows in sorted(solo.items()):
            if len(rows)!=2:raise ValueError('Expected exactly two solo repeats per selected input')
            first,second=rows;family=raw['resident_aliases'][key[0]]
            ratio=second['handler_ms']/first['handler_ms'];paths[family+'/'+key[1]].append(ratio)
            profiles.append(dict(family=family,task=key[1],pool_id=key[2],observations=rows,
                later_over_earlier_handler=ratio,
                later_over_earlier_bands={k:second['bands_ms'][k]/v for k,v in first['bands_ms'].items()}))
        runs.append(dict(run_id=name,source=bundle['source'],scope=bundle['plan']['runtime_scope'],
            profiles=profiles,path_later_over_earlier={k:summarize(v) for k,v in paths.items()}))
    return dict(schema='gsdeploy.runtime_input_recurrence_description.v1',runs=runs,
        scope='development-only within-capture repeat description; no fit or heldout qualification',
        limitations=['request order, predecessor history and elapsed time covary with repeated input',
            'earlier input count includes co-runs and priming; no first-use cache mechanism asserted',
            'different execution contexts require their own training and observer checks'],
        physical_predictor_qualified=False,formal_utility_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    report=analyze(a.base.resolve());report['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({r['run_id']:r['path_later_over_earlier'].get('ocr/ocr') for r in report['runs']}))
