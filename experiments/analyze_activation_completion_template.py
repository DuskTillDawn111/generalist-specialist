"""Training-only baseline residuals under the explicit completion-band policy."""
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
from experiments.fit_activation_completion_bands import build_band_template
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_order import NativeOrderLaw, request_orders


def analyze(bundles, pool):
    template,_,_=build_band_template(bundles,pool)
    order=NativeOrderLaw(template['native_order_law'],template['execution_context_sha256'])
    lookup={(p['family'],p['task'],p['pool_id']):p for p in template['profiles']}
    groups=defaultdict(list)
    for bundle in bundles:
        orders=request_orders(bundle['raw'])
        for row in bundle['raw']['request_records']:
            family=bundle['raw']['resident_aliases'][row['model']]
            profile=lookup[family,row['task'],row['pool_id']]
            means={s['id']:s['duration_ms']['mean'] for s in profile['segments']}
            for span in completion_band_layout(row,family):
                if span['kind']!='gpu':continue
                factor=order.factor(profile,dict(id=span['native_root_id']),orders[row['request_id']])
                error=abs((span['end_ns']-span['start_ns'])/1e6/(means[span['id']]*factor)-1.)
                groups[family+'/'+row['task']+'/'+span['native_band_role']].append(error)
    return dict(schema='gsdeploy.completion_template_solo_diagnostic.v1',
        scope='training-only solo baseline after root-order adjustment; no concurrent progress, FIFO or energy qualification',
        execution_context_sha256=template['execution_context_sha256'],sources=[b['source'] for b in bundles],
        physical_predictor_qualified=False,formal_utility_qualified=False,
        paths={k:dict(summarize(v),over_point2=sum(x>.2 for x in v)) for k,v in sorted(groups.items())})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,action='append',required=True)
    p.add_argument('--pool',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();content=a.pool.read_bytes();bundles=[load_bundle(path.resolve()) for path in a.capture]
    if any(b['context']['input_pool_sha256']!=hashlib.sha256(content).hexdigest() for b in bundles):
        raise ValueError('Frozen training input pool changed')
    result=analyze(bundles,[json.loads(line) for line in content.decode().splitlines()])
    result['analysis_sources']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__).resolve(),ROOT/'experiments/fit_activation_completion_bands.py',
        ROOT/'experiments/activation_completion_bands.py',ROOT/'experiments/activation_semantic_bands.py',
        ROOT/'gsdeploy/forecasting/native_order.py',ROOT/'gsdeploy/forecasting/native_training.py')}
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result['paths']))
