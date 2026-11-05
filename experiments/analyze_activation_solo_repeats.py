"""Full-input fresh solo repeat diagnosis, without fitting future templates."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.activation_semantic_bands import band_layout
from experiments.analyze_activation_semantic import output_identity
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_training import digest, validate_capture


def analyze(bundles):
    contexts = {validate_capture(bundle) for bundle in bundles}
    if len(contexts) != 1 or len({b['raw']['run_id'] for b in bundles}) != len(bundles) or len(bundles) < 2:
        raise ValueError('Independent complete training context repeats required')
    roster = None
    samples = defaultdict(list)
    for bundle in bundles:
        if bundle['plan']['mode'] != 'solo' or any(len(j['requests']) != 1 for j in bundle['plan']['jobs']):
            raise ValueError('Solo training data required')
        keys = set()
        for row in bundle['raw']['request_records']:
            family = bundle['raw']['resident_aliases'][row['model']]
            key = family, row['task'], row['pool_id']
            if key in keys:
                raise ValueError('Unique full-input solo profile required')
            keys.add(key)
            bands = band_layout(row, family)
            samples[key].append(dict(run_id=bundle['raw']['run_id'], request_id=row['request_id'],
                output_identity_sha256=digest(output_identity(row['worker_response'])),
                handler_ms=row['worker_response']['worker_latency_ms'],
                bands_ms={s['id']: (s['end_ns']-s['start_ns'])/1e6 for s in bands if s['kind']=='gpu'},
                generated_tokens=row['worker_response'].get('generated_tokens')))
        if roster is not None and keys != roster:
            raise ValueError('Repeated complete input rosters differ')
        roster = keys
    contrasts = []
    paths = defaultdict(list)
    for key, rows in sorted(samples.items()):
        if any(set(r['bands_ms']) != set(rows[0]['bands_ms']) for r in rows):
            raise ValueError('Cross-run observed band layouts differ')
        durations = [r['handler_ms'] for r in rows]
        ratio = max(durations)/min(durations)
        paths[key[0]+'/'+key[1]].append(ratio)
        contrasts.append(dict(family=key[0], task=key[1], pool_id=key[2],
            output_changed=len({r['output_identity_sha256'] for r in rows})>1,
            handler_maximum_to_minimum=ratio,
            band_maximum_to_minimum={band: max(r['bands_ms'][band] for r in rows)/min(r['bands_ms'][band] for r in rows)
                                     for band in rows[0]['bands_ms']}, observations=rows))
    return dict(schema='gsdeploy.complete_solo_repeat_description.v1', execution_context_sha256=next(iter(contexts)),
        sources=[b['source'] for b in bundles], profiles=len(contrasts),
        changed_output_profiles=[{k:r[k] for k in ('family','task','pool_id')} for r in contrasts if r['output_changed']],
        path_handler_repeat_ratios={path: dict(profiles=len(v), mean=statistics.mean(v), maximum=max(v)) for path,v in paths.items()},
        contrasts=contrasts, physical_predictor_qualified=False, formal_utility_qualified=False,
        limitations=['training-only repeat diagnosis; no measured future stopping or forward count may enter prediction',
                     'stable solo output or band layout does not qualify concurrent progress or FIFO accuracy'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', required=True, action='append', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args=p.parse_args()
    report=analyze([load_bundle(path.resolve()) for path in args.capture])
    report['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('contrasts','sources')}))


if __name__=='__main__':main()
