"""Score full solo outputs against unchanged references and quality anchors."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_training import validate_capture
from gsdeploy.normalization import quality_loss
from gsdeploy.quality import quality_digest, sample_totals
from gsdeploy.tasks import page_ocr_text, score_result


def describe(bundle, data):
    validate_capture(bundle)
    if bundle['plan']['mode'] != 'solo':
        raise ValueError('Complete full-input solo training required')
    quality = data['quality']
    if (quality_digest(quality['references']) != quality['identity']['reference_sha256']
            or bundle['context']['input_pool_sha256'] != quality['identity']['input_pool_sha256']):
        raise ValueError('Exact unchanged input/reference identity required')
    expected = {(family, task, pool_id) for family in bundle['raw']['resident_aliases'].values()
        for task in quality['input_order'] if family == 'generalist_2' or task == family
        for pool_id in quality['input_order'][task]}
    seen = set()
    totals = defaultdict(lambda: [0., 0., 0., 0., 0])
    scores = []
    for row in bundle['raw']['request_records']:
        family = bundle['raw']['resident_aliases'][row['model']]
        task, pool_id = row['task'], row['pool_id']
        key = family, task, pool_id
        if key in seen:
            raise ValueError('Exactly one output per full-input profile required')
        seen.add(key)
        result = dict(row['worker_response'])
        if result.get('success') is not True:
            raise ValueError('Unsuccessful output cannot be scored as ordinary quality')
        result['task'] = result.get('task', result.get('task_type'))
        if task == 'ocr':
            result['prediction'] = page_ocr_text(result['prediction'])
        score = score_result(quality['references'][task][pool_id], result)
        a, z = sample_totals(task, score)
        old = quality['paths'][family][task]['samples'][pool_id]
        oa, oz = sample_totals(task, old)
        total = totals[family, task]
        for index, value in enumerate((a, z, oa, oz, 1)):
            total[index] += value
        scores.append(dict(family=family, task=task, pool_id=pool_id, request_id=row['request_id'],
            error_numerator=a, reference_denominator=z,
            catalog_error_numerator=oa, catalog_reference_denominator=oz))
    if seen != expected:
        raise ValueError('Complete declared input pool missing from solo quality cohort')
    summary = {}
    for (family, task), (a, z, oa, oz, n) in totals.items():
        loss, old_loss = a/z, oa/oz
        norm, old_norm = quality_loss(data, task, loss), quality_loss(data, task, old_loss)
        summary[family+'/'+task] = dict(profiles=n, native_loss=loss, catalog_native_loss=old_loss,
            normalized_loss=norm, catalog_normalized_loss=old_norm,
            normalized_loss_delta=norm-old_norm if norm is not None and old_norm is not None else None)
    return dict(source=bundle['source'], execution_context_sha256=bundle['plan']['execution_context_sha256'],
                per_path=summary, scores=scores)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', action='append', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    reference=args.reference/'system-frozen.json'
    data=json.loads(reference.read_text())['catalog']
    runs=[describe(load_bundle(path.resolve()),data) for path in args.capture]
    report=dict(schema='gsdeploy.actual_solo_output_quality.v1', runs=runs,
        reference_identity=data['quality']['identity'],
        quality_anchors=data['utility_normalization']['tasks'],
        system_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='complete actual solo output quality; unchanged quality references and anchors; no episode energy denominator',
        Q_norm_is_loss=True, negative_delta_means_quality_improvement=True,
        physical_predictor_qualified=False, formal_utility_qualified=False, U=None)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(args.output), runs=[r['per_path'] for r in runs])))


if __name__=='__main__':main()
