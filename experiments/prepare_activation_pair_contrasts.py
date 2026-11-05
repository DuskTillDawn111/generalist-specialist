"""Repeat a missing G2/text + image activity vertex in an unchanged resident domain."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import POOL, prepare, save
from gsdeploy.forecasting.native_training import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', required=True, type=Path)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', required=True, type=int)
    args = p.parse_args()
    fit = json.loads(args.fit.read_text())
    origin = Path(fit['source_receipts'][0].split(':sha256:')[0]).parent
    context = json.loads((origin/'context.json').read_text())
    aliases = context['resident_aliases']
    ocr_count = sum(f == 'ocr' for f in aliases.values())
    expected = {'generalist_2#0': 'generalist_2', 'image#0': 'image',
                **{f'ocr#{i}': 'ocr' for i in range(ocr_count)}}
    if aliases != expected or ocr_count not in (1, 2):
        raise ValueError('Matched G2 + image + OCR resident domain required')
    output = args.output.resolve()
    prepare(output, args.reference.resolve(), image=True, ocr_replicas=ocr_count,
            mode='overlap', seed=args.seed)
    plan = json.loads((output/'plan.json').read_text())
    if plan['execution_context_sha256'] != fit['profiles']['execution_context_sha256']:
        raise ValueError('Collector or residency changed since original fit')
    pool = [json.loads(line) for line in POOL.read_text().splitlines()]
    text = [r for r in pool if r['task_type'] == 'text']
    image = [r for r in pool if r['task_type'] == 'image']
    if len(image) != 2*len(text):
        raise ValueError('Complete 32 text / 64 image marginal cohorts required')
    jobs = []
    for i, item in enumerate(text):
        # Two image calls share one owner. The second queues behind the first,
        # covering a later G2 call age without observing its eventual end.
        requests = [dict(alias='generalist_2#0', task='text', pool_id=item['pool_id'], offset_ms=0.)]
        for j in range(2):
            requests.append(dict(alias='image#0', task='image',
                pool_id=image[(2*i+j+13) % len(image)]['pool_id'], offset_ms=0.))
        jobs.append(dict(id=f'pair_text_image:{i}', requests=requests))
    random.Random(args.seed).shuffle(jobs)
    provenance = dict(schema='gsdeploy.native_pair_contrast_design.v1', split='training',
        source_fit=str(args.fit.resolve()), source_fit_sha256=digest(fit),
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reason='G2/text and image-only count vertices appeared in only one independent training run',
        design='all 32 text and 64 image inputs; two FIFO image requests per text call; OCR remains resident and idle',
        independent_job_order_seed=args.seed, no_validation_or_confirmation_data_used=True,
        no_future_stopping_features=True, admission_enabled=False)
    plan.update(jobs=jobs, diagnostic_supplement=True, pair_contrast_design=provenance,
                not_formal_utility=True)
    save(output/'plan.json', plan)
    save(output/'pair-contrast-design.json', provenance)
    print(json.dumps(dict(output=str(output), jobs=len(jobs), requests=3*len(jobs),
                          context=plan['execution_context_sha256'])))


if __name__ == '__main__':
    main()
