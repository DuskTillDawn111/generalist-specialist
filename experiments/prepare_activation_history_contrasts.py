"""Training-only paired histories; target requests stay serial and unchanged."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import POOL, digest, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    args = p.parse_args()
    if args.seed in (6110491, 6110492):
        raise ValueError('Confirmation seeds reserved')
    fit = json.loads(args.fit.read_text())
    diagnostic = json.loads(args.diagnostic.read_text())
    if diagnostic['training_fit_sha256'] != digest(fit):
        raise ValueError('Exact training-only diagnostic required')
    source = Path(fit['source_receipts'][0].rsplit(':sha256:', 1)[0]).parent
    context = json.loads((source/'context.json').read_text())
    aliases = {name+'#0': name for name in ('text', 'image', 'ocr', 'asr')}
    if context['resident_aliases'] != aliases or context.get('observer_diagnostic') != 'specialists_peft_base_semantic':
        raise ValueError('Exact observer-qualified specialist runtime required')
    if hashlib.sha256(POOL.read_bytes()).hexdigest() != context['input_pool_sha256']:
        raise ValueError('Frozen pool changed')
    pool = [json.loads(line) for line in POOL.read_text().splitlines()]
    profiles = fit['profiles']['profiles']
    ranked = {family: sorted([r for r in profiles if r['family'] == family],
        key=lambda r: (sum(s['duration_ms']['mean'] for s in r['segments'] if s['kind'] == 'gpu'), r['pool_id']))
        for family in aliases.values()}
    def request(family, pool_id):
        return dict(alias=family+'#0', task=family, pool_id=pool_id, offset_ms=0.)
    targets = []
    for family in ('text', 'ocr'):
        selected = []
        for row in diagnostic['solo_band_contrasts']:
            if row['family'] == family and row['pool_id'] not in selected:
                selected.append(row['pool_id'])
            if len(selected) == 3:
                break
        for item in selected:
            if item not in {r['pool_id'] for r in pool if r['task_type'] == family}:
                raise ValueError('Target absent from frozen pool')
            targets.append((family, item))
    blocks = []
    for family, pool_id in targets:
        histories = [('self', request(family, pool_id)),
            ('text_short', request('text', ranked['text'][0]['pool_id'])),
            ('text_long', request('text', ranked['text'][-1]['pool_id'])),
            ('image', request('image', ranked['image'][-1]['pool_id'])),
            ('asr', request('asr', ranked['asr'][-1]['pool_id'])),
            ('ocr', request('ocr', ranked['ocr'][-1]['pool_id']))]
        for repeat in range(2):
            for history, precursor in histories:
                identity = f'{family}:{pool_id}:{history}:r{repeat}'
                blocks.append([dict(id='history:'+identity, requests=[copy.deepcopy(precursor)]),
                               dict(id='target:'+identity, requests=[request(family, pool_id)])])
        for seconds in (10, 30):
            identity = f'{family}:{pool_id}:idle{seconds}'
            blocks.append([dict(id='history:'+identity, requests=[request(family, pool_id)]),
                dict(id='idle:'+identity, requests=[], duration_ms=seconds*1000),
                dict(id='target:'+identity, requests=[request(family, pool_id)])])
    random.Random(args.seed).shuffle(blocks)
    jobs = [job for block in blocks for job in block]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    for index, receipt in enumerate(json.loads((source/'sources.json').read_text())):
        content = Path(receipt['snapshot']).read_bytes()
        if any(hashlib.sha256(Path(receipt[key]).read_bytes()).hexdigest() != receipt['sha256']
               for key in ('path', 'snapshot')):
            raise ValueError('Frozen observer source drift: '+receipt['path'])
        snapshot = output/'source-snapshot'/str(index)/Path(receipt['path']).name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
        receipts.append(dict(receipt, snapshot=str(snapshot)))
    design = dict(schema='gsdeploy.specialist_history_contrasts.v1', split='training',
        training_fit_sha256=digest(fit), training_diagnostic_sha256=digest(diagnostic),
        targets=[dict(family=f, pool_id=i) for f, i in targets],
        selection='three largest training-only solo band repeat ratios per text/OCR family',
        histories='same input, short/long text, image, ASR, OCR; completed serial precursor then target',
        idle_seconds=[10, 30], block_order_randomized=True, requests_within_blocks_serial=True,
        previous_model_and_input_are_causal=True, no_future_forward_or_output_features=True,
        no_heldout_data_used=True, no_causal_explanation_claimed=True,
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    plan = dict(schema='gsdeploy.native_activity_training_plan.v1', mode='overlap', split='training',
        execution_context_sha256=digest(context), run_id=output.name, seed=args.seed,
        aliases=aliases, jobs=jobs, budget_seconds=1800, priming='largest_three_per_supported_task',
        confirmation_seeds_reserved=[6110491, 6110492], resident_domain=dict.fromkeys(aliases.values(), 1),
        purpose='training history diagnosis only; not independent prediction or utility qualification',
        history_design=design, pure_gpu_stage_claim=False)
    save(output/'context.json', context)
    save(output/'sources.json', receipts)
    save(output/'plan.json', plan)
    save(output/'design.json', design)
    (output/'history-preparation-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(output=str(output), jobs=len(jobs), requests=sum(len(j['requests']) for j in jobs),
                          planned_idle_seconds=240, context=digest(context))))


if __name__ == '__main__':
    main()
