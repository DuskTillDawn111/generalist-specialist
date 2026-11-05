"""Training-only size extremes and sustained owner rosters for specialist support."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import save, digest
from experiments.prepare_activation_resident import prepare_resident


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--ocr-replicas', type=int, choices=(1, 2), default=1)
    args = p.parse_args()
    fit = json.loads(args.fit.read_text())
    origin = Path(fit['source_receipts'][0].split(':sha256:')[0]).parent
    aliases = json.loads((origin/'context.json').read_text())['resident_aliases']
    deployment = dict(text=1, image=1, ocr=args.ocr_replicas, asr=1)
    if aliases != {f'{m}#{i}': m for m, n in deployment.items() for i in range(n)}:
        raise ValueError('This design requires matching specialist residency and OCR replica count')
    prepare_resident(args.output.resolve(), args.reference.resolve(),
                     deployment, mode='overlap', seed=args.seed)
    plan = json.loads((args.output/'plan.json').read_text())
    if plan['execution_context_sha256'] != fit['profiles']['execution_context_sha256']:
        raise ValueError('Support supplement changed collector or residency')
    ordered = {}
    for family in aliases.values():
        ordered[family] = sorted([r for r in fit['profiles']['profiles'] if r['family'] == family],
            key=lambda r: (sum(s['duration_ms']['mean'] for s in r['segments'] if s['kind'] == 'gpu'), r['pool_id']))

    def request(alias, index, offset=0.):
        family = aliases[alias]
        return dict(alias=alias, task=family, pool_id=ordered[family][index]['pool_id'], offset_ms=offset)

    # Fixed first job puts each largest training-baseline input at ordinal zero.
    # Later random ordering cannot accidentally omit that cold extreme again.
    first = dict(id='cold_largest_all_owners', requests=[request(a, -1) for a in aliases])
    jobs = []
    for size in (0, -1):
        for n in range(2, len(aliases)+1):
            for subset in itertools.combinations(aliases, n):
                jobs.append(dict(id=f'extreme:{size}:'+','.join(subset),
                    requests=[request(a, size) for a in subset]))
    # Deliberately maintain all owners across changing complete request inputs.
    # Work remains nonpreemptive and batch one; this is ordinary FIFO queueing.
    for phase in range(4):
        requests = []
        for alias in aliases:
            family = aliases[alias]
            n = len(ordered[family])
            for i in range(8):
                index = (phase*8+i) % n
                requests.append(request(alias, index, 25.*i))
        jobs.append(dict(id=f'sustained_rank_inputs:{phase}', requests=requests))
    # Every image input also meets the other owners' largest observed solo input.
    for index in range(len(ordered['image'])):
        requests = [request('image#0', index)]
        requests += [request(a, -1) for a in aliases if a != 'image#0']
        jobs.append(dict(id=f'image_full_extreme_peers:{index}', requests=requests))
    random.Random(args.seed).shuffle(jobs)
    design = dict(schema='gsdeploy.specialist_support_contrasts.v1', split='training',
        source_fit=str(args.fit.resolve()), source_fit_sha256=digest(fit),
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection='smallest/largest pinned solo native baseline and complete rank cohorts, all from training receipts',
        first_job_fixed='largest training-baseline input on every owner at post-prime ordinal zero',
        no_heldout_samples_or_outputs_used=True, no_future_service_or_stopping_features=True,
        independent_validation_required_again=True, unchanged_acceptance_criteria=True)
    plan.update(jobs=[first, *jobs], diagnostic_supplement=True, support_design=design,
                admission_enabled=False, not_formal_utility=True)
    save(args.output/'plan.json', plan)
    save(args.output/'support-design.json', design)
    (args.output/'support-preparation-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(output=str(args.output), jobs=len(plan['jobs']),
                          requests=sum(len(j['requests']) for j in plan['jobs']),
                          context=plan['execution_context_sha256'])))


if __name__ == '__main__':
    main()
