"""Long idle contrasts in a declared resident domain, using the frozen collector."""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import POOL, save
from experiments.prepare_activation_resident import prepare_resident


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--deployment', required=True, type=Path)
    p.add_argument('--seed', required=True, type=int)
    args = p.parse_args()
    output = args.output.resolve()
    prepare_resident(output, args.reference.resolve(), json.loads(args.deployment.read_text()),
                     mode='overlap', seed=args.seed)
    plan = json.loads((output/'plan.json').read_text())
    by_task = {}
    for line in POOL.read_text().splitlines():
        row = json.loads(line)
        by_task.setdefault(row['task_type'], []).append(row)
    owners = list(plan['aliases'])
    random.Random(args.seed).shuffle(owners)
    jobs = [dict(id='resident_idle_before', requests=[], duration_ms=30000)]
    for i in range(4):
        requests = []
        for alias in owners:
            family = plan['aliases'][alias]
            task = ('text', 'image', 'ocr', 'asr')[i] if family == 'generalist_2' else family
            for j in range(4):
                item = by_task[task][(7*i+j) % len(by_task[task])]
                requests.append(dict(alias=alias, task=task, pool_id=item['pool_id'], offset_ms=75.*j))
        jobs.append(dict(id=f'controlled_busy:{i}', requests=requests))
        if i in (1, 3):
            jobs.append(dict(id=f'resident_idle_after_busy:{i}', requests=[], duration_ms=30000))
    plan.update(jobs=jobs, diagnostic_supplement=True, admission_enabled=False,
                design='three fixed 30 second resident idle windows before and after controlled bursts',
                idle_is_not_unloaded=True)
    save(output/'plan.json', plan)
    print(json.dumps(dict(output=str(output), jobs=len(jobs), context=plan['execution_context_sha256'])))


if __name__ == '__main__':
    main()
