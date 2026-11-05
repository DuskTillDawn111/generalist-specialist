"""Freeze independent open-loop pressure tests of an already trained resident predictor."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import POOL, save
from gsdeploy.forecasting.native_training import digest
from gsdeploy.trace import make_trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', nargs=2, type=int, required=True)
    args = p.parse_args()
    fit = json.loads(args.fit.read_text())
    families = {r['family'] for r in fit['profiles']['profiles']}
    if families != {'text', 'image', 'ocr', 'asr'}:
        raise ValueError('This open-loop check covers the full specialist resident domain')
    subprocess.run([sys.executable, str(ROOT/'experiments/validate_activation_native.py'), 'prepare',
        '--fit', str(args.fit), '--reference', str(args.reference), '--output', str(args.output),
        '--seeds', *map(str, args.seeds)], check=True)
    protocol = json.loads((args.output/'protocol.json').read_text())
    by_task = {}
    for line in POOL.read_text().splitlines():
        row = json.loads(line)
        by_task.setdefault(row['task_type'], []).append(row)
    for directory in map(Path, protocol['runs']):
        plan = json.loads((directory/'plan.json').read_text())
        save(directory/'plan-before-openloop-design.json', plan)
        jobs = []
        for case, asr_rate in enumerate((1., 1.2)):
            rates = dict(text=1., image=.0748360842705697, ocr=.0748360842705697, asr=asr_rate)
            trace = make_trace(rates, 120., plan['seed']+100*case, 'poisson')
            counts = dict.fromkeys(by_task, 0)
            shuffled = {t: list(items) for t, items in by_task.items()}
            for task, items in shuffled.items():
                random.Random(f"{plan['seed']}:openloop:{case}:{task}").shuffle(items)
            requests = []
            for offset, task in trace:
                n = counts[task]
                counts[task] += 1
                item = shuffled[task][n % len(shuffled[task])]
                requests.append(dict(alias=task+'#0', task=task, pool_id=item['pool_id'], offset_ms=offset*1000.))
            jobs.append(dict(id=f'heldout_pressure:asr{asr_rate}', requests=requests, duration_ms=120000,
                rates=rates, arrival_process='poisson', offered_trace_sha256=digest(trace)))
            jobs.append(dict(id=f'heldout_pressure_cooldown:asr{asr_rate}', requests=[], duration_ms=20000))
        plan.update(jobs=jobs, independent_openloop_validation=True, not_for_training=True,
            validation_design='two 120 second open-loop task/input cohorts and two 20 second cooldowns; no within-job FIFO reset',
            not_formal_utility=True, stability_capacity_qualified=False)
        save(directory/'plan.json', plan)
    path = Path(__file__).resolve()
    content = path.read_bytes()
    snapshot = args.output/'openloop-design-source'/path.name
    snapshot.parent.mkdir(exist_ok=False)
    snapshot.write_bytes(content)
    protocol['prediction_sources'].append(dict(path=str(path), snapshot=str(snapshot.resolve()),
                                               sha256=hashlib.sha256(content).hexdigest()))
    protocol.update(independent_openloop_validation=True, startup_energy_qualified=False,
        openloop_design='fixed before capture; full FIFO across each 120 second open-loop cohort; original criteria unchanged',
        run_plan_sha256={d: hashlib.sha256((Path(d)/'plan.json').read_bytes()).hexdigest() for d in protocol['runs']})
    save(args.output/'protocol.json', protocol)
    print(json.dumps(dict(output=str(args.output), runs=protocol['runs'], criteria_unchanged=True)))


if __name__ == '__main__':
    main()
