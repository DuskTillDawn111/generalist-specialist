"""Matched five-model small-offload probes; all specialists remain resident."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import POOL, save, digest
from experiments.prepare_activation_resident import prepare_resident
from gsdeploy.trace import make_trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--matched-specialist', required=True, type=Path)
    args = p.parse_args()
    control_path = args.matched_specialist.resolve()/'plan.json'
    control = json.loads(control_path.read_text())
    if (control['split'] != 'development_shared_pressure'
            or control['resident_domain'] != dict(text=1, image=1, ocr=1, asr=1)):
        raise ValueError('Declared all-specialist pressure control required')
    seed = control['seed']
    deployment = dict(generalist_2=1, text=1, image=1, ocr=1, asr=1)
    output = args.output.resolve()
    prepare_resident(output, args.reference.resolve(), deployment, mode='overlap', seed=seed)
    plan = json.loads((output/'plan.json').read_text())
    by_task = {}
    for line in POOL.read_text().splitlines():
        row = json.loads(line)
        by_task.setdefault(row['task_type'], []).append(row)
    controls = {job['rates']['asr']: job for job in control['jobs']}
    # Declare the same five coordinates before either round is observed.
    coordinates = [(0., 0.), (.1, 0.), (0., .1), (.1, .1), (.25, .25)]
    jobs = []
    for case, asr_rate in enumerate((1., 1.2)):
        rates = dict(text=1., image=.0748360842705697, ocr=.0748360842705697, asr=asr_rate)
        trace = make_trace(rates, 45., seed+100*case, 'poisson')
        counters = dict.fromkeys(by_task, 0)
        cohort = []
        rng = random.Random(seed+100*case+10000)
        for offset, task in trace:
            n = counters[task]
            counters[task] += 1
            item = by_task[task][n % len(by_task[task])]
            cohort.append(dict(task=task, pool_id=item['pool_id'], offset_ms=offset*1000.,
                               route_uniform=rng.random()))
        if digest(cohort) != controls[asr_rate]['shared_trace_sha256']:
            raise ValueError('Five-model arrival/input/routing-uniform trace differs from control')
        for text_share, asr_share in coordinates:
            shares = dict(text=text_share, asr=asr_share)
            requests = []
            for row in cohort:
                task = row['task']
                alias = 'generalist_2#0' if row['route_uniform'] < shares.get(task, 0.) else task+'#0'
                requests.append(dict(alias=alias, **{k: row[k] for k in ('task', 'pool_id', 'offset_ms')}))
            jobs.append(dict(id=f'five_pressure:asr{asr_rate}:g_text{text_share}:g_asr{asr_share}',
                requests=requests, duration_ms=45000, rates=rates,
                text_generalist_share=text_share, asr_generalist_share=asr_share,
                shared_trace_sha256=digest(cohort), finite_probe_not_capacity_or_utility=True))
    random.Random(seed+17000).shuffle(jobs)
    plan.update(jobs=jobs, mode='five_model_pressure_probe', split='development_shared_pressure',
        not_for_training=True, admission_enabled=False, no_utility_normalization=True,
        design='five resident models, small text/ASR offload on unchanged matched cohorts',
        declared_coordinates=coordinates, all_specialists_remain_resident=True,
        matched_control_plan=str(control_path),
        matched_control_plan_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest(),
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['finite 45 second diagnostic; no utility or stability qualification',
                    'new five-model runs occur after original controls; chronology is recorded, not called counterbalanced',
                    'zero offload still pays for idle G2 residency; compare to four-model control separately'])
    save(output/'plan.json', plan)
    print(json.dumps(dict(output=str(output), jobs=len(jobs), requests=sum(len(j['requests']) for j in jobs),
                          deployment=deployment, context=plan['execution_context_sha256'])))


if __name__ == '__main__':
    main()
