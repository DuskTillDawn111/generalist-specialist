"""Matched pressure controls with extra ASR/text specialist replicas."""
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
from experiments.prepare_activation_resident import prepare_resident
from gsdeploy.config import SystemConfig
from gsdeploy.trace import make_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--matched-specialist', type=Path, required=True)
    parser.add_argument('--text-replicas', type=int, choices=(1, 2), required=True)
    parser.add_argument('--asr-replicas', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    reference = args.reference.resolve()
    control_path = args.matched_specialist.resolve()/'plan.json'
    control = json.loads(control_path.read_text())
    if (control['split'] != 'development_shared_pressure'
            or control['resident_domain'] != dict(text=1, image=1, ocr=1, asr=1)):
        raise ValueError('Matched original four-specialist pressure control required')
    deployment = dict(text=args.text_replicas, image=1, ocr=1, asr=args.asr_replicas)
    system = SystemConfig.load(reference/'system-frozen.json')
    # Preserve the original catalog memory limit. Physical peak remains a
    # measured outcome; catalog feasibility is not a physical upper bound.
    memory = system.catalog['safe_factor']*sum(
        system.catalog['memory_table_mib'][family][str(count)] for family, count in deployment.items())
    if memory > system.gpu_memory_mib:
        raise ValueError('Replica control exceeds original catalog GPU memory')
    output = args.output.resolve()
    seed = control['seed']
    prepare_resident(output, reference, dict.fromkeys(deployment, 1), mode='overlap', seed=seed)
    plan = json.loads((output/'plan.json').read_text())
    context = json.loads((output/'context.json').read_text())
    registry = context['worker_bindings']
    for family, count in deployment.items():
        for replica in range(1, count):
            registry[f'{family}#{replica}'] = copy.deepcopy(registry[family+'#0'])
    aliases = {alias: alias.split('#')[0] for alias in sorted(registry)}
    registry = {alias: registry[alias] for alias in aliases}
    sources = json.loads((output/'sources.json').read_text())
    content = Path(__file__).read_bytes()
    snapshot = output/'replica-preparation-source.py'
    snapshot.write_bytes(content)
    sources.append(dict(path=str(Path(__file__).resolve()), snapshot=str(snapshot),
                        sha256=hashlib.sha256(content).hexdigest()))
    sources.sort(key=lambda source: source['path'])
    context.update(resident_aliases=aliases, worker_bindings=registry,
        preparation_workflow='diagnostic_replica_pressure_v1',
        source_content=[{key: source[key] for key in ('path','sha256')} for source in sources])
    plan.update(aliases=aliases, resident_domain=deployment, execution_context_sha256=digest(context))
    save(output/'context.json', context)
    save(output/'sources.json', sources)
    by_task = {}
    for line in POOL.read_text().splitlines():
        item = json.loads(line)
        by_task.setdefault(item['task_type'], []).append(item)
    controls = {job['rates']['asr']: job for job in control['jobs']}
    jobs = []
    for case, asr_rate in enumerate((1., 1.2)):
        rates = dict(text=1., image=.0748360842705697, ocr=.0748360842705697, asr=asr_rate)
        cohort = []
        counters = dict.fromkeys(by_task, 0)
        rng = random.Random(seed+100*case+10000)
        requests = []
        for offset, task in make_trace(rates, 45., seed+100*case, 'poisson'):
            ordinal = counters[task]
            counters[task] += 1
            pool_id = by_task[task][ordinal % len(by_task[task])]['pool_id']
            cohort.append(dict(task=task, pool_id=pool_id, offset_ms=offset*1000., route_uniform=rng.random()))
            requests.append(dict(alias=f'{task}#{ordinal % deployment[task]}', task=task,
                                 pool_id=pool_id, offset_ms=offset*1000.))
        if digest(cohort) != controls[asr_rate]['shared_trace_sha256']:
            raise ValueError('Replica control arrival/input cohort differs from original control')
        jobs.append(dict(id=f'replica_pressure:asr{asr_rate}:text_copies{args.text_replicas}:asr_copies{args.asr_replicas}',
            requests=requests, duration_ms=45000, rates=rates,
            asr_generalist_share=0., text_generalist_share=0.,
            shared_trace_sha256=digest(cohort), finite_probe_not_capacity_or_utility=True))
    random.Random(seed+18000).shuffle(jobs)
    plan.update(jobs=jobs, mode='replica_pressure_probe', split='development_shared_pressure',
        not_for_training=True, admission_enabled=False, no_utility_normalization=True,
        matched_control_plan=str(control_path),
        matched_control_plan_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest(),
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        routing='per-task round robin over resident specialist replicas',
        limitations=['45 second offered intervals; no steady-state, rho or formal utility certificate',
                    'later than historical controls; no claim of counterbalanced chronology',
                    'energy measured through batch drain; whole startup energy not allocated'])
    save(output/'plan.json', plan)
    print(json.dumps(dict(output=str(output), deployment=deployment, jobs=len(jobs),
                          context=plan['execution_context_sha256'])))


if __name__ == '__main__':
    main()
