"""Prospective five-model training for aggregate prediction, not qualification.

Reuses the accepted runtime and completed solo training. Local stage residuals
are recorded by the old reports, but do not veto new aggregate training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import POOL, digest, save
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_training import build_template
from gsdeploy.trace import make_trace

TASKS = ('asr', 'image', 'ocr', 'text')
RESERVED = {6110491, 6110492}


def make_jobs(template, aliases, seed):
    rng = random.Random(seed)
    paths = {}
    for profile in template['profiles']:
        paths.setdefault((profile['family'], profile['task']), []).append(profile)
    for values in paths.values():
        values.sort(key=lambda p: (sum(s['duration_ms']['mean'] for s in p['segments']), p['pool_id']))
    means = {k: statistics.mean(sum(s['duration_ms']['mean'] for s in p['segments'])
                                for p in values) for k, values in paths.items()}
    tiers = {k: [values[round(q * (len(values) - 1))] for q in (.1, .5, .9)]
             for k, values in paths.items()}
    jobs = []
    for n in range(2, len(aliases) + 1):
        for roster in itertools.combinations(sorted(aliases), n):
            options = [TASKS if aliases[a] == 'generalist_2' else (aliases[a],) for a in roster]
            for tasks in itertools.product(*options):
                for tier in range(3):
                    requests = []
                    for i, (alias, task) in enumerate(zip(roster, tasks)):
                        profile = tiers[aliases[alias], task][(tier + i) % 3]
                        # Prospective stagger uses only the existing solo mean.
                        offset = 0. if tier == 0 else min(250., .2 * means[aliases[alias], task]) * (i % 2)
                        requests.append(dict(alias=alias, task=task, pool_id=profile['pool_id'], offset_ms=offset))
                    jobs.append(dict(id='roster:' + digest([roster, tasks, tier])[:20], requests=requests,
                                     design=dict(kind='crossed_roster', tier=tier, intended_owner_count=n,
                                                 realized_overlap_guaranteed=False)))
    for (family, task), values in sorted(paths.items()):
        alias = next(a for a, f in aliases.items() if f == family)
        requests = [dict(alias=alias, task=task, pool_id=tiers[family, task][i % 3]['pool_id'],
                         offset_ms=i * .2 * means[family, task]) for i in range(6)]
        jobs.append(dict(id=f'fifo-burst:{family}:{task}', requests=requests,
                         design=dict(kind='finite_same_owner_burst', formal_rho_qualification=False)))
    rng.shuffle(jobs)
    # Continuous arrivals are training exposure, not measured deployment gains.
    # Common arrival traces are shared by the three routing variants at a load.
    # The conservative nominal scale is computed over all three variants.
    routes = (0., .5, 1.)
    demand = []
    for share in routes:
        owner = dict.fromkeys(aliases, 0.)
        for task in TASKS:
            owner['generalist_2#0'] += share * means['generalist_2', task] / 1000.
            owner[task + '#0'] += (1. - share) * means[task, task] / 1000.
        demand.append(max(owner.values()))
    for load_index, load in enumerate((.4, .8)):
        rate = load / max(demand)
        arrivals = make_trace(dict.fromkeys(TASKS, rate), 45., seed + 100 + load_index, 'poisson')
        # Input choice is identical across routing variants and known in advance.
        cohort = [(offset, task, rng.choice(paths[task, task])['pool_id']) for offset, task in arrivals]
        for share in routes:
            requests = []
            route_rng = random.Random(seed + 200 + load_index)
            for offset, task, pool_id in cohort:
                alias = 'generalist_2#0' if route_rng.random() < share else task + '#0'
                requests.append(dict(alias=alias, task=task, pool_id=pool_id, offset_ms=offset * 1000.))
            jobs.append(dict(id=f'arrivals:{load}:{share}', requests=requests, duration_ms=45000.,
                             design=dict(kind='continuous_arrivals', nominal_solo_load=load,
                                         generalist_share=share, task_rate=rate,
                                         formal_rho_qualification=False)))
    # Native work resets idle history; all idle samples remain in the record.
    anchor = tiers['image', 'image'][1]
    for seconds in (1, 3, 12, 30, 90):
        jobs.append(dict(id=f'idle-anchor:{seconds}', requests=[
            dict(alias='image#0', task='image', pool_id=anchor['pool_id'], offset_ms=0.)]))
        jobs.append(dict(id=f'idle:{seconds}', requests=[], duration_ms=seconds * 1000.))
    return jobs


def prepare(solo, observer, output, seed, policy):
    if {seed, seed + 1} & RESERVED:
        raise ValueError('Confirmation seeds are reserved')
    gate = json.loads(observer.read_text())
    if gate.get('observer_repeat_check_passed') is not True or gate.get('issues'):
        raise ValueError('Accepted observer required')
    policy_doc = json.loads(policy.read_text())
    if policy_doc.get('schema') != 'gsdeploy.practical_prediction_policy.v1':
        raise ValueError('Explicit new practical policy required')
    bundles = [load_bundle(solo / f'solo-round{i}') for i in (1, 2)]
    context = bundles[0]['context']
    aliases = {f + '#0': f for f in ('generalist_2', *TASKS)}
    if context['resident_aliases'] != aliases or any(b['context'] != context for b in bundles):
        raise ValueError('Matching complete five-model solo context required')
    if context.get('observer_diagnostic') != 'five_model_scoped_semantic':
        raise ValueError('Accepted scoped observer runtime required')
    for name, source in gate['sources'].items():
        if not name.startswith('semantic-'):
            continue
        bundle = load_bundle(Path(source.rsplit(':sha256:', 1)[0]).parent)
        if bundle['source'] != source or bundle['context'] != context:
            raise ValueError('Observer source/context changed')
    repeat = json.loads((solo / 'description.json').read_text())
    if repeat['sources'] != [b['source'] for b in bundles] or repeat['changed_output_profiles']:
        raise ValueError('Complete output-consistent solo baselines required')
    content = POOL.read_bytes()
    if hashlib.sha256(content).hexdigest() != context['input_pool_sha256']:
        raise ValueError('Training input pool changed')
    template, evidence = build_template(bundles, [json.loads(line) for line in content.decode().splitlines()])
    sources = json.loads((solo / 'solo-round1/sources.json').read_text())
    for source in sources:
        if any(hashlib.sha256(Path(source[k]).read_bytes()).hexdigest() != source['sha256']
               for k in ('path', 'snapshot')):
            raise ValueError('Frozen execution source changed: ' + source['path'])
    output.mkdir(parents=True, exist_ok=False)
    copied = []
    for i, source in enumerate(sources):
        target = output / 'runtime-source-snapshot' / str(i) / Path(source['path']).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source['snapshot']).read_bytes())
        copied.append(dict(source, snapshot=str(target)))
    save(output / 'solo-template.json', template)
    save(output / 'solo-baseline-evidence.json', evidence)
    captures = []
    for i in range(2):
        directory = output / f'aggregate-training-round{i + 1}'
        directory.mkdir()
        jobs = make_jobs(template, aliases, seed + i)
        plan = dict(schema='gsdeploy.native_activity_training_plan.v1', mode='overlap', split='training',
                    execution_context_sha256=digest(context), run_id=directory.name, seed=seed + i,
                    aliases=aliases, jobs=jobs, budget_seconds=3600,
                    priming='largest_three_per_supported_task', confirmation_seeds_reserved=sorted(RESERVED),
                    local_stage_errors_diagnostic_only=True, physical_predictor_qualified=False,
                    formal_utility_qualified=False, purpose='aggregate co-resident model training')
        save(directory / 'context.json', context)
        save(directory / 'sources.json', copied)
        save(directory / 'plan.json', plan)
        captures.append(str(directory))
    design_paths = [Path(__file__).resolve(), policy, observer, solo / 'description.json',
                    solo / 'quality-description.json', solo / 'completion-template-diagnostic.json',
                    ROOT / 'experiments/run_activation_queue.py', ROOT / 'experiments/run_activation_capture.py',
                    ROOT / 'gsdeploy/trace.py', ROOT / 'gsdeploy/forecasting/native_training.py']
    receipts = []
    for i, path in enumerate(design_paths):
        target = output / 'design-source-snapshot' / str(i) / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        receipts.append(dict(path=str(path), snapshot=str(target), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(output / 'queue.json', dict(schema='gsdeploy.native_capture_queue.v1',
                                     state=str(output / 'state.json'), captures=captures))
    save(output / 'design.json', dict(schema='gsdeploy.aggregate_training_design.v1',
        baseline_sources=[b['source'] for b in bundles], source_receipts=receipts,
        execution_context_sha256=digest(context), seeds=[seed, seed + 1],
        scope='five models resident, one replica each; activity rosters, input tiers, FIFO bursts, arrivals and idle',
        fit_direction='coarse native progress with low-dimensional peer effects; aggregate replay decides usability',
        baseline_stage_gate_required=False, observer_and_output_integrity_required=True,
        physical_predictor_qualified=False, independent_validation_scheduled=False,
        future_input_generalization_required=True))
    print(json.dumps(dict(output=str(output), captures=captures, jobs_per_round=len(jobs),
                          requests_last_round=sum(len(j['requests']) for j in jobs))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('solo-base', 'observer-report', 'output', 'policy'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    prepare(args.solo_base.resolve(), args.observer_report.resolve(), args.output.resolve(),
            args.seed, args.policy.resolve())
