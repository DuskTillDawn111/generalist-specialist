"""Seal fresh continuous-arrival histories before collecting validation.

This evaluates new histories on the explicitly profiled input pool. It does
not claim unseen-input generalization or qualify utility/candidate ranking.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import digest, save
from experiments.replay_activation_aggregate import predict_plan

TASKS = ('asr', 'image', 'ocr', 'text')


def make_jobs(template, seed, *, per_task=24, nominal_load=.7, burst_pairs=False, reverse_routes=False):
    if per_task < 20 or per_task % 2 or not 0 < nominal_load <= .8:
        raise ValueError('Even cohort >=20 and nominal load <= original rho ceiling required')
    profiles = {}
    for p in template['profiles']:
        profiles.setdefault((p['family'], p['task']), []).append(p)
    g2_work_seconds = sum(statistics.mean(sum(s['duration_ms']['mean'] for s in p['segments'])
                                         for p in profiles['generalist_2', task])/1000. for task in TASKS)
    # All-generalist is the common reference for the offered rate, not a
    # claim about measured dynamic utilization of the other routing policies.
    horizon_seconds = per_task * g2_work_seconds / nominal_load
    spacing = horizon_seconds / per_task
    rng = random.Random(seed)
    cohort = []
    for task in TASKS:
        inputs = sorted(p['pool_id'] for p in profiles['generalist_2', task])
        if len(inputs) < per_task:
            raise ValueError('Distinct profiled inputs required for every task cohort')
        rng.shuffle(inputs)
        for i, pool_id in enumerate(inputs[:per_task]):
            if burst_pairs:
                # Paired bursts preserve the same mean offered rate and count.
                slot = 2*(i//2) + .1 + .02*(i % 2) + rng.uniform(0., .005)
            else:
                slot = i + rng.uniform(.05, .95)
            cohort.append(dict(task=task, pool_id=pool_id, offset_ms=slot*spacing*1000., task_index=i))
    cohort.sort(key=lambda r: (r['offset_ms'], r['task']))
    shares = [0., .5, 1.]
    if reverse_routes:
        shares.reverse()
    jobs = []
    for share in shares:
        route_rng = random.Random(seed+100)
        selected = {task: set(route_rng.sample(range(per_task), round(share*per_task))) for task in TASKS}
        requests = [dict(task=r['task'], pool_id=r['pool_id'], offset_ms=r['offset_ms'],
                         alias='generalist_2#0' if r['task_index'] in selected[r['task']] else r['task']+'#0')
                    for r in cohort]
        jobs.append(dict(id=f'validation-arrivals:{share}', requests=requests,
                         duration_ms=horizon_seconds*1000., design=dict(kind='continuous_arrivals',
                         generalist_share=share, nominal_solo_load=nominal_load,
                         arrival_regime='paired_bursts' if burst_pairs else 'stratified_jitter',
                         offered_per_task=per_task, task_rate=per_task/horizon_seconds,
                         formal_rho_qualification=False),
                         shared_offered_cohort_sha256=digest([{k:r[k] for k in ('task','pool_id','offset_ms')} for r in requests])))
    return jobs


def prepare(training, output, seeds, policy):
    if len(set(seeds)) != 2 or set(seeds) & {6110491, 6110492}:
        raise ValueError('Two unused development seeds required')
    fit_path = training/'analysis-v1/fit.json'
    fit = json.loads(fit_path.read_text())
    replay_path = training/'analysis-v1/training-replay.json'
    replay = json.loads(replay_path.read_text())
    if not all(r['assessment']['complete_cohort'] for r in replay['reports']):
        raise ValueError('Complete training coverage required before fresh history validation')
    for i in (1, 2):
        audit = json.loads((training/f'capture-audit-v1/aggregate-training-round{i}.json').read_text())
        if audit['runs'][0]['issues']:
            raise ValueError('Training capture integrity has unresolved issues')
    source = training/'aggregate-training-round1'
    context = json.loads((source/'context.json').read_text())
    if fit['profiles']['execution_context_sha256'] != digest(context):
        raise ValueError('Model/context mismatch')
    output.mkdir(parents=True, exist_ok=False)
    (output/'fit-frozen.json').write_bytes(fit_path.read_bytes())
    (output/'policy-frozen.json').write_bytes(policy.read_bytes())
    receipts = []
    for i, old in enumerate(json.loads((source/'sources.json').read_text())):
        if any(hashlib.sha256(Path(old[k]).read_bytes()).hexdigest() != old['sha256'] for k in ('path','snapshot')):
            raise ValueError('Frozen runtime source changed')
        target = output/'runtime-source-snapshot'/str(i)/Path(old['path']).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(old['snapshot']).read_bytes())
        receipts.append(dict(old, snapshot=str(target)))
    captures, predictions = [], []
    for i, seed in enumerate(seeds):
        directory = output/f'aggregate-validation-round{i+1}'
        directory.mkdir()
        jobs = make_jobs(fit['profiles'], seed, burst_pairs=bool(i), reverse_routes=bool(i))
        plan = dict(schema='gsdeploy.native_activity_training_plan.v1', mode='overlap', split='development_validation',
            execution_context_sha256=digest(context), run_id=directory.name, aliases=context['resident_aliases'],
            jobs=jobs, seed=seed, budget_seconds=math.ceil(sum(j['duration_ms'] for j in jobs)/1000.+700),
            priming='largest_three_per_supported_task', confirmation_seeds_reserved=[6110491,6110492],
            frozen_fit_sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest(), not_for_training=True,
            input_domain='same profiled inputs; fresh arrivals, route assignment and cross-model histories',
            unseen_input_generalization=False)
        save(directory/'context.json', context)
        save(directory/'sources.json', receipts)
        save(directory/'plan.json', plan)
        prediction = predict_plan(fit, plan)
        if not prediction['complete'] or prediction['predicted_episode_board_joules'] is None:
            save(directory/'rejected-prediction.json', prediction)
            raise ValueError('Prospective whole prediction has unknown coverage; do not launch capture')
        save(directory/'prediction-sealed.json', prediction)
        captures.append(str(directory))
        predictions.append(dict(run_id=plan['run_id'], plan_sha256=hashlib.sha256((directory/'plan.json').read_bytes()).hexdigest(),
            prediction_sha256=hashlib.sha256((directory/'prediction-sealed.json').read_bytes()).hexdigest()))
    queue_path = output/'queue.json'
    save(queue_path, dict(schema='gsdeploy.native_capture_queue.v1', state=str(output/'capture-state.json'), captures=captures))
    source_paths = {Path(__file__).resolve(), ROOT/'experiments/continue_activation_aggregate_validation.py',
        ROOT/'experiments/replay_activation_aggregate.py', ROOT/'experiments/review_activation_aggregate_accuracy.py',
        ROOT/'experiments/audit_activation_aggregate_capture.py', ROOT/'experiments/analyze_activation_semantic.py',
        ROOT/'experiments/analyze_activation_trim_episode_energy.py', ROOT/'experiments/fit_activation_native.py',
        ROOT/'experiments/run_activation_queue.py', ROOT/'experiments/run_activation_capture.py',
        *[ROOT/'gsdeploy'/f for f in ('energy.py','quality.py','normalization.py','tasks.py')],
        *sorted((ROOT/'gsdeploy/forecasting').glob('*.py'))}
    frozen = []
    for i, path in enumerate(sorted(source_paths)):
        target = output/'prediction-source-snapshot'/str(i)/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        frozen.append(dict(path=str(path), snapshot=str(target), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(output/'protocol.json', dict(schema='gsdeploy.aggregate_known_input_validation_protocol.v1',
        prediction_sealed_unix_ns=time.time_ns(), fit_sha256=hashlib.sha256((output/'fit-frozen.json').read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256((output/'policy-frozen.json').read_bytes()).hexdigest(),
        queue_sha256=hashlib.sha256(queue_path.read_bytes()).hexdigest(), captures=captures, predictions=predictions,
        source_receipts=frozen, primary_cells='each run, routing share and task (24 requests per cell)',
        workload_regimes=['stratified_jitter','paired_bursts'], comparison_scope='descriptive new-history development accuracy',
        input_overlap_with_training=True, independent_input_qualification=False, candidate_ranking_qualified=False,
        no_between_round_refit=True, formal_utility_qualified=False))
    print(json.dumps(dict(output=str(output), captures=captures, requests_per_round=288,
                          window_seconds_per_route=jobs[0]['duration_ms']/1000.)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training-base','output','policy'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--seeds', type=int, nargs=2, required=True)
    a = p.parse_args()
    prepare(a.training_base.resolve(), a.output.resolve(), a.seeds, a.policy.resolve())
