"""Seal at most three candidates and two paired confirmation histories.

Candidate choice uses development search only. Confirmation plans/predictions
are sealed together before any GPU collection; no fit or adaptive resampling.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import digest, save
from experiments.prepare_activation_aggregate_validation import make_jobs
from experiments.scoped_routing_objective import file_receipt, predict_candidate
from gsdeploy.routing import Router

TASKS = ('asr', 'image', 'ocr', 'text')


def realized_key(routing, per_task=24):
    router = Router(routing)
    return tuple((task, tuple(router.choose(task) for _ in range(per_task))) for task in TASKS)


def category(routing):
    shares = [row.get('generalist_2', 0.)/sum(row.values()) for row in routing.values()]
    endpoint = all(min(abs(s), abs(1-s)) < 1e-10 for s in shares)
    grid = all(abs(2*s-round(2*s)) < 1e-10 for s in shares)
    return 'endpoint' if endpoint else 'grid' if grid else 'nongrid'


def select_panel(runs, rates):
    pool, winners = [], {}
    for method, seed, raw in runs:
        winner = raw.get('winner')
        if winner:
            route = winner.get('routing', winner.get('routing_qps'))
            value = winner.get('score', winner.get('best_feasible_utility'))
            item = dict(method=method, seed=seed, routing=route, development_utility=value, source='method_winner')
            if method not in winners or value > winners[method]['development_utility']:
                winners[method] = item
            pool.append(item)
        for row in raw.get('events', []):
            value = row.get('utility', row.get('lb'))
            if value is None:
                continue
            route = row.get('routing')
            if route is None:
                route = {t: {m: s*rates[t] for m, s in v.items()} for t, v in row['routing_shares'].items()}
            pool.append(dict(method=method, seed=seed, routing=route, development_utility=value, source='search_observation'))
    if set(winners) != {'Joint-K2', 'Activation-BO-MADS'}:
        raise ValueError('Both methods need complete feasible development winners')
    panel, keys = [], set()
    def add(item):
        key = realized_key(item['routing'])
        if key in keys:
            return
        if len(panel) >= 3:
            raise ValueError('Three-candidate budget cannot include both method winners and missing route categories; review selection impact, do not expand automatically')
        keys.add(key)
        panel.append(dict(copy.deepcopy(item), category=category(item['routing']), candidate_id=f'candidate-{len(panel)+1}'))
    for method in ('Joint-K2', 'Activation-BO-MADS'):
        add(winners[method])
    for required in ('endpoint', 'nongrid'):
        if any(r['category'] == required for r in panel):
            continue
        eligible = sorted((r for r in pool if category(r['routing']) == required
                           and realized_key(r['routing']) not in keys), key=lambda r: -r['development_utility'])
        if eligible:
            add(eligible[0])
        else:
            # A distinct non-grid route is an explicit control, not a newly
            # optimized winner. Its prediction is charged to panel setup.
            share = 0. if required == 'endpoint' else .3141592653589793
            add(dict(method='predeclared_control', seed=None,
                     routing={t: {t: rates[t]*(1-share), 'generalist_2': rates[t]*share} for t in TASKS},
                     development_utility=None, source='fixed_control_not_search_winner'))
    if len(panel) < 3 and not any(r['category'] == 'grid' for r in panel):
        add(dict(method='predeclared_control', seed=None,
                 routing={t: {t: rates[t]*.5, 'generalist_2': rates[t]*.5} for t in TASKS},
                 development_utility=None, source='fixed_half_share_grid_control'))
    for item in panel:
        item['represents_methods'] = [m for m, w in winners.items() if realized_key(w['routing']) == realized_key(item['routing'])]
    return panel


def prepare(comparison, validation, evidence_path, reference, output):
    state = json.loads((comparison/'state.json').read_text())
    if state['status'] != 'bounded_search_completed':
        raise ValueError('Completed bounded search required; software smoke cannot select confirmation candidates')
    specification = json.loads((comparison/'plan.json').read_text())
    for receipt in specification['source_receipts']:
        if file_receipt(receipt['path']) != receipt:
            raise ValueError('Optimization source/input changed since search')
    if specification['seeds'] != [11, 29, 47] or specification['methods'] != ['Joint-K2', 'Activation-BO-MADS']:
        raise ValueError('Comparison methods/seeds changed')
    fit = json.loads((validation/'fit-frozen.json').read_text())
    evidence = json.loads(evidence_path.read_text())
    system = json.loads(reference.read_text())
    original = json.loads((validation/'aggregate-validation-round1/plan.json').read_text())
    rate = 24/(original['jobs'][0]['duration_ms']/1000.)
    rates = dict.fromkeys(TASKS, rate)
    runs = [(m, s, json.loads((comparison/f'{m}-{s}.json').read_text()))
            for s in specification['seeds'] for m in specification['methods']]
    panel = select_panel(runs, rates)
    # A persisted seed in an earlier capture plan means it cannot be silently
    # reused as a fresh confirmation. Reserved-seed metadata is not usage.
    for path in (ROOT/'artifacts/activation-gpu-20261104').rglob('plan.json'):
        previous = json.loads(path.read_text())
        if previous.get('seed') in (6110491, 6110492):
            raise ValueError('Confirmation seed already bound to a plan: '+str(path))
    output.mkdir(parents=True, exist_ok=False)
    save(output/'panel.json', dict(candidates=panel, methods='best development result over three search seeds',
        max_candidates=3, paired_rounds=2, no_additional_candidates=True,
        nongrid_control_if_needed='fixed pi/10 share, not claimed as BO winner'))
    old_context = json.loads((validation/'aggregate-validation-round1/context.json').read_text())
    runtime_receipts = json.loads((validation/'aggregate-validation-round1/sources.json').read_text())
    for i, receipt in enumerate(runtime_receipts):
        if any(file_receipt(receipt[k])['sha256'] != receipt['sha256'] for k in ('path', 'snapshot')):
            raise ValueError('Runtime source changed')
        snapshot = output/'runtime-source-snapshot'/str(i)/Path(receipt['path']).name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(Path(receipt['snapshot']).read_bytes())
        receipt['snapshot'] = str(snapshot.resolve())
    captures, sealed = [], []
    for round_index, seed in enumerate((6110491, 6110492)):
        # Same per-task count, profiled pool, offered rate and two arrival
        # families as the closing validation; no new coverage dimension.
        template = copy.deepcopy(original)
        template.update(seed=seed, split='confirmation', not_for_training=True,
                        evaluation_role='locked_candidate_confirmation')
        template.pop('confirmation_seeds_reserved', None)
        template['jobs'] = [make_jobs(fit['profiles'], seed, burst_pairs=bool(round_index))[0]]
        order = panel if round_index == 0 else list(reversed(panel))
        for item in order:
            directory = output/f'round{round_index+1}-{item["candidate_id"]}'
            directory.mkdir()
            template['run_id'] = directory.name
            prediction = predict_candidate(fit, template, item['routing'], evidence, system)
            if prediction.get('performance_unknown') or not prediction['constraints_eligible'] or prediction['U'] is None:
                save(directory/'rejected-prediction.json', prediction)
                raise ValueError('Candidate confirmation forecast incomplete/infeasible; no GPU launch')
            plan = prediction['candidate_plan']
            plan.update(candidate_id=item['candidate_id'], round=round_index+1,
                        budget_seconds=math.ceil(plan['jobs'][0]['duration_ms']/1000.+700))
            plan['jobs'][0]['id'] = 'candidate-routing'
            # Match the sealed forecast job identity to the actual capture;
            # no duration, routing, prediction value or arrival is changed.
            prediction['prediction']['jobs'][0]['id'] = 'candidate-routing'
            save(directory/'plan.json', plan)
            save(directory/'prediction-sealed.json', prediction)
            save(directory/'context.json', old_context)
            save(directory/'sources.json', runtime_receipts)
            cohort = [{k: r[k] for k in ('task', 'pool_id', 'offset_ms')} for r in plan['jobs'][0]['requests']]
            sealed.append(dict(candidate_id=item['candidate_id'], round=round_index+1, seed=seed,
                directory=str(directory.resolve()), offered_cohort_sha256=digest(cohort),
                plan=file_receipt(directory/'plan.json'), prediction=file_receipt(directory/'prediction-sealed.json')))
            captures.append(str(directory.resolve()))
    for round_index in (1, 2):
        if len({r['offered_cohort_sha256'] for r in sealed if r['round'] == round_index}) != 1:
            raise ValueError('Candidate offered histories differ')
    save(output/'queue.json', dict(schema='gsdeploy.native_capture_queue.v1',
                                  state=str((output/'capture-state.json').resolve()), captures=captures))
    sources = {Path(__file__).resolve(), ROOT/'experiments/score_scoped_confirmation.py',
               ROOT/'experiments/continue_scoped_confirmation.py',
               ROOT/'experiments/scoped_routing_objective.py',
               *[Path(r['path']) for r in json.loads((validation/'protocol.json').read_text())['source_receipts']]}
    receipts = []
    for i, source in enumerate(sorted(sources)):
        snapshot = output/'analysis-source-snapshot'/str(i)/source.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        receipts.append(dict(**file_receipt(source), snapshot=str(snapshot.resolve())))
    save(output/'protocol.json', dict(schema='gsdeploy.scoped_candidate_confirmation.v1',
        sealed_unix_ns=time.time_ns(), candidates=panel, predictions=sealed, source_receipts=receipts,
        input_receipts=[file_receipt(p) for p in (reference, evidence_path, validation/'fit-frozen.json',
            validation/'policy-frozen.json', comparison/'comparison.json', output/'panel.json', output/'queue.json')],
        maximum_gpu_episodes=6, no_between_round_changes=True, no_automatic_retries=True,
        inference='two paired histories on the known input pool; descriptive panel selection only, no unseen-input/general population claim',
        energy_boundary='startup_begin through observation_end, including priming/drain/tail; excludes process teardown and host energy'))
    return dict(candidates=len(panel), gpu_episodes=len(captures), sealed_before_collection=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('comparison', 'validation', 'evidence', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.comparison.resolve(), a.validation.resolve(), a.evidence.resolve(), a.reference.resolve(), a.output.resolve())))
