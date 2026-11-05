"""Matched Phase 1 ablation on the existing validated real-data CPU replay.

No new training, measurement profiles, GPU work or outer optimizer changes.
All setup is timed; sessions are private and closed after each individual run.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsdeploy import DeploymentSystem, SystemConfig
from gsdeploy.continuous import (Budget, ContinuousProblem, ImmutableContinuousProblem,
    ContinuousRoutingEvaluator, DERoutingOptimizer, LegacyPredictorOracle, SessionPredictorOracle)
from gsdeploy.continuous.contracts import digest
from gsdeploy.search_session import SearchSession
from experiments.diagnose_continuous_phase1 import DEFAULT_SOURCE, ROOT, assignment, domain_size, save, subset


def manifest():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'gsdeploy').rglob('*.py')}


def run(c, states, algorithm, seconds, outer, seed, route=None):
    start = time.perf_counter()
    session, oracle, history = None, None, []
    try:
        if algorithm.startswith('old_'):
            name = algorithm.split('_')[1]
            if algorithm.endswith('_session'):
                session = SearchSession(ROOT, c.as_dict())
                setup = time.perf_counter()-start
                opt = session.optimizer('dfs' if name == 'dfs' else 'milp',
                    milp_time_limit_seconds=max(.001, seconds-setup))
                raw = opt.solve(K=c.route_grid, replica_options=states)
            else:
                config = replace(c, solver='dfs' if name == 'dfs' else 'milp')
                system = DeploymentSystem(config)
                setup = time.perf_counter()-start
                raw = system.plan(replica_options=states)
            w = raw['winner']
            winner = dict(deployment=w['deployment'], routing=w['routing'], utility=w['score'],
                          prediction=w['prediction']) if w and w['prediction'].get('U') is not None else None
            metrics = raw['metrics']
            counts = metrics['full_prediction_calls']
            certified = raw['optimality_certified']
            raw.pop('problem', None)
        else:
            budget = Budget(seconds, 10000, outer)
            frozen = algorithm in {'cache', 'both'}
            p = (ImmutableContinuousProblem if frozen else ContinuousProblem)(c, replica_options=states)
            oracle = SessionPredictorOracle(p) if frozen else LegacyPredictorOracle()
            mode = 'enumeration' if algorithm == 'enumeration' else ('milp_bounds' if algorithm in {'bounds', 'both'} else 'milp')
            e = ContinuousRoutingEvaluator(p, oracle, mode, budget)
            setup = time.perf_counter()-start
            if route is not None:
                row = e.evaluate(route)
                raw = dict(status=row['status'], winner=row if row['feasible'] else None)
                certified = row['certified']
            else:
                raw = DERoutingOptimizer(e, seed=seed, popsize=3, maxiter=100, face_limit=128).solve()
                certified = False
            history = e.history
            w = raw['winner']
            winner = dict(deployment=w['deployment'], routing=w['routing_qps'],
                          utility=w['best_feasible_utility'], prediction=w['prediction']) if w else None
            metrics = budget.report()
            counts = budget.counts['predictions']
        elapsed = time.perf_counter()-start
        stats = (oracle.session.stats() if isinstance(oracle, SessionPredictorOracle)
                 else session.stats() if session else None)
        fifo = assignment(c, winner['deployment'], winner['routing']) if winner else None
        return dict(algorithm=algorithm, scope='fixed_route' if route else 'joint',
            seconds=elapsed, setup_seconds=setup, search_seconds=elapsed-setup,
            predictions=counts, utility=winner['utility'] if winner else None,
            certified=certified, status=raw['status'], winner=winner, fifo=fifo,
            metrics=metrics, session_stats=stats, history=history, raw=raw,
            input_sha256=digest(dict(config=c.as_dict(), states=states)),
            seed=seed, seconds_limit=seconds, outer_limit=outer,
            fixed_shares=route)
    finally:
        if isinstance(oracle, SessionPredictorOracle):
            oracle.close()
        if session:
            session.close()


def main(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = manifest()
    save(out/'source-manifest.json', before)
    protected = {p: v for p,v in before.items() if not p.startswith('gsdeploy/continuous/')}
    previous = json.loads((ROOT/'artifacts/continuous-routing-diagnostics-v1-20261101/source-manifest-before.json').read_text())
    assert all(previous[p] == v for p,v in protected.items()), 'Old optimizer/predictor changed'
    for path in list((ROOT/'gsdeploy/continuous').glob('*.py'))+[Path(__file__).resolve(), ROOT/'tests/test_continuous_validated.py']:
        target = out/'source-snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    source = ROOT/args.source
    base = SystemConfig.load(source)
    save(out/'environment.json', dict(python=sys.version, platform=platform.platform(),
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        predictor_status=base.catalog['calibrated_replay_v2']['status'], no_new_data=True,
        no_gpu=True, no_outer_changes=True, setup_included=True))
    cases = [r for r in json.loads((ROOT/'artifacts/continuous-routing-diagnostics-v1-20261101/domains.json').read_text())
             if r['case'] in args.cases]
    all_runs = []
    global_start = time.perf_counter()
    for case in cases:
        name, states = case['case'], case['replica_options']
        c = subset(base, list(states), args.seconds, 2)
        save(out/f'{name}-config.json', c.as_dict())
        save(out/f'{name}-domain.json', domain_size(c, states, 2))
        jobs = [(a, repeat) for repeat in range(args.repeats) for a in
                ('old_dfs', 'old_joint', 'old_dfs_session', 'old_joint_session', 'baseline', 'bounds', 'cache', 'both')]
        random.Random(20261101).shuffle(jobs)
        for algorithm, repeat in jobs:
            result = run(c, states, algorithm, args.seconds, args.outer, [11,29][repeat % 2])
            result.update(case=name, repeat=repeat)
            save(out/f'{name}-{algorithm}-r{repeat}.json', result)
            all_runs.append(result)
            print(name, algorithm, repeat, round(result['seconds'],3), result['predictions'], result['utility'], flush=True)
        # Fixed-route independent reference and all four ablations are separate
        # experiments; never rank them as end-to-end joint searches.
        if name in {'pool3', 'pool5'}:
            routes = {'generalist': {t:{'generalist_2':1.} for t in c.workload},
                      'winner': {t:{('image' if t=='image' else 'ocr' if t=='ocr' else 'generalist_2'):1.} for t in c.workload}}
            for route_name, route in routes.items():
                fixed = []
                jobs = [(a, repeat) for repeat in range(args.repeats) for a in
                        ('enumeration','baseline','bounds','cache','both')]
                random.Random(20261102).shuffle(jobs)
                for algorithm, repeat in jobs:
                    result = run(c, states, algorithm, 120., 100, 11, route)
                    result.update(case=name, repeat=repeat, route_name=route_name)
                    save(out/f'{name}-fixed-{route_name}-{algorithm}-r{repeat}.json',result)
                    fixed.append(result); all_runs.append(result)
                    print(name, 'fixed', route_name, algorithm, round(result['seconds'],3),result['predictions'],flush=True)
                assert len({(r['utility'],r['certified']) for r in fixed})==1
                # Equal complete winning prediction JSON, not only equal U.
                refs = [r for r in fixed if r['algorithm']=='enumeration']
                if refs[0]['winner']:
                    prediction = refs[0]['winner']['prediction']
                    for r in fixed:
                        assert r['winner']['prediction']==prediction
    assert manifest()==before, 'Sources changed during benchmark'
    save(out/'summary.json', dict(seconds=time.perf_counter()-global_start, runs=len(all_runs),
        rows=[{k:r[k] for k in ('case','scope','algorithm','repeat','seconds','setup_seconds','search_seconds',
                               'predictions','utility','certified','status','input_sha256')} |
              dict(route_name=r.get('route_name'), coverage=r['raw'].get('activation_coverage'),
                   outer_evaluations=len(r['history']),
                   milp_calls=r['metrics'].get('milp_solve_calls',r['metrics'].get('milp_calls',0)),
                   bound_seconds=sum(h.get('bound_seconds',0) for h in r['history'])) for r in all_runs],
        source_unchanged=True, old_sources_match_previous_diagnosis=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=DEFAULT_SOURCE)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=float,default=8.)
    parser.add_argument('--outer',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--cases',nargs='+',default=['pool3','pool5','pool7','replicas2_pool5'])
    main(parser.parse_args())
