"""Bounded cross-domain strategy comparison, without modifying production code.

All score queries use the same counted evaluator. Structural feasibility and
flow repair use only capability/memory metadata. Reference objectives, gradients
and reference winners are unavailable to search. The log records EVERY paid
prediction, including initialization and unfinished inner solves.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.bilevel_scalability_benchmark import (
    QueueCostOracle, build_problem, read, receipt, save, validate_rows, reference_objective)
from experiments.verify_bilevel_scheduler import ScheduledControlledOptimizer
from gsdeploy.continuous import ContinuousRoutingEvaluator
from gsdeploy.continuous.contracts import Budget, BudgetExhausted, digest

PROTOCOL = ROOT/'configs/crossdomain_strategy_comparison_20261105.json'
ARCHIVE = ROOT/'artifacts/bilevel-scalability-ablation-20261105'


class LoggedOracle(QueueCostOracle):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.phase = 'initialization'

    def predict(self, problem, deployment, routing, trace, budget, deadline=None):
        prediction = super().predict(problem, deployment, routing, trace, budget, deadline)
        self.calls.append(dict(index=budget.counts['predictions'],
            elapsed_seconds=time.perf_counter()-budget.start, phase=self.phase,
            deployment=copy.deepcopy(deployment), routing_qps=copy.deepcopy(routing),
            U=prediction['U'], eligible=prediction['constraints_eligible']))
        return prediction


def legal(problem, deployment):
    c = problem.config
    return (bool(deployment)
        and all(m in problem.states and type(n) is int and n > 0 and n in problem.states[m]
                for m, n in deployment.items())
        and sum(c.catalog['memory_table_mib'][m][str(n)] for m, n in deployment.items()) <= c.gpu_memory_mib+1e-9
        and all(any(t in c.catalog['models'][m]['supported_tasks'] for m in deployment) for t in c.workload))


def neighbors(problem, deployment):
    """One replica add/drop/transfer, with capability and memory filtering only."""
    seen = {}
    def add(candidate):
        candidate = {m: n for m, n in candidate.items() if n}
        if candidate != deployment and legal(problem, candidate):
            seen[digest(candidate)] = candidate
    for m in problem.states:
        a = dict(deployment);a[m] = a.get(m, 0)+1;add(a)
    for m in deployment:
        a = dict(deployment);a[m] -= 1;add(a)
        for other in problem.states:
            if other == m:continue
            b = dict(a);b[other] = b.get(other, 0)+1;add(b)
    return list(seen.values())


def repaired_route(problem, domain, deployment, previous=None):
    """Uniform new routes; preserve retained weights and activate new choices.

    A newly available model gets 1/k share. Existing weights receive the
    remaining mass proportionally. No objective/quality/service-time query.
    """
    result = {}
    for task in domain.edges:
        models = sorted(m for m in deployment if m in domain.edges[task])
        old = {m: (previous or {}).get(task, {}).get(m, 0.) for m in models}
        retained = sum(old.values())
        new = [m for m in models if previous is not None and m not in previous[task]]
        # Normalized routes include zero entries; positive membership defines
        # which choices had flow, including zero-flow residents as new choices.
        if previous is not None:new = [m for m in models if old[m] <= 0]
        if retained <= 0:
            result[task] = {m: 1/len(models) for m in models}
        else:
            mass = len(new)/len(models)
            result[task] = {m: (1/len(models) if m in new else (1-mass)*old[m]/retained) for m in models}
    return domain.validate(result)


def route_neighbors(problem, domain, deployment, shares, step):
    candidates = {}
    for task in domain.edges:
        models = sorted(m for m in deployment if m in domain.edges[task])
        for donor, recipient in itertools.permutations(models, 2):
            amount = min(step, shares[task].get(donor, 0.))
            if amount <= 1e-12:continue
            trial = copy.deepcopy(shares)
            trial[task][donor] -= amount
            trial[task][recipient] = trial[task].get(recipient, 0.)+amount
            trial = domain.validate(trial)
            candidates[digest(trial)] = trial
    return list(candidates.values())


class DirectSearch:
    def __init__(self, evaluator, initializer, seed, protocol):
        self.e, self.init, self.cfg = evaluator, initializer, protocol
        self.rng = np.random.default_rng(seed)
        self.events = []
        self.seen = {}
        for r in evaluator.oracle.calls:
            shares = evaluator.domain.validate({t: {m: q/evaluator.problem.config.workload[t] for m, q in row.items()}
                                                for t, row in r['routing_qps'].items()})
            row = dict(deployment=r['deployment'], shares=shares, U=r['U'])
            self.seen[digest([row['deployment'], shares])] = row
        self.incumbent = max((x for x in self.seen.values() if x['U'] is not None), key=lambda x: x['U'])

    def evaluate(self, deployment, shares, phase):
        self.e.budget.check()
        shares = self.e.domain.validate(shares)
        assert legal(self.e.problem, deployment)
        qps = self.e.domain.to_qps(shares)
        assert all(m in deployment for row in qps.values() for m in row)
        key = digest([deployment, shares])
        if key in self.seen:
            self.e.budget.counts['cache_hits'] += 1
            return self.seen[key]
        self.e.budget.consume('outer')
        self.e.oracle.phase = phase
        result = self.e.candidate(deployment, qps)
        row = dict(deployment=copy.deepcopy(deployment), shares=shares, U=result['utility'])
        self.seen[key] = row
        if row['U'] is not None and row['U'] > self.incumbent['U']:
            self.incumbent = row
        self.events.append(dict(phase=phase, deployment=deployment, shares=shares,
                               U=row['U'], budget=self.e.budget.report()))
        return row

    def random_candidate(self):
        # Bounded repair on metadata. Includes one/two replicas and two-model
        # deployments, including zero-flow residents through endpoint routes.
        names = list(self.e.problem.states)
        for _ in range(100):
            k = int(self.rng.integers(1, 3))
            selected = self.rng.choice(names, size=k, replace=False).tolist()
            d = dict.fromkeys(selected, 1)
            if k == 1:d[selected[0]] = int(self.rng.integers(1, 3))
            if legal(self.e.problem, d):
                return d, repaired_route(self.e.problem, self.e.domain, d)
        raise ValueError('NO_LEGAL_RANDOM_START')

    @staticmethod
    def better(a, b):
        return a['U'] is not None and (b['U'] is None or a['U'] > b['U']+1e-12)

    def route_poll(self, current, step, phase):
        trials = route_neighbors(self.e.problem, self.e.domain, current['deployment'], current['shares'], step)
        self.rng.shuffle(trials)
        best = current
        for shares in trials:
            row = self.evaluate(current['deployment'], shares, phase)
            if self.better(row, best):best = row
        return best

    def facility(self):
        current = self.incumbent
        step = self.cfg['local']['initial_step']
        for _ in range(self.cfg['budget']['routes']):
            self.e.budget.check()
            candidates = neighbors(self.e.problem, current['deployment'])
            self.rng.shuffle(candidates)
            best = current
            for d in candidates:
                shares = repaired_route(self.e.problem, self.e.domain, d, current['shares'])
                row = self.evaluate(d, shares, 'facility_add_drop_swap')
                if self.better(row, best):best = row
            best = self.route_poll(best, step, 'facility_flow_polish')
            if self.better(best, current):
                current = best;step = self.cfg['local']['initial_step']
            else:
                step /= 2
                if step < self.cfg['local']['minimum_step']:return 'LOCAL_STATIONARY'
        return 'ITERATION_GUARD'

    def alternating(self):
        # Round-robin starts prevent the first block-local optimum consuming
        # the full budget. Restart addresses the zero-flow activation trap.
        starts = []
        for r in self.init.inner.history[:4]:
            starts.append(dict(deployment=r['deployment'], shares=r['routing_shares'],
                               U=r['best_feasible_utility'], step=self.cfg['local']['initial_step']))
        for iteration in range(self.cfg['budget']['routes']):
            self.e.budget.check()
            index = iteration % len(starts)
            old = starts[index];current = old
            current = self.route_poll(current, old['step'], 'alternating_route_block')
            candidates = neighbors(self.e.problem, current['deployment'])
            active = {m for row in current['shares'].values() for m, s in row.items() if s > 0}
            candidates = [d for d in candidates if active <= d.keys()]
            self.rng.shuffle(candidates)
            best = current
            for d in candidates:
                row = self.evaluate(d, current['shares'], 'alternating_deployment_block')
                if self.better(row, best):best = row
            improved = self.better(best, old)
            step = self.cfg['local']['initial_step'] if improved else old['step']/2
            if step < self.cfg['local']['minimum_step']:
                d, shares = self.random_candidate()
                best = self.evaluate(d, shares, 'alternating_restart')
                step = self.cfg['local']['initial_step']
            starts[index] = dict(best, step=step)
        return 'ITERATION_GUARD'


class MixedCodec:
    """Random keys for identities, cardinality/replicas and per-task flow.

    Feasibility repair uses capability only. Memory domain is the declared
    benchmark's at-most-two-unit-replica domain, not a universal encoding.
    """
    def __init__(self, problem, domain):
        self.problem, self.domain = problem, domain
        self.names = list(problem.states)
        self.tasks = list(domain.edges)
        self.n = len(self.names)
        self.dimension = self.n+2+len(self.tasks)*self.n

    def decode(self, vector):
        z = np.clip(vector, 0., 1.)
        order = sorted(range(self.n), key=lambda i: (-z[i], i))
        k = 1 if z[self.n] < .5 else 2
        chosen = [self.names[i] for i in order[:k]]
        if not all(any(m in self.domain.edges[t] for m in chosen) for t in self.tasks):
            if k == 1:
                chosen = [next(self.names[i] for i in order if all(self.names[i] in self.domain.edges[t] for t in self.tasks))]
            else:
                for i in order[1:]:
                    proposal = [chosen[0], self.names[i]]
                    if len(set(proposal)) == 2 and all(any(m in self.domain.edges[t] for m in proposal) for t in self.tasks):
                        chosen = proposal;break
        d = dict.fromkeys(chosen, 1)
        if k == 1:d[chosen[0]] = 1 if z[self.n+1] < .5 else 2
        assert legal(self.problem, d)
        shares = {}
        for j, t in enumerate(self.tasks):
            ms = [m for m in chosen if m in self.domain.edges[t]]
            scores = [float(z[self.n+2+j*self.n+self.names.index(m)]) for m in ms]
            total = sum(scores)
            shares[t] = {m: v/total if total else 1/len(ms) for m, v in zip(ms, scores)}
        return d, self.domain.validate(shares)

    def encode(self, deployment, shares):
        z = np.full(self.dimension, .1)
        for m in deployment:z[self.names.index(m)] = .9
        z[self.n] = .25 if len(deployment) == 1 else .75
        z[self.n+1] = .75 if max(deployment.values()) == 2 else .25
        for j, t in enumerate(self.tasks):
            for m in self.names:z[self.n+2+j*self.n+self.names.index(m)] = shares[t].get(m, 0.)
        return z


def de_trial(population, index, rng, factor, crossover):
    others = [i for i in range(len(population)) if i != index]
    a, b, c = rng.choice(others, 3, replace=False)
    mutant = np.clip(population[a]+factor*(population[b]-population[c]), 0., 1.)
    mask = rng.random(population.shape[1]) < crossover
    mask[int(rng.integers(population.shape[1]))] = True
    return np.where(mask, mutant, population[index])


def mixed_de(search):
    cfg = search.cfg['de'];codec = MixedCodec(search.e.problem, search.e.domain)
    population = search.rng.random((cfg['population'], codec.dimension))
    for i, r in enumerate(search.init.inner.history[:4]):
        population[i] = codec.encode(r['deployment'], r['routing_shares'])
    values = []
    for z in population:
        row = search.evaluate(*codec.decode(z), 'de_population')
        values.append(row['U'] if row['U'] is not None else -math.inf)
    values = np.asarray(values)
    for _ in range(search.cfg['budget']['routes']):
        search.e.budget.check()
        frozen, old_values = population.copy(), values.copy()
        for i in range(len(population)):
            z = de_trial(frozen, i, search.rng, cfg['F'], cfg['CR'])
            row = search.evaluate(*codec.decode(z), 'de_rand1bin')
            value = row['U'] if row['U'] is not None else -math.inf
            if value >= old_values[i]:population[i], values[i] = z, value
    return 'ITERATION_GUARD'


def initialize(opt):
    opt.spaces.initialize()
    routes = [opt.spaces.center(k) for k in list(opt.spaces.spaces)]
    for shares in routes:opt._evaluate(shares, 'initialization', force=True)
    assert opt.initial_count == 4


def run_one(size, method, seed, cfg, base):
    protocol = copy.deepcopy(base);protocol['budget'] = cfg['budget']
    problem = build_problem(size, protocol)
    b = cfg['budget']
    oracle = LoggedOracle()
    evaluator = ContinuousRoutingEvaluator(problem, oracle,
        budget=Budget(b['seconds'], b['predictions'], b['routes'], b['milp']))
    opt = ScheduledControlledOptimizer(evaluator, 'BO-MADS', seed, protocol)
    if method == 'BO-MADS':
        # Log phase without changing the released controller's decisions.
        original = opt._evaluate
        def evaluate(shares, source, **kwargs):
            oracle.phase = source
            return original(shares, source, **kwargs)
        opt._evaluate = evaluate
        raw = opt.solve()
    else:
        initialize(opt)
        search = DirectSearch(evaluator, opt, seed, cfg)
        try:
            status = {'Add-drop-swap': search.facility, 'Alternating': search.alternating,
                      'Mixed-DE': lambda: mixed_de(search)}[method]()
        except BudgetExhausted as exc:status = str(exc)
        raw = dict(status=status, events=search.events,
                   problem_sha256=problem.identity, oracle_version=oracle.version)
    winner = max((r for r in oracle.calls if r['eligible']), key=lambda r: r['U'])
    raw.update(size=size, method=method, seed=seed, winner_observed=winner,
        budget=evaluator.budget.report(), initial_signature=opt.initial_signature,
        calls=oracle.calls, inner_evaluations=opt.inner.history,
        initial_U=opt.initial_signature[-1]['work']['best_feasible_utility'],
        initial_calls=opt.initial_signature[-1]['work']['full_prediction_calls'],
        evidence=cfg['evidence'], formal_optimality_claim=False)
    return raw, problem


def audit(raw, problem, reference, protocol):
    count = validate_rows(raw, problem, reference, protocol)
    assert len(raw['calls']) == raw['budget']['full_prediction_calls']
    oracle = QueueCostOracle()
    for i, r in enumerate(raw['calls'], 1):
        assert r['index'] == i and legal(problem, r['deployment'])
        shares = {t: {m: q/problem.config.workload[t] for m, q in row.items()} for t, row in r['routing_qps'].items()}
        from gsdeploy.continuous.route_domain import ContinuousRouteDomain
        ContinuousRouteDomain(problem).validate(shares)
        assert all(m in r['deployment'] for row in r['routing_qps'].values() for m in row)
        expected = oracle.score(problem, r['deployment'], r['routing_qps'])
        assert (expected is not None) == r['eligible']
        if expected is not None:
            assert math.isclose(expected['U'], r['U'], abs_tol=1e-12)
            free_edges = {t: sorted(m for m in r['deployment'] if t in problem.config.catalog['models'][m]['supported_tasks'])
                          for t in sorted(problem.config.workload)}
            vector = [r['routing_qps'][t].get(ms[0], 0.) for t, ms in free_edges.items() if len(ms) == 2]
            independent_cost = reference_objective(problem, r['deployment'], vector)[0]
            assert math.isclose(-independent_cost, r['U'], abs_tol=1e-12)
            assert r['U'] <= reference['upper_U']+1e-8
    assert raw['winner_observed']['U'] >= raw['initial_U']-1e-12
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = read(PROTOCOL);base = read(ARCHIVE/'plan.json')['protocol']
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), PROTOCOL, ROOT/'tests/test_crossdomain_strategies.py',
        ROOT/'experiments/bilevel_scalability_benchmark.py', ROOT/'experiments/verify_bilevel_scheduler.py',
        ROOT/'gsdeploy/config.py', ROOT/'gsdeploy/linear_solver.py',
        *sorted((ROOT/'gsdeploy/continuous').glob('*.py'))]
    sources = [receipt(p) for p in paths]
    inputs = [receipt(ARCHIVE/f'reference-{n}.json') for n in cfg['sizes']]
    inputs.append(receipt(ROOT/'artifacts/crossdomain-method-references-20261105/references.json'))
    inputs += [receipt(ROOT/f'artifacts/bilevel-scheduler-improvement-20261105/{n}-balanced-{s}.json')
               for n in cfg['sizes'] for s in cfg['seeds']]
    schedule = list(itertools.product(cfg['sizes'], cfg['methods'], cfg['seeds']))
    random.Random(6110509).shuffle(schedule)
    assert len(schedule) == cfg['maximum_searches']
    save(out/'plan.json', dict(protocol=cfg, schedule=schedule, sources=sources, inputs=inputs,
        base_protocol=base, frozen_before_search=True, python=sys.version, platform=platform.platform(),
        numpy=np.__version__, blas_threads=1))
    summaries, errors, checked = [], [], 0
    for n, method, seed in schedule:
        raw, problem = run_one(n, method, seed, cfg, base)
        save(out/f'{n}-{method}-{seed}.json', raw)
        ref = read(ARCHIVE/f'reference-{n}.json')
        try:
            prior = read(ROOT/f'artifacts/bilevel-scheduler-improvement-20261105/{n}-balanced-{seed}.json')
            assert raw['initial_signature'] == prior['initial_signature']
            assert raw['problem_sha256'] == prior['problem_sha256']
            assert raw['oracle_version'] == prior['oracle_version']
            validation_protocol = dict(base, budget=cfg['budget'])
            checked += audit(raw, problem, ref, validation_protocol)
        except AssertionError as exc:
            errors.append(dict(size=n, method=method, seed=seed, error=str(exc) or 'AUDIT_FAILURE'))
        target = raw['initial_U']+cfg['target_gain']
        reached = next((r for r in raw['calls'] if r['U'] is not None and r['U'] >= target), None)
        row = dict(size=n, method=method, seed=seed, U=raw['winner_observed']['U'],
            initial_U=raw['initial_U'], gain=raw['winner_observed']['U']-raw['initial_U'],
            regret_upper=ref['upper_U']-raw['winner_observed']['U'], budget=raw['budget'],
            status=raw['status'], target_U=target, target_reached=reached is not None,
            target_calls=reached['index'] if reached else None,
            target_seconds=reached['elapsed_seconds'] if reached else None,
            initial_calls=raw['initial_calls'], initial_sha256=digest(raw['initial_signature']),
            distinct_deployments=len({digest(r['deployment']) for r in raw['calls']}))
        summaries.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    assert all(receipt(r['path']) == r for r in sources+inputs)
    save(out/'summary.json', dict(rows=summaries, audits=errors, checked_inner_results=checked,
        searches=len(schedule), new_gpu_runs=0, real_predictor_calls=0, predictor_refits=0,
        evidence=cfg['evidence'], raw_calls_audited=sum(r['budget']['full_prediction_calls'] for r in summaries)))
    save(out/'completion.json', dict(status='COMPLETE' if not errors else 'COMPLETE_WITH_AUDIT_FAILURES',
        searches=len(schedule), audits=errors, source_integrity=True, remaining_automatic_experiments=[]))
    if errors:raise SystemExit('Audit failures retained, no performance retries')


if __name__ == '__main__':main()
