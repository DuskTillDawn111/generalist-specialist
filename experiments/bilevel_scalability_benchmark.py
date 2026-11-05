"""Bounded, manufactured CPU benchmark for the unchanged bilevel controller.

This is NOT a FIFO predictor, fitted performance law, or physical model catalog.
The synthetic convex queue cost admits independent reference intervals. Sources
of the released optimizer are not modified. Initialization is explicitly shared;
component switches take effect only after its charged evaluations finish.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsdeploy.config import SystemConfig
from gsdeploy.continuous import ActivationAwareBilevelOptimizer, ContinuousRoutingEvaluator
from gsdeploy.continuous.activation_search import ActivationSpaces
from gsdeploy.continuous.benchmark import make_problem
from gsdeploy.continuous.contracts import Budget, BudgetExhausted, digest
from gsdeploy.continuous.deployment_pool import DeploymentCandidatePool
from gsdeploy.continuous.route_domain import ContinuousRouteDomain

PROTOCOL = ROOT / 'configs/bilevel_scalability_ablation_20261105.json'
WEIGHTS = dict(Q=.25, T=1/12, L=1/12, P=1/12, M=.25, E=.25)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def receipt(path):
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def build_problem(size, protocol):
    if size not in protocol['sizes']:
        raise ValueError('UNDECLARED_SIZE')
    # Fixed deterministic construction, never fitted to optimizer outputs.
    rows = [dict(name='g', role='generalist', tasks=['text', 'image'], mu=1.05,
                 quality=dict(text=.08, image=.09)),
            dict(name='s_shared', role='specialist', tasks=['text', 'image'], mu=1.10,
                 quality=dict(text=.18, image=.12)),
            dict(name='s_text', role='specialist', tasks=['text'], mu=1.15,
                 quality=dict(text=.11))]
    for j in range(size-3):
        rows.append(dict(name=f'x{j:02d}', role='generalist' if j % 3 == 0 else 'specialist',
            tasks=['text', 'image'], mu=1.02+.03*(j % 5),
            quality=dict(text=.06+.025*((j*3+1) % 9), image=.06+.025*((j*5+2) % 9))))
    models = {r['name']: dict(role=r['role'], supported_tasks=r['tasks'], executable_tasks=r['tasks'],
        service_time_ms={t: 1000/r['mu'] for t in r['tasks']}, benchmark_mu=r['mu'],
        benchmark_quality=r['quality']) for r in rows}
    catalog = dict(safe_factor=1., capability_contract='verified_executable_v1', models=models,
        capacity_factors={m: {str(i): f for i, f in enumerate(protocol['replica_capacity_factors'])} for m in models},
        memory_table_mib={m: {str(i): i*protocol['memory_per_replica'] for i in range(3)} for m in models})
    config = SystemConfig(catalog=catalog, workload=protocol['workload'],
        gpu_memory_mib=protocol['memory_budget'], latency_reference_ms=dict(text=4000., image=4000.),
        weights=WEIGHTS, utility_schema='six_term_quality_v1', horizon_seconds=20., random_seed=101)
    return make_problem(config, {'train': [101]})


class QueueCostOracle:
    """Analytical convex synthetic losses, independent of finite trace assignment.

    x_m=sum_t q_tm, c_m=mu_m*capacity(n_m), total offered rate=1.
    Q=sum q_tm*error_tm; T=sum x^3/c^2; L=sum n*x/[4(c-x)];
    P=sum x^2/c; M=workers/2.4; E=.08*workers+.12*sum x^2/c.
    U=-(Q+(T+L+P)/3+M+E)/4. These are invented test losses,
    not the measured Q/T/L/P semantics or calibrated reference-v3 metrics.
    """
    evidence_kind = 'manufactured_convex_queue_cost_not_hardware_or_FIFO'
    supports_stress = False
    cost_lower_bound = 0.

    def __init__(self):
        self.version = digest(dict(source=receipt(__file__), objective='synthetic_four_group_queue_v1'))

    def check(self):
        if self.version != digest(dict(source=receipt(__file__), objective='synthetic_four_group_queue_v1')):
            raise ValueError('BENCHMARK_SOURCE_CHANGED')

    def memory_states(self, problem):
        c = problem.config.catalog
        return {(m, n): c['memory_table_mib'][m][str(n)] for m, ns in problem.states.items() for n in ns}

    def state_costs(self, problem):
        # Explicit nonnegative M + idle E floor, identical in all ablations.
        return {(m, n): .25*v/problem.config.gpu_memory_mib+.25*.08*n
                for (m, n), v in self.memory_states(problem).items()}

    def score(self, problem, deployment, routing):
        c = problem.config
        workers = sum(deployment.values())
        memory = sum(c.catalog['memory_table_mib'][m][str(n)] for m, n in deployment.items())
        if memory > c.gpu_memory_mib+1e-9:
            return None
        terms = dict(Q=0., T=0., L=0., P=0., M=memory/c.gpu_memory_mib, E=.08*workers)
        for m, n in deployment.items():
            model = c.catalog['models'][m]
            x = sum(row.get(m, 0.) for row in routing.values())
            cap = model['benchmark_mu']*c.catalog['capacity_factors'][m][str(n)]
            if x >= cap:
                return None
            terms['Q'] += sum(row.get(m, 0.)*model['benchmark_quality'].get(t, 0.) for t, row in routing.items())
            terms['T'] += x**3/cap**2
            terms['L'] += n*x/(4*(cap-x))
            terms['P'] += x*x/cap
            terms['E'] += .12*x*x/cap
        return dict(U=-math.fsum(c.weights[k]*v for k, v in terms.items()), losses=terms)

    def predict(self, problem, deployment, routing, trace, budget, deadline=None):
        budget.consume('predictions', deadline)
        budget.counts['predictor_leaves'] += 1
        result = self.score(problem, deployment, routing)
        return dict(constraints_eligible=result is not None, U=None if result is None else result['U'],
            losses=None if result is None else result['losses'], physical_evidence=False)


def reference_deployments(problem):
    """Complete memory-feasible set: at most two unit-memory replicas fit.

    Independent of tested MILP, activation search, candidate pool and pruning.
    """
    c = problem.config
    if c.gpu_memory_mib != 2.4 or any(c.catalog['memory_table_mib'][m] != {'0': 0., '1': 1., '2': 2.}
                                   for m in problem.states):
        raise ValueError('REFERENCE_ENUMERATION_ASSUMPTION_CHANGED')
    names = list(problem.states)
    candidates = [{m: n} for m in names for n in (1, 2)]
    candidates += [{a: 1, b: 1} for a, b in itertools.combinations(names, 2)]
    return [x for x in candidates if all(any(t in c.catalog['models'][m]['supported_tasks'] for m in x)
                                         for t in c.workload)]


def reference_objective(problem, deployment, vector):
    """Separate formula + analytical gradient in per-task QPS box coordinates.

    All terms are convex for 0<=load<capacity. The tangent minimum over this
    box is a valid lower cost bound, even if SLSQP stops early.
    """
    c = problem.config
    edges = {t: [m for m in sorted(deployment) if t in c.catalog['models'][m]['supported_tasks']]
             for t in sorted(c.workload)}
    flow, pos, bounds = {}, 0, []
    for t, models in edges.items():
        rate = c.workload[t]
        if len(models) == 1:
            flow[t] = {models[0]: rate}
        else:
            flow[t] = {models[0]: float(vector[pos]), models[1]: rate-float(vector[pos])}
            bounds.append((0., rate)); pos += 1
    value = .25*sum(deployment.values())/c.gpu_memory_mib+.02*sum(deployment.values())
    derivatives = {}
    for m, n in deployment.items():
        info = c.catalog['models'][m]
        load = sum(row.get(m, 0.) for row in flow.values())
        capacity = info['benchmark_mu']*(1. if n == 1 else 1.65)
        value += load**3/(12*capacity**2)+n*load/(48*(capacity-load))+(1/12+.03)*load**2/capacity
        marginal = load**2/(4*capacity**2)+n*capacity/(48*(capacity-load)**2)+(1/6+.06)*load/capacity
        for t, row in flow.items():
            if m in row:
                error = info['benchmark_quality'][t]
                value += .25*row[m]*error
                derivatives[t, m] = marginal+.25*error
    gradient = np.array([derivatives[t, ms[0]]-derivatives[t, ms[1]]
                         for t, ms in edges.items() if len(ms) == 2])
    return value, gradient, flow, bounds


def reference(problem, settings):
    started = time.perf_counter()
    records = []
    deployments = reference_deployments(problem)
    if len(deployments) > settings['max_deployments_per_size']:
        raise ValueError('REFERENCE_SIZE_LIMIT')
    for x in deployments:
        if time.perf_counter()-started > settings['seconds_per_size']:
            raise ValueError('REFERENCE_TIME_LIMIT')
        free = [t for t in sorted(problem.config.workload) if sum(
            t in problem.config.catalog['models'][m]['supported_tasks'] for m in x) == 2]
        bounds = [(0., problem.config.workload[t]) for t in free]
        if free:
            fit = minimize(lambda z: reference_objective(problem, x, z)[:2],
                [hi/2 for _, hi in bounds], jac=True, bounds=bounds, method='SLSQP',
                options=dict(maxiter=settings['max_iterations_per_deployment'], ftol=1e-13))
            vector = np.clip(fit.x, 0, [hi for _, hi in bounds])
            status = dict(success=bool(fit.success), iterations=int(fit.nit), message=str(fit.message))
        else:
            vector = np.array([]); status = dict(success=True, iterations=0, message='singleton')
        cost, grad, route, bounds = reference_objective(problem, x, vector)
        tangent = cost+sum(g*((lo if g >= 0 else hi)-z) for g, z, (lo, hi) in zip(grad, vector, bounds))
        lower_cost = tangent-settings['outward_slack']
        # Cross-check two separately implemented arithmetic paths.
        actual = QueueCostOracle().score(problem, x, route)
        if actual is None or not math.isclose(actual['U'], -cost, abs_tol=1e-12):
            raise ValueError('REFERENCE_ORACLE_DISAGREEMENT')
        records.append(dict(deployment=x, routing_qps=route, feasible_U=-cost,
            valid_upper_U=-lower_cost, tangent_gap=cost-lower_cost, optimizer=status))
    best = max(records, key=lambda r: r['feasible_U'])
    return dict(status='COMPLETE_FINITE_DEPLOYMENT_REFERENCE', records=records, winner=best,
        lower_U=best['feasible_U'], upper_U=max(r['valid_upper_U'] for r in records),
        continuous_reference_gap=max(r['valid_upper_U'] for r in records)-best['feasible_U'],
        deployments=len(records), nominal_cartesian_states=3**len(problem.states),
        seconds=time.perf_counter()-started, proof='convex tangent bound per complete memory-feasible deployment',
        numerical_slack=settings['outward_slack'], hardware_evidence=False)


def initial_routes():
    return [{t: {'g': 1.} for t in ('text', 'image')},
            {t: {'s_shared': 1.} for t in ('text', 'image')},
            dict(text={'s_text': 1.}, image={'s_shared': 1.}),
            {t: {'g': .5, 's_shared': .5} for t in ('text', 'image')}]


class CommonInitialSpaces(ActivationSpaces):
    def initialize(self):
        for route in initial_routes():
            self.add({m for row in route.values() for m in row}, 'common_initialization')

    def center(self, key):
        if key == ('s_shared', 's_text'):
            return self.domain.validate(initial_routes()[2])
        return super().center(key)


class FullDomainSpaces(ActivationSpaces):
    """Same domain, endpoints, GP features; no activation faces/transitions."""
    def initialize(self):
        self.add(self.models, 'full_domain_no_activation_search')

    def observe(self, row):
        space = self.spaces[self.models]
        space['visits'] += 1
        if row['best_feasible_utility'] is not None and (space['best'] is None or
                row['best_feasible_utility'] > space['best']['best_feasible_utility']):
            space['best'] = row
        return self.models

    def expand(self, key):
        pass


class SwitchablePool(DeploymentCandidatePool):
    disabled = False

    def compatible(self, routing):
        if self.disabled:
            self.check()
            return []
        return super().compatible(routing)


def coarse_routes(problem):
    # All memory-feasible K=4 task-simplex points, independently generated.
    domain = ContinuousRouteDomain(problem)
    candidates = {}
    for x in reference_deployments(problem):
        options = []
        for t, rate in sorted(problem.config.workload.items()):
            ms = [m for m in sorted(x) if t in problem.config.catalog['models'][m]['supported_tasks']]
            options.append([{ms[0]: 1.}] if len(ms) == 1 else
                           [{ms[0]: k/4, ms[1]: 1-k/4} for k in range(5)])
        for rows in itertools.product(*options):
            route = domain.validate(dict(zip(sorted(problem.config.workload), rows)))
            candidates[digest(route)] = route
    return [candidates[k] for k in sorted(candidates)]


class ControlledOptimizer(ActivationAwareBilevelOptimizer):
    """Uses released solve/scheduler/GP/polls/MILP; changes only declared hooks."""
    def __init__(self, evaluator, method, seed, protocol):
        inner = protocol['inner']
        super().__init__(evaluator, seed=seed, pool=SwitchablePool(evaluator.problem),
            initial_endpoints=0, poll_evaluations=4, inner_seconds=inner['seconds'],
            inner_predictions=inner['predictions'], inner_milp_calls=inner['milp'],
            use_bo=method != 'MADS-only', use_mads=method not in ('BO-only', 'Grid-K4'), exact_inner=True)
        self.method, self.initial_count = method, 0
        self.spaces = CommonInitialSpaces(evaluator.domain, evaluator.problem.config.catalog)
        self.grid = None
        if method == 'Grid-K4':
            points = coarse_routes(evaluator.problem)
            np.random.default_rng(seed).shuffle(points)
            self.grid = iter(points)

    def _evaluate(self, shares, source, **kwargs):
        row = super()._evaluate(shares, source, **kwargs)
        if source == 'initialization':
            self.initial_count += 1
            if self.initial_count == 4:
                self.initial_signature = [dict(shares=r['routing_shares'], lb=r['lb'], ub=r['ub'],
                    certified=r['certified'], work={k: c[k] for k in ('full_prediction_calls',
                    'outer_evaluations', 'milp_solve_calls', 'best_feasible_utility')})
                    for r, c in zip(self.events, self.convergence)]
                self.exact_inner = self.method == 'No-progressive'
                self.pool.disabled = self.method == 'No-pool'
                if self.method == 'No-activation':
                    self.spaces = FullDomainSpaces(self.e.domain, self.e.problem.config.catalog)
                    self.spaces.initialize()
                    for r in self.rows.values():
                        self.spaces.observe(r)
                else:
                    self.spaces.add(self.spaces.models, 'full_cover_after_common_initialization')
                    for key in list(self.spaces.spaces):
                        self.spaces.expand(key)
        return row

    def _global_step(self):
        if self.method != 'Grid-K4':
            return super()._global_step()
        for point in self.grid:
            if digest(point) not in self.rows:
                return self._evaluate(point, 'coarse_grid_K4')
        return None


def run_one(problem, method, seed, protocol):
    b = protocol['budget']
    e = ContinuousRoutingEvaluator(problem, QueueCostOracle(),
        budget=Budget(b['seconds'], b['predictions'], b['routes'], b['milp']))
    opt = ControlledOptimizer(e, method, seed, protocol)
    raw = opt.solve()
    raw.update(method=method, size=len(problem.states), initial_signature=getattr(opt, 'initial_signature', None),
        inner_evaluations=opt.inner.history, switches_after_initialization=True,
        search_domain_dimension=e.domain.dimension, grid_order='seeded_random_without_replacement' if method == 'Grid-K4' else None)
    if raw['initial_signature'] is None:
        raise ValueError('COMMON_INITIALIZATION_INCOMPLETE')
    return raw


def summarize(raw, ref, problem):
    budget = raw['budget']
    init = raw['initial_signature'][-1]['work']['best_feasible_utility']
    winner = raw['winner']
    value = winner['best_feasible_utility'] if winner else None
    hist = raw['inner_evaluations']
    first = next((r for r in raw['convergence'] if r['best_feasible_utility'] is not None), None)
    gaps = [r['optimality_gap'] for r in hist if r['optimality_gap'] is not None]
    neighborhood = []
    if winner:
        for t, row in winner['routing_shares'].items():
            positive = [m for m, s in row.items() if s > 1e-12]
            for a, b in itertools.combinations(positive, 2):
                for delta in (-.01, .01):
                    shares = copy.deepcopy(winner['routing_shares'])
                    if min(shares[t][a]+delta, shares[t][b]-delta) < 0:
                        continue
                    shares[t][a] += delta; shares[t][b] -= delta
                    route = ContinuousRouteDomain(problem).to_qps(shares)
                    best = max((QueueCostOracle().score(problem, x, route)['U'] for x in reference_deployments(problem)
                        if all(m in x for rr in route.values() for m in rr)), default=None)
                    neighborhood.append(dict(task=t, from_model=b, to_model=a, delta=delta, U=best))
    return dict(size=raw['size'], method=raw['method'], seed=raw['seed'], status=raw['status'], U=value,
        initialization_U=init, gain_over_initialization=None if value is None else value-init,
        reference_lower_U=ref['lower_U'], reference_upper_U=ref['upper_U'],
        regret_upper=None if value is None else ref['upper_U']-value,
        budget=budget, dimension=raw['search_domain_dimension'],
        first_feasible=None if first is None else {k: first[k] for k in ('elapsed_seconds','full_prediction_calls','milp_solve_calls')},
        certified_evaluations=sum(r['certified'] for r in hist), evaluations=len(hist),
        mean_inner_gap=sum(gaps)/len(gaps) if gaps else None, max_inner_gap=max(gaps) if gaps else None,
        pool_size=raw['pool']['size'], pool_distinct_discoveries=sum(r['action']=='ADD' for r in raw['pool']['events']),
        initialization_sha256=digest(raw['initial_signature']),
        mixed=bool(winner and any(sum(s>1e-9 for s in row.values())>1 for row in winner['routing_shares'].values())),
        non_K4=bool(winner and any(abs(s*4-round(s*4))>1e-8 for row in winner['routing_shares'].values() for s in row.values())),
        winner_deployment=winner['deployment'] if winner else None,
        winner_shares=winner['routing_shares'] if winner else None,
        neighbor_checks=neighborhood,
        bo_ei_proposals=sum(r['source']=='bo_expected_improvement' for r in raw['events']),
        mads_polls=sum(r['source']=='mads_poll' for r in raw['events']))


def validate_rows(raw, problem, ref, protocol):
    b, limits = raw['budget'], protocol['budget']
    assert b['full_prediction_calls'] <= limits['predictions']
    assert b['outer_evaluations'] <= limits['routes'] and b['milp_solve_calls'] <= limits['milp']
    # Independent fixed-route enumeration audits every observed interval.
    oracle = QueueCostOracle()
    deployments = reference_deployments(problem)
    count = 0
    for row in raw['inner_evaluations']:
        routing = row['routing_qps']
        compatible = [x for x in deployments if all(m in x for rr in routing.values() for m in rr)]
        values = [oracle.score(problem, x, routing)['U'] for x in compatible]
        truth = max(values) if values else None
        lower, upper = row['best_feasible_utility'], row['valid_upper_bound']
        assert lower is None or truth is not None and lower <= truth+1e-8
        assert truth is None or upper is None or upper >= truth-1e-8
        if row['certified']:
            assert lower is None and truth is None or lower is not None and truth is not None and abs(lower-truth)<1e-8
        assert lower is None or lower <= ref['upper_U']+1e-8
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), PROTOCOL, *sorted((ROOT/'gsdeploy/continuous').glob('*.py')),
             ROOT/'gsdeploy/config.py', ROOT/'gsdeploy/linear_solver.py']
    receipts = [receipt(p) for p in paths]
    schedule = [(n, 'BO-MADS', s) for n in protocol['sizes'] for s in protocol['seeds']]
    schedule += [(7, m, s) for s in protocol['seeds'] for m in protocol['methods_at_seven'] if m != 'BO-MADS']
    assert len(schedule) == protocol['maximum_search_runs']
    save(out/'plan.json', dict(protocol=protocol, schedule=schedule, sources=receipts,
        benchmark_parameters_fixed_before_reference_and_search=True))
    problems, references = {}, {}
    for size in protocol['sizes']:
        problems[size] = build_problem(size, protocol)
        save(out/f'problem-{size}.json', dict(config=problems[size].config.as_dict(),
            identity=problems[size].identity, states=problems[size].states))
        references[size] = reference(problems[size], protocol['reference'])
        save(out/f'reference-{size}.json', references[size])
        print(json.dumps(dict(stage='reference', size=size, U=references[size]['lower_U'],
            gap=references[size]['continuous_reference_gap'], deployments=references[size]['deployments'])), flush=True)
    summaries, checked = [], 0
    for size, method, seed in schedule:
        raw = run_one(problems[size], method, seed, protocol)
        save(out/f'{size}-{method}-{seed}.json', raw)
        checked += validate_rows(raw, problems[size], references[size], protocol)
        row = summarize(raw, references[size], problems[size])
        summaries.append(row)
        print(json.dumps({k: row[k] for k in ('size','method','seed','U','gain_over_initialization','regret_upper','budget')}, ensure_ascii=False), flush=True)
    for size in protocol['sizes']:
        assert len({r['initialization_sha256'] for r in summaries if r['size']==size}) == 1
    assert all(receipt(r['path']) == r for r in receipts)
    save(out/'summary.json', dict(rows=summaries, checked_inner_evaluations=checked,
        searches=len(summaries), evidence=protocol['evidence'], new_gpu_runs=0, predictor_refits=0))
    save(out/'completion.json', dict(status='BOUNDED_SCALABILITY_CONTINUITY_ABLATION_COMPLETE',
        searches=len(summaries), inner_intervals_checked=checked, source_integrity=True,
        common_initialization_verified=True, remaining_automatic_work=[]))


if __name__ == '__main__':
    main()
