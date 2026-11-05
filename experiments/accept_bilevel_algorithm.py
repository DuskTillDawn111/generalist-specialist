"""Finite final acceptance: matched initialization and nontrivial deployment domains.

Read existing sources, use frozen real predictor on CPU, and separately verify
software mechanisms with the existing synthetic fixture. Never collects data.
"""
import argparse
import copy
from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.scoped_routing_objective import file_receipt
from experiments.search_scoped_utility_v3 import evaluator_v3, CONTRACT
from gsdeploy.config import SystemConfig
from gsdeploy.continuous import ActivationAwareBilevelOptimizer, ContinuousRoutingEvaluator
from gsdeploy.continuous.contracts import Budget, ContinuousProblem, digest, load_experiment
from gsdeploy.continuous.benchmark import oracle_from_spec, make_problem
from gsdeploy.continuous.deployment_pool import DeploymentCandidatePool
from gsdeploy.continuous.progressive import ProgressiveFixedRoutingSolver
from gsdeploy.continuous.route_domain import ContinuousRouteDomain
from gsdeploy.continuous.predictor_adapter import normalize_prediction


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def initial_signature(raw):
    events = [r for r in raw['events'] if r['source'] == 'initialization']
    convergence = [r for r in raw['convergence'] if r['source'] == 'initialization']
    require(len(events) == len(convergence) > 0, 'Complete initialization ledger required')
    return [dict(shares=e['routing_shares'], lower=e['lb'], upper=e['ub'], certified=e['certified'],
                 **{k: c[k] for k in ('full_prediction_calls', 'outer_evaluations', 'milp_solve_calls', 'best_feasible_utility')})
            for e, c in zip(events, convergence)]


def outer_summary(raw, method, seed, reused):
    initial = initial_signature(raw)
    w = raw['winner']
    require(w is not None, 'Feasible initial control required')
    value = w['best_feasible_utility']
    return dict(method=method, seed=seed, reused=reused, status=raw['status'], budget=raw['budget'],
        U=value, initialization_U=initial[-1]['best_feasible_utility'],
        improvement_after_initialization=value-initial[-1]['best_feasible_utility'],
        routing=w['routing_qps'], deployment=w['deployment'],
        initialization_sha256=digest(initial),
        bo_ei_proposals=sum(e['source'] == 'bo_expected_improvement' for e in raw['events']),
        mads_polls=sum(e['source'] == 'mads_poll' for e in raw['events']))


def outer_acceptance(output, cfg):
    campaign = ROOT / 'artifacts/activation-gpu-20261104'
    archive = ROOT / 'artifacts/utility-v3-search-20261105'
    audit = read(archive / 'verification.json')
    # Source snapshots bind the actual search code used for the archived runs.
    for r in read(archive / 'plan.json')['source_receipts']:
        require(file_receipt(r['snapshot'])['sha256'] == r['sha256'], 'Archived search snapshot changed')
        if Path(r['path']).name != 'search_scoped_utility_v3.py':
            require(file_receipt(r['path'])['sha256'] == r['sha256'], 'Archived search source changed')
    for revision in audit['postprocessing_revisions']:
        require(file_receipt(revision['current']['path']) == revision['current'], 'Search adapter revision changed')
    system = read(campaign / 'references-v2/system-frozen.json')
    fit = read(campaign / 'five-model-aggregate-validation-v1/fit-frozen.json')
    evidence = read(campaign / 'scoped-optimization-preparation-v1/training-evidence.json')
    templates = []
    for i in (1, 2):
        p = read(campaign / f'five-model-aggregate-validation-v1/aggregate-validation-round{i}/plan.json')
        p['jobs'] = [p['jobs'][0]]
        templates.append(p)
    archived, rows = {}, []
    for seed in cfg['seeds']:
        path = archive / f'Activation-BO-MADS-{seed}.json'
        raw = read(path)
        expected = next(r for r in read(archive/'decision.json')['source_runs'] if r['path'] == str(path))
        require(file_receipt(path) == expected, 'Archived run changed')
        require(raw['utility_contract'] == CONTRACT, 'Objective mismatch')
        archived[seed] = raw
        rows.append(outer_summary(raw, 'BO-MADS', seed, True))
    for seed in cfg['seeds']:
        methods = cfg['new_methods'] if seed == 29 else list(reversed(cfg['new_methods']))
        for method in methods:
            e = evaluator_v3(system, fit, evidence, templates, Budget(cfg['seconds_per_run'],
                cfg['predictions_per_run'], cfg['routes_per_run'], cfg['milp_per_run']))
            raw = ActivationAwareBilevelOptimizer(e, seed=seed, initial_endpoints=4,
                inner_seconds=20., inner_predictions=4, inner_milp_calls=8, poll_evaluations=4,
                use_bo=method != 'MADS-only', use_mads=method != 'BO-only').solve()
            raw.update(utility_contract=CONTRACT, physical_ranking_qualified=False)
            save(output / f'{method}-{seed}.json', raw)
            require(e.problem.objective_id == archived[seed]['objective_id'] and
                    e.oracle.version == archived[seed]['oracle_version'], 'Problem/predictor identity mismatch')
            require(initial_signature(raw) == initial_signature(archived[seed]), 'Initialization not matched')
            row = outer_summary(raw, method, seed, False)
            require(row['budget']['full_prediction_calls'] <= 64 and row['budget']['outer_evaluations'] <= 32
                    and row['budget']['milp_solve_calls'] <= 128, 'Outer budget exceeded')
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    require(sum(not r['reused'] for r in rows) == cfg['maximum_new_runs'], 'Run count changed')
    result = dict(status='MATCHED_INITIALIZATION_ACCEPTANCE_COMPLETED', rows=rows,
        initialization_exactly_matched=True, new_runs=6, reused_runs=3,
        combo_improves_over_initialization=any(r['improvement_after_initialization'] > 1e-12 for r in rows if r['method'] == 'BO-MADS'),
        combo_strictly_best_all_seeds=all(next(r['U'] for r in rows if r['seed'] == s and r['method'] == 'BO-MADS') >
            max(r['U'] for r in rows if r['seed'] == s and r['method'] != 'BO-MADS') + 1e-12 for s in cfg['seeds']),
        physical_method_ranking_confirmed=False, new_gpu_episodes=0,
        interpretation='Complete methods share identical charged initialization; negative/zero gains retained; archived combo wall times not concurrent')
    save(output / 'outer-comparison.json', result)
    return result


def fixed_routes(problem):
    tasks = problem.config.workload
    pure_g = {t: {'g': 1.} for t in tasks}
    shared = {t: {'s_shared': 1.} for t in tasks}
    specialists = {'text': {'s_text': 1.}, 'image': {'s_shared': 1.}}
    mixed = lambda q: {t: {'g': q, 's_shared': 1-q} for t in tasks}
    return [pure_g, shared, specialists, mixed(.3141592653589793), mixed(.6180339887498948), mixed(.3141592653589793)]


def exhaustive(problem, oracle, shares):
    """Independent Cartesian deployment enumeration, no MILP/pool/bounds."""
    routing = ContinuousRouteDomain(problem).to_qps(shares)
    names = sorted(problem.states)
    active = {m for row in routing.values() for m, q in row.items() if q > 0}
    records, best, best_x = [], None, None
    budget = Budget(120, 128, 1, 1)
    for counts in itertools.product(*(problem.states[m] for m in names)):
        x = {m: n for m, n in zip(names, counts) if n}
        if not active <= set(x):
            continue
        predictions = [oracle.predict(problem, x, routing, trace, budget) for trace in problem.train_traces]
        states_values = [normalize_prediction(p, problem.config.search_policy) for p in predictions]
        require(all(s != 'UNKNOWN' for s, _ in states_values), 'Synthetic reference contains unknown evidence')
        feasible = all(s == 'FEASIBLE' for s, _ in states_values)
        value = (sum(t.weight*v for t, (_, v) in zip(problem.train_traces, states_values)) /
                 sum(t.weight for t in problem.train_traces)) if feasible else None
        records.append(dict(deployment=x, feasible=feasible, U=value))
        if value is not None and (best is None or value > best):
            best, best_x = value, x
    return dict(U=best, deployment=best_x, domain_size=len(records), records=records,
                oracle_calls=budget.report()['full_prediction_calls'])


def verify_inner(row, truth):
    value = row['best_feasible_utility']
    upper = row.get('valid_upper_bound', row.get('valid_utility_upper_bound'))
    target = truth['U']
    if target is None:
        require(value is None, 'Feasible value invented in infeasible domain')
    else:
        require(value is None or value <= target + 1e-8, 'Feasible lower bound exceeds enumeration')
        require(upper is None or upper + 1e-8 >= target, 'Upper bound excludes true optimum')
    if row['certified']:
        require((value is None and target is None) or
                value is not None and target is not None and math.isclose(value, target, abs_tol=1e-8),
                'False fixed-route certification')


def inner_acceptance(output, cfg):
    spec = read(ROOT / cfg['fixture'])
    reports = []
    for workload in cfg['workloads']:
        config = SystemConfig(**copy.deepcopy(spec['system']))
        config = replace(config, workload=spec['workloads'][workload])
        problem = make_problem(config, {'train': [cfg['trace_seed']]})
        require(math.prod(len(ns) for ns in problem.states.values()) == cfg['maximum_cartesian_states'], 'Replica domain changed')
        routes = fixed_routes(problem)
        reference = [exhaustive(problem, oracle_from_spec(spec), s) for s in routes]
        results = []
        for policy in cfg['policies']:
            e = ContinuousRoutingEvaluator(problem, oracle_from_spec(spec), inner='milp', cache=True,
                budget=Budget(cfg['seconds_per_policy'], cfg['predictions_per_policy'], cfg['routes_per_policy'], cfg['milp_per_policy']))
            pool = DeploymentCandidatePool(problem)
            solver = ProgressiveFixedRoutingSolver(e, pool, inner_seconds=5., inner_predictions=2,
                inner_milp_calls=4, exploration_period=4)
            rows = []
            for shares, truth in zip(routes, reference):
                row = e.evaluate(shares) if policy == 'fresh_exact_milp' else solver.evaluate(shares, exact=policy == 'shared_exact')
                verify_inner(row, truth)
                if policy != 'shared_budget':
                    require(row['certified'], 'Exact inner did not complete within fixed allowance')
                rows.append(row)
            results.append(dict(policy=policy, rows=rows, budget=e.budget.report(), pool=pool.report()))
        # Full joint controller exercises the same non-singleton deployment domain.
        e = ContinuousRoutingEvaluator(problem, oracle_from_spec(spec), budget=Budget(
            cfg['outer_seconds'], cfg['outer_predictions'], cfg['outer_routes'], 512))
        raw = ActivationAwareBilevelOptimizer(e, seed=11, inner_seconds=5., inner_predictions=8,
            inner_milp_calls=16, poll_evaluations=4).solve()
        truth = exhaustive(problem, oracle_from_spec(spec), raw['winner']['routing_shares'])
        require(raw['winner']['best_feasible_utility'] <= truth['U'] + 1e-8, 'Joint incumbent exceeds independent reference')
        save(output / f'joint-fixture-{workload}.json', raw)
        report = dict(workload=workload, problem_sha256=problem.identity, states=problem.states,
            references=reference, policies=results, joint_winner=raw['winner'], joint_budget=raw['budget'],
            joint_winner_route_reference=truth, joint_pool_size=raw['pool']['size'],
            joint_visited_deployments=raw['pool']['entries'], evidence='synthetic FIFO fixture only',
            objective_schema=problem.config.utility_schema, not_measured_v3_benefit=True)
        save(output / f'inner-{workload}.json', report)
        reports.append(report)
        print(json.dumps(dict(workload=workload, exact_agreement=True, pool_size=raw['pool']['size'],
            policies=[dict(policy=r['policy'], budget=r['budget']) for r in results]), ensure_ascii=False), flush=True)
    result = dict(status='NONTRIVIAL_INNER_ACCEPTANCE_COMPLETED', reports=reports,
        independent_enumeration=True, exact_results_agree=True, partial_bounds_valid=True,
        real_multi_deployment_support=False, new_gpu_episodes=0)
    save(output / 'inner-comparison.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/bilevel_final_acceptance_20261105.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = read(a.config)
    require(cfg == read(ROOT / 'configs/bilevel_final_acceptance_20261105.json'), 'Fixed acceptance protocol required')
    a.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), a.config, ROOT/'experiments/search_scoped_utility_v3.py',
             ROOT/'experiments/scoped_routing_objective.py', ROOT/cfg['inner']['fixture'],
             *sorted((ROOT/'gsdeploy').rglob('*.py'))]
    receipts = [file_receipt(path) for path in paths]
    save(a.output/'plan.json', dict(config=cfg, source_receipts=receipts))
    try:
        outer = outer_acceptance(a.output, cfg['outer'])
        inner = inner_acceptance(a.output, cfg['inner'])
        require(all(file_receipt(r['path']) == r for r in receipts), 'Source changed during acceptance')
        save(a.output/'completion.json', dict(status='BOUNDED_ALGORITHM_ACCEPTANCE_COMPLETED',
            outer_status=outer['status'], inner_status=inner['status'],
            combo_synergy_demonstrated=outer['combo_strictly_best_all_seeds'],
            software_joint_optimization_verified=True, real_scope='fixed residency only',
            new_gpu_episodes=0, predictor_refits=0, remaining_automatic_work=[]))
    except BaseException as exc:
        save(a.output/'failure.json', dict(status='FAILED_NO_AUTOMATIC_RETRY', error=type(exc).__name__+': '+str(exc)))
        raise


if __name__ == '__main__':
    main()
