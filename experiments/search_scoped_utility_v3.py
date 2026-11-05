"""Bounded CPU search with default grouped v3 utility and frozen prediction laws.

Only the reviewed default profile is supported: original numerical references,
SLO k=2, and unit energy/quality scale factors. No collectors or refits invoked.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_scoped_routing import make_problem, ScopedOracle, run_joint
from experiments.run_scoped_decision_search import check_release
from experiments.scoped_routing_objective import file_receipt
from experiments.scoped_v3_search_decision import summarize_selection
from gsdeploy.continuous import ActivationAwareBilevelOptimizer, ContinuousRoutingEvaluator
from gsdeploy.continuous.contracts import Budget, digest
from gsdeploy.continuous.predictor_adapter import global_cost_floor
from gsdeploy.utility_tradeoffs import SCHEMA, TERMS, WEIGHT_PROFILES, score_terms

PROFILE = 'four_groups_v3'
CONTRACT = dict(utility_schema=SCHEMA, profile=PROFILE, weights=WEIGHT_PROFILES[PROFILE],
    normalization_schema='six_term_reference_v2', deadline_multiplier=2,
    latency_scale='frozen_v2_deadline_ms', energy_scale_factor=1, quality_span_factor=1,
    joint_delivery='diagnostic_only', feasibility='all_development_histories',
    interpretation='retrospective objective design; frozen predictor proposal search')
METHODS = ['Joint-K2', 'Activation-BO-MADS']
SEEDS = [11, 29, 47]
LIMITS = dict(seconds=300, predictions=64, routes=32, master_calls=128)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ScopedV3Oracle(ScopedOracle):
    """Both scoring and pruning use the same runtime weights; no old caches."""
    evidence_kind = 'frozen_scoped_predictor_grouped_v3_proposal'

    def _identity(self):
        return digest(dict(underlying=super()._identity(), objective=CONTRACT))

    def predict(self, problem, deployment, routing, trace, budget, deadline=None):
        p = super().predict(problem, deployment, routing, trace, budget, deadline)
        # An incomplete prediction stays unknown; never fill absent terms.
        if p.get('U') is None:
            return dict(p, utility_schema=SCHEMA, objective_profile=PROFILE)
        scored = score_terms({t: p[t + '_norm'] for t in TERMS}, CONTRACT['weights'])
        require(math.isclose(scored['U'], p['U'], abs_tol=1e-12, rel_tol=1e-12),
                'Search prediction and v3 evaluator objectives disagree')
        for term in TERMS:
            require(math.isclose(scored['contributions'][term], p[term + '_contribution'],
                                 abs_tol=1e-12, rel_tol=1e-12), 'Mixed-version contribution')
        p.update(utility_schema=SCHEMA, objective_profile=PROFILE,
                 normalization_schema='six_term_reference_v2', utility_contract=copy.deepcopy(CONTRACT))
        return p


def evaluator_v3(system, fit, evidence, templates, budget):
    runtime = copy.deepcopy(system)
    runtime['weights'] = copy.deepcopy(CONTRACT['weights'])
    spec = runtime['catalog']['utility_normalization']
    require(spec['deadline_multiplier'] == 2, 'Only reviewed k=2 default supported')
    require(all(runtime['latency_reference_ms'][t] == r['deadline_ms'] and
                math.isclose(r['deadline_ms'], 2*r['reference_mean_e2e_ms'], abs_tol=1e-10)
                for t, r in spec['tasks'].items()), 'Default v3 requires frozen deadline/latency scales')
    problem = make_problem(runtime, evidence, templates)
    # Keep the validated v2 normalization substrate; version the optimization
    # objective separately so no old prediction/cut/GP/cache identity can match.
    problem.objective_id = digest(dict(substrate_objective=problem.objective_id, contract=CONTRACT))
    oracle = ScopedV3Oracle(problem, fit, evidence, runtime, templates)
    return ContinuousRoutingEvaluator(problem, oracle, budget=budget)


def run_v3(system, fit, evidence, templates, method, seed, *, seconds=300., predictions=64, routes=32):
    require(method in METHODS, 'Unsupported method')
    e = evaluator_v3(system, fit, evidence, templates, Budget(seconds, predictions, routes, 128))
    if method == 'Joint-K2':
        result = run_joint(e)
    else:
        result = ActivationAwareBilevelOptimizer(e, seed=seed, initial_endpoints=4,
            inner_seconds=20., inner_predictions=4, inner_milp_calls=8, poll_evaluations=4).solve()
    result.update(utility_contract=copy.deepcopy(CONTRACT), objective_id=e.problem.objective_id,
        oracle_version=e.oracle.version, global_cost_floor=global_cost_floor(e.oracle, e.problem),
        physical_ranking_qualified=False, continuous_global_optimality_claim=False)
    return result


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def same_route(left, right):
    return (set(left) == set(right) and all(
        math.isclose(left[t].get(m, 0), right[t].get(m, 0), abs_tol=1e-12, rel_tol=1e-10)
        for t in left for m in set(left[t]) | set(right[t])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'artifacts/activation-gpu-20261104')
    parser.add_argument('--review', type=Path, default=ROOT / 'artifacts/utility-tradeoffs-v3-20261105/results.json')
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    require(not a.output.exists(), 'Output already exists; no automatic retries or overwrites')
    campaign = a.campaign.resolve()
    validation = campaign / 'five-model-aggregate-validation-v1'
    release_path = campaign / 'scoped-release-handoff-v1/decision-use-release-v2.json'
    release = read(release_path)
    check_release(release)
    review = read(a.review)
    verification_path = a.review.parent / 'verification.json'
    verification = read(verification_path)
    require(verification['status'] == 'PASS_OFFLINE_UTILITY_V3_REVIEW' and
            file_receipt(a.review) == verification['result'], 'Reviewed result receipt mismatch')
    require(review['status'] == 'OFFLINE_REVIEW_COMPLETED' and
            review['default']['profile'] == PROFILE and review['evaluation_role'] == 'retrospective_exploratory' and
            review['weight_profiles'][PROFILE] == CONTRACT['weights'], 'Completed matching v3 review required')
    for r in review['source_receipts'] + review['implementation_receipts']:
        require(file_receipt(r['path']) == r, 'Reviewed input or implementation changed: ' + r['path'])
    paths = dict(reference=campaign / 'references-v2/system-frozen.json',
                 fit=validation / 'fit-frozen.json',
                 evidence=campaign / 'scoped-optimization-preparation-v1/training-evidence.json')
    system, fit, evidence = (read(paths[k]) for k in ('reference', 'fit', 'evidence'))
    require(file_receipt(paths['reference']) in evidence['sources'], 'Reference identity mismatch')
    require(file_receipt(paths['fit']) in release['source_receipts'], 'Released fit identity mismatch')
    require(set(evidence['solo_sources']) <= set(fit['source_receipts']), 'Solo source mismatch')
    for r in evidence['sources']:
        require(file_receipt(r['path']) == r, 'Training evidence changed')
    templates = []
    template_paths = [validation / f'aggregate-validation-round{i}/plan.json' for i in (1, 2)]
    for path in template_paths:
        plan = read(path)
        plan['jobs'] = [plan['jobs'][0]]
        templates.append(plan)
    require(all(p['split'] != 'confirmation' for p in templates), 'Confirmation cannot become search data')
    for r in read(validation / 'protocol.json')['source_receipts']:
        require(file_receipt(r['path'])['sha256'] == r['sha256'], 'Frozen predictor source changed')
    old_path = campaign / 'scoped-optimization-decision-v2/comparison/comparison.json'
    old_plan_path = campaign / 'scoped-optimization-decision-v2/plan.json'
    old_plan = read(old_plan_path)
    for r in old_plan['source_receipts']:
        require(file_receipt(r['path']) == r, 'Original search source/input changed')
    require(all(file_receipt(p) in old_plan['source_receipts'] for p in template_paths),
            'Development templates differ from original search')
    original = read(old_path)
    # Snapshot before execution. All inputs stay versioned; old files are read only.
    sources = {Path(__file__).resolve(), ROOT / 'experiments/benchmark_scoped_routing.py',
        ROOT / 'experiments/scoped_v3_search_decision.py',
        ROOT / 'experiments/run_scoped_decision_search.py', ROOT / 'experiments/scoped_routing_objective.py',
        a.review.resolve(), verification_path.resolve(), release_path, old_path, old_plan_path, *paths.values(), *template_paths,
        *[Path(r['path']) for r in read(validation / 'protocol.json')['source_receipts']],
        *sorted((ROOT / 'gsdeploy').rglob('*.py'))}
    receipts = [file_receipt(p) for p in sorted(sources)]
    a.output.mkdir(parents=True, exist_ok=False)
    snapshots = []
    for i, r in enumerate(receipts):
        target = a.output / 'source-snapshot' / str(i) / Path(r['path']).name
        target.parent.mkdir(parents=True)
        target.write_bytes(Path(r['path']).read_bytes())
        snapshots.append(dict(r, snapshot=str(target.resolve())))
    save(a.output / 'plan.json', dict(schema='gsdeploy.scoped_v3_search_plan.v1',
        objective=CONTRACT, methods=METHODS, seeds=SEEDS, budget_per_run=LIMITS, maximum_runs=6,
        bound_contract='Cost >= w_Q * sum(alpha_t * (0-best_t)/(worst_t-best_t)); other losses nonnegative; '
                       'profiled standalone memory/service bounds disabled; identity feedback from current v3 full cost only',
        source_receipts=snapshots, new_gpu_episodes=0, refits=0, automatic_retries=0,
        stop_rule='Finish these six CPU runs; report existing evidence or unconfirmed new proposals; no new experiments',
        scope='Fixed five-model residency, one replica, same two development histories; no confirmation feedback',
        original_strict_latency_gate_pass=False))
    rows = []
    try:
        for seed in SEEDS:
            for method in (list(reversed(METHODS)) if seed % 3 == 2 else METHODS):
                result = run_v3(system, fit, evidence, templates, method, seed)
                save(a.output / f'{method}-{seed}.json', result)
                w = result.get('winner')
                require(w is not None, 'No feasible winner; bounded run stops without retry')
                row = dict(method=method, seed=seed, status=result['status'], budget=result['budget'],
                    utility=w.get('score', w.get('best_feasible_utility')),
                    routing=w.get('routing', w.get('routing_qps')), deployment=w['deployment'])
                prior = next(r for r in original['rows'] if r['method'] == method and r['seed'] == seed)
                row['same_route_and_deployment_as_v2'] = same_route(row['routing'], prior['routing']) and row['deployment'] == prior['deployment']
                row['matches_archived_pure_specialist'] = (
                    all({m for m, q in route.items() if q > 0} == {t} for t, route in row['routing'].items())
                    and row['deployment'] == {m: 1 for m in evidence['resident_aliases'].values()})
                b = result['budget']
                require(b['full_prediction_calls'] <= 64 and b['outer_evaluations'] <= 32 and b['milp_solve_calls'] <= 128,
                        'Search count budget exceeded')
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        for r in receipts:
            require(file_receipt(r['path']) == r, 'Search input changed during execution')
        matched = all(r['matches_archived_pure_specialist'] for r in rows)
        save(a.output / 'comparison.json', dict(schema='gsdeploy.scoped_v3_search.v1', rows=rows,
            utility_contract=CONTRACT, all_same_as_v2=all(r['same_route_and_deployment_as_v2'] for r in rows),
            all_match_archived_control=matched, selection=summarize_selection(rows, review),
            archived_panel_evidence=file_receipt(a.review), independent_v3_confirmation=False,
            new_gpu_episodes=0,
            new_predictor_fits=0, scope_expansion=False, continuous_global_optimality_claim=False,
            disposition='Select best proposal with explicit evidence coverage; no automatic GPU runs'))
        save(a.output / 'completion.json', dict(status='BOUNDED_V3_SEARCH_COMPLETED', runs=6,
            all_match_archived_control=matched, new_gpu_episodes=0, remaining_automatic_work=[]))
    except BaseException as exc:
        save(a.output / 'failure.json', dict(status='FAILED_NO_RETRY', completed_runs=len(rows),
                                           error=type(exc).__name__ + ': ' + str(exc)))
        raise


if __name__ == '__main__':
    main()
