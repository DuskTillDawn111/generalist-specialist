"""Bounded Joint K2 / activation BO-MADS comparison at fixed residency.

Experimental adapter only. Both methods use the same whole-episode oracle,
six-term objective, source inputs, and non-renewing budgets. Run only after
the scoped release, except an explicitly limited software wiring smoke.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.scoped_routing_objective import predict_candidate, scoped_reference, file_receipt
from gsdeploy.config import SystemConfig
from gsdeploy.continuous import ActivationAwareBilevelOptimizer, ContinuousRoutingEvaluator
from gsdeploy.continuous.contracts import Budget, BudgetExhausted, ContinuousProblem, TraceSpec, digest
from gsdeploy import milp, normalization as norm


def make_problem(system, evidence, templates):
    if len(templates) != 2:
        raise ValueError('Two declared development histories required')
    data, rates = scoped_reference(system, templates[0])
    roster = set(evidence['resident_aliases'].values())
    data['models'] = {m: copy.deepcopy(data['models'][m]) for m in sorted(roster)}
    for m in data['models']:
        data['models'][m]['service_time_ms'] = copy.deepcopy(evidence['capacity_service_ms'][m])
    data['capacity_factors'] = {m: {'0': 0., '1': 1.} for m in roster}
    data['memory_table_mib'] = {m: {k: data['memory_table_mib'][m][k] for k in ('0', '1')} for m in roster}
    # The default predictor is never invoked. Profiled mode disables invalid
    # standalone service/memory bounds in the retained original Joint master.
    data.update(performance_mode='profiled', memory_mode='profiled', colocation_profiles={})
    config = copy.deepcopy(system)
    config.update(catalog=data, workload=rates, horizon_seconds=templates[0]['jobs'][0]['duration_ms']/1000.,
                  search_policy='nominal', route_grid=2, route_pair_denominators_by_task={},
                  milp_capacity_witness=False)
    traces = []
    for i, plan in enumerate(templates):
        _, other_rates = scoped_reference(system, plan)
        if other_rates != rates or plan['aliases'] != evidence['resident_aliases']:
            raise ValueError('Comparison histories must share rates and residency')
        rows = sorted(plan['jobs'][0]['requests'], key=lambda r: r['offset_ms'])
        traces.append(TraceSpec(f'development-history-{i}', plan['seed'],
                                tuple((r['offset_ms']/1000., r['task']) for r in rows)))
    problem = ContinuousProblem(SystemConfig(**config), traces, {m: [1] for m in roster})
    problem.states = {m: (1,) for m in sorted(roster)}
    problem.identity = problem._identity()
    problem.check()
    return problem


class ScopedOracle:
    evidence_kind = 'measured_training_fit_scoped_routing_estimate'
    supports_stress = False
    cost_lower_bound = None

    def __init__(self, problem, fit, evidence, system, templates):
        self.problem = problem
        self.fit, self.evidence, self.system = map(copy.deepcopy, (fit, evidence, system))
        self.templates = {t.identity: copy.deepcopy(p) for t, p in zip(problem.train_traces, templates)}
        self.version = self._identity()

    def _identity(self):
        return digest(dict(fit=self.fit, evidence=self.evidence, reference=self.system, templates=self.templates))

    def check(self):
        if self._identity() != self.version:
            raise ValueError('Scoped oracle inputs changed')

    def state_costs(self, problem):
        return {}

    def memory_states(self, problem):
        return {}

    def predict(self, problem, deployment, routing, trace, budget, deadline=None):
        if problem is not self.problem or deployment != {m: 1 for m in problem.states}:
            raise ValueError('Only the frozen all-resident single-replica deployment is supported')
        budget.consume('predictions', deadline)
        budget.counts['predictor_leaves'] += 1
        return predict_candidate(self.fit, self.templates[trace.identity], routing, self.evidence, self.system)


class ConservativeJoint(milp.JointMilpOptimizer):
    def floor(self, task, deployment, routing):
        # Old per-input quality routing may select a different input cohort.
        # Native loss >=0 gives a valid bound for our exact offered cohort.
        return self.weights['Q']*self.rates[task]/sum(self.rates.values())*norm.quality_loss(self.data, task, 0.)


def run_joint(evaluator):
    e, budget = evaluator, evaluator.budget
    c = e.problem.config
    opt = ConservativeJoint(c.catalog, c.workload, c.gpu_memory_mib, c.latency_reference_ms, c.weights,
        horizon=c.horizon_seconds, seed=c.random_seed, arrival_process=c.arrival_process,
        milp_capacity_witness=False, milp_time_limit_seconds=max(.001, budget.deadline-time.perf_counter()))
    opt.trace = list(e.problem.train_traces[0].arrivals)
    opt.counts = {t: sum(task == t for _, task in opt.trace) for t in c.workload}
    events = []
    original_master = milp.solve_master
    def master(*args, **kwargs):
        budget.consume('milp')
        kwargs['options']['time_limit'] = max(.001, min(kwargs['options']['time_limit'], budget.deadline-time.perf_counter()))
        started = time.perf_counter()
        try:
            return original_master(*args, **kwargs)
        finally:
            budget.times['milp'] += time.perf_counter()-started
    def predict(data, deployment, routing, *args, **kwargs):
        budget.consume('outer')
        candidate = e.candidate(deployment, routing)
        parts = [row['prediction'] for row in candidate['traces']]
        # Both histories have equal declared weight. Preserve per-trace
        # evidence; eligibility requires every history, never an average.
        result = dict(constraints_eligible=candidate['constraint_state'] == 'FEASIBLE',
            performance_unknown=candidate['constraint_state'] == 'UNKNOWN', U=candidate['utility'],
            reason=candidate['constraint_state'], scoped_candidate=candidate)
        for key in ('M_norm', 'Q_contribution', 'T_contribution', 'L_contribution', 'P_contribution'):
            result[key] = sum(p[key] for p in parts)/len(parts) if all(key in p for p in parts) else None
        result['scenario_delay_costs'] = [-sum(result[k+'_contribution'] for k in ('T', 'L', 'P'))] if result['U'] is not None else [None]
        events.append(dict(**budget.report(), routing=copy.deepcopy(routing), utility=candidate['utility']))
        return result
    with patch.object(milp, 'solve_master', master), patch.object(milp, 'predict_scenarios', predict):
        try:
            raw = opt.solve(K=2, deployment={m: 1 for m in e.problem.states})
        except BudgetExhausted as exc:
            raw = opt.result(2, str(exc), False)
    return dict(algorithm='Joint-K2', status=raw['status'], winner=raw['winner'], raw=raw,
                events=events, budget=budget.report(), physical_ranking_qualified=False)


def run(system, fit, evidence, templates, method, seed, *, seconds=300., predictions=64, routes=32):
    budget = Budget(seconds, predictions, routes, 128)
    problem = make_problem(system, evidence, templates)
    oracle = ScopedOracle(problem, fit, evidence, system, templates)
    e = ContinuousRoutingEvaluator(problem, oracle, budget=budget)
    if method == 'Joint-K2':
        return run_joint(e)
    if method != 'Activation-BO-MADS':
        raise ValueError('Only the two predeclared methods are supported')
    result = ActivationAwareBilevelOptimizer(e, seed=seed, initial_endpoints=4,
        inner_seconds=20., inner_predictions=4, inner_milp_calls=8, poll_evaluations=4).solve()
    result['physical_ranking_qualified'] = False
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-base', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--release', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--software-smoke', action='store_true')
    args = p.parse_args()
    if not args.software_smoke:
        if args.release is None:
            p.error('Completed scoped release required')
        release = json.loads(args.release.read_text())
        if release['status'] != 'SCOPED_PASS':
            p.error('Restricted/failed validation requires decision-impact review; no automatic broad route search')
        actual = file_receipt(args.validation_base/'fit-frozen.json')
        if release['frozen_inputs_sha256'].get(actual['path']) != actual['sha256']:
            p.error('Release fit identity differs from optimization fit')
        receipt = file_receipt(args.validation_base/'validation.json')
        if release['evidence_sha256'].get(receipt['path']) != receipt['sha256']:
            p.error('Release validation identity differs')
        for path, expected in {**release['frozen_inputs_sha256'], **release['evidence_sha256']}.items():
            if file_receipt(path)['sha256'] != expected:
                p.error('Release evidence changed: '+path)
    system = json.loads(args.reference.read_text())
    fit = json.loads((args.validation_base/'fit-frozen.json').read_text())
    evidence = json.loads(args.evidence.read_text())
    if file_receipt(args.reference) not in evidence['sources']:
        p.error('Quality/resource evidence uses a different reference')
    if not set(evidence['solo_sources']) <= set(fit['source_receipts']):
        p.error('Capacity/quality sources differ from fitted solo sources')
    for receipt in evidence['sources']:
        if file_receipt(receipt['path']) != receipt:
            p.error('Training evidence source changed')
    templates = []
    for i in (1, 2):
        plan = json.loads((args.validation_base/f'aggregate-validation-round{i}/plan.json').read_text())
        plan['jobs'] = [plan['jobs'][0]]
        templates.append(plan)
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    paths = [Path(__file__), ROOT/'experiments/scoped_routing_objective.py', args.evidence, args.reference,
             args.validation_base/'fit-frozen.json', *sorted((ROOT/'gsdeploy/continuous').glob('*.py')),
             ROOT/'gsdeploy/milp.py', ROOT/'gsdeploy/prediction.py']
    protocol = json.loads((args.validation_base/'protocol.json').read_text())
    for receipt in protocol['source_receipts']:
        if file_receipt(receipt['path'])['sha256'] != receipt['sha256']:
            p.error('Sealed prediction source changed')
        paths.append(Path(receipt['path']))
    paths.extend(args.validation_base/f'aggregate-validation-round{i}/plan.json' for i in (1, 2))
    if args.release is not None:
        paths.append(args.release)
    receipts = [file_receipt(path) for path in paths]
    save('plan.json', dict(scope='software wiring only' if args.software_smoke else 'scoped prediction search; measured ranking pending',
        source_receipts=receipts, fixed_residency=True, replica_counts=[1],
        methods=['Joint-K2', 'Activation-BO-MADS'], seeds=[11] if args.software_smoke else [11, 29, 47],
        budget=dict(seconds=60 if args.software_smoke else 300, predictions=4 if args.software_smoke else 64,
                    routes=2 if args.software_smoke else 32, master_calls=128),
        reference_reuse=scoped_reference(system, templates[0])[0][norm.KEY],
        extra_gpu_runs=0, predictor_refits=0, original_reference_file_unchanged=True))
    save('state.json', dict(status='running_software_smoke' if args.software_smoke else 'running_bounded_optimization'))
    try:
        summaries = []
        for seed in ([11] if args.software_smoke else [11, 29, 47]):
            methods = ['Joint-K2', 'Activation-BO-MADS']
            if seed % 3 == 2:
                methods.reverse()
            for method in methods:
                raw = run(system, fit, evidence, templates, method, seed,
                    seconds=60 if args.software_smoke else 300,
                    predictions=4 if args.software_smoke else 64, routes=2 if args.software_smoke else 32)
                save(f'{method}-{seed}.json', raw)
                winner = raw.get('winner')
                summaries.append(dict(method=method, seed=seed, status=raw['status'], budget=raw['budget'],
                    utility=(winner.get('score', winner.get('best_feasible_utility')) if winner else None),
                    routing=(winner.get('routing', winner.get('routing_qps')) if winner else None),
                    deployment=winner.get('deployment') if winner else None))
                print(json.dumps(dict(method=method, seed=seed, status=raw['status'], budget=raw['budget'])), flush=True)
        if any(file_receipt(r['path']) != r for r in receipts):
            raise ValueError('Comparison input/source changed during execution')
        save('comparison.json', dict(rows=summaries, software_smoke=args.software_smoke,
            selection_scope='prediction search on the two development histories; fixed residency',
            input_generalization_claim=False, measured_ranking_qualified=False,
            deployment_optimization_claim=False, continuous_global_optimality_claim=False))
        save('state.json', dict(status='software_smoke_completed' if args.software_smoke else 'bounded_search_completed',
                               measured_ranking_qualified=False))
    except BaseException as exc:
        save('state.json', dict(status='failed', error=type(exc).__name__+': '+str(exc)))
        raise


if __name__ == '__main__':
    main()
