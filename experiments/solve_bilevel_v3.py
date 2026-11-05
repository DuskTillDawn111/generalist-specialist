"""Run the complete BO/MADS + deployment/replica MILP with explicit v3 scoring.

Accepts only ready reference-v2 numerical contracts and their declared predictor
coverage. This entry point grants no new physical support and never collects data.
The frozen real five-model scope remains available via search_scoped_utility_v3.
"""
import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.scoped_routing_objective import file_receipt
from gsdeploy.config import SystemConfig
from gsdeploy.continuous import (ActivationAwareBilevelOptimizer, Budget, ContinuousRoutingEvaluator,
                                ImmutableContinuousProblem, SessionPredictorOracle)
from gsdeploy.continuous.contracts import digest
from gsdeploy.continuous.budget_scheduled_bilevel import BudgetScheduledBilevelOptimizer
from gsdeploy.utility_tradeoffs import SCHEMA, TERMS, WEIGHT_PROFILES, score_terms

OBJECTIVE = dict(schema=SCHEMA, profile='four_groups_v3', weights=WEIGHT_PROFILES['four_groups_v3'],
                 scales='unchanged ready explicit reference-v2 contract', joint_delivery='diagnostic_only')


class GroupedReferenceOracle:
    """Verify complete six-term scores and isolate cache identity from v2."""
    evidence_kind = 'declared_reference_predictor_grouped_v3'

    def __init__(self, base, problem):
        self.base, self.problem = base, problem
        if problem.config.weights != OBJECTIVE['weights']:
            raise ValueError('The solver and oracle must use the same v3 weights')
        self.supports_stress = base.supports_stress
        self.cost_lower_bound = None  # Reference-aware solver computes the quality floor.
        self.version = self._identity()

    def _identity(self):
        return digest(dict(base=self.base.version, problem=self.problem.identity,
                           objective=OBJECTIVE, implementation=file_receipt(__file__)))

    def check(self):
        self.base.check()
        self.problem.check()
        if self.version != self._identity():
            raise ValueError('Grouped objective/oracle changed during search')

    def state_costs(self, problem):
        return self.base.state_costs(problem)

    def memory_states(self, problem):
        return self.base.memory_states(problem)

    def predict(self, problem, deployment, routing, trace, budget, deadline=None):
        if problem is not self.problem:
            raise ValueError('Oracle/problem mismatch')
        result = self.base.predict(problem, deployment, routing, trace, budget, deadline)
        if result.get('U') is not None:
            scored = score_terms({t: result[t+'_norm'] for t in TERMS}, OBJECTIVE['weights'])
            if not math.isclose(result['U'], scored['U'], abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError('Underlying predictor used a different utility objective')
            for t in TERMS:
                if not math.isclose(result[t+'_contribution'], scored['contributions'][t], abs_tol=1e-12):
                    raise ValueError('Mixed utility contribution versions')
        return dict(result, utility_schema=SCHEMA, normalization_schema='six_term_reference_v2',
                    objective_profile='four_groups_v3')


def solve(config, *, seed=11, seconds=300., predictions=64, routes=32, milp=128,
          use_bo=True, use_mads=True, scheduler='balanced'):
    if scheduler not in {'balanced', 'legacy'}:
        raise ValueError('Unknown bilevel scheduler')
    if config.utility_schema != 'six_term_reference_v2' or config.search_policy != 'nominal':
        raise ValueError('Ready reference-v2 numerical contract and nominal policy required')
    runtime = replace(config, weights=copy.deepcopy(OBJECTIVE['weights']))
    problem = ImmutableContinuousProblem(runtime)
    # Numeric objective identity already includes the new weights. The wrapper
    # adds the v3 schema to oracle/evaluator cache identity without mutating the
    # immutable problem or bypassing its guard.
    with SessionPredictorOracle(problem) as base:
        oracle = GroupedReferenceOracle(base, problem)
        evaluator = ContinuousRoutingEvaluator(problem, oracle,
            budget=Budget(seconds, predictions, routes, milp))
        optimizer = BudgetScheduledBilevelOptimizer if scheduler == 'balanced' else ActivationAwareBilevelOptimizer
        result = optimizer(evaluator, seed=seed, initial_endpoints=4,
            inner_seconds=min(20., seconds), inner_predictions=8, inner_milp_calls=16,
            poll_evaluations=4, use_bo=use_bo, use_mads=use_mads).solve()
        result.update(objective_contract=copy.deepcopy(OBJECTIVE), declared_replica_states=problem.states,
            scheduler_mode=scheduler,
            objective_contract_sha256=digest(OBJECTIVE),
            normalization_schema='six_term_reference_v2', physical_scope_expansion=False,
            physical_qualification_granted=False, engine_stats=base.engine_stats())
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--predictions', type=int, default=64)
    parser.add_argument('--routes', type=int, default=32)
    parser.add_argument('--milp-calls', type=int, default=128)
    parser.add_argument('--method', choices=['BO-MADS', 'BO-only', 'MADS-only'], default='BO-MADS')
    parser.add_argument('--scheduler', choices=['balanced', 'legacy'], default='balanced')
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), a.config, *sorted((ROOT/'gsdeploy').rglob('*.py'))]
    receipts = [file_receipt(p) for p in paths]
    def save(name, value):
        with (a.output/name).open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    save('plan.json', dict(objective=OBJECTIVE, source_receipts=receipts,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, new_gpu_episodes=0))
    try:
        result = solve(SystemConfig.load(a.config), seed=a.seed, seconds=a.seconds,
            predictions=a.predictions, routes=a.routes, milp=a.milp_calls,
            use_bo=a.method != 'MADS-only', use_mads=a.method != 'BO-only', scheduler=a.scheduler)
        if any(file_receipt(r['path']) != r for r in receipts):
            raise ValueError('Input or implementation changed during solve')
        save('result.json', result)
        print(json.dumps(dict(status=result['status'], budget=result['budget'],
                              winner=result['winner'], physical_qualification_granted=False), ensure_ascii=False))
    except BaseException as exc:
        save('failure.json', dict(error=type(exc).__name__+': '+str(exc), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
