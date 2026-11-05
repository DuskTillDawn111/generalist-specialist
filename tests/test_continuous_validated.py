"""Independent full-predictor checks for optional validated bounds and sessions.

Fitted fixtures here are explicitly CPU test data, never research measurements.
"""

from dataclasses import FrozenInstanceError, replace
import math
import sys
import time
import unittest
from unittest.mock import patch

from gsdeploy.continuous import (Budget, ContinuousProblem, ContinuousRoutingEvaluator,
    ImmutableContinuousProblem, LegacyPredictorOracle, SessionPredictorOracle, TraceSpec)
from gsdeploy.continuous.activation import deployments
from gsdeploy.continuous.validated_bounds import ValidatedRouteBounds
from gsdeploy.continuous.fixed_routing_milp import FixedRoutingMilpOptimizer
from gsdeploy.trace import make_trace
from tests.test_calibrated_search_bounds_v6 import search_fixture, seal
from tests.test_calibrated_replay_v2 import fitted_replay_fixture
from tests.test_continuous_routing import TableOracle, tiny


def traces(c):
    # A declared custom realization exercises the explicit-trace override in
    # the bound helpers as well as the private prediction adapter.
    arrivals = make_trace(c.workload, c.horizon_seconds, 51, c.arrival_process)
    custom = tuple((a*.91, t) for a, t in arrivals)
    return [TraceSpec('train-a', 42, make_trace(c.workload, c.horizon_seconds, 42, c.arrival_process)),
            TraceSpec('train-custom', 51, custom, weight=2),
            TraceSpec('heldout', 61, make_trace(c.workload, c.horizon_seconds, 61, c.arrival_process), 'test')]


class ContinuousValidatedTests(unittest.TestCase):
    def test_every_bound_dominates_complete_prediction_with_custom_multitrace(self):
        c = replace(fitted_replay_fixture(quality_mode='per_sample_replay'), horizon_seconds=20.)
        checked = 0
        for policy in ('nominal', 'robust'):
            p = ContinuousProblem(replace(c, search_policy=policy), traces(c))
            e = ContinuousRoutingEvaluator(p, LegacyPredictorOracle(), budget=Budget(60, 1000))
            b = ValidatedRouteBounds(e)
            self.assertTrue(b.available)
            routes = list(e.domain.endpoints())+[e.domain.decode([v]*e.domain.dimension) for v in (1e-14, .314159)]
            for shares in routes:
                routing = e.domain.to_qps(shares)
                for x in deployments(p, routing):
                    opt = b.optimizers[0][1]
                    memory = math.fsum(opt.state_memory(m, n) for m, n in x.items())
                    floor = c.weights['M']*memory/c.gpu_memory_mib + c.weights['E']*opt.surrogate.state_energy(sum(x.values()))
                    floor += math.fsum(b.task_floor(t, {m: x[m] for m, q in row.items() if q > 0}, row, None)
                                       for t, row in routing.items())
                    actual = e.candidate(x, routing)
                    if actual['utility'] is not None:
                        self.assertLessEqual(floor, -actual['utility']+1e-10)
                    if memory > c.gpu_memory_mib+1e-6:
                        self.assertEqual(actual['constraint_state'], 'INFEASIBLE')
                    checked += 1
        self.assertGreater(checked, 40)

    def test_four_ablations_equal_independent_enumeration(self):
        c = search_fixture()
        for policy in ('nominal', 'robust'):
            c = replace(c, search_policy=policy)
            plain = ContinuousProblem(c, traces(c))
            frozen = ImmutableContinuousProblem(c, traces(c))
            with SessionPredictorOracle(frozen) as oracle:
                es = [ContinuousRoutingEvaluator(p, o, mode, Budget(60, 1000, 100), cache=False)
                      for p, o, mode in [(plain, LegacyPredictorOracle(), 'enumeration'),
                          (plain, LegacyPredictorOracle(), 'milp'),
                          (plain, LegacyPredictorOracle(), 'milp_bounds'),
                          (frozen, oracle, 'milp'), (frozen, oracle, 'milp_bounds')]]
                for route in list(es[0].domain.endpoints())+[es[0].domain.decode([1e-14]*2), es[0].domain.decode([.31]*2)]:
                    rows = [e.evaluate(route) for e in es]
                    for row in rows:
                        self.assertEqual(row['best_feasible_utility'], rows[0]['best_feasible_utility'])
                        self.assertEqual(row['certified'], rows[0]['certified'])
                        self.assertEqual(row['remaining_deployments'], 0)
                        self.assertEqual(row['evaluated_deployments']+row['safely_pruned_deployments'], row['deployment_domain_size'])
                self.assertLess(es[2].budget.counts['predictions'], es[1].budget.counts['predictions'])

    def test_full_prediction_json_equal_for_explicit_train_and_heldout(self):
        c = replace(fitted_replay_fixture(quality_mode='per_sample_replay'), horizon_seconds=20.)
        ps = [ContinuousProblem(c, traces(c)), ImmutableContinuousProblem(c, traces(c))]
        self.assertEqual(ps[0].identity, ps[1].identity)
        with SessionPredictorOracle(ps[1]) as oracle:
            es = [ContinuousRoutingEvaluator(ps[0], LegacyPredictorOracle(), cache=False),
                  ContinuousRoutingEvaluator(ps[1], oracle, cache=False)]
            for v in (0, 1e-14, .3159, 1):
                r = es[0].domain.to_qps(es[0].domain.decode([v]*2))
                x = {'generalist': 2, 'specialist': 1}
                a, b = [e.candidate(x, r, traces=e.problem.traces) for e in es]
                self.assertEqual(a, b)
            self.assertEqual(es[1].budget.counts['predictions'], 12)
            self.assertGreater(oracle.session.memo.hits, 0)

    def test_frozen_inputs_detached_and_all_normal_mutations_rejected(self):
        c = search_fixture()
        p = ImmutableContinuousProblem(c)
        identity = p.identity
        c.catalog['safe_factor'] = 999
        self.assertNotEqual(c.catalog['safe_factor'], p.config.catalog['safe_factor'])
        for mutate in (lambda: p.config.catalog.update(safe_factor=9),
                       lambda: p.states.clear(), lambda: p.states['generalist'].append(3),
                       lambda: setattr(p, 'config', c), lambda: setattr(p, 'identity', 'changed'),
                       lambda: setattr(p.config, 'horizon_seconds', 2)):
            with self.assertRaises((TypeError, FrozenInstanceError)):
                mutate()
        p.check()
        self.assertEqual(identity, p.identity)
        mutable = ContinuousProblem(search_fixture())
        mutable.config.catalog['safe_factor'] = 999
        with self.assertRaisesRegex(ValueError, 'PROBLEM_CHANGED'):
            mutable.check()

    def test_session_isolation_close_source_and_binding_invalidation(self):
        p = ImmutableContinuousProblem(search_fixture())
        with SessionPredictorOracle(p) as a, SessionPredictorOracle(p) as b:
            self.assertNotEqual(a.session.prefix, b.session.prefix)
            self.assertIsNot(a.session.memo.cache, b.session.memo.cache)
            with self.assertRaisesRegex(ValueError, 'PROBLEM_MISMATCH'):
                ContinuousRoutingEvaluator(ImmutableContinuousProblem(search_fixture()), a)
            with patch('gsdeploy.continuous.predictor_adapter.source_identity', return_value='changed'):
                with self.assertRaisesRegex(ValueError, 'SOURCE_CHANGED'):
                    a.check()
            prefix = a.session.prefix
        self.assertFalse(any(k.startswith(prefix) for k in sys.modules))
        with self.assertRaisesRegex(ValueError, 'SESSION_CLOSED'):
            a.check()
        with self.assertRaises(TypeError):
            SessionPredictorOracle(ContinuousProblem(search_fixture()))

    def test_nonseparable_unknown_and_positive_utility_fallback(self):
        route = {'text': {'g': 1}, 'image': {'g': 1}}
        for score in (lambda x,r,t: -.1 if 's_shared' in x else -.9,
                      lambda x,r,t: 'unknown' if 's_shared' in x else -.1,
                      lambda x,r,t: 2 if 's_shared' in x else 1):
            o = TableOracle(score)
            o.cost_lower_bound = None
            a = ContinuousRoutingEvaluator(tiny(), o, 'enumeration').evaluate(route)
            b = ContinuousRoutingEvaluator(tiny(), o, 'milp_bounds').evaluate(route)
            self.assertEqual(a['best_feasible_utility'], b['best_feasible_utility'])
            self.assertEqual(a['certified'], b['certified'])
            self.assertFalse(b['validated_bounds_enabled'])

    def test_unknown_calibration_gate_uses_conservative_fallback(self):
        c = search_fixture()
        c.catalog['calibrated_surrogate']['status'] = 'UNVALIDATED'
        p = ContinuousProblem(c)
        r = ContinuousRoutingEvaluator(p, LegacyPredictorOracle(), 'milp_bounds').evaluate(
            {'text': {'generalist': 1}, 'image': {'specialist': 1}})
        self.assertFalse(r['validated_bounds_enabled'])
        self.assertFalse(r['certified'])
        self.assertIsNone(r['best_feasible_utility'])

    def test_column_budget_has_no_certificate_and_does_not_cache_partial_route(self):
        p = ContinuousProblem(replace(search_fixture(), milp_max_columns=6))
        e = ContinuousRoutingEvaluator(p, LegacyPredictorOracle(), 'milp_bounds')
        r = e.evaluate({'text': {'generalist': 1}, 'image': {'specialist': 1}})
        self.assertEqual(r['termination_reason'], 'COLUMN_LIMIT')
        self.assertFalse(r['certified'])
        self.assertFalse(e.routing_cache)
        self.assertEqual(r['full_prediction_calls'], 0)
        self.assertEqual(r['remaining_deployments'], 4)

    def test_bad_bound_clears_certificate_and_upper(self):
        p = ContinuousProblem(search_fixture())
        with patch.object(ValidatedRouteBounds, 'task_floor', return_value=100.):
            r = ContinuousRoutingEvaluator(p, LegacyPredictorOracle(), 'milp_bounds').evaluate(
                {'text': {'generalist': 1}, 'image': {'specialist': 1}})
        self.assertEqual(r['termination_reason'], 'BOUND_CONTRACT_VIOLATION')
        self.assertEqual(r['status'], 'SOLVER_ERROR')
        self.assertFalse(r['certified'])
        self.assertIsNone(r['valid_utility_upper_bound'])

    def test_sparse_test_predictor_keeps_missing_state_unknown(self):
        # A separately sealed CPU fixture, never a modification to real data.
        from gsdeploy.calibration import _baseline_identity
        c = search_fixture()
        del c.catalog['task_replica_profiles']['generalist']['text']['1']
        c.catalog['calibrated_surrogate']['baseline_identities'] = _baseline_identity(c.catalog)
        seal(c.catalog)
        p = ContinuousProblem(c, replica_options={'generalist':[1], 'specialist':[1]})
        e = ContinuousRoutingEvaluator(p, LegacyPredictorOracle(), 'milp_bounds')
        r = e.evaluate({'text': {'generalist': 1}, 'image': {'specialist': 1}})
        self.assertTrue(r['validated_bounds_enabled'])
        self.assertEqual(r['status'], 'INCOMPLETE_EVIDENCE')
        self.assertFalse(r['certified'])
        self.assertIsNone(r['best_feasible_utility'])
        self.assertEqual(len(r['unresolved_candidates']), 1)
        self.assertEqual(r['full_prediction_calls'], 1)

    def test_shared_prediction_and_deadline_budgets_are_not_renewed(self):
        p = ImmutableContinuousProblem(search_fixture())
        with SessionPredictorOracle(p) as oracle:
            budget = Budget(60, 1, 10)
            e = ContinuousRoutingEvaluator(p, oracle, 'milp_bounds', budget)
            route = {'text': {'generalist':1}, 'image': {'generalist':1}}
            a = e.evaluate(route)
            # Tight inner bounds may prove completion with a single call. In
            # either case a different route must not receive a renewed budget.
            b = e.evaluate({'text': {'specialist':1}, 'image': {'specialist':1}})
            self.assertEqual(budget.counts['predictions'], 1)
            self.assertFalse(b['certified'])
            self.assertEqual(b['termination_reason'], 'PREDICTIONS_LIMIT')
            self.assertFalse(any(v['routing_shares']==b['routing_shares'] for v in e.history[:-1]))
            # Absolute expiry applies before provider construction and does
            # not become an infeasibility or optimality certificate.
            budget.deadline = time.perf_counter()-1
            before = budget.counts.copy()
            r = FixedRoutingMilpOptimizer(e).solve(e.domain.to_qps(route))
            self.assertEqual(r['termination_reason'], 'TIME_LIMIT')
            self.assertFalse(r['certified'])
            self.assertEqual(budget.counts, before)


if __name__ == '__main__':
    unittest.main()
