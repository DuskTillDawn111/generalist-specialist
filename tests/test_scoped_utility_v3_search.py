import copy
import unittest
from unittest.mock import patch

from experiments.benchmark_scoped_routing import ConservativeJoint, make_problem, run
from experiments.search_scoped_utility_v3 import CONTRACT, evaluator_v3, run_v3, same_route
from gsdeploy.continuous.contracts import Budget
from gsdeploy.continuous.predictor_adapter import global_cost_floor
from gsdeploy.continuous.safe_bounds import candidate_bound
from gsdeploy import normalization as norm
from gsdeploy.utility_tradeoffs import SCHEMA, TERMS, score_terms
from tests import test_scoped_routing_benchmark as benchmark_fixture


class ScopedV3SearchTests(unittest.TestCase):
    def setUp(self):
        fixture = benchmark_fixture.ScopedBenchmarkTests()
        fixture.setUp()
        self.system, self.evidence, self.templates = fixture.system, fixture.evidence, fixture.templates
        spec = self.system['catalog']['utility_normalization']
        spec['deadline_multiplier'] = 2
        for task, ref in spec['tasks'].items():
            ref['reference_mean_e2e_ms'] = ref['deadline_ms'] / 2

    @staticmethod
    def prediction(fit, template, routing, evidence, system):
        share = routing['text'].get('g', 0) / 2
        # v2 prefers faster g; v3 prefers higher quality s. This catches a
        # score-only reporting change while the actual search still uses v2.
        terms = dict(Q=2*share, T=1-share, L=1-share, P=1-share, M=.1, E=.1)
        scored = score_terms(terms, system['weights'])
        return dict(U=scored['U'], constraints_eligible=True, performance_unknown=False,
                    **{t+'_norm': terms[t] for t in TERMS},
                    **{t+'_contribution': scored['contributions'][t] for t in TERMS})

    def test_v3_changes_actual_search_winner_in_both_algorithms(self):
        before = copy.deepcopy(self.system)
        with patch('experiments.benchmark_scoped_routing.predict_candidate', self.prediction):
            old = run(self.system, {}, self.evidence, self.templates, 'Joint-K2', 11,
                      predictions=10, routes=5, seconds=10)
            self.assertEqual(old['winner']['routing']['text'], {'g': 2.0})
            for method in ('Joint-K2', 'Activation-BO-MADS'):
                result = run_v3(self.system, {}, self.evidence, self.templates, method, 11,
                                predictions=16, routes=8, seconds=10)
                w = result['winner']
                route = w.get('routing', w.get('routing_qps'))
                self.assertTrue(same_route(route, {'text': {'s': 2.0}}))
                self.assertAlmostEqual(w.get('score', w.get('best_feasible_utility')), -.3)
                self.assertLessEqual(result['budget']['full_prediction_calls'], 16)
                self.assertLessEqual(result['budget']['outer_evaluations'], 8)
                self.assertEqual(result['utility_contract']['utility_schema'], SCHEMA)
        self.assertEqual(self.system, before)

    def test_bounds_use_new_weight_and_remain_valid_for_every_grid_point(self):
        e = evaluator_v3(self.system, {}, self.evidence, self.templates, Budget())
        c = e.problem.config
        opt = ConservativeJoint(c.catalog, c.workload, c.gpu_memory_mib, c.latency_reference_ms,
                                c.weights, horizon=c.horizon_seconds)
        expected = .25*norm.quality_floor(c.catalog, c.workload)
        self.assertAlmostEqual(global_cost_floor(e.oracle, e.problem), expected)
        self.assertAlmostEqual(opt.lower, expected)
        self.assertEqual(opt.state_memory('g', 1), 0)
        deployment = {'g': 1, 's': 1}
        with patch('experiments.benchmark_scoped_routing.predict_candidate', self.prediction):
            for share in (0., .5, 1.):
                route = {'text': {'g': 2*share, 's': 2*(1-share)}}
                cost_floor = opt.floor('text', deployment, route['text'])
                value = e.candidate(deployment, route)['utility']
                self.assertGreaterEqual(-value + 1e-12, cost_floor)
                self.assertLessEqual(value, candidate_bound(e, deployment, route)['upper'])

    def test_objective_and_cache_identity_are_distinct_and_metadata_explicit(self):
        old = make_problem(self.system, self.evidence, self.templates)
        e = evaluator_v3(self.system, {}, self.evidence, self.templates, Budget())
        self.assertNotEqual(old.objective_id, e.problem.objective_id)
        with patch('experiments.benchmark_scoped_routing.predict_candidate', self.prediction):
            p = e.candidate({'g': 1, 's': 1}, {'text': {'s': 2.}})['traces'][0]['prediction']
        self.assertEqual(p['utility_schema'], SCHEMA)
        self.assertEqual(p['normalization_schema'], 'six_term_reference_v2')
        self.assertEqual(p['utility_contract'], CONTRACT)
        e.oracle.system['weights']['Q'] = .1
        with self.assertRaises(ValueError):
            e.check()

    def test_changed_slo_cannot_silently_reuse_old_scales(self):
        bad = copy.deepcopy(self.system)
        bad['catalog']['utility_normalization']['deadline_multiplier'] = 4
        with self.assertRaises(ValueError):
            evaluator_v3(bad, {}, self.evidence, self.templates, Budget())

    def test_incomplete_prediction_stays_unknown(self):
        e = evaluator_v3(self.system, {}, self.evidence, self.templates, Budget())
        with patch('experiments.benchmark_scoped_routing.predict_candidate', return_value=dict(
                U=None, constraints_eligible=None, performance_unknown=True)):
            result = e.candidate({'g': 1, 's': 1}, {'text': {'s': 2.}})
        self.assertEqual(result['constraint_state'], 'UNKNOWN')
        self.assertIsNone(result['utility'])

    def test_mixed_version_scores_rejected(self):
        e = evaluator_v3(self.system, {}, self.evidence, self.templates, Budget())
        def bad(*args):
            out = self.prediction(*args)
            out['U'] -= .1
            return out
        with patch('experiments.benchmark_scoped_routing.predict_candidate', bad), self.assertRaises(ValueError):
            e.candidate({'g': 1, 's': 1}, {'text': {'s': 2.}})


if __name__ == '__main__':
    unittest.main()
