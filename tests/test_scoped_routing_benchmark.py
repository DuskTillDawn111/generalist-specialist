import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.benchmark_scoped_routing import make_problem, run, ScopedOracle
from gsdeploy.config import SystemConfig
from gsdeploy.continuous.activation import deployments
from gsdeploy.continuous.contracts import Budget


class ScopedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.system = SystemConfig.load(Path(__file__).resolve().parents[1]/'configs/paper_fifo_reference_v2_cpu_fixture.json').as_dict()
        self.system['catalog']['utility_normalization']['episode_protocol'] = {'scope': 'synthetic test episode'}
        self.system['catalog']['utility_normalization']['provenance'] = {'scope': 'synthetic test fixture'}
        self.evidence = dict(resident_aliases={'g#0': 'g', 's#0': 's'},
                             capacity_service_ms={'g': {'text': 2.}, 's': {'text': 4.}})
        self.templates = [dict(seed=i, execution_context_sha256='test', aliases=self.evidence['resident_aliases'],
            jobs=[dict(id='episode', duration_ms=1000., requests=[dict(task='text', offset_ms=o, pool_id=str(j), alias='s#0')
                  for j, o in enumerate((0., 500.))])]) for i in (1, 2)]

    @staticmethod
    def prediction(fit, template, routing, evidence, system):
        share = routing['text'].get('g', 0)/2.
        value = -1.-(share-.5)**2
        return dict(U=value, constraints_eligible=True, performance_unknown=False,
                    M_norm=.1, Q_contribution=0., T_contribution=-.1,
                    L_contribution=-.2, P_contribution=-.2)

    def test_inactive_models_stay_resident_and_original_inputs_unchanged(self):
        before = copy.deepcopy(self.system)
        p = make_problem(self.system, self.evidence, self.templates)
        self.assertEqual(p.states, {'g': (1,), 's': (1,)})
        self.assertEqual(list(deployments(p, {'text': {'g': 2., 's': 0.}})), [{'g': 1, 's': 1}])
        self.assertEqual(self.system, before)
        self.assertEqual(len(p.train_traces), 2)

    def test_oracle_refuses_missing_resident_before_spending_budget(self):
        p = make_problem(self.system, self.evidence, self.templates)
        oracle = ScopedOracle(p, {}, self.evidence, self.system, self.templates)
        budget = Budget()
        with self.assertRaises(ValueError):
            oracle.predict(p, {'g': 1}, {'text': {'g': 2.}}, p.train_traces[0], budget)
        self.assertEqual(budget.counts['predictions'], 0)

    def test_joint_matches_known_finite_grid_optimum_and_counts_two_histories(self):
        with patch('experiments.benchmark_scoped_routing.predict_candidate', self.prediction):
            result = run(self.system, {}, self.evidence, self.templates, 'Joint-K2', 11,
                         predictions=10, routes=5, seconds=10.)
        self.assertAlmostEqual(result['winner']['score'], -1.)
        self.assertEqual(result['winner']['deployment'], {'g': 1, 's': 1})
        self.assertEqual(result['budget']['full_prediction_calls'], 6)
        self.assertTrue(result['raw']['optimality_certified'])

    def test_shared_budget_no_extra_predictions_for_either_method(self):
        for method in ('Joint-K2', 'Activation-BO-MADS'):
            with self.subTest(method=method), patch('experiments.benchmark_scoped_routing.predict_candidate', self.prediction):
                result = run(self.system, {}, self.evidence, self.templates, method, 11,
                             predictions=4, routes=2, seconds=10.)
                self.assertLessEqual(result['budget']['full_prediction_calls'], 4)
                self.assertLessEqual(result['budget']['outer_evaluations'], 2)
                self.assertFalse(result['physical_ranking_qualified'])
                self.assertIsNotNone(result['winner'])
                self.assertEqual(result['winner']['deployment'], {'g': 1, 's': 1})


if __name__ == '__main__':
    unittest.main()
