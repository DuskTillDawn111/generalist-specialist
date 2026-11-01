import copy
import unittest

from tools.predictor_validation_20261027.duration_selection import evaluate


class TrainingSelectionScoreTests(unittest.TestCase):
    def setUp(self):
        self.units = {'seeds': [1], 'routes': ['base', 'spill']}
        self.contracts = {1: {'baseline': 'base', 'tie_tolerance': 0., 'tasks': ['text']}}
        self.limits = {'service_mare_max': .2, 'e2e_mare_max': .2,
                       'queue_depth_mean_absolute_error_max_requests': 1.,
                       'each_task_p95_absolute_relative_error_max': .2}
        self.report = {'candidates': []}
        for cid, value in (('base', -2.), ('spill', -1.)):
            self.report['candidates'].append({'seed': 1, 'candidate_id': cid,
                'predicted_comparison_U': value, 'observed_U': value,
                'timing_errors': {k: {'mean_absolute_relative_error': .1} for k in ('service', 'e2e')},
                'queue_error': {'mae': .5},
                'per_task_p95': {'text': {'absolute_relative_error': .1}}})

    def evaluate(self, report=None):
        return evaluate(report or self.report, self.units, self.limits, self.contracts)

    def test_all_declared_metrics_and_pairs_counted(self):
        score = self.evaluate()
        self.assertEqual(len(score['metrics']), 8)
        self.assertEqual(score['decision_errors'], 0)
        self.assertEqual(score['temporal_failed_or_unknown'], 0)
        self.assertEqual(score['normalized_temporal_error_mean'], .5)
        self.assertTrue(score['original_training_cv_thresholds_passed'])

    def test_missing_utility_counts_sign_and_rank_as_errors(self):
        report = copy.deepcopy(self.report)
        report['candidates'][1]['predicted_comparison_U'] = None
        self.assertEqual(self.evaluate(report)['decision_errors'], 2)

    def test_unknown_metric_has_fixed_denominator_and_infinite_third_score(self):
        report = copy.deepcopy(self.report)
        report['candidates'][0]['queue_error']['mae'] = None
        score = self.evaluate(report)
        self.assertEqual(len(score['metrics']), 8)
        self.assertEqual(score['temporal_failed_or_unknown'], 1)
        self.assertIsNone(score['normalized_temporal_error_mean'])

    def test_omitted_candidate_cannot_shrink_score(self):
        report = copy.deepcopy(self.report); report['candidates'].pop()
        with self.assertRaises(ValueError):
            self.evaluate(report)

    def test_nonfinite_metrics_remain_unknown(self):
        report = copy.deepcopy(self.report)
        report['candidates'][0]['per_task_p95']['text']['absolute_relative_error'] = float('nan')
        self.assertEqual(self.evaluate(report)['temporal_failed_or_unknown'], 1)


if __name__ == '__main__':
    unittest.main()
