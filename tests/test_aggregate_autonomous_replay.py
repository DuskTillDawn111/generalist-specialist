import copy
import unittest

from experiments.replay_activation_aggregate import predict_plan, score_prediction
from gsdeploy.forecasting.native_training import fit_native_captures
from tests.test_native_training import capture, POOL, ALIASES


class AggregateAutonomousReplayTests(unittest.TestCase):
    def setUp(self):
        self.fit = fit_native_captures([capture('s1', 'solo'), capture('s2', 'solo'),
                                       capture('o1', 'overlap'), capture('o2', 'overlap')], POOL, block_seconds=.5)
        self.fit['resident_aliases'] = dict(ALIASES)
        self.fit['episode_envelope'] = dict(scope='CPU analytic envelope', means=dict(
            startup_ms=1000., startup_board_joules=50., initial_idle_age_ms=0.,
            formal_lead_ms=10., batch_closure_ms=5., between_batch_ms=3., formal_tail_ms=7.))
        self.plan = copy.deepcopy(capture('planned', 'overlap')['plan'])
        for job in self.plan['jobs']:
            for row in job['requests']:
                row['offset_ms'] = 0.

    def test_stitches_public_jobs_from_training_envelope_and_shared_fifo(self):
        result = predict_plan(self.fit, self.plan)
        self.assertTrue(result['complete'], result['failure'])
        first, second = result['jobs']
        self.assertEqual(first['predicted_start_ns'], 1_010_000_000)
        self.assertEqual(second['predicted_start_ns'], first['predicted_drain_end_ns']+8_000_000)
        self.assertAlmostEqual(first['requests'][0]['predicted_e2e_ms'], 22.0001, places=3)
        self.assertAlmostEqual(result['predicted_episode_ms'], second['predicted_drain_end_ns']/1e6+12., places=6)
        self.assertFalse(result['observed_timestamps_used'])
        self.assertFalse(result['physical_predictor_qualified'])

    def test_fixed_window_and_same_owner_fifo_are_preserved(self):
        request = dict(alias='generalist_2#0', task='text', pool_id='t', offset_ms=0.)
        self.plan['jobs'] = [dict(id='queued', requests=[request, dict(request)], duration_ms=100.)]
        result = predict_plan(self.fit, self.plan)
        self.assertTrue(result['complete'], result['failure'])
        first, second = result['jobs'][0]['requests']
        self.assertGreater(second['predicted_e2e_ms'], first['predicted_e2e_ms']+10.)
        self.assertGreaterEqual(result['jobs'][0]['predicted_drain_end_ns'], 1_110_000_000)

    def test_changed_residency_rejected_and_unknown_cohort_not_dropped(self):
        bad = copy.deepcopy(self.plan)
        bad['aliases']['image#1'] = 'image'
        with self.assertRaises(ValueError):
            predict_plan(self.fit, bad)
        bad = copy.deepcopy(self.plan)
        bad['jobs'][0]['requests'][0]['pool_id'] = 'unseen'
        result = predict_plan(self.fit, bad)
        self.assertFalse(result['complete'])
        self.assertEqual(result['requested_requests'], 4)
        self.assertIsNone(result['predicted_episode_ms'])

    def test_partial_prediction_cannot_publish_full_cohort_aggregate_metrics(self):
        bundle = capture('partial', 'overlap')
        bundle['raw']['startup_begin_time_ns'] = 0
        requests = [dict(request_id=r['request_id'], alias=r['model'], task=r['task'],
                         pool_id=r['pool_id'], predicted_e2e_ms=22.0001)
                    for r in bundle['raw']['request_records'] if r['profile_job_id']=='0']
        prediction = dict(complete=False, jobs=[dict(id='0', status='predicted', requests=requests, design={})],
                          predicted_episode_ms=None, predicted_episode_board_joules=None)
        policy = dict(thresholds=dict(per_candidate_load_task_aggregate_mean_relative_error_max=.15,
                                     per_candidate_load_task_aggregate_p95_relative_error_max=.25))
        report = score_prediction(prediction, bundle, policy)
        self.assertEqual(report['requested'], 4)
        self.assertEqual(report['predicted'], 2)
        self.assertEqual(report['aggregate_metrics'], {})
        self.assertTrue(report['partial_prefix_metrics'])
        self.assertFalse(report['complete_cohort'])


if __name__ == '__main__':
    unittest.main()
