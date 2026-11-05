import copy
import unittest

from experiments.prepare_activation_aggregate_validation import make_jobs, TASKS
from experiments.continue_activation_aggregate_validation import assess_cells


class AggregateValidationDesignTests(unittest.TestCase):
    def setUp(self):
        self.template = dict(profiles=[dict(family='generalist_2', task=t, pool_id=f'{t}:{i}',
             segments=[dict(duration_ms={'mean': 1000.+i})]) for t in TASKS for i in range(32)])

    def test_candidate_cohorts_match_and_route_shares_are_exact(self):
        jobs = make_jobs(self.template, 17)
        self.assertEqual(len({j['shared_offered_cohort_sha256'] for j in jobs}), 1)
        for job in jobs:
            self.assertEqual(len(job['requests']), 96)
            for task in TASKS:
                rows = [r for r in job['requests'] if r['task']==task]
                self.assertEqual(len({r['pool_id'] for r in rows}), 24)
                self.assertEqual(sum(r['alias']=='generalist_2#0' for r in rows),
                                 round(24*job['design']['generalist_share']))
            self.assertTrue(all(0 <= r['offset_ms'] < job['duration_ms'] for r in job['requests']))

    def test_burst_regime_preserves_window_and_mean_load(self):
        normal = make_jobs(self.template, 17)
        burst = make_jobs(self.template, 18, burst_pairs=True, reverse_routes=True)
        self.assertEqual([j['design']['generalist_share'] for j in burst], [1., .5, 0.])
        self.assertEqual(normal[0]['duration_ms'], burst[0]['duration_ms'])
        times = sorted(r['offset_ms'] for r in burst[0]['requests'] if r['task']=='ocr')
        self.assertLess(times[1]-times[0], (times[2]-times[1])/20)
        with self.assertRaises(ValueError):
            make_jobs(self.template, 17, nominal_load=.9)

    def test_one_failed_task_cannot_be_hidden_by_other_cells(self):
        jobs = make_jobs(self.template, 17)
        metrics = {j['id']+'/task:'+t: dict(n=24, descriptive_within_latency_tolerances=True)
                   for j in jobs for t in TASKS}
        report = dict(complete_cohort=True, aggregate_metrics=metrics,
            throughput=[dict(job_id=j['id'], relative_error=0.) for j in jobs],
            episode=dict(relative_energy_error=.01))
        policy = dict(thresholds=dict(fixed_window_throughput_relative_error_max=.1,
                                     full_gpu_board_episode_energy_relative_error_max=.15))
        self.assertEqual(assess_cells(report, dict(jobs=jobs), policy), [])
        bad = copy.deepcopy(report)
        key = jobs[0]['id']+'/task:ocr'
        bad['aggregate_metrics'][key]['descriptive_within_latency_tolerances'] = False
        self.assertEqual(assess_cells(bad, dict(jobs=jobs), policy), [key+':LATENCY_TOLERANCE'])


if __name__ == '__main__':
    unittest.main()
