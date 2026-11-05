import copy
import unittest

from experiments.continue_activation_aggregate_validation import assess_cells
from experiments.decide_activation_scoped_release import decide


class ScopedReleaseTests(unittest.TestCase):
    def setUp(self):
        self.policy = dict(thresholds=dict(fixed_window_throughput_relative_error_max=.1,
                                          full_gpu_board_episode_energy_relative_error_max=.15))
        self.plans = [dict(run_id=f'round{i}', jobs=[dict(id=f'route:{s}', design=dict(
            generalist_share=s, arrival_regime=regime, offered_per_task=24)) for s in (0., .5, 1.)])
            for i, regime in enumerate(('stratified_jitter', 'paired_bursts'))]
        self.validation = dict(reports=[dict(run_id=p['run_id'], complete_cohort=True,
            aggregate_metrics={j['id']+'/task:'+t: dict(n=24, descriptive_within_latency_tolerances=True)
                               for j in p['jobs'] for t in ('asr', 'image', 'ocr', 'text')},
            throughput=[dict(job_id=j['id'], relative_error=.01) for j in p['jobs']],
            episode=dict(relative_energy_error=.02)) for p in self.plans],
            execution_integrity_passed=True, physical_predictor_qualified=False,
            candidate_ranking_qualified=False)
        self.refresh()

    def refresh(self):
        reasons = []
        for report, plan in zip(self.validation['reports'], self.plans):
            report['tolerance_reasons'] = assess_cells(report, plan, self.policy)
            reasons.extend(report['run_id']+':'+r for r in report['tolerance_reasons'])
        self.validation['reasons'] = reasons
        self.validation['known_input_history_accuracy_within_tolerances'] = not reasons

    def test_general_qualification_false_does_not_block_scoped_acceptance(self):
        result = decide(self.validation, self.plans, self.policy)
        self.assertEqual(result['status'], 'SCOPED_PASS')
        self.assertTrue(result['scoped_prediction_accepted'])
        self.assertFalse(result['candidate_ranking_qualified'])
        self.assertEqual(result['automatic_extra_gpu_runs'], 0)

    def test_one_task_failure_excludes_route_across_both_histories(self):
        self.validation['reports'][1]['aggregate_metrics']['route:0.5/task:ocr'][
            'descriptive_within_latency_tolerances'] = False
        self.refresh()
        result = decide(self.validation, self.plans, self.policy)
        self.assertEqual(result['status'], 'SCOPED_RESTRICTED')
        self.assertEqual(result['tested_route_shares_eligible_in_both_histories'], [0., 1.])
        self.assertEqual(len(result['validation_reasons']), 1)

    def test_whole_episode_energy_failure_cannot_be_assigned_to_one_route(self):
        self.validation['reports'][1]['episode']['relative_energy_error'] = .16
        self.refresh()
        result = decide(self.validation, self.plans, self.policy)
        self.assertEqual(result['status'], 'DECISION_IMPACT_REVIEW')
        self.assertEqual(result['tested_route_shares_eligible_in_both_histories'], [])
        self.assertEqual(result['automatic_refits'], 0)

    def test_capture_integrity_failure_prevents_release(self):
        self.validation.update(execution_integrity_passed=False,
            known_input_history_accuracy_within_tolerances=False, reasons=['round0:BAD_OUTPUT'])
        result = decide(self.validation, self.plans, self.policy)
        self.assertEqual(result['status'], 'CAPTURE_INTEGRITY_REVIEW')
        self.assertFalse(result['scoped_prediction_accepted'])

    def test_missing_duplicate_or_inconsistent_reports_rejected(self):
        for change in ('missing', 'duplicate', 'flag', 'reason'):
            value = copy.deepcopy(self.validation)
            if change == 'missing':
                value['reports'].pop()
            elif change == 'duplicate':
                value['reports'][1] = copy.deepcopy(value['reports'][0])
            elif change == 'flag':
                value['known_input_history_accuracy_within_tolerances'] = False
            else:
                value['reasons'] = ['invented']
            with self.subTest(change=change), self.assertRaises(ValueError):
                decide(value, self.plans, self.policy)


if __name__ == '__main__':
    unittest.main()
