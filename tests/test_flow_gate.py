import copy
import unittest
from tools.predictor_validation_20261027.flow_gate import assess


class FlowGateTests(unittest.TestCase):
    def fixture(self):
        limits=dict(complete_request_coverage=1.,complete_candidate_coverage=1.,service_mare_max=.2,
            e2e_mare_max=.2,each_task_p95_absolute_relative_error_max=.2,
            queue_depth_mean_absolute_error_max_requests=1.,paired_delta_sign_accuracy_min=1.,
            candidate_pair_rank_accuracy_min=1.,scope='CPU fixture')
        plan=dict(validation_thresholds=limits,candidates=[{'id':'a'},{'id':'b'}],trace=[{},{}],workload_rps={'text':1.})
        rows=[dict(candidate_id=c,known=2,timing_errors={k:{'mean_absolute_relative_error':.1} for k in ('service','e2e')},
            queue_error={'mae':.5},prediction_metrics={'per_task':{'text':{'p95_ms':100.}}},
            observed_metrics={'per_task':{'text':{'p95_ms':100.}}}) for c in ('a','b')]
        report=dict(candidate_coverage={'observed':2},candidates=rows,paired_delta_coverage={'requested':1,'known':1},
            rank_coverage={'requested':1,'known':1},delta_sign_accuracy=1.,rank_accuracy=1.)
        return plan,report

    def test_pass_does_not_enable_final_confirmation(self):
        p,r=self.fixture();g=assess(p,r)
        self.assertEqual(g['status'],'PASS');self.assertFalse(g['final_four_groups_enabled'])

    def test_one_task_tail_failure_cannot_hide_in_mean(self):
        p,r=self.fixture();r['candidates'][0]['prediction_metrics']['per_task']['text']['p95_ms']=125.
        g=assess(p,r);self.assertEqual(g['status'],'FAIL');self.assertIn('a:text_p95_are',g['failed_checks'])

    def test_missing_predictions_and_wrong_rank_not_pass(self):
        p,r=self.fixture();r['candidates'][0]['timing_errors']['e2e']['mean_absolute_relative_error']=None
        r['rank_accuracy']=0.
        g=assess(p,r);self.assertEqual(g['status'],'INCOMPLETE');self.assertIn('rank_accuracy',g['failed_checks'])
        self.assertIn('a:e2e_mare',g['unknown_checks'])

    def test_nan_cannot_be_known_evidence(self):
        p,r=self.fixture();r['delta_sign_accuracy']=float('nan')
        self.assertEqual(assess(p,r)['status'],'INCOMPLETE')
