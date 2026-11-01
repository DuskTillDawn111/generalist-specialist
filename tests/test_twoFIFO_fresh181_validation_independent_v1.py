"""Independent quantile, mixed-mass and full-denominator decision checks."""
import unittest
from fractions import Fraction as F

from tests import test_twoFIFO_fresh181_validation_metrics_v1 as fixture
from tools.predictor_validation_20261027 import twoFIFO_fresh181_validation_metrics_v1 as metrics
from tools.predictor_validation_20261027 import twoFIFO_fresh181_validation_evaluation_v1 as evaluation
from tools.predictor_validation_20261027.actual_worker_twoFIFO_reference_v5_2 import decode
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import encode


class IndependentValidationMathTests(unittest.TestCase):
    def test_expected_world_P95_and_point_error_are_not_pooled_world_statistics(self):
        requests,raw=fixture.population()
        for row in raw['request_records']:
            row.update(completion_time_ns=200,response_reader_event_enqueue_time_ns=200,
                previous_handler_next_request_ready=dict(perf_ns=191),worker_response=dict(success=True,end_perf_ns=190))
        report=fixture.ValidationMetricsTests().report(requests)
        forecast=decode(report['replay']['forecast'])
        for index,(draw,projection) in enumerate(zip(forecast['ensemble']['draws'],forecast['complete_draw_projection'])):
            finish=100 if index<6 else 300
            draw['queue_diagnostics']['reader_E2E_P95_ns']=finish
            draw['queue_diagnostics']['event_slices'][2]['state']['at_ns']=finish-9
            draw['queue_diagnostics']['event_slices'][3]['state']['at_ns']=finish
            draw['queue_diagnostics']['cutoff_state']['at_ns']=500
            for row in projection['requests']:row['completion_time_ns']=finish
        report['replay']['forecast']=encode(forecast)
        observed=metrics.observed_labels(raw,requests,500)
        timeline=metrics.actual_timeline(raw,requests,0,500,[])
        result=decode(metrics.prediction_errors(report,observed,timeline,requests,0,500))
        self.assertEqual(result['point_reader_MARE'],0)
        self.assertTrue(all(row['predicted_mean_reader_E2E_ns']==200 for row in result['requests']))
        self.assertTrue(all(row['expected_draw_P95_ns']==200 for row in result['per_task_P95'].values()))
        pooled=[row['completion_time_ns'] for world in forecast['complete_draw_projection'] for row in world['requests']]
        self.assertEqual(metrics.percentile(pooled),300)
        self.assertTrue(all(abs(world['P95_error_ns'])==100 for world in result['worlds']))

    def test_one_unknown_world_prevents_point_estimates_without_dropping_eleven_known_worlds(self):
        requests,raw=fixture.population();report=fixture.ValidationMetricsTests().report(requests)
        forecast=decode(report['replay']['forecast'])
        forecast['ensemble']['draws'][-1]['status']='UNKNOWN_WORLD'
        forecast['ensemble']['world_unknown_probability_mass']=F(1,12)
        projection=forecast['complete_draw_projection'][-1];projection['status']='UNKNOWN_WORLD'
        for row in projection['requests']:row['completion_time_ns']=None
        report['replay']['forecast']=encode(forecast)
        observed=metrics.observed_labels(raw,requests,210)
        result=decode(metrics.prediction_errors(report,observed,metrics.actual_timeline(raw,requests,0,210,[]),
            requests,0,210,sample_times=[100]))
        self.assertEqual(len(result['worlds']),12)
        self.assertEqual(result['world_unknown_probability_mass'],F(1,12))
        for row in result['requests']:
            self.assertEqual(row['known_complete_probability_mass'],F(11,12))
            self.assertIsNone(row['predicted_mean_reader_E2E_ns'])
        self.assertEqual(result['point_error_request_coverage'],0)
        self.assertIsNone(result['predeclared_queue_sample_grid'][0]['queues']['arrived_RESULT_unfinished']['predicted_mean'])

    def test_failed_requirement_and_irrecoverable_pair_error_survive_missing_evidence(self):
        self.assertFalse(evaluation.tri_state_all([None,True,False]))
        self.assertIsNone(evaluation.tri_state_all([None,True]))
        self.assertFalse(evaluation.accuracy_threshold(0,1,6,1.))
        self.assertIsNone(evaluation.accuracy_threshold(1,1,6,.8))
        self.assertFalse(evaluation.accuracy_threshold(4,6,6,.8))


if __name__=='__main__':unittest.main()
