"""CPU regression for complete population, full horizon and censored errors."""
import copy
from fractions import Fraction as F
import unittest
from tools.predictor_validation_20261027 import twoFIFO_fresh181_validation_metrics_v1 as m
from tools.predictor_validation_20261027.actual_worker_twoFIFO_reference_v5_2 import decode
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import encode


def population():
    requests=[dict(request_id=str(i),owner='x',task='text' if i<167 else ['image','asr','ocr'][i%3],arrival_ns=0) for i in range(181)]
    rows=[dict(request_id=r['request_id'],status='completed',worker_response=dict(success=True,end_perf_ns=90),
        enqueued_time_ns=0,actual_submit_time_ns=0,observed_worker_start_time_ns=1,
        previous_handler_next_request_ready=dict(perf_ns=91),completion_time_ns=100,response_reader_event_enqueue_time_ns=100) for r in requests]
    return requests,dict(request_records=rows)


def state(at,ids,phase):
    running=ids if phase=='running' else [];released=ids if phase in ('released','finished') else []
    waiting=ids if phase=='waiting' else []
    return dict(at_ns=at,lifecycle_count=dict(arrived_not_submitted=0,submitted_waiting_START=len(waiting),running_START_to_READY=len(running)),
        parent_sender_FIFO_heads=[],arrived_RESULT_unfinished_request_ids=[] if phase=='finished' else ids,
        worker_lifecycle=dict(running_START_to_READY=running,worker_released=released))


class ValidationMetricsTests(unittest.TestCase):
    def test_censored_all181_P95_bound_and_identifiable_small_missing_tail(self):
        requests,raw=population();raw['request_records'][-1].update(status='incomplete',worker_response=None,completion_time_ns=None)
        labels=m.observed_labels(raw,requests,210)
        self.assertEqual(labels['overall']['offered'],181);self.assertEqual(labels['overall']['successful_within_drain'],180)
        self.assertEqual(labels['overall']['P95_ns'],100);self.assertTrue(labels['overall']['P95_identifiable'])
        for row in raw['request_records'][-10:]:row.update(status='incomplete',worker_response=None,completion_time_ns=None)
        labels=m.observed_labels(raw,requests,210)
        self.assertIsNone(labels['overall']['P95_ns']);self.assertIsNone(labels['overall']['P95_upper_ns'])
        self.assertEqual(labels['overall']['P95_lower_ns'],210)
        self.assertEqual(len(labels['requests']),181)

    def test_RESULT_reader_does_not_release_missing_READY_and_end_perf_is_integration_cut(self):
        requests=[dict(request_id='r',arrival_ns=0)];row=dict(enqueued_time_ns=0,actual_submit_time_ns=0,
            observed_worker_start_time_ns=2,worker_response=dict(end_perf_ns=6,success=False),response_reader_event_enqueue_time_ns=8)
        raw=dict(request_records=[row]);timeline=m.actual_timeline(raw,requests,0,10,[])
        self.assertIn(6,[s['at_ns'] for s in timeline])
        self.assertEqual(m.actual_snapshot(raw,requests,9)['lower']['arrived_RESULT_unfinished'],0)
        self.assertEqual(m.actual_snapshot(raw,requests,9)['upper']['running_START_to_READY'],1)
        diag=dict(event_slices=[dict(state=state(0,[],'finished'))],cutoff_state=state(10,[],'finished'))
        result=m.queue_errors(timeline,diag,0,10)
        self.assertEqual(result['time_mean_absolute_error_bounds_requests']['running_START_to_READY'],[F(4,10),F(8,10)])
        self.assertTrue(result['RESULT_and_READY_not_interchanged'])

    def test_censored_signed_and_absolute_errors_keep_infinite_side(self):
        self.assertEqual(m.signed_error_bounds(20,10,None),dict(signed_lower=None,signed_upper=10,absolute_lower=0,absolute_upper=None))
        self.assertEqual(m.signed_error_bounds(5,10,15),dict(signed_lower=-10,signed_upper=-5,absolute_lower=5,absolute_upper=10))

    def test_missing_submit_has_possible_sender_to_START_queue_mass(self):
        result=m.actual_snapshot(dict(request_records=[dict(enqueued_time_ns=1)]),[dict(request_id='r',arrival_ns=0)],2)
        self.assertEqual(result['lower']['submitted_waiting_START'],0)
        self.assertEqual(result['upper']['submitted_waiting_START'],1)

    def report(self,requests,*,unknown=False):
        ids=[r['request_id'] for r in requests];draws=[];projections=[]
        for i in range(12):
            status='UNKNOWN_WORLD' if unknown else 'GENERATED_WORLD'
            draw=dict(draw_id=str(i),weight=F(1,12),status=status)
            if not unknown:
                draw.update(queue_diagnostics=dict(event_slices=[dict(state=state(t,ids,p)) for t,p in [(0,'waiting'),(1,'running'),(91,'released'),(100,'finished')]],
                    cutoff_state=state(210,ids,'finished'),reader_E2E_P95_ns=100),scoring=dict(original_score=dict(terms={k:None for k in 'QTLPME'},full_U=None)))
            draws.append(draw);projections.append(dict(draw_id=str(i),weight=F(1,12),status=status,
                requests=[dict(request_id=rid,completion_time_ns=None if unknown else 100) for rid in ids]))
        forecast=dict(ensemble=dict(draws=draws,public_requests=requests,world_unknown_probability_mass=F(1) if unknown else F(0)),complete_draw_projection=projections)
        return dict(replay=dict(forecast=encode(forecast)))

    def test_all12_unknown_worlds_and_all181_keep_mass_and_no_point_fallback(self):
        requests,raw=population();labels=m.observed_labels(raw,requests,210);timeline=m.actual_timeline(raw,requests,0,210,[])
        result=decode(m.prediction_errors(self.report(requests,unknown=True),labels,timeline,requests,0,210))
        self.assertEqual(len(result['requests']),181);self.assertEqual(len(result['worlds']),12)
        self.assertEqual(result['world_unknown_probability_mass'],1);self.assertEqual(result['point_error_request_coverage'],0)
        self.assertIsNone(result['point_reader_MARE']);self.assertFalse(result['unknown_worlds_renormalized'])
        self.assertTrue(all(r['remaining_probability_mass']==1 for r in result['requests']))

    def test_full12world_weighted_point_service_and_per_task_P95_errors(self):
        requests,raw=population();labels=m.observed_labels(raw,requests,210);timeline=m.actual_timeline(raw,requests,0,210,[])
        result=decode(m.prediction_errors(self.report(requests),labels,timeline,requests,0,210))
        self.assertEqual(result['point_error_request_coverage'],1)
        self.assertEqual(result['point_reader_MARE'],0);self.assertEqual(result['point_START_to_READY_MARE'],0)
        self.assertTrue(all(t['absolute_relative_error']==0 for t in result['per_task_P95'].values()))
        for row in raw['request_records'][-10:]:row.update(status='incomplete',worker_response=None,completion_time_ns=None)
        labels=m.observed_labels(raw,requests,210)
        result=decode(m.prediction_errors(self.report(requests),labels,m.actual_timeline(raw,requests,0,210,[]),requests,0,210))
        bound=result['worlds'][0]['P95_signed_error_bounds_ns']
        self.assertIsNone(bound['signed_lower']);self.assertEqual(bound['signed_upper'],-110)
        self.assertEqual(result['requests'][-1]['signed_error_bounds_ns']['signed_upper'],-110)

if __name__=='__main__':unittest.main()
