import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.native_queue_risk_rows_v9 import (
    features, offline_outcome, protocol, visible_projection,
)


class NativeQueueRiskRowsTests(unittest.TestCase):
    def setUp(self):
        self.trace = [dict(request_id='a',alias='s',task='text',pool_id='short',arrival_ns=0),
                      dict(request_id='b',alias='s',task='text',pool_id='long',arrival_ns=100),
                      dict(request_id='c',alias='g',task='image',pool_id='image',arrival_ns=1000)]
        self.costs = {'short':1.,'long':10.,'image':7.}
        self.plan = dict(split='training',training_only=True,execution_context_sha256='ctx',
                         risk_target=dict(horizon_ns=500,recovery_threshold_ms=500.),
                         latency_reference_ms=dict(text=.00005,image=.00005))

    def predict(self, visible=(),now=150,trace=None,cost=None):
        return features(self.trace if trace is None else trace,list(visible),now_ns=now,
            declared_at_ns=0,context='ctx',cost_provider=cost or (lambda r:self.costs[r['pool_id']]),
            scales_ns=(500,2000))

    def records(self):
        return [dict(request_id=r['request_id'],model=r['alias'],
                     execution_events=[dict(perf_ns=t,received_time_ns=t+5)],
                     response_reader_event_enqueue_time_ns=t+100)
                for r,t in zip(self.trace,(100,200,1000))]

    def test_visible_start_and_hol_keep_not_yet_received_start_pending(self):
        seen=[dict(request_id='a',alias='s',kind='START',received_ns=10)]
        actual=self.predict(seen)
        self.assertEqual(actual['queue_work_ms'],10.)
        self.assertEqual(actual['waiting_requests'],1)
        self.assertEqual(actual['hol_blocked_ms'],.00005)

    def test_complete_future_not_just_current_burst(self):
        actual=self.predict()
        self.assertEqual([e['max_offered_solo_work_ms'] for e in actual['future_work_envelopes']],[7.,7.])
        self.assertEqual(actual['queue_work_ms'],11.)

    def test_projection_future_outputs_and_worker_times_do_not_enter_features(self):
        rows=self.records();before=self.predict(visible_projection(rows,150))
        for r in rows:
            r['worker_response']={'output':'secret future quality', 'end_perf_ns':9999999}
            r['response_reader_event_enqueue_time_ns']=9999999
            for e in r['execution_events']:
                if e['received_time_ns']>150:e['perf_ns']=9999998
        self.assertEqual(before,self.predict(visible_projection(rows,150)))

    def test_cost_provider_gets_only_public_identity(self):
        def cost(row):
            self.assertEqual(set(row),{'alias','task','pool_id'})
            return self.costs[row['pool_id']]
        self.predict(cost=cost)

    def test_future_fact_and_extra_outcome_field_rejected(self):
        with self.assertRaises(ValueError):
            self.predict([dict(request_id='a',alias='s',kind='START',received_ns=151)])
        rows=copy.deepcopy(self.trace);rows[0]['generated_tokens']=3
        with self.assertRaises(ValueError):self.predict(trace=rows)

    def test_missing_future_cost_does_not_drop_request(self):
        with self.assertRaises(UnknownEvidence):
            self.predict(cost=lambda r:None if r['task']=='image' else 1.)

    def test_offline_shared_endpoint_has_no_artificial_empty_queue(self):
        outcome=offline_outcome(self.trace,self.records(),now_ns=50,observation_end_ns=1200,
                                risk_protocol=protocol(self.plan))
        self.assertEqual(outcome['recovery_ms'],.00015)
        self.assertTrue(outcome['violation'])

    def test_half_open_target_arrivals(self):
        rows=self.records();rows[2]['response_reader_event_enqueue_time_ns']=1100
        actual=offline_outcome(self.trace,rows,now_ns=500,observation_end_ns=1200,
                               risk_protocol=protocol(self.plan))
        self.assertFalse(actual['violation'])  # c arrives exactly at window end

    def test_horizon_and_missing_target_receipt_are_censored(self):
        for end,rows in ((549,self.records()),(1200,self.records())):
            rows[1]['response_reader_event_enqueue_time_ns']=None
            with self.assertRaises(UnknownEvidence):
                offline_outcome(self.trace,rows,now_ns=50,observation_end_ns=end,
                                risk_protocol=protocol(self.plan))

    def test_offered_cohort_cannot_disappear(self):
        with self.assertRaises(ValueError):
            offline_outcome(self.trace,self.records()[:-1],now_ns=50,observation_end_ns=1200,
                            risk_protocol=protocol(self.plan))


if __name__=='__main__':unittest.main()
