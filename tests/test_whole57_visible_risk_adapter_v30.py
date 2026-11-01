import copy
import unittest

from tools.predictor_validation_20261027.whole57_visible_risk_adapter_v30 import (
    OfflineQueueLabels, public_registry, snapshot_at)


def fixture():
    plan=dict(inputs=[dict(task='text',pool_id='input',input_size=123,exact_input_sha256='inputsha')],
        trace=[dict(request_id='a',task='text',pool_id='input',offset_ns=0),
               dict(request_id='b',task='text',pool_id='input',offset_ns=10)],
        routes={'a':'text#0','b':'text#0'},resource_configuration={'allocations':{'text#0':{}}})
    records=[dict(request_id='a',model='text#0',execution_events=[
        dict(request_id='a',perf_ns=2,received_time_ns=20),
        dict(request_id='a',perf_ns=3,received_time_ns=21,event='start',kind='gpu',segment_id='prefill',semantic_phase='prefill',input_sequence_length=9),
        dict(request_id='a',perf_ns=30,received_time_ns=31,event='end',kind='gpu',segment_id='prefill')],
        worker_response=dict(end_perf_ns=40),response_reader_event_enqueue_time_ns=50,
        previous_handler_next_request_ready=dict(request_id='a',perf_ns=41,received_time_ns=51,clock='host_perf_counter_ns',boundary='previous_handler_next_request_ready',scope='after_response_flush_before_next_read')),
        dict(request_id='b',model='text#0',execution_events=[dict(request_id='b',perf_ns=42,received_time_ns=52)],
        worker_response=dict(end_perf_ns=60),response_reader_event_enqueue_time_ns=70)]
    return plan,records


class Whole57VisibleRiskAdapterTests(unittest.TestCase):
    def test_late_START_uses_reader_receipt_not_earlier_worker_time(self):
        plan,records=fixture()
        early=snapshot_at(plan,records,0,'ctx',15)
        self.assertEqual(early['waiting_request_ids'],['a','b'])
        self.assertEqual(early['oldest_waiting_age_ns'],15)
        late=snapshot_at(plan,records,0,'ctx',20)
        self.assertEqual(late['waiting_request_ids'],['b'])
        self.assertEqual(late['per_owner']['text#0']['handler_active_request_ids'],['a'])
        self.assertIsNone(late['GPUcursor'])

    def test_future_label_and_phase_metadata_changes_cannot_change_prefix(self):
        plan,records=fixture();changed=copy.deepcopy(records)
        changed[0]['worker_response']['end_perf_ns']=90
        changed[0]['response_reader_event_enqueue_time_ns']=100
        changed[0]['previous_handler_next_request_ready']['perf_ns']=91
        changed[0]['previous_handler_next_request_ready']['received_time_ns']=101
        changed[0]['execution_events'][1]['input_sequence_length']=99999
        changed[0]['execution_events'][2]['perf_ns']=80
        changed[0]['execution_events'][2]['received_time_ns']=81
        self.assertEqual(snapshot_at(plan,records,0,'ctx',20),snapshot_at(plan,changed,0,'ctx',20))

    def test_future_public_arrivals_and_input_are_preserved_without_tokens(self):
        plan,records=fixture();features=snapshot_at(plan,records,0,'ctx',5)
        self.assertEqual(features['future_declared_arrivals'][0]['request_id'],'b')
        self.assertEqual(features['future_declared_arrivals'][0]['input_features']['input_size'],123)
        self.assertNotIn('input_sequence_length',str(features))
        after=snapshot_at(plan,records,0,'ctx',21)
        self.assertEqual(after['observed_requests'][0]['active_phase_segments'][0]['input_sequence_length'],9)

    def test_label_recovery_keeps_future_arrival_and_atomic_ties(self):
        plan,records=fixture();registry=public_registry(plan,0)
        labels=OfflineQueueLabels(registry,records,80)
        outcome=labels.outcome(5,60)
        self.assertTrue(outcome['recovery']['event_observed'])
        self.assertEqual(outcome['recovery']['elapsed_or_censor_ns'],47)
        self.assertEqual(outcome['observed_waiting_peak'],2)
        self.assertAlmostEqual(outcome['exact_tail_p95_ms'],59.5/1e6)

    def test_arrival_at_another_START_does_not_create_false_empty_recovery(self):
        plan,records=fixture();plan['trace'][1]['offset_ns']=20
        labels=OfflineQueueLabels(public_registry(plan,0),records,80)
        outcome=labels.outcome(5,60)
        self.assertEqual(outcome['recovery']['elapsed_or_censor_ns'],47)
        self.assertEqual(outcome['observed_waiting_peak'],1)

    def test_future_label_change_changes_label_but_preserves_same_prefix(self):
        plan,records=fixture();changed=copy.deepcopy(records)
        changed[1]['worker_response']['end_perf_ns']=90
        changed[1]['response_reader_event_enqueue_time_ns']=100
        registry=public_registry(plan,0)
        self.assertEqual(snapshot_at(plan,records,0,'ctx',15),snapshot_at(plan,changed,0,'ctx',15))
        before=OfflineQueueLabels(registry,records,120).outcome(15,70)
        after=OfflineQueueLabels(registry,changed,120).outcome(15,70)
        self.assertNotEqual(before['exact_tail_p95_ms'],after['exact_tail_p95_ms'])

    def test_truncated_labels_are_censored_never_negative_examples(self):
        plan,records=fixture();labels=OfflineQueueLabels(public_registry(plan,0),records,25)
        outcome=labels.outcome(15,30)
        self.assertFalse(outcome['full_horizon_observed'])
        self.assertFalse(outcome['recovery']['event_observed'])
        self.assertEqual(outcome['recovery']['elapsed_or_censor_ns'],10)
        self.assertIsNone(outcome['exact_tail_p95_ms'])
        self.assertTrue(all(not row['event_observed'] for row in outcome['tail_cohort']))
        self.assertIsNone(outcome['latency_violation'])

    def test_READY_is_not_fabricated_from_result(self):
        plan,records=fixture()
        features=snapshot_at(plan,records,0,'ctx',50)
        self.assertEqual(features['per_owner']['text#0']['awaiting_next_ready_request_ids'],['a'])
        self.assertEqual(snapshot_at(plan,records,0,'ctx',51)['per_owner']['text#0']['awaiting_next_ready_request_ids'],[])

    def test_future_pipe_write_return_is_hidden_while_dispatch_elapsed_is_visible(self):
        plan,records=fixture()
        records[0].update(enqueued_time_ns=0,actual_submit_time_ns=1,submit_write_completed_time_ns=100)
        before=snapshot_at(plan,records,0,'ctx',15)
        row=before['observed_requests'][0]
        self.assertEqual(row['host_pipeline_status'],'DISPATCHED_PIPE_WRITE_NOT_OBSERVED_COMPLETED')
        self.assertEqual(row['elapsed_since_dispatch_without_write_completion_ns'],14)
        self.assertNotIn('submit_write_completed_time_ns',row['visible_host_boundaries'])
        changed=copy.deepcopy(records);changed[0]['submit_write_completed_time_ns']=1000
        self.assertEqual(before,snapshot_at(plan,changed,0,'ctx',15))


if __name__=='__main__':unittest.main()
