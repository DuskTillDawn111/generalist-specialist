"""Independent causal-prefix and identity review; no CUDA execution."""
import copy
import unittest

from tests.test_visible_device_event_ledger_v1 import fixture
from tools.predictor_validation_20261027 import visible_device_event_ledger_v1 as ledger


class FutureBody(dict):
    def get(self,key,*default):
        if key!='received_time_ns':raise AssertionError('future body read: '+key)
        return super().get(key,*default)
    def __deepcopy__(self,memo):raise AssertionError('future body copied')


class FutureResult(dict):
    def get(self,key,*default):raise AssertionError('future RESULT body read')


class IndependentLedgerReview(unittest.TestCase):
    def state(self):return ledger.new_state({'a':{},'b':{}},'context')
    def completed(self,rows,*,sequence=2,received=40):
        query=copy.deepcopy(rows[0]['execution_events'][-1])
        query.update(observation_sequence=sequence,host_query_begin_ns=31,host_query_end_ns=32,perf_ns=32,received_time_ns=received,end_event_complete=True,recorded_stream_elapsed_ms=1.)
        rows[0]['execution_events'].append(query)
    def test_future_event_RESULT_and_ready_bodies_not_touched(self):
        rows=fixture();rows[0]['execution_events'].append(FutureBody(received_time_ns=100))
        rows[0]['response_reader_event_enqueue_time_ns']=100;rows[0]['worker_response']=FutureResult()
        rows[0]['previous_handler_next_request_ready']=FutureBody(received_time_ns=100)
        result=ledger.advance(self.state(),rows,30)
        self.assertEqual(len(result['visible_device_events']),1)
        self.assertIsNone(result['host']['requests']['r']['handler_end_ns'])
    def test_query_is_invisible_until_its_own_parent_receipt(self):
        rows=fixture();before=ledger.advance(self.state(),rows,29)
        self.assertEqual(before['visible_device_events'],[])
        visible=ledger.advance(before,rows,30)
        self.assertEqual(visible['visible_device_events'][0]['received_time_ns'],30)
    def test_late_forward_boundary_cannot_support_earlier_query_cutoff(self):
        rows=fixture();rows[0]['execution_events'][2]['received_time_ns']=35
        with self.assertRaisesRegex(ValueError,'matching host forward'):
            ledger.advance(self.state(),rows,30)
        self.assertEqual(len(ledger.advance(self.state(),rows,35)['visible_device_events']),1)
    def test_history_truncation_deletion_and_insert_before_prefix_rejected(self):
        for mutation in ('delete_device','delete_host','earlier_request'):
            rows=fixture();first=ledger.advance(self.state(),rows,30)
            if mutation=='delete_device':rows[0]['execution_events'].pop()
            elif mutation=='delete_host':rows[0]['execution_events'].pop(0)
            else:
                extra=copy.deepcopy(rows[0]);extra['request_id']='early';extra['model']='b';extra['execution_events']=[dict(request_id='early',perf_ns=1,received_time_ns=2)];rows.append(extra)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):ledger.advance(first,rows,40)
    def test_alias_and_request_binding_cannot_change(self):
        for mutation in ('alias','request','duplicate'):
            rows=fixture();first=ledger.advance(self.state(),rows,30)
            if mutation=='alias':rows[0]['model']='b'
            elif mutation=='request':rows[0]['execution_events'][-1]['request_id']='other'
            else:rows.append(copy.deepcopy(rows[0]))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):ledger.advance(first,rows,40)
    def test_false_true_does_not_complete_request_or_other_streams(self):
        rows=fixture();self.completed(rows);state=ledger.advance(self.state(),rows,40)
        self.assertEqual([e['end_event_complete'] for e in state['visible_device_events']],[False,True])
        self.assertIsNone(state['host']['requests']['r']['handler_end_ns'])
        self.assertIsNone(state['host']['requests']['r']['worker_next_ready_ns'])
        self.assertIsNone(state['host']['requests']['r']['kernel_cursor'])
        self.assertIsNone(state['kernel_cursor']);self.assertFalse(state['main_lifecycle_resume_eligible'])
    def test_stream_identity_change_and_requery_after_true_rejected(self):
        for mutation in ('stream_id','device_index','forward_index','requery'):
            rows=fixture();self.completed(rows)
            if mutation=='requery':
                extra=copy.deepcopy(rows[0]['execution_events'][-1]);extra.update(observation_sequence=3,host_query_begin_ns=41,host_query_end_ns=42,perf_ns=42,received_time_ns=45);rows[0]['execution_events'].append(extra)
            else:rows[0]['execution_events'][-1][mutation]=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):ledger.advance(self.state(),rows,45)
    def test_each_request_restarts_sequence_and_has_its_own_clock(self):
        rows=fixture();other=copy.deepcopy(rows[0]);other['request_id']='s';other['model']='b'
        for event in other['execution_events']:
            event['request_id']='s'
            for key in ('perf_ns','host_query_begin_ns','host_query_end_ns','received_time_ns'):
                if key in event:event[key]+=10
            if 'event_id' in event:event['event_id']='s:forward:0:stream_end'
        rows.append(other);self.completed(rows,received=50)
        result=ledger.advance(self.state(),rows,50)
        self.assertEqual([(e['request_id'],e['observation_sequence']) for e in result['visible_device_events']],[('r',1),('s',1),('r',2)])
        self.assertEqual([e['sequence'] for e in result['host_feedback_prefix']],list(range(1,7)))
    def test_observation_reorder_cannot_invent_perrequest_sequence(self):
        rows=fixture();self.completed(rows);rows[0]['execution_events'][-1]['received_time_ns']=29
        with self.assertRaises(ValueError):ledger.advance(self.state(),rows,40)
    def test_inputs_and_old_checkpoints_stay_immutable(self):
        rows=fixture();original=copy.deepcopy(rows);old=self.state();saved=copy.deepcopy(old)
        result=ledger.advance(old,rows,30);self.assertEqual(rows,original);self.assertEqual(old,saved)
        rows[0]['execution_events'][-1]['stream_id']=99;self.assertEqual(result['visible_device_events'][0]['stream_id'],0)
        result['host']['aliases'].append('other')
        with self.assertRaisesRegex(ValueError,'Mutated'):ledger.advance(result,original,31)
    def test_nonfinite_or_boolean_elapsed_cannot_be_measured_interval(self):
        for value in (float('nan'),float('inf'),-1,True):
            rows=fixture();self.completed(rows);rows[0]['execution_events'][-1]['recorded_stream_elapsed_ms']=value
            with self.subTest(value=value),self.assertRaises(ValueError):ledger.advance(self.state(),rows,40)


if __name__=='__main__':unittest.main()
