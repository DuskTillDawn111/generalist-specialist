"""Independent CPU boundary tests; no native collection or numeric fitting."""
import copy
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import twoFIFO_native_busy_state_v1 as busy
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v5_2 as engine
from tests.test_twoFIFO_settled_history_initial_v1 import snapshot, seeded_inputs


def fixture():
    row = dict(request_id='a', model='owner', scheduled_time_ns=0,
        enqueued_time_ns=0, actual_submit_time_ns=1, submit_write_completed_time_ns=2,
        execution_events=[dict(request_id='a',pid=41,perf_ns=2,received_time_ns=3,
            collector_dequeued_time_ns=4,collector_processing_time_ns=5)])
    snap = dict(snapshot_time_ns=20, request_records=[row], pending_request_ids=['a'],
        ready={'owner':dict(pid=41,spec_sha256='a'*64,clock='time.perf_counter_ns')},
        queued_runtime_events=[],runtime_errors=[],empty_initial_state_assumed=False)
    cohort = [dict(request_id=r,owner='owner',task='text',pool_id='p',arrival_ns=t)
              for r,t in [('a',0),('b',12)]]
    return snap, cohort


def inspect(snap, cohort, **kw):
    return busy.from_snapshot(snap,public_cohort=cohort,cutoff_ns=5,
        owner_producers={'owner':'stdout-41'},execution_context_sha256='b'*64,**kw)


class Poison(dict):
    def get(self,key,default=None):
        if key != 'collector_processing_time_ns':
            raise AssertionError('future body access: '+key)
        return super().get(key,default)
    def __deepcopy__(self,memo):
        raise AssertionError('future body copy')


class NativeBusyIndependentTests(unittest.TestCase):
    def test_collector_processing_alone_opens_message_body(self):
        s,c=fixture()
        s['request_records'][0]['execution_events'].append(Poison(collector_processing_time_ns=6))
        s['queued_runtime_events']=[Poison()]
        out=inspect(s,c)
        self.assertEqual([e['kind'] for e in out['collector_processed_prefix']],['START'])
        self.assertIsNone(out['reader_queued_envelope_count'])
        self.assertFalse(out['queued_message_bodies_read'])

    def test_processed_queries_and_phases_without_pid_cannot_create_start(self):
        s,c=fixture()
        for event in [dict(event='start',kind='gpu',segment_id='p'),
                      dict(event='observation',kind='device_event_query',schema=busy.QUERY_SCHEMA),
                      dict(kind='device_event_query',schema=busy.QUERY_SCHEMA)]:
            s['request_records'][0]['execution_events'].append(dict(event,request_id='a',
                perf_ns=5,received_time_ns=5,collector_dequeued_time_ns=5,
                collector_processing_time_ns=5))
        out=inspect(s,c)
        self.assertEqual([e['kind'] for e in out['collector_processed_prefix']],
                         ['START','PHASE','QUERY','QUERY'])
        self.assertEqual(set(out['native_request_states']['a']['observed_boundaries']),{'START'})
        self.assertIsNone(out['native_request_states']['a']['exact_remaining_work'])

    def test_busy_engine_is_never_called_even_after_complete_query(self):
        s,c=fixture()
        s['request_records'][0]['execution_events'].append(dict(request_id='a',
            event='observation',kind='device_event_query',schema=busy.QUERY_SCHEMA,
            end_event_complete=True,all_request_GPU_work_complete=True,
            perf_ns=5,received_time_ns=5,collector_dequeued_time_ns=5,
            collector_processing_time_ns=5))
        out=inspect(s,c)
        with patch.object(engine,'simulate',side_effect=AssertionError('unsafe busy resume')) as call:
            with self.assertRaisesRegex(UnknownEvidence,'busy_state_support_not_identified'):
                busy.forecast(out,requests=c,window_start_ns=5)
            call.assert_not_called()

    def test_same_processed_prefix_different_result_futures_produces_same_state(self):
        s,c=fixture();t=copy.deepcopy(s)
        for source,end in [(s,9),(t,90)]:
            source['snapshot_time_ns']=100
            source['request_records'][0].update(worker_response={'end_perf_ns':end,'success':True},
                response_reader_event_enqueue_time_ns=end+1,
                response_collector_dequeued_time_ns=end+2,
                response_collector_processing_time_ns=end+3)
        self.assertEqual(inspect(s,c),inspect(t,c))

    def test_future_registry_row_does_not_change_historical_prefix(self):
        s,c=fixture();t=copy.deepcopy(s)
        t['request_records'].append(dict(request_id='b',model='owner',scheduled_time_ns=12,
            enqueued_time_ns=12,actual_submit_time_ns=13,submit_write_completed_time_ns=14,
            execution_events=[Poison(collector_processing_time_ns=18)]))
        t['pending_request_ids'].append('b')
        self.assertEqual(inspect(s,c),inspect(t,c))

    def test_past_scheduled_calendar_alone_does_not_reveal_future_registration(self):
        s,c=fixture();c[1]['arrival_ns']=3;t=copy.deepcopy(s)
        t['request_records'].append(dict(request_id='b',model='owner',scheduled_time_ns=3,
            enqueued_time_ns=12,actual_submit_time_ns=13,submit_write_completed_time_ns=14,
            execution_events=[Poison(collector_processing_time_ns=18)]))
        t['pending_request_ids'].append('b')
        self.assertEqual(inspect(s,c),inspect(t,c))

    def test_actual_sender_enqueue_can_reveal_registration_before_start(self):
        s,c=fixture();c[1]['arrival_ns']=3
        s['request_records'].append(dict(request_id='b',model='owner',scheduled_time_ns=3,
            enqueued_time_ns=4,actual_submit_time_ns=13,submit_write_completed_time_ns=14,
            execution_events=[Poison(collector_processing_time_ns=18)]))
        s['pending_request_ids'].append('b')
        state=inspect(s,c)
        self.assertIn('b',state['native_request_states'])
        self.assertEqual(state['sender_prefix']['b'],dict(scheduled_time_ns=3,enqueued_time_ns=4))
        self.assertIn('STARTED_NOTICE_PENDING',
            state['native_request_states']['b']['possible_current_states'])

    def test_previous_prefix_rejects_inserting_old_processed_fact(self):
        s,c=fixture();before=inspect(s,c)
        s['request_records'][0]['execution_events'].append(dict(request_id='a',event='start',
            kind='gpu',segment_id='p',perf_ns=5,received_time_ns=5,
            collector_dequeued_time_ns=5,collector_processing_time_ns=5))
        with self.assertRaisesRegex(UnknownEvidence,'changed_processed_prefix'):
            inspect(s,c,previous=before)

    def test_settled_adapter_calls_real_engine_and_preserves_resume(self):
        s=snapshot();kw=seeded_inputs();kw.pop('settled_initial_state')
        for row in s['request_records']:
            row['execution_events'][0]['request_id']=row['request_id']
        state=busy.from_snapshot(s,public_cohort=kw['requests'],cutoff_ns=50,
            forecast_origin_ns=100,owner_producers={'owner':'owner','idle':'idle:stdout'},
            execution_context_sha256='b'*64)
        self.assertTrue(state['reference_entry_eligible'])
        with patch.object(engine,'simulate',wraps=engine.simulate) as call:
            whole=busy.forecast(state,**kw)
            call.assert_called_once()
        partial=busy.forecast(state,**dict(kw,cutoff_ns=105))
        self.assertEqual(engine.simulate(**kw,continuation_state=partial['continuation_state']),whole)
        self.assertFalse(state['native_source_authenticated'])
        self.assertFalse(state['busy_numeric_resume_eligible'])

    def test_mutated_state_and_forecast_override_are_rejected_before_engine(self):
        s,c=fixture();out=inspect(s,c)
        corrupt=copy.deepcopy(out);corrupt['reference_entry_eligible']=True
        with patch.object(engine,'simulate',side_effect=AssertionError('unsafe call')):
            with self.assertRaisesRegex(UnknownEvidence,'mutated_state'):
                busy.forecast(corrupt,requests=c,window_start_ns=5)
            with self.assertRaisesRegex(UnknownEvidence,'caller_cannot_override_native_state'):
                busy.forecast(out,requests=c,window_start_ns=5,continuation_state={})


if __name__=='__main__':
    unittest.main()
