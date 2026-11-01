import copy
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_twoFIFO_settled_history_initial_v1 import snapshot, seeded_inputs
from tools.predictor_validation_20261027 import twoFIFO_native_busy_state_v1 as adapter
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v5_2 as engine


class FutureMessage(dict):
    def get(self, key, default=None):
        if key != 'collector_processing_time_ns':
            raise AssertionError('Future event body was read: ' + key)
        return super().get(key, default)

    def __deepcopy__(self, memo):
        raise AssertionError('Future event body was copied')


class FutureResponse(dict):
    def get(self, key, default=None):
        raise AssertionError('Future RESULT was read: ' + key)

    def __deepcopy__(self, memo):
        raise AssertionError('Future RESULT was copied')


def busy_snapshot(end=100):
    ready = dict(owner=dict(pid=111, spec_sha256='a' * 64, clock='time.perf_counter_ns'))
    start = dict(request_id='r0', pid=111, perf_ns=1, received_time_ns=2,
        collector_dequeued_time_ns=3, collector_processing_time_ns=4)
    row = dict(request_id='r0', model='owner', scheduled_time_ns=0,
        enqueued_time_ns=0, actual_submit_time_ns=0, submit_write_completed_time_ns=1,
        execution_events=[start], worker_response=dict(success=True, end_perf_ns=end),
        response_reader_event_enqueue_time_ns=end+1,
        response_collector_dequeued_time_ns=end+2,
        response_collector_processing_time_ns=end+3,
        previous_handler_next_request_ready=dict(request_id='r0',pid=111,
            perf_ns=end+4, received_time_ns=end+5,
            collector_dequeued_time_ns=end+6, collector_processing_time_ns=end+7))
    return dict(snapshot_time_ns=5, request_records=[row], ready=ready,
        pending_request_ids=['r0'], queued_runtime_events=[], runtime_errors=[],
        empty_initial_state_assumed=False)


def public():
    return [dict(request_id='r0',owner='owner',task='text',pool_id='p',arrival_ns=0),
            dict(request_id='r1',owner='owner',task='text',pool_id='p',arrival_ns=10)]


def inspect(s, **kw):
    return adapter.from_snapshot(s,public_cohort=public(),cutoff_ns=5,
        owner_producers={'owner':'owner'},execution_context_sha256='b'*64,**kw)


class NativeBusyStateTests(unittest.TestCase):
    def test_processing_cutoff_precedes_body_and_not_reader_visibility(self):
        s = busy_snapshot()
        row = s['request_records'][0]
        row['execution_events'].append(FutureMessage(collector_processing_time_ns=6))
        row['previous_handler_next_request_ready'] = FutureMessage(collector_processing_time_ns=7)
        row['worker_response'] = FutureResponse()
        # These envelopes have reached reader/dequeue; still not collector-visible.
        row['response_reader_event_enqueue_time_ns'] = 3
        row['response_collector_dequeued_time_ns'] = 4
        row['response_collector_processing_time_ns'] = 6
        out = inspect(s)
        self.assertEqual([m['kind'] for m in out['collector_processed_prefix']], ['START'])
        self.assertIsNone(out['native_request_states']['r0']['exact_remaining_work'])
        self.assertFalse(out['busy_numeric_resume_eligible'])

    def test_full_future_cohort_and_four_stations_retained(self):
        out = inspect(busy_snapshot())
        self.assertEqual(out['offered_requests'], 2)
        self.assertEqual([r['request_id'] for r in out['full_offered_cohort']], ['r0','r1'])
        self.assertTrue(out['full_offered_cohort'][1]['public_future_arrival'])
        start = out['collector_processed_prefix'][0]
        self.assertEqual([start[k] for k in ('perf_ns','received_time_ns',
            'collector_dequeued_time_ns','collector_processing_time_ns')], [1,2,3,4])
        self.assertFalse(out['native_request_states']['r0']['host_elapsed_is_completed_work'])

    def test_same_visible_prefix_has_distinct_legal_future_and_blocks_resume(self):
        source_a, source_b = busy_snapshot(10), busy_snapshot(100)
        # Genuine retrospective snapshots are captured after their respective
        # futures; cutoff 5 selects the identical collector-processed prefix.
        source_a['snapshot_time_ns'], source_b['snapshot_time_ns'] = 20, 110
        a, b = inspect(source_a), inspect(source_b)
        self.assertEqual(a, b)
        self.assertEqual(a['native_request_states']['r0']['possible_current_states'],
            ['GPU_OR_HOST_WORK_ACTIVE','HANDLER_FINISHED_NOTICE_PENDING'])
        for state in (a, b):
            with patch.object(engine,'simulate',side_effect=AssertionError('unsafe engine call')):
                with self.assertRaisesRegex(UnknownEvidence,'busy_state_support_not_identified'):
                    adapter.forecast(state,requests=public(),window_start_ns=5)

    def test_missing_start_does_not_prove_waiting_and_queue_body_is_opaque(self):
        s = busy_snapshot()
        s['request_records'][0]['execution_events'] = []
        s['queued_runtime_events'] = [FutureResponse()]
        out = inspect(s)
        self.assertIn('STARTED_NOTICE_PENDING',out['native_request_states']['r0']['possible_current_states'])
        self.assertFalse(out['native_request_states']['r0']['missing_START_proves_waiting'])
        self.assertEqual(out['reader_queued_envelope_count'], 1)
        self.assertTrue(out['dequeued_unprocessed_message_possible'])

    def test_prefix_cannot_be_rewritten_or_removed_between_cutoffs(self):
        s = busy_snapshot(); first = inspect(s)
        self.assertEqual(inspect(s,previous=first),first)
        for mutate in (lambda x:x['request_records'][0]['execution_events'].clear(),
            lambda x:x['request_records'][0]['execution_events'][0].update(perf_ns=0),
            lambda x:x['request_records'][0].update(actual_submit_time_ns=1)):
            changed = copy.deepcopy(s); mutate(changed)
            with self.subTest(change=changed),self.assertRaises(UnknownEvidence):
                inspect(changed,previous=first)

    def test_later_processing_admits_original_fact_without_using_future_end(self):
        s = busy_snapshot()
        s['request_records'][0]['execution_events'][0]['collector_processing_time_ns'] = 6
        first = inspect(s)
        self.assertEqual(first['collector_processed_prefix'], [])
        s['snapshot_time_ns'] = 6
        second = adapter.from_snapshot(s,public_cohort=public(),cutoff_ns=6,
            owner_producers={'owner':'owner'},execution_context_sha256='b'*64,previous=first)
        self.assertEqual(len(second['collector_processed_prefix']),1)
        self.assertEqual(second['collector_processed_prefix'][0]['perf_ns'],1)
        self.assertIsNone(second['native_request_states']['r0']['exact_remaining_work'])

    def test_missing_or_invalid_processing_clock_is_rejected(self):
        for value in (None,True,-1,1.5):
            s = busy_snapshot()
            s['request_records'][0]['execution_events'][0]['collector_processing_time_ns'] = value
            with self.subTest(value=value),self.assertRaises(UnknownEvidence):
                inspect(s)

    def test_real_query_emitter_without_pid_is_a_query_not_START_or_phase(self):
        from tools.predictor_validation_20261027.device_progress_observer_v1 import DeviceProgressObserver
        emitted = []
        # Execute the real native payload producer without initializing CUDA.
        producer = DeviceProgressObserver.__new__(DeviceProgressObserver)
        producer.sequence, producer.request_id = 0, 'r0'
        producer.queries, producer.sink = [], emitted.append
        for done, clock in ((False, 6), (True, 7)):
            producer._observation(dict(phase_id='prefill_0000', stream_id=123,
                event_id='r0:prefill_0000:stream_end', device_index=0, forward_index=0,
                host_query_begin_ns=clock,host_query_end_ns=clock,
                end_event_complete=done,in_flight_remaining_fraction=None,
                evidence_scope='recorded_current_stream_only',all_request_GPU_work_complete=False,
                completion_time_ns=None,physical_SM_service_ns=None,
                recorded_stream_elapsed_ms=0.001 if done else None))
        s = busy_snapshot()
        s['snapshot_time_ns'] = 8
        row = s['request_records'][0]
        # Both real semantic phase payloads and queries lack native PID fields.
        for edge, clock in (('start', 4), ('end', 5)):
            row['execution_events'].append(dict(request_id='r0',event=edge,kind='gpu',
                segment_id='prefill_0000',perf_ns=clock,received_time_ns=clock,
                collector_dequeued_time_ns=clock,collector_processing_time_ns=clock))
        for query in emitted:
            self.assertNotIn('pid', query)
            clock = query['perf_ns']
            query.update(received_time_ns=clock,collector_dequeued_time_ns=clock,
                collector_processing_time_ns=clock)
            row['execution_events'].append(query)
        out = adapter.from_snapshot(s,public_cohort=public(),cutoff_ns=8,
            owner_producers={'owner':'owner'},execution_context_sha256='b'*64)
        self.assertEqual([e['kind'] for e in out['collector_processed_prefix']],
            ['START','PHASE','PHASE','QUERY','QUERY'])
        self.assertEqual(set(out['native_request_states']['r0']['observed_boundaries']), {'START'})
        self.assertIsNone(out['native_request_states']['r0']['exact_remaining_work'])
        self.assertFalse(out['busy_numeric_resume_eligible'])
        # An explicitly typed query stays a query even if an upstream exporter
        # omits the optional observation marker; it must not become a START.
        del row['execution_events'][-1]['event']
        out = adapter.from_snapshot(s,public_cohort=public(),cutoff_ns=8,
            owner_producers={'owner':'owner'},execution_context_sha256='b'*64)
        self.assertEqual(out['collector_processed_prefix'][-1]['kind'], 'QUERY')

    def test_foreign_phase_pid_and_unclassified_payload_rejected(self):
        for event in (dict(request_id='r0',kind='gpu',event='start',segment_id='p',pid=222),
                      dict(request_id='r0',kind='unknown_observation',pid=111)):
            s = busy_snapshot()
            event.update(perf_ns=4,received_time_ns=4,collector_dequeued_time_ns=4,
                collector_processing_time_ns=5)
            s['request_records'][0]['execution_events'].append(event)
            with self.subTest(event=event),self.assertRaises(UnknownEvidence):
                inspect(s)

    def test_future_registry_existence_and_overdue_schedule_do_not_reveal_state(self):
        for arrival in (0,10):
            s = busy_snapshot(); s['snapshot_time_ns'] = 30
            cohort = public(); cohort[1]['arrival_ns'] = arrival
            before = adapter.from_snapshot(s,public_cohort=cohort,cutoff_ns=5,
                owner_producers={'owner':'owner'},execution_context_sha256='b'*64)
            s['request_records'].append(dict(request_id='r1',model='owner',
                scheduled_time_ns=arrival,enqueued_time_ns=12,actual_submit_time_ns=13,
                submit_write_completed_time_ns=14,
                execution_events=[FutureMessage(collector_processing_time_ns=20)]))
            s['pending_request_ids'].append('r1')
            after = adapter.from_snapshot(s,public_cohort=cohort,cutoff_ns=5,
                owner_producers={'owner':'owner'},execution_context_sha256='b'*64)
            self.assertEqual(before,after)

    def test_settled_native_boundary_reaches_real_v5_2_and_exact_resume(self):
        s = snapshot()
        for row in s['request_records']:
            row['execution_events'][0]['request_id'] = row['request_id']
        kw = seeded_inputs()
        seed = kw.pop('settled_initial_state')
        state = adapter.from_snapshot(s,public_cohort=kw['requests'],cutoff_ns=50,
            forecast_origin_ns=100,owner_producers={'owner':'owner','idle':'idle:stdout'},
            execution_context_sha256='b'*64)
        self.assertTrue(state['reference_entry_eligible'])
        self.assertFalse(state['busy_numeric_resume_eligible'])
        direct_seed = copy.deepcopy(seed)
        # Snapshot SHA includes realistic native START identity added above.
        direct_seed['snapshot_sha256'] = state['settled_initial_state']['snapshot_sha256']
        expected = engine.simulate(**kw,settled_initial_state=direct_seed)
        actual = adapter.forecast(state,**kw)
        self.assertEqual(actual,expected)
        partial = adapter.forecast(state,**dict(kw,cutoff_ns=105))
        self.assertEqual(engine.simulate(**kw,continuation_state=partial['continuation_state']),actual)

    def test_caller_cannot_change_cohort_or_supply_generated_checkpoint(self):
        state = inspect(busy_snapshot())
        for args in (dict(requests=[],window_start_ns=5),
            dict(requests=public(),window_start_ns=5,continuation_state={})):
            with self.subTest(args=args),self.assertRaises(UnknownEvidence):
                adapter.forecast(state,**args)


if __name__ == '__main__':
    unittest.main()
