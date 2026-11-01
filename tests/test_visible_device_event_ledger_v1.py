import copy
import unittest

from tools.predictor_validation_20261027.visible_device_event_ledger_v1 import (
    PROBE, advance, new_state,
)


def fixture():
    events = [dict(request_id='r', perf_ns=10, received_time_ns=11),
        dict(request_id='r', perf_ns=12, received_time_ns=13,
             event='start', kind='gpu', segment_id='forward:0'),
        dict(request_id='r', perf_ns=20, received_time_ns=21,
             event='end', kind='gpu', segment_id='forward:0')]
    query = dict(request_id='r', perf_ns=23, received_time_ns=30,
        event='observation', kind='device_event_query', schema=PROBE,
        observation_sequence=1, phase_id='forward:0', segment_id='forward:0',
        event_id='r:forward:0:stream_end', stream_id=0, device_index=0,
        forward_index=0, host_query_begin_ns=22, host_query_end_ns=23,
        end_event_complete=False, evidence_scope='recorded_current_stream_only',
        all_request_GPU_work_complete=False, in_flight_remaining_fraction=None,
        completion_time_ns=None, physical_SM_service_ns=None,
        recorded_stream_elapsed_ms=None)
    events.append(query)
    return [dict(request_id='r', model='a', execution_events=events)]


class LedgerTests(unittest.TestCase):
    def state(self, **kwargs):
        return new_state({'a': {}}, 'context', **kwargs)

    def test_delayed_receipt_controls_visibility_and_no_cursor(self):
        rows = fixture()
        before = advance(self.state(), rows, 29)
        self.assertEqual(before['visible_device_events'], [])
        self.assertIsNotNone(before['host']['requests']['r']['phases']['forward:0']['end_ns'])
        after = advance(before, rows, 30)
        self.assertEqual(len(after['visible_device_events']), 1)
        self.assertIsNone(after['kernel_cursor'])
        self.assertIsNone(after['host']['requests']['r']['kernel_cursor'])
        self.assertFalse(after['main_lifecycle_resume_eligible'])
        self.assertEqual(before['visible_device_events'], [])

    def test_false_then_true_preserves_limited_evidence(self):
        rows = fixture()
        first = advance(self.state(), rows, 30)
        query = copy.deepcopy(rows[0]['execution_events'][-1])
        query.update(observation_sequence=2, host_query_begin_ns=31,
            host_query_end_ns=32, perf_ns=32, received_time_ns=40,
            end_event_complete=True, recorded_stream_elapsed_ms=2.5)
        rows[0]['execution_events'].append(query)
        result = advance(first, rows, 40)
        self.assertEqual(len(result['visible_device_events']), 2)
        self.assertIsNone(result['kernel_cursor'])
        self.assertFalse(result['main_lifecycle_resume_eligible'])

    def test_future_unknown_body_not_consumed(self):
        rows = fixture()
        rows[0]['execution_events'].append(dict(received_time_ns=100))
        result = advance(self.state(), rows, 30)
        self.assertEqual(len(result['visible_device_events']), 1)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            advance(result, rows, 100)

    def test_old_visible_history_cannot_be_rewritten(self):
        rows = fixture()
        result = advance(self.state(), rows, 30)
        rows[0]['execution_events'][-1]['stream_id'] = 9
        with self.assertRaisesRegex(ValueError, 'history changed'):
            advance(result, rows, 31)
        rows = fixture()
        rows[0]['execution_events'][0]['perf_ns'] = 9
        with self.assertRaisesRegex(ValueError, 'host history changed'):
            advance(result, rows, 31)

    def test_global_progress_and_clock_fabrication_rejected(self):
        for key, value in [('all_request_GPU_work_complete', True),
                           ('in_flight_remaining_fraction', .5),
                           ('physical_SM_service_ns', 1),
                           ('host_query_end_ns', 40),
                           ('observation_sequence', 2)]:
            with self.subTest(key=key):
                rows = fixture()
                rows[0]['execution_events'][-1][key] = value
                with self.assertRaises(ValueError):
                    advance(self.state(), rows, 30)

    def test_missing_host_forward_and_alias_rejected(self):
        rows = fixture()
        del rows[0]['execution_events'][1:3]
        with self.assertRaisesRegex(ValueError, 'matching host forward'):
            advance(self.state(), rows, 30)
        rows = fixture()
        rows[0]['model'] = 'unknown'
        with self.assertRaises(ValueError):
            advance(self.state(), rows, 30)

    def test_checkpoint_and_cutoff_integrity(self):
        rows = fixture()
        result = advance(self.state(frozen_cutoff_ns=30), rows, 30)
        with self.assertRaisesRegex(ValueError, 'Frozen'):
            advance(result, rows, 31)
        with self.assertRaisesRegex(ValueError, 'Backward'):
            advance(result, rows, 29)
        result['main_lifecycle_resume_eligible'] = True
        with self.assertRaisesRegex(ValueError, 'Mutated'):
            advance(result, rows, 30)


if __name__ == '__main__':
    unittest.main()
