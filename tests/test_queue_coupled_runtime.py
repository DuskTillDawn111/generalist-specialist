"""Feedback visibility and immutable evidence boundaries for the new runtime."""
from copy import deepcopy
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from tools.queue_coupled_utility_20261023.runtime_window import feedback, choose_and_enqueue
from tools.queue_coupled_utility_20261023 import evaluation


class FeedbackTests(unittest.TestCase):
    def previous(self):
        return dict(request_id='measurement:0', phase='measurement', model='text#0',
                    task='text', pool_id='pool', status='completed', enqueued_time_ns=100,
                    actual_submit_time_ns=110, completion_time_ns=150,
                    worker_response=dict(service_ms=0.01, raw_output='PRIVATE', answers=['PRIVATE']))

    def test_feedback_does_not_expose_outputs_or_worker_future_fields(self):
        source = self.previous()
        value = feedback(source)
        self.assertEqual(set(value), {'request_id', 'model', 'task', 'pool_id', 'enqueued_ns',
                                     'actual_submit_ns', 'completion_time_ns', 'worker_service_ms'})
        altered = deepcopy(source)
        altered['worker_response'].update(raw_output='OTHER', generated_tokens=999, start_ns=999999)
        self.assertEqual(value, feedback(altered))

    def test_feedback_precedes_choose_once_and_under_lock(self):
        lock = threading.RLock()
        previous = self.previous()
        runtime = SimpleNamespace(lock=lock, pending={previous['request_id']: previous})
        calls = []
        class Dispatcher:
            aliases = {'text#0': {}}
            def observe_completed(self, rows, now):
                self.rows = rows
                calls.append(('feedback', len(rows), lock._is_owned()))
            def set_arrival(self, rid, when):
                calls.append(('arrival', rid, when))
            def choose(self, task, pool, now, snapshot):
                calls.append(('choose', lock._is_owned()))
                return {'alias': 'text#0'}
        dispatcher = Dispatcher()
        runtime.enqueue = lambda row, sample: calls.append(('enqueue', lock._is_owned()))
        row = dict(request_id='measurement:1', phase='measurement', task='text', pool_id='p', scheduled_time_ns=180)
        with patch('tools.queue_coupled_utility_20261023.runtime_window.time.perf_counter_ns', return_value=200):
            choose_and_enqueue(runtime, dispatcher, row, {})
            choose_and_enqueue(runtime, dispatcher, dict(row, request_id='measurement:2'), {})
        self.assertEqual(calls[0], ('feedback', 1, True))
        self.assertEqual(calls[2], ('choose', True))
        self.assertEqual(calls[3], ('enqueue', True))
        self.assertEqual(calls[4], ('feedback', 0, True))
        self.assertEqual(row['routing_decision']['newly_observed_completions'], [feedback(previous)])

    def test_replay_rejects_extra_feedback_fields_before_dispatch(self):
        previous = self.previous()
        previous['routing_decision'] = dict(newly_observed_completions=[])
        leaked = dict(feedback(previous), answer='PRIVATE')
        current = dict(request_id='measurement:1', routing_decision=dict(newly_observed_completions=[leaked]))
        raw = dict(request_records=[previous, current], measurement_window={}, policy_contract={})
        with patch('tools.queue_coupled_utility_20261023.protocol.validate_contract'):
            audit = evaluation.audit_phase(None, raw)
        self.assertFalse(audit['verified'])
        self.assertIn('FEEDBACK_ALLOWLIST', audit['errors'][0])


if __name__ == '__main__':
    unittest.main()
