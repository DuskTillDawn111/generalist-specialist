"""Synthetic cutoff and censoring tests; no real training/validation labels."""
import copy
import math
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import native_queue_risk_rows_v9 as risk


def public(rid, arrival, alias='a'):
    return dict(request_id=rid, alias=alias, task='text', pool_id='synthetic-' + rid, arrival_ns=arrival)


def record(row, start, received, receipt):
    return dict(request_id=row['request_id'], model=row['alias'], status='completed',
                execution_events=[dict(perf_ns=start, received_time_ns=received)],
                response_reader_event_enqueue_time_ns=receipt,
                previous_handler_next_request_ready=dict(received_time_ns=receipt + 1))


def protocol(horizon=10, latency_ns=5):
    return risk.protocol(dict(split='training', training_only=True, execution_context_sha256='a' * 64,
        risk_target=dict(horizon_ns=horizon, recovery_threshold_ms=500.),
        latency_reference_ms=dict(text=latency_ns / 1e6)))


class IndependentRiskRowsTests(unittest.TestCase):
    def feature(self, trace, records, now=10, provider=None):
        return risk.features(trace, risk.visible_projection(records, now), now_ns=now, declared_at_ns=0,
            context='a' * 64, cost_provider=provider or (lambda public_input: 1.), scales_ns=(5, 10, 20))

    def outcome(self, trace, records, now=10, observed=100, target=None):
        return risk.offline_outcome(trace, records, now_ns=now, observation_end_ns=observed,
                                    risk_protocol=target or protocol())

    def test_current_visible_queue_differs_from_offline_worker_queue(self):
        trace = [public('r', 0)]
        records = [record(trace[0], 5, 15, 20)]
        feature = self.feature(trace, records)
        self.assertEqual(feature['waiting_requests'], 1)
        self.assertEqual(feature['oldest_waiting_ms'], 10 / 1e6)
        self.assertEqual(self.outcome(trace, records)['recovery_ms'], 0.)

    def test_future_feedback_outputs_and_quality_do_not_change_features(self):
        trace = [public('r', 0), public('future', 20, alias='b')]
        records = [record(trace[0], 5, 15, 30), record(trace[1], 21, 25, 40)]
        expected = self.feature(trace, records)
        mutated = copy.deepcopy(records)
        for row in mutated:
            row.update(worker_response=dict(prediction='UNREAD_SENTINEL', answers=['UNREAD_SENTINEL']),
                       quality_score=math.nan, completion_time_ns=-123)
            row['execution_events'][0]['perf_ns'] += 999
            row['response_reader_event_enqueue_time_ns'] += 999
        self.assertEqual(self.feature(trace, mutated), expected)

    def test_cost_model_receives_only_public_static_identity(self):
        trace = [public('waiting', 0), public('future', 20)]
        calls = []
        def cost(fields):
            self.assertEqual(set(fields), {'alias', 'task', 'pool_id'})
            calls.append(fields)
            return 2.
        self.feature(trace, [], provider=cost)
        self.assertTrue(calls)

    def test_declared_envelope_covers_later_bursts_and_half_open_windows(self):
        trace = [public('at-cutoff', 10), public('a', 20), public('b', 25), public('c', 26), public('d', 29)]
        result = self.feature(trace, [])
        self.assertEqual(result['waiting_requests'], 1)
        self.assertEqual([e['max_offered_solo_work_ms'] for e in result['future_work_envelopes']], [3., 4., 4.])

    def test_arrival_at_cutoff_included_and_horizon_endpoint_excluded(self):
        trace = [public('included', 10), public('excluded', 20)]
        records = [record(trace[0], 10, 10, 15), record(trace[1], 20, 20, 99)]
        result = self.outcome(trace, records)
        self.assertFalse(result['violation'])
        records[0]['response_reader_event_enqueue_time_ns'] = 16
        self.assertTrue(self.outcome(trace, records)['violation'])

    def test_missing_target_receipt_and_short_observation_are_censored(self):
        trace = [public('r', 10)]
        records = [record(trace[0], 11, 11, 15)]
        records[0]['response_reader_event_enqueue_time_ns'] = None
        with self.assertRaisesRegex(UnknownEvidence, 'CENSORED_RISK_LATENCY'):
            self.outcome(trace, records)
        with self.assertRaisesRegex(UnknownEvidence, 'CENSORED_RISK_FUTURE_HORIZON'):
            self.outcome(trace, records, observed=19)

    def test_recovery_merges_touching_waits_without_false_empty_instant(self):
        trace = [public('first', 0), public('second', 15, alias='b')]
        records = [record(trace[0], 15, 15, 16), record(trace[1], 20, 20, 21)]
        self.assertEqual(self.outcome(trace, records)['recovery_ms'], 10 / 1e6)

    def test_nonfinite_receipt_cannot_become_negative_risk_label(self):
        trace = [public('r', 10)]
        records = [record(trace[0], 11, 11, 15)]
        records[0]['response_reader_event_enqueue_time_ns'] = math.nan
        with self.assertRaises((ValueError, UnknownEvidence)):
            self.outcome(trace, records)

    def test_partial_future_capture_is_explicitly_conservatively_censored(self):
        trace = [public('near', 10), public('far', 200)]
        records = [record(trace[0], 10, 10, 15), dict(request_id='far', execution_events=[])]
        self.assertIn('require full original cohort STARTs', protocol()['censoring'])
        with self.assertRaisesRegex(UnknownEvidence, 'CENSORED_RISK_START_LABEL'):
            self.outcome(trace, records, observed=100)


if __name__ == '__main__':
    unittest.main()
