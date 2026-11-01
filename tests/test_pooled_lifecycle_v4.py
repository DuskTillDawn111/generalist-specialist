import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.lifecycle import export_training_observations
from tools.predictor_validation_20261027.pooled_lifecycle_v4 import fit, PooledLifecycleProvider


ALIASES = {'text#0': {'family': 'text'}}
CONTEXT = 'same-source-bound-execution'


def observations():
    result = []
    for capture in range(2):
        for index, pool in enumerate(('short', 'long')):
            row = dict(request_id='r' + str(index), split='training', status='completed',
                       model='text#0', task='text', pool_id=pool,
                       worker_response={'end_perf_ns': 10_000_000},
                       completion_time_ns=12_000_000 + capture * 2_000_000,
                       response_reader_event_enqueue_time_ns=12_000_000 + capture * 2_000_000,
                       previous_handler_next_request_ready=dict(request_id='r' + str(index), pid=123,
                           perf_ns=11_000_000 + capture * 1_000_000,
                           received_time_ns=14_000_000, clock='host_perf_counter_ns',
                           boundary='previous_handler_next_request_ready', scope='after_response_flush_before_next_read'))
            raw = dict(split='training', kind='cpu_fixture', admission_enabled=False,
                       execution_context_sha256=CONTEXT, request_records=[row])
            result.extend(export_training_observations(raw, ALIASES, execution_context_sha256=CONTEXT,
                source='independent-capture-' + str(capture), clock_contract=dict(handler='host_perf_counter_ns',
                    runtime='host_perf_counter_ns', worker_next_ready='host_perf_counter_ns',
                    same_host_verified=True, source='cpu-clock-contract')))
    return result


class PooledLifecycleTests(unittest.TestCase):
    def model(self, rows=None):
        return fit(observations() if rows is None else rows, CONTEXT, allow_cpu_fixture=True)

    def provider(self, model=None, context=CONTEXT):
        return PooledLifecycleProvider(self.model() if model is None else model, context,
                                       allow_pooling_hypothesis=True, allow_cpu_fixture=True)

    def test_unseen_input_uses_frozen_joint_training_mean_without_fitting_it(self):
        model = self.model()
        before = copy.deepcopy(model)
        provider = self.provider(model)
        result = provider.delays(dict(alias='text#0', task='text', pool_id='unseen'), ALIASES, 'mean')
        self.assertEqual(result['parent_receipt_ms'], 3.)
        self.assertEqual(result['worker_next_ready_ms'], 1.5)
        self.assertIsNone(result['permit_release_ms'])
        self.assertFalse(result['load_independence_proven'])
        self.assertEqual(model, before)
        self.assertEqual(model['support'][0]['input_ids'], ['long', 'short'])

    def test_one_capture_or_single_input_is_not_supported(self):
        for rows in (observations()[:2], [r for r in observations() if r['pool_id'] == 'short']):
            model = self.model(rows)
            self.assertEqual(model['status'], 'UNKNOWN')
            with self.assertRaises(UnknownEvidence):
                self.provider(model)

    def test_training_split_context_and_duplicates_are_checked(self):
        for field, value in (('split', 'validation'), ('execution_context_sha256', 'other')):
            rows = observations()
            rows[-1][field] = value
            with self.assertRaises(ValueError):
                self.model(rows)
        rows = observations()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.model(rows + [rows[0]])

    def test_missing_ready_remains_unknown(self):
        rows = observations()
        rows[0]['timestamps']['worker_next_ready_ns'] = None
        rows[0]['delays']['worker_next_ready_ms'] = None
        rows[0]['ready_source'] = None
        result = self.provider(self.model(rows)).delays(
            dict(alias='text#0', task='text', pool_id='unseen'), ALIASES, 'mean')
        self.assertIsNone(result['worker_next_ready_ms'])
        self.assertFalse(result['known'])

    def test_new_task_or_changed_execution_is_unknown_or_rejected(self):
        with self.assertRaises(ValueError):
            self.provider(context='different')
        with self.assertRaises(UnknownEvidence):
            self.provider().delays(dict(alias='text#0', task='asr', pool_id='unseen'), ALIASES, 'mean')

    def test_hypothesis_and_fixture_require_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'Measured'):
            fit(observations(), CONTEXT)
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            PooledLifecycleProvider(self.model(), CONTEXT, allow_cpu_fixture=True)


if __name__ == '__main__':
    unittest.main()
