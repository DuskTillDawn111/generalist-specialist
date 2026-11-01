"""Independent counterexamples for the frozen fresh181 CPU-only collector.

These tests exercise the real collector control flow with the author's CPU
transport fixture. They neither authenticate a native forecast nor use a GPU.
"""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tests import test_twoFIFO_fresh181_execute_v1 as fixtures
from tools.predictor_validation_20261027 import twoFIFO_fresh181_prefix_v1 as p
from tools.predictor_validation_20261027 import twoFIFO_fresh181_execute_v1 as m


class IndependentFreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.FreshTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.FreshTests.tearDownClass()

    def collector(self, *args, **kwargs):
        return fixtures.FreshTests(methodName='runTest').run_collector(*args, **kwargs)

    def test_future_reader_ready_and_unprocessed_messages_cannot_seed_forecast(self):
        _, _, _, seen = self.collector()
        source = seen['prefix']['prefix_snapshot']
        cutoff = source['snapshot_time_ns']
        epoch = seen['gate']['forecast_origin_ns']
        mutations = [
            lambda s: s['request_records'][-1].update(response_reader_event_enqueue_time_ns=cutoff+1),
            lambda s: s['request_records'][-1]['previous_handler_next_request_ready'].update(collector_processing_time_ns=cutoff+1),
            lambda s: s['request_records'][-1].update(status='incomplete'),
            lambda s: s['request_records'][-1]['worker_response'].update(success=False),
            lambda s: s.update(runtime_errors=['reader failed']),
            lambda s: s.update(empty_initial_state_assumed=True),
        ]
        for mutate in mutations:
            snapshot = copy.deepcopy(source)
            mutate(snapshot)
            with self.subTest(mutation=mutate), self.assertRaises((ValueError, UnknownEvidence)):
                p.derive(fixtures.FreshTests.plan, snapshot, epoch)

    def test_expensive_authentication_crossing_origin_sends_no_target(self):
        original = p.audit_seal

        def setup(gate_ref, plan, clock):
            epoch = p.checked(gate_ref)['forecast_origin_ns']

            def slow_audit(*args, **kwargs):
                value = original(*args, **kwargs)
                clock.wait(epoch)
                return value

            self.audit_patch = patch.object(p, 'audit_seal', side_effect=slow_audit)
            self.audit_patch.start()

        try:
            raw, runtime, _, _ = self.collector(setup)
        finally:
            self.audit_patch.stop()
        self.assertEqual(raw['status'], 'failed')
        self.assertEqual(len(runtime.enqueued), 18)
        self.assertEqual(len(raw['request_records']), 181)
        self.assertTrue(all(r['status'] == 'not_started' for r in raw['request_records']))
        self.assertFalse(raw['target_response_labels_collected'])

    def test_changed_then_restored_sealed_forecast_rejected_before_first_send(self):
        original_wait = fixtures.Clock.wait

        def hooked_wait(clock, at):
            if hasattr(clock, 'before_wait'):
                clock.before_wait(at)
            original_wait(clock, at)

        def setup(gate_ref, plan, clock):
            gate = p.checked(gate_ref)
            path = Path(gate_ref['path']).parent/'forecast.json'
            fired = False

            def wait(at):
                nonlocal fired
                if not fired and at == gate['forecast_origin_ns']:
                    fired = True
                    original = path.read_bytes()
                    path.write_bytes(original+b' ')
                    path.write_bytes(original)
            clock.before_wait = wait

        with patch.object(fixtures.Clock, 'wait', hooked_wait):
            raw, runtime, _, _ = self.collector(setup)
        self.assertEqual(raw['status'], 'failed', raw['errors'])
        self.assertEqual(len(runtime.enqueued), 18)
        self.assertTrue(runtime.closed)
        self.assertEqual(len(raw['request_records']), 181)
        self.assertFalse(raw['target_response_labels_collected'])

    def test_native_gate_cannot_accept_relabelled_cpu_seal(self):
        def setup(gate_ref, plan, clock):
            gate = p.checked(gate_ref)
            prediction = p.save(Path(gate_ref['path']).parent/'test-prediction.json', fixtures.forecast(gate))
            seal = p.seal_prediction(gate_ref['path'], prediction['path'], gate['expected_seal_path'], clock_ns=clock.clock)
            value = p.checked(seal)
            value['CPU_TEST_ONLY'] = False
            Path(seal['path']).write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                p.audit_seal(p.receipt(seal['path']), gate_ref, now_ns=clock.now)
            Path(seal['path']).unlink()

        raw, runtime, _, _ = self.collector(setup)
        self.assertEqual(raw['status'], 'completed', raw['errors'])
        self.assertEqual(len(runtime.enqueued), 199)
        self.assertEqual(raw['backend'], 'CPU_TEST_ONLY')
        self.assertFalse(raw['main_queue_provider_eligible'])

    def test_unknown_probability_cannot_be_dropped_or_numerically_renamed(self):
        _, _, _, seen = self.collector()
        gate = seen['gate']
        value = fixtures.forecast(gate)
        first = value['draws'][0]
        first['weight'] = dict(numerator=1, denominator=3)
        second = copy.deepcopy(first)
        second.update(draw_id='second', weight=dict(numerator=2, denominator=3))
        value['draws'].append(second)
        p.validate_forecast(value, gate)
        for mutation in ('remove', 'partial', 'invent'):
            changed = copy.deepcopy(value)
            if mutation == 'remove':
                changed['draws'].pop()
            elif mutation == 'partial':
                changed['draws'][1]['requests'].pop()
            else:
                changed['draws'][1]['requests'][0]['completion_time_ns'] = gate['public_requests'][0]['arrival_ns']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                p.validate_forecast(changed, gate)


if __name__ == '__main__':
    unittest.main()
