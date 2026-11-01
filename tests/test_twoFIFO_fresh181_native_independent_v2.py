"""Independent CPU deadline/watch counterexamples; no native source claim."""
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_fresh181_native_v2 import NativeV2Tests
from tools.predictor_validation_20261027 import twoFIFO_fresh181_native_v2 as m
from tools.predictor_validation_20261027 import twoFIFO_matching_prospective_bridge_v2 as typed
from tools.predictor_validation_20261027.stage_response_features_v1 import checked


class IndependentNativeV2Tests(unittest.TestCase):
    setUpClass = classmethod(NativeV2Tests.setUpClass.__func__)
    tearDownClass = classmethod(NativeV2Tests.tearDownClass.__func__)
    authentication = NativeV2Tests.authentication
    collect_cpu = NativeV2Tests.collect_cpu

    def assert_zero_targets(self, raw, runtime):
        self.assertEqual(raw['status'], 'failed', raw['errors'])
        self.assertEqual(len(runtime.enqueued), 18)
        self.assertEqual(len(raw['request_records']), 181)
        self.assertTrue(all(r['status'] == 'not_started' for r in raw['request_records']))
        self.assertTrue(runtime.closed)

    def test_audited_durable_seal_return_at_origin_sends_zero_targets(self):
        self.item = self.manifest['attempts'][0]
        original = typed.seal
        def late(*args, **kwargs):
            result = original(*args, **kwargs)
            kwargs['clock_ns'].__self__.wait(checked(result)['forecast_origin_ns'])
            return result
        with patch.object(typed, 'seal', side_effect=late):
            raw, runtime, _ = self.collect_cpu()
        self.assert_zero_targets(raw, runtime)
        self.assertFalse(raw['predictions_durable_before_cohort'])

    def test_acceptance_checkpoint_crossing_origin_sends_zero_targets(self):
        self.item = self.manifest['attempts'][1]
        original_seal, original_save = typed.seal, m.atomic_save
        clock_holder = {}
        def seal(*args, **kwargs):
            clock_holder['clock'] = kwargs['clock_ns'].__self__
            return original_seal(*args, **kwargs)
        def save(path, value):
            result = original_save(path, value)
            if value.get('checkpoint_context', {}).get('kind') == 'precohort_prediction_accepted':
                clock_holder['clock'].wait(value['planned_cohort_epoch_ns'])
            return result
        with patch.object(typed, 'seal', side_effect=seal), patch.object(m, 'atomic_save', side_effect=save):
            raw, runtime, _ = self.collect_cpu()
        self.assert_zero_targets(raw, runtime)
        self.assertTrue(any('checkpoint crossed' in error for error in raw['errors']))

    def test_prefix_change_restore_after_seal_acceptance_sends_zero_targets(self):
        self.item = self.manifest['attempts'][2]
        original = typed.audit_seal_structure
        def changed(*args, **kwargs):
            value = original(*args, **kwargs)
            path = Path(value['prefix']['path'])
            content = path.read_bytes()
            path.write_bytes(content + b' ')
            path.write_bytes(content)
            return value
        with patch.object(typed, 'audit_seal_structure', side_effect=changed):
            raw, runtime, _ = self.collect_cpu()
        self.assert_zero_targets(raw, runtime)
        self.assertTrue(raw['errors'])


if __name__ == '__main__':
    unittest.main()
