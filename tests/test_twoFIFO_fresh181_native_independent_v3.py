"""Independent v3 deadline/identity controls; CPU law sources explicitly mocked."""
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_fresh181_native_v3_collector import NativeV3CollectorTests
from tools.predictor_validation_20261027 import twoFIFO_fresh181_native_v3 as m
from tools.predictor_validation_20261027 import twoFIFO_matching_prospective_bridge_v3 as typed


class IndependentNativeV3Tests(unittest.TestCase):
    setUpClass = classmethod(NativeV3CollectorTests.setUpClass.__func__)
    tearDownClass = classmethod(NativeV3CollectorTests.tearDownClass.__func__)
    authentication = NativeV3CollectorTests.authentication
    collect_cpu = NativeV3CollectorTests.collect_cpu

    def zero_targets(self, raw, runtime):
        self.assertEqual(raw['status'], 'failed', raw['errors'])
        self.assertEqual(len(runtime.enqueued), 18)
        self.assertEqual(len(raw['request_records']), 181)
        self.assertTrue(all(row['status']=='not_started' for row in raw['request_records']))
        self.assertTrue(runtime.closed)

    def test_new_native_entry_cannot_accept_injected_transport_or_caller_sessions(self):
        with patch.object(m.supervisor, 'validate_lease'), patch.object(m, 'audit_plan', return_value={}), \
                patch.object(m, 'require_native_ready'), patch.object(typed, 'open_target_session') as session:
            with self.assertRaisesRegex(ValueError, 'cannot inject'):
                m.collect('unused', 'unused', lease_fd=-1, runtime_factory=object, sessions=(object(), object()))
            session.assert_not_called()

    def test_v3_late_producer_retains_all181_and_sends_zero(self):
        self.item = self.manifest['attempts'][0]
        raw, runtime, _ = self.collect_cpu(late=True)
        self.zero_targets(raw, runtime)
        self.assertFalse(raw['predictions_durable_before_cohort'])

    def test_v3_producer_session_cannot_audit_itself(self):
        self.item = self.manifest['attempts'][1]
        raw, runtime, _ = self.collect_cpu(same_session=True)
        self.zero_targets(raw, runtime)
        self.assertTrue(any('separately authenticated' in error for error in raw['errors']))

    def test_v3_prefix_change_restore_after_acceptance_still_blocks_cohort(self):
        self.item = self.manifest['attempts'][2]
        original = typed.audit_seal_structure
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            path = Path(result['prefix']['path']); content = path.read_bytes()
            path.write_bytes(content+b' '); path.write_bytes(content)
            return result
        with patch.object(typed, 'audit_seal_structure', side_effect=mutate):
            raw, runtime, _ = self.collect_cpu()
        self.zero_targets(raw, runtime)


if __name__=='__main__': unittest.main()
