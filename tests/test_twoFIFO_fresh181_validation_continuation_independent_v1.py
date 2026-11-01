"""Independent ordering checks for the new one-shot validation controller."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_fresh181_validation_continuation_v1 as fixtures
from tools.predictor_validation_20261027 import twoFIFO_fresh181_validation_continuation_v1 as m


class OrderingTests(unittest.TestCase):
    def test_restored_first_review_rejects_second_capture_and_any_partial_aggregate(self):
        fixture = fixtures.ContinuationTests()
        capture = m.capture_one
        def restore_after_first(*args, **kwargs):
            capture(*args, **kwargs)
            index, result = args[1], args[4]
            if index == 0:
                path = Path(result['candidate_outcomes'][0]['review']['path'])
                data = path.read_bytes()
                path.write_bytes(data+b' ')
                path.write_bytes(data)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(m, 'capture_one', side_effect=restore_after_first):
            result, calls, prepares, _ = fixture.fixture(Path(directory))
        self.assertEqual(len(prepares), 1)
        self.assertEqual(len(calls), 2)  # one native stage plus its CPU review
        self.assertEqual(result['native_capture_invocations'], 1)
        self.assertFalse(result['native_validation_complete'])
        self.assertFalse(result['aggregate_source_authenticated'])
        self.assertNotIn('aggregate', result)
        self.assertEqual(result['candidate_denominator'], 4)
        self.assertEqual(result['pair_denominator'], 6)
        self.assertEqual(len(result['candidate_outcomes']), 4)
        self.assertTrue(all(row['offered_requests'] == 181 for row in result['candidate_outcomes']))
        self.assertTrue(all(not row['source_authenticated'] for row in result['candidate_outcomes']))
        self.assertTrue(all(row['empirical_outcome'] == 'UNKNOWN_UNAUTHENTICATED'
                            for row in result['candidate_outcomes']))

    def test_capacity_guard_failure_precedes_started_marker_and_every_native_command(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as watches:
            root = Path(directory)
            item = dict(plan={'path': str(root/'plan.json')}, claim=str(root/'claim.json'),
                        collection=str(root/'collection'))
            result = dict(candidate_outcomes=[dict(status='NOT_STARTED')],
                          native_capture_invocations=0, GPU_started=False)
            with patch.object(m.native, 'audit_plan', return_value={}), \
                 patch.object(m.native, 'require_native_ready'), \
                 patch.object(m, 'wait_capacity', side_effect=ValueError('source changed while waiting')), \
                 patch.object(m, 'run_stage') as stage:
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    m.capture_one(root, 0, item, {}, result, watches, lambda: None, lambda *a: None)
                stage.assert_not_called()
            self.assertFalse((root/'capture-0-started.json').exists())
            self.assertFalse((root/'claim.json').exists())
            self.assertEqual(result['candidate_outcomes'][0]['status'], 'NOT_STARTED')
            self.assertEqual(result['native_capture_invocations'], 0)
            self.assertFalse(result['GPU_started'])


if __name__ == '__main__':
    unittest.main()
