"""Independent recovery role/causality checks, without native labels or fitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_recovery_phase12_v1 as m


class IndependentRecoveryPhase12Tests(unittest.TestCase):
    def fixture(self, root):
        def put(name, value):
            path = root / name
            path.write_text(json.dumps(value))
            return m.receipt(path)
        original = put('original.json', {})
        recovery = put('recovery.json', {})
        stopped = put('stopped.json', {})
        prefix = [dict(attempt_id='prefix' + str(i)) for i in range(4)]
        tail = [dict(attempt_id='tail' + str(i)) for i in range(8)]
        final = dict(schema=m.recovery.SCHEMA + '.run', status=m.recovery.FINAL,
            policy=m.recovery.POLICY, recovery_manifest=recovery, original_manifest=original,
            stopped_report=stopped, prefix_attempts=prefix, completed_tail=tail,
            completed_attempts=prefix + tail, no_started_capture_retries=True, fit_eligible=False,
            numerical_fit_run=False, automatic_fit=False, validation_labels_read=False)
        final_ref = put('final.json', final)
        manifest = dict(recovery_manifest=recovery, original_executor_manifest=original,
                        expected_continuation_report_path=final_ref['path'])
        recovered = dict(proof=dict(stopped_report=stopped, prefix=prefix, tail=tail))
        intake = dict(schema=m.intake.SCHEMA, status=m.INTAKE_STATUS,
            recovery_manifest=recovery, manifest=original, continuation=final_ref,
            recovery_policy=m.recovery.POLICY, stopped_report=stopped,
            preserved_prefix_attempts=4, new_policy_tail_attempts=8,
            attempt_count=12, measured_requests=546, priming_requests=216,
            numerical_fit_run=False, validation_labels_read=False, sources=[])
        return manifest, recovered, final, intake

    def test_final_does_not_accept_old_producer_missing_tail_or_reordered_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, recovered, final, _ = self.fixture(Path(directory))
            with patch.object(m.recovery, 'audit', return_value=recovered):
                self.assertTrue(m.complete_final(manifest, final))
                changes = [dict(schema=m.executor.CONT_SCHEMA + '.run'),
                           dict(completed_tail=final['completed_tail'][:-1]),
                           dict(automatic_fit=True), dict(no_started_capture_retries=False),
                           dict(completed_attempts=final['completed_attempts'][::-1])]
                for change in changes:
                    with self.subTest(change=list(change)):
                        self.assertFalse(m.complete_final(manifest, {**final, **change}))

    def test_intake_requires_new_producer_preserved_stop_and_current_final_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, recovered, _, intake = self.fixture(Path(directory))
            with patch.object(m.recovery, 'audit', return_value=recovered):
                m.check_intake(intake, manifest)
                for change in [dict(schema=m.numeric.intake.SCHEMA), dict(measured_requests=545),
                               dict(stopped_report={'path': 'foreign', 'sha256': 'f'*64}),
                               dict(new_policy_tail_attempts=7), dict(numerical_fit_run=True)]:
                    with self.subTest(change=list(change)), self.assertRaises(ValueError):
                        m.check_intake({**intake, **change}, manifest)
                Path(manifest['expected_continuation_report_path']).write_text('{}')
                with self.assertRaisesRegex(ValueError, 'receipt changed'):
                    m.check_intake(intake, manifest)

    def test_callback_failure_still_checks_final_and_manifest_without_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, recovered, final, intake = self.fixture(root)
            path = root / 'manifest.json'
            path.write_text('{}')
            verified = []
            real_verify = m.verify
            def verify(ref):
                verified.append(ref)
                return real_verify(ref)
            def native(*args, output, **kwargs):
                output.mkdir()
                report = output / 'report.json'
                report.write_text(json.dumps(intake))
                return m.receipt(report)
            def callback(bundle, document):
                self.assertEqual(document, intake)
                raise RuntimeError('downstream failed')
            with patch.object(m, 'audit', return_value=manifest), \
                 patch.object(m.recovery, 'audit', return_value=recovered), \
                 patch.object(m.intake, 'intake', side_effect=native), \
                 patch.object(m, 'row_bundle', return_value={'verified': True}), \
                 patch.object(m, 'verify', side_effect=verify), \
                 patch.object(m.numeric, 'fit_bundle', side_effect=AssertionError('fit forbidden')):
                with self.assertRaisesRegex(RuntimeError, 'downstream failed'):
                    m.with_verified_bundle(path, manifest['expected_continuation_report_path'], callback)
            self.assertIn(m.receipt(path), verified)
            self.assertIn(intake['continuation'], verified)
            self.assertFalse((root / 'phase12-fit-claim.json').exists())


if __name__ == '__main__':
    unittest.main()
