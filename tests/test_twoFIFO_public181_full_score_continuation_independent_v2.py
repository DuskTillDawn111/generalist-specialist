"""Independent late-output authentication counterexamples; no GPU or fitting."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_public181_full_score_continuation_v2 as m


class LateArtifactTests(unittest.TestCase):
    def scenario(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(name, value):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
                return m.receipt(path)
            source = write('source.json', {'CPU_TEST_ONLY': True})
            public = write('public.json', {'CPU_TEST_ONLY': True})
            manifest = dict(output=str(root/'run'), public_preparation=public)
            manifest_ref = write('manifest.json', manifest)
            inputs = dict(matching=dict(artifacts=[source], documents=[], fit=source),
                          resource=dict(artifacts=[source], documents=[], resource_manifest=source, reviews=[source]*8))
            invocations = []
            artifact = []
            def execute(command, **kwargs):
                invocations.append(command)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                output = Path(command[command.index('--output')+1])
                artifact.append(write(str((output/'timing.json').relative_to(root)), {'duration': 1}))
                write(str((output/'report.json').relative_to(root)), {'timing': artifact[0]})
                return SimpleNamespace(returncode=0)
            def restore(path):
                path = Path(path)
                original = path.read_bytes()
                path.write_bytes(original+b' ')
                path.write_bytes(original)
            def audit_budget(*args, **kwargs):
                if mode == 'declared_artifact_after_audit':
                    restore(artifact[0]['path'])
                return dict(fits_predeclared_budget=True, public_preparation=public)
            atomic = m.atomic_save
            def save(path, value):
                atomic(path, value)
                if mode == 'eligibility_after_success_write' and value.get('budget_qualified') is True:
                    restore(value['eligibility']['path'])
            with patch.object(m, 'audit', return_value=manifest), \
                 patch.object(m, 'wait_inputs', return_value=inputs), \
                 patch.object(m, 'qualification', return_value=dict(eligible=True, reasons=[])), \
                 patch.object(m.budget, 'audit', side_effect=audit_budget), \
                 patch.object(m, 'atomic_save', side_effect=save):
                result = m.checked(m.run(manifest_ref['path'], run_command=execute))
            self.assertEqual(len(invocations), 1)
            self.assertEqual(result['status'], 'COMPLETE_HOLD_SOURCE_OR_CPU_STAGE_FAILURE_NO_RETRY')
            self.assertFalse(result['full_score_budget_ready'])
            self.assertFalse(result['budget_qualified'])
            self.assertIsNone(result['fits_predeclared_budget'])
            self.assertNotIn('budget', result)
            self.assertNotIn('eligibility', result)
            self.assertFalse(result['automatic_retry'])
            return result

    def test_declared_timing_artifact_change_restore_is_not_hidden_by_unchanged_budget_report(self):
        self.scenario('declared_artifact_after_audit')

    def test_eligibility_change_restore_after_success_write_revokes_success_receipt(self):
        result = self.scenario('eligibility_after_success_write')
        self.assertTrue(result['unauthenticated_diagnostics']['previous_outputs']['fits_predeclared_budget'])


if __name__ == '__main__':
    unittest.main()
