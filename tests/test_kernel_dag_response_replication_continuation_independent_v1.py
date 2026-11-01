"""Independent predecessor, capacity-watch and actual preparation checks."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import kernel_dag_response_replication_continuation_v1 as m


class IndependentContinuationTests(unittest.TestCase):
    def test_actual_preparation_binds_root_cpu04_review_and_same_matching_writer(self):
        path = 'artifacts/predictor-kernel-dag-response-replication-continuation-v1-20261029-cpu01-prepared/manifest.json'
        manifest, campaign = m.audit(path)
        actual = m.queue_identity(manifest['queue_identity']['pid'], manifest['queue_claim'], manifest['matching_final_path'])
        self.assertEqual(actual, manifest['queue_identity'])
        self.assertEqual(len(campaign['runs']), 3)
        self.assertIn('cpu05-independent', manifest['independent_review']['path'])
        self.assertFalse(Path(manifest['output']).exists())
        self.assertFalse(manifest['requires_matching_fit_success'])

    def test_capacity_wait_catches_source_change_restore_before_any_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frozen'; path.write_text('frozen')
            ref = m.receipt(path); manifest = dict(output=str(Path(directory)/'run-v1'))
            def pause(_):
                content = path.read_bytes(); path.write_bytes(content+b'changed'); path.write_bytes(content)
            with m.SourceChangeGuard([ref]) as guard:
                with self.assertRaises(ValueError):
                    m.wait_capacity(manifest, 100, guard.check, phase='native', poll_seconds=1,
                        pause=pause, disk_usage=lambda _:types.SimpleNamespace(free=99))
            self.assertEqual(m.receipt(path), ref)
            self.assertFalse(Path(manifest['output']).exists())

    def test_queue_complete_cannot_name_another_native_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); native=root/'native.json'; native.write_text('{}')
            foreign=root/'foreign.json'; foreign.write_text('{"foreign":true}')
            manifest=dict(matching_final_path=str(native),queue_final_path=str(root/'queue.json'))
            campaign=dict(after_matching_manifest={'path':'pinned'})
            result=dict(schema=m.QUEUE_SCHEMA,status=m.QUEUE_STATUS,
                matching_manifest=campaign['after_matching_manifest'],matching_report=m.receipt(foreign))
            Path(manifest['queue_final_path']).write_text(json.dumps(result))
            with patch.object(m.producer,'require_matching_completion'):
                with self.assertRaisesRegex(ValueError,'queue final'):m.complete_matching(manifest,campaign)


if __name__=='__main__':unittest.main()
