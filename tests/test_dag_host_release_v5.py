import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_dag_host_release_v4 import profile,natural
from tools.predictor_validation_20261027 import dag_host_release_v5 as tool


class NativeCompletionBoundaryTests(unittest.TestCase):
    def test_ocr_return_is_explicit_upper_bound_with_unchanged_sync_adapter_receipt(self):
        p=profile();p.update(task='ocr');row=natural();row['task']='ocr'
        row['worker_response']['task_type']='ocr'
        row['worker_response']['observed_stages']=[dict(id='native_ocr',end_ns=1100)]
        result=tool.fit_observation(p,row)
        self.assertEqual(result['completion_boundary_kind'],'synchronous_native_ocr_call_return_upper_bound')
        self.assertEqual(result['completion_upper_bound_offset_ns'],100)
        self.assertEqual(result['terminal_tail_ns'],7)
        self.assertFalse(result['exact_device_completion_identified'])
        self.assertEqual(len(result['completion_boundary_sources']),1)
        with patch.object(tool,'receipt',return_value=dict(sha256='changed')):
            with self.assertRaisesRegex(UnknownEvidence,'SYNCHRONOUS_OCR_ADAPTER'):tool.natural_boundaries(row)

    def test_native_return_cannot_silently_replace_other_task_sync(self):
        row=natural();row['task']='text';row['worker_response']['task_type']='text'
        row['worker_response']['observed_stages']=[dict(id='native_ocr',end_ns=1100)]
        with self.assertRaisesRegex(UnknownEvidence,'COMPLETION_BOUND'):tool.natural_boundaries(row)


if __name__=='__main__':unittest.main()
