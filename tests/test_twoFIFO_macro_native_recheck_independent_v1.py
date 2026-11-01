"""Independent negative tests for public v2 native-macro extraction.

Fakes below never claim to be native evidence. They test that self-declared
review eligibility cannot skip rederivation or replace an exact report.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_composable_macros_v2 as m


class NativeRecheckIndependentTests(unittest.TestCase):
    def fixture(self,directory):
        directory=Path(directory)
        plan=directory/'plan.json';plan.write_text('{}')
        raw=directory/'raw.json';raw.write_text(json.dumps(dict(backend='CPU_TEST_ONLY',status='completed',plan=m.receipt(plan))))
        execution=directory/'execution.json';execution.write_text('{}')
        value=dict(schema=m.REVIEW_SCHEMA,status='PASS_COMPLETE_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION',
            training_feature_extraction_eligible=True,issues=[],validation_labels_read=False,
            execution=m.receipt(execution),raw=m.receipt(raw),sources=[])
        path=directory/'review.json';path.write_text(json.dumps(value))
        return path,value

    def test_self_declared_eligibility_calls_actual_reviewer_and_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path,value=self.fixture(directory)
            with self.assertRaises(ValueError):m.extract(path)

    def test_source_receipt_change_rejected_before_recheck(self):
        with tempfile.TemporaryDirectory() as directory:
            path,value=self.fixture(directory)
            source=Path(directory)/'source.txt';source.write_text('original')
            value['sources']=[m.receipt(source)];path.write_text(json.dumps(value));source.write_text('changed')
            with patch.object(m,'recheck_review') as recheck:
                with self.assertRaisesRegex(ValueError,'source changed'):m.extract(path)
                recheck.assert_not_called()

    def test_full_rederived_report_must_equal_claim_not_just_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path,value=self.fixture(directory);changed=copy.deepcopy(value);changed['additional_fact']='not in claim'
            def recheck(raw_path,output,execution_path):
                self.assertEqual(raw_path,value['raw']['path']);self.assertEqual(execution_path,value['execution']['path'])
                output.mkdir();target=output/'report.json';target.write_text(json.dumps(changed));return m.receipt(target)
            with patch.object(m,'recheck_review',side_effect=recheck) as called:
                with self.assertRaisesRegex(ValueError,'rederive exactly'):m.extract(path)
                called.assert_called_once()

    def test_even_equal_recheck_does_not_make_CPU_backend_native(self):
        with tempfile.TemporaryDirectory() as directory:
            path,value=self.fixture(directory)
            with patch.object(m,'recheck_review',return_value=m.receipt(path)):
                with self.assertRaisesRegex(ValueError,'Native complete attempt'):m.extract(path)


if __name__=='__main__':unittest.main()
