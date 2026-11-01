"""Independent V4 authorization/source CPU regressions; no GPU."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tests import test_admission_validation_continuation_v1 as fixtures
from tools.predictor_validation_20261027 import admission_validation_continuation_v1 as m


class IndependentV4ContinuationTests(unittest.TestCase):
    setUp=fixtures.AdmissionValidationContinuationTests.setUp
    prepare=fixtures.AdmissionValidationContinuationTests.prepare
    review=fixtures.AdmissionValidationContinuationTests.review
    complete_training=fixtures.AdmissionValidationContinuationTests.complete_training
    fake_seal=fixtures.AdmissionValidationContinuationTests.fake_seal

    def test_completed_file_cannot_hide_recycled_original_T8_identity(self):
        protocol=self.prepare();m.save(self.paths['training_completed'],{'cpu_fixture':True})
        changed=copy.deepcopy(m.expected_identity());changed['start_ticks']+=1
        self.identity.return_value=changed
        with self.assertRaisesRegex(ValueError,'identity changed'):m.upstream_status(protocol)

    def test_root_review_drift_during_CPU_seal_stops_before_GPU(self):
        self.prepare();self.review();self.complete_training()
        def seal(*args):
            self.fake_seal(*args)
            value=m.read(self.review_path);value['rationale']='Changed after launch; same schema fields'
            self.review_path.write_text(__import__('json').dumps(value))
        with patch.object(m,'wait_upstream'),patch.object(m.validation,'seal',side_effect=seal),\
                patch.object(m.runner,'run') as launch:
            with self.assertRaises(ValueError):m.run(self.path,self.review_path)
            launch.assert_not_called()
        self.assertTrue((self.output/'failed.json').exists())

    def test_training_native_source_drift_after_CPU_seal_stops_before_GPU(self):
        self.prepare();self.review();_,jobs=self.complete_training()
        def seal(*args):
            self.fake_seal(*args)
            (Path(jobs[0]['output'])/'raw.json').write_text('{"changed":true}')
        with patch.object(m,'wait_upstream'),patch.object(m.validation,'seal',side_effect=seal),\
                patch.object(m.runner,'run') as launch:
            with self.assertRaises(ValueError):m.run(self.path,self.review_path)
            launch.assert_not_called()
        self.assertTrue((self.output/'failed.json').exists())


if __name__=='__main__':unittest.main()
