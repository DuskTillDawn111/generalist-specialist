"""New run entry exercises the OCR checker and stops before GPU on bad inputs."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from tests import test_ocr_trained_validation as fixtures
from tools.predictor_validation_20261027 import ocr_validation_run as run
from tools.predictor_validation_20261027 import ocr_trained_validation as entry
from tools.predictor_validation_20261027.native_flow import receipt
from tools.predictor_validation_20261027.profiling_analysis import digest


class OcrRunTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.OcrValidationTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture.fixture;self.f=f;self.root=f.root
        self.plan=f.plan;self.plan['seed']=f.plan.get('seed',1);f.write('plan.json',self.plan)
        with self.fixture._verifier(),patch.object(entry,'audit_plan',return_value=f.plan):
            entry.seal(f.plan_path,f.model_path,f.resources_path,f.quality_path,f.root/'sealed')
        sealed=run.read(self.root/'sealed/predictions.json')
        self.qualification=f.write('qualification.json',{'target_plan':receipt(f.plan_path),
            'target_seed':self.plan['seed'],'created_ns':sealed['created_ns']})
        self.patches=[self.fixture._verifier(),patch.object(entry,'audit_plan',return_value=self.plan),
            patch.object(run,'audit_plan',return_value=self.plan),
            patch.object(run,'verify_protocol',side_effect=lambda path:run.read(path)),
            patch('tools.predictor_validation_20261027.flow_suite_audit.condition',return_value={}),
            patch('tools.predictor_validation_20261027.flow_suite_audit._generation',return_value={})]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def prepare(self):
        return run.prepare(self.f.plan_path,self.root/'sealed',self.qualification,self.root/'suite')

    def test_real_new_checker_supports_prepare_and_prospective_audit_without_old_entry(self):
        with patch('tools.predictor_validation_20261027.flow_trained_validation.check_sealed',side_effect=AssertionError('old entry called')):
            suite=self.prepare();audit=run.audit_inputs(self.root/'suite/suite.json')
        self.assertEqual(audit['status'],'PASS');self.assertFalse(audit['runtime_paths_opened'])
        self.assertEqual(audit['training_verification']['variant'],'ocr_handler')
        self.assertTrue(suite['no_retries']);self.assertTrue(suite['no_seed_replacement'])

    def test_empty_predictions_cannot_prepare_a_suite(self):
        path=self.root/'sealed/predictions.json';sealed=run.read(path);sealed['candidates'][0]['requests']=[]
        self.f.write('sealed/predictions.json',sealed)
        # Keep the audit's prediction receipt current to isolate full cohort checking.
        a=run.read(self.root/'sealed/training-audit.json');a['predictions']=receipt(path);self.f.write('sealed/training-audit.json',a)
        with self.assertRaisesRegex(ValueError,'complete unique offered cohort'):self.prepare()
        self.assertFalse((self.root/'suite').exists())

    def test_pre_run_wrong_source_stops_before_execute(self):
        self.prepare();path=self.root/'suite/suite.json';suite=run.read(path)
        suite['sources']['source']=receipt(self.f.plan_path);self.f.write('suite/suite.json',suite)
        with patch.object(run,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'generator source'):run.run(path)
            execute.assert_not_called()

    def test_first_native_failure_stops_without_retry(self):
        self.prepare()
        with patch.object(run,'execute',return_value={'status':'failed','qualification_issues':[]}) as execute:
            with self.assertRaisesRegex(RuntimeError,'without retry'):run.run(self.root/'suite/suite.json')
        self.assertEqual(execute.call_count,1)


if __name__=='__main__':unittest.main()
