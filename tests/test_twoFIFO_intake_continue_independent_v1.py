"""Independent CPU-only negative tests; no collector/model is launched."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_training_intake_v3 as intake_fixture
from tests import test_twoFIFO_campaign_continue_v1 as driver_fixture
from tools.predictor_validation_20261027 import twoFIFO_training_intake_v3 as intake
from tools.predictor_validation_20261027 import twoFIFO_campaign_continue_v1 as driver


class IntakeIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):intake_fixture.IntakeTests.setUpClass()

    def test_incomplete_metadata_with_poison_raw_path_never_opens_raw_or_extracts(self):
        helper=intake_fixture.IntakeTests()
        with tempfile.TemporaryDirectory() as directory:
            paths=helper.make_reviews(directory,count=1)
            value=json.loads(Path(paths[0]).read_text())
            value['raw']={'path':'/this/raw/must/not/be/opened','sha256':'0'*64}
            Path(paths[0]).write_text(json.dumps(value))
            with patch.object(intake.campaign_v2,'audit',return_value=helper.campaign),patch.object(intake.macros_v2,'extract') as extract:
                result=intake.checked(intake.run(intake_fixture.CAMPAIGN,paths,Path(directory)/'report'))
            extract.assert_not_called();self.assertTrue(result['status'].startswith('WAITING'))
            self.assertEqual(result['response_documents_extracted'],0)

    def test_duplicate_review_and_false_same_phase_count_rejected(self):
        helper=intake_fixture.IntakeTests()
        with tempfile.TemporaryDirectory() as directory:
            paths=helper.make_reviews(directory,count=1);ref=intake.receipt(paths[0]);value=intake.checked(ref)
            with self.assertRaises(ValueError):intake.planned_intake(helper.plans,[(ref,value),(ref,value)])
        doc=copy.deepcopy(helper.docs[0]);piece=doc['requests'][0]['macros'][0]['causal_covariate_path'][0]
        piece['state'][intake.phase.SAME]=1
        with self.assertRaises(ValueError):intake.prepared_rows([doc],intake.phase.DESIGNS['global_phase_aware'])


class ContinueIndependentTests(unittest.TestCase):
    def test_execute_without_final_receipt_stops_before_any_review_or_next_attempt(self):
        helper=driver_fixture.ContinueTests()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,manifest=helper.setup_campaign(root)
            with patch.object(driver,'audit',return_value=manifest),patch.object(driver.shutil,'disk_usage',return_value=type('Disk',(),dict(free=100*2**30))()),patch.object(driver.collector,'execute') as execute,patch.object(driver,'review') as review:
                result=driver.run(path,root/'run',execute=True)
            execute.assert_called_once();review.assert_not_called()
            self.assertEqual(result['status'],'STOPPED_NO_RETRY')

    def test_changed_existing_review_or_macro_stops_before_next_attempt(self):
        helper=driver_fixture.ContinueTests()
        for which in ('review','macro'):
            with self.subTest(which=which),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);path,manifest=helper.setup_campaign(root);item=manifest['attempts'][0]
                driver_fixture.write(Path(item['collection'])/'execution.json',helper.execution(item))
                native=dict(status=driver.COMPLETE,training_feature_extraction_eligible=True,issues=[])
                prior=copy.deepcopy(native)
                if which=='review':prior['changed']='receipt'
                driver_fixture.write(Path(item['existing_review']),prior)
                old=dict(requests=[],macros=1);driver_fixture.write(Path(item['existing_macros']),old)
                report=root/'new-review.json';driver_fixture.write(report,native)
                def extract(path):return dict(requests=[],macros=2)
                with patch.object(driver,'audit',return_value=manifest),patch.object(driver.collector,'execute') as execute,patch.object(driver,'review',return_value=driver.receipt(report)),patch.object(driver.macros,'extract',side_effect=extract):
                    result=driver.run(path,root/'run',execute=True)
                execute.assert_not_called();self.assertEqual(result['status'],'STOPPED_NO_RETRY')


if __name__=='__main__':unittest.main()
