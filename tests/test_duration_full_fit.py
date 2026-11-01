"""Only a duration winner may produce a fully source-audited full model."""
import copy
import unittest
from pathlib import Path
from tests import test_duration_fit as fixtures
from tools.predictor_validation_20261027 import duration_full_fit as full
from tools.predictor_validation_20261027 import duration_fit
from tools.predictor_validation_20261027.background_profiles import receipt
from tools.predictor_validation_20261027.flow_fit import read
from tools.predictor_validation_20261027.profiling_analysis import digest


class DurationFullFitTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.DurationFitTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.selection=self.root/'selection-result.json'
        self.value={'schema':'gsdeploy.duration_loss_training_cv_selection_result.v1',
            'selected_algorithm':'duration_weighted_relative_work_v12','independent_validation':False,
            'final_gain_eligible':False,'original_training_cv_thresholds_passed':False,
            'source_receipts':[receipt(self.fixture.selection),receipt(self.fixture.addendum)]}
        self.fixture.fixture.write(self.selection,self.value)

    def test_full_model_has_all_eight_flows_and_preserves_cv_failure(self):
        protocol=self.root/'full-protocol.json';full.prepare_protocol(self.fixture.protocol,self.selection,protocol)
        output=self.root/'full';manifest=full.fit_full_protocol(protocol,output)
        verified=full.verify_duration_full_model(manifest['model']['path'])
        self.assertTrue(verified['duration_full_model_verified']);self.assertEqual(verified['training_flow_count'],8)
        self.assertEqual(verified['omitted_flow_ids'],[]);self.assertIsNone(verified['omitted_seed'])
        self.assertFalse(manifest['original_training_cv_thresholds_passed'])
        with self.assertRaises(ValueError):duration_fit.verify_duration_model(manifest['model']['path'])
        with self.assertRaises(FileExistsError):full.fit_full_protocol(protocol,output)

    def test_old_winner_is_rejected_before_full_protocol_or_fit(self):
        value=copy.deepcopy(self.value);value['selected_algorithm']='relative_work_v11'
        self.fixture.fixture.write(self.selection,value);protocol=self.root/'no-fit.json'
        with self.assertRaisesRegex(ValueError,'duration winner'):full.prepare_protocol(self.fixture.protocol,self.selection,protocol)
        self.assertFalse(protocol.exists())

    def test_recursive_selection_source_tampering_is_rejected(self):
        protocol=self.root/'full-protocol.json';full.prepare_protocol(self.fixture.protocol,self.selection,protocol)
        self.fixture.addendum.write_text('{}')
        with self.assertRaises(ValueError):full.fit_full_protocol(protocol,self.root/'bad')
        self.assertFalse((self.root/'bad').exists())

    def test_full_model_cannot_omit_a_flow_or_mislabel_numerics(self):
        protocol=self.root/'full-protocol.json';full.prepare_protocol(self.fixture.protocol,self.selection,protocol)
        output=self.root/'full';manifest=full.fit_full_protocol(protocol,output);path=Path(manifest['model']['path'])
        model=read(path);model['training_evidence_tree']['omitted_seed']=1
        model.pop('model_sha256');model['model_sha256']=digest(model);self.fixture.fixture.write(path,model)
        with self.assertRaises(ValueError):full.verify_duration_full_model(path)


if __name__=='__main__':unittest.main()
