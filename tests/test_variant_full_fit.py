"""Full-fit evidence must cover every training flow and frozen OCR numerics."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
from tests.test_duration_variant_fit import VariantFitTests
from tools.predictor_validation_20261027 import variant_full_fit as full
from tools.predictor_validation_20261027.background_profiles import receipt
from tools.predictor_validation_20261027.flow_fit import read
from tools.predictor_validation_20261027.profiling_analysis import digest


class VariantFullFitTests(unittest.TestCase):
    def setUp(self):
        f=VariantFitTests();f.setUp();self.addCleanup(f.doCleanups)
        self.f=f;self.root=f.root;self.selection=self.root/'winner.json'
        f.fixture.fixture.write(self.selection,{'selected_variant':'ocr_handler'})
        self.verification={'selection':receipt(self.selection),
            'recursive_source_receipts':[receipt(f.selection),receipt(f.fixture.addendum)],
            'selected_variant':'ocr_handler','original_training_cv_thresholds_passed':False}
        self.mock=patch.object(full,'verify_selection',return_value=self.verification)
        self.mock.start();self.addCleanup(self.mock.stop)

    def fit(self):
        protocol=self.root/'full-protocol.json'
        full.prepare_protocol(self.f.protocol,self.selection,protocol)
        return full.fit_full_protocol(protocol,self.root/'full')

    def test_all_training_flows_preserve_failed_cv_and_numeric_identity(self):
        manifest=self.fit();result=full.verify_variant_full_model(manifest['model']['path'])
        self.assertTrue(result['variant_full_model_verified'])
        self.assertEqual(result['variant'],'ocr_handler')
        self.assertEqual(result['training_flow_count'],8)
        self.assertEqual(result['omitted_flow_ids'],[])
        self.assertFalse(manifest['original_training_cv_thresholds_passed'])
        with self.assertRaises(FileExistsError):
            full.fit_full_protocol(self.root/'full-protocol.json',self.root/'full')

    def test_omitted_flow_and_changed_variant_are_rejected(self):
        manifest=self.fit();path=Path(manifest['model']['path']);original=read(path)
        for alter in ('variant','omitted'):
            model=copy.deepcopy(original)
            if alter=='variant':model['variant']='phase_ridge1'
            else:model['training_evidence_tree']['omitted_seed']=1
            model.pop('model_sha256');model['model_sha256']=digest(model)
            self.f.fixture.fixture.write(path,model)
            with self.assertRaises(ValueError):full.verify_variant_full_model(path)

    def test_unselected_variant_rejected_before_protocol(self):
        self.mock.stop()
        self.f.fixture.fixture.write(self.selection,{'schema':'gsdeploy.duration_variants_training_cv_selection_result.v1',
            'selected_variant':'phase_ridge1','independent_validation':False,'final_gain_eligible':False,
            'original_training_cv_thresholds_passed':False,'source_receipts':[receipt(self.f.selection)]})
        with self.assertRaisesRegex(ValueError,'OCR variant winner'):
            full.prepare_protocol(self.f.protocol,self.selection,self.root/'no.json')
        self.assertFalse((self.root/'no.json').exists())


if __name__=='__main__':unittest.main()
