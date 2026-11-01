"""Four new whole-seed fits; baseline remains the original frozen receipt."""
import copy
import unittest
from tests import test_duration_fit as fixtures
from tools.predictor_validation_20261027 import duration_variant_fit as variant
from tools.predictor_validation_20261027.background_profiles import receipt
from tools.predictor_validation_20261027.flow_fit import read
from tools.predictor_validation_20261027.profiling_analysis import digest


class VariantFitTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.DurationFitTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;self.root=f.root;self.selection=self.root/'variant-selection.json'
        value={'schema':'gsdeploy.duration_variants_training_cv_selection_protocol.v1',
            'original_training_protocol':receipt(f.original),'duration_fit_protocol':receipt(f.protocol),
            'baseline_seed_manifest':receipt(f.selection),'prior_selection_protocol':receipt(f.selection),
            'original_threshold_addendum':receipt(f.addendum),'no_independent_results_read':True,
            'candidates':[{'id':v} for v in ['duration_v12','ocr_handler','phase_ridge1']]}
        value['protocol_sha256']=digest(value);f.fixture.write(self.selection,value)
        self.protocol=self.root/'variant-fit-protocol.json'
        variant.prepare_protocol(f.original,self.selection,self.protocol)

    def test_four_fits_have_whole_seed_exclusion_new_definitions_and_sources(self):
        out=self.root/'variants';value=variant.fit_protocol(self.protocol,out)
        self.assertEqual(value['new_whole_seed_refits'],4);self.assertEqual(value['baseline_refits'],0)
        for item in value['variants']:
            manifest=read(item['manifest']['path']);self.assertEqual(manifest['variant'],item['variant'])
            self.assertEqual(manifest['variant_protocol'],receipt(self.protocol))
            self.assertEqual(manifest['selection_protocol'],receipt(self.selection))
            for fold in manifest['folds']:
                checked=variant.verify_variant_model(fold['model']['path'])
                self.assertTrue(checked['variant_model_verified']);self.assertEqual(checked['variant'],item['variant'])
                self.assertEqual(checked['training_flow_count'],4);self.assertEqual(len(checked['omitted_flow_ids']),4)
                self.assertTrue(all(h['heldout_raw'] not in checked['training_raw_receipts'] for h in fold['heldout_flows']))
        with self.assertRaises(FileExistsError):variant.fit_protocol(self.protocol,out)

    def test_changed_candidate_plan_rejected_before_output(self):
        selection=read(self.selection);selection['candidates'].reverse();self.fixture.fixture.write(self.selection,selection)
        with self.assertRaises(ValueError):variant.fit_protocol(self.protocol,self.root/'bad')
        self.assertFalse((self.root/'bad').exists())

    def test_variant_relabelling_is_rejected(self):
        out=self.root/'variants';value=variant.fit_protocol(self.protocol,out)
        m=read(value['variants'][0]['manifest']['path']);path=m['folds'][0]['model']['path'];model=read(path)
        model['variant']='phase_ridge1';model.pop('model_sha256');model['model_sha256']=digest(model)
        self.fixture.fixture.write(path,model)
        with self.assertRaisesRegex(ValueError,'mislabelled'):variant.verify_variant_model(path)


if __name__=='__main__':unittest.main()
