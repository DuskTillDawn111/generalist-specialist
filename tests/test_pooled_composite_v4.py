import json
from pathlib import Path
import tempfile
import unittest

from tests.test_forecast_gate_v3 import fixtures
from tests.test_pooled_lifecycle_v4 import observations
from tools.predictor_validation_20261027.forecast_gate_v3 import execute_bundle
from tools.predictor_validation_20261027.pooled_composite_v4 import derive
from tools.predictor_validation_20261027.pooled_lifecycle_v4 import fit
from tools.predictor_validation_20261027.native_collect import receipt


class PooledCompositeTests(unittest.TestCase):
    def adapted(self,root):
        bundle,public,save,_=fixtures(root)
        rows=observations()
        for row in rows:row.update(family='shared-family',execution_context_sha256='natural')
        model=fit(rows,'natural',allow_cpu_fixture=True)
        model['sources']=[bundle['artifacts']['lifecycle']]
        pooled=save('pooled.json',model)
        old_ref=bundle['artifacts']['composite'];original=Path(old_ref['path']).read_bytes()
        contract=derive(old_ref['path'],pooled['path'],root/'derived',allow_cpu_fixture=True)
        bundle['artifacts'].update(lifecycle=pooled,composite=receipt(root/'derived/contract.json'))
        bundle.update(lifecycle_provider_kind='pooled_joint_mean_hypothesis_v4',allow_pooling_hypothesis=True)
        return bundle,public,contract,old_ref,original

    def test_new_model_identity_preserves_original_target_and_old_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,_,new,old_ref,old_bytes=self.adapted(Path(temp));old=json.loads(old_bytes)
            self.assertEqual(Path(old_ref['path']).read_bytes(),old_bytes)
            self.assertNotEqual(new['composite_execution_sha256'],old['composite_execution_sha256'])
            self.assertEqual(new['target'],old['target'])
            self.assertEqual(new['components']['service'],old['components']['service'])
            self.assertEqual(new['components']['lifecycle']['original_context_sha256'],'natural')
            self.assertEqual(new['components']['lifecycle']['model'],bundle['artifacts']['lifecycle'])

    def test_new_provider_actually_runs_through_gate_and_original_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,public,contract,_,_=self.adapted(Path(temp))
            result=execute_bundle(bundle,public,allow_cpu_fixture=True)
            self.assertEqual(result['forecast']['finishes']['a']['finish_ns'],213000000)
            self.assertEqual(result['model_composite_execution_sha256'],contract['composite_execution_sha256'])
            self.assertEqual(result['metrics']['terms']['Q'],.1)
            self.assertIsNone(result['metrics']['full_U'])
            bundle['allow_pooling_hypothesis']=False
            with self.assertRaisesRegex(ValueError,'opt-in'):execute_bundle(bundle,public,allow_cpu_fixture=True)

    def test_changed_component_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle,_,save,_=fixtures(root)
            rows=observations();model=fit(rows,'same-source-bound-execution',allow_cpu_fixture=True)
            model['sources']=[bundle['artifacts']['lifecycle']];ref=save('other-context.json',model)
            with self.assertRaisesRegex(ValueError,'original component execution identity'):
                derive(bundle['artifacts']['composite']['path'],ref['path'],root/'bad',allow_cpu_fixture=True)


if __name__=='__main__':unittest.main()
