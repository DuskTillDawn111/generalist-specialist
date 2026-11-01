"""Source equality permits an explicit collector transfer hypothesis only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.native_queue_busy_replay_v9 import queue_context_bridge
from tools.predictor_validation_20261027.dag_checkpoint_v9 import receipt


class QueueContextBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        native=self.root/'native.txt';native.write_text('CPU source fixture')
        catalog=self.root/'runtime-inputs.jsonl';catalog.write_text('{"pool_id":"fixture","sample":"original"}\n')
        weights=self.root/'weights.json';weights.write_text('[]')
        ref=receipt(native)
        source=dict(shared_native_execution_sources=[ref],worker_bindings={},residents={},threads={},execution={},
            device={},measurement_domain='natural',profiler={},affine_observation={},worker_variant='fixture',
            large_weight_identity_scope='fixture',large_weight_stats=receipt(weights),
            sources=[ref],execution_sources=[dict(original_path=str(native),frozen=ref),
                dict(original_path=str(catalog),frozen=receipt(catalog))])
        path=self.root/'source.json';path.write_text(json.dumps(source));source_ref=receipt(path)
        composite=self.root/'composite.json';composite.write_text(json.dumps(dict(target=dict(context=source_ref))))
        self.bundle=dict(composite=receipt(composite));self.target=copy.deepcopy(source)
        self.target.update(measurement_domain='natural_predeclared_queue_training_v8',queue_training=True,training_only=True)

    def target_ref(self):
        path=self.root/'target.json';path.write_text(json.dumps(self.target));return receipt(path)

    def test_exact_fields_one_domain_difference_is_hypothesis_not_full_equality(self):
        result=queue_context_bridge(self.bundle,self.target_ref())
        self.assertTrue(result['collector_invariance_is_hypothesis'])
        self.assertFalse(result['full_context_equality_claimed'])
        self.assertFalse(result['natural_timing_transfer_validated'])
        self.assertFalse(result['physical_accuracy_validated'])
        self.assertNotEqual(result['source_full_context_sha256'],result['target_full_context_sha256'])

    def test_changed_execution_threads_or_native_source_cannot_use_domain_exception(self):
        for field in ('threads','execution','worker_variant'):
            old=self.target[field];self.target[field]='changed'
            with self.subTest(field=field),self.assertRaisesRegex(UnknownEvidence,'PROJECTION_MISMATCH'):
                queue_context_bridge(self.bundle,self.target_ref())
            self.target[field]=old

    def test_validation_domain_and_changed_source_receipt_rejected(self):
        self.target['measurement_domain']='validation'
        with self.assertRaisesRegex(ValueError,'domains'):queue_context_bridge(self.bundle,self.target_ref())
        self.target['measurement_domain']='natural_predeclared_queue_training_v8'
        self.target['shared_native_execution_sources'][0]['sha256']='0'*64
        with self.assertRaises(ValueError):queue_context_bridge(self.bundle,self.target_ref())

    def test_same_pool_id_changed_public_content_cannot_transfer_exact_template(self):
        replacement=self.root/'changed-runtime-inputs.jsonl'
        replacement.write_text('{"pool_id":"fixture","sample":"changed"}\n')
        self.target['execution_sources'][1]['frozen']=receipt(replacement)
        with self.assertRaisesRegex(UnknownEvidence,'exact_public_runtime_input_catalog_content'):
            queue_context_bridge(self.bundle,self.target_ref())


if __name__=='__main__':unittest.main()
