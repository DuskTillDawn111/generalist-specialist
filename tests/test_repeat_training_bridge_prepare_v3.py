import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.native_collect import context_identity,receipt
from tools.predictor_validation_20261027.repeat_training_bridge_prepare_v3 import relocated_context


class RepeatPreparationTests(unittest.TestCase):
    def test_relocation_preserves_context_and_original_resource_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'a.py';source.write_text('example')
            frozen=receipt(source)
            context=dict(execution_sources=[dict(original_path='/original/a.py',frozen=frozen)],sources=[frozen],
                         large_weight_stats=dict(path='/stable/weight-stats.json',sha256='verified'))
            repeated=relocated_context(context,root/'repeat')
            self.assertEqual(context_identity(context),context_identity(repeated))
            self.assertNotEqual(context['sources'][0]['path'],repeated['sources'][0]['path'])
            self.assertEqual(context['large_weight_stats'],repeated['large_weight_stats'])
            self.assertEqual(context['sources'][0]['path'],str(source))

    def test_changed_reference_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'a.py';source.write_text('first');frozen=receipt(source)
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Changed reference'):
                relocated_context(dict(execution_sources=[dict(original_path='/a.py',frozen=frozen)]),root/'repeat')


if __name__=='__main__':unittest.main()
