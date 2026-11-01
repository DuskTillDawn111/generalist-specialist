"""Provider construction rejects edits to nested reviewers and training data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import natural_resources_v6 as tool


class SealedResourceEvidenceTests(unittest.TestCase):
    def make_model(self, root):
        def write(name, value):
            p = root / name
            p.write_text(json.dumps(value))
            return tool.receipt(p)
        self.evaluator = root / 'reviewer.py'
        self.evaluator.write_text('original evaluator\n')
        row = dict(split='training', status='completed', request_id='r', pool_id='p', task='text')
        raw = write('raw.json', dict(request_records=[row]))
        review = write('review.json', dict(raw=raw, evaluator_sources=[tool.receipt(self.evaluator)]))
        run = 'raw-sha256:' + raw['sha256']
        self.index = root / 'index.json'
        index = write('index.json', dict(schema='gsdeploy.independent_prediction_gate_training_index.v2',
                split='training', records=[dict(run_id=run, request_id='r', pool_id='p', task='text')]))
        model = dict(schema=tool.original.SCHEMA, provenance=dict(kind='measured', split='training'),
            training_index=index, captures=[dict(raw=raw, run_id=run)],
            native_projection_sha256='native', states={}, sources=[review])
        self.base = write('base.json', model)
        return tool.seal(self.base['path'], root / 'sealed')

    def provider(self, model):
        with patch.object(tool.original, 'native_execution_projection', return_value=dict(
                projection_sha256='native', original_full_context_sha256='full')):
            return tool.NaturalResourceProvider(model, {}, allow_conditional_resource_hypothesis=True)

    def test_nested_reviewer_is_checked_again_when_provider_loads(self):
        with tempfile.TemporaryDirectory() as temp:
            model = self.make_model(Path(temp))
            self.provider(model)
            self.evaluator.write_text('changed evaluator\n')
            with self.assertRaisesRegex(ValueError, 'Changed receipt'):
                self.provider(model)

    def test_training_index_receipt_is_checked_again(self):
        with tempfile.TemporaryDirectory() as temp:
            model = self.make_model(Path(temp))
            self.index.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Changed resource evidence'):
                self.provider(model)

    def test_edited_estimates_in_sealed_wrapper_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            model = self.make_model(Path(temp))
            changed = copy.deepcopy(model)
            changed['states'] = {'invented': dict(known=True)}
            with self.assertRaisesRegex(ValueError, 'differs from original'):
                self.provider(changed)

    def test_index_cannot_omit_training_request_even_with_correct_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            self.make_model(Path(temp))
            model = json.loads(Path(self.base['path']).read_text())
            index = json.loads(self.index.read_text())
            index['records'] = []
            self.index.write_text(json.dumps(index))
            model['training_index'] = tool.receipt(self.index)
            with self.assertRaisesRegex(ValueError, 'exactly cover'):
                tool.verify_training_index(model)

    def test_exact_historical_snapshot_is_explicit_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_model(root)
            before = Path(self.base['path']).read_bytes()
            historical = tool.receipt(self.evaluator)
            snapshot = root / 'snapshot.py'
            snapshot.write_bytes(self.evaluator.read_bytes())
            self.evaluator.write_text('current implementation\n')
            manifest = root / 'resolution.json'
            manifest.write_text(json.dumps(dict(schema='gsdeploy.historical_receipt_snapshot_resolution.v4',
                historical_artifacts_rewritten=False, mappings=[dict(historical=historical, snapshot=tool.receipt(snapshot))])))
            model = tool.seal(self.base['path'], root / 'sealed-history', historical_resolution_path=manifest)
            self.provider(model)
            self.assertEqual(len(model['training_evidence_closure']['historical_substitutions']), 1)
            self.assertEqual(Path(self.base['path']).read_bytes(), before)
            snapshot.write_text('corrupted original\n')
            with self.assertRaisesRegex(ValueError, 'exact original bytes'):
                self.provider(model)


if __name__ == '__main__':
    unittest.main()
