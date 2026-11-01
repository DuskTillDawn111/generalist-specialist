import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.historical_evidence_v4 import SCHEMA, receipt, resolve_closure, HistoricalReceiptResolver


class HistoricalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / 'executor.py'
        self.old.write_text('original source\n')
        self.original = receipt(self.old)
        self.snapshot = self.root / 'snapshot.py'
        self.snapshot.write_bytes(self.old.read_bytes())
        self.old.write_text('current source\n')
        self.model = self.write('model.json', dict(source=self.original))
        self.manifest = self.write('resolution.json', dict(schema=SCHEMA, historical_artifacts_rewritten=False,
            mappings=[dict(historical=self.original, snapshot=receipt(self.snapshot))]))

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def test_original_bytes_are_bound_without_rewriting_history_or_current_code(self):
        before = self.model.read_bytes()
        result = resolve_closure([self.model], self.manifest)
        self.assertIn(receipt(self.snapshot), result['materialized_sources'])
        self.assertNotIn(self.original, result['materialized_sources'])
        self.assertEqual(result['historical_substitutions'][0]['historical'], self.original)
        self.assertEqual(self.model.read_bytes(), before)
        self.assertEqual(self.old.read_text(), 'current source\n')
        self.assertFalse(result['current_execution_equivalence_claimed'])

    def test_changed_source_without_snapshot_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'without an exact'):
            resolve_closure([self.model])

    def test_snapshot_with_different_bytes_cannot_satisfy_old_receipt(self):
        manifest = json.loads(self.manifest.read_text())
        manifest['mappings'][0]['snapshot'] = receipt(self.old)
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'exact original bytes'):
            resolve_closure([self.model], self.manifest)

    def test_changed_snapshot_is_rejected_after_resolver_construction(self):
        resolver = HistoricalReceiptResolver(self.manifest)
        self.snapshot.write_text('corrupted\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            resolver.resolve(self.original)

    def test_two_historical_versions_of_one_path_keep_separate_snapshots(self):
        second = receipt(self.old)
        second_snapshot = self.root / 'snapshot2.py'
        second_snapshot.write_bytes(self.old.read_bytes())
        manifest = json.loads(self.manifest.read_text())
        manifest['mappings'].append(dict(historical=second, snapshot=receipt(second_snapshot)))
        self.manifest.write_text(json.dumps(manifest))
        model = self.write('model2.json', dict(sources=[self.original, second]))
        self.old.write_text('third version\n')
        result = resolve_closure([model], self.manifest)
        self.assertEqual(len(result['historical_substitutions']), 2)
        self.assertIn(receipt(second_snapshot), result['materialized_sources'])


if __name__ == '__main__':
    unittest.main()
