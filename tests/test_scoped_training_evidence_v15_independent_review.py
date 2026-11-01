"""Independent synthetic checks for historical aliases and exact audit edges."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.predictor_validation_20261027.historical_evidence_v4 import SCHEMA as HISTORICAL
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.profiling import digest
from tools.predictor_validation_20261027.scoped_training_evidence_v15 import (
    METADATA_REASON, resolve_closure, training_json_read_guard)


class ScopedEvidenceIndependentReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def historical_alias(self, name):
        historical = self.write(name, {'synthetic_only': 'never an actual outcome'})
        original = receipt(historical)
        snapshot = self.root / 'renamed-snapshot.json'
        snapshot.write_bytes(historical.read_bytes())
        historical.write_text('original path drifted')
        manifest = self.write('manifest.json', dict(schema=HISTORICAL,
            historical_artifacts_rewritten=False,
            mappings=[dict(historical=original, snapshot=receipt(snapshot))]))
        return original, snapshot, manifest

    def assert_guarded_normal_alias(self, name, message):
        original, snapshot, manifest = self.historical_alias(name)
        context = self.write('training/context.json', dict(training_only=True,
            split='training', execution_sources=[], residents={}, prior_failed_evidence=[original]))
        model = self.write('model.json', dict(actual_numerical_dependency=original))
        read_text = Path.read_text
        for roots in ([model, context], [context, model]):
            parsed = []

            def recording_read(path, *args, **kwargs):
                parsed.append(path)
                return read_text(path, *args, **kwargs)

            with self.subTest(roots=[p.name for p in roots]), mock.patch.object(
                    Path, 'read_text', recording_read):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_closure(roots, manifest,
                        training_context_references=[receipt(context)],
                        before_json_read=training_json_read_guard([]))
            self.assertNotIn(snapshot, parsed, 'forbidden target was parsed before refusal')

    def test_unindexed_raw_normal_upgrade_rejects_renamed_snapshot_before_parse(self):
        self.assert_guarded_normal_alias('unindexed/raw.json', 'Normal nonindexed raw JSON parse refused')

    def test_observed_normal_upgrade_rejects_renamed_snapshot_before_parse(self):
        self.assert_guarded_normal_alias('validation/observed.json', 'Validation outcome JSON parse refused')

    def test_original_alias_guard_cannot_be_skipped_after_materialized_source_seen(self):
        original, snapshot, manifest = self.historical_alias('unindexed/raw.json')
        model = self.write('model.json', dict(actual_numerical_dependency=original))
        # The explicit generic snapshot root can be visited first. A later
        # forbidden original reference must still refuse the complete closure.
        with self.assertRaisesRegex(ValueError, 'Normal nonindexed raw JSON parse refused'):
            resolve_closure([model, snapshot], manifest,
                            before_json_read=training_json_read_guard([]))

    def test_exact_metadata_snapshot_alias_upgrades_and_calls_guard_both_orders(self):
        inventory = self.write('prep/source-before.json', dict(audit_only='synthetic'))
        original = receipt(inventory)
        snapshot = self.root / 'inventory-snapshot.json'
        snapshot.write_bytes(inventory.read_bytes())
        manifest = self.write('manifest.json', dict(schema=HISTORICAL,
            historical_artifacts_rewritten=False,
            mappings=[dict(historical=original, snapshot=receipt(snapshot))]))
        parent = self.write('prep/prepared-repeat.json',
            dict(schema='gsdeploy.corun_exact_repeat_preparation.v8', training_only=True,
                 status='CPU_PREPARED_SOURCE_RESOURCE_AUDIT_PASS', source_before=original))
        policy = dict(schema='gsdeploy.explicit_operational_metadata_boundary_policy.v15',
            implementation=receipt(__file__), boundaries=[dict(parent=receipt(parent),
                jsonpath=['source_before'], ref=original, reason=METADATA_REASON)])
        policy['policy_sha256'] = digest(policy)

        def guard(reference):
            if reference['path'] == str(snapshot):
                raise ValueError('INDEPENDENT_INVENTORY_PARSE_REFUSED')

        opaque = resolve_closure([parent], metadata_boundary_policy=policy,
                                 before_json_read=guard)
        self.assertIn(original, opaque['materialized_sources'])
        self.assertEqual(opaque['boundaries'][0]['parent'], receipt(parent))
        self.assertEqual(opaque['boundaries'][0]['ref'], original)
        self.assertTrue(opaque['boundaries'][0]['effective_opaque_boundary'])
        # The manifest itself contains a normal snapshot reference, which is
        # an independent normal edge and must upgrade even without a root.
        with self.assertRaisesRegex(ValueError, 'INDEPENDENT_INVENTORY_PARSE_REFUSED'):
            resolve_closure([parent], manifest, metadata_boundary_policy=policy,
                            before_json_read=guard)
        for roots in ([snapshot, parent], [parent, snapshot]):
            with self.subTest(roots=[p.name for p in roots]):
                with self.assertRaisesRegex(ValueError, 'INDEPENDENT_INVENTORY_PARSE_REFUSED'):
                    resolve_closure(roots, manifest, metadata_boundary_policy=policy,
                                    before_json_read=guard)


if __name__ == '__main__':
    unittest.main()
