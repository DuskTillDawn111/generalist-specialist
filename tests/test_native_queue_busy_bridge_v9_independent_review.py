"""Independent CPU receipt/input mutations for the queue native bridge."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import native_queue_busy_replay_v9 as replay
from tools.predictor_validation_20261027.background_collect import context_identity


class QueueBridgeIndependentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_code = self.root / 'worker.py'
        self.source_code.write_text('# synthetic worker source\n')
        spec = self.root / 'spec.json'
        spec.write_text('{}')
        weights = self.root / 'weight-stats.json'
        weights.write_text('[]')
        self.source_pool = self.root / 'source/runtime-inputs.jsonl'
        self.target_pool = self.root / 'target/runtime-inputs.jsonl'
        for p in (self.source_pool, self.target_pool):
            p.parent.mkdir()
            p.write_text(json.dumps(dict(pool_id='same-id', task_type='text',
                sample=dict(id='same-id', task='text', context='original public text', question='question'))) + '\n')
        self.source = dict(shared_native_execution_sources=[replay.receipt(self.source_code)],
            worker_bindings=dict(text=dict(spec=str(spec), spec_sha256=replay.receipt(spec)['sha256'])),
            residents={'a': 'text'}, threads={}, execution=dict(physical_owners=1),
            device='synthetic-device', measurement_domain='natural', profiler='none',
            affine_observation=None, worker_variant='semantic_worker.py', large_weight_identity_scope='stat_only',
            large_weight_stats=replay.receipt(weights),
            execution_sources=[dict(original_path=str(self.source_pool), frozen=replay.receipt(self.source_pool))])
        self.target = copy.deepcopy(self.source)
        self.target.update(measurement_domain='natural_predeclared_queue_training_v8', queue_training=True, training_only=True)
        self.target['execution_sources'] = [dict(original_path=str(self.target_pool), frozen=replay.receipt(self.target_pool))]
        self.source_ref = self.write('source-context.json', self.source)
        self.target_ref = self.write('target-context.json', self.target)
        self.composite = self.write('composite.json', dict(target=dict(context=self.source_ref,
            execution_context_sha256=context_identity(self.source))))
        self.bundle = dict(composite=self.composite)

    def write(self, name, document):
        p = self.root / name
        p.write_text(json.dumps(document))
        return replay.receipt(p)

    def test_only_domain_projection_matches_full_contexts_remain_distinct(self):
        result = replay.queue_context_bridge(self.bundle, self.target_ref)
        self.assertNotEqual(result['source_full_context_sha256'], result['target_full_context_sha256'])
        self.assertFalse(result['full_context_equality_claimed'])
        self.assertTrue(result['collector_invariance_is_hypothesis'])
        self.assertFalse(result['natural_timing_transfer_validated'])

    def test_changed_native_field_is_refused(self):
        self.target['threads'] = {'OMP_NUM_THREADS': '999'}
        with self.assertRaisesRegex(UnknownEvidence, 'threads'):
            replay.queue_context_bridge(self.bundle, self.write('target-context.json', self.target))

    def test_different_weight_manifest_bytes_are_refused(self):
        p = self.root / 'other-weight-stats.json'
        p.write_text('[{"synthetic":"different"}]')
        self.target['large_weight_stats'] = replay.receipt(p)
        with self.assertRaisesRegex(UnknownEvidence, 'large_weight'):
            replay.queue_context_bridge(self.bundle, self.write('target-context.json', self.target))

    def test_changed_public_content_cannot_reuse_same_pool_identity(self):
        text = json.loads(self.target_pool.read_text())
        text['sample']['context'] = 'different public content under the same ID'
        self.target_pool.write_text(json.dumps(text) + '\n')
        self.target['execution_sources'][0]['frozen'] = replay.receipt(self.target_pool)
        with self.assertRaises((ValueError, UnknownEvidence)):
            replay.queue_context_bridge(self.bundle, self.write('target-context.json', self.target))

    def test_missing_explicit_native_field_is_refused_even_when_source_value_is_null(self):
        del self.target['affine_observation']
        with self.assertRaises((ValueError, UnknownEvidence)):
            replay.queue_context_bridge(self.bundle, self.write('target-context.json', self.target))

    def test_numeric_fit_receipts_are_verified_before_materializing(self):
        refs = {name: self.write(name + '.json', dict(synthetic=name)) for name in
                ('component_support', 'host_supply_components', 'semantic_fit', 'copy_memset_fit', 'two_capture_provisional_fit')}
        bundle = dict(self.bundle, forecast_defaults=dict(aliases={'a': dict(supported_tasks=['text'])}),
            host_supply_scenario='synthetic', generation=dict(component_support=refs['component_support']),
            **{name: refs[name] for name in ('host_supply_components', 'semantic_fit', 'copy_memset_fit', 'two_capture_provisional_fit')})
        Path(refs['copy_memset_fit']['path']).write_text('{"different":"valid-json"}')
        adapter = types.SimpleNamespace(plans={('a', 'text', 'same-id'): dict(nodes=[dict(id='gpu', kind='gpu')])})
        generation = types.SimpleNamespace(check=lambda rows: {}, public_bindings={})
        durations = types.SimpleNamespace(coverage={})
        class Duration:
            coverage = {}
            def __call__(self, node, row): return 10.
        with patch.object(replay, 'MultiNativeProvider', return_value=adapter), \
             patch.object(replay, 'ComponentGenerationSupport', return_value=generation), \
             patch.object(replay, 'ComponentHostSupply', return_value=lambda n, r: 0), \
             patch.object(replay, 'NumericDurations', return_value=Duration()):
            with self.assertRaisesRegex(ValueError, 'source receipt|receipt|Changed'):
                replay.frozen_training_providers(bundle, [dict(request_id='r', alias='a', task='text', pool_id='same-id', arrival_ns=0)])


if __name__ == '__main__':
    unittest.main()
