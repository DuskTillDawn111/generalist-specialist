import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_startup_prediction_fit_v4 import raw_fixture
from tools.predictor_validation_20261027 import startup_prediction_v6 as tool


class FlowBoundStartupTests(unittest.TestCase):
    def native(self, context):
        return dict(projection_sha256='native', original_full_context_sha256=tool.context_identity(context),
                    measurement_domain='natural')

    def write(self, path, value):
        path.write_text(json.dumps(value))
        return tool.receipt(path)

    def fixture(self, root):
        base = root / 'base'
        base.mkdir()
        rows, inputs = [], []
        for task in ('text', 'image', 'asr', 'ocr'):
            for size in range(2):
                pool = task + str(size)
                if task == 'text':
                    sample = dict(context='c' * (size + 1), question='q')
                    exact = tool.digest(sample)
                    amount = len(sample['context']) + 1
                else:
                    media = root / (pool + '.media')
                    media.write_bytes((pool * 2).encode())
                    exact = tool.receipt(media)['sha256']
                    sample = dict(input=str(media), input_sha256=exact)
                    amount = media.stat().st_size
                sample.update(id=pool, task=task)
                rows.append(dict(pool_id=pool, task_type=task, sample=sample))
                inputs.append(dict(task=task, pool_id=pool, input_size=amount, exact_input_sha256=exact))
        live_pool = base / 'runtime-inputs.jsonl'
        live_pool.write_text('\n'.join(map(json.dumps, rows)))
        snapshot = root / 'frozen-runtime-inputs.jsonl'
        snapshot.write_bytes(live_pool.read_bytes())
        stats = self.write(root / 'stats.json', [])
        context = dict(residents=tool.RESIDENTS, execution_sources=[dict(
            original_path=str(live_pool), frozen=tool.receipt(snapshot))], large_weight_stats=stats)
        context_ref = self.write(root / 'context.json', context)
        plan = dict(domain='natural', split='training', inputs=inputs,
                    execution_context_sha256=tool.context_identity(context), jobs=[dict(split='training')])
        plan['plan_sha256'] = tool.digest(plan)
        plan_ref = self.write(root / 'plan.json', plan)
        reviewer = root / 'reviewer.py'
        reviewer.write_text('strict original CPU test fixture\n')
        reviews = []
        for i in range(2):
            raw = raw_fixture()
            raw.update(capture=i, execution_context_sha256=plan['execution_context_sha256'],
                plan=plan_ref, context=context_ref,
                request_records=[dict(split='training', status='completed')],
                priming_request_records=[dict(r, status='completed') for r in
                    tool.original.planned_priming(inputs, context['residents'])])
            raw_ref = self.write(root / ('raw' + str(i) + '.json'), raw)
            report = dict(schema='gsdeploy.startup_training_collection_strict_review.v4', status='PASS',
                issues=[], owners={}, raw=raw_ref, plan=plan_ref, context=context_ref,
                capture_run_id='raw-sha256:' + raw_ref['sha256'], evaluator_sources=[tool.receipt(reviewer)])
            reviews.append(self.write(root / ('review' + str(i) + '.json'), report)['path'])
        with patch.object(tool.original, 'native_execution_projection', side_effect=self.native):
            tool.original.fit(reviews, root / 'fit')
        model_path = root / 'fit/model.json'
        target_context = dict(context, target_collector='different honest complete context')
        target_context_ref = self.write(root / 'target-context.json', target_context)
        aliases = {a: dict(family=f, supported_tasks=['text','image','asr','ocr']) for a, f in tool.RESIDENTS.items()}
        full = tool.context_identity(target_context)
        candidate = dict(id='shared_balanced', execution_context_sha256=full,
            resource_configuration=tool.flow.resource_configuration(), routes={'measurement:0':'text#0'})
        catalog_ref = self.write(base / 'catalog.json', dict(safe_factor=1.1))
        flow_plan = dict(schema=tool.flow.SCHEMA,
            validation_scope='FRESH_HELDOUT_CO_RUN_REQUESTS_OF_EXACT_TRAINED_PUBLIC_INPUTS',
            context=target_context_ref, profile_plan=plan_ref, catalog=catalog_ref,
            execution_context_sha256=full, runner_source=tool.receipt(tool.flow.__file__),
            collector_sources=[tool.receipt(p) for p in tool.flow.collection_source_paths()],
            warmup_seconds=0., window_seconds=1., drain_seconds=2., memory_budget_mib=49140.,
            memory_safe_factor=1.1, energy_reference_watts=450., startup_amortization_seconds=3600.,
            aliases=aliases, workload_rps={'text':1.}, latency_reference_ms={'text':1000.},
            weights={k:1/6 for k in 'QTLPME'}, baseline_candidate_id='shared_balanced',
            candidates=[candidate], trace=[dict(request_id='measurement:0', offset_ns=0, task='text', pool_id='text0')])
        flow_plan['plan_sha256'] = tool.digest(flow_plan)
        flow_path = root / 'flow.json'
        self.write(flow_path, flow_plan)
        return model_path, flow_path, base, inputs, reviewer

    def call(self, model, plan):
        return tool.prediction(model, plan, 'shared_balanced', allow_native_priming_transfer_hypothesis=True)

    def test_real_targets_and_honest_distinct_contexts_are_bound(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(tool, 'native_execution_projection', side_effect=self.native):
            model, plan, base, _, _ = self.fixture(Path(temp))
            with patch.object(tool.flow, 'BASE', base):
                result = self.call(model, plan)
            self.assertEqual(result['energy_joules'], 400.)
            self.assertEqual(result['peak_memory_mib'], 800.)
            self.assertTrue(result['energy_includes_priming'])
            self.assertFalse(result['safety_factor_applied'])
            self.assertFalse(result['full_context_equality_claimed'])
            self.assertNotEqual(result['execution_context_sha256'], result['profile_execution_context_sha256'])
            self.assertEqual(len(result['target_contract']['actual_priming_content']), 12)
            self.assertEqual(result['flow_plan'], tool.receipt(plan))

    def test_negative_positive_mean_and_observation_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(tool, 'native_execution_projection', side_effect=self.native):
            root = Path(temp)
            model, plan, base, _, _ = self.fixture(root)
            document = json.loads(model.read_text())
            with patch.object(tool.flow, 'BASE', base):
                for value in (-1., 123., float('nan')):
                    changed = copy.deepcopy(document)
                    changed['startup_prediction']['energy_joules'] = value
                    modified = root / ('model' + str(value) + '.json')
                    self.write(modified, changed)
                    with self.assertRaises(ValueError):
                        self.call(modified, plan)
                changed = copy.deepcopy(document)
                changed['observations'][0]['peak_memory_mib'] = 1.
                modified = root / 'changed-observation.json'
                self.write(modified, changed)
                with self.assertRaisesRegex(ValueError, 'recomputed'):
                    self.call(modified, plan)

    def test_warmup_admission_plan_hash_and_collector_changes_fail(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(tool, 'native_execution_projection', side_effect=self.native):
            root = Path(temp)
            model, plan, base, _, _ = self.fixture(root)
            document = json.loads(plan.read_text())
            with patch.object(tool.flow, 'BASE', base):
                for kind in ('warmup', 'admission', 'collector', 'hash'):
                    changed = copy.deepcopy(document)
                    if kind == 'warmup':
                        changed['warmup_seconds'] = 1.
                    elif kind == 'admission':
                        changed['candidates'][0]['resource_configuration'] = tool.flow.resource_configuration(1)
                    elif kind == 'collector':
                        changed['collector_sources'] = changed['collector_sources'][:-1]
                    changed.pop('plan_sha256')
                    changed['plan_sha256'] = tool.digest(changed) if kind != 'hash' else 'wrong'
                    modified = root / ('plan-' + kind + '.json')
                    self.write(modified, changed)
                    with self.assertRaises(ValueError):
                        self.call(model, modified)

    def test_target_profile_and_catalog_content_are_validated_not_only_pool_id(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(tool, 'native_execution_projection', side_effect=self.native):
            root = Path(temp)
            model, plan, base, inputs, _ = self.fixture(root)
            document = json.loads(plan.read_text())
            profile_path = Path(document['profile_plan']['path'])
            profile = json.loads(profile_path.read_text())
            profile['inputs'][0]['exact_input_sha256'] = 'wrong-same-pool-id'
            profile.pop('plan_sha256')
            profile['plan_sha256'] = tool.digest(profile)
            # New profile document, separate from immutable training plan.
            modified_profile = root / 'different-profile.json'
            document['profile_plan'] = self.write(modified_profile, profile)
            document.pop('plan_sha256')
            document['plan_sha256'] = tool.digest(document)
            modified_flow = root / 'different-profile-flow.json'
            self.write(modified_flow, document)
            with patch.object(tool.flow, 'BASE', base):
                with self.assertRaisesRegex(ValueError, 'priming hash/size'):
                    self.call(model, modified_flow)

    def test_nested_training_source_or_actual_catalog_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(tool, 'native_execution_projection', side_effect=self.native):
            root = Path(temp)
            model, plan, base, _, reviewer = self.fixture(root)
            with patch.object(tool.flow, 'BASE', base):
                reviewer.write_text('changed historical evaluator\n')
                with self.assertRaisesRegex(ValueError, 'Changed receipt'):
                    self.call(model, plan)

    def test_live_catalog_and_media_bytes_are_verified(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(tool, 'native_execution_projection', side_effect=self.native):
            root = Path(temp)
            model, plan, base, _, _ = self.fixture(root)
            with patch.object(tool.flow, 'BASE', base):
                runtime_pool = base / 'runtime-inputs.jsonl'
                original = runtime_pool.read_bytes()
                runtime_pool.write_bytes(original + b'\n')
                with self.assertRaisesRegex(ValueError, 'source closure'):
                    self.call(model, plan)
                runtime_pool.write_bytes(original)
                media = root / 'image0.media'
                media.write_bytes(b'changed actual media')
                with self.assertRaisesRegex(ValueError, 'media bytes'):
                    self.call(model, plan)

    def test_transfer_requires_explicit_optin(self):
        with self.assertRaisesRegex(ValueError, 'Explicit native'):
            tool.prediction('not-read', 'not-read', 'not-read')


if __name__ == '__main__':
    unittest.main()
