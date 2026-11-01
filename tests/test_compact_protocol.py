"""Behavioral tests for the new task boundary; no model libraries or GPU."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
import tempfile
import hashlib

from gsdeploy.backends.compact import CompactAdapter, fingerprint
from gsdeploy.tasks import asr_normalize, summarize_quality


ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    policy = json.loads((ROOT / 'configs/compact_protocol.json').read_text())
    study = json.loads((ROOT / 'configs/study.json').read_text())
    classes = {str(i): [f'n{i:08d}', f'class_{i}'] for i in range(1000)}
    samples = {
        'text': {'id': 'qa', 'task': 'text', 'dataset': 'SQuAD2-dev', 'context': 'The city is Paris.',
                 'question': 'Which city?', 'answers': ['Paris'], 'label': 'Paris'},
        'image': {'id': 'image', 'task': 'image', 'dataset': 'ImageNet-A', 'input': '/fixture/image.jpg',
                  'input_sha256': 'a' * 64, 'label': 'n00000001'},
        'ocr': {'id': 'page', 'task': 'ocr', 'dataset': 'SROIE-test-pages', 'input': '/fixture/page.jpg',
                'input_sha256': 'b' * 64, 'label': 'ITEM PRICE A, B -1.20'},
        'asr': {'id': 'audio', 'task': 'asr', 'dataset': 'LibriSpeech-dev-other', 'input': '/fixture/audio.flac',
                'input_sha256': 'c' * 64, 'label': "I CAN'T GO", 'duration_s': 5.0},
    }
    return CompactAdapter(policy, classes), study, samples


class CompactProtocolTests(unittest.TestCase):
    def test_all_sixteen_paths_build_and_references_never_change_requests(self):
        adapter, study, samples = fixtures()
        paths = 0
        for candidate in study['generalists'] + study['specialists']:
            for task in adapter.supported_tasks(candidate):
                sample = samples[task]
                before = copy.deepcopy(sample)
                first = adapter.build(candidate, sample)
                changed = copy.deepcopy(sample)
                changed.update(label='SECRET REFERENCE', answers=['SECRET ANSWER'], entities={'total': 'SECRET'})
                self.assertEqual(first, adapter.build(candidate, changed))
                self.assertNotIn('SECRET', json.dumps(first))
                self.assertEqual(sample, before)
                self.assertFalse(first['input_truncation'])
                if task == 'image' and candidate in study['generalists']:
                    self.assertIn('n00000000:', first['instruction'])
                    self.assertIn('n00000999:', first['instruction'])
                    self.assertEqual(first['instruction'].count(': class '), 1000)
                paths += 1
        self.assertEqual(paths, 16)

    def test_legacy_tasks_and_wrong_model_task_routes_are_rejected(self):
        adapter, study, samples = fixtures()
        for task in ['lid', 'visual_qa']:
            with self.assertRaises(ValueError):
                adapter.build(study['generalists'][0], {'task': task})
        with self.assertRaises(ValueError):
            adapter.build(study['specialists'][0], samples['image'])
        old = {**samples['text'], 'dataset': 'SST-2'}
        with self.assertRaises(ValueError):
            adapter.build(study['generalists'][0], old)
        for duration in [0, 30, float('nan')]:
            with self.assertRaises(ValueError):
                adapter.build(study['generalists'][0], {**samples['asr'], 'duration_s': duration})

    def test_native_formatting_and_cache_identity(self):
        adapter, study, samples = fixtures()
        qa, image, ocr, speech = study['specialists']
        qa_request = adapter.build(qa, samples['text'])
        self.assertEqual(qa_request['native']['text'], 'which city? \\n the city is paris.')
        self.assertEqual(qa_request['native']['max_input_tokens'], 512)
        key = adapter.build(image, samples['image'])['native']['cache_key']
        altered = {**image, 'precision': 'INT8'}
        self.assertNotEqual(key, adapter.build(altered, samples['image'])['native']['cache_key'])
        self.assertEqual(adapter.build(ocr, samples['ocr'])['native']['options']['eval_mode'], True)
        self.assertEqual(adapter.build(speech, samples['asr'])['native']['generate_options']['task'], 'transcribe')
        policy = copy.deepcopy(adapter.policy)
        policy['tasks']['ocr']['max_new_tokens'] = 1024
        another = CompactAdapter(policy, adapter.classmap)
        with self.assertRaisesRegex(ValueError, 'hardcodes 8192'):
            another.build(ocr, samples['ocr'])

    def finish(self, adapter, candidate, sample, output, reason='eos', tokens=3, **extra):
        request = adapter.build(candidate, sample)
        return adapter.finish(request, sample, output, finish_reason=reason, generated_tokens=tokens,
                              terminal_token_id=2 if reason == 'eos' else None,
                              eos_token_ids=[2], **extra)

    def test_missing_eos_truncation_and_failure_never_score_as_no_answer(self):
        adapter, study, samples = fixtures()
        sample = {**samples['text'], 'answers': [], 'label': ''}
        model = study['generalists'][0]
        for reason in ['length', 'unknown', 'error']:
            result = self.finish(adapter, model, sample, '<No Answer>', reason)
            self.assertFalse(result['success'])
            self.assertEqual(result['score']['EM'], 0)
        result = self.finish(adapter, model, sample, 'no answer>')
        self.assertTrue(result['success'])
        self.assertEqual(result['score']['EM'], 1)
        request = adapter.build(model, sample)
        invalid = adapter.finish(request, sample, '<No Answer>', finish_reason='eos', generated_tokens=3,
                                 terminal_token_id=9, eos_token_ids=[2])
        self.assertFalse(invalid['success'])
        too_long = self.finish(adapter, model, sample, '<No Answer>', tokens=129)
        self.assertEqual(too_long['failure_reason'], 'OUTPUT_BUDGET_EXCEEDED')

    def test_same_task_scoring_across_models_and_strict_class_output(self):
        adapter, study, samples = fixtures()
        model = study['generalists'][0]
        for task, raw in [('text', 'The Paris'), ('ocr', '<table><tr><td>ITEM</td><td>PRICE</td></tr><tr><td>A, B</td><td>-1.20</td></tr></table>'),
                          ('asr', "I can’t go.")]:
            result = self.finish(adapter, model, samples[task], raw)
            self.assertTrue(result['success'])
            self.assertEqual(result['score'].get('errors', 0), 0)
            if task == 'text':
                self.assertEqual(result['score']['EM'], 1)
        correct = self.finish(adapter, model, samples['image'], 'n00000001')
        self.assertTrue(correct['score']['correct'])
        for output in ['class_1', 'Answer: n00000001', 'n00000001 n00000002', 'n99999999']:
            self.assertFalse(self.finish(adapter, model, samples['image'], output)['success'])
        self.assertEqual(asr_normalize('TWENTY-ONE'), 'twenty one')
        self.assertEqual(asr_normalize("'Hello' — can’t."), "hello can't")

    def test_feature_classification_requires_finite_full_catalog_and_argmax(self):
        adapter, study, samples = fixtures()
        model = study['specialists'][1]
        scores = [0.0] * 1000
        scores[1] = 2.0
        result = self.finish(adapter, model, samples['image'], 'n00000001', 'classified', 0, class_scores=scores)
        self.assertTrue(result['score']['correct'])
        for invalid in [None, scores[:200], [float('nan')] * 1000]:
            result = self.finish(adapter, model, samples['image'], 'n00000001', 'classified', 0, class_scores=invalid)
            self.assertFalse(result['success'])
        result = self.finish(adapter, model, samples['image'], 'n00000002', 'classified', 0, class_scores=scores)
        self.assertEqual(result['failure_reason'], 'CLASS_ARGMAX_MISMATCH')
        with self.assertRaises(ValueError):
            self.finish(adapter, study['generalists'][0], samples['image'], 'n00000001', 'classified', 0, class_scores=scores)

    def test_rounds_cannot_be_merged_or_failed_samples_dropped(self):
        adapter, study, samples = fixtures()
        a = samples['asr']
        b = {**a, 'id': 'other', 'label': 'ONE TWO THREE'}
        good = self.finish(adapter, study['generalists'][0], a, a['label'])
        failed = self.finish(adapter, study['generalists'][0], b, '', 'error', 0)
        result = summarize_quality([a, b], [good, failed])
        self.assertEqual(result['unique_samples'], 2)
        self.assertEqual(result['failed_samples'], 1)
        self.assertEqual(result['errors'], 3)
        self.assertEqual(result['reference_units'], 6)
        self.assertAlmostEqual(result['WER'], 3 / 6)
        with self.assertRaises(ValueError):
            summarize_quality([a, b], [good])
        with self.assertRaises(ValueError):
            summarize_quality([a, b], [good, good, failed])

    def test_import_and_request_construction_do_not_import_gpu_libraries(self):
        script = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'pynvml', 'auto_gptq', 'bitsandbytes'}:
        raise RuntimeError('Forbidden model/GPU library import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tests.test_compact_protocol import fixtures
adapter, study, samples = fixtures()
for candidate in study['generalists'] + study['specialists']:
    for task in adapter.supported_tasks(candidate):
        adapter.build(candidate, samples[task])
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class CompactPacketTests(unittest.TestCase):
    def fixture(self, root):
        adapter, study, samples = fixtures()
        config = root / 'configs'
        config.mkdir()
        policy = config / 'protocol.json'
        policy.write_text(json.dumps(adapter.policy))
        environments = config / 'environments.json'
        environments.write_bytes((ROOT / 'configs/compact_environments.json').read_bytes())
        classmap = root / 'imagenet-class-index.json'
        classmap.write_text(json.dumps(adapter.classmap))
        source = root / 'quality'
        source.mkdir()
        asset = root / 'media'
        asset.write_bytes(b'CPU data-flow fixture, no inference')
        task_manifests = {}
        for task, sample in samples.items():
            sample = copy.deepcopy(sample)
            if task != 'text':
                sample.update(input=str(asset), input_sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
            path = source / (task + '.jsonl')
            path.write_text(json.dumps(sample) + '\n')
            task_manifests[task] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'samples': 1}
        manifest = source / 'evaluation-status.json'
        manifest.write_text(json.dumps({
            'status': 'FOUR_TASK_INPUTS_FROZEN_PROTOCOL_REVIEW_PENDING_NOT_ADMITTED',
            'tasks': task_manifests,
            'source_hashes': {str(classmap): hashlib.sha256(classmap.read_bytes()).hexdigest()}}))
        study['quality_preparation'].update(manifest=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        study['task_adapter_protocol'].update(sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                                             environment_plan_sha256=hashlib.sha256(environments.read_bytes()).hexdigest())
        study_path = config / 'study.json'
        study_path.write_text(json.dumps(study))
        return (study_path, policy, environments, classmap), asset

    def test_packet_stays_unexecuted_and_never_overwrites(self):
        from gsdeploy.profiling.compact import prepare_compact
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, _ = self.fixture(root)
            output = root / 'packet'
            report = prepare_compact(*inputs, output)
            self.assertEqual(report['total_paths'], 16)
            self.assertEqual(report['constructed_requests'], 16)
            self.assertEqual(report['executed_requests'], 0)
            self.assertEqual(report['models_loaded'], 0)
            self.assertFalse(report['gpu_started'])
            self.assertFalse(report['native_execution_ready'])
            self.assertIsNone(report['utility'])
            with self.assertRaises(FileExistsError):
                prepare_compact(*inputs, output)

    def test_changed_media_or_protocol_cannot_create_a_valid_packet(self):
        from gsdeploy.profiling.compact import prepare_compact
        for mutate in ['media', 'protocol', 'classmap']:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs, asset = self.fixture(root)
                if mutate == 'media':
                    asset.write_bytes(b'changed media')
                elif mutate == 'protocol':
                    inputs[1].write_text(inputs[1].read_text() + ' ')
                else:
                    catalog = json.loads(inputs[3].read_text())
                    catalog['0'][1] = 'different label catalog'
                    inputs[3].write_text(json.dumps(catalog))
                output = root / 'packet'
                with self.assertRaises(ValueError):
                    prepare_compact(*inputs, output)
                self.assertFalse(output.exists())


class EditDistanceConsistencyTests(unittest.TestCase):
    def test_fast_and_fallback_distance_agree_for_characters_and_words(self):
        from unittest.mock import patch
        from gsdeploy.tasks import distance
        cases = [('kitten', 'sitting'), ('a, b -1.20', 'a b 1.20'), ('', 'abc'),
                 (['i', "can't", 'go'], ['i', 'can', 'go']), (['one'], []), ('价格 12', '价格 13')]
        fast = [distance(a, b) for a, b in cases]
        with patch.dict(sys.modules, {'rapidfuzz.distance': None}):
            fallback = [distance(a, b) for a, b in cases]
        self.assertEqual(fast, fallback)
        self.assertEqual(fast[0], 3)
        self.assertEqual(fast[3], 1)


if __name__ == '__main__':
    unittest.main()
