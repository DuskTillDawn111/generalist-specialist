"""Explicit new-protocol behavior; legacy frozen semantics remain available."""
import copy
import unittest
from types import SimpleNamespace

from gsdeploy.backends.compact import CompactAdapter, CLASS_NAME_DECODING
from gsdeploy.backends.compact_memory import exllama_request_scratch
from tests.test_compact_protocol import fixtures


EOS = {'finish_reason': 'eos', 'generated_tokens': 2, 'terminal_token_id': 2, 'eos_token_ids': [2]}


class ProtocolRepairTests(unittest.TestCase):
    def test_task_controls_are_effective_fingerprinted_and_checked_on_finish(self):
        old, study, samples = fixtures()
        policy = copy.deepcopy(old.policy)
        for task in ['ocr', 'asr']:
            policy['tasks'][task]['generation_controls'] = {'no_repeat_ngram_size': 35}
        adapter = CompactAdapter(policy, old.classmap)
        for candidate in study['generalists'] + study['specialists']:
            for task in adapter.supported_tasks(candidate):
                request = adapter.build(candidate, samples[task])
                self.assertEqual(request['generation'].get('no_repeat_ngram_size'),
                                 35 if task in ['ocr', 'asr'] else None)
        candidate = study['generalists'][0]
        request = adapter.build(candidate, samples['asr'])
        result = adapter.finish(request, samples['asr'], 'hello', evaluate_quality=False, **EOS)
        self.assertTrue(result['success'])
        request['generation']['no_repeat_ngram_size'] = 0
        with self.assertRaisesRegex(ValueError, 'generation policy changed'):
            adapter.finish(request, samples['asr'], 'hello', **EOS)
        self.assertNotIn('no_repeat_ngram_size', old.build(candidate, samples['asr'])['generation'])

    def test_invalid_or_silently_ignored_controls_are_rejected(self):
        old, _, _ = fixtures()
        for controls in [{'num_beams': 3}, {'no_repeat_ngram_size': True},
                         {'no_repeat_ngram_size': -1}, {'no_repeat_ngram_size': 35.0}]:
            policy = copy.deepcopy(old.policy)
            policy['tasks']['asr']['generation_controls'] = controls
            with self.assertRaises(ValueError):
                CompactAdapter(policy, old.classmap)

    def test_declared_repetition_penalty_reaches_every_native_family(self):
        old, study, samples = fixtures()
        policy = copy.deepcopy(old.policy)
        for task in ['ocr', 'asr']:
            policy['tasks'][task]['generation_controls'] = {
                'no_repeat_ngram_size': 35, 'repetition_penalty': 1.2}
        adapter = CompactAdapter(policy, old.classmap)
        for candidate in study['generalists'] + study['specialists']:
            for task in adapter.supported_tasks(candidate):
                request = adapter.build(candidate, samples[task])
                self.assertEqual(request['generation'].get('repetition_penalty'),
                                 1.2 if task in ['ocr', 'asr'] else None)
                if candidate['slot'] == 'generalist_2':
                    self.assertEqual(request['native']['options']['repetition_penalty'],
                                     1.2 if task in ['ocr', 'asr'] else 1.0)
        candidate = study['generalists'][0]
        request = adapter.build(candidate, samples['asr'])
        request['generation']['repetition_penalty'] = 1.0
        with self.assertRaisesRegex(ValueError, 'generation policy changed'):
            adapter.finish(request, samples['asr'], 'hello', evaluate_quality=False, **EOS)
        for invalid in [True, 0, -1, float('nan'), float('inf'), 3, '1.2']:
            policy['tasks']['asr']['generation_controls']['repetition_penalty'] = invalid
            with self.assertRaises(ValueError):
                CompactAdapter(policy, old.classmap)

    def test_decoder_only_catalog_keeps_all_classes_without_reference_leakage(self):
        old, study, samples = fixtures()
        policy = copy.deepcopy(old.policy)
        policy['tasks']['image'].update(generative_decoding=CLASS_NAME_DECODING,
                                       catalog_presentation='decoder_only')
        adapter = CompactAdapter(policy, old.classmap)
        self.assertEqual(len(adapter.image_output_labels), 1000)
        self.assertEqual(set(adapter.class_name_to_id.values()), set(old.class_ids))
        candidate = study['generalists'][0]
        original = adapter.build(candidate, samples['image'])
        altered = copy.deepcopy(samples['image'])
        altered['label'] = 'n99999999'
        self.assertEqual(original, adapter.build(candidate, altered))
        self.assertNotIn('LABEL CATALOG:', original['instruction'])
        self.assertNotIn(adapter.catalog, original['instruction'])
        legacy = old.build(candidate, samples['image'])
        self.assertIn(old.catalog, legacy['instruction'])
        feature_request = adapter.build(study['specialists'][1], samples['image'])
        self.assertEqual(len(feature_request['native']['class_texts']), 1000)
        self.assertEqual(feature_request['native']['class_ids'], old.class_ids)
        policy['tasks']['image']['generative_decoding'] = 'complete_label_token_trie_greedy_v1'
        with self.assertRaisesRegex(ValueError, 'complete canonical'):
            CompactAdapter(policy, old.classmap)
        policy['tasks']['image']['catalog_presentation'] = 'first_ten_only'
        with self.assertRaisesRegex(ValueError, 'catalog presentation'):
            CompactAdapter(policy, old.classmap)

    def test_canonical_class_names_preserve_every_id_and_resolve_duplicates(self):
        old, study, samples = fixtures()
        classes = copy.deepcopy(old.classmap)
        for i, name in enumerate(['crane', 'crane', 'Cardigan', 'cardigan']):
            classes[str(i)][1] = name
        policy = copy.deepcopy(old.policy)
        policy['tasks']['image']['generative_decoding'] = CLASS_NAME_DECODING
        adapter = CompactAdapter(policy, classes)
        self.assertEqual(len(adapter.class_name_to_id), 1000)
        self.assertEqual(set(adapter.class_name_to_id.values()), set(old.class_ids))
        self.assertEqual(adapter.image_output_labels[1], 'crane [n00000001]')
        self.assertNotEqual(adapter.image_output_labels[2], adapter.image_output_labels[3])
        sample = samples['image']
        request = adapter.build(study['generalists'][0], sample)
        label = adapter.image_output_labels[1]
        result = adapter.finish(request, sample, label, evaluate_quality=False, **EOS)
        self.assertEqual(result['raw_output'], label)
        self.assertEqual(result['prediction'], 'n00000001')
        self.assertTrue(result['success'])
        for invalid in ['crane', 'n00000001', 'a crane', 'class 9999']:
            self.assertFalse(adapter.finish(request, sample, invalid, evaluate_quality=False, **EOS)['success'])

    def test_feature_classification_still_uses_all_thousand_scores(self):
        old, study, samples = fixtures()
        policy = copy.deepcopy(old.policy)
        policy['tasks']['image']['generative_decoding'] = CLASS_NAME_DECODING
        adapter = CompactAdapter(policy, old.classmap)
        request = adapter.build(study['specialists'][1], samples['image'])
        scores = [0.0] * 1000
        scores[1] = 1.0
        result = adapter.finish(request, samples['image'], 'n00000001', evaluate_quality=False,
                                finish_reason='classified', generated_tokens=0, class_scores=scores)
        self.assertTrue(result['success'])
        self.assertEqual(result['prediction'], 'n00000001')

    def test_optional_no_answer_grammar_is_reference_independent_and_legacy_is_unchanged(self):
        old, study, samples = fixtures()
        policy = copy.deepcopy(old.policy)
        policy['tasks']['text']['no_answer_normalization'] = 'optional_brackets_terminal_punctuation_v2'
        adapter = CompactAdapter(policy, old.classmap)
        candidate = study['generalists'][0]
        request = adapter.build(candidate, samples['text'])
        for value in ['No Answer.', 'No Answer', '<No Answer>', 'no answer>']:
            result = adapter.finish(request, samples['text'], value, evaluate_quality=False, **EOS)
            self.assertEqual(result['prediction'], '')
        for value in ['No', 'No Answer. Paris', 'Paris']:
            self.assertNotEqual(adapter.finish(request, samples['text'], value, evaluate_quality=False, **EOS)['prediction'], '')
        legacy = old.finish(old.build(candidate, samples['text']), samples['text'], 'No Answer.', evaluate_quality=False, **EOS)
        self.assertEqual(legacy['prediction'], 'No Answer.')

    def test_native_scratch_keeps_full_input_and_releases_on_exception(self):
        events = []
        model = object()
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: events.append('sync')))
        def resize(owner, rows):
            self.assertIs(owner, model)
            events.append(rows)
        with exllama_request_scratch(model, True, 16384, torch, resize=resize) as result:
            events.append('full 13499 token projection')
        self.assertEqual(events, ['sync', 16384, 'full 13499 token projection', 'sync', 1, 'sync'])
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with exllama_request_scratch(model, True, 16384, torch, resize=resize):
                raise RuntimeError('test failure')
        self.assertEqual(events[-3:], ['sync', 1, 'sync'])
        with exllama_request_scratch(None, False, None, None):
            pass


if __name__ == '__main__':
    unittest.main()
