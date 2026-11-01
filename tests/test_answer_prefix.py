"""Native decoding presets, answer boundaries and beam-aware class constraints."""
import copy
import tempfile
from types import SimpleNamespace
import unittest

from gsdeploy.backends.compact import CompactAdapter, NATIVE_IMAGE_OBSERVATION, CLASS_NAME_DECODING, CLASS_NAME_DECODING_V2
from gsdeploy.backends.compact_native import ClassLabelLogitsProcessor, CompleteMiniProcessor, label_token_trie
from tests.test_compact_protocol import fixtures
from tests import test_compact_native as native_fixtures


class AnswerPrefixTests(unittest.TestCase):
    def test_leading_space_surface_keeps_all_semantic_classes_and_exact_scoring(self):
        original, study, samples = fixtures()
        policy = copy.deepcopy(original.policy)
        policy['tasks']['image'].update(generative_decoding=CLASS_NAME_DECODING_V2,
                                       catalog_presentation='decoder_only', class_label_surface='leading_space_v1')
        adapter = CompactAdapter(policy, original.classmap)
        self.assertEqual(len(adapter.image_output_labels), 1000)
        self.assertTrue(all(label.startswith(' ') for label in adapter.image_output_labels))
        self.assertEqual(adapter.class_ids, original.class_ids)
        request = adapter.build(study['generalists'][0], samples['image'])
        result = adapter.finish(request, samples['image'], ' class 1', finish_reason='eos',
                                generated_tokens=4, terminal_token_id=2, eos_token_ids=[2])
        self.assertTrue(result['score']['correct'])
        self.assertEqual(result['prediction'], 'n00000001')
        policy['tasks']['image']['class_label_surface'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'surface'):
            CompactAdapter(policy, original.classmap)

    def test_pinned_ocr_optical_preset_is_explicit_and_unknown_presets_fail(self):
        original, study, samples = fixtures()
        candidate = next(c for c in study['specialists'] if c['slot'] == 'ocr')
        before = original.build(candidate, samples['ocr'])
        policy = copy.deepcopy(original.policy)
        policy['tasks']['ocr']['deepseek_image_preset'] = 'native_large_v1'
        after = CompactAdapter(policy, original.classmap).build(candidate, samples['ocr'])
        self.assertEqual(after['media'], before['media'])
        self.assertEqual(after['generation'], before['generation'])
        self.assertEqual({k:after['native']['options'][k] for k in ['base_size','image_size','crop_mode']},
                         {'base_size':1280,'image_size':1280,'crop_mode':False})
        self.assertEqual(after['native']['options']['prompt'], before['native']['options']['prompt'])
        policy['tasks']['ocr']['deepseek_image_preset'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'image preset'):
            CompactAdapter(policy, original.classmap).build(candidate, samples['ocr'])

    def test_native_preset_is_explicit_model_checked_and_not_a_global_sampling_change(self):
        original, study, samples = fixtures()
        policy = copy.deepcopy(original.policy)
        mini = study['generalists'][1]
        policy['native_decoding_presets'] = {mini['model']: {'asr': 'minicpm_deterministic_chat_v1'}}
        adapter = CompactAdapter(policy, original.classmap)
        request = adapter.build(mini, samples['asr'])
        self.assertEqual(request['generation']['num_beams'], 3)
        self.assertEqual(request['generation']['repetition_penalty'], 1.2)
        self.assertFalse(request['generation']['do_sample'])
        self.assertEqual(adapter.build(study['generalists'][0], samples['asr'])['generation']['num_beams'], 1)
        self.assertEqual(adapter.build(mini, samples['image'])['generation']['num_beams'], 1)
        policy['native_decoding_presets'] = {study['generalists'][0]['model']: {'asr': 'minicpm_deterministic_chat_v1'}}
        with self.assertRaisesRegex(ValueError, 'family'):
            CompactAdapter(policy, original.classmap)

    def test_beam_classification_has_an_explicit_mode_without_relabeling_legacy_greedy(self):
        original, study, samples = fixtures()
        policy = copy.deepcopy(original.policy)
        policy['diagnostic_only'] = True
        mini = study['generalists'][1]
        policy['tasks']['image']['generative_decoding'] = CLASS_NAME_DECODING
        policy['model_generation_controls'] = {mini['model']: {'image': {'num_beams': 3}}}
        with self.assertRaisesRegex(ValueError, 'Legacy greedy'):
            CompactAdapter(policy, original.classmap).build(mini, samples['image'])
        policy['tasks']['image']['generative_decoding'] = CLASS_NAME_DECODING_V2
        adapter = CompactAdapter(policy, original.classmap)
        self.assertEqual(len(adapter.image_output_labels), 1000)
        self.assertEqual(adapter.build(mini, samples['image'])['generation']['num_beams'], 3)

    def test_prefix_is_bound_before_tokenization_in_all_three_native_paths(self):
        native_fixtures.NativeBindingTests.setUpClass()
        helper = native_fixtures.NativeBindingTests()
        original, study, samples = fixtures()
        policy = copy.deepcopy(original.policy)
        policy['diagnostic_only'] = True
        policy['tasks']['image'].update(generative_decoding=NATIVE_IMAGE_OBSERVATION,
                                       catalog_presentation='decoder_only', assistant_prefix='Class name:\n')
        adapter = CompactAdapter(policy, original.classmap)
        class Spy:
            def __init__(self, inner):
                self.inner, self.text = inner, None
            def __getattr__(self, name):
                return getattr(self.inner, name)
            def __call__(self, *args, **kwargs):
                self.text = kwargs.get('text', args[0] if args else None)
                return self.inner(*args, **kwargs)
        with helper.no_cuda(), tempfile.TemporaryDirectory() as folder:
            for candidate in study['generalists']:
                executor, processor, model = helper.binding(candidate, samples['image'], folder)
                spy = Spy(processor)
                executor.processor, executor.adapter = spy, adapter
                executor.infer(samples['image'])
                texts = spy.text if isinstance(spy.text, list) else [spy.text]
                self.assertTrue(all(t.endswith('Class name:\n') for t in texts))
                self.assertEqual(model.calls[-1].get('thinker_num_beams', model.calls[-1].get('num_beams')), 1)

    def test_label_boundaries_must_stay_identical_after_answer_prefix(self):
        def encode(text, **kwargs):
            if text == 'Class name:\n':
                return [9]
            if text.startswith('Class name:\n'):
                return [9] + [int(c) for c in text[len('Class name:\n'):]]
            if text.startswith('unstable'):
                return [8]
            return [int(c) for c in text]
        tokenizer = SimpleNamespace(encode=encode)
        label_token_trie(tokenizer, ['12', '13', '24'], [7], 3, answer_prefix='Class name:\n')
        with self.assertRaisesRegex(ValueError, 'token boundaries'):
            label_token_trie(tokenizer, ['12'], [7], 3, answer_prefix='unstable')

    def test_three_beams_are_constrained_independently_and_still_require_eos(self):
        native_fixtures.NativeBindingTests.setUpClass()
        torch = native_fixtures.NativeBindingTests.torch
        trie = label_token_trie(SimpleNamespace(encode=lambda text, **kw: [int(c) for c in text]), ['12', '13', '24'], [9], 3)
        processor = ClassLabelLogitsProcessor(trie, num_beams=3)
        scores = torch.zeros(3, 10)
        processor(torch.empty((3, 0), dtype=torch.long), scores)
        masked = processor(torch.tensor([[1], [1], [2]]), scores)
        self.assertEqual([torch.isfinite(r).nonzero()[:, 0].tolist() for r in masked], [[2, 3], [2, 3], [4]])
        masked = processor(torch.tensor([[1, 2], [1, 3], [2, 4]]), scores)
        self.assertTrue(all(torch.isfinite(r).nonzero()[:, 0].tolist() == [9] for r in masked))
        with self.assertRaisesRegex(ValueError, 'beam count'):
            processor(torch.empty((1, 0), dtype=torch.long), scores[:1])

    def test_impossible_beams_are_eliminated_without_crashing_or_allowing_invalid_classes(self):
        native_fixtures.NativeBindingTests.setUpClass()
        torch = native_fixtures.NativeBindingTests.torch
        trie = label_token_trie(SimpleNamespace(encode=lambda text, **kw: [int(c) for c in text]), ['12', '13', '24'], [9], 3)
        processor = ClassLabelLogitsProcessor(trie, num_beams=3)
        scores = torch.zeros(3, 10)
        processor(torch.empty((3, 0), dtype=torch.long), scores)
        masked = processor(torch.tensor([[1, 2, 9], [1, 3, 0], [2, 4, 9]]), scores)
        self.assertFalse(torch.isfinite(masked).any())
        self.assertEqual(processor.discarded_beam_rows, 3)
        single = ClassLabelLogitsProcessor(trie)
        single(torch.empty((1, 0), dtype=torch.long), scores[:1])
        with self.assertRaisesRegex(ValueError, 'left'):
            single(torch.tensor([[1, 4]]), scores[:1])

    def test_actual_transformers_beam_loop_handles_early_finished_class_names(self):
        native_fixtures.NativeBindingTests.setUpClass()
        torch = native_fixtures.NativeBindingTests.torch
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                            eos_token_id=2, pad_token_id=0, bos_token_id=1)
        config._attn_implementation = 'eager'
        model = Qwen2ForCausalLM(config).eval()
        for parameter in model.parameters():
            parameter.data.zero_()
        trie = label_token_trie(SimpleNamespace(encode=lambda text, **kw: [int(c) for c in text]), ['3', '45', '678'], [2], 8)
        processor = ClassLabelLogitsProcessor(trie, num_beams=3)
        with torch.inference_mode():
            result = model.generate(input_ids=torch.tensor([[11, 12]]), attention_mask=torch.ones(1, 2, dtype=torch.long),
                                    num_beams=3, do_sample=False, max_new_tokens=8, logits_processor=[processor])
        self.assertIn(result[0, 2:].tolist(), [[3, 2], [4, 5, 2], [6, 7, 8, 2]])

    def test_v2_class_mode_can_declare_the_same_beam_search_for_all_generalists(self):
        original, study, samples = fixtures()
        policy = copy.deepcopy(original.policy)
        policy['tasks']['image'].update(generative_decoding=CLASS_NAME_DECODING_V2,
                                       generation_controls={'num_beams': 3})
        adapter = CompactAdapter(policy, original.classmap)
        for candidate in study['generalists']:
            self.assertEqual(adapter.build(candidate, samples['image'])['generation']['num_beams'], 3)

    def test_prefix_does_not_reenable_hidden_minicpm_input_truncation(self):
        calls = []
        def processor(text, **kwargs):
            calls.append((text, kwargs))
            return {'input_ids': SimpleNamespace(shape=(1, 101))}
        wrapper = CompleteMiniProcessor(processor, 100, 'Class name:\n')
        with self.assertRaisesRegex(ValueError, 'full multimodal input'):
            wrapper(['original full image prompt'], max_length=1, truncation=True)
        self.assertEqual(calls[0][0], ['original full image promptClass name:\n'])
        self.assertIsNone(calls[0][1]['max_length'])
        self.assertFalse(calls[0][1]['truncation'])

    def test_unvalidated_deepseek_prompt_cannot_silently_enter_formal_experiments(self):
        original, study, samples = fixtures()
        policy = copy.deepcopy(original.policy)
        policy['tasks']['ocr']['deepseek_prompt'] = '<image>\nNew unvalidated long instruction'
        adapter = CompactAdapter(policy, original.classmap)
        with self.assertRaisesRegex(ValueError, 'diagnostic-only'):
            adapter.build(study['specialists'][2], samples['ocr'])
        policy['diagnostic_only'] = True
        adapter = CompactAdapter(policy, original.classmap)
        self.assertIn('New unvalidated', adapter.build(study['specialists'][2], samples['ocr'])['native']['options']['prompt'])

    def test_deferred_scratch_counts_all_beams_without_shrinking_the_context_budget(self):
        from contextlib import contextmanager
        from unittest.mock import patch
        native_fixtures.NativeBindingTests.setUpClass()
        helper = native_fixtures.NativeBindingTests()
        original, study, samples = fixtures()
        mini = study['generalists'][1]
        policy = copy.deepcopy(original.policy)
        policy['native_decoding_presets'] = {mini['model']: {'asr': 'minicpm_deterministic_chat_v1'}}
        observed = []
        @contextmanager
        def scratch(model, enabled, active_rows, torch, **kwargs):
            observed.append((enabled, active_rows))
            yield
        with helper.no_cuda(), tempfile.TemporaryDirectory() as folder:
            executor, processor, model = helper.binding(mini, samples['asr'], folder)
            executor.adapter = CompactAdapter(policy, original.classmap)
            executor.input_limit = 16384
            executor.memory_options = {'defer_exllama_scratch': True}
            with patch('gsdeploy.backends.compact_native.exllama_request_scratch', scratch):
                result = executor.infer(samples['asr'])
            self.assertTrue(result['success'])
            self.assertEqual(observed, [(True, 49152)])
            self.assertEqual(executor.input_limit, 16384)


if __name__ == '__main__':
    unittest.main()
