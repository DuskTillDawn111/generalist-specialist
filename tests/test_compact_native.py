"""Exercise all native bindings with CPU tensors; no checkpoint or CUDA calls."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.backends.compact_native import (NativeExecutor, CompleteMiniProcessor, capture_generation,
                                            load_native, label_token_trie, label_prefix_constraint,
                                            ClassLabelLogitsProcessor)
from tests.test_compact_protocol import fixtures


class NativeBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch = torch
        except ImportError:
            raise unittest.SkipTest('CPU tensor dependency unavailable')

    @contextmanager
    def no_cuda(self):
        def forbidden(*args, **kwargs):
            raise AssertionError('CPU binding tests must not call CUDA')
        with patch.object(self.torch.cuda, '_lazy_init', forbidden), patch.object(self.torch.Tensor, 'cuda', forbidden):
            yield

    def binding(self, candidate, sample, directory):
        torch = self.torch
        raw = {'text': 'Paris', 'image': 'n00000042', 'ocr': 'ITEM PRICE A, B -1.20', 'asr': "I can't go"}[sample['task']]
        class Tokenizer:
            eos_token_id = 2
            def encode(self, label, **kwargs):
                return [100 + int(label[1:])]
            def decode(self, ids, **kwargs):
                return raw
        tokenizer = Tokenizer()
        class Processor:
            last = None
            def __init__(self):
                self.tokenizer = tokenizer
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return 'native formatted prompt'
            def get_decoder_prompt_ids(self, **kwargs):
                return [(1, 3), (2, 4), (3, 5)]
            def __call__(self, *args, **kwargs):
                self.last = kwargs
                if isinstance(kwargs.get('text'), list) and kwargs['text'] and kwargs['text'][0].startswith('a photo'):
                    import re
                    indexes = [int(re.search(r'class (\d+)', text).group(1)) for text in kwargs['text']]
                    return {'input_ids': torch.tensor(indexes).reshape(-1, 1)}
                if 'images' in kwargs and 'text' not in kwargs:
                    return {'pixel_values': torch.ones(1, 2)}
                return {'input_ids': torch.tensor([[10, 11, 12]]),
                        'attention_mask': torch.ones(1, 3, dtype=torch.long),
                        'pixel_values': torch.ones(1, 2)}
        processor = Processor()
        class Model:
            def __init__(self):
                self.config = SimpleNamespace(max_position_embeddings=32768, decoder_start_token_id=0,
                                              max_target_positions=448)
                self.generation_config = SimpleNamespace(eos_token_id=2)
                self.llm = SimpleNamespace(generate=self.llm_generate, generation_config=self.generation_config)
                self.calls = []
            def generate(self, *args, **kwargs):
                self.calls.append(kwargs)
                if candidate['slot'] == 'asr':
                    prefix = kwargs['decoder_input_ids'][0].tolist()
                elif candidate['slot'] == 'text':
                    prefix = [0]
                else:
                    value = kwargs.get('input_ids', args[0] if args else None)
                    prefix = value[0].tolist()
                sequence = torch.tensor([prefix + [7, 2]])
                return SimpleNamespace(sequences=sequence) if kwargs.get('return_dict_in_generate') else sequence
            def llm_generate(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(sequences=torch.tensor([[7, 2]]))
            def chat(self, msgs, processor, **kwargs):
                processor(['expanded multimodal prompt'], max_length=32768)
                self.llm.generate(inputs_embeds=torch.ones(1, 4, 2), eos_token_id=[2], num_beams=3,
                                  return_dict_in_generate=True)
                return raw
            def infer(self, tokenizer, output_path, **kwargs):
                if not Path(output_path).is_dir():
                    raise AssertionError('Missing isolated output directory')
                self.generate(torch.tensor([[10, 11, 12]]), max_new_tokens=8192, temperature=0.0)
                return raw
            def get_text_features(self, input_ids):
                values = torch.zeros(len(input_ids), 2)
                for i, index in enumerate(input_ids[:, 0].tolist()):
                    values[i, 1 if index == 42 else 0] = 1
                return values
            def get_image_features(self, pixel_values):
                return torch.tensor([[0.0, 1.0]])
        model = Model()
        adapter, _, _ = fixtures()
        executor = NativeExecutor(adapter, candidate, model, processor, torch=torch, device='cpu',
                                  dtype=torch.float32, image_loader=lambda _: 'CPU image fixture',
                                  audio_loader=lambda _: [0.0] * 80000, output_root=directory)
        return executor, processor, model

    def test_all_sixteen_bindings_capture_eos_and_score_without_cuda(self):
        adapter, study, samples = fixtures()
        samples['image']['label'] = 'n00000042'
        paths = 0
        with self.no_cuda(), tempfile.TemporaryDirectory() as directory:
            for candidate in study['generalists'] + study['specialists']:
                for task in adapter.supported_tasks(candidate):
                    with self.subTest(model=candidate['model'], task=task):
                        executor, processor, model = self.binding(candidate, samples[task], directory)
                        result = executor.infer(samples[task])
                        self.assertTrue(result['success'], result)
                        self.assertEqual(result['score'].get('errors', 0), 0)
                        if task == 'image':
                            self.assertTrue(result['score']['correct'])
                        if task == 'text':
                            self.assertEqual(result['score']['EM'], 1)
                        if candidate['slot'] == 'generalist_2':
                            self.assertIsNone(processor.last['max_length'])
                            self.assertFalse(processor.last['truncation'])
                            self.assertEqual(model.calls[-1]['num_beams'], 1)
                            self.assertFalse(model.calls[-1]['do_sample'])
                        if candidate['slot'] == 'generalist_1':
                            self.assertIn('thinker_max_new_tokens', model.calls[-1])
                            self.assertFalse(model.calls[-1]['return_audio'])
                        paths += 1
        self.assertEqual(paths, 16)

    def test_capture_restores_native_method_even_on_failure(self):
        owner = SimpleNamespace(generate=lambda **kwargs: (_ for _ in ()).throw(RuntimeError('native failure')))
        original = owner.generate
        with self.assertRaises(RuntimeError):
            with capture_generation(owner, None, {'max_new_tokens': 3, 'do_sample': False, 'num_beams': 1}):
                owner.generate()
        self.assertIs(owner.generate, original)

    def test_class_constraint_preserves_full_catalog_and_requires_eos(self):
        class Tokenizer:
            def encode(self, label, **kwargs):
                return [int(c) for c in label]
        trie = label_token_trie(Tokenizer(), ['12', '13', '24'], [9], 3)
        for prompt in [[], [7, 8, 7]]:
            allowed = label_prefix_constraint(trie)
            tensor = lambda tokens: self.torch.tensor(tokens, dtype=self.torch.long)
            self.assertEqual(allowed(0, tensor(prompt)), [1, 2])
            self.assertEqual(allowed(0, tensor(prompt + [1])), [2, 3])
            self.assertEqual(allowed(0, tensor(prompt + [1, 3])), [9])
            with self.assertRaisesRegex(ValueError, 'left'):
                allowed(0, tensor(prompt + [1, 4]))

    def test_class_constraint_rejects_ambiguous_or_over_budget_tokenizations(self):
        tokenizer = SimpleNamespace(encode=lambda label, **kwargs: [1, 2])
        for labels, budget in [(['a', 'b'], 3), (['a'], 2)]:
            with self.assertRaises(ValueError):
                label_token_trie(tokenizer, labels, [9], budget)

    def test_class_logits_constraint_accepts_empty_native_decoder_sequence(self):
        torch = self.torch
        trie = label_token_trie(SimpleNamespace(encode=lambda label, **kwargs: [1, int(label)]), ['2', '3'], [9], 3)
        processor = ClassLabelLogitsProcessor(trie)
        scores = torch.arange(10, dtype=torch.float32).unsqueeze(0)
        masked = processor(torch.empty((1, 0), dtype=torch.long), scores)
        self.assertEqual(torch.isfinite(masked).nonzero()[:, 1].tolist(), [1])
        masked = processor(torch.tensor([[1]]), scores)
        self.assertEqual(torch.isfinite(masked).nonzero()[:, 1].tolist(), [2, 3])
        self.assertEqual(masked.argmax(-1).item(), 3)
        masked = processor(torch.tensor([[1, 3]]), scores)
        self.assertEqual(torch.isfinite(masked).nonzero()[:, 1].tolist(), [9])

    def test_whisper_pads_complete_audio_and_retains_transcription_prefix(self):
        adapter, study, samples = fixtures()
        candidate = next(c for c in study['specialists'] if c['slot'] == 'asr')
        with self.no_cuda(), tempfile.TemporaryDirectory() as directory:
            executor, processor, model = self.binding(candidate, samples['asr'], directory)
            result = executor.infer(samples['asr'])
            self.assertTrue(result['success'])
            self.assertEqual(processor.last['padding'], 'max_length')
            self.assertEqual(processor.last['max_length'], 480000)
            self.assertFalse(processor.last['truncation'])
            self.assertEqual(model.calls[-1]['language'], 'en')
            self.assertEqual(model.calls[-1]['task'], 'transcribe')
            self.assertTrue(model.calls[-1]['return_dict_in_generate'])
            self.assertTrue(model.calls[-1]['force_unique_generate_call'])
            self.assertEqual(result['generated_tokens'], 2)

    def test_qwen_preview_template_batch_is_not_wrapped_twice(self):
        adapter, study, samples = fixtures()
        with self.no_cuda(), tempfile.TemporaryDirectory() as directory:
            executor, processor, _ = self.binding(study['generalists'][0], samples['text'], directory)
            processor.apply_chat_template = lambda *args, **kwargs: ['one native conversation']
            executor.prepare(adapter.build(study['generalists'][0], samples['text']))
            self.assertEqual(processor.last['text'], ['one native conversation'])

    def test_qwen_audio_processor_cannot_mutate_frozen_request_identity(self):
        from gsdeploy.backends.compact import fingerprint
        adapter, study, samples = fixtures()
        with self.no_cuda(), tempfile.TemporaryDirectory() as directory:
            executor, processor, _ = self.binding(study['generalists'][0], samples['asr'], directory)
            class MutatingProcessor(type(processor)):
                def __call__(self, *args, **kwargs):
                    self.consumed = kwargs['audio_kwargs'].pop('max_length')
                    return super().__call__(*args, **kwargs)
            executor.processor = MutatingProcessor()
            request = adapter.build(study['generalists'][0], samples['asr'])
            before = fingerprint(request)
            executor.prepare(request)
            self.assertEqual(executor.processor.consumed, 480000)
            self.assertEqual(fingerprint(request), before)

    def test_measurement_paths_do_not_compute_gold_reference_quality(self):
        adapter, study, samples = fixtures()
        with self.no_cuda(), tempfile.TemporaryDirectory() as directory, \
                patch('gsdeploy.backends.compact.score_result', side_effect=AssertionError('Gold scoring entered serving path')):
            for candidate in study['generalists'] + study['specialists']:
                for task in adapter.supported_tasks(candidate):
                    executor, _, _ = self.binding(candidate, samples[task], directory)
                    result = executor.infer(samples[task], evaluate_quality=False)
                    self.assertTrue(result['success'])
                    self.assertNotIn('score', result)

    def test_native_input_overflow_rejected_before_generation(self):
        torch = self.torch
        class Processor:
            def __call__(self, *args, **kwargs):
                self.kwargs = kwargs
                return {'input_ids': torch.zeros(1, 6, dtype=torch.long)}
        processor = Processor()
        wrapped = CompleteMiniProcessor(processor, 5)
        with self.assertRaises(ValueError):
            wrapped('long input', max_length=5)
        self.assertIsNone(processor.kwargs['max_length'])
        self.assertFalse(processor.kwargs['truncation'])

    def test_gpu_loader_refuses_before_importing_or_reading_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, 'GPU model loading is disabled'):
            load_native(None, None, None, None)

    def test_binding_keeps_integer_ids_and_casts_only_floating_inputs(self):
        adapter, study, samples = fixtures()
        with self.no_cuda(), tempfile.TemporaryDirectory() as directory:
            executor, _, _ = self.binding(study['generalists'][0], samples['text'], directory)
            executor.dtype = self.torch.float64
            result = executor.move({'input_ids': self.torch.ones(1, 2, dtype=self.torch.long),
                                    'pixel_values': self.torch.ones(1, 2),
                                    'image_attention_mask': self.torch.ones(1, 4897)})
            self.assertEqual(result['input_ids'].dtype, self.torch.long)
            self.assertEqual(result['pixel_values'].dtype, self.torch.float64)
            self.assertEqual(result['image_attention_mask'].dtype, self.torch.float32)
            self.assertEqual(result['image_attention_mask'].sum().item(), 4897)


class CompactTransportFailureTests(unittest.TestCase):
    def test_explicit_inference_failure_is_not_a_completed_runtime_request(self):
        import threading
        import time
        from gsdeploy.runtime.deployment import DeploymentRuntime
        runtime = DeploymentRuntime(SimpleNamespace(), {})
        row = {'request_id': 'r', 'model': 'm', 'task': 'ocr', 'status': 'submitted'}
        runtime.pending['r'] = row
        thread = threading.Thread(target=runtime._collect, daemon=True)
        thread.start()
        runtime.events.put(('m', 'RESULT', {'request_id': 'r', 'task_type': 'ocr', 'success': False,
                                          'failure_reason': 'GENERATION_LENGTH', 'raw_output': 'retained text'}))
        deadline = time.monotonic() + 2
        while row['status'] == 'submitted' and time.monotonic() < deadline:
            time.sleep(.005)
        runtime.collect_stop.set()
        thread.join(1)
        self.assertEqual(row['status'], 'error')
        self.assertEqual(row['worker_response']['raw_output'], 'retained text')


if __name__ == '__main__':
    unittest.main()
