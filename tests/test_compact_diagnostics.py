"""Diagnostic output must never become formal quality or admission evidence."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from gsdeploy.backends.compact import CompactAdapter, NATIVE_IMAGE_OBSERVATION
from gsdeploy.backends.compact_native import NativeExecutor, capture_generation, generation_audit
from gsdeploy.profiling.locking import gpu_lease
from gsdeploy.tasks import score_result, summarize_quality, parse_result
from tests.test_compact_protocol import fixtures
from tests import test_compact_native as native_fixtures

EOS = {'finish_reason': 'eos', 'generated_tokens': 4, 'terminal_token_id': 2, 'eos_token_ids': [2]}


class DiagnosticBindingTests(unittest.TestCase):
    def adapter(self):
        original, study, samples = fixtures()
        policy = copy.deepcopy(original.policy)
        policy['diagnostic_only'] = True
        policy['tasks']['image'].update(generative_decoding=NATIVE_IMAGE_OBSERVATION,
                                       catalog_presentation='decoder_only')
        policy['model_generation_controls'] = {
            study['generalists'][1]['model']: {'image': {'num_beams': 3, 'repetition_penalty': 1.2}}}
        return CompactAdapter(policy, original.classmap), study, samples

    def test_free_image_observation_keeps_eos_and_is_rejected_by_quality_scoring(self):
        adapter, study, samples = self.adapter()
        request = adapter.build(study['generalists'][1], samples['image'])
        result = adapter.finish(request, samples['image'], 'a bird on a branch', **EOS)
        self.assertTrue(result['success'])
        self.assertFalse(result['quality_eligible'])
        self.assertTrue(result['diagnostic_only'])
        self.assertNotIn('score', result)
        self.assertEqual(result['prediction'], 'a bird on a branch')
        for fn in [lambda: score_result(samples['image'], result),
                   lambda: summarize_quality([samples['image']], [result])]:
            with self.assertRaisesRegex(ValueError, 'observations'):
                fn()
        self.assertNotIn('LABEL CATALOG', request['instruction'])
        changed = copy.deepcopy(samples['image'])
        changed['label'] = 'n99999999'
        self.assertEqual(request, adapter.build(study['generalists'][1], changed))

    def test_observation_does_not_turn_empty_length_or_unknown_into_success(self):
        adapter, study, samples = self.adapter()
        request = adapter.build(study['generalists'][0], samples['image'])
        self.assertFalse(adapter.finish(request, samples['image'], '', **EOS)['success'])
        for reason in ['length', 'unknown']:
            observation = {**EOS, 'finish_reason': reason}
            self.assertFalse(adapter.finish(request, samples['image'], 'a bird', **observation)['success'])
        with self.assertRaises(ValueError):
            parse_result('asr', 'hello', max_new_tokens=32, diagnostic_only=True, **EOS)

    def test_native_beam_override_is_fingerprinted_and_formal_protocols_reject_it(self):
        adapter, study, samples = self.adapter()
        candidate = study['generalists'][1]
        request = adapter.build(candidate, samples['image'])
        self.assertEqual(request['generation']['num_beams'], 3)
        self.assertEqual(request['native']['options']['num_beams'], 3)
        self.assertEqual(request['native']['options']['repetition_penalty'], 1.2)
        request['generation']['num_beams'] = 1
        with self.assertRaisesRegex(ValueError, 'generation policy changed'):
            adapter.finish(request, samples['image'], 'bird', **EOS)
        for key in ['diagnostic_only', 'model_generation_controls']:
            policy = copy.deepcopy(adapter.policy)
            if key == 'diagnostic_only':
                policy[key] = False
            else:
                policy[key][candidate['model']]['image']['max_new_tokens'] = 999
            with self.assertRaises(ValueError):
                CompactAdapter(policy, adapter.classmap)

    def test_native_bindings_preserve_beams_without_installing_class_constraint(self):
        # Reuse the CPU-only native binding fixtures, with actual tensor shapes.
        native_fixtures.NativeBindingTests.setUpClass()
        helper = native_fixtures.NativeBindingTests()
        adapter, study, samples = self.adapter()
        with helper.no_cuda(), tempfile.TemporaryDirectory() as directory:
            for candidate in study['generalists']:
                executor, processor, model = helper.binding(candidate, samples['image'], directory)
                executor.adapter = adapter
                result = executor.infer(samples['image'])
                self.assertFalse(result['quality_eligible'])
                self.assertNotIn('logits_processor', model.calls[-1])
                key = 'thinker_num_beams' if candidate['slot'] == 'generalist_1' else 'num_beams'
                self.assertEqual(model.calls[-1][key], 3 if candidate['slot'] == 'generalist_2' else 1)

    def test_actual_image_and_audio_bytes_are_verified_before_model_preparation(self):
        from PIL import Image
        import soundfile as sf
        adapter, study, samples = self.adapter()
        executor = NativeExecutor(adapter, study['generalists'][0], None, None,
                                  torch=None, device='cpu', dtype=None)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'image.png'
            Image.new('RGB', (10, 7), (12, 34, 56)).save(image)
            audio = Path(directory) / 'audio.wav'
            sf.write(str(audio), [0.0] * 80000, 16000)
            for task, path in [('image', image), ('asr', audio)]:
                sample = copy.deepcopy(samples[task])
                sample.update(input=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                request = adapter.build(study['generalists'][0], sample)
                value = executor.media(request)
                binding = executor.last_media_binding
                self.assertEqual(binding['sha256'], sample['input_sha256'])
                if task == 'image':
                    self.assertEqual(value.size, (10, 7))
                    self.assertEqual(binding['size'], [10, 7])
                else:
                    self.assertEqual(len(value), 80000)
                    self.assertEqual(binding['sampling_rate'], 16000)
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'bytes changed'):
                    executor.media(request)

    def test_effective_native_generation_audit_retains_native_defaults_and_overrides(self):
        from types import SimpleNamespace
        native_fixtures.NativeBindingTests.setUpClass()
        torch = native_fixtures.NativeBindingTests.torch
        owner = SimpleNamespace(generate=lambda **kwargs: torch.tensor([[7, 2]]),
                                generation_config=SimpleNamespace(eos_token_id=2))
        tokenizer = SimpleNamespace(eos_token_id=2)
        declared = {'do_sample': False, 'num_beams': 3, 'repetition_penalty': 1.0, 'max_new_tokens': 32}
        with capture_generation(owner, tokenizer, declared) as calls:
            owner.generate(inputs_embeds=torch.ones(1, 4, 2), num_beams=1,
                           repetition_penalty=1.2, no_repeat_ngram_size=35)
        self.assertEqual(calls[0]['effective_generation'], {**declared, 'no_repeat_ngram_size': 35, 'class_constraint': False})
        self.assertEqual(generation_audit({'thinker_num_beams': 3, 'thinker_max_new_tokens': 32}),
                         {'num_beams': 3, 'max_new_tokens': 32, 'class_constraint': False})

    def test_gpu_lease_is_global_and_rejects_another_controller(self):
        with gpu_lease():
            with self.assertRaises(BlockingIOError):
                with gpu_lease():
                    self.fail('A second result tree must not acquire the same GPU')


if __name__ == '__main__':
    unittest.main()
