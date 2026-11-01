"""CPU mathematical/ownership checks for the new inference memory policies."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gsdeploy.backends.compact_memory import (last_position_logits, install_phi_vision_microbatch,
                                             retire_minicpm_audio_output, reclaim_unused_cache,
                                             allocator_environment, install_linear_row_microbatch)


class CompactMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch = torch
        except ImportError:
            raise unittest.SkipTest('CPU tensor dependency unavailable')

    def test_projection_slices_only_last_position_and_preserves_greedy_outputs(self):
        torch = self.torch
        torch.manual_seed(42)
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(31, 8)
                self.lm_head = torch.nn.Linear(8, 31)
                self.input_lengths, self.projected_lengths = [], []
                self.lm_head.register_forward_hook(lambda m, a, output: self.projected_lengths.append(output.shape[1]))
            def forward(self, ids):
                self.input_lengths.append(ids.shape[1])
                # The full prefix participates in the hidden state computation.
                return self.lm_head(self.embedding(ids).cumsum(dim=1))
            def generate(self, ids):
                for _ in range(5):
                    token = self(ids)[:, -1].argmax(-1, keepdim=True)
                    ids = torch.cat([ids, token], dim=-1)
                return ids
        model = Decoder().eval()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.inference_mode():
            expected = model.generate(ids)
            lengths = list(model.input_lengths)
            model.input_lengths.clear()
            model.projected_lengths.clear()
            with last_position_logits(model, True):
                actual = model.generate(ids)
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(model.input_lengths, lengths)
            self.assertEqual(model.projected_lengths, [1] * 5)
            self.assertEqual(model(ids).shape, (1, 5, 31))
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with last_position_logits(model, True):
                    raise RuntimeError('test failure')
            self.assertEqual(model(ids).shape[1], 5)
        model.train()
        with self.assertRaises(ValueError):
            with last_position_logits(model, True):
                pass

    def test_visual_microbatch_keeps_all_crops_masks_and_order(self):
        torch = self.torch
        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batches = []
            def get_img_features(self, images, attention_mask=None):
                self.batches.append(len(images))
                return images * attention_mask.unsqueeze(-1)
        vision = Vision().eval()
        images = torch.arange(30, dtype=torch.float32).reshape(5, 3, 2)
        mask = torch.tensor([[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0]])
        expected = vision.get_img_features(images, mask)
        vision.batches.clear()
        original = images.clone()
        audit = install_phi_vision_microbatch(vision, 2, torch)
        actual = vision.get_img_features(images, mask)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(images, original))
        self.assertEqual(vision.batches, [2, 2, 1])
        self.assertEqual(audit['crops'], 5)
        self.assertEqual(audit['maximum_batch_seen'], 2)
        with self.assertRaises(ValueError):
            vision.get_img_features(images, mask[:2])

    def test_audio_output_retirement_preserves_all_input_modules(self):
        torch = self.torch
        model = torch.nn.Module()
        model.config = SimpleNamespace(init_tts=True)
        names = ('llm', 'vpm', 'apm', 'audio_projection_layer', 'resampler')
        for name in (*names, 'tts', 'vocos'):
            setattr(model, name, torch.nn.Linear(2, 2))
        model.tts_processor = object()
        required = {name: getattr(model, name) for name in names}
        result = retire_minicpm_audio_output(model)
        self.assertEqual(result['retired_parameter_bytes'], 48)
        self.assertFalse(model.config.init_tts)
        self.assertFalse(hasattr(model, 'tts'))
        self.assertTrue(all(getattr(model, n) is m for n, m in required.items()))

    def test_cache_reclamation_is_real_without_resetting_peak_counters(self):
        events, reserved = [], [1024 * 1024 * 4]
        cuda = SimpleNamespace(memory_reserved=lambda: reserved[0], memory_allocated=lambda: 1024 * 1024,
                               empty_cache=lambda: (events.append('release'), reserved.__setitem__(0, 1024 * 1024)))
        result = reclaim_unused_cache(SimpleNamespace(cuda=cuda), 2)
        self.assertTrue(result['performed'])
        self.assertEqual(events, ['release'])
        self.assertEqual(result['unused_reserved_after_mib'], 0)
        self.assertFalse(reclaim_unused_cache(SimpleNamespace(cuda=cuda), 2)['performed'])

    def test_linear_microbatch_retains_every_row_shape_bias_and_small_decode(self):
        torch = self.torch
        torch.manual_seed(42)
        layer = torch.nn.Linear(8, 13, bias=True).eval()
        layer.infeatures = 8
        inputs = torch.randn(2, 15, 8)
        expected = layer(inputs)
        pointers = [p.data_ptr() for p in layer.parameters()]
        audit = install_linear_row_microbatch(layer, 7, torch)
        with torch.inference_mode():
            actual = layer(inputs)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.shape, (2, 15, 13))
            self.assertEqual(audit['maximum_rows_per_kernel'], 7)
            self.assertEqual(audit['split_calls'], 1)
            self.assertEqual(layer(inputs[:, :1]).shape, (2, 1, 13))
            self.assertEqual(audit['split_calls'], 1)
        self.assertEqual(pointers, [p.data_ptr() for p in layer.parameters()])
        layer.train()
        with self.assertRaisesRegex(ValueError, 'inference-only'):
            layer(inputs)

    def test_reclamation_happens_inside_profile_service_timer(self):
        from gsdeploy.backends.compact_native import ProfileAdapter
        events = []
        cuda = SimpleNamespace(synchronize=lambda: events.append('sync'),
                               memory_reserved=lambda: 4 * 1024**2, memory_allocated=lambda: 1024**2,
                               empty_cache=lambda: events.append('release'))
        adapter = ProfileAdapter.__new__(ProfileAdapter)
        adapter.torch = SimpleNamespace(cuda=cuda)
        adapter.cache_release_threshold_mib = 2
        adapter.executor = SimpleNamespace(infer=lambda *a, **kw: (events.append('infer') or {'success': True}))
        def stamp():
            events.append('stamp')
            return len(events) * 1000000
        with patch('time.perf_counter_ns', side_effect=stamp):
            result = adapter.run({'id': 'sample'})
        self.assertLess(events.index('stamp'), events.index('release'))
        self.assertEqual(events[-1], 'stamp')
        self.assertTrue(result['cache_reclamation']['performed'])

    def test_allocator_policy_is_explicit_and_does_not_import_torch(self):
        self.assertEqual(allocator_environment({}), {})
        self.assertEqual(allocator_environment({'allocator_conf': 'expandable_segments:True'}),
                         {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})


if __name__ == '__main__':
    unittest.main()
