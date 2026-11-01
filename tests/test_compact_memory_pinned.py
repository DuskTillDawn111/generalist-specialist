"""Run in compact-minicpm to test its actual pinned eager/SDPA implementations."""
from types import SimpleNamespace
import unittest


class PinnedMiniAttentionTests(unittest.TestCase):
    def test_native_owner_resolution_with_actual_pinned_gptq_wrapper(self):
        try:
            import torch
            from auto_gptq.modeling._base import BaseGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            raise unittest.SkipTest('Run with the pinned compact-minicpm interpreter')
        from gsdeploy.backends.compact_memory import minicpm_native_model, retire_minicpm_audio_output
        native = torch.nn.Module()
        native.config = SimpleNamespace(model_type='minicpmo', init_tts=True)
        names = ('llm', 'vpm', 'apm', 'audio_projection_layer', 'resampler')
        for name in (*names, 'tts', 'vocos'):
            setattr(native, name, torch.nn.Linear(2, 2))
        native.tts_processor = object()
        required = {name: getattr(native, name) for name in names}
        wrapper = BaseGPTQForCausalLM(native, quantized=True, quantize_config=BaseQuantizeConfig())
        self.assertTrue(hasattr(wrapper, 'llm'))
        self.assertNotIn('llm', vars(wrapper).get('_modules', {}))
        self.assertIs(minicpm_native_model(wrapper), native)
        self.assertIs(minicpm_native_model(native), native)
        result = retire_minicpm_audio_output(minicpm_native_model(wrapper))
        self.assertEqual(result['retired_parameter_bytes'], 48)
        self.assertFalse(hasattr(native, 'tts'))
        self.assertFalse(hasattr(wrapper, 'tts'))
        self.assertIs(wrapper.model, native)
        self.assertTrue(all(getattr(native, n) is m for n, m in required.items()))

    def test_sdpa_switch_preserves_projection_parameters_and_full_prefix_logits(self):
        try:
            import torch
            from transformers import Qwen2Config, Qwen2ForCausalLM
            from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2SdpaAttention
        except ImportError:
            raise unittest.SkipTest('Run with the pinned compact-minicpm interpreter')
        from gsdeploy.backends.compact_memory import configure_minicpm_language_sdpa, last_position_logits
        torch.manual_seed(42)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=64, attn_implementation='eager',
                            eos_token_id=63, bos_token_id=1, pad_token_id=0)
        model = Qwen2ForCausalLM(config).eval()
        if type(model.model.layers[0].self_attn) is not Qwen2Attention:
            self.skipTest('This interpreter is not the pinned Qwen2 eager implementation')
        ids = torch.tensor([[1, 3, 5, 7, 9, 11]])
        mask = torch.ones_like(ids)
        tensors = {n: p.data_ptr() for n, p in model.named_parameters()}
        with torch.inference_mode():
            expected = model(ids, attention_mask=mask).logits
            tokens = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False)
            audit = configure_minicpm_language_sdpa(SimpleNamespace(llm=model))
            actual = model(ids, attention_mask=mask).logits
            torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)
            with last_position_logits(model, True):
                projected = model(ids, attention_mask=mask).logits
                torch.testing.assert_close(projected[:, -1], actual[:, -1], atol=1e-6, rtol=1e-5)
                result = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False)
            self.assertTrue(torch.equal(tokens, result))
        self.assertEqual(tensors, {n: p.data_ptr() for n, p in model.named_parameters()})
        self.assertEqual(audit['actual_attention_classes'], ['Qwen2SdpaAttention'])


if __name__ == '__main__':
    unittest.main()
