"""New QA candidate: native formatting, complete context and actual EOS IDs."""
import copy
from types import SimpleNamespace
import unittest

from gsdeploy.backends.compact import CompactAdapter
from gsdeploy.backends.compact_native import NativeExecutor
from tests.test_compact_protocol import fixtures


class ChatQABindingTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.adapter, _, self.samples = fixtures()
        self.candidate = {'slot': 'text', 'model': 'nvidia/Llama3-ChatQA-1.5-8B',
                          'revision': '3b98162e3f97550d62aeeb19ea50208f968c678a', 'precision': 'NF4 candidate'}

    def executor(self, terminal=3, limit=8192):
        torch = self.torch
        class Tokenizer:
            bos_token, bos_token_id, eos_token_id, unk_token_id = '<BOS>', 1, 2, None
            def __call__(self, text, **options):
                self.text, self.options = text, options
                return {'input_ids': torch.tensor([[1, 9, 10]]), 'attention_mask': torch.ones((1, 3), dtype=torch.long)}
            def convert_tokens_to_ids(self, token):
                return {'<|end_of_text|>': 2, '<|eot_id|>': 3}.get(token)
            def convert_ids_to_tokens(self, token):
                return {2: '<|end_of_text|>', 3: '<|eot_id|>'}.get(token)
            def decode(self, ids, **kwargs):
                self.decoded = ids
                return 'Paris'
        class Model:
            config = SimpleNamespace(max_position_embeddings=limit)
            # Test that inference honors the explicit native terminator list,
            # even if the saved generic generation config knows only one EOS.
            generation_config = SimpleNamespace(eos_token_id=2)
            def generate(self, **kwargs):
                self.options = kwargs
                return torch.tensor([[1, 9, 10, 7, terminal]])
        tokenizer, model = Tokenizer(), Model()
        return NativeExecutor(self.adapter, self.candidate, model, tokenizer, torch=torch,
                              device='cpu', dtype=torch.float32), tokenizer, model

    def test_full_case_preserving_context_without_answer_leakage(self):
        request = self.adapter.build(self.candidate, self.samples['text'])
        native = request['native']
        self.assertEqual(self.adapter.supported_tasks(self.candidate), ('text',))
        self.assertIn('\n\nThe city is Paris.\n\nUser: Which city?\n\nAssistant:', native['text'])
        changed = copy.deepcopy(self.samples['text'])
        changed.update(answers=['SECRET'], label='SECRET', is_impossible=True)
        self.assertEqual(request, self.adapter.build(self.candidate, changed))
        with self.assertRaises(ValueError):
            self.adapter.build(self.candidate, self.samples['image'])

    def test_causal_prefix_and_both_native_terminators(self):
        for terminal in [2, 3]:
            executor, tokenizer, model = self.executor(terminal)
            result = executor.infer(self.samples['text'])
            self.assertTrue(result['success'], result)
            self.assertEqual(result['score']['EM'], 1)
            self.assertEqual(tokenizer.decoded, [7, terminal])
            self.assertEqual(result['generated_tokens'], 2)
            self.assertEqual(result['eos_token_ids'], [2, 3])
            self.assertEqual(model.options['eos_token_id'], [2, 3])
            self.assertEqual(result['effective_generation']['eos_token_id'], [2, 3])
            self.assertFalse(tokenizer.options['add_special_tokens'])
            self.assertFalse(tokenizer.options['truncation'])
            self.assertTrue(tokenizer.text.startswith('<BOS>System: '))

    def test_full_input_plus_output_overflow_rejected(self):
        executor, _, _ = self.executor(limit=3)
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            executor.infer(self.samples['text'])

    def test_unknown_eot_token_rejected(self):
        executor, tokenizer, _ = self.executor()
        tokenizer.convert_tokens_to_ids = lambda _: None
        with self.assertRaisesRegex(ValueError, 'terminator'):
            executor.infer(self.samples['text'])

    def test_native_system_format_keeps_task_request_in_the_user_turn(self):
        policy = copy.deepcopy(self.adapter.policy)
        policy['tasks']['text']['chatqa_prompt_format'] = 'native_system_v2'
        adapter = CompactAdapter(policy, self.adapter.classmap)
        request = adapter.build(self.candidate, self.samples['text'])
        text = request['native']['text']
        self.assertTrue(text.startswith('System: This is a chat'))
        self.assertIn('\n\nThe city is Paris.\n\nUser: ' + policy['tasks']['text']['instruction'] + ' Which city?', text)
        self.assertTrue(text.endswith('\n\nAssistant:'))
        self.assertEqual(request['generation'], self.adapter.build(self.candidate, self.samples['text'])['generation'])
        policy['tasks']['text']['chatqa_prompt_format'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'ChatQA prompt format'):
            CompactAdapter(policy, self.adapter.classmap).build(self.candidate, self.samples['text'])


if __name__ == '__main__':
    unittest.main()
