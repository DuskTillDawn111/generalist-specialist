"""Train/evaluation isolation and answer-only LoRA updates on a tiny CPU model."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
prepare = module('prepare_chatqa_training')
training = module('train_chatqa_lora')


class TrainOnlyQATests(unittest.TestCase):
    def document(self):
        return {'data': [{'title': 'Train_article', 'paragraphs': [{'context': 'Paris is a city.', 'qas': [
            {'id': 'yes', 'question': 'Which city?', 'is_impossible': False,
             'answers': [{'text': 'Paris', 'answer_start': 0}]},
            {'id': 'no', 'question': 'Who founded it?', 'is_impossible': True, 'answers': []}]}]}]}

    def test_training_split_and_targets_are_separate_from_evaluation(self):
        groups = prepare.train_questions(self.document(), {'eval'}, {'Eval_article'}, set())
        self.assertEqual(groups[True][0]['target'], 'Paris')
        self.assertEqual(groups[False][0]['target'], '<No Answer>')
        self.assertEqual(groups[False][0]['dataset'], 'SQuAD2-train')
        for ids, articles, contexts in [({'yes'}, set(), set()), (set(), {'Train_article'}, set()),
                (set(), set(), {hashlib.sha256(b'Paris is a city.').hexdigest()})]:
            with self.assertRaisesRegex(ValueError, 'overlap'):
                prepare.train_questions(self.document(), ids, articles, contexts)

    def test_complete_prompt_and_padding_never_become_training_targets(self):
        import torch
        class Tokenizer:
            bos_token, eos_token_id = 'B', 2
            def encode(self, text, **kwargs):
                return [ord(c) for c in text]
        ids, labels = prepare.training_ids(Tokenizer(), 'prompt', 'answer')
        self.assertEqual(ids[:-1], [ord(c) for c in 'Bprompt answer'])
        self.assertEqual(labels[:7], [-100] * 7)
        self.assertEqual(labels[-1], 2)
        rows = [{'input_ids': [1, 3, 7, 2], 'labels': [-100, -100, 7, 2]},
                {'input_ids': [1, 5, 2], 'labels': [-100, 5, 2]}]
        before = copy.deepcopy(rows)
        result = training.collate(rows, torch, 2)
        self.assertEqual(result['attention_mask'].tolist(), [[1,1,1,1],[1,1,1,0]])
        self.assertEqual(result['labels'].tolist()[1][-1], -100)
        self.assertEqual(rows, before)

    def test_real_peft_cpu_step_updates_only_new_adapter_parameters(self):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(20261018)
        base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
               num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
               max_position_embeddings=64, bos_token_id=1, eos_token_id=2))
        model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4,
                               lora_dropout=0, target_modules=['q_proj','v_proj']))
        frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        original = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        batch = training.collate([{'input_ids':[1,3,4,5,2], 'labels':[-100,-100,-100,5,2]},
                                 {'input_ids':[1,6,7,2], 'labels':[-100,-100,7,2]}], torch, 2)
        model.train()
        loss = model(**batch).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        after = dict(model.named_parameters())
        self.assertTrue(all(torch.equal(value, after[name]) for name,value in frozen.items()))
        self.assertTrue(any(not torch.equal(value, after[name]) for name,value in original.items()))


if __name__ == '__main__':
    unittest.main()
