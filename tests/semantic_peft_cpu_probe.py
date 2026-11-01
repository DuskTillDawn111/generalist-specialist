"""Explicit tiny random PEFT generate check; requires CUDA_VISIBLE_DEVICES=''."""
import inspect
import json
import os
from types import SimpleNamespace

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' before importing PEFT dependencies")

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from tools.predictor_validation_20261027.semantic_observer import language_module, observe_language_forwards

torch.set_num_threads(1)
torch.manual_seed(527)
base = LlamaForCausalLM(LlamaConfig(
    vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
    bos_token_id=1, eos_token_id=None, pad_token_id=0))
model = get_peft_model(base, LoraConfig(
    r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM))
model.eval()
executor = SimpleNamespace(family="chatqa", model=model)
assert language_module(executor) is base
signatures = [inspect.signature(fn) for fn in (model.forward, base.forward, model.generate)]
inputs = torch.tensor([[1, 4, 8, 3]], dtype=torch.long)
kwargs = dict(input_ids=inputs, attention_mask=torch.ones_like(inputs),
              max_new_tokens=3, do_sample=False, eos_token_id=None, pad_token_id=0)
with torch.inference_mode():
    expected = model.generate(**kwargs)
    with observe_language_forwards(executor, "cpu:peft:probe") as observer:
        actual = model.generate(**kwargs)
assert torch.equal(expected, actual)
assert [r["semantic_phase"] for r in observer.records] == ["prefill", "decode", "decode"]
assert [r["input_sequence_length"] for r in observer.records] == [4, 1, 1]
assert signatures == [inspect.signature(fn) for fn in (model.forward, base.forward, model.generate)]
assert not base._forward_hooks and not base._forward_pre_hooks
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status="PASS", CUDA_initialized=False, checkpoint_loaded=False,
                     random_model_only=True, output_equal=True,
                     observation=observer.document())))
