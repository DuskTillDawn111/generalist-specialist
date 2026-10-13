import os
import json
import time
from pathlib import Path
import torch

PROMPT_FILE = Path(
    os.environ.get(
        "GS_PROMPT_FILE",
        str(Path(__file__).resolve().parents[3] / "configs/prompts.json"),
    )
)


def load_task_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts["text_classification"]


def build_user_text(instruction, sentence):
    return instruction + "\n\nSentence:\n" + sentence


def build_chat_prompt(tokenizer, user_text):
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def normalize_closed_label(text):
    return text.strip().lower()


def sync():
    torch.cuda.synchronize()


def move_to_device(inputs, device):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def generate_one(model, processor, instruction, sentence, device, max_new_tokens):
    """
    返回：
        response
        input_tokens
        output_tokens
        generate latency ms
        e2e latency ms
    """
    sync()
    e2e_start = time.perf_counter()
    user_text = build_user_text(instruction, sentence)
    prompt = build_chat_prompt(processor.tokenizer, user_text)
    inputs = processor(text=prompt, return_tensors="pt")
    inputs = move_to_device(inputs, device)
    sync()
    input_tokens = int(inputs["input_ids"].shape[-1])
    model_start = time.perf_counter()
    generated_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
    )
    sync()
    model_end = time.perf_counter()
    new_ids = generated_ids[:, input_tokens:]
    output_tokens = int(new_ids.shape[-1])
    response = processor.tokenizer.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    normalized = normalize_closed_label(response)
    e2e_end = time.perf_counter()
    model_latency_ms = (model_end - model_start) * 1000
    e2e_latency_ms = (e2e_end - e2e_start) * 1000
    return {
        "response": response,
        "normalized_response": normalized,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_latency_ms": model_latency_ms,
        "e2e_latency_ms": e2e_latency_ms,
    }
