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


def load_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts["text_classification"]


def normalize_label(text):
    return text.strip().lower()


def sync():
    torch.cuda.synchronize()


def make_instruction(task_prompt, sentence):
    return task_prompt + "\n\nSentence:\n" + sentence


def run_one(model, processor, task_prompt, sample, max_new_tokens):
    sync()
    e2e_start = time.perf_counter()
    instruction = make_instruction(task_prompt, sample["input"])
    messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    formatted_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=formatted_text, return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")
    input_length = inputs["input_ids"].shape[1]
    sync()
    model_start = time.perf_counter()
    output = model.generate(
        **inputs,
        thinker_max_new_tokens=max_new_tokens,
        do_sample=False,
        return_audio=False,
    )
    sync()
    model_end = time.perf_counter()
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected Qwen generate() Tensor, got {type(output)}")
    if output.shape[1] < input_length:
        raise RuntimeError("Generated sequence is shorter than input sequence.")
    generated_ids = output[:, input_length:]
    decoded = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    normalized = normalize_label(decoded)
    e2e_end = time.perf_counter()
    return {
        "raw_response": decoded,
        "normalized_response": normalized,
        "input_tokens": int(input_length),
        "output_tokens": int(generated_ids.shape[1]),
        "model_latency_ms": (model_end - model_start) * 1000,
        "e2e_latency_ms": (e2e_end - e2e_start) * 1000,
    }
