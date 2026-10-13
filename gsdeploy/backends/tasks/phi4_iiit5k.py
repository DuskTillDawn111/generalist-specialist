import os
import json
import re
import time
from pathlib import Path
from PIL import Image
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
    return prompts["ocr"]


def normalize_ocr(text):
    text = text.lower()
    text = re.sub("[^a-z0-9]", "", text)
    return text


def build_chat_prompt(tokenizer, instruction):
    user_text = "<|image_1|>\n" + instruction
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


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


def generate_one(model, processor, prompt, image_path, device, max_new_tokens):
    sync()
    e2e_start = time.perf_counter()
    with Image.open(image_path) as img:
        image_width = img.width
        image_height = img.height
        image = img.convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
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
    normalized = normalize_ocr(response)
    e2e_end = time.perf_counter()
    return {
        "response": response,
        "normalized_response": normalized,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "image_width": image_width,
        "image_height": image_height,
        "model_latency_ms": (model_end - model_start) * 1000,
        "e2e_latency_ms": (e2e_end - e2e_start) * 1000,
    }
