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


def load_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts["ocr"]


def normalize_ocr(text):
    text = text.lower()
    text = re.sub("[^a-z0-9]", "", text)
    return text


def sync():
    torch.cuda.synchronize()


def run_one(model, instruction, image_path, max_new_tokens):
    sync()
    e2e_start = time.perf_counter()
    with Image.open(image_path) as img:
        width = img.width
        height = img.height
        image = img.convert("RGB")
    msgs = [{"role": "user", "content": [image, instruction]}]
    sync()
    chat_start = time.perf_counter()
    response = model.chat(
        msgs=msgs,
        use_tts_template=False,
        enable_thinking=False,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=max_new_tokens,
    )
    sync()
    chat_end = time.perf_counter()
    if not isinstance(response, str):
        raise TypeError(f"Expected model.chat() to return str, got {type(response)}")
    normalized = normalize_ocr(response)
    e2e_end = time.perf_counter()
    return {
        "raw_response": response,
        "normalized_response": normalized,
        "image_width": width,
        "image_height": height,
        "chat_latency_ms": (chat_end - chat_start) * 1000,
        "e2e_latency_ms": (e2e_end - e2e_start) * 1000,
    }
