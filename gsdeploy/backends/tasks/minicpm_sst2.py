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


def make_user_text(instruction, sentence):
    return instruction + "\n\nSentence:\n" + sentence


def make_msgs(instruction, sentence):
    return [{"role": "user", "content": [make_user_text(instruction, sentence)]}]


def sync():
    torch.cuda.synchronize()


def chat_once(model, instruction, sentence, max_new_tokens):
    e2e_start = time.perf_counter()
    msgs = make_msgs(instruction, sentence)
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
    normalized = normalize_label(response)
    e2e_end = time.perf_counter()
    return {
        "raw_response": response,
        "normalized_response": normalized,
        "chat_latency_ms": (chat_end - chat_start) * 1000,
        "e2e_latency_ms": (e2e_end - e2e_start) * 1000,
    }
