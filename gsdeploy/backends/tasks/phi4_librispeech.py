import os
import json
import re
import time
from pathlib import Path
import soundfile as sf
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
    return prompts["asr"]


def normalize_asr(text):
    text = text.lower()
    text = re.sub("[^a-z0-9'\\s]", " ", text)
    return " ".join(text.split())


def levenshtein_words(reference_words, prediction_words):
    n = len(reference_words)
    m = len(prediction_words)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if reference_words[i - 1] == prediction_words[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def build_chat_prompt(tokenizer, instruction):
    messages = [{"role": "user", "content": "<|audio_1|>\n" + instruction}]
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


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise RuntimeError(f"Expected 16000 Hz, got {sr}: {path}")
    return (audio, sr)


def generate_one(model, processor, prompt, audio_path, device, max_new_tokens):
    sync()
    e2e_start = time.perf_counter()
    audio, sr = load_audio(audio_path)
    inputs = processor(text=prompt, audios=[(audio, sr)], return_tensors="pt")
    audio_frames = None
    if (
        "input_audio_embeds" in inputs
        and torch.is_tensor(inputs["input_audio_embeds"])
        and (inputs["input_audio_embeds"].ndim >= 2)
    ):
        audio_frames = int(inputs["input_audio_embeds"].shape[1])
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
    normalized = normalize_asr(response)
    e2e_end = time.perf_counter()
    return {
        "raw_response": response,
        "normalized_response": normalized,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "audio_frames": audio_frames,
        "model_latency_ms": (model_end - model_start) * 1000,
        "e2e_latency_ms": (e2e_end - e2e_start) * 1000,
    }
