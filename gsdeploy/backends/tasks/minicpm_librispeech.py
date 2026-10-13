import os
import json
import re
import time
from pathlib import Path
import numpy as np
import soundfile as sf
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
    return prompts["asr"]


def normalize_asr(text):
    text = text.lower()
    text = re.sub("[^a-z0-9'\\s]", " ", text)
    text = re.sub("\\s+", " ", text)
    return text.strip()


def sync():
    torch.cuda.synchronize()


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sr} Hz for {path}")
    return (audio, sr)


def run_one(model, instruction, sample, max_new_tokens):
    duration_s = float(sample["duration_s"])
    sync()
    e2e_start = time.perf_counter()
    audio, sr = load_audio(sample["input"])
    msgs = [{"role": "user", "content": [instruction, audio]}]
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
    normalized = normalize_asr(response)
    e2e_end = time.perf_counter()
    chat_latency_ms = (chat_end - chat_start) * 1000
    e2e_latency_ms = (e2e_end - e2e_start) * 1000
    return {
        "raw_response": response,
        "normalized_response": normalized,
        "sample_rate": sr,
        "audio_samples": int(audio.shape[0]),
        "response_chars": len(response),
        "prediction_words": len(normalized.split()),
        "chat_latency_ms": chat_latency_ms,
        "e2e_latency_ms": e2e_latency_ms,
        "chat_rtf": chat_latency_ms / 1000.0 / duration_s,
        "e2e_rtf": e2e_latency_ms / 1000.0 / duration_s,
    }
