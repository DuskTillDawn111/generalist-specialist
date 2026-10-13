import re
import soundfile as sf
import torch


def normalize_asr(text):
    """
    与当前 benchmark/evaluation/metrics.py
    的 ASR normalization 保持同一原则：

    - lowercase
    - 保留 apostrophe
    - 去除其他 punctuation
    - normalize whitespace
    """
    text = text.lower()
    text = re.sub("[^a-z0-9'\\s]", " ", text)
    text = re.sub("\\s+", " ", text)
    return text.strip()


def levenshtein_words(ref_words, hyp_words):
    n = len(ref_words)
    m = len(hyp_words)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def sync():
    torch.cuda.synchronize()


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"Unexpected sample rate {sr} for {path}; expected 16000 Hz.")
    return (audio, sr)


def prepare_inputs(processor, audio, sample_rate, duration_s, device):
    """
    <= 30 s:
        Whisper short-form

    > 30 s:
        保留完整音频，
        交给 HF Whisper long-form generation。
    """
    if duration_s <= 30.0:
        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        long_form = False
    else:
        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=False,
            padding="longest",
        )
        long_form = True
    inputs = {key: value.to(device) for (key, value) in inputs.items()}
    return (inputs, long_form)


def generate_transcription(model, inputs, long_form):
    kwargs = {"language": "en", "task": "transcribe", "do_sample": False}
    if long_form:
        kwargs["return_timestamps"] = True
    generated_ids = model.generate(**inputs, **kwargs)
    return generated_ids
