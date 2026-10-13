import os
import json
import re
import time
from pathlib import Path
import torch
from qwen_omni_utils import process_mm_info

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
    text = re.sub("[^a-z0-9' ]+", " ", text)
    text = re.sub("\\s+", " ", text)
    return text.strip()


def word_distance(reference, prediction):
    ref = reference.split()
    hyp = prediction.split()
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, hw in enumerate(hyp, start=1):
            cost = 0 if rw == hw else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return (prev[-1], len(ref))


def sync():
    torch.cuda.synchronize()


def run_one(model, processor, instruction, sample, max_new_tokens):
    duration_s = float(sample["duration_s"])
    sync()
    e2e_start = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": sample["input"]},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    if audios is None:
        raise RuntimeError("process_mm_info returned no audio.")
    if len(audios) != 1:
        raise RuntimeError(f"Expected exactly one audio, got {len(audios)}.")
    if images is not None:
        raise RuntimeError("Unexpected image input.")
    if videos is not None:
        raise RuntimeError("Unexpected video input.")
    audio_samples = None
    audio_dtype = None
    if hasattr(audios[0], "shape"):
        audio_samples = int(audios[0].shape[0])
    if hasattr(audios[0], "dtype"):
        audio_dtype = str(audios[0].dtype)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    if "input_features" not in inputs or "feature_attention_mask" not in inputs:
        raise RuntimeError("Audio processor outputs missing.")
    input_features_shape = list(inputs["input_features"].shape)
    feature_attention_mask_shape = list(inputs["feature_attention_mask"].shape)
    inputs = inputs.to("cuda")
    input_length = int(inputs["input_ids"].shape[1])
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
        raise TypeError(f"Expected Tensor output, got {type(output)}")
    if output.shape[1] < input_length:
        raise RuntimeError("Generated sequence shorter than input sequence.")
    generated_ids = output[:, input_length:]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    prediction = normalize_asr(response)
    reference = normalize_asr(sample["label"])
    e2e_end = time.perf_counter()
    model_ms = (model_end - model_start) * 1000.0
    e2e_ms = (e2e_end - e2e_start) * 1000.0
    model_rtf = model_ms / 1000.0 / duration_s
    e2e_rtf = e2e_ms / 1000.0 / duration_s
    word_errors, ref_words = word_distance(reference, prediction)
    utterance_wer = word_errors / ref_words if ref_words > 0 else 0.0
    output_tokens = int(generated_ids.shape[1])
    return {
        "raw_response": response,
        "prediction": prediction,
        "reference": reference,
        "input_tokens": input_length,
        "output_tokens": output_tokens,
        "max_new_tokens_hit": output_tokens >= max_new_tokens,
        "audio_samples": audio_samples,
        "audio_dtype": audio_dtype,
        "input_features_shape": input_features_shape,
        "feature_attention_mask_shape": feature_attention_mask_shape,
        "word_errors": word_errors,
        "reference_words": ref_words,
        "utterance_wer": utterance_wer,
        "model_latency_ms": model_ms,
        "e2e_latency_ms": e2e_ms,
        "model_rtf": model_rtf,
        "e2e_rtf": e2e_rtf,
    }
