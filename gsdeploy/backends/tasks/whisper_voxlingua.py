import time
from gsdeploy.backends.tasks import whisper_librispeech as asr_base


def normalize_lang_code(value):
    value = str(value).strip().lower()
    if value.startswith("<|") and value.endswith("|>"):
        value = value[2:-2]
    return value


def validate_sample(sample):
    required = [
        "id",
        "task",
        "dataset",
        "split",
        "input_type",
        "input",
        "label",
        "duration_s",
        "sample_rate",
    ]
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"{sample.get('id')}: missing keys {missing}")
    if sample["task"] != "lid":
        raise ValueError(f"{sample['id']}: unexpected task={sample['task']}")
    if sample["input_type"] != "audio":
        raise ValueError(f"{sample['id']}: input_type must be audio")
    if int(sample["sample_rate"]) != 16000:
        raise ValueError(
            f"{sample['id']}: expected 16000 Hz, got {sample['sample_rate']}"
        )


def token_id_to_language(token_id, generation_config):
    inverse = {
        int(token_id): normalize_lang_code(token)
        for (token, token_id) in generation_config.lang_to_id.items()
    }
    token_id = int(token_id)
    if token_id not in inverse:
        raise RuntimeError(
            f"Whisper detect_language returned unknown token id {token_id}"
        )
    return inverse[token_id]


def run_one(model, processor, sample, device):
    asr_base.sync()
    e2e_start = time.perf_counter()
    audio, sr = asr_base.load_audio(sample["input"])
    processor_start = time.perf_counter()
    inputs = processor(
        audio, sampling_rate=sr, return_tensors="pt", return_attention_mask=True
    )
    input_features = inputs["input_features"].to(device)
    processor_end = time.perf_counter()
    asr_base.sync()
    model_start = time.perf_counter()
    language_tokens = model.detect_language(
        input_features=input_features, generation_config=model.generation_config
    )
    asr_base.sync()
    model_end = time.perf_counter()
    token_id = int(language_tokens.detach().cpu().reshape(-1)[0].item())
    prediction = token_id_to_language(token_id, model.generation_config)
    reference = normalize_lang_code(sample["label"])
    e2e_end = time.perf_counter()
    return {
        "prediction": prediction,
        "reference": reference,
        "language_token_id": token_id,
        "correct": prediction == reference,
        "audio_samples": int(audio.shape[0]),
        "sample_rate": int(sr),
        "duration_s": float(sample["duration_s"]),
        "input_features_shape": list(input_features.shape),
        "processor_latency_ms": (processor_end - processor_start) * 1000.0,
        "model_latency_ms": (model_end - model_start) * 1000.0,
        "e2e_latency_ms": (e2e_end - e2e_start) * 1000.0,
    }
