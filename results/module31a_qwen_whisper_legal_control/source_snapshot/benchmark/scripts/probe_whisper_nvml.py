import json
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

MODEL_DIR = (
    ROOT / "models/whisper-small"
)

MANIFEST = (
    ROOT
    / "benchmark/manifests/asr.jsonl"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/asr_full.jsonl"
)


def read_manifest(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def sync():
    torch.cuda.synchronize()


def marker(name):
    sync()

    print(
        f"NVML_MARKER|{name}",
        flush=True
    )

    # Stable observation window for
    # the external 50 ms NVML sampler.
    time.sleep(1.0)


def load_audio(path):
    audio, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False
    )

    if audio.ndim == 2:
        audio = audio.mean(
            axis=1
        )

    if sr != 16000:
        raise ValueError(
            f"Unexpected sample rate "
            f"{sr} for {path}; "
            f"expected 16000 Hz."
        )

    return audio, sr


def prepare_inputs(
    processor,
    audio,
    sample_rate,
    duration_s,
    device
):
    # Same short-form / long-form split
    # as the formal Whisper runner.

    if duration_s <= 30.0:

        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True
        )

        long_form = False

    else:

        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=False,
            padding="longest"
        )

        long_form = True

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    return inputs, long_form


def generate_transcription(
    model,
    inputs,
    long_form
):
    kwargs = {
        "language": "en",
        "task": "transcribe",
        "do_sample": False,
    }

    if long_form:
        kwargs[
            "return_timestamps"
        ] = True

    return model.generate(
        **inputs,
        **kwargs
    )


# --------------------------------------------------
# CUDA setup
# --------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available."
    )


device = torch.device(
    "cuda:0"
)

torch.cuda.set_device(
    device
)


print(
    "Device:",
    device,
    flush=True
)

print(
    "GPU:",
    torch.cuda.get_device_name(0),
    flush=True
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

samples = read_manifest(
    MANIFEST
)

full_samples = read_manifest(
    FULL_MANIFEST
)


if len(samples) != 200:
    raise RuntimeError(
        f"Expected 200 benchmark samples, "
        f"got {len(samples)}."
    )


long_samples = [
    x
    for x in samples
    if float(
        x["duration_s"]
    ) > 30.0
]


print(
    "Benchmark samples:",
    len(samples),
    flush=True
)

print(
    "Long-form samples (>30s):",
    len(long_samples),
    flush=True
)

if long_samples:
    for x in long_samples:
        print(
            "  long:",
            x["id"],
            "duration=",
            x["duration_s"],
            flush=True
        )


# --------------------------------------------------
# Processor
# --------------------------------------------------

print(
    "Loading processor...",
    flush=True
)


processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )
)


print(
    "Sampling rate:",
    processor
    .feature_extractor
    .sampling_rate,
    flush=True
)


# --------------------------------------------------
# Model
# --------------------------------------------------

print(
    "Loading Whisper-small...",
    flush=True
)


model = (
    AutoModelForSpeechSeq2Seq
    .from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )
)

model.eval()
model.to(
    device
)


# Same forced-decoder cleanup
# as formal benchmark.

if hasattr(
    model.config,
    "forced_decoder_ids"
):
    model.config.forced_decoder_ids = None


if hasattr(
    model.generation_config,
    "forced_decoder_ids"
):
    model.generation_config.forced_decoder_ids = None


sync()


dtype = str(
    next(
        model.parameters()
    ).dtype
).replace(
    "torch.",
    ""
)


print(
    "Model dtype:",
    dtype,
    flush=True
)


marker(
    "post_load"
)


# --------------------------------------------------
# Warmup
#
# Formal benchmark:
#   3 runs
#   separate sample
#   duration in [4, 6)
# --------------------------------------------------

benchmark_ids = {
    x["id"]
    for x in samples
}


warmup_sample = None


for candidate in full_samples:

    duration = float(
        candidate["duration_s"]
    )

    if (
        candidate["id"]
        not in benchmark_ids
        and 4.0 <= duration < 6.0
    ):
        warmup_sample = candidate
        break


if warmup_sample is None:
    raise RuntimeError(
        "Could not find separate "
        "4-6 second warmup sample."
    )


print(
    "Warmup audio:",
    warmup_sample["input"],
    flush=True
)

print(
    "Warmup duration:",
    warmup_sample["duration_s"],
    flush=True
)

print(
    "Warmup runs: 3",
    flush=True
)


with torch.inference_mode():

    for _ in range(3):

        audio, sr = load_audio(
            warmup_sample[
                "input"
            ]
        )

        inputs, long_form = (
            prepare_inputs(
                processor,
                audio,
                sr,
                float(
                    warmup_sample[
                        "duration_s"
                    ]
                ),
                device
            )
        )

        generated_ids = (
            generate_transcription(
                model,
                inputs,
                long_form
            )
        )

        sync()

        _ = (
            processor
            .batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        )

        del inputs
        del generated_ids


# Same as formal runner.
# This resets only PyTorch's peak-stat
# counter; it does not free NVML-visible
# process memory.

torch.cuda.reset_peak_memory_stats(
    device
)


marker(
    "post_warmup"
)


# --------------------------------------------------
# Actual 200-sample formal workload
# --------------------------------------------------

print(
    "Running 200 LibriSpeech "
    "benchmark samples...",
    flush=True
)


observed_long_form = 0


with torch.inference_mode():

    for i, sample in enumerate(
        samples,
        start=1
    ):

        duration_s = float(
            sample[
                "duration_s"
            ]
        )

        audio, sr = load_audio(
            sample[
                "input"
            ]
        )

        inputs, long_form = (
            prepare_inputs(
                processor,
                audio,
                sr,
                duration_s,
                device
            )
        )

        if long_form:
            observed_long_form += 1

        generated_ids = (
            generate_transcription(
                model,
                inputs,
                long_form
            )
        )

        sync()

        _ = (
            processor
            .batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
        )

        del inputs
        del generated_ids

        if (
            i == 1
            or i % 25 == 0
            or long_form
            or i == len(samples)
        ):
            print(
                f"Processed "
                f"{i}/{len(samples)} "
                f"duration={duration_s:.3f}s "
                f"long_form={long_form}",
                flush=True
            )


print(
    "Observed long-form count:",
    observed_long_form,
    flush=True
)


marker(
    "post_inference"
)


print(
    "Probe complete.",
    flush=True
)
