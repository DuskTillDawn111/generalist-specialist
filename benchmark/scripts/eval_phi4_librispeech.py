import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)


ROOT = Path("/root/autodl-tmp/generalist-specialist")

MODEL_DIR = ROOT / "models/phi4-multimodal"

MANIFEST = (
    ROOT
    / "benchmark/manifests/asr.jsonl"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/asr_full.jsonl"
)

PROMPT_FILE = (
    ROOT
    / "benchmark/prompts/task_prompts.json"
)

RESULT_DIR = (
    ROOT
    / "benchmark/results/formal"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "phi4_librispeech_summary.json"
)

PREDICTION_PATH = (
    RESULT_DIR
    / "phi4_librispeech_predictions.jsonl"
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


def load_task_prompt():
    with open(
        PROMPT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        prompts = json.load(f)

    return prompts["asr"]


def normalize_asr(text):
    text = text.lower()

    text = re.sub(
        r"[^a-z0-9'\s]",
        " ",
        text
    )

    return " ".join(
        text.split()
    )


def levenshtein_words(
    reference_words,
    prediction_words
):
    n = len(
        reference_words
    )

    m = len(
        prediction_words
    )

    prev = list(
        range(m + 1)
    )

    for i in range(
        1,
        n + 1
    ):
        curr = [i] + [0] * m

        for j in range(
            1,
            m + 1
        ):
            cost = (
                0
                if reference_words[i - 1]
                == prediction_words[j - 1]
                else 1
            )

            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost
            )

        prev = curr

    return prev[m]


def build_chat_prompt(
    tokenizer,
    instruction
):
    messages = [
        {
            "role": "user",
            "content": (
                "<|audio_1|>\n"
                + instruction
            ),
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def stats(values):
    if not values:
        return None

    arr = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "mean_ms":
            round(
                float(
                    np.mean(arr)
                ),
                3
            ),

        "p50_ms":
            round(
                float(
                    np.percentile(
                        arr,
                        50
                    )
                ),
                3
            ),

        "p95_ms":
            round(
                float(
                    np.percentile(
                        arr,
                        95
                    )
                ),
                3
            ),

        "p99_ms":
            round(
                float(
                    np.percentile(
                        arr,
                        99
                    )
                ),
                3
            ),

        "min_ms":
            round(
                float(
                    np.min(arr)
                ),
                3
            ),

        "max_ms":
            round(
                float(
                    np.max(arr)
                ),
                3
            ),
    }


def distribution_stats(values):
    if not values:
        return None

    arr = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "mean":
            round(
                float(
                    np.mean(arr)
                ),
                6
            ),

        "p50":
            round(
                float(
                    np.percentile(
                        arr,
                        50
                    )
                ),
                6
            ),

        "p95":
            round(
                float(
                    np.percentile(
                        arr,
                        95
                    )
                ),
                6
            ),

        "p99":
            round(
                float(
                    np.percentile(
                        arr,
                        99
                    )
                ),
                6
            ),
    }


def sync():
    torch.cuda.synchronize()


def checkpoint_size_mb(path):
    total = 0

    for p in path.glob(
        "*.safetensors"
    ):
        total += p.stat().st_size

    return total / (1024 ** 2)


def move_to_device(
    inputs,
    device
):
    moved = {}

    for key, value in inputs.items():

        if torch.is_tensor(
            value
        ):
            moved[key] = (
                value.to(
                    device
                )
            )
        else:
            moved[key] = value

    return moved


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
        raise RuntimeError(
            f"Expected 16000 Hz, "
            f"got {sr}: {path}"
        )

    return audio, sr


def generate_one(
    model,
    processor,
    prompt,
    audio_path,
    device,
    max_new_tokens
):
    sync()

    e2e_start = (
        time.perf_counter()
    )

    # --------------------------------
    # Local FLAC read / decode
    # --------------------------------

    audio, sr = load_audio(
        audio_path
    )

    # --------------------------------
    # Phi multimodal processor
    # --------------------------------

    inputs = processor(
        text=prompt,
        audios=[
            (
                audio,
                sr
            )
        ],
        return_tensors="pt"
    )

    audio_frames = None

    if (
        "input_audio_embeds"
        in inputs
        and torch.is_tensor(
            inputs[
                "input_audio_embeds"
            ]
        )
        and inputs[
            "input_audio_embeds"
        ].ndim >= 2
    ):
        audio_frames = int(
            inputs[
                "input_audio_embeds"
            ].shape[1]
        )

    inputs = move_to_device(
        inputs,
        device
    )

    sync()

    input_tokens = int(
        inputs[
            "input_ids"
        ].shape[-1]
    )

    # --------------------------------
    # Audio encoder + LM generation
    # --------------------------------

    model_start = (
        time.perf_counter()
    )

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True
    )

    sync()

    model_end = (
        time.perf_counter()
    )

    # --------------------------------
    # Decode generated continuation
    # --------------------------------

    new_ids = generated_ids[
        :,
        input_tokens:
    ]

    output_tokens = int(
        new_ids.shape[-1]
    )

    response = (
        processor.tokenizer
        .batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    )

    normalized = (
        normalize_asr(
            response
        )
    )

    e2e_end = (
        time.perf_counter()
    )

    return {
        "raw_response":
            response,

        "normalized_response":
            normalized,

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "audio_frames":
            audio_frames,

        "model_latency_ms":
            (
                model_end
                - model_start
            ) * 1000,

        "e2e_latency_ms":
            (
                e2e_end
                - e2e_start
            ) * 1000,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--warmup",
        type=int,
        default=5
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0
    )

    parser.add_argument(
        "--longest-only",
        action="store_true"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256
    )

    parser.add_argument(
        "--latency-min",
        type=float,
        default=4.0
    )

    parser.add_argument(
        "--latency-max",
        type=float,
        default=6.0
    )

    args = parser.parse_args()

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
        device
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        )
    )

    print(
        "Torch:",
        torch.__version__
    )

    print(
        "Transformers:",
        transformers.__version__
    )

    print(
        "Precision: bfloat16"
    )

    print(
        "Attention backend: SDPA"
    )

    print(
        "Decoding: do_sample=False"
    )

    print(
        "Max new tokens:",
        args.max_new_tokens
    )

    print(
        "Latency bucket:",
        f"[{args.latency_min}, "
        f"{args.latency_max}) s"
    )

    # --------------------------------
    # Prompt
    # --------------------------------

    prompt_info = (
        load_task_prompt()
    )

    instruction = (
        prompt_info[
            "prompt"
        ]
    )

    print(
        "Prompt version:",
        prompt_info[
            "version"
        ]
    )

    # --------------------------------
    # Dataset
    # --------------------------------

    all_benchmark_samples = (
        read_manifest(
            MANIFEST
        )
    )

    if args.longest_only:

        longest = max(
            all_benchmark_samples,
            key=lambda x:
                x[
                    "duration_s"
                ]
        )

        samples = [
            longest
        ]

        print(
            "Mode: longest sample only"
        )

    else:
        samples = list(
            all_benchmark_samples
        )

        if args.limit > 0:
            samples = samples[
                :args.limit
            ]

    print(
        "Evaluation samples:",
        len(samples)
    )

    if args.longest_only:
        print(
            "Longest duration:",
            samples[0][
                "duration_s"
            ]
        )

        print(
            "Longest sample:",
            samples[0][
                "id"
            ]
        )

    # --------------------------------
    # Processor
    # --------------------------------

    processor_start = (
        time.perf_counter()
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=False
        )
    )

    processor_load_time_s = (
        time.perf_counter()
        - processor_start
    )

    chat_prompt = (
        build_chat_prompt(
            processor.tokenizer,
            instruction
        )
    )

    # --------------------------------
    # Model
    # --------------------------------

    print(
        "Loading Phi-4-multimodal..."
    )

    load_start = (
        time.perf_counter()
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation="sdpa"
        )
    )

    model.eval()

    model.to(
        device
    )

    model.config.use_cache = True

    sync()

    model_load_time_s = (
        time.perf_counter()
        - load_start
    )

    parameters_m = (
        sum(
            p.numel()
            for p in model.parameters()
        )
        / 1e6
    )

    actual_dtype = str(
        next(
            model.parameters()
        ).dtype
    ).replace(
        "torch.",
        ""
    )

    actual_attention_backend = (
        model.config
        ._attn_implementation
    )

    idle_allocated_mb = (
        torch.cuda.memory_allocated(
            device
        )
        / 1024**2
    )

    print(
        f"Parameters: "
        f"{parameters_m:.3f} M"
    )

    print(
        "Actual dtype:",
        actual_dtype
    )

    print(
        "Actual attention backend:",
        actual_attention_backend
    )

    print(
        f"Model load time: "
        f"{model_load_time_s:.3f} s"
    )

    print(
        f"CUDA allocated after load: "
        f"{idle_allocated_mb:.3f} MB"
    )

    # --------------------------------
    # Separate 4-6 second warmup
    # --------------------------------

    benchmark_ids = {
        x["id"]
        for x in all_benchmark_samples
    }

    warmup_sample = None

    for candidate in read_manifest(
        FULL_MANIFEST
    ):

        duration = float(
            candidate[
                "duration_s"
            ]
        )

        if (
            candidate["id"]
            not in benchmark_ids
            and args.latency_min
            <= duration
            < args.latency_max
        ):
            warmup_sample = (
                candidate
            )
            break

    if warmup_sample is None:
        raise RuntimeError(
            "Could not find separate "
            "4-6 s warmup audio."
        )

    print(
        "Full-pipeline warmup runs:",
        args.warmup
    )

    print(
        "Warmup audio:",
        warmup_sample[
            "input"
        ]
    )

    print(
        "Warmup duration:",
        warmup_sample[
            "duration_s"
        ]
    )

    with torch.inference_mode():

        for _ in range(
            args.warmup
        ):

            _ = generate_one(
                model=model,
                processor=processor,
                prompt=chat_prompt,
                audio_path=(
                    warmup_sample[
                        "input"
                    ]
                ),
                device=device,
                max_new_tokens=(
                    args.max_new_tokens
                )
            )

    torch.cuda.reset_peak_memory_stats(
        device
    )

    # --------------------------------
    # Evaluation
    # --------------------------------

    predictions = []

    model_latencies = []
    e2e_latencies = []

    model_rtfs = []
    e2e_rtfs = []

    bucket_model_latencies = []
    bucket_e2e_latencies = []

    bucket_model_rtfs = []
    bucket_e2e_rtfs = []

    input_token_counts = []
    output_token_counts = []
    audio_frame_counts = []

    total_word_errors = 0
    total_reference_words = 0

    utterance_wers = []

    total_audio_duration_s = 0.0
    total_model_time_s = 0.0
    total_e2e_time_s = 0.0

    over_30s_samples = 0
    hit_max_new_tokens = 0

    print()
    print(
        "Running evaluation..."
    )

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

            if duration_s > 30.0:
                over_30s_samples += 1

            result = generate_one(
                model=model,
                processor=processor,
                prompt=chat_prompt,
                audio_path=(
                    sample[
                        "input"
                    ]
                ),
                device=device,
                max_new_tokens=(
                    args.max_new_tokens
                )
            )

            reference = (
                normalize_asr(
                    sample[
                        "label"
                    ]
                )
            )

            prediction = (
                result[
                    "normalized_response"
                ]
            )

            reference_words = (
                reference.split()
            )

            prediction_words = (
                prediction.split()
            )

            word_errors = (
                levenshtein_words(
                    reference_words,
                    prediction_words
                )
            )

            num_reference_words = len(
                reference_words
            )

            utterance_wer = (
                word_errors
                / num_reference_words
                if num_reference_words
                > 0
                else 0.0
            )

            total_word_errors += (
                word_errors
            )

            total_reference_words += (
                num_reference_words
            )

            utterance_wers.append(
                utterance_wer
            )

            model_latency_ms = (
                result[
                    "model_latency_ms"
                ]
            )

            e2e_latency_ms = (
                result[
                    "e2e_latency_ms"
                ]
            )

            model_rtf = (
                model_latency_ms
                / 1000.0
                / duration_s
            )

            e2e_rtf = (
                e2e_latency_ms
                / 1000.0
                / duration_s
            )

            model_latencies.append(
                model_latency_ms
            )

            e2e_latencies.append(
                e2e_latency_ms
            )

            model_rtfs.append(
                model_rtf
            )

            e2e_rtfs.append(
                e2e_rtf
            )

            input_token_counts.append(
                result[
                    "input_tokens"
                ]
            )

            output_token_counts.append(
                result[
                    "output_tokens"
                ]
            )

            if (
                result[
                    "audio_frames"
                ]
                is not None
            ):
                audio_frame_counts.append(
                    result[
                        "audio_frames"
                    ]
                )

            if (
                result[
                    "output_tokens"
                ]
                >= args.max_new_tokens
            ):
                hit_max_new_tokens += 1

            total_audio_duration_s += (
                duration_s
            )

            total_model_time_s += (
                model_latency_ms
                / 1000.0
            )

            total_e2e_time_s += (
                e2e_latency_ms
                / 1000.0
            )

            in_bucket = (
                args.latency_min
                <= duration_s
                < args.latency_max
            )

            if in_bucket:

                bucket_model_latencies.append(
                    model_latency_ms
                )

                bucket_e2e_latencies.append(
                    e2e_latency_ms
                )

                bucket_model_rtfs.append(
                    model_rtf
                )

                bucket_e2e_rtfs.append(
                    e2e_rtf
                )

            record = {
                "id":
                    sample["id"],

                "task":
                    "automatic_speech_recognition",

                "dataset":
                    "librispeech",

                "split":
                    "test-clean",

                "input":
                    sample["input"],

                "duration_s":
                    duration_s,

                "reference":
                    sample[
                        "label"
                    ],

                "normalized_reference":
                    reference,

                "raw_response":
                    result[
                        "raw_response"
                    ],

                "normalized_prediction":
                    prediction,

                "word_errors":
                    word_errors,

                "reference_words":
                    num_reference_words,

                "wer":
                    round(
                        utterance_wer,
                        6
                    ),

                "input_tokens":
                    result[
                        "input_tokens"
                    ],

                "output_tokens":
                    result[
                        "output_tokens"
                    ],

                "audio_frames":
                    result[
                        "audio_frames"
                    ],

                "hit_max_new_tokens":
                    (
                        result[
                            "output_tokens"
                        ]
                        >=
                        args.max_new_tokens
                    ),

                "model_latency_ms":
                    round(
                        model_latency_ms,
                        3
                    ),

                "e2e_latency_ms":
                    round(
                        e2e_latency_ms,
                        3
                    ),

                "model_rtf":
                    round(
                        model_rtf,
                        6
                    ),

                "e2e_rtf":
                    round(
                        e2e_rtf,
                        6
                    ),

                "in_fixed_duration_bucket":
                    in_bucket,
            }

            predictions.append(
                record
            )

            long_mark = (
                " LONG"
                if duration_s > 30
                else ""
            )

            print(
                f"[{i:03d}/"
                f"{len(samples):03d}] "
                f"dur={duration_s:5.2f}s "
                f"WER={utterance_wer:.3f} "
                f"out="
                f"{result['output_tokens']:>3} "
                f"audio_frames="
                f"{str(result['audio_frames']):>5} "
                f"model="
                f"{model_latency_ms:8.2f}ms "
                f"e2e="
                f"{e2e_latency_ms:8.2f}ms "
                f"RTF="
                f"{e2e_rtf:.4f}"
                f"{long_mark}"
            )

    # --------------------------------
    # Summary
    # --------------------------------

    corpus_wer = (
        total_word_errors
        / total_reference_words
        if total_reference_words > 0
        else 0.0
    )

    mean_utterance_wer = (
        float(
            np.mean(
                utterance_wers
            )
        )
    )

    aggregate_model_rtf = (
        total_model_time_s
        / total_audio_duration_s
    )

    aggregate_e2e_rtf = (
        total_e2e_time_s
        / total_audio_duration_s
    )

    peak_allocated_mb = (
        torch.cuda.max_memory_allocated(
            device
        )
        / 1024**2
    )

    summary = {
        "model":
            "Phi-4-multimodal-instruct",

        "task":
            "automatic_speech_recognition",

        "dataset":
            "LibriSpeech",

        "split":
            "test-clean",

        "language":
            "en",

        "model_role":
            "generalist",

        "precision":
            actual_dtype,

        "attention_backend":
            actual_attention_backend,

        "batch_size":
            1,

        "decoding": {
            "do_sample":
                False,

            "max_new_tokens":
                args.max_new_tokens,
        },

        "prompt_version":
            prompt_info[
                "version"
            ],

        "prompt":
            instruction,

        "prompt_composition":
            (
                "<|audio_1|>\\n"
                "<ASR prompt>"
            ),

        "normalization":
            (
                "lowercase; remove punctuation "
                "except apostrophes; "
                "normalize whitespace"
            ),

        "num_samples":
            len(samples),

        "total_audio_duration_s":
            round(
                total_audio_duration_s,
                3
            ),

        "samples_over_30s":
            over_30s_samples,

        "hit_max_new_tokens":
            hit_max_new_tokens,

        "corpus_word_errors":
            total_word_errors,

        "corpus_reference_words":
            total_reference_words,

        "wer":
            round(
                corpus_wer,
                6
            ),

        "mean_utterance_wer":
            round(
                mean_utterance_wer,
                6
            ),

        "parameters_m":
            round(
                parameters_m,
                3
            ),

        "checkpoint_size_mb":
            round(
                checkpoint_size_mb(
                    MODEL_DIR
                ),
                3
            ),

        "processor_load_time_s":
            round(
                processor_load_time_s,
                3
            ),

        "model_load_time_s":
            round(
                model_load_time_s,
                3
            ),

        "warmup_runs":
            args.warmup,

        "input_tokens": {
            "mean":
                round(
                    float(
                        np.mean(
                            input_token_counts
                        )
                    ),
                    3
                ),

            "min":
                int(
                    min(
                        input_token_counts
                    )
                ),

            "max":
                int(
                    max(
                        input_token_counts
                    )
                ),
        },

        "output_tokens": {
            "total":
                int(
                    sum(
                        output_token_counts
                    )
                ),

            "mean":
                round(
                    float(
                        np.mean(
                            output_token_counts
                        )
                    ),
                    3
                ),

            "min":
                int(
                    min(
                        output_token_counts
                    )
                ),

            "max":
                int(
                    max(
                        output_token_counts
                    )
                ),
        },

        "audio_frames": {
            "mean":
                (
                    round(
                        float(
                            np.mean(
                                audio_frame_counts
                            )
                        ),
                        3
                    )
                    if audio_frame_counts
                    else None
                ),

            "min":
                (
                    int(
                        min(
                            audio_frame_counts
                        )
                    )
                    if audio_frame_counts
                    else None
                ),

            "max":
                (
                    int(
                        max(
                            audio_frame_counts
                        )
                    )
                    if audio_frame_counts
                    else None
                ),
        },

        "all_samples": {
            "model_latency":
                stats(
                    model_latencies
                ),

            "e2e_latency":
                stats(
                    e2e_latencies
                ),

            "model_rtf_distribution":
                distribution_stats(
                    model_rtfs
                ),

            "e2e_rtf_distribution":
                distribution_stats(
                    e2e_rtfs
                ),

            "aggregate_model_rtf":
                round(
                    aggregate_model_rtf,
                    6
                ),

            "aggregate_e2e_rtf":
                round(
                    aggregate_e2e_rtf,
                    6
                ),
        },

        "fixed_duration_latency": {
            "range":
                (
                    f"[{args.latency_min}, "
                    f"{args.latency_max}) "
                    f"seconds"
                ),

            "num_samples":
                len(
                    bucket_e2e_latencies
                ),

            "model_latency":
                stats(
                    bucket_model_latencies
                ),

            "e2e_latency":
                stats(
                    bucket_e2e_latencies
                ),

            "model_rtf_distribution":
                distribution_stats(
                    bucket_model_rtfs
                ),

            "e2e_rtf_distribution":
                distribution_stats(
                    bucket_e2e_rtfs
                ),
        },

        "torch_idle_allocated_mb":
            round(
                idle_allocated_mb,
                3
            ),

        "torch_peak_allocated_mb":
            round(
                peak_allocated_mb,
                3
            ),

        "memory_note":
            (
                "Torch allocator memory is "
                "auxiliary only; final deployment "
                "VRAM comparison will use "
                "process-level NVML."
            ),

        "latency_definition": {
            "model_latency":
                (
                    "multimodal model.generate only "
                    "after FLAC decoding, "
                    "audio preprocessing and H2D; "
                    "includes audio encoding and "
                    "autoregressive generation"
                ),

            "e2e_latency":
                (
                    "local FLAC read/decode + "
                    "audio preprocessing + H2D + "
                    "audio encoding + "
                    "autoregressive generation + "
                    "text decoding + "
                    "ASR normalization"
                ),
        },

        "device":
            "cuda:0",

        "gpu":
            torch.cuda.get_device_name(
                0
            ),

        "torch_version":
            torch.__version__,

        "transformers_version":
            transformers.__version__,

        "timestamp":
            datetime.now()
            .astimezone()
            .isoformat(),
    }

    # --------------------------------
    # Save
    #
    # Longest-only testing uses
    # separate filenames so it cannot
    # overwrite formal results.
    # --------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.longest_only:

        prediction_path = (
            RESULT_DIR
            / "phi4_librispeech_longest_probe.jsonl"
        )

        summary_path = (
            RESULT_DIR
            / "phi4_librispeech_longest_probe_summary.json"
        )

    elif args.limit > 0:

        prediction_path = (
            RESULT_DIR
            / "phi4_librispeech_smoke_predictions.jsonl"
        )

        summary_path = (
            RESULT_DIR
            / "phi4_librispeech_smoke_summary.json"
        )

    else:

        prediction_path = (
            PREDICTION_PATH
        )

        summary_path = (
            SUMMARY_PATH
        )

    with open(
        prediction_path,
        "w",
        encoding="utf-8"
    ) as f:

        for record in predictions:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(
        "========== SUMMARY =========="
    )

    print(
        "Samples:",
        len(samples)
    )

    print(
        "Total audio:",
        round(
            total_audio_duration_s,
            3
        ),
        "s"
    )

    print(
        "Samples >30 s:",
        over_30s_samples
    )

    print(
        "Hit max_new_tokens:",
        hit_max_new_tokens
    )

    print(
        "Corpus WER:",
        f"{corpus_wer:.4f}"
    )

    print(
        "Mean utterance WER:",
        f"{mean_utterance_wer:.4f}"
    )

    print()

    print(
        "All model latency:",
        stats(
            model_latencies
        )
    )

    print(
        "All E2E latency:",
        stats(
            e2e_latencies
        )
    )

    print(
        "Aggregate model RTF:",
        round(
            aggregate_model_rtf,
            6
        )
    )

    print(
        "Aggregate E2E RTF:",
        round(
            aggregate_e2e_rtf,
            6
        )
    )

    print()

    print(
        "Fixed-duration samples:",
        len(
            bucket_e2e_latencies
        )
    )

    print(
        "Bucket model latency:",
        stats(
            bucket_model_latencies
        )
    )

    print(
        "Bucket E2E latency:",
        stats(
            bucket_e2e_latencies
        )
    )

    print()

    print(
        "Saved predictions:",
        prediction_path
    )

    print(
        "Saved summary:",
        summary_path
    )


if __name__ == "__main__":
    main()
