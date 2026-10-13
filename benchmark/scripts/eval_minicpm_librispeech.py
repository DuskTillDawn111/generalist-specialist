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
from transformers import AutoModel


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

MODEL_DIR = (
    ROOT / "models/minicpm-o-4_5"
)

MANIFEST = (
    ROOT / "benchmark/manifests/asr.jsonl"
)

FULL_MANIFEST = (
    ROOT / "benchmark/manifests/asr_full.jsonl"
)

PROMPT_FILE = (
    ROOT / "benchmark/prompts/task_prompts.json"
)

RESULT_DIR = (
    ROOT / "benchmark/results/formal"
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


def load_prompt():
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

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def levenshtein_words(
    reference,
    prediction
):
    ref = reference.split()
    hyp = prediction.split()

    prev = list(
        range(
            len(hyp) + 1
        )
    )

    for i in range(
        1,
        len(ref) + 1
    ):
        curr = [
            i
        ] + [
            0
        ] * len(hyp)

        for j in range(
            1,
            len(hyp) + 1
        ):
            cost = (
                0
                if ref[i - 1]
                == hyp[j - 1]
                else 1
            )

            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )

        prev = curr

    return prev[-1]


def sync():
    torch.cuda.synchronize()


def latency_stats(values):
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


def scalar_stats(values):
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

        "min":
            round(
                float(
                    np.min(arr)
                ),
                6
            ),

        "max":
            round(
                float(
                    np.max(arr)
                ),
                6
            ),
    }


def checkpoint_size_mb(path):
    total = 0

    for p in path.glob(
        "*.safetensors"
    ):
        total += p.stat().st_size

    return total / (1024 ** 2)


def load_audio(path):
    audio, sr = sf.read(
        path,
        dtype="float32"
    )

    if audio.ndim == 2:
        audio = np.mean(
            audio,
            axis=1,
            dtype=np.float32
        )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if sr != 16000:
        raise ValueError(
            f"Expected 16 kHz audio, "
            f"got {sr} Hz for {path}"
        )

    return audio, sr


def run_one(
    model,
    instruction,
    sample,
    max_new_tokens
):
    duration_s = float(
        sample["duration_s"]
    )

    # ---------------------------------
    # E2E starts before local FLAC
    # file read / decode
    # ---------------------------------

    sync()

    e2e_start = (
        time.perf_counter()
    )

    audio, sr = load_audio(
        sample["input"]
    )

    msgs = [
        {
            "role": "user",
            "content": [
                instruction,
                audio,
            ],
        }
    ]

    # ---------------------------------
    # MiniCPM chat latency
    #
    # Audio has already been decoded.
    # model.chat internally performs
    # multimodal audio processing,
    # audio encoding, generation and
    # response decoding.
    # ---------------------------------

    sync()

    chat_start = (
        time.perf_counter()
    )

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

    chat_end = (
        time.perf_counter()
    )

    if not isinstance(
        response,
        str
    ):
        raise TypeError(
            "Expected model.chat() "
            f"to return str, got "
            f"{type(response)}"
        )

    normalized = normalize_asr(
        response
    )

    e2e_end = (
        time.perf_counter()
    )

    chat_latency_ms = (
        chat_end
        - chat_start
    ) * 1000

    e2e_latency_ms = (
        e2e_end
        - e2e_start
    ) * 1000

    return {
        "raw_response":
            response,

        "normalized_response":
            normalized,

        "sample_rate":
            sr,

        "audio_samples":
            int(audio.shape[0]),

        "response_chars":
            len(response),

        "prediction_words":
            len(
                normalized.split()
            ),

        "chat_latency_ms":
            chat_latency_ms,

        "e2e_latency_ms":
            e2e_latency_ms,

        "chat_rtf":
            (
                chat_latency_ms
                / 1000.0
                / duration_s
            ),

        "e2e_rtf":
            (
                e2e_latency_ms
                / 1000.0
                / duration_s
            ),
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
        torch.cuda.get_device_name(0)
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
        "init_vision=True"
    )

    print(
        "init_audio=True"
    )

    print(
        "init_tts=False"
    )

    print(
        "use_tts_template=False"
    )

    print(
        "enable_thinking=False"
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

    # ---------------------------------
    # Prompt
    # ---------------------------------

    prompt_info = load_prompt()

    instruction = (
        prompt_info["prompt"]
    )

    print(
        "Prompt version:",
        prompt_info["version"]
    )

    # ---------------------------------
    # Dataset
    # ---------------------------------

    all_samples = read_manifest(
        MANIFEST
    )

    if args.longest_only:
        samples = [
            max(
                all_samples,
                key=lambda x: float(
                    x["duration_s"]
                )
            )
        ]

        print(
            "Mode: longest sample only"
        )

    else:
        samples = list(
            all_samples
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
            samples[0]["duration_s"]
        )

        print(
            "Longest sample:",
            samples[0]["id"]
        )

    # ---------------------------------
    # Load model
    # ---------------------------------

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(
        "Loading MiniCPM-o-4.5..."
    )

    load_start = (
        time.perf_counter()
    )

    model = AutoModel.from_pretrained(
        MODEL_DIR,

        trust_remote_code=True,
        local_files_only=True,

        torch_dtype=torch.bfloat16,

        attn_implementation="sdpa",

        init_vision=True,
        init_audio=True,
        init_tts=False,

        low_cpu_mem_usage=True,
    )

    model.eval().cuda()

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

    actual_attention = getattr(
        model.config,
        "_attn_implementation",
        getattr(
            model.config,
            "attn_implementation",
            "unknown"
        )
    )

    idle_allocated_mb = (
        torch.cuda.memory_allocated()
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
        actual_attention
    )

    print(
        f"Model load time: "
        f"{model_load_time_s:.3f} s"
    )

    print(
        f"CUDA allocated after load: "
        f"{idle_allocated_mb:.3f} MB"
    )

    # ---------------------------------
    # Separate warmup sample
    # Must not be in benchmark subset.
    # Prefer a 4-6 second sample.
    # ---------------------------------

    benchmark_ids = {
        x["id"]
        for x in all_samples
    }

    warmup_sample = None

    for candidate in read_manifest(
        FULL_MANIFEST
    ):
        duration = float(
            candidate["duration_s"]
        )

        if (
            candidate["id"]
            not in benchmark_ids
            and args.latency_min
            <= duration
            < args.latency_max
        ):
            warmup_sample = candidate
            break

    if warmup_sample is None:
        raise RuntimeError(
            "No separate 4-6 s "
            "warmup sample found."
        )

    print(
        "Full-pipeline warmup runs:",
        args.warmup
    )

    print(
        "Warmup audio:",
        warmup_sample["input"]
    )

    print(
        "Warmup duration:",
        warmup_sample["duration_s"]
    )

    with torch.inference_mode():
        for _ in range(
            args.warmup
        ):
            _ = run_one(
                model=model,
                instruction=instruction,
                sample=warmup_sample,
                max_new_tokens=(
                    args.max_new_tokens
                ),
            )

    torch.cuda.reset_peak_memory_stats()

    # ---------------------------------
    # Evaluation
    # ---------------------------------

    predictions = []

    chat_latencies = []
    e2e_latencies = []

    chat_rtfs = []
    e2e_rtfs = []

    bucket_chat_latencies = []
    bucket_e2e_latencies = []

    bucket_chat_rtfs = []
    bucket_e2e_rtfs = []

    response_chars = []
    prediction_words = []

    total_word_errors = 0
    total_reference_words = 0

    utterance_wers = []

    total_audio_duration_s = 0.0

    total_chat_time_s = 0.0
    total_e2e_time_s = 0.0

    samples_over_30s = 0

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
                sample["duration_s"]
            )

            result = run_one(
                model=model,
                instruction=instruction,
                sample=sample,
                max_new_tokens=(
                    args.max_new_tokens
                ),
            )

            reference = normalize_asr(
                sample["label"]
            )

            prediction = result[
                "normalized_response"
            ]

            word_errors = (
                levenshtein_words(
                    reference,
                    prediction
                )
            )

            reference_words = len(
                reference.split()
            )

            utterance_wer = (
                word_errors
                / reference_words
                if reference_words > 0
                else 0.0
            )

            total_word_errors += (
                word_errors
            )

            total_reference_words += (
                reference_words
            )

            utterance_wers.append(
                utterance_wer
            )

            chat_ms = result[
                "chat_latency_ms"
            ]

            e2e_ms = result[
                "e2e_latency_ms"
            ]

            chat_rtf = result[
                "chat_rtf"
            ]

            e2e_rtf = result[
                "e2e_rtf"
            ]

            chat_latencies.append(
                chat_ms
            )

            e2e_latencies.append(
                e2e_ms
            )

            chat_rtfs.append(
                chat_rtf
            )

            e2e_rtfs.append(
                e2e_rtf
            )

            total_audio_duration_s += (
                duration_s
            )

            total_chat_time_s += (
                chat_ms
                / 1000.0
            )

            total_e2e_time_s += (
                e2e_ms
                / 1000.0
            )

            response_chars.append(
                result[
                    "response_chars"
                ]
            )

            prediction_words.append(
                result[
                    "prediction_words"
                ]
            )

            if duration_s > 30.0:
                samples_over_30s += 1

            in_bucket = (
                args.latency_min
                <= duration_s
                < args.latency_max
            )

            if in_bucket:
                bucket_chat_latencies.append(
                    chat_ms
                )

                bucket_e2e_latencies.append(
                    e2e_ms
                )

                bucket_chat_rtfs.append(
                    chat_rtf
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
                    "LibriSpeech",

                "split":
                    "test-clean",

                "input":
                    sample["input"],

                "duration_s":
                    duration_s,

                "reference":
                    sample["label"],

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
                    reference_words,

                "wer":
                    round(
                        utterance_wer,
                        6
                    ),

                "response_chars":
                    result[
                        "response_chars"
                    ],

                "prediction_words":
                    result[
                        "prediction_words"
                    ],

                "audio_samples":
                    result[
                        "audio_samples"
                    ],

                "sample_rate":
                    result[
                        "sample_rate"
                    ],

                "chat_latency_ms":
                    round(
                        chat_ms,
                        3
                    ),

                "e2e_latency_ms":
                    round(
                        e2e_ms,
                        3
                    ),

                "chat_rtf":
                    round(
                        chat_rtf,
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
                f"dur="
                f"{duration_s:5.2f}s "
                f"WER="
                f"{utterance_wer:.3f} "
                f"words="
                f"{result['prediction_words']:3d} "
                f"chat="
                f"{chat_ms:8.2f}ms "
                f"e2e="
                f"{e2e_ms:8.2f}ms "
                f"RTF="
                f"{e2e_rtf:.4f}"
                f"{long_mark}"
            )

    # ---------------------------------
    # Metrics
    # ---------------------------------

    corpus_wer = (
        total_word_errors
        / total_reference_words
        if total_reference_words > 0
        else 0.0
    )

    mean_utterance_wer = float(
        np.mean(
            utterance_wers
        )
    )

    aggregate_chat_rtf = (
        total_chat_time_s
        / total_audio_duration_s
    )

    aggregate_e2e_rtf = (
        total_e2e_time_s
        / total_audio_duration_s
    )

    all_chat_stats = (
        latency_stats(
            chat_latencies
        )
    )

    all_e2e_stats = (
        latency_stats(
            e2e_latencies
        )
    )

    bucket_chat_stats = (
        latency_stats(
            bucket_chat_latencies
        )
    )

    bucket_e2e_stats = (
        latency_stats(
            bucket_e2e_latencies
        )
    )

    peak_allocated_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    all_chat_qps_proxy = (
        1000.0
        / all_chat_stats["mean_ms"]
    )

    all_e2e_qps_proxy = (
        1000.0
        / all_e2e_stats["mean_ms"]
    )

    if bucket_e2e_stats is not None:
        bucket_chat_qps_proxy = (
            1000.0
            / bucket_chat_stats[
                "mean_ms"
            ]
        )

        bucket_e2e_qps_proxy = (
            1000.0
            / bucket_e2e_stats[
                "mean_ms"
            ]
        )

    else:
        bucket_chat_qps_proxy = None
        bucket_e2e_qps_proxy = None

    # ---------------------------------
    # Output paths
    # ---------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.longest_only:
        prediction_path = (
            RESULT_DIR
            / "minicpm_librispeech_longest_probe.jsonl"
        )

        summary_path = (
            RESULT_DIR
            / "minicpm_librispeech_longest_probe_summary.json"
        )

    elif args.limit > 0:
        prediction_path = (
            RESULT_DIR
            / "minicpm_librispeech_smoke_predictions.jsonl"
        )

        summary_path = (
            RESULT_DIR
            / "minicpm_librispeech_smoke_summary.json"
        )

    else:
        prediction_path = (
            RESULT_DIR
            / "minicpm_librispeech_predictions.jsonl"
        )

        summary_path = (
            RESULT_DIR
            / "minicpm_librispeech_summary.json"
        )

    # ---------------------------------
    # Summary
    # ---------------------------------

    summary = {
        "model":
            "MiniCPM-o-4.5",

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
            actual_attention,

        "batch_size":
            1,

        "modalities_initialized": {
            "vision": True,
            "audio": True,
            "tts": False,
        },

        "decoding": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "enable_thinking": False,
            "use_tts_template": False,
            "max_new_tokens":
                args.max_new_tokens,
        },

        "generation_length_note":
            (
                "MiniCPM model.chat() returns "
                "decoded text rather than generated "
                "token IDs, so exact counting of "
                "max_new_tokens hits is not exposed "
                "by this runner. A separate 33.91 s "
                "longest-sample probe completed a "
                "96-word transcript with "
                "max_new_tokens=256."
            ),

        "prompt_version":
            prompt_info["version"],

        "prompt":
            instruction,

        "message_format":
            "[ASR prompt text, float32 16-kHz mono numpy audio]",

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
            samples_over_30s,

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

        "response_chars": {
            "mean":
                round(
                    float(
                        np.mean(
                            response_chars
                        )
                    ),
                    3
                ),

            "min":
                int(
                    min(
                        response_chars
                    )
                ),

            "max":
                int(
                    max(
                        response_chars
                    )
                ),
        },

        "prediction_words": {
            "mean":
                round(
                    float(
                        np.mean(
                            prediction_words
                        )
                    ),
                    3
                ),

            "min":
                int(
                    min(
                        prediction_words
                    )
                ),

            "max":
                int(
                    max(
                        prediction_words
                    )
                ),
        },

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

        "model_load_time_s":
            round(
                model_load_time_s,
                3
            ),

        "warmup_runs":
            args.warmup,

        "all_samples": {
            "chat_latency":
                all_chat_stats,

            "e2e_latency":
                all_e2e_stats,

            "chat_rtf_distribution":
                scalar_stats(
                    chat_rtfs
                ),

            "e2e_rtf_distribution":
                scalar_stats(
                    e2e_rtfs
                ),

            "aggregate_chat_rtf":
                round(
                    aggregate_chat_rtf,
                    6
                ),

            "aggregate_e2e_rtf":
                round(
                    aggregate_e2e_rtf,
                    6
                ),

            "chat_request_service_rate_proxy_qps":
                round(
                    all_chat_qps_proxy,
                    3
                ),

            "e2e_request_service_rate_proxy_qps":
                round(
                    all_e2e_qps_proxy,
                    3
                ),
        },

        "fixed_duration_latency": {
            "range":
                (
                    f"[{args.latency_min}, "
                    f"{args.latency_max}) seconds"
                ),

            "num_samples":
                len(
                    bucket_e2e_latencies
                ),

            "chat_latency":
                bucket_chat_stats,

            "e2e_latency":
                bucket_e2e_stats,

            "chat_rtf_distribution":
                scalar_stats(
                    bucket_chat_rtfs
                ),

            "e2e_rtf_distribution":
                scalar_stats(
                    bucket_e2e_rtfs
                ),

            "chat_request_service_rate_proxy_qps":
                (
                    round(
                        bucket_chat_qps_proxy,
                        3
                    )
                    if bucket_chat_qps_proxy
                    is not None
                    else None
                ),

            "e2e_request_service_rate_proxy_qps":
                (
                    round(
                        bucket_e2e_qps_proxy,
                        3
                    )
                    if bucket_e2e_qps_proxy
                    is not None
                    else None
                ),
        },

        "throughput_note":
            (
                "Request QPS values are sequential "
                "batch-1 service-rate proxies "
                "(1000 / mean latency), not "
                "load-tested concurrent throughput."
            ),

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
                "framework-specific auxiliary "
                "data; final deployment VRAM "
                "comparison will use "
                "process-level NVML."
            ),

        "latency_definition": {
            "chat_latency":
                (
                    "MiniCPM model.chat() after "
                    "local FLAC decoding; includes "
                    "MiniCPM internal audio "
                    "preprocessing, audio encoding, "
                    "autoregressive generation and "
                    "response decoding"
                ),

            "e2e_latency":
                (
                    "local FLAC read/decode + "
                    "mono float32 preparation + "
                    "message construction + "
                    "model.chat() + "
                    "ASR normalization"
                ),
        },

        "device":
            "cuda:0",

        "gpu":
            torch.cuda.get_device_name(0),

        "torch_version":
            torch.__version__,

        "transformers_version":
            transformers.__version__,

        "timestamp":
            datetime.now()
            .astimezone()
            .isoformat(),
    }

    # ---------------------------------
    # Save
    # ---------------------------------

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

    # ---------------------------------
    # Console summary
    # ---------------------------------

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
        samples_over_30s
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
        "All chat latency:",
        all_chat_stats
    )

    print(
        "All E2E latency:",
        all_e2e_stats
    )

    print(
        "Aggregate chat RTF:",
        round(
            aggregate_chat_rtf,
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
        "Bucket chat latency:",
        bucket_chat_stats
    )

    print(
        "Bucket E2E latency:",
        bucket_e2e_stats
    )

    print()
    print(
        "Max prediction words:",
        max(
            prediction_words
        )
    )

    print(
        "Max response chars:",
        max(
            response_chars
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
