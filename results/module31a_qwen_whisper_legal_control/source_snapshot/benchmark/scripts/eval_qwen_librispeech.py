import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import transformers

from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

from qwen_omni_utils import process_mm_info


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

MODEL_DIR = (
    ROOT / "models/qwen2.5-omni-7b"
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

    # Keep:
    # letters, digits, apostrophe, spaces
    text = re.sub(
        r"[^a-z0-9' ]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def word_distance(
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

    for i, rw in enumerate(
        ref,
        start=1
    ):
        curr = [i] + (
            [0] * len(hyp)
        )

        for j, hw in enumerate(
            hyp,
            start=1
        ):
            cost = (
                0
                if rw == hw
                else 1
            )

            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )

        prev = curr

    return (
        prev[-1],
        len(ref)
    )


def sync():
    torch.cuda.synchronize()


def stats(values):
    if len(values) == 0:
        return None

    arr = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "mean":
            round(
                float(np.mean(arr)),
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
                float(np.min(arr)),
                6
            ),

        "max":
            round(
                float(np.max(arr)),
                6
            ),
    }


def latency_stats(values):
    x = stats(values)

    if x is None:
        return None

    return {
        "mean_ms":
            round(x["mean"], 3),

        "p50_ms":
            round(x["p50"], 3),

        "p95_ms":
            round(x["p95"], 3),

        "p99_ms":
            round(x["p99"], 3),

        "min_ms":
            round(x["min"], 3),

        "max_ms":
            round(x["max"], 3),
    }


def rtf_stats(values):
    x = stats(values)

    if x is None:
        return None

    return {
        "mean":
            x["mean"],

        "p50":
            x["p50"],

        "p95":
            x["p95"],

        "p99":
            x["p99"],

        "min":
            x["min"],

        "max":
            x["max"],
    }


def checkpoint_size_mb(path):
    total = 0

    for p in path.glob(
        "*.safetensors"
    ):
        total += p.stat().st_size

    return total / (1024 ** 2)


def run_one(
    model,
    processor,
    instruction,
    sample,
    max_new_tokens
):
    duration_s = float(
        sample["duration_s"]
    )

    # ------------------------------------------------
    # E2E begins BEFORE process_mm_info().
    #
    # Therefore it includes local FLAC read/decode,
    # audio loading, processor, H2D, model.generate,
    # text decoding and ASR normalization.
    # ------------------------------------------------

    sync()

    e2e_start = (
        time.perf_counter()
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio": sample["input"],
                },
                {
                    "type": "text",
                    "text": instruction,
                },
            ],
        }
    ]

    text = (
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    audios, images, videos = (
        process_mm_info(
            messages,
            use_audio_in_video=False,
        )
    )

    if audios is None:
        raise RuntimeError(
            "process_mm_info returned "
            "no audio."
        )

    if len(audios) != 1:
        raise RuntimeError(
            "Expected exactly one audio, "
            f"got {len(audios)}."
        )

    if images is not None:
        raise RuntimeError(
            "Unexpected image input."
        )

    if videos is not None:
        raise RuntimeError(
            "Unexpected video input."
        )

    audio_samples = None
    audio_dtype = None

    if hasattr(
        audios[0],
        "shape"
    ):
        audio_samples = int(
            audios[0].shape[0]
        )

    if hasattr(
        audios[0],
        "dtype"
    ):
        audio_dtype = str(
            audios[0].dtype
        )

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )

    if (
        "input_features" not in inputs
        or
        "feature_attention_mask"
        not in inputs
    ):
        raise RuntimeError(
            "Audio processor outputs missing."
        )

    input_features_shape = list(
        inputs[
            "input_features"
        ].shape
    )

    feature_attention_mask_shape = list(
        inputs[
            "feature_attention_mask"
        ].shape
    )

    inputs = inputs.to(
        "cuda"
    )

    input_length = int(
        inputs[
            "input_ids"
        ].shape[1]
    )

    # ------------------------------------------------
    # Model latency starts AFTER:
    # local file IO, process_mm_info,
    # audio preprocessing and H2D.
    #
    # model.generate still includes:
    # audio encoder + LLM autoregressive generation.
    # ------------------------------------------------

    sync()

    model_start = (
        time.perf_counter()
    )

    output = model.generate(
        **inputs,
        thinker_max_new_tokens=max_new_tokens,
        do_sample=False,
        return_audio=False,
    )

    sync()

    model_end = (
        time.perf_counter()
    )

    if not isinstance(
        output,
        torch.Tensor
    ):
        raise TypeError(
            "Expected Tensor output, "
            f"got {type(output)}"
        )

    if output.shape[1] < input_length:
        raise RuntimeError(
            "Generated sequence shorter "
            "than input sequence."
        )

    generated_ids = (
        output[
            :,
            input_length:
        ]
    )

    response = (
        processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    )

    prediction = normalize_asr(
        response
    )

    reference = normalize_asr(
        sample["label"]
    )

    e2e_end = (
        time.perf_counter()
    )

    model_ms = (
        model_end
        - model_start
    ) * 1000.0

    e2e_ms = (
        e2e_end
        - e2e_start
    ) * 1000.0

    model_rtf = (
        model_ms
        / 1000.0
        / duration_s
    )

    e2e_rtf = (
        e2e_ms
        / 1000.0
        / duration_s
    )

    word_errors, ref_words = (
        word_distance(
            reference,
            prediction
        )
    )

    utterance_wer = (
        word_errors / ref_words
        if ref_words > 0
        else 0.0
    )

    output_tokens = int(
        generated_ids.shape[1]
    )

    return {
        "raw_response":
            response,

        "prediction":
            prediction,

        "reference":
            reference,

        "input_tokens":
            input_length,

        "output_tokens":
            output_tokens,

        "max_new_tokens_hit":
            (
                output_tokens
                >= max_new_tokens
            ),

        "audio_samples":
            audio_samples,

        "audio_dtype":
            audio_dtype,

        "input_features_shape":
            input_features_shape,

        "feature_attention_mask_shape":
            feature_attention_mask_shape,

        "word_errors":
            word_errors,

        "reference_words":
            ref_words,

        "utterance_wer":
            utterance_wer,

        "model_latency_ms":
            model_ms,

        "e2e_latency_ms":
            e2e_ms,

        "model_rtf":
            model_rtf,

        "e2e_rtf":
            e2e_rtf,
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

    if (
        args.longest_only
        and
        args.limit > 0
    ):
        raise ValueError(
            "--longest-only and --limit "
            "cannot be used together."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    torch.cuda.set_device(0)

    print(
        "Device: cuda:0"
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
        "enable_audio_output=False"
    )

    print(
        "return_audio=False"
    )

    print(
        "use_audio_in_video=False"
    )

    print(
        "Decoding: do_sample=False"
    )

    print(
        "Max new tokens:",
        args.max_new_tokens
    )

    print(
        "Fixed latency bin:",
        f"[{args.latency_min}, "
        f"{args.latency_max}) s"
    )

    # ------------------------------------------------
    # Prompt
    # ------------------------------------------------

    prompt_info = load_prompt()

    instruction = (
        prompt_info["prompt"]
    )

    print(
        "Prompt version:",
        prompt_info["version"]
    )

    # ------------------------------------------------
    # Manifest
    # ------------------------------------------------

    all_samples = read_manifest(
        MANIFEST
    )

    samples = list(
        all_samples
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

    elif args.limit > 0:
        samples = samples[
            :args.limit
        ]

    print(
        "Evaluation samples:",
        len(samples)
    )

    print(
        "Total audio duration:",
        round(
            sum(
                float(
                    x["duration_s"]
                )
                for x in samples
            ),
            3
        ),
        "s"
    )

    # ------------------------------------------------
    # Processor
    # ------------------------------------------------

    print(
        "Loading processor..."
    )

    processor = (
        Qwen2_5OmniProcessor
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
    )

    # ------------------------------------------------
    # Model
    # ------------------------------------------------

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(
        "Loading Qwen2.5-Omni-7B..."
    )

    load_start = (
        time.perf_counter()
    )

    model = (
        Qwen2_5OmniForConditionalGeneration
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            enable_audio_output=False,
        )
    )

    model.eval()

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
        f"Model load time: "
        f"{model_load_time_s:.3f} s"
    )

    print(
        f"CUDA allocated after load: "
        f"{idle_allocated_mb:.3f} MB"
    )

    # ------------------------------------------------
    # Separate warmup sample:
    # not part of benchmark 200,
    # and duration is also in [4,6).
    # ------------------------------------------------

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
            and
            args.latency_min
            <= duration
            < args.latency_max
        ):
            warmup_sample = candidate
            break

    if warmup_sample is None:
        raise RuntimeError(
            "No separate [4,6) "
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
        warmup_sample["duration_s"],
        "s"
    )

    with torch.inference_mode():

        for _ in range(
            args.warmup
        ):
            _ = run_one(
                model=model,
                processor=processor,
                instruction=instruction,
                sample=warmup_sample,
                max_new_tokens=(
                    args.max_new_tokens
                ),
            )

    torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------
    # Evaluation
    # ------------------------------------------------

    predictions = []

    model_latencies = []
    e2e_latencies = []

    model_rtfs = []
    e2e_rtfs = []

    utterance_wers = []

    input_tokens = []
    output_tokens = []

    total_word_errors = 0
    total_reference_words = 0

    total_audio_duration_s = 0.0

    total_model_time_s = 0.0
    total_e2e_time_s = 0.0

    long_utterances_gt_30s = 0
    max_new_tokens_hits = 0

    print()
    print(
        "Running evaluation..."
    )

    with torch.inference_mode():

        for i, sample in enumerate(
            samples,
            start=1
        ):

            result = run_one(
                model=model,
                processor=processor,
                instruction=instruction,
                sample=sample,
                max_new_tokens=(
                    args.max_new_tokens
                ),
            )

            duration_s = float(
                sample["duration_s"]
            )

            if duration_s > 30.0:
                long_utterances_gt_30s += 1

            if result[
                "max_new_tokens_hit"
            ]:
                max_new_tokens_hits += 1

            total_word_errors += (
                result[
                    "word_errors"
                ]
            )

            total_reference_words += (
                result[
                    "reference_words"
                ]
            )

            utterance_wers.append(
                result[
                    "utterance_wer"
                ]
            )

            model_latencies.append(
                result[
                    "model_latency_ms"
                ]
            )

            e2e_latencies.append(
                result[
                    "e2e_latency_ms"
                ]
            )

            model_rtfs.append(
                result[
                    "model_rtf"
                ]
            )

            e2e_rtfs.append(
                result[
                    "e2e_rtf"
                ]
            )

            input_tokens.append(
                result[
                    "input_tokens"
                ]
            )

            output_tokens.append(
                result[
                    "output_tokens"
                ]
            )

            total_audio_duration_s += (
                duration_s
            )

            total_model_time_s += (
                result[
                    "model_latency_ms"
                ]
                / 1000.0
            )

            total_e2e_time_s += (
                result[
                    "e2e_latency_ms"
                ]
                / 1000.0
            )

            predictions.append(
                {
                    "id":
                        sample["id"],

                    "utterance_id":
                        sample[
                            "utterance_id"
                        ],

                    "task":
                        "asr",

                    "dataset":
                        "LibriSpeech",

                    "split":
                        "test-clean",

                    "input":
                        sample["input"],

                    "duration_s":
                        duration_s,

                    "sample_rate":
                        sample[
                            "sample_rate"
                        ],

                    "raw_reference":
                        sample["label"],

                    "normalized_reference":
                        result[
                            "reference"
                        ],

                    "raw_response":
                        result[
                            "raw_response"
                        ],

                    "normalized_prediction":
                        result[
                            "prediction"
                        ],

                    "word_errors":
                        result[
                            "word_errors"
                        ],

                    "reference_words":
                        result[
                            "reference_words"
                        ],

                    "wer":
                        round(
                            result[
                                "utterance_wer"
                            ],
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

                    "max_new_tokens_hit":
                        result[
                            "max_new_tokens_hit"
                        ],

                    "audio_samples":
                        result[
                            "audio_samples"
                        ],

                    "audio_dtype":
                        result[
                            "audio_dtype"
                        ],

                    "input_features_shape":
                        result[
                            "input_features_shape"
                        ],

                    "feature_attention_mask_shape":
                        result[
                            "feature_attention_mask_shape"
                        ],

                    "model_latency_ms":
                        round(
                            result[
                                "model_latency_ms"
                            ],
                            3
                        ),

                    "e2e_latency_ms":
                        round(
                            result[
                                "e2e_latency_ms"
                            ],
                            3
                        ),

                    "model_rtf":
                        round(
                            result[
                                "model_rtf"
                            ],
                            6
                        ),

                    "e2e_rtf":
                        round(
                            result[
                                "e2e_rtf"
                            ],
                            6
                        ),
                }
            )

            print(
                f"[{i:03d}/"
                f"{len(samples):03d}] "
                f"dur={duration_s:6.2f}s "
                f"err={result['word_errors']:2d}/"
                f"{result['reference_words']:2d} "
                f"in={result['input_tokens']:4d} "
                f"out={result['output_tokens']:3d} "
                f"model="
                f"{result['model_latency_ms']:8.2f}ms "
                f"e2e="
                f"{result['e2e_latency_ms']:8.2f}ms "
                f"RTF="
                f"{result['e2e_rtf']:.4f}"
            )

            if result[
                "word_errors"
            ] > 0:
                print(
                    "    REF :",
                    result[
                        "reference"
                    ]
                )

                print(
                    "    PRED:",
                    result[
                        "prediction"
                    ]
                )

    # ------------------------------------------------
    # Overall quality
    # ------------------------------------------------

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
        if utterance_wers
        else 0.0
    )

    # ------------------------------------------------
    # Fixed [4,6) latency subset
    # ------------------------------------------------

    fixed_records = [
        x
        for x in predictions
        if (
            args.latency_min
            <= x["duration_s"]
            < args.latency_max
        )
    ]

    fixed_model_ms = [
        x["model_latency_ms"]
        for x in fixed_records
    ]

    fixed_e2e_ms = [
        x["e2e_latency_ms"]
        for x in fixed_records
    ]

    fixed_model_rtf = [
        x["model_rtf"]
        for x in fixed_records
    ]

    fixed_e2e_rtf = [
        x["e2e_rtf"]
        for x in fixed_records
    ]

    fixed_duration_s = sum(
        x["duration_s"]
        for x in fixed_records
    )

    fixed_model_time_s = sum(
        x["model_latency_ms"]
        for x in fixed_records
    ) / 1000.0

    fixed_e2e_time_s = sum(
        x["e2e_latency_ms"]
        for x in fixed_records
    ) / 1000.0

    fixed_summary = {
        "duration_range_s": [
            args.latency_min,
            args.latency_max,
        ],

        "interval":
            "[min, max)",

        "num_samples":
            len(
                fixed_records
            ),

        "model_latency":
            latency_stats(
                fixed_model_ms
            ),

        "e2e_latency":
            latency_stats(
                fixed_e2e_ms
            ),

        "model_rtf":
            rtf_stats(
                fixed_model_rtf
            ),

        "e2e_rtf":
            rtf_stats(
                fixed_e2e_rtf
            ),

        "aggregate_model_rtf":
            (
                round(
                    fixed_model_time_s
                    / fixed_duration_s,
                    6
                )
                if fixed_duration_s > 0
                else None
            ),

        "aggregate_e2e_rtf":
            (
                round(
                    fixed_e2e_time_s
                    / fixed_duration_s,
                    6
                )
                if fixed_duration_s > 0
                else None
            ),

        "model_request_service_rate_proxy_qps":
            (
                round(
                    1000.0
                    / np.mean(
                        fixed_model_ms
                    ),
                    3
                )
                if fixed_model_ms
                else None
            ),

        "e2e_request_service_rate_proxy_qps":
            (
                round(
                    1000.0
                    / np.mean(
                        fixed_e2e_ms
                    ),
                    3
                )
                if fixed_e2e_ms
                else None
            ),
    }

    # ------------------------------------------------
    # Memory
    # ------------------------------------------------

    peak_allocated_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    # ------------------------------------------------
    # Summary
    # ------------------------------------------------

    summary = {
        "model":
            "Qwen2.5-Omni-7B",

        "task":
            "asr",

        "dataset":
            "LibriSpeech",

        "split":
            "test-clean",

        "model_role":
            "generalist",

        "precision":
            actual_dtype,

        "batch_size":
            1,

        "audio_output_enabled":
            False,

        "generation": {
            "do_sample":
                False,

            "return_audio":
                False,

            "max_new_tokens":
                args.max_new_tokens,

            "max_new_tokens_hits":
                max_new_tokens_hits,
        },

        "multimodal_processing": {
            "qwen_omni_utils":
                True,

            "qwen_omni_utils_version":
                "0.0.9",

            "use_audio_in_video":
                False,

            "required_audio_inputs": [
                "input_features",
                "feature_attention_mask",
            ],
        },

        "prompt_version":
            prompt_info["version"],

        "prompt":
            instruction,

        "normalization":
            (
                "lowercase + remove punctuation "
                "except apostrophe + normalize spaces"
            ),

        "reference_source":
            (
                "raw manifest label; "
                "normalized_label field is not "
                "used for WER"
            ),

        "generated_token_extraction":
            (
                "output[:, input_length:] "
                "before batch_decode"
            ),

        "num_samples":
            len(samples),

        "total_audio_duration_s":
            round(
                total_audio_duration_s,
                3
            ),

        "long_utterances_gt_30s":
            long_utterances_gt_30s,

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

        "input_tokens": {
            "mean":
                round(
                    float(
                        np.mean(
                            input_tokens
                        )
                    ),
                    3
                ),

            "min":
                int(
                    min(
                        input_tokens
                    )
                ),

            "max":
                int(
                    max(
                        input_tokens
                    )
                ),
        },

        "output_tokens": {
            "total":
                int(
                    sum(
                        output_tokens
                    )
                ),

            "mean":
                round(
                    float(
                        np.mean(
                            output_tokens
                        )
                    ),
                    3
                ),

            "min":
                int(
                    min(
                        output_tokens
                    )
                ),

            "max":
                int(
                    max(
                        output_tokens
                    )
                ),
        },

        "parameters_m":
            round(
                parameters_m,
                3
            ),

        "checkpoint_safetensors_total_mb":
            round(
                checkpoint_size_mb(
                    MODEL_DIR
                ),
                3
            ),

        "checkpoint_note":
            (
                "Checkpoint disk size includes "
                "components not necessarily loaded "
                "when enable_audio_output=False."
            ),

        "model_load_time_s":
            round(
                model_load_time_s,
                3
            ),

        "warmup_runs":
            args.warmup,

        "all_samples": {
            "model_latency":
                latency_stats(
                    model_latencies
                ),

            "e2e_latency":
                latency_stats(
                    e2e_latencies
                ),

            "model_rtf":
                rtf_stats(
                    model_rtfs
                ),

            "e2e_rtf":
                rtf_stats(
                    e2e_rtfs
                ),

            "aggregate_model_rtf":
                round(
                    total_model_time_s
                    / total_audio_duration_s,
                    6
                ),

            "aggregate_e2e_rtf":
                round(
                    total_e2e_time_s
                    / total_audio_duration_s,
                    6
                ),

            "model_request_service_rate_proxy_qps":
                round(
                    1000.0
                    / np.mean(
                        model_latencies
                    ),
                    3
                ),

            "e2e_request_service_rate_proxy_qps":
                round(
                    1000.0
                    / np.mean(
                        e2e_latencies
                    ),
                    3
                ),
        },

        "fixed_latency_subset":
            fixed_summary,

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
                "framework-specific auxiliary data; "
                "final deployment VRAM comparison "
                "will use process-level NVML."
            ),

        "latency_definition": {
            "model_latency":
                (
                    "synchronized model.generate() "
                    "after local audio loading, "
                    "process_mm_info, processor "
                    "execution and H2D; includes "
                    "audio encoder and autoregressive "
                    "text generation"
                ),

            "e2e_latency":
                (
                    "message construction + local "
                    "FLAC read/decode in "
                    "process_mm_info + chat template "
                    "+ audio processor + H2D + "
                    "model.generate() + generated "
                    "token slicing + decode + ASR "
                    "normalization"
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

    # ------------------------------------------------
    # Save
    # ------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.longest_only:

        prediction_path = (
            RESULT_DIR
            / "qwen_librispeech_longest_predictions.jsonl"
        )

        summary_path = (
            RESULT_DIR
            / "qwen_librispeech_longest_summary.json"
        )

    elif args.limit > 0:

        prediction_path = (
            RESULT_DIR
            / "qwen_librispeech_smoke_predictions.jsonl"
        )

        summary_path = (
            RESULT_DIR
            / "qwen_librispeech_smoke_summary.json"
        )

    else:

        prediction_path = (
            RESULT_DIR
            / "qwen_librispeech_predictions.jsonl"
        )

        summary_path = (
            RESULT_DIR
            / "qwen_librispeech_summary.json"
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

    # ------------------------------------------------
    # Console summary
    # ------------------------------------------------

    print()
    print(
        "========== SUMMARY =========="
    )

    print(
        "Samples:",
        len(samples)
    )

    print(
        "Total audio duration:",
        round(
            total_audio_duration_s,
            3
        ),
        "s"
    )

    print(
        "Long utterances >30s:",
        long_utterances_gt_30s
    )

    print(
        "Max-new-token hits:",
        max_new_tokens_hits
    )

    print(
        "Word errors / reference words:",
        f"{total_word_errors}/"
        f"{total_reference_words}"
    )

    print(
        "WER:",
        f"{corpus_wer:.6f}"
    )

    print(
        "Mean utterance WER:",
        f"{mean_utterance_wer:.6f}"
    )

    print(
        "Output tokens total:",
        sum(
            output_tokens
        )
    )

    print()
    print(
        "All model latency:",
        summary[
            "all_samples"
        ][
            "model_latency"
        ]
    )

    print(
        "All E2E latency:",
        summary[
            "all_samples"
        ][
            "e2e_latency"
        ]
    )

    print(
        "All model RTF:",
        summary[
            "all_samples"
        ][
            "model_rtf"
        ]
    )

    print(
        "All E2E RTF:",
        summary[
            "all_samples"
        ][
            "e2e_rtf"
        ]
    )

    print(
        "Aggregate model RTF:",
        summary[
            "all_samples"
        ][
            "aggregate_model_rtf"
        ]
    )

    print(
        "Aggregate E2E RTF:",
        summary[
            "all_samples"
        ][
            "aggregate_e2e_rtf"
        ]
    )

    print()
    print(
        "Fixed subset n:",
        fixed_summary[
            "num_samples"
        ]
    )

    print(
        "Fixed model latency:",
        fixed_summary[
            "model_latency"
        ]
    )

    print(
        "Fixed E2E latency:",
        fixed_summary[
            "e2e_latency"
        ]
    )

    print(
        "Fixed model RTF:",
        fixed_summary[
            "model_rtf"
        ]
    )

    print(
        "Fixed E2E RTF:",
        fixed_summary[
            "e2e_rtf"
        ]
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
