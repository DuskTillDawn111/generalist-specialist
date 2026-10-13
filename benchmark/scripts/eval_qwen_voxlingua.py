import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

import eval_qwen_librispeech as audio_base


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

MODEL_DIR = audio_base.MODEL_DIR

MANIFEST = (
    ROOT
    / "benchmark/manifests/lid.jsonl"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/lid_full.jsonl"
)

PROMPT_FILE = (
    ROOT
    / "benchmark/prompts/task_prompts.json"
)

RESULT_DIR = (
    ROOT
    / "benchmark/results/formal"
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

    if "lid" not in prompts:
        raise KeyError(
            "lid missing from "
            "task_prompts.json"
        )

    return prompts[
        "lid"
    ]


def normalize_lang_code(value):
    """
    Strict LID normalization.

    Only surrounding whitespace and
    case are normalized.

    Also accepts Whisper-style tokens
    such as <|en|> for compatibility.

    We deliberately DO NOT convert
    'English' -> 'en' or parse sentences.
    """

    value = (
        str(value)
        .strip()
        .lower()
    )

    if (
        value.startswith("<|")
        and value.endswith("|>")
    ):
        value = value[
            2:-2
        ]

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

    missing = [
        key
        for key in required
        if key not in sample
    ]

    if missing:
        raise ValueError(
            f"{sample.get('id')}: "
            f"missing keys {missing}"
        )

    if sample["task"] != "lid":
        raise ValueError(
            f"{sample['id']}: "
            f"unexpected task="
            f"{sample['task']}"
        )

    if sample["input_type"] != "audio":
        raise ValueError(
            f"{sample['id']}: "
            "input_type must be audio"
        )

    if int(
        sample[
            "sample_rate"
        ]
    ) != 16000:
        raise ValueError(
            f"{sample['id']}: "
            f"expected 16000 Hz, got "
            f"{sample['sample_rate']}"
        )


def scalar_stats(values):

    if not values:
        return {
            "total": 0,
            "mean": None,
            "min": None,
            "max": None,
        }

    return {
        "total":
            int(sum(values)),

        "mean":
            round(
                sum(values)
                / len(values),
                3
            ),

        "min":
            int(min(values)),

        "max":
            int(max(values)),
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
        default=0,
        help=(
            "0 means all benchmark samples"
        )
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16
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

    # ==================================
    # Prompt
    # ==================================

    prompt_info = load_prompt()

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

    # ==================================
    # Dataset
    # ==================================

    all_samples = read_manifest(
        MANIFEST
    )

    for sample in all_samples:
        validate_sample(
            sample
        )

    samples = list(
        all_samples
    )

    if args.limit > 0:
        samples = samples[
            :args.limit
        ]

    if not samples:
        raise RuntimeError(
            "No evaluation samples."
        )

    valid_labels = sorted(
        {
            normalize_lang_code(
                sample[
                    "label"
                ]
            )
            for sample in samples
        }
    )

    print(
        "Evaluation samples:",
        len(samples)
    )

    print(
        "Benchmark languages:",
        valid_labels
    )

    # ==================================
    # Processor
    # ==================================

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

    # ==================================
    # Model
    # ==================================

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

    audio_base.sync()

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
        "Model load time:",
        f"{model_load_time_s:.3f}s"
    )

    print(
        "Parameters:",
        f"{parameters_m:.3f}M"
    )

    # ==================================
    # Warmup
    # ==================================

    if args.warmup > 0:

        benchmark_ids = {
            sample[
                "id"
            ]
            for sample in samples
        }

        warmup_sample = None

        for candidate in read_manifest(
            FULL_MANIFEST
        ):

            validate_sample(
                candidate
            )

            if (
                candidate["id"]
                not in benchmark_ids
            ):
                warmup_sample = candidate
                break

        if warmup_sample is None:
            raise RuntimeError(
                "No separate LID warmup "
                "sample found."
            )

        print(
            "Full-pipeline warmup runs:",
            args.warmup
        )

        print(
            "Warmup sample:",
            warmup_sample[
                "id"
            ]
        )

        with torch.inference_mode():

            for _ in range(
                args.warmup
            ):

                audio_base.run_one(
                    model=model,
                    processor=processor,
                    instruction=instruction,
                    sample=warmup_sample,
                    max_new_tokens=(
                        args.max_new_tokens
                    ),
                )

    audio_base.sync()

    torch.cuda.reset_peak_memory_stats()

    # ==================================
    # Evaluation
    # ==================================

    predictions = []

    correct = 0
    valid_outputs = 0

    model_latencies = []
    e2e_latencies = []

    input_tokens = []
    output_tokens = []

    predicted_language_counts = (
        Counter()
    )

    per_language = {
        label: {
            "total": 0,
            "correct": 0,
        }
        for label in valid_labels
    }

    print()
    print(
        "Running Qwen LID evaluation..."
    )

    with torch.inference_mode():

        for i, sample in enumerate(
            samples,
            start=1
        ):

            result = (
                audio_base.run_one(
                    model=model,
                    processor=processor,
                    instruction=instruction,
                    sample=sample,
                    max_new_tokens=(
                        args.max_new_tokens
                    ),
                )
            )

            gt = normalize_lang_code(
                sample[
                    "label"
                ]
            )

            pred = normalize_lang_code(
                result[
                    "raw_response"
                ]
            )

            valid = (
                pred
                in valid_labels
            )

            is_correct = (
                valid
                and pred == gt
            )

            if valid:
                valid_outputs += 1

            if is_correct:
                correct += 1

            per_language[
                gt
            ][
                "total"
            ] += 1

            if is_correct:
                per_language[
                    gt
                ][
                    "correct"
                ] += 1

            predicted_language_counts[
                pred
            ] += 1

            model_ms = float(
                result[
                    "model_latency_ms"
                ]
            )

            e2e_ms = float(
                result[
                    "e2e_latency_ms"
                ]
            )

            model_latencies.append(
                model_ms
            )

            e2e_latencies.append(
                e2e_ms
            )

            input_tokens.append(
                int(
                    result[
                        "input_tokens"
                    ]
                )
            )

            output_tokens.append(
                int(
                    result[
                        "output_tokens"
                    ]
                )
            )

            record = {
                "id":
                    sample[
                        "id"
                    ],

                "task":
                    "lid",

                "dataset":
                    "VoxLingua107",

                "split":
                    sample[
                        "split"
                    ],

                "input":
                    sample[
                        "input"
                    ],

                "label":
                    gt,

                "language_name":
                    sample.get(
                        "language_name"
                    ),

                "source_label":
                    sample.get(
                        "source_label"
                    ),

                "raw_response":
                    result[
                        "raw_response"
                    ],

                "prediction":
                    pred,

                "valid_closed_set_output":
                    valid,

                "correct":
                    is_correct,

                "duration_s":
                    float(
                        sample[
                            "duration_s"
                        ]
                    ),

                "sample_rate":
                    int(
                        sample[
                            "sample_rate"
                        ]
                    ),

                "audio_samples":
                    result.get(
                        "audio_samples"
                    ),

                "audio_dtype":
                    result.get(
                        "audio_dtype"
                    ),

                "input_tokens":
                    int(
                        result[
                            "input_tokens"
                        ]
                    ),

                "output_tokens":
                    int(
                        result[
                            "output_tokens"
                        ]
                    ),

                "model_latency_ms":
                    round(
                        model_ms,
                        3
                    ),

                "e2e_latency_ms":
                    round(
                        e2e_ms,
                        3
                    ),
            }

            predictions.append(
                record
            )

            print(
                f"[{i:03d}/"
                f"{len(samples):03d}] "
                f"GT={gt:<5} "
                f"Pred={pred!r:<12} "
                f"{'OK' if is_correct else 'WRONG'} "
                f"model={model_ms:.2f}ms "
                f"e2e={e2e_ms:.2f}ms"
            )

    # ==================================
    # Summary
    # ==================================

    accuracy = (
        correct
        / len(samples)
    )

    valid_rate = (
        valid_outputs
        / len(samples)
    )

    model_stats = (
        audio_base.latency_stats(
            model_latencies
        )
    )

    e2e_stats = (
        audio_base.latency_stats(
            e2e_latencies
        )
    )

    per_language_summary = {}

    for label in valid_labels:

        total = (
            per_language[
                label
            ][
                "total"
            ]
        )

        lang_correct = (
            per_language[
                label
            ][
                "correct"
            ]
        )

        per_language_summary[
            label
        ] = {
            "total":
                total,

            "correct":
                lang_correct,

            "accuracy":
                (
                    round(
                        lang_correct
                        / total,
                        6
                    )
                    if total
                    else None
                ),
        }

    peak_allocated_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    model_mean = (
        model_stats[
            "mean_ms"
        ]
    )

    e2e_mean = (
        e2e_stats[
            "mean_ms"
        ]
    )

    summary = {
        "model":
            "Qwen2.5-Omni-7B",

        "task":
            "lid",

        "dataset":
            "VoxLingua107",

        "split":
            "dev",

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
        },

        "prompt_version":
            prompt_info[
                "version"
            ],

        "prompt":
            instruction,

        "output_normalization":
            (
                "strip + lowercase only; "
                "Whisper-style <|xx|> token "
                "wrapper accepted; no mapping "
                "from language names to codes"
            ),

        "evaluation_setting":
            (
                "closed-set spoken language "
                "identification over languages "
                "present in the benchmark manifest"
            ),

        "num_samples":
            len(samples),

        "num_languages":
            len(valid_labels),

        "benchmark_languages":
            valid_labels,

        "correct":
            correct,

        "accuracy":
            round(
                accuracy,
                6
            ),

        "invalid_closed_set_outputs":
            (
                len(samples)
                - valid_outputs
            ),

        "valid_closed_set_output_rate":
            round(
                valid_rate,
                6
            ),

        "per_language":
            per_language_summary,

        "predicted_language_counts":
            dict(
                predicted_language_counts
            ),

        "input_tokens":
            scalar_stats(
                input_tokens
            ),

        "output_tokens":
            scalar_stats(
                output_tokens
            ),

        "parameters_m":
            round(
                parameters_m,
                3
            ),

        "checkpoint_safetensors_total_mb":
            round(
                audio_base.checkpoint_size_mb(
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

        "model_latency":
            model_stats,

        "e2e_latency":
            e2e_stats,

        "model_request_service_rate_proxy_qps":
            (
                round(
                    1000.0
                    / model_mean,
                    6
                )
                if model_mean
                else None
            ),

        "e2e_request_service_rate_proxy_qps":
            (
                round(
                    1000.0
                    / e2e_mean,
                    6
                )
                if e2e_mean
                else None
            ),

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
                "Final deployment VRAM comparison "
                "uses process-level NVML."
            ),

        "latency_definition": {
            "model_latency":
                (
                    "synchronized model.generate() "
                    "including Qwen audio encoding "
                    "and autoregressive generation"
                ),

            "e2e_latency":
                (
                    "local audio read/decode + "
                    "multimodal audio preprocessing "
                    "+ H2D + model.generate() + "
                    "decode + language-code "
                    "normalization"
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
            datetime.now(
            ).astimezone(
            ).isoformat(),
    }

    # ==================================
    # Save
    # ==================================

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.limit > 0:

        prediction_path = (
            RESULT_DIR
            / (
                "qwen_voxlingua_"
                "smoke_predictions.jsonl"
            )
        )

        summary_path = (
            RESULT_DIR
            / (
                "qwen_voxlingua_"
                "smoke_summary.json"
            )
        )

    else:

        prediction_path = (
            RESULT_DIR
            / (
                "qwen_voxlingua_"
                "predictions.jsonl"
            )
        )

        summary_path = (
            RESULT_DIR
            / (
                "qwen_voxlingua_"
                "summary.json"
            )
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
        "===== SUMMARY ====="
    )

    print(
        "Accuracy:",
        f"{accuracy:.4f}"
    )

    print(
        "Valid closed-set output rate:",
        f"{valid_rate:.4f}"
    )

    print(
        "Model latency mean:",
        model_stats[
            "mean_ms"
        ],
        "ms"
    )

    print(
        "E2E latency mean:",
        e2e_stats[
            "mean_ms"
        ],
        "ms"
    )

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
