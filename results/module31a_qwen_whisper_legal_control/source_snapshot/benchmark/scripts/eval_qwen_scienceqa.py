import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

import eval_qwen_imagenette as image_base


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

MANIFEST = (
    ROOT
    / "benchmark/manifests/visual_qa.jsonl"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/visual_qa_full.jsonl"
)

PROMPT_PATH = (
    ROOT
    / "benchmark/prompts/task_prompts.json"
)

RESULT_DIR = (
    ROOT
    / "benchmark/results/formal"
)

MODEL_DIR = image_base.MODEL_DIR


def read_manifest(path):
    return image_base.read_manifest(
        path
    )


def load_prompt():
    with open(
        PROMPT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        prompts = json.load(f)

    if "visual_qa" not in prompts:
        raise KeyError(
            "visual_qa missing from "
            "task_prompts.json"
        )

    return prompts[
        "visual_qa"
    ]


def option_letter(index):
    if not 0 <= index < 26:
        raise ValueError(
            f"Unsupported option index: {index}"
        )

    return chr(
        ord("A") + index
    )


def valid_options(sample):
    choices = sample[
        "choices"
    ]

    return {
        option_letter(i)
        for i in range(
            len(choices)
        )
    }


def normalize_option(text):
    """
    Strict closed-set normalization.

    Only whitespace/case are normalized.

    Examples:
      "B"   -> "B"
      " b " -> "B"

    Outputs such as:
      "B."
      "(B)"
      "Answer: B"
      "The answer is B"

    remain invalid closed-set outputs.
    """

    return (
        str(text)
        .strip()
        .upper()
    )


def validate_sample(sample):

    required = [
        "id",
        "task",
        "dataset",
        "split",
        "input",
        "question",
        "choices",
        "label_id",
        "label",
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

    if sample["task"] != "visual_qa":
        raise ValueError(
            f"{sample['id']}: "
            f"unexpected task="
            f"{sample['task']}"
        )

    choices = sample[
        "choices"
    ]

    if not isinstance(
        choices,
        list
    ):
        raise TypeError(
            f"{sample['id']}: "
            "choices must be list"
        )

    if not 2 <= len(choices) <= 26:
        raise ValueError(
            f"{sample['id']}: "
            f"invalid choices count="
            f"{len(choices)}"
        )

    label_id = int(
        sample[
            "label_id"
        ]
    )

    if not (
        0
        <= label_id
        < len(choices)
    ):
        raise ValueError(
            f"{sample['id']}: "
            f"invalid label_id="
            f"{label_id}"
        )

    expected = option_letter(
        label_id
    )

    actual = normalize_option(
        sample[
            "label"
        ]
    )

    if actual != expected:
        raise ValueError(
            f"{sample['id']}: "
            f"label mismatch: "
            f"label_id={label_id}, "
            f"label={actual}, "
            f"expected={expected}"
        )


def build_instruction(
    task_prompt,
    sample
):

    parts = [
        task_prompt.strip()
    ]

    hint = (
        sample.get(
            "hint"
        )
        or ""
    ).strip()

    if hint:
        parts.extend(
            [
                "",
                "Context:",
                hint,
            ]
        )

    parts.extend(
        [
            "",
            "Question:",
            sample[
                "question"
            ].strip(),
            "",
            "Choices:",
        ]
    )

    for i, choice in enumerate(
        sample["choices"]
    ):
        parts.append(
            f"{option_letter(i)}. "
            f"{str(choice).strip()}"
        )

    return "\n".join(
        parts
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
                float(
                    sum(values)
                    / len(values)
                ),
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

    task_prompt = (
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

    print(
        "Evaluation samples:",
        len(samples)
    )

    choice_count_distribution = Counter(
        len(
            sample[
                "choices"
            ]
        )
        for sample in samples
    )

    print(
        "Choice counts:",
        dict(
            sorted(
                choice_count_distribution.items()
            )
        )
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

    image_base.sync()

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

    torch_idle_allocated_mb = (
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
    # Full-pipeline warmup
    # ==================================

    benchmark_ids = {
        sample["id"]
        for sample in samples
    }

    warmup_sample = None

    if args.warmup > 0:

        full_samples = (
            read_manifest(
                FULL_MANIFEST
            )
        )

        for candidate in full_samples:

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
                "No separate warmup sample "
                "found in visual_qa_full.jsonl."
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

        warmup_instruction = (
            build_instruction(
                task_prompt,
                warmup_sample
            )
        )

        with torch.inference_mode():

            for _ in range(
                args.warmup
            ):

                image_base.run_one(
                    model=model,
                    processor=processor,
                    instruction=(
                        warmup_instruction
                    ),
                    sample=(
                        warmup_sample
                    ),
                    max_new_tokens=(
                        args.max_new_tokens
                    ),
                )

    image_base.sync()

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

    answer_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
        }
    )

    print()
    print(
        "Running ScienceQA-IMG evaluation..."
    )

    with torch.inference_mode():

        for i, sample in enumerate(
            samples,
            start=1
        ):

            instruction = (
                build_instruction(
                    task_prompt,
                    sample
                )
            )

            result = (
                image_base.run_one(
                    model=model,
                    processor=processor,
                    instruction=instruction,
                    sample=sample,
                    max_new_tokens=(
                        args.max_new_tokens
                    ),
                )
            )

            gt = normalize_option(
                sample[
                    "label"
                ]
            )

            pred = normalize_option(
                result[
                    "raw_response"
                ]
            )

            allowed = (
                valid_options(
                    sample
                )
            )

            valid = (
                pred
                in allowed
            )

            is_correct = (
                valid
                and pred == gt
            )

            if valid:
                valid_outputs += 1

            if is_correct:
                correct += 1

            answer_stats[
                gt
            ][
                "total"
            ] += 1

            if is_correct:
                answer_stats[
                    gt
                ][
                    "correct"
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
                    sample["id"],

                "task":
                    "visual_qa",

                "dataset":
                    "ScienceQA",

                "split":
                    sample[
                        "split"
                    ],

                "input":
                    sample[
                        "input"
                    ],

                "question":
                    sample[
                        "question"
                    ],

                "choices":
                    sample[
                        "choices"
                    ],

                "hint":
                    sample.get(
                        "hint"
                    ),

                "label_id":
                    int(
                        sample[
                            "label_id"
                        ]
                    ),

                "label":
                    gt,

                "raw_response":
                    result[
                        "raw_response"
                    ],

                "normalized_response":
                    pred,

                "valid_options":
                    sorted(
                        allowed
                    ),

                "valid_closed_set_output":
                    valid,

                "correct":
                    is_correct,

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

                "processed_image_width":
                    result.get(
                        "image_width"
                    ),

                "processed_image_height":
                    result.get(
                        "image_height"
                    ),

                "pixel_values_shape":
                    result.get(
                        "pixel_values_shape"
                    ),

                "image_grid_thw":
                    result.get(
                        "image_grid_thw"
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
                f"GT={gt:<2} "
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
        image_base.latency_stats(
            model_latencies
        )
    )

    e2e_stats = (
        image_base.latency_stats(
            e2e_latencies
        )
    )

    answer_summary = {}

    for label in sorted(
        answer_stats
    ):

        total = (
            answer_stats[
                label
            ][
                "total"
            ]
        )

        cls_correct = (
            answer_stats[
                label
            ][
                "correct"
            ]
        )

        answer_summary[
            label
        ] = {
            "total":
                total,

            "correct":
                cls_correct,

            "accuracy":
                (
                    round(
                        cls_correct
                        / total,
                        6
                    )
                    if total
                    else None
                ),
        }

    torch_peak_allocated_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    model_mean = (
        model_stats.get(
            "mean_ms"
        )
    )

    e2e_mean = (
        e2e_stats.get(
            "mean_ms"
        )
    )

    summary = {
        "model":
            "Qwen2.5-Omni-7B",

        "task":
            "visual_qa",

        "dataset":
            "ScienceQA",

        "subset":
            "IMG",

        "split":
            "validation",

        "evaluation_setting":
            (
                "ScienceQA validation samples "
                "with image context; "
                "multiple-choice closed-set "
                "answer prediction"
            ),

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

        "multimodal_processing": {
            "qwen_omni_utils":
                True,

            "use_audio_in_video":
                False,

            "required_vision_inputs": [
                "pixel_values",
                "image_grid_thw",
            ],
        },

        "prompt_version":
            prompt_info[
                "version"
            ],

        "prompt":
            task_prompt,

        "prompt_composition":
            (
                "<task prompt> + optional Context "
                "+ Question + enumerated Choices"
            ),

        "output_normalization":
            (
                "strip + uppercase only; "
                "prediction must exactly equal "
                "one valid option letter"
            ),

        "generated_token_extraction":
            (
                "output[:, input_length:] "
                "before batch_decode"
            ),

        "num_samples":
            len(samples),

        "choice_count_distribution": {
            str(k): v
            for k, v in sorted(
                choice_count_distribution.items()
            )
        },

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

        "per_answer":
            answer_summary,

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
                image_base.checkpoint_size_mb(
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
                torch_idle_allocated_mb,
                3
            ),

        "torch_peak_allocated_mb":
            round(
                torch_peak_allocated_mb,
                3
            ),

        "memory_note":
            (
                "Torch allocator memory is "
                "framework-specific auxiliary data; "
                "final deployment VRAM comparison "
                "uses process-level NVML."
            ),

        "latency_definition": {
            "model_latency":
                (
                    "synchronized model.generate() "
                    "including visual encoding"
                ),

            "e2e_latency":
                (
                    "instruction construction + "
                    "local image read/decode + "
                    "multimodal preprocessing + "
                    "H2D + model.generate() + "
                    "decode + closed-set "
                    "postprocessing"
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
                "qwen_scienceqa_"
                "smoke_predictions.jsonl"
            )
        )

        summary_path = (
            RESULT_DIR
            / (
                "qwen_scienceqa_"
                "smoke_summary.json"
            )
        )

    else:

        prediction_path = (
            RESULT_DIR
            / "qwen_scienceqa_predictions.jsonl"
        )

        summary_path = (
            RESULT_DIR
            / "qwen_scienceqa_summary.json"
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
        model_stats.get(
            "mean_ms"
        ),
        "ms"
    )

    print(
        "E2E latency mean:",
        e2e_stats.get(
            "mean_ms"
        ),
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
