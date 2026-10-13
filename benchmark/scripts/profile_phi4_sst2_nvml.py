import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)


ROOT = Path("/root/autodl-tmp/generalist-specialist")

MODEL_DIR = (
    ROOT / "models/phi4-multimodal"
)

MANIFEST = (
    ROOT
    / "benchmark/manifests/text_classification.jsonl"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/text_classification_full.jsonl"
)

PROMPT_FILE = (
    ROOT
    / "benchmark/prompts/task_prompts.json"
)

RESULT_DIR = (
    ROOT
    / "benchmark/results/nvml_probe_temp"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "phi4_sst2_summary.json"
)

PREDICTION_PATH = (
    RESULT_DIR
    / "phi4_sst2_predictions.jsonl"
)

LABELS = [
    "negative",
    "positive",
]


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

    return prompts[
        "text_classification"
    ]


def build_user_text(
    instruction,
    sentence
):
    return (
        instruction
        + "\n\nSentence:\n"
        + sentence
    )


def build_chat_prompt(
    tokenizer,
    user_text
):
    messages = [
        {
            "role": "user",
            "content": user_text,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def normalize_closed_label(text):
    return text.strip().lower()


def percentile(values, q):
    return float(
        np.percentile(
            values,
            q
        )
    )


def stats(values):
    values = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "mean_ms":
            round(
                float(
                    np.mean(values)
                ),
                3
            ),

        "p50_ms":
            round(
                percentile(
                    values,
                    50
                ),
                3
            ),

        "p95_ms":
            round(
                percentile(
                    values,
                    95
                ),
                3
            ),

        "p99_ms":
            round(
                percentile(
                    values,
                    99
                ),
                3
            ),

        "min_ms":
            round(
                float(
                    np.min(values)
                ),
                3
            ),

        "max_ms":
            round(
                float(
                    np.max(values)
                ),
                3
            ),
    }


def sync():
    torch.cuda.synchronize()


def checkpoint_size_mb(path):
    """
    只统计 safetensors checkpoint，
    避免目录中若存在兼容性 bin 文件时重复计算。
    """
    total = 0

    files = list(
        path.glob("*.safetensors")
    )

    for p in files:
        total += p.stat().st_size

    return (
        total / (1024 ** 2)
    )


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


def generate_one(
    model,
    processor,
    instruction,
    sentence,
    device,
    max_new_tokens
):
    """
    返回：
        response
        input_tokens
        output_tokens
        generate latency ms
        e2e latency ms
    """

    sync()

    e2e_start = (
        time.perf_counter()
    )

    # -----------------------------
    # Prompt construction
    # -----------------------------

    user_text = build_user_text(
        instruction,
        sentence
    )

    prompt = build_chat_prompt(
        processor.tokenizer,
        user_text
    )

    # -----------------------------
    # Tokenization
    # -----------------------------

    inputs = processor(
        text=prompt,
        return_tensors="pt"
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

    # -----------------------------
    # Autoregressive generation
    # -----------------------------

    model_start = (
        time.perf_counter()
    )

    generated_ids = (
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True
        )
    )

    sync()

    model_end = (
        time.perf_counter()
    )

    # -----------------------------
    # Decode only generated tokens
    # -----------------------------

    new_ids = generated_ids[
        :,
        input_tokens:
    ]

    output_tokens = int(
        new_ids.shape[-1]
    )

    response = (
        processor.tokenizer.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    )

    normalized = (
        normalize_closed_label(
            response
        )
    )

    e2e_end = (
        time.perf_counter()
    )

    model_latency_ms = (
        model_end
        - model_start
    ) * 1000

    e2e_latency_ms = (
        e2e_end
        - e2e_start
    ) * 1000

    return {
        "response":
            response,

        "normalized_response":
            normalized,

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "model_latency_ms":
            model_latency_ms,

        "e2e_latency_ms":
            e2e_latency_ms,
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
        help="0 means evaluate all 200 samples"
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

    # --------------------------------
    # Prompt
    # --------------------------------

    prompt_info = (
        load_task_prompt()
    )

    instruction = (
        prompt_info["prompt"]
    )

    print(
        "Prompt version:",
        prompt_info["version"]
    )

    # --------------------------------
    # Dataset
    # --------------------------------

    samples = read_manifest(
        MANIFEST
    )

    for sample in samples:

        label_id = int(
            sample["label_id"]
        )

        if label_id not in (
            0,
            1
        ):
            raise ValueError(
                f"Unexpected label_id: "
                f"{label_id}"
            )

        expected = LABELS[
            label_id
        ]

        if (
            sample["label"]
            .strip()
            .lower()
            != expected
        ):
            raise ValueError(
                f"Manifest label mismatch "
                f"for {sample['id']}"
            )

    if args.limit > 0:
        samples = samples[
            :args.limit
        ]

    print(
        "Evaluation samples:",
        len(samples)
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

    print(
        "NVML_MARKER|post_load",
        flush=True
    )

    time.sleep(1.0)

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
        model.config._attn_implementation
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
    # Full-pipeline warmup
    # --------------------------------

    benchmark_ids = {
        sample["id"]
        for sample in samples
    }

    warmup_sample = None

    for candidate in read_manifest(
        FULL_MANIFEST
    ):

        if (
            candidate["id"]
            not in benchmark_ids
        ):
            warmup_sample = candidate
            break

    if warmup_sample is None:
        raise RuntimeError(
            "Unable to find separate warmup sample."
        )

    print(
        "Full-pipeline warmup runs:",
        args.warmup
    )

    print(
        "Warmup text:",
        warmup_sample[
            "input"
        ][:100]
    )

    with torch.inference_mode():

        for _ in range(
            args.warmup
        ):

            _ = generate_one(
                model=model,
                processor=processor,
                instruction=instruction,
                sentence=warmup_sample[
                    "input"
                ],
                device=device,
                max_new_tokens=args.max_new_tokens
            )

    torch.cuda.reset_peak_memory_stats(
        device
    )

    sync()

    print(
        "NVML_MARKER|post_warmup",
        flush=True
    )

    time.sleep(1.0)

    # --------------------------------
    # Formal evaluation
    # --------------------------------

    predictions = []

    model_latencies = []
    e2e_latencies = []

    input_token_counts = []
    output_token_counts = []

    correct = 0

    total_model_time_s = 0.0
    total_e2e_time_s = 0.0

    print()
    print(
        "Running evaluation..."
    )

    with torch.inference_mode():

        for i, sample in enumerate(
            samples,
            start=1
        ):

            result = generate_one(
                model=model,
                processor=processor,
                instruction=instruction,
                sentence=sample[
                    "input"
                ],
                device=device,
                max_new_tokens=args.max_new_tokens
            )

            gt_id = int(
                sample[
                    "label_id"
                ]
            )

            gt_label = LABELS[
                gt_id
            ]

            predicted_label = (
                result[
                    "normalized_response"
                ]
            )

            is_correct = (
                predicted_label
                == gt_label
            )

            if is_correct:
                correct += 1

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

            model_latencies.append(
                model_latency_ms
            )

            e2e_latencies.append(
                e2e_latency_ms
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

            total_model_time_s += (
                model_latency_ms
                / 1000.0
            )

            total_e2e_time_s += (
                e2e_latency_ms
                / 1000.0
            )

            record = {
                "id":
                    sample["id"],

                "task":
                    "text_classification",

                "dataset":
                    "sst2",

                "split":
                    "validation",

                "input":
                    sample["input"],

                "label_id":
                    gt_id,

                "label":
                    gt_label,

                "raw_response":
                    result[
                        "response"
                    ],

                "normalized_response":
                    predicted_label,

                "correct":
                    is_correct,

                "input_tokens":
                    result[
                        "input_tokens"
                    ],

                "output_tokens":
                    result[
                        "output_tokens"
                    ],

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
            }

            predictions.append(
                record
            )

            print(
                f"[{i:03d}/"
                f"{len(samples):03d}] "
                f"GT={gt_label:<8} "
                f"Pred="
                f"{predicted_label[:20]:<20} "
                f"{'OK' if is_correct else 'WRONG'} "
                f"in={result['input_tokens']:>3} "
                f"out={result['output_tokens']:>2} "
                f"model={model_latency_ms:7.2f}ms "
                f"e2e={e2e_latency_ms:7.2f}ms"
            )

    # --------------------------------
    sync()

    print(
        "NVML_MARKER|post_inference",
        flush=True
    )

    time.sleep(1.0)

    # Summary
    # --------------------------------

    accuracy = (
        correct
        / len(samples)
    )

    model_stats = stats(
        model_latencies
    )

    e2e_stats = stats(
        e2e_latencies
    )

    peak_allocated_mb = (
        torch.cuda.max_memory_allocated(
            device
        )
        / 1024**2
    )

    total_output_tokens = int(
        sum(
            output_token_counts
        )
    )

    generation_token_rate = (
        total_output_tokens
        / total_model_time_s
        if total_model_time_s > 0
        else 0.0
    )

    summary = {
        "model":
            "Phi-4-multimodal-instruct",

        "task":
            "text_classification",

        "dataset":
            "SST-2",

        "split":
            "validation",

        "model_role":
            "generalist",

        "precision":
            actual_dtype,

        "attention_backend":
            "sdpa",

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
                "<task prompt>\\n\\n"
                "Sentence:\\n"
                "<original SST-2 sentence>"
            ),

        "output_normalization":
            "strip + lowercase only",

        "num_samples":
            len(samples),

        "correct":
            correct,

        "accuracy":
            round(
                accuracy,
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
                total_output_tokens,

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

        "model_latency":
            model_stats,

        "e2e_latency":
            e2e_stats,

        "model_request_throughput_qps":
            round(
                len(samples)
                / total_model_time_s,
                3
            ),

        "e2e_request_throughput_qps":
            round(
                len(samples)
                / total_e2e_time_s,
                3
            ),

        "generation_output_tokens_per_s":
            round(
                generation_token_rate,
                3
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
                "cross-model deployment VRAM will "
                "later use process-level NVML."
            ),

        "latency_definition": {
            "model_latency":
                (
                    "autoregressive model.generate "
                    "only, after tokenization and H2D"
                ),

            "e2e_latency":
                (
                    "task/chat prompt construction + "
                    "tokenization + H2D + "
                    "autoregressive generation + "
                    "generated-text decoding + "
                    "label normalization"
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
    # --------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        PREDICTION_PATH,
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
        SUMMARY_PATH,
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
        "Correct:",
        correct
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        "Model latency:",
        model_stats
    )

    print(
        "E2E latency:",
        e2e_stats
    )

    print(
        "Model request throughput:",
        summary[
            "model_request_throughput_qps"
        ],
        "qps"
    )

    print(
        "E2E request throughput:",
        summary[
            "e2e_request_throughput_qps"
        ],
        "qps"
    )

    print(
        "Generation rate:",
        summary[
            "generation_output_tokens_per_s"
        ],
        "output tokens/s"
    )

    print()
    print(
        "Saved predictions:",
        PREDICTION_PATH
    )

    print(
        "Saved summary:",
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()
