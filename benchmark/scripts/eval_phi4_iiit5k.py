import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

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
    / "benchmark/manifests/ocr.jsonl"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/ocr_full.jsonl"
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
    / "phi4_iiit5k_summary.json"
)

PREDICTION_PATH = (
    RESULT_DIR
    / "phi4_iiit5k_predictions.jsonl"
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

    return prompts["ocr"]


def normalize_ocr(text):
    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]",
        "",
        text
    )

    return text


def levenshtein_chars(a, b):
    n = len(a)
    m = len(b)

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
                if a[i - 1]
                == b[j - 1]
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
    user_text = (
        "<|image_1|>\n"
        + instruction
    )

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
                float(np.mean(values)),
                3
            ),

        "p50_ms":
            round(
                percentile(values, 50),
                3
            ),

        "p95_ms":
            round(
                percentile(values, 95),
                3
            ),

        "p99_ms":
            round(
                percentile(values, 99),
                3
            ),

        "min_ms":
            round(
                float(np.min(values)),
                3
            ),

        "max_ms":
            round(
                float(np.max(values)),
                3
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

        if torch.is_tensor(value):
            moved[key] = value.to(
                device
            )
        else:
            moved[key] = value

    return moved


def generate_one(
    model,
    processor,
    prompt,
    image_path,
    device,
    max_new_tokens
):
    sync()

    e2e_start = (
        time.perf_counter()
    )

    # -----------------------------
    # Local image read / decode
    # -----------------------------

    with Image.open(
        image_path
    ) as img:

        image_width = img.width
        image_height = img.height

        image = img.convert(
            "RGB"
        )

        # -------------------------
        # Multimodal preprocessing
        # -------------------------

        inputs = processor(
            text=prompt,
            images=image,
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
    # Vision + language generation
    # -----------------------------

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

    # -----------------------------
    # Decode continuation
    # -----------------------------

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
        normalize_ocr(
            response
        )
    )

    e2e_end = (
        time.perf_counter()
    )

    return {
        "response":
            response,

        "normalized_response":
            normalized,

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "image_width":
            image_width,

        "image_height":
            image_height,

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
        "--max-new-tokens",
        type=int,
        default=32
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
        "Decoding: do_sample=False"
    )

    print(
        "Max new tokens:",
        args.max_new_tokens
    )

    # -----------------------------
    # Prompt
    # -----------------------------

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

    # -----------------------------
    # Dataset
    # -----------------------------

    all_benchmark_samples = (
        read_manifest(
            MANIFEST
        )
    )

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

    # -----------------------------
    # Processor
    # -----------------------------

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

    # -----------------------------
    # Model
    # -----------------------------

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

    # -----------------------------
    # Separate warmup sample
    # -----------------------------

    benchmark_ids = {
        x["id"]
        for x in all_benchmark_samples
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
            "Could not find separate "
            "warmup image."
        )

    print(
        "Full-pipeline warmup runs:",
        args.warmup
    )

    print(
        "Warmup image:",
        warmup_sample["input"]
    )

    with torch.inference_mode():

        for _ in range(
            args.warmup
        ):

            _ = generate_one(
                model=model,
                processor=processor,
                prompt=chat_prompt,
                image_path=(
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

    # -----------------------------
    # Evaluation
    # -----------------------------

    predictions = []

    model_latencies = []
    e2e_latencies = []

    input_token_counts = []
    output_token_counts = []

    correct = 0

    total_char_errors = 0
    total_reference_chars = 0

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
                prompt=chat_prompt,
                image_path=(
                    sample["input"]
                ),
                device=device,
                max_new_tokens=(
                    args.max_new_tokens
                )
            )

            reference = (
                normalize_ocr(
                    sample[
                        "normalized_label"
                    ]
                )
            )

            prediction = (
                result[
                    "normalized_response"
                ]
            )

            is_correct = (
                prediction
                == reference
            )

            if is_correct:
                correct += 1

            char_errors = (
                levenshtein_chars(
                    reference,
                    prediction
                )
            )

            ref_chars = len(
                reference
            )

            total_char_errors += (
                char_errors
            )

            total_reference_chars += (
                ref_chars
            )

            sample_cer = (
                char_errors
                / ref_chars
                if ref_chars > 0
                else 0.0
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
                    "ocr",

                "dataset":
                    "iiit5k",

                "split":
                    "test",

                "input":
                    sample["input"],

                "label":
                    sample["label"],

                "normalized_reference":
                    reference,

                "raw_response":
                    result[
                        "response"
                    ],

                "normalized_response":
                    prediction,

                "correct":
                    is_correct,

                "char_errors":
                    char_errors,

                "reference_chars":
                    ref_chars,

                "cer":
                    round(
                        sample_cer,
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

                "image_width":
                    result[
                        "image_width"
                    ],

                "image_height":
                    result[
                        "image_height"
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
                f"GT={reference:<18} "
                f"Pred={prediction[:22]:<22} "
                f"{'OK' if is_correct else 'WRONG'} "
                f"in={result['input_tokens']:>4} "
                f"out={result['output_tokens']:>2} "
                f"model={model_latency_ms:8.2f}ms "
                f"e2e={e2e_latency_ms:8.2f}ms"
            )

    # -----------------------------
    # Summary
    # -----------------------------

    exact_match = (
        correct
        / len(samples)
    )

    corpus_cer = (
        total_char_errors
        / total_reference_chars
        if total_reference_chars > 0
        else 0.0
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

    output_token_rate = (
        total_output_tokens
        / total_model_time_s
        if total_model_time_s > 0
        else 0.0
    )

    summary = {
        "model":
            "Phi-4-multimodal-instruct",

        "task":
            "optical_character_recognition",

        "dataset":
            "IIIT5K",

        "split":
            "test",

        "evaluation_mode":
            "single-word recognition",

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
                "<|image_1|>\\n"
                "<OCR prompt>"
            ),

        "normalization":
            (
                "lowercase; remove all "
                "non-alphanumeric characters"
            ),

        "num_samples":
            len(samples),

        "correct":
            correct,

        "exact_match_accuracy":
            round(
                exact_match,
                6
            ),

        "total_character_errors":
            total_char_errors,

        "total_reference_characters":
            total_reference_chars,

        "cer":
            round(
                corpus_cer,
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

        "effective_output_tokens_per_s":
            round(
                output_token_rate,
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
                "auxiliary only; final deployment "
                "VRAM comparison will use "
                "process-level NVML."
            ),

        "latency_definition": {
            "model_latency":
                (
                    "multimodal model.generate only "
                    "after image decoding, "
                    "multimodal preprocessing and "
                    "H2D; includes vision encoding "
                    "and autoregressive generation"
                ),

            "e2e_latency":
                (
                    "local image read/decode + "
                    "multimodal preprocessing + "
                    "H2D + vision encoding + "
                    "autoregressive generation + "
                    "text decoding + "
                    "OCR normalization"
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

    # -----------------------------
    # Save
    # -----------------------------

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
        "Exact Match:",
        f"{exact_match:.4f}"
    )

    print(
        "CER:",
        f"{corpus_cer:.4f}"
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
