import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import transformers
from transformers import AutoModel


ROOT = Path("/root/autodl-tmp/generalist-specialist")

MODEL_DIR = ROOT / "models/minicpm-o-4_5"

MANIFEST = (
    ROOT
    / "benchmark/manifests/image_classification.jsonl"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/image_classification_full.jsonl"
)

PROMPT_FILE = (
    ROOT
    / "benchmark/prompts/task_prompts.json"
)

RESULT_DIR = (
    ROOT
    / "benchmark/results/formal"
)

PREDICTION_PATH = (
    RESULT_DIR
    / "minicpm_imagenette_predictions.jsonl"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "minicpm_imagenette_summary.json"
)


TARGET_NAMES = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
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


def load_prompt():
    with open(
        PROMPT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        prompts = json.load(f)

    return prompts[
        "image_classification"
    ]


def normalize_label(text):
    return text.strip().lower()


def sync():
    torch.cuda.synchronize()


def stats(values):
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


def checkpoint_size_mb(path):
    total = 0

    for p in path.glob(
        "*.safetensors"
    ):
        total += p.stat().st_size

    return total / (1024 ** 2)


def run_one(
    model,
    instruction,
    image_path,
    max_new_tokens
):
    # --------------------------------
    # E2E starts before local
    # image read / decode
    # --------------------------------

    sync()

    e2e_start = (
        time.perf_counter()
    )

    with Image.open(
        image_path
    ) as img:

        image_width = img.width
        image_height = img.height

        image = img.convert(
            "RGB"
        )

    msgs = [
        {
            "role": "user",
            "content": [
                image,
                instruction,
            ],
        }
    ]

    # --------------------------------
    # chat latency:
    # MiniCPM internal vision
    # preprocessing + vision encoder
    # + generation + decoding
    # --------------------------------

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

    normalized = (
        normalize_label(
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

        "image_width":
            image_width,

        "image_height":
            image_height,

        "chat_latency_ms":
            (
                chat_end
                - chat_start
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

    # --------------------------------
    # Prompt
    # --------------------------------

    prompt_info = (
        load_prompt()
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

    all_samples = read_manifest(
        MANIFEST
    )

    for sample in all_samples:

        label_id = int(
            sample[
                "label_id"
            ]
        )

        if not (
            0
            <= label_id
            < len(
                TARGET_NAMES
            )
        ):
            raise ValueError(
                f"Invalid label_id: "
                f"{label_id}"
            )

        expected = (
            TARGET_NAMES[
                label_id
            ]
        )

        if (
            sample[
                "label"
            ]
            != expected
        ):
            raise ValueError(
                "Manifest label "
                f"mismatch for "
                f"{sample['id']}"
            )

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

    # --------------------------------
    # Model
    # --------------------------------

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

    actual_attention = (
        getattr(
            model.config,
            "_attn_implementation",
            getattr(
                model.config,
                "attn_implementation",
                "unknown"
            )
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

    # --------------------------------
    # Separate warmup image
    # --------------------------------

    benchmark_ids = {
        x["id"]
        for x in all_samples
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
            "No separate warmup "
            "image found."
        )

    print(
        "Full-pipeline warmup runs:",
        args.warmup
    )

    print(
        "Warmup image:",
        warmup_sample[
            "input"
        ]
    )

    with torch.inference_mode():

        for _ in range(
            args.warmup
        ):

            _ = run_one(
                model=model,
                instruction=instruction,
                image_path=(
                    warmup_sample[
                        "input"
                    ]
                ),
                max_new_tokens=(
                    args.max_new_tokens
                )
            )

    torch.cuda.reset_peak_memory_stats()

    # --------------------------------
    # Evaluation
    # --------------------------------

    predictions = []

    chat_latencies = []
    e2e_latencies = []

    correct = 0
    valid_output_count = 0

    class_total = {
        name: 0
        for name in TARGET_NAMES
    }

    class_correct = {
        name: 0
        for name in TARGET_NAMES
    }

    normalized_targets = {
        x.lower()
        for x in TARGET_NAMES
    }

    total_chat_time_s = 0.0
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

            result = run_one(
                model=model,
                instruction=instruction,
                image_path=(
                    sample[
                        "input"
                    ]
                ),
                max_new_tokens=(
                    args.max_new_tokens
                )
            )

            gt_label = (
                sample[
                    "label"
                ]
            )

            gt_norm = (
                normalize_label(
                    gt_label
                )
            )

            pred = (
                result[
                    "normalized_response"
                ]
            )

            valid_output = (
                pred
                in normalized_targets
            )

            if valid_output:
                valid_output_count += 1

            is_correct = (
                pred
                == gt_norm
            )

            class_total[
                gt_label
            ] += 1

            if is_correct:

                correct += 1

                class_correct[
                    gt_label
                ] += 1

            chat_ms = (
                result[
                    "chat_latency_ms"
                ]
            )

            e2e_ms = (
                result[
                    "e2e_latency_ms"
                ]
            )

            chat_latencies.append(
                chat_ms
            )

            e2e_latencies.append(
                e2e_ms
            )

            total_chat_time_s += (
                chat_ms
                / 1000.0
            )

            total_e2e_time_s += (
                e2e_ms
                / 1000.0
            )

            record = {
                "id":
                    sample["id"],

                "task":
                    "image_classification",

                "dataset":
                    "imagenette",

                "split":
                    "validation",

                "input":
                    sample["input"],

                "label_id":
                    int(
                        sample[
                            "label_id"
                        ]
                    ),

                "label":
                    gt_label,

                "raw_response":
                    result[
                        "raw_response"
                    ],

                "normalized_response":
                    pred,

                "valid_closed_set_output":
                    valid_output,

                "correct":
                    is_correct,

                "image_width":
                    result[
                        "image_width"
                    ],

                "image_height":
                    result[
                        "image_height"
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
            }

            predictions.append(
                record
            )

            print(
                f"[{i:03d}/"
                f"{len(samples):03d}] "
                f"GT={gt_label:<18} "
                f"Pred="
                f"{pred[:22]:<22} "
                f"{'OK' if is_correct else 'WRONG'} "
                f"chat="
                f"{chat_ms:8.2f}ms "
                f"e2e="
                f"{e2e_ms:8.2f}ms"
            )

    # --------------------------------
    # Summary
    # --------------------------------

    accuracy = (
        correct
        / len(samples)
    )

    valid_output_rate = (
        valid_output_count
        / len(samples)
    )

    per_class_accuracy = {}

    for name in TARGET_NAMES:

        total = (
            class_total[
                name
            ]
        )

        cls_correct = (
            class_correct[
                name
            ]
        )

        per_class_accuracy[
            name
        ] = {
            "correct":
                cls_correct,

            "total":
                total,

            "accuracy":
                (
                    round(
                        cls_correct
                        / total,
                        6
                    )
                    if total > 0
                    else None
                ),
        }

    chat_stats = stats(
        chat_latencies
    )

    e2e_stats = stats(
        e2e_latencies
    )

    peak_allocated_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    summary = {
        "model":
            "MiniCPM-o-4.5",

        "task":
            "image_classification",

        "dataset":
            "Imagenette",

        "split":
            "validation",

        "evaluation_mode":
            "10-class closed-set",

        "model_role":
            "generalist",

        "precision":
            actual_dtype,

        "attention_backend":
            actual_attention,

        "batch_size":
            1,

        "modalities_initialized": {
            "vision":
                True,

            "audio":
                True,

            "tts":
                False,
        },

        "decoding": {
            "do_sample":
                False,

            "temperature":
                None,

            "top_p":
                None,

            "top_k":
                None,

            "enable_thinking":
                False,

            "use_tts_template":
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

        "message_format":
            "[PIL RGB image, prompt text]",

        "output_normalization":
            "strip + lowercase only",

        "target_classes":
            TARGET_NAMES,

        "num_samples":
            len(samples),

        "correct":
            correct,

        "accuracy":
            round(
                accuracy,
                6
            ),

        "valid_closed_set_outputs":
            valid_output_count,

        "valid_closed_set_output_rate":
            round(
                valid_output_rate,
                6
            ),

        "per_class_accuracy":
            per_class_accuracy,

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

        "chat_latency":
            chat_stats,

        "e2e_latency":
            e2e_stats,

        "chat_request_throughput_qps":
            round(
                len(samples)
                / total_chat_time_s,
                3
            ),

        "e2e_request_throughput_qps":
            round(
                len(samples)
                / total_e2e_time_s,
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
            "chat_latency":
                (
                    "MiniCPM model.chat() on "
                    "a pre-decoded PIL RGB image; "
                    "includes MiniCPM internal "
                    "vision preprocessing, vision "
                    "encoding, language generation "
                    "and response decoding"
                ),

            "e2e_latency":
                (
                    "local image read/decode + "
                    "RGB conversion + message "
                    "construction + model.chat() + "
                    "closed-label normalization"
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
        "Valid closed-set output rate:",
        f"{valid_output_rate:.4f}"
    )

    print(
        "Chat latency:",
        chat_stats
    )

    print(
        "E2E latency:",
        e2e_stats
    )

    print(
        "Chat throughput:",
        summary[
            "chat_request_throughput_qps"
        ],
        "qps"
    )

    print(
        "E2E throughput:",
        summary[
            "e2e_request_throughput_qps"
        ],
        "qps"
    )

    print()
    print(
        "Per-class accuracy:"
    )

    for name in TARGET_NAMES:

        x = (
            per_class_accuracy[
                name
            ]
        )

        print(
            f"  {name:<18} "
            f"{x['correct']}/"
            f"{x['total']} "
            f"= {x['accuracy']}"
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
