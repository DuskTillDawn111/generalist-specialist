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
from transformers import AutoModel


ROOT = Path("/root/autodl-tmp/generalist-specialist")

MODEL_DIR = ROOT / "models/minicpm-o-4_5"

MANIFEST = ROOT / "benchmark/manifests/ocr.jsonl"
FULL_MANIFEST = ROOT / "benchmark/manifests/ocr_full.jsonl"

PROMPT_FILE = ROOT / "benchmark/prompts/task_prompts.json"

RESULT_DIR = ROOT / "benchmark/results/formal"

FORMAL_PREDICTIONS = RESULT_DIR / "minicpm_iiit5k_predictions.jsonl"
FORMAL_SUMMARY = RESULT_DIR / "minicpm_iiit5k_summary.json"


def read_manifest(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def load_prompt():
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


def levenshtein_chars(reference, prediction):
    n = len(reference)
    m = len(prediction)

    prev = list(range(m + 1))

    for i in range(1, n + 1):
        curr = [i] + [0] * m

        for j in range(1, m + 1):
            cost = (
                0
                if reference[i - 1] == prediction[j - 1]
                else 1
            )

            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )

        prev = curr

    return prev[m]


def sync():
    torch.cuda.synchronize()


def latency_stats(values):
    arr = np.asarray(values, dtype=np.float64)

    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def checkpoint_size_mb(path):
    total = 0

    for p in path.glob("*.safetensors"):
        total += p.stat().st_size

    return total / (1024 ** 2)


def run_one(
    model,
    instruction,
    image_path,
    max_new_tokens,
):
    sync()

    e2e_start = time.perf_counter()

    with Image.open(image_path) as img:
        width = img.width
        height = img.height

        image = img.convert("RGB")

    msgs = [
        {
            "role": "user",
            "content": [
                image,
                instruction,
            ],
        }
    ]

    sync()

    chat_start = time.perf_counter()

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

    chat_end = time.perf_counter()

    if not isinstance(response, str):
        raise TypeError(
            "Expected model.chat() to return str, "
            f"got {type(response)}"
        )

    normalized = normalize_ocr(response)

    e2e_end = time.perf_counter()

    return {
        "raw_response": response,
        "normalized_response": normalized,

        "image_width": width,
        "image_height": height,

        "chat_latency_ms":
            (chat_end - chat_start) * 1000,

        "e2e_latency_ms":
            (e2e_end - e2e_start) * 1000,
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
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda:0")

    torch.cuda.set_device(device)

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print("Transformers:", transformers.__version__)

    print("Precision: bfloat16")
    print("Attention backend: SDPA")

    print("init_vision=True")
    print("init_audio=True")
    print("init_tts=False")

    print("use_tts_template=False")
    print("enable_thinking=False")

    print("Decoding: do_sample=False")
    print("Max new tokens:", args.max_new_tokens)

    prompt_info = load_prompt()
    instruction = prompt_info["prompt"]

    print(
        "Prompt version:",
        prompt_info["version"]
    )

    all_samples = read_manifest(MANIFEST)

    samples = list(all_samples)

    if args.limit > 0:
        samples = samples[:args.limit]

    print(
        "Evaluation samples:",
        len(samples)
    )

    # -----------------------------
    # Model
    # -----------------------------

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Loading MiniCPM-o-4.5...")

    load_start = time.perf_counter()

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
        next(model.parameters()).dtype
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
        f"Parameters: {parameters_m:.3f} M"
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

    # -----------------------------
    # Separate warmup sample
    # -----------------------------

    benchmark_ids = {
        x["id"]
        for x in all_samples
    }

    warmup_sample = None

    for candidate in read_manifest(
        FULL_MANIFEST
    ):
        if candidate["id"] not in benchmark_ids:
            warmup_sample = candidate
            break

    if warmup_sample is None:
        raise RuntimeError(
            "No separate warmup image found."
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
        for _ in range(args.warmup):
            _ = run_one(
                model=model,
                instruction=instruction,
                image_path=warmup_sample["input"],
                max_new_tokens=args.max_new_tokens,
            )

    torch.cuda.reset_peak_memory_stats()

    # -----------------------------
    # Evaluation
    # -----------------------------

    predictions = []

    chat_latencies = []
    e2e_latencies = []

    correct = 0

    total_character_errors = 0
    total_reference_characters = 0

    total_chat_time_s = 0.0
    total_e2e_time_s = 0.0

    raw_lengths = []
    normalized_lengths = []

    print()
    print("Running evaluation...")

    with torch.inference_mode():

        for i, sample in enumerate(
            samples,
            start=1
        ):
            result = run_one(
                model=model,
                instruction=instruction,
                image_path=sample["input"],
                max_new_tokens=args.max_new_tokens,
            )

            reference_raw = sample["label"]

            reference = normalize_ocr(
                reference_raw
            )

            prediction = result[
                "normalized_response"
            ]

            char_errors = levenshtein_chars(
                reference,
                prediction
            )

            reference_chars = len(reference)

            sample_cer = (
                char_errors / reference_chars
                if reference_chars > 0
                else 0.0
            )

            is_correct = (
                prediction == reference
            )

            if is_correct:
                correct += 1

            total_character_errors += (
                char_errors
            )

            total_reference_characters += (
                reference_chars
            )

            chat_ms = result[
                "chat_latency_ms"
            ]

            e2e_ms = result[
                "e2e_latency_ms"
            ]

            chat_latencies.append(
                chat_ms
            )

            e2e_latencies.append(
                e2e_ms
            )

            total_chat_time_s += (
                chat_ms / 1000.0
            )

            total_e2e_time_s += (
                e2e_ms / 1000.0
            )

            raw_lengths.append(
                len(
                    result["raw_response"]
                )
            )

            normalized_lengths.append(
                len(prediction)
            )

            record = {
                "id": sample["id"],

                "task":
                    "optical_character_recognition",

                "dataset":
                    "IIIT5K",

                "split":
                    "test",

                "input":
                    sample["input"],

                "reference":
                    reference_raw,

                "normalized_reference":
                    reference,

                "raw_response":
                    result["raw_response"],

                "normalized_prediction":
                    prediction,

                "correct":
                    is_correct,

                "character_errors":
                    char_errors,

                "reference_characters":
                    reference_chars,

                "cer":
                    round(
                        sample_cer,
                        6
                    ),

                "image_width":
                    result["image_width"],

                "image_height":
                    result["image_height"],

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

            predictions.append(record)

            print(
                f"[{i:03d}/{len(samples):03d}] "
                f"GT={reference:<18} "
                f"Pred={prediction[:22]:<22} "
                f"{'OK' if is_correct else 'WRONG'} "
                f"err={char_errors:<2} "
                f"chat={chat_ms:8.2f}ms "
                f"e2e={e2e_ms:8.2f}ms"
            )

    # -----------------------------
    # Summary
    # -----------------------------

    exact_match_accuracy = (
        correct / len(samples)
    )

    cer = (
        total_character_errors
        / total_reference_characters
        if total_reference_characters > 0
        else 0.0
    )

    chat_stats = latency_stats(
        chat_latencies
    )

    e2e_stats = latency_stats(
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

        "prompt_version":
            prompt_info["version"],

        "prompt":
            instruction,

        "message_format":
            "[PIL RGB image, OCR prompt text]",

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
                exact_match_accuracy,
                6
            ),

        "total_character_errors":
            total_character_errors,

        "total_reference_characters":
            total_reference_characters,

        "cer":
            round(
                cer,
                6
            ),

        "raw_response_length_chars": {
            "mean":
                round(
                    float(
                        np.mean(raw_lengths)
                    ),
                    3
                ),

            "min":
                int(min(raw_lengths)),

            "max":
                int(max(raw_lengths)),
        },

        "normalized_response_length_chars": {
            "mean":
                round(
                    float(
                        np.mean(
                            normalized_lengths
                        )
                    ),
                    3
                ),

            "min":
                int(
                    min(
                        normalized_lengths
                    )
                ),

            "max":
                int(
                    max(
                        normalized_lengths
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
                    "MiniCPM model.chat() on a "
                    "pre-decoded PIL RGB image; "
                    "includes internal visual "
                    "preprocessing, visual encoding, "
                    "language generation and "
                    "response decoding"
                ),

            "e2e_latency":
                (
                    "local image read/decode + "
                    "RGB conversion + message "
                    "construction + model.chat() + "
                    "OCR normalization"
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

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.limit > 0:
        prediction_path = (
            RESULT_DIR
            / "minicpm_iiit5k_smoke_predictions.jsonl"
        )

        summary_path = (
            RESULT_DIR
            / "minicpm_iiit5k_smoke_summary.json"
        )

    else:
        prediction_path = FORMAL_PREDICTIONS
        summary_path = FORMAL_SUMMARY

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
    print("========== SUMMARY ==========")

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
        f"{exact_match_accuracy:.4f}"
    )

    print(
        "CER:",
        f"{cer:.4f}"
    )

    print(
        "Character errors:",
        total_character_errors,
        "/",
        total_reference_characters
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
        "Saved predictions:",
        prediction_path
    )

    print(
        "Saved summary:",
        summary_path
    )


if __name__ == "__main__":
    main()
