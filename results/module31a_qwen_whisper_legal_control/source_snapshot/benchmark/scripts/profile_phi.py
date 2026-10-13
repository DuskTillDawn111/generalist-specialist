import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)


ROOT = Path("/root/autodl-tmp/generalist-specialist")
MODEL_DIR = ROOT / "models/phi4-multimodal"

RESULT_DIR = ROOT / "benchmark/results"
RESULT_FILE = RESULT_DIR / "phi4_multimodal_profile.json"


def safetensors_size_mb(path):
    total = 0

    for p in path.glob("*.safetensors"):
        total += p.stat().st_size

    return total / (1024 ** 2)


def percentile(values, q):
    return float(np.percentile(values, q))


def sync():
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--warmup",
        type=int,
        default=1
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16
    )

    parser.add_argument(
        "--attn",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"]
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = "cuda:0"

    torch.cuda.set_device(0)

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Total VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GiB"
    )

    print("Attention backend:", args.attn)
    print("Precision: bfloat16")

    # -----------------------------
    # Processor
    # -----------------------------

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        local_files_only=True
    )

    # -----------------------------
    # Model loading
    # -----------------------------

    print("Loading model...")

    sync()
    load_start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation=args.attn,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()

    sync()

    load_time_s = (
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

    idle_vram_mb = (
        torch.cuda.memory_allocated(0)
        / 1024**2
    )

    print(
        "Model loaded."
    )

    print(
        "Parameters:",
        round(parameters_m, 3),
        "M"
    )

    print(
        "Idle VRAM:",
        round(idle_vram_mb, 3),
        "MB"
    )

    # -----------------------------
    # Generation config
    # -----------------------------

    generation_config = (
        GenerationConfig.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True
        )
    )

    # -----------------------------
    # Fixed text-only prompt
    # -----------------------------

    user_prompt = "<|user|>"
    assistant_prompt = "<|assistant|>"
    prompt_suffix = "<|end|>"

    question = (
        "Classify the sentiment of the following sentence "
        "as exactly positive or negative. "
        "Output only the label. "
        "Sentence: The movie was wonderful and enjoyable."
    )

    prompt = (
        f"{user_prompt}"
        f"{question}"
        f"{prompt_suffix}"
        f"{assistant_prompt}"
    )

    print("\nPrompt:")
    print(prompt)

    inputs = processor(
        prompt,
        images=None,
        return_tensors="pt"
    ).to(device)

    input_tokens = int(
        inputs["input_ids"].shape[1]
    )

    print(
        "Input tokens:",
        input_tokens
    )

    # -----------------------------
    # Warmup
    # -----------------------------

    print(
        f"\nWarmup: {args.warmup}"
    )

    with torch.inference_mode():

        for _ in range(args.warmup):

            _ = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                generation_config=generation_config,
                do_sample=False,
            )

            sync()

    # -----------------------------
    # Peak memory reset
    # -----------------------------

    torch.cuda.reset_peak_memory_stats(0)

    # -----------------------------
    # Timed inference
    # -----------------------------

    latencies = []

    final_generate_ids = None

    with torch.inference_mode():

        for i in range(args.runs):

            sync()

            start = time.perf_counter()

            generate_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                generation_config=generation_config,
                do_sample=False,
            )

            sync()

            elapsed_ms = (
                time.perf_counter()
                - start
            ) * 1000

            latencies.append(
                elapsed_ms
            )

            final_generate_ids = (
                generate_ids
            )

            print(
                f"Run {i + 1}/{args.runs}: "
                f"{elapsed_ms:.3f} ms"
            )

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(0)
        / 1024**2
    )

    # -----------------------------
    # Decode final output
    # -----------------------------

    generated_only = (
        final_generate_ids[
            :,
            inputs["input_ids"].shape[1]:
        ]
    )

    response = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    output_tokens = int(
        generated_only.shape[1]
    )

    # -----------------------------
    # Statistics
    # -----------------------------

    mean_latency_ms = float(
        np.mean(latencies)
    )

    result = {
        "model":
            "Phi-4-multimodal-instruct",

        "task":
            "text_generation",

        "benchmark_mode":
            "text_only_microbenchmark",

        "precision":
            "bfloat16",

        "attention_backend":
            args.attn,

        "parameters_m":
            round(parameters_m, 3),

        "weight_size_mb":
            round(
                safetensors_size_mb(
                    MODEL_DIR
                ),
                3
            ),

        "load_time_s":
            round(load_time_s, 3),

        "idle_vram_mb":
            round(idle_vram_mb, 3),

        "peak_vram_mb":
            round(peak_vram_mb, 3),

        "mean_latency_ms":
            round(mean_latency_ms, 3),

        "p50_latency_ms":
            round(
                percentile(
                    latencies,
                    50
                ),
                3
            ),

        "p95_latency_ms":
            round(
                percentile(
                    latencies,
                    95
                ),
                3
            ),

        "p99_latency_ms":
            round(
                percentile(
                    latencies,
                    99
                ),
                3
            ),

        "throughput_qps":
            round(
                1000.0
                / mean_latency_ms,
                3
            ),

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "max_new_tokens":
            args.max_new_tokens,

        "warmup_runs":
            args.warmup,

        "test_runs":
            args.runs,

        "device":
            device,

        "gpu":
            torch.cuda.get_device_name(0),

        "torch_version":
            torch.__version__,

        "response":
            response,
    }

    print()
    print(
        "========== RESULT =========="
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "Saved to:",
        RESULT_FILE
    )


if __name__ == "__main__":
    main()
