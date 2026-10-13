import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch


ROOT = Path("/root/autodl-tmp/generalist-specialist")
MODEL_DIR = ROOT / "models/qwen2.5-omni-7b"


def checkpoint_size_mb(model_dir):
    files = list(Path(model_dir).glob("*.safetensors"))
    return sum(f.stat().st_size for f in files) / 1024**2


def checkpoint_parameter_count(model_dir):
    """
    只读取 safetensors 元数据，不把整个模型加载进内存。
    这里统计的是完整 checkpoint 参数，包括音频输出模块。
    """
    from safetensors import safe_open

    total = 0

    for file in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(
            str(file),
            framework="pt",
            device="cpu"
        ) as f:
            for key in f.keys():
                shape = f.get_slice(key).get_shape()
                total += math.prod(shape)

    return total


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0

    k = (len(values) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def smoke_test():
    from transformers import (
        Qwen2_5OmniConfig,
        Qwen2_5OmniProcessor
    )

    print("No GPU detected -> Qwen smoke test only")

    t0 = time.perf_counter()

    config = Qwen2_5OmniConfig.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    load_time = time.perf_counter() - t0

    checkpoint_params = checkpoint_parameter_count(
        MODEL_DIR
    )

    result = {
        "model": "Qwen2.5-Omni-7B",
        "model_type": config.model_type,
        "task": "multimodal_generalist",
        "parameters_m": None,
        "checkpoint_parameters_m":
            checkpoint_params / 1e6,
        "weight_size_mb":
            checkpoint_size_mb(MODEL_DIR),
        "load_time_s": load_time,
        "idle_vram_mb": None,
        "peak_vram_mb": None,
        "mean_latency_ms": None,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "p99_latency_ms": None,
        "throughput_qps": None,
        "warmup_runs": 0,
        "test_runs": 0,
        "device": "cpu",
        "processor_loaded": True,
        "smoke_test_only": True,
        "note":
            "No-GPU mode: config/processor/checkpoint metadata only"
    }

    return result


def gpu_profile(warmup, runs):
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor
    )

    print("Device: cuda:0")
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

    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()

    model = (
        Qwen2_5OmniForConditionalGeneration
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            enable_audio_output=False
        )
    )

    model.eval()

    sync()

    load_time = time.perf_counter() - t0

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    idle_vram = (
        torch.cuda.memory_allocated()
        / 1024**2
    )

    torch.cuda.reset_peak_memory_stats()

    # 先使用固定文本作为 microbenchmark。
    # 后面正式实验会分别加入 text / image / audio / OCR workload。
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":
                        "Classify the sentiment of this sentence "
                        "as positive or negative: "
                        "This movie is surprisingly good."
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        return_tensors="pt",
        padding=True
    )

    inputs = inputs.to("cuda")

    generation_kwargs = {
        "max_new_tokens": 16,
        "do_sample": False,
        "return_audio": False
    }

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.generate(
                **inputs,
                **generation_kwargs
            )

    sync()

    latencies = []

    with torch.inference_mode():
        for _ in range(runs):

            sync()

            start = time.perf_counter()

            _ = model.generate(
                **inputs,
                **generation_kwargs
            )

            sync()

            end = time.perf_counter()

            latencies.append(
                (end - start) * 1000
            )

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    mean_latency = statistics.mean(latencies)

    checkpoint_params = checkpoint_parameter_count(
        MODEL_DIR
    )

    return {
        "model": "Qwen2.5-Omni-7B",
        "task": "multimodal_generalist",
        "benchmark_task": "text_generation",
        "parameters_m": parameters / 1e6,
        "checkpoint_parameters_m":
            checkpoint_params / 1e6,
        "weight_size_mb":
            checkpoint_size_mb(MODEL_DIR),
        "dtype": "bfloat16",
        "audio_output_enabled": False,
        "max_new_tokens": 16,
        "load_time_s": load_time,
        "idle_vram_mb": idle_vram,
        "peak_vram_mb": peak_vram,
        "mean_latency_ms": mean_latency,
        "p50_latency_ms":
            percentile(latencies, 50),
        "p95_latency_ms":
            percentile(latencies, 95),
        "p99_latency_ms":
            percentile(latencies, 99),
        "throughput_qps":
            1000 / mean_latency,
        "warmup_runs": warmup,
        "test_runs": runs,
        "device": "cuda:0",
        "gpu":
            torch.cuda.get_device_name(0)
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--warmup",
        type=int,
        default=3
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=10
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        result = gpu_profile(
            args.warmup,
            args.runs
        )
    else:
        result = smoke_test()

    print(
        "\n========== RESULT =========="
    )

    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

    output_dir = (
        ROOT / "benchmark/results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "qwen_profile.json"
    )

    with open(
        output_path,
        "w"
    ) as f:
        json.dump(
            result,
            f,
            indent=2
        )

    print(
        "\nSaved to:",
        output_path
    )


if __name__ == "__main__":
    main()
