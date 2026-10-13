import argparse
import json
import statistics
import time
from pathlib import Path

import torch


ROOT = Path("/root/autodl-tmp/generalist-specialist")
MODEL_DIR = ROOT / "models/minicpm-o-4_5"


def checkpoint_size_mb(model_dir):
    files = list(
        Path(model_dir).glob("*.safetensors")
    )

    return sum(
        f.stat().st_size
        for f in files
    ) / 1024**2


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0

    k = (len(values) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    return (
        values[f]
        + (values[c] - values[f])
        * (k - f)
    )


def sync():
    torch.cuda.synchronize()


def profile_minicpm(warmup, runs):

    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError(
            "MiniCPM profiler requires GPU. "
            "Please run after GPU startup."
        )

    print("Device: cuda:0")
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "Total VRAM:",
        round(
            torch.cuda
            .get_device_properties(0)
            .total_memory
            / 1024**3,
            2
        ),
        "GiB"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ========================================
    # Model loading
    # ========================================

    t0 = time.perf_counter()

    model = AutoModel.from_pretrained(
        MODEL_DIR,

        trust_remote_code=True,
        local_files_only=True,

        # 与另外两个 Generalist 保持一致
        torch_dtype=torch.bfloat16,

        # baseline 统一 SDPA
        attn_implementation="sdpa",

        # 保留我们实验需要的视觉 + 音频输入
        init_vision=True,
        init_audio=True,

        # 不测试语音输出
        init_tts=False,

        low_cpu_mem_usage=True,
    )

    model.eval().cuda()

    sync()

    load_time = (
        time.perf_counter() - t0
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    idle_vram = (
        torch.cuda.memory_allocated()
        / 1024**2
    )

    torch.cuda.reset_peak_memory_stats()

    # ========================================
    # Text-only microbenchmark
    #
    # 后续正式任务实验还会分别加入：
    # image / OCR / audio / text
    # ========================================

    msgs = [
        {
            "role": "user",
            "content": [
                (
                    "Classify the sentiment of the "
                    "following sentence as positive "
                    "or negative. Answer with only "
                    "one word: "
                    "This movie is surprisingly good."
                )
            ],
        }
    ]

    generation_kwargs = {
        "msgs": msgs,

        # 不启用 TTS 模板
        "use_tts_template": False,

        # 不使用 thinking，避免额外 token
        "enable_thinking": False,

        # 固定 deterministic generation
        "do_sample": False,

        "max_new_tokens": 16,
    }

    # ========================================
    # Warmup
    # ========================================

    with torch.inference_mode():

        for _ in range(warmup):

            _ = model.chat(
                **generation_kwargs
            )

    sync()

    # ========================================
    # Benchmark
    # ========================================

    latencies = []

    with torch.inference_mode():

        for _ in range(runs):

            sync()

            start = time.perf_counter()

            _ = model.chat(
                **generation_kwargs
            )

            sync()

            end = time.perf_counter()

            latencies.append(
                (end - start) * 1000
            )

    mean_latency = statistics.mean(
        latencies
    )

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    return {

        "model":
            "MiniCPM-o-4.5",

        "task":
            "multimodal_generalist",

        "benchmark_task":
            "text_generation",

        "parameters_m":
            params / 1e6,

        "weight_size_mb":
            checkpoint_size_mb(
                MODEL_DIR
            ),

        "dtype":
            "bfloat16",

        "attention":
            "sdpa",

        "vision_enabled":
            True,

        "audio_input_enabled":
            True,

        "tts_enabled":
            False,

        "thinking_enabled":
            False,

        "max_new_tokens":
            16,

        "load_time_s":
            load_time,

        "idle_vram_mb":
            idle_vram,

        "peak_vram_mb":
            peak_vram,

        "mean_latency_ms":
            mean_latency,

        "p50_latency_ms":
            percentile(
                latencies,
                50
            ),

        "p95_latency_ms":
            percentile(
                latencies,
                95
            ),

        "p99_latency_ms":
            percentile(
                latencies,
                99
            ),

        "throughput_qps":
            1000 / mean_latency,

        "warmup_runs":
            warmup,

        "test_runs":
            runs,

        "device":
            "cuda:0",

        "gpu":
            torch.cuda.get_device_name(0),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    result = profile_minicpm(
        args.warmup,
        args.runs,
    )

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
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "minicpm_profile.json"
    )

    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        "\nSaved to:",
        output_path,
    )


if __name__ == "__main__":
    main()
