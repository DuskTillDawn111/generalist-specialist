import argparse
import json
import statistics
import time
from pathlib import Path

import torch


ROOT = Path("/root/autodl-tmp/generalist-specialist")


def file_size_mb(path):
    return Path(path).stat().st_size / 1024 / 1024


def safetensors_size_mb(model_dir):
    model_dir = Path(model_dir)
    files = list(model_dir.glob("*.safetensors"))
    return sum(f.stat().st_size for f in files) / 1024 / 1024


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


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_gpu_memory(device):
    if device.type != "cuda":
        return None

    return torch.cuda.memory_allocated() / 1024**2


def get_peak_gpu_memory(device):
    if device.type != "cuda":
        return None

    return torch.cuda.max_memory_allocated() / 1024**2


# =========================================================
# ResNet50
# =========================================================

def profile_resnet(device, warmup, runs):

    from torchvision.models import resnet50

    model_dir = ROOT / "models/resnet50"
    weight_path = model_dir / "resnet50-11ad3fa6.pth"

    t0 = time.perf_counter()

    model = resnet50(weights=None)

    state = torch.load(
        weight_path,
        map_location="cpu"
    )

    model.load_state_dict(state)
    model.eval()
    model.to(device)

    sync(device)

    load_time = time.perf_counter() - t0

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    # 固定输入
    x = torch.randn(
        1,
        3,
        224,
        224,
        device=device
    )

    idle_vram = get_gpu_memory(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    with torch.inference_mode():

        for _ in range(warmup):
            _ = model(x)

    sync(device)

    latencies = []

    # Benchmark
    with torch.inference_mode():

        for _ in range(runs):

            sync(device)

            start = time.perf_counter()

            _ = model(x)

            sync(device)

            end = time.perf_counter()

            latencies.append(
                (end - start) * 1000
            )

    peak_vram = get_peak_gpu_memory(device)

    mean_latency = statistics.mean(latencies)

    return {

        "model": "ResNet50",

        "task": "image_classification",

        "parameters_m":
            params / 1e6,

        "weight_size_mb":
            file_size_mb(weight_path),

        "load_time_s":
            load_time,

        "idle_vram_mb":
            idle_vram,

        "peak_vram_mb":
            peak_vram,

        "mean_latency_ms":
            mean_latency,

        "p50_latency_ms":
            percentile(latencies, 50),

        "p95_latency_ms":
            percentile(latencies, 95),

        "p99_latency_ms":
            percentile(latencies, 99),

        "throughput_qps":
            1000 / mean_latency,

        "warmup_runs":
            warmup,

        "test_runs":
            runs
    }


# =========================================================
# DistilBERT
# =========================================================

def profile_distilbert(device, warmup, runs):

    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification
    )

    model_dir = (
        ROOT
        / "models/distilbert-sst2"
    )

    weight_path = (
        model_dir
        / "model.safetensors"
    )

    t0 = time.perf_counter()

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_dir,
            local_files_only=True
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_dir,
            local_files_only=True
        )
    )

    model.eval()
    model.to(device)

    sync(device)

    load_time = (
        time.perf_counter()
        - t0
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    text = (
        "This movie is surprisingly good "
        "and I really enjoyed watching it."
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    encoded = {
        k: v.to(device)
        for k, v in encoded.items()
    }

    idle_vram = get_gpu_memory(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    with torch.inference_mode():

        for _ in range(warmup):
            _ = model(**encoded)

    sync(device)

    latencies = []

    # Benchmark
    with torch.inference_mode():

        for _ in range(runs):

            sync(device)

            start = time.perf_counter()

            _ = model(**encoded)

            sync(device)

            end = time.perf_counter()

            latencies.append(
                (end - start) * 1000
            )

    peak_vram = get_peak_gpu_memory(device)

    mean_latency = statistics.mean(
        latencies
    )

    return {

        "model":
            "DistilBERT-SST2",

        "task":
            "text_classification",

        "parameters_m":
            params / 1e6,

        "weight_size_mb":
            file_size_mb(
                weight_path
            ),

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
            1000
            / mean_latency,

        "warmup_runs":
            warmup,

        "test_runs":
            runs
    }


# =========================================================
# Whisper-small
# =========================================================

def profile_whisper(device, warmup, runs):

    import numpy as np

    from transformers import (
        AutoProcessor,
        AutoModelForSpeechSeq2Seq
    )

    model_dir = (
        ROOT
        / "models/whisper-small"
    )

    t0 = time.perf_counter()

    processor = (
        AutoProcessor
        .from_pretrained(
            model_dir,
            local_files_only=True
        )
    )

    model = (
        AutoModelForSpeechSeq2Seq
        .from_pretrained(
            model_dir,
            local_files_only=True
        )
    )

    model.eval()
    model.to(device)

    sync(device)

    load_time = (
        time.perf_counter()
        - t0
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    # 5 秒静音音频
    # 这里只用于 Profiler smoke test，
    # 正式 accuracy / E2E 实验会换真实数据集
    sample_rate = 16000

    audio = np.zeros(
        sample_rate * 5,
        dtype=np.float32
    )

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    input_features = (
        inputs.input_features.to(device)
    )

    idle_vram = get_gpu_memory(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    generation_kwargs = {
        "max_new_tokens": 32
    }

    # Warmup
    with torch.inference_mode():

        for _ in range(warmup):

            _ = model.generate(
                input_features,
                **generation_kwargs
            )

    sync(device)

    latencies = []

    # Benchmark
    with torch.inference_mode():

        for _ in range(runs):

            sync(device)

            start = time.perf_counter()

            _ = model.generate(
                input_features,
                **generation_kwargs
            )

            sync(device)

            end = time.perf_counter()

            latencies.append(
                (end - start) * 1000
            )

    peak_vram = (
        get_peak_gpu_memory(device)
    )

    mean_latency = (
        statistics.mean(latencies)
    )

    return {

        "model":
            "Whisper-small",

        "task":
            "automatic_speech_recognition",

        "parameters_m":
            params / 1e6,

        "weight_size_mb":
            safetensors_size_mb(
                model_dir
            ),

        "audio_duration_s":
            5,

        "max_new_tokens":
            32,

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
            1000
            / mean_latency,

        "realtime_factor":
            (
                mean_latency
                / 1000
                / 5
            ),

        "warmup_runs":
            warmup,

        "test_runs":
            runs
    }


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "resnet50",
            "distilbert",
            "whisper"
        ]
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=100
    )

    args = parser.parse_args()

    if torch.cuda.is_available():

        device = torch.device(
            "cuda:0"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

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

    if args.model == "resnet50":

        result = profile_resnet(
            device,
            args.warmup,
            args.runs
        )

    elif args.model == "distilbert":

        result = profile_distilbert(
            device,
            args.warmup,
            args.runs
        )

    else:

        result = profile_whisper(
            device,
            args.warmup,
            args.runs
        )

    result["device"] = str(
        device
    )

    if device.type == "cuda":

        result["gpu"] = (
            torch.cuda
            .get_device_name(0)
        )

    print(
        "\n========== RESULT =========="
    )

    for k, v in result.items():

        if isinstance(v, float):

            print(
                f"{k}: {v:.3f}"
            )

        else:

            print(
                f"{k}: {v}"
            )

    output_dir = (
        ROOT
        / "benchmark/results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / f"{args.model}_profile.json"
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