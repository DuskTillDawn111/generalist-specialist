import argparse
import json
import time
from pathlib import Path

import numpy as np
import paddle
from paddleocr import TextRecognition


ROOT = Path("/root/autodl-tmp/generalist-specialist")

MODEL_DIR = (
    ROOT
    / "models/paddleocr/official_models/PP-OCRv6_medium_rec"
)

MANIFEST = ROOT / "benchmark/manifests/ocr.jsonl"

RESULT_DIR = ROOT / "benchmark/results"
RESULT_FILE = RESULT_DIR / "paddleocr_rec_profile.json"


def directory_size_mb(path):
    total = 0

    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size

    return total / (1024 ** 2)


def percentile(values, q):
    return float(np.percentile(values, q))


def sync():
    paddle.device.cuda.synchronize()


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

    if not paddle.device.is_compiled_with_cuda():
        raise RuntimeError(
            "PaddlePaddle CUDA support is not available."
        )

    paddle.set_device("gpu:0")

    print("Device:", paddle.device.get_device())

    print(
        "GPU:",
        paddle.device.cuda.get_device_name()
    )

    # -----------------------------
    # 读取第一张真实 IIIT5K benchmark
    # -----------------------------

    with open(
        MANIFEST,
        "r",
        encoding="utf-8"
    ) as f:

        sample = json.loads(
            f.readline()
        )

    image_path = sample["input"]
    ground_truth = sample["label"]

    print("Input:", image_path)
    print("Ground truth:", ground_truth)

    # -----------------------------
    # 模型加载
    # -----------------------------

    sync()

    start = time.perf_counter()

    model = TextRecognition(
        model_name="PP-OCRv6_medium_rec",
        model_dir=str(MODEL_DIR),
        device="gpu:0"
    )

    sync()

    load_time = (
        time.perf_counter()
        - start
    )

    idle_vram_mb = (
        paddle.device.cuda.memory_allocated()
        / 1024 ** 2
    )

    # -----------------------------
    # Warmup
    # -----------------------------

    print(
        f"Warmup: {args.warmup}"
    )

    for _ in range(args.warmup):

        _ = model.predict(
            input=image_path,
            batch_size=1
        )

        sync()

    # -----------------------------
    # 重置峰值统计
    # -----------------------------

    paddle.device.cuda.reset_max_memory_allocated()

    # -----------------------------
    # 正式计时
    # -----------------------------

    latencies = []

    last_output = None

    for _ in range(args.runs):

        sync()

        start = time.perf_counter()

        last_output = model.predict(
            input=image_path,
            batch_size=1
        )

        sync()

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        latencies.append(
            elapsed_ms
        )

    peak_vram_mb = (
        paddle.device.cuda.max_memory_allocated()
        / 1024 ** 2
    )

    mean_latency_ms = float(
        np.mean(latencies)
    )

    result = {
        "model":
            "PP-OCRv6-medium-rec",

        "task":
            "optical_character_recognition",

        "benchmark_mode":
            "recognition_only",

        "parameters_m":
            None,

        "weight_size_mb":
            round(
                directory_size_mb(MODEL_DIR),
                3
            ),

        "load_time_s":
            round(load_time, 3),

        "idle_vram_mb":
            round(idle_vram_mb, 3),

        "peak_vram_mb":
            round(peak_vram_mb, 3),

        "mean_latency_ms":
            round(mean_latency_ms, 3),

        "p50_latency_ms":
            round(
                percentile(latencies, 50),
                3
            ),

        "p95_latency_ms":
            round(
                percentile(latencies, 95),
                3
            ),

        "p99_latency_ms":
            round(
                percentile(latencies, 99),
                3
            ),

        "throughput_qps":
            round(
                1000.0
                / mean_latency_ms,
                3
            ),

        "warmup_runs":
            args.warmup,

        "test_runs":
            args.runs,

        "input_dataset":
            "IIIT5K",

        "sample_id":
            sample["id"],

        "ground_truth":
            ground_truth,

        "device":
            paddle.device.get_device(),
    }

    if last_output:

        first = last_output[0]

        try:
            result["prediction"] = str(
                first["rec_text"]
            )

            result["rec_score"] = float(
                first["rec_score"]
            )

        except Exception:
            pass

    print()
    print("========== RESULT ==========")

    for key, value in result.items():
        print(f"{key}: {value}")

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
