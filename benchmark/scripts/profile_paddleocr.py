import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import paddle
from paddleocr import PaddleOCR


ROOT = Path("/root/autodl-tmp/generalist-specialist")

MODEL_ROOT = ROOT / "models/paddleocr/official_models"
DET_DIR = MODEL_ROOT / "PP-OCRv6_medium_det"
REC_DIR = MODEL_ROOT / "PP-OCRv6_medium_rec"


def dir_size_mb(*paths):
    total = 0

    for path in paths:
        path = Path(path)

        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size

    return total / 1024 / 1024


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


def cuda_available():
    try:
        return (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
    except Exception:
        return False


def get_gpu_memory_mb():
    try:
        return paddle.device.cuda.memory_allocated() / 1024**2
    except Exception:
        return None


def get_peak_gpu_memory_mb():
    try:
        return paddle.device.cuda.max_memory_allocated() / 1024**2
    except Exception:
        return None


def profile_paddleocr(warmup, runs):

    has_cuda = cuda_available()

    device = "gpu:0" if has_cuda else "cpu"

    print("Device:", device)

    if has_cuda:
        try:
            print(
                "CUDA devices:",
                paddle.device.cuda.device_count()
            )
        except Exception:
            pass

    # 强制使用本地模型目录
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(
        ROOT / "models/paddleocr"
    )

    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    t0 = time.perf_counter()

    ocr = PaddleOCR(
        device=device,
        text_detection_model_dir=str(DET_DIR),
        text_recognition_model_dir=str(REC_DIR),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    load_time = time.perf_counter() - t0

    weight_size_mb = dir_size_mb(
        DET_DIR,
        REC_DIR
    )

    # 无卡阶段只验证初始化，不跑正式推理
    if not has_cuda:

        return {
            "model": "PP-OCRv6-medium",
            "task": "optical_character_recognition",
            "parameters_m": None,
            "weight_size_mb": weight_size_mb,
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
            "smoke_test_only": True,
            "note": "CPU/no-GPU mode: OCR inference benchmark skipped"
        }

    # ------------------------------------------------------
    # GPU 正式 microbenchmark 用固定合成图
    # 后续 accuracy / E2E 实验会换真实 OCR 数据集
    # ------------------------------------------------------

    import cv2

    image = np.full(
        (640, 1280, 3),
        255,
        dtype=np.uint8
    )

    cv2.putText(
        image,
        "OpenAI Generalist Specialist AI Model 2026",
        (60, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        "PaddleOCR benchmark sample 123456",
        (60, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    idle_vram = get_gpu_memory_mb()

    try:
        paddle.device.cuda.reset_max_memory_allocated()
    except Exception:
        pass

    # Warmup
    for _ in range(warmup):
        result = ocr.predict(image)

        # 强制真正执行 generator / lazy result
        list(result)

    latencies = []

    for _ in range(runs):

        start = time.perf_counter()

        result = ocr.predict(image)

        list(result)

        end = time.perf_counter()

        latencies.append(
            (end - start) * 1000
        )

    peak_vram = get_peak_gpu_memory_mb()

    mean_latency = statistics.mean(latencies)

    return {
        "model": "PP-OCRv6-medium",
        "task": "optical_character_recognition",
        "parameters_m": None,
        "weight_size_mb": weight_size_mb,
        "input_resolution": "1280x640",
        "load_time_s": load_time,
        "idle_vram_mb": idle_vram,
        "peak_vram_mb": peak_vram,
        "mean_latency_ms": mean_latency,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "throughput_qps": 1000 / mean_latency,
        "warmup_runs": warmup,
        "test_runs": runs,
        "device": "gpu:0",
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--warmup",
        type=int,
        default=10
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=50
    )

    args = parser.parse_args()

    result = profile_paddleocr(
        args.warmup,
        args.runs
    )

    print("\n========== RESULT ==========")

    for k, v in result.items():

        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

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
        / "paddleocr_profile.json"
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
