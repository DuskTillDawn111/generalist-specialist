import json
import time
from pathlib import Path

import numpy as np
import paddle
from PIL import Image
from paddleocr import TextRecognition


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

MODEL_DIR = (
    ROOT
    / "models/paddleocr/official_models/PP-OCRv6_medium_rec"
)

MANIFEST = (
    ROOT
    / "benchmark/manifests/ocr.jsonl"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/ocr_full.jsonl"
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


def sync():
    paddle.device.synchronize()


def marker(name):
    sync()

    print(
        f"NVML_MARKER|{name}",
        flush=True
    )

    # Give the external sampler a stable
    # observation window.
    time.sleep(1.0)


def load_image_bgr(path):
    with Image.open(path) as img:
        rgb = np.asarray(
            img.convert("RGB")
        )

    # Same BGR ndarray convention as
    # the formal PaddleOCR runner.
    bgr = rgb[
        :,
        :,
        ::-1
    ].copy()

    return bgr


if not paddle.device.is_compiled_with_cuda():
    raise RuntimeError(
        "Paddle CUDA support is not available."
    )


paddle.set_device(
    "gpu:0"
)


print(
    "Paddle device:",
    paddle.device.get_device(),
    flush=True
)

print(
    "GPU:",
    paddle.device.cuda.get_device_name(),
    flush=True
)


samples = read_manifest(
    MANIFEST
)

full_samples = read_manifest(
    FULL_MANIFEST
)


if len(samples) != 200:
    raise RuntimeError(
        f"Expected 200 benchmark samples, "
        f"got {len(samples)}."
    )


benchmark_ids = {
    x["id"]
    for x in samples
}


warmup_sample = None

for candidate in full_samples:
    if candidate["id"] not in benchmark_ids:
        warmup_sample = candidate
        break


if warmup_sample is None:
    raise RuntimeError(
        "Could not find separate warmup sample."
    )


# ------------------------------------------------
# Load model exactly as formal runner
# ------------------------------------------------

print(
    "Loading PP-OCRv6-medium-rec...",
    flush=True
)


model = TextRecognition(
    model_name="PP-OCRv6_medium_rec",
    model_dir=str(
        MODEL_DIR
    ),
    device="gpu:0",
)


sync()

marker(
    "post_load"
)


# ------------------------------------------------
# Full-pipeline warmup
# Same count as formal benchmark.
# ------------------------------------------------

print(
    "Warmup image:",
    warmup_sample["input"],
    flush=True
)

warmup_bgr = load_image_bgr(
    warmup_sample["input"]
)


for _ in range(10):
    _ = model.predict(
        input=warmup_bgr,
        batch_size=1,
    )

    sync()


marker(
    "post_warmup"
)


# ------------------------------------------------
# Run actual 200-image benchmark set once.
#
# The objective here is VRAM profiling,
# not latency or quality re-evaluation.
# ------------------------------------------------

print(
    "Running 200 benchmark images...",
    flush=True
)


for i, sample in enumerate(
    samples,
    start=1
):

    image_bgr = load_image_bgr(
        sample["input"]
    )

    _ = model.predict(
        input=image_bgr,
        batch_size=1,
    )

    sync()

    if (
        i == 1
        or i % 50 == 0
        or i == len(samples)
    ):
        print(
            f"Processed {i}/{len(samples)}",
            flush=True
        )


marker(
    "post_inference"
)


print(
    "Probe complete.",
    flush=True
)
