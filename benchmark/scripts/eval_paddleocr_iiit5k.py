import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import paddle
import paddleocr
from PIL import Image
from paddleocr import TextRecognition


ROOT = Path("/root/autodl-tmp/generalist-specialist")

MODEL_DIR = (
    ROOT
    / "models/paddleocr/official_models/PP-OCRv6_medium_rec"
)

PARAM_FILE = MODEL_DIR / "inference.pdiparams"

MANIFEST = (
    ROOT
    / "benchmark/manifests/ocr.jsonl"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/ocr_full.jsonl"
)

RESULT_DIR = (
    ROOT
    / "benchmark/results/formal"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "paddleocr_iiit5k_summary.json"
)

PREDICTION_PATH = (
    RESULT_DIR
    / "paddleocr_iiit5k_predictions.jsonl"
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


def directory_size_mb(path):
    total = 0

    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size

    return total / (1024 ** 2)


def sync():
    paddle.device.synchronize()


def load_image_bgr(path):
    """
    PaddleOCR/PaddleX image modules conventionally
    operate on BGR numpy arrays.

    E2E timing includes:
        file read
        decode
        RGB -> BGR conversion
    """

    with Image.open(
        path
    ) as img:

        rgb = np.asarray(
            img.convert("RGB")
        )

    bgr = rgb[
        :,
        :,
        ::-1
    ].copy()

    return bgr


def extract_result(output):
    if not output:
        return "", 0.0

    result = output[0]

    try:
        text = str(
            result["rec_text"]
        )

        score = float(
            result["rec_score"]
        )

    except Exception as e:
        raise RuntimeError(
            f"Unable to parse PaddleOCR result: {result}"
        ) from e

    return text, score


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--warmup",
        type=int,
        default=10
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means use all 200 samples"
    )

    args = parser.parse_args()

    if not paddle.device.is_compiled_with_cuda():
        raise RuntimeError(
            "Paddle CUDA support is not available."
        )

    paddle.set_device(
        "gpu:0"
    )

    print(
        "Device:",
        paddle.device.get_device()
    )

    print(
        "GPU:",
        paddle.device.cuda.get_device_name()
    )

    print(
        "Paddle:",
        paddle.__version__
    )

    print(
        "PaddleOCR:",
        paddleocr.__version__
    )

    # --------------------------------
    # Dataset
    # --------------------------------

    samples = read_manifest(
        MANIFEST
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

    print(
        "Loading PP-OCRv6-medium-rec..."
    )

    sync()

    load_start = (
        time.perf_counter()
    )

    model = TextRecognition(
        model_name="PP-OCRv6_medium_rec",
        model_dir=str(
            MODEL_DIR
        ),
        device="gpu:0"
    )

    sync()

    model_load_time_s = (
        time.perf_counter()
        - load_start
    )

    print(
        f"Model load time: "
        f"{model_load_time_s:.3f} s"
    )

    # --------------------------------
    # Full-pipeline warmup
    # --------------------------------

    benchmark_ids = {
        x["id"]
        for x in samples
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
            "Could not find separate warmup image."
        )

    print(
        "Full-pipeline warmup runs:",
        args.warmup
    )

    print(
        "Warmup image:",
        warmup_sample["input"]
    )

    for _ in range(
        args.warmup
    ):

        image_bgr = load_image_bgr(
            warmup_sample["input"]
        )

        _ = model.predict(
            input=image_bgr,
            batch_size=1
        )

        sync()

    # Paddle allocator stats are kept only
    # as framework-specific auxiliary metadata.
    try:
        paddle.device.cuda.reset_max_memory_allocated()
    except Exception:
        pass

    # --------------------------------
    # Formal evaluation
    #
    # model_latency:
    #   TextRecognition.predict()
    #   on an already decoded BGR ndarray.
    #
    # This includes PaddleOCR recognition-module
    # preprocessing + GPU inference + postprocessing.
    #
    # E2E latency:
    #   local image read/decode
    #   + RGB->BGR
    #   + recognition module
    #   + OCR normalization.
    # --------------------------------

    predictions = []

    model_latencies = []
    e2e_latencies = []

    correct = 0

    total_char_errors = 0
    total_reference_chars = 0

    confidence_scores = []

    print()
    print(
        "Running evaluation..."
    )

    for i, sample in enumerate(
        samples,
        start=1
    ):

        sync()

        e2e_start = (
            time.perf_counter()
        )

        # -----------------------------
        # Image file read + decode
        # -----------------------------

        image_bgr = load_image_bgr(
            sample["input"]
        )

        sync()

        # -----------------------------
        # Recognition module latency
        # -----------------------------

        model_start = (
            time.perf_counter()
        )

        output = model.predict(
            input=image_bgr,
            batch_size=1
        )

        sync()

        model_end = (
            time.perf_counter()
        )

        prediction, rec_score = (
            extract_result(
                output
            )
        )

        normalized_prediction = (
            normalize_ocr(
                prediction
            )
        )

        normalized_reference = (
            normalize_ocr(
                sample[
                    "normalized_label"
                ]
            )
        )

        e2e_end = (
            time.perf_counter()
        )

        # -----------------------------
        # Accuracy + CER
        # -----------------------------

        is_correct = (
            normalized_prediction
            == normalized_reference
        )

        if is_correct:
            correct += 1

        char_errors = (
            levenshtein_chars(
                normalized_reference,
                normalized_prediction
            )
        )

        reference_chars = len(
            normalized_reference
        )

        total_char_errors += (
            char_errors
        )

        total_reference_chars += (
            reference_chars
        )

        sample_cer = (
            char_errors
            / reference_chars
            if reference_chars > 0
            else 0.0
        )

        # -----------------------------
        # Timing
        # -----------------------------

        model_latency_ms = (
            model_end
            - model_start
        ) * 1000

        e2e_latency_ms = (
            e2e_end
            - e2e_start
        ) * 1000

        model_latencies.append(
            model_latency_ms
        )

        e2e_latencies.append(
            e2e_latency_ms
        )

        confidence_scores.append(
            rec_score
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
                normalized_reference,

            "prediction":
                prediction,

            "normalized_prediction":
                normalized_prediction,

            "rec_score":
                round(
                    rec_score,
                    8
                ),

            "correct":
                is_correct,

            "word_length":
                int(
                    sample[
                        "word_length"
                    ]
                ),

            "char_errors":
                char_errors,

            "reference_chars":
                reference_chars,

            "cer":
                round(
                    sample_cer,
                    6
                ),

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
            f"GT={normalized_reference:<18} "
            f"Pred={normalized_prediction:<18} "
            f"{'OK' if is_correct else 'WRONG'} "
            f"score={rec_score:.4f} "
            f"model={model_latency_ms:.2f}ms "
            f"e2e={e2e_latency_ms:.2f}ms"
        )

    # --------------------------------
    # Summary
    # --------------------------------

    exact_match = (
        correct
        / len(samples)
    )

    cer = (
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

    try:
        peak_allocated_mb = (
            paddle.device.cuda.max_memory_allocated()
            / 1024**2
        )

        peak_allocated_mb = round(
            float(
                peak_allocated_mb
            ),
            3
        )

    except Exception:
        peak_allocated_mb = None

    summary = {
        "model":
            "PP-OCRv6-medium-rec",

        "task":
            "optical_character_recognition",

        "dataset":
            "IIIT5K",

        "split":
            "test",

        "evaluation_mode":
            "recognition_only",

        "batch_size":
            1,

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
                cer,
                6
            ),

        "mean_rec_score":
            round(
                float(
                    np.mean(
                        confidence_scores
                    )
                ),
                6
            ),

        "parameter_file":
            "inference.pdiparams",

        "parameter_file_size_mb":
            round(
                PARAM_FILE.stat().st_size
                / 1024**2,
                3
            ),

        "model_dir_size_mb":
            round(
                directory_size_mb(
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

        "model_latency":
            model_stats,

        "e2e_latency":
            e2e_stats,

        "model_throughput_qps":
            round(
                1000.0
                / model_stats[
                    "mean_ms"
                ],
                3
            ),

        "e2e_throughput_qps":
            round(
                1000.0
                / e2e_stats[
                    "mean_ms"
                ],
                3
            ),

        "paddle_peak_allocated_mb":
            peak_allocated_mb,

        "normalization":
            (
                "lowercase; remove all "
                "non-alphanumeric characters"
            ),

        "latency_definition": {
            "model_latency":
                (
                    "PP-OCRv6 TextRecognition.predict "
                    "on a pre-decoded BGR numpy image; "
                    "includes recognition-module "
                    "preprocessing, GPU inference "
                    "and postprocessing"
                ),

            "e2e_latency":
                (
                    "local image read/decode + "
                    "RGB-to-BGR conversion + "
                    "TextRecognition.predict + "
                    "OCR normalization"
                ),
        },

        "memory_note":
            (
                "Paddle allocator memory is "
                "framework-specific auxiliary data; "
                "cross-framework VRAM comparison "
                "will use process-level NVML later"
            ),

        "device":
            paddle.device.get_device(),

        "gpu":
            paddle.device.cuda.get_device_name(),

        "paddle_version":
            paddle.__version__,

        "paddleocr_version":
            paddleocr.__version__,

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
        "Exact Match:",
        f"{exact_match:.4f}"
    )

    print(
        "CER:",
        f"{cer:.4f}"
    )

    print(
        "Mean confidence:",
        f"{np.mean(confidence_scores):.4f}"
    )

    print()

    print(
        "Model latency:",
        model_stats
    )

    print(
        "E2E latency:",
        e2e_stats
    )

    print(
        "Model throughput:",
        summary[
            "model_throughput_qps"
        ],
        "qps"
    )

    print(
        "E2E throughput:",
        summary[
            "e2e_throughput_qps"
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
