import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


ROOT = Path("/root/autodl-tmp/generalist-specialist")

MODEL_DIR = ROOT / "models/distilbert-sst2"

MANIFEST = (
    ROOT
    / "benchmark/manifests/text_classification.jsonl"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/text_classification_full.jsonl"
)

RESULT_DIR = ROOT / "benchmark/results/formal"

SUMMARY_PATH = (
    RESULT_DIR
    / "distilbert_sst2_summary.json"
)

PREDICTION_PATH = (
    RESULT_DIR
    / "distilbert_sst2_predictions.jsonl"
)

LABELS = [
    "negative",
    "positive",
]


def read_manifest(path):
    samples = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                samples.append(
                    json.loads(line)
                )

    return samples


def percentile(values, q):
    return float(
        np.percentile(values, q)
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


def sync():
    torch.cuda.synchronize()


def weight_size_mb(path):
    total = 0

    for p in path.iterdir():
        if (
            p.is_file()
            and (
                p.suffix == ".safetensors"
                or p.name.endswith(".bin")
            )
        ):
            total += p.stat().st_size

    return total / (1024 ** 2)


def validate_manifest(samples):
    for sample in samples:
        label_id = int(
            sample["label_id"]
        )

        if label_id not in (0, 1):
            raise ValueError(
                f"Invalid label_id: {label_id}"
            )

        expected = LABELS[
            label_id
        ]

        if sample["label"].lower() != expected:
            raise ValueError(
                f"Manifest mismatch: "
                f"{sample['id']} "
                f"label_id={label_id} "
                f"label={sample['label']} "
                f"expected={expected}"
            )


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
        help="0 means use all samples"
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

    # --------------------------------
    # Dataset
    # --------------------------------

    samples = read_manifest(
        MANIFEST
    )

    validate_manifest(
        samples
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
    # Tokenizer
    # --------------------------------

    tokenizer_start = (
        time.perf_counter()
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            use_fast=True
        )
    )

    tokenizer_load_time_s = (
        time.perf_counter()
        - tokenizer_start
    )

    # --------------------------------
    # Model
    # --------------------------------

    print(
        "Loading DistilBERT..."
    )

    load_start = (
        time.perf_counter()
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True
        )
    )

    model.eval()
    model.to(device)

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

    print(
        f"Parameters: "
        f"{parameters_m:.3f} M"
    )

    print(
        "id2label:",
        model.config.id2label
    )

    print(
        f"Model load time: "
        f"{load_time_s:.3f} s"
    )

    # --------------------------------
    # Full-pipeline warmup
    # --------------------------------

    benchmark_ids = {
        sample["id"]
        for sample in samples
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
            "Could not find a separate warmup sample."
        )

    print(
        "Full-pipeline warmup runs:",
        args.warmup
    )

    print(
        "Warmup text:",
        warmup_sample["input"][:100]
    )

    with torch.inference_mode():

        for _ in range(
            args.warmup
        ):

            encoded = tokenizer(
                warmup_sample["input"],
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            encoded = {
                key: value.to(device)
                for key, value
                in encoded.items()
            }

            outputs = model(
                **encoded
            )

            sync()

            _ = int(
                torch.argmax(
                    outputs.logits,
                    dim=-1
                ).item()
            )

            del encoded
            del outputs

    torch.cuda.reset_peak_memory_stats(
        device
    )

    # --------------------------------
    # Evaluation
    #
    # model latency:
    # GPU forward only
    #
    # E2E:
    # tokenizer
    # + CPU -> GPU
    # + forward
    # + argmax/label mapping
    # --------------------------------

    predictions = []

    model_latencies = []
    e2e_latencies = []

    correct = 0

    print()
    print("Running evaluation...")

    with torch.inference_mode():

        for i, sample in enumerate(
            samples,
            start=1
        ):

            sync()

            e2e_start = (
                time.perf_counter()
            )

            encoded = tokenizer(
                sample["input"],
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            encoded = {
                key: value.to(device)
                for key, value
                in encoded.items()
            }

            sync()

            model_start = (
                time.perf_counter()
            )

            outputs = model(
                **encoded
            )

            sync()

            model_end = (
                time.perf_counter()
            )

            pred_id = int(
                torch.argmax(
                    outputs.logits,
                    dim=-1
                ).item()
            )

            pred_label = LABELS[
                pred_id
            ]

            e2e_end = (
                time.perf_counter()
            )

            gt_id = int(
                sample["label_id"]
            )

            is_correct = (
                pred_id == gt_id
            )

            if is_correct:
                correct += 1

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

            record = {
                "id":
                    sample["id"],

                "task":
                    "text_classification",

                "dataset":
                    "sst2",

                "input":
                    sample["input"],

                "label_id":
                    gt_id,

                "label":
                    sample["label"],

                "prediction_id":
                    pred_id,

                "prediction":
                    pred_label,

                "correct":
                    is_correct,

                "input_tokens":
                    int(
                        encoded[
                            "input_ids"
                        ].shape[1]
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
                f"GT={sample['label']:<8} "
                f"Pred={pred_label:<8} "
                f"{'OK' if is_correct else 'WRONG'} "
                f"tokens={record['input_tokens']:>3} "
                f"model={model_latency_ms:.2f}ms "
                f"e2e={e2e_latency_ms:.2f}ms"
            )

            del encoded
            del outputs

    # --------------------------------
    # Summary
    # --------------------------------

    accuracy = (
        correct
        / len(samples)
    )

    model_stats = stats(
        model_latencies
    )

    e2e_stats = stats(
        e2e_latencies
    )

    peak_allocated_mb = (
        torch.cuda.max_memory_allocated(
            device
        )
        / 1024**2
    )

    summary = {
        "model":
            "DistilBERT-SST2",

        "task":
            "text_classification",

        "dataset":
            "SST-2",

        "split":
            "validation",

        "precision":
            "float32",

        "batch_size":
            1,

        "num_samples":
            len(samples),

        "correct":
            correct,

        "accuracy":
            round(
                accuracy,
                6
            ),

        "parameters_m":
            round(
                parameters_m,
                3
            ),

        "weight_size_mb":
            round(
                weight_size_mb(
                    MODEL_DIR
                ),
                3
            ),

        "tokenizer_load_time_s":
            round(
                tokenizer_load_time_s,
                3
            ),

        "model_load_time_s":
            round(
                load_time_s,
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

        "torch_peak_allocated_mb":
            round(
                peak_allocated_mb,
                3
            ),

        "label_mapping": {
            "0":
                "negative",

            "1":
                "positive",
        },

        "latency_definition": {
            "model_latency":
                "GPU forward only",

            "e2e_latency":
                (
                    "tokenization + "
                    "host-to-device transfer + "
                    "GPU forward + "
                    "argmax and label mapping"
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
