import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


ROOT = Path("/root/autodl-tmp/generalist-specialist")

MANIFEST = ROOT / "benchmark/manifests/image_classification.jsonl"

WEIGHT_PATH = (
    ROOT
    / "models/resnet50/resnet50-11ad3fa6.pth"
)

RESULT_DIR = ROOT / "benchmark/results/formal"

SUMMARY_PATH = (
    RESULT_DIR
    / "resnet50_imagenette_summary.json"
)

PREDICTION_PATH = (
    RESULT_DIR
    / "resnet50_imagenette_predictions.jsonl"
)


TARGET_NAMES = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]


def percentile(values, q):
    return float(np.percentile(values, q))


def stats(values):
    values = np.asarray(values, dtype=np.float64)

    return {
        "mean_ms": round(float(np.mean(values)), 3),
        "p50_ms": round(percentile(values, 50), 3),
        "p95_ms": round(percentile(values, 95), 3),
        "p99_ms": round(percentile(values, 99), 3),
        "min_ms": round(float(np.min(values)), 3),
        "max_ms": round(float(np.max(values)), 3),
    }


def sync():
    torch.cuda.synchronize()


def read_manifest(path):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            samples.append(
                json.loads(line)
            )

    return samples


def validate_manifest(samples):
    for sample in samples:
        label_id = int(sample["label_id"])

        if not 0 <= label_id < len(TARGET_NAMES):
            raise ValueError(
                f"Invalid label_id: {label_id}"
            )

        expected = TARGET_NAMES[label_id]

        if sample["label"] != expected:
            raise ValueError(
                f"Manifest mismatch: "
                f"id={sample['id']} "
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

    device = torch.device("cuda:0")

    torch.cuda.set_device(device)

    # 固定输入为 224x224，允许 cuDNN 对固定形状优化。
    torch.backends.cudnn.benchmark = True

    print("Device:", device)
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )
    print(
        "Torch:",
        torch.__version__
    )
    print(
        "Torchvision:",
        torchvision.__version__
    )

    # -----------------------------------
    # Dataset
    # -----------------------------------

    samples = read_manifest(
        MANIFEST
    )

    validate_manifest(samples)

    if args.limit > 0:
        samples = samples[:args.limit]

    print(
        "Evaluation samples:",
        len(samples)
    )

    # -----------------------------------
    # Preprocessing + closed-set mapping
    # -----------------------------------

    weights = (
        ResNet50_Weights.IMAGENET1K_V2
    )

    preprocess = weights.transforms()

    categories = weights.meta[
        "categories"
    ]

    imagenet_indices = [
        categories.index(name)
        for name in TARGET_NAMES
    ]

    print()
    print("Closed-set ImageNet mapping:")

    for label_id, (
        name,
        imagenet_index
    ) in enumerate(
        zip(
            TARGET_NAMES,
            imagenet_indices
        )
    ):
        print(
            f"{label_id}: "
            f"{name} -> "
            f"ImageNet {imagenet_index}"
        )

    index_tensor = torch.tensor(
        imagenet_indices,
        dtype=torch.long,
        device=device
    )

    # -----------------------------------
    # Model
    # -----------------------------------

    print()
    print("Loading ResNet50...")

    load_start = time.perf_counter()

    model = resnet50(
        weights=None
    )

    state_dict = torch.load(
        WEIGHT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        state_dict,
        strict=True
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
        f"Load time: "
        f"{load_time_s:.3f} s"
    )

    # -----------------------------------
    # Full-pipeline warmup
    #
    # 使用不在当前 benchmark samples 中的真实图片，
    # 预热 JPEG 解码、V2 transform、H2D 和 GPU forward。
    # warmup 不计入准确率和 latency。
    # -----------------------------------

    full_manifest = (
        ROOT
        / "benchmark/manifests/image_classification_full.jsonl"
    )

    benchmark_ids = {
        sample["id"]
        for sample in samples
    }

    warmup_sample = None

    for candidate in read_manifest(
        full_manifest
    ):
        if candidate["id"] not in benchmark_ids:
            warmup_sample = candidate
            break

    if warmup_sample is None:
        raise RuntimeError(
            "Could not find a separate warmup image."
        )

    print(
        f"Full-pipeline warmup runs: "
        f"{args.warmup}"
    )

    print(
        "Warmup image:",
        warmup_sample["input"]
    )

    with torch.inference_mode():

        for _ in range(
            args.warmup
        ):

            with Image.open(
                warmup_sample["input"]
            ) as img:

                img = img.convert(
                    "RGB"
                )

                warmup_tensor = (
                    preprocess(img)
                    .unsqueeze(0)
                    .to(device)
                )

            warmup_logits = model(
                warmup_tensor
            )

            warmup_logits_10 = (
                torch.index_select(
                    warmup_logits,
                    dim=1,
                    index=index_tensor
                )
            )

            _ = torch.argmax(
                warmup_logits_10,
                dim=1
            ).item()

            sync()

            del warmup_tensor
            del warmup_logits
            del warmup_logits_10

    torch.cuda.reset_peak_memory_stats(
        device
    )

    # -----------------------------------
    # Formal evaluation
    #
    # model_latency:
    #   GPU forward only
    #
    # e2e_latency:
    #   image file decode
    #   + V2 preprocessing
    #   + host-to-device transfer
    #   + model forward
    #   + closed-set postprocessing
    # -----------------------------------

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

            with Image.open(
                sample["input"]
            ) as img:

                img = img.convert(
                    "RGB"
                )

                tensor = (
                    preprocess(img)
                    .unsqueeze(0)
                    .to(device)
                )

            sync()

            model_start = (
                time.perf_counter()
            )

            logits = model(
                tensor
            )

            sync()

            model_end = (
                time.perf_counter()
            )

            # 1000-class -> Imagenette
            # closed-set 10-class logits
            logits_10 = torch.index_select(
                logits,
                dim=1,
                index=index_tensor
            )

            pred_label_id = int(
                torch.argmax(
                    logits_10,
                    dim=1
                ).item()
            )

            pred_label = TARGET_NAMES[
                pred_label_id
            ]

            is_correct = (
                pred_label_id
                == int(
                    sample["label_id"]
                )
            )

            if is_correct:
                correct += 1

            model_latency_ms = (
                model_end
                - model_start
            ) * 1000

            e2e_latency_ms = (
                time.perf_counter()
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
                    "image_classification",

                "dataset":
                    "imagenette",

                "input":
                    sample["input"],

                "label_id":
                    int(
                        sample["label_id"]
                    ),

                "label":
                    sample["label"],

                "wnid":
                    sample["wnid"],

                "prediction_id":
                    pred_label_id,

                "prediction":
                    pred_label,

                "correct":
                    is_correct,

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
                f"GT={sample['label']:<16} "
                f"Pred={pred_label:<16} "
                f"{'OK' if is_correct else 'WRONG'} "
                f"model={model_latency_ms:.2f}ms "
                f"e2e={e2e_latency_ms:.2f}ms"
            )

            del tensor
            del logits
            del logits_10

    # -----------------------------------
    # Summary
    # -----------------------------------

    accuracy = (
        correct / len(samples)
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
            "ResNet50",

        "task":
            "image_classification",

        "dataset":
            "Imagenette",

        "split":
            "validation",

        "evaluation_mode":
            "closed_set_10_class",

        "weights":
            "IMAGENET1K_V2",

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
                WEIGHT_PATH.stat().st_size
                / 1024**2,
                3
            ),

        "load_time_s":
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

        "latency_definition": {
            "model_latency":
                "GPU forward only",

            "e2e_latency":
                (
                    "local image decode + "
                    "IMAGENET1K_V2 preprocessing + "
                    "host-to-device transfer + "
                    "GPU forward + "
                    "closed-set postprocessing"
                ),
        },

        "class_mapping": [
            {
                "label_id":
                    i,

                "label":
                    name,

                "imagenet_index":
                    imagenet_indices[i],
            }
            for i, name
            in enumerate(
                TARGET_NAMES
            )
        ],

        "device":
            "cuda:0",

        "gpu":
            torch.cuda.get_device_name(
                0
            ),

        "torch_version":
            torch.__version__,

        "torchvision_version":
            torchvision.__version__,

        "timestamp":
            datetime.now().astimezone(
            ).isoformat(),
    }

    # -----------------------------------
    # Save
    # -----------------------------------

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
        f"Samples: "
        f"{len(samples)}"
    )

    print(
        f"Correct: "
        f"{correct}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
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
