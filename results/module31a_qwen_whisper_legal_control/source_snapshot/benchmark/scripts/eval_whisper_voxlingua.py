import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)

import eval_whisper_librispeech as asr_base


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

MODEL_DIR = (
    ROOT
    / "models/whisper-small"
)

MANIFEST = (
    ROOT
    / "benchmark/manifests/lid.jsonl"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/lid_full.jsonl"
)

RESULT_DIR = (
    ROOT
    / "benchmark/results/formal"
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


def normalize_lang_code(value):

    value = (
        str(value)
        .strip()
        .lower()
    )

    if (
        value.startswith("<|")
        and value.endswith("|>")
    ):
        value = value[
            2:-2
        ]

    return value


def validate_sample(sample):

    required = [
        "id",
        "task",
        "dataset",
        "split",
        "input_type",
        "input",
        "label",
        "duration_s",
        "sample_rate",
    ]

    missing = [
        key
        for key in required
        if key not in sample
    ]

    if missing:
        raise ValueError(
            f"{sample.get('id')}: "
            f"missing keys {missing}"
        )

    if sample["task"] != "lid":
        raise ValueError(
            f"{sample['id']}: "
            f"unexpected task="
            f"{sample['task']}"
        )

    if sample["input_type"] != "audio":
        raise ValueError(
            f"{sample['id']}: "
            "input_type must be audio"
        )

    if int(
        sample[
            "sample_rate"
        ]
    ) != 16000:
        raise ValueError(
            f"{sample['id']}: "
            f"expected 16000 Hz, got "
            f"{sample['sample_rate']}"
        )


def token_id_to_language(
    token_id,
    generation_config
):

    inverse = {
        int(token_id):
            normalize_lang_code(
                token
            )
        for token, token_id
        in generation_config.lang_to_id.items()
    }

    token_id = int(
        token_id
    )

    if token_id not in inverse:
        raise RuntimeError(
            "Whisper detect_language returned "
            f"unknown token id {token_id}"
        )

    return inverse[
        token_id
    ]


def run_one(
    model,
    processor,
    sample,
    device
):

    # E2E begins before local file decode.
    asr_base.sync()

    e2e_start = (
        time.perf_counter()
    )

    audio, sr = (
        asr_base.load_audio(
            sample[
                "input"
            ]
        )
    )

    # LID only needs Whisper acoustic
    # features. For language detection,
    # Whisper operates on its standard
    # input_features representation.
    processor_start = (
        time.perf_counter()
    )

    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_features = (
        inputs[
            "input_features"
        ]
        .to(
            device
        )
    )

    processor_end = (
        time.perf_counter()
    )

    asr_base.sync()

    model_start = (
        time.perf_counter()
    )

    language_tokens = (
        model.detect_language(
            input_features=(
                input_features
            ),
            generation_config=(
                model.generation_config
            ),
        )
    )

    asr_base.sync()

    model_end = (
        time.perf_counter()
    )

    token_id = int(
        language_tokens
        .detach()
        .cpu()
        .reshape(-1)[0]
        .item()
    )

    prediction = (
        token_id_to_language(
            token_id,
            model.generation_config
        )
    )

    reference = (
        normalize_lang_code(
            sample[
                "label"
            ]
        )
    )

    e2e_end = (
        time.perf_counter()
    )

    return {
        "prediction":
            prediction,

        "reference":
            reference,

        "language_token_id":
            token_id,

        "correct":
            (
                prediction
                == reference
            ),

        "audio_samples":
            int(
                audio.shape[0]
            ),

        "sample_rate":
            int(sr),

        "duration_s":
            float(
                sample[
                    "duration_s"
                ]
            ),

        "input_features_shape":
            list(
                input_features.shape
            ),

        "processor_latency_ms":
            (
                processor_end
                - processor_start
            ) * 1000.0,

        "model_latency_ms":
            (
                model_end
                - model_start
            ) * 1000.0,

        "e2e_latency_ms":
            (
                e2e_end
                - e2e_start
            ) * 1000.0,
    }


def mean(values):

    if not values:
        return None

    return (
        sum(values)
        / len(values)
    )


def percentile(
    values,
    q
):

    if not values:
        return None

    values = sorted(
        values
    )

    if len(values) == 1:
        return values[0]

    pos = (
        (len(values) - 1)
        * q
        / 100.0
    )

    lo = int(pos)

    hi = min(
        lo + 1,
        len(values) - 1
    )

    frac = (
        pos
        - lo
    )

    return (
        values[lo]
        * (1.0 - frac)
        + values[hi]
        * frac
    )


def latency_stats(values):

    if not values:
        return {
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
        }

    return {
        "mean_ms":
            round(
                mean(values),
                3
            ),

        "p50_ms":
            round(
                percentile(
                    values,
                    50
                ),
                3
            ),

        "p95_ms":
            round(
                percentile(
                    values,
                    95
                ),
                3
            ),

        "p99_ms":
            round(
                percentile(
                    values,
                    99
                ),
                3
            ),

        "min_ms":
            round(
                min(values),
                3
            ),

        "max_ms":
            round(
                max(values),
                3
            ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--warmup",
        type=int,
        default=5
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "0 means all benchmark samples"
        )
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

    # ----------------------------
    # Dataset
    # ----------------------------

    all_samples = (
        read_manifest(
            MANIFEST
        )
    )

    for sample in all_samples:
        validate_sample(
            sample
        )

    samples = list(
        all_samples
    )

    if args.limit > 0:
        samples = samples[
            :args.limit
        ]

    if not samples:
        raise RuntimeError(
            "No evaluation samples."
        )

    valid_labels = sorted(
        {
            normalize_lang_code(
                sample[
                    "label"
                ]
            )
            for sample in samples
        }
    )

    print(
        "Evaluation samples:",
        len(samples)
    )

    print(
        "Benchmark languages:",
        valid_labels
    )

    # ----------------------------
    # Processor
    # ----------------------------

    print(
        "Loading Whisper processor..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
    )

    print(
        "Sampling rate:",
        processor
        .feature_extractor
        .sampling_rate
    )

    # ----------------------------
    # Model
    # ----------------------------

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(
        "Loading Whisper-small..."
    )

    load_start = (
        time.perf_counter()
    )

    model = (
        AutoModelForSpeechSeq2Seq
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True
        )
    )

    model.eval()

    model.to(
        device
    )

    asr_base.sync()

    model_load_time_s = (
        time.perf_counter()
        - load_start
    )

    if not hasattr(
        model,
        "detect_language"
    ):
        raise RuntimeError(
            "Loaded Whisper model does not "
            "provide detect_language()."
        )

    if not getattr(
        model.generation_config,
        "lang_to_id",
        None
    ):
        raise RuntimeError(
            "Whisper generation config does "
            "not provide lang_to_id."
        )

    whisper_languages = {
        normalize_lang_code(
            token
        )
        for token in (
            model
            .generation_config
            .lang_to_id
            .keys()
        )
    }

    unsupported = sorted(
        set(valid_labels)
        - whisper_languages
    )

    if unsupported:
        raise RuntimeError(
            "Benchmark language labels are "
            "not supported by Whisper: "
            f"{unsupported}"
        )

    parameters_m = (
        sum(
            p.numel()
            for p in model.parameters()
        )
        / 1e6
    )

    idle_allocated_mb = (
        torch.cuda.memory_allocated(
            device
        )
        / 1024**2
    )

    print(
        "Model load time:",
        f"{model_load_time_s:.3f}s"
    )

    print(
        "Parameters:",
        f"{parameters_m:.3f}M"
    )

    print(
        "Whisper language tokens:",
        len(
            whisper_languages
        )
    )

    # ----------------------------
    # Warmup
    # ----------------------------

    if args.warmup > 0:

        benchmark_ids = {
            sample[
                "id"
            ]
            for sample in samples
        }

        warmup_sample = None

        for candidate in read_manifest(
            FULL_MANIFEST
        ):

            validate_sample(
                candidate
            )

            if (
                candidate["id"]
                not in benchmark_ids
            ):
                warmup_sample = candidate
                break

        if warmup_sample is None:
            raise RuntimeError(
                "No separate LID warmup "
                "sample found."
            )

        print(
            "Warmup runs:",
            args.warmup
        )

        print(
            "Warmup sample:",
            warmup_sample[
                "id"
            ]
        )

        with torch.inference_mode():

            for _ in range(
                args.warmup
            ):
                run_one(
                    model=model,
                    processor=processor,
                    sample=warmup_sample,
                    device=device,
                )

    torch.cuda.reset_peak_memory_stats(
        device
    )

    # ----------------------------
    # Formal evaluation
    # ----------------------------

    predictions = []

    correct = 0

    model_latencies = []
    e2e_latencies = []
    processor_latencies = []

    per_language = {
        label: {
            "total": 0,
            "correct": 0,
        }
        for label in valid_labels
    }

    predicted_language_counts = (
        Counter()
    )

    print()
    print(
        "Running Whisper LID evaluation..."
    )

    with torch.inference_mode():

        for i, sample in enumerate(
            samples,
            start=1
        ):

            result = run_one(
                model=model,
                processor=processor,
                sample=sample,
                device=device,
            )

            pred = result[
                "prediction"
            ]

            gt = result[
                "reference"
            ]

            is_correct = (
                result[
                    "correct"
                ]
            )

            if is_correct:
                correct += 1

            per_language[
                gt
            ][
                "total"
            ] += 1

            if is_correct:
                per_language[
                    gt
                ][
                    "correct"
                ] += 1

            predicted_language_counts[
                pred
            ] += 1

            model_latencies.append(
                result[
                    "model_latency_ms"
                ]
            )

            e2e_latencies.append(
                result[
                    "e2e_latency_ms"
                ]
            )

            processor_latencies.append(
                result[
                    "processor_latency_ms"
                ]
            )

            record = {
                "id":
                    sample[
                        "id"
                    ],

                "task":
                    "lid",

                "dataset":
                    "VoxLingua107",

                "split":
                    sample[
                        "split"
                    ],

                "input":
                    sample[
                        "input"
                    ],

                "label":
                    gt,

                "language_name":
                    sample.get(
                        "language_name"
                    ),

                "source_label":
                    sample.get(
                        "source_label"
                    ),

                "prediction":
                    pred,

                "language_token_id":
                    result[
                        "language_token_id"
                    ],

                "correct":
                    is_correct,

                "duration_s":
                    result[
                        "duration_s"
                    ],

                "sample_rate":
                    result[
                        "sample_rate"
                    ],

                "audio_samples":
                    result[
                        "audio_samples"
                    ],

                "input_features_shape":
                    result[
                        "input_features_shape"
                    ],

                "processor_latency_ms":
                    round(
                        result[
                            "processor_latency_ms"
                        ],
                        3
                    ),

                "model_latency_ms":
                    round(
                        result[
                            "model_latency_ms"
                        ],
                        3
                    ),

                "e2e_latency_ms":
                    round(
                        result[
                            "e2e_latency_ms"
                        ],
                        3
                    ),
            }

            predictions.append(
                record
            )

            print(
                f"[{i:03d}/"
                f"{len(samples):03d}] "
                f"GT={gt:<4} "
                f"Pred={pred:<4} "
                f"{'OK' if is_correct else 'WRONG'} "
                f"model="
                f"{result['model_latency_ms']:.2f}ms "
                f"e2e="
                f"{result['e2e_latency_ms']:.2f}ms"
            )

    accuracy = (
        correct
        / len(samples)
    )

    per_language_summary = {}

    for label in valid_labels:

        total = (
            per_language[
                label
            ][
                "total"
            ]
        )

        lang_correct = (
            per_language[
                label
            ][
                "correct"
            ]
        )

        per_language_summary[
            label
        ] = {
            "total":
                total,

            "correct":
                lang_correct,

            "accuracy":
                (
                    round(
                        lang_correct
                        / total,
                        6
                    )
                    if total
                    else None
                ),
        }

    model_stats = (
        latency_stats(
            model_latencies
        )
    )

    e2e_stats = (
        latency_stats(
            e2e_latencies
        )
    )

    processor_stats = (
        latency_stats(
            processor_latencies
        )
    )

    peak_allocated_mb = (
        torch.cuda.max_memory_allocated(
            device
        )
        / 1024**2
    )

    summary = {
        "model":
            "Whisper-small",

        "task":
            "lid",

        "dataset":
            "VoxLingua107",

        "split":
            "dev",

        "model_role":
            "specialist",

        "inference_method":
            "Whisper.detect_language",

        "num_samples":
            len(samples),

        "num_languages":
            len(valid_labels),

        "benchmark_languages":
            valid_labels,

        "whisper_supported_languages":
            len(
                whisper_languages
            ),

        "correct":
            correct,

        "accuracy":
            round(
                accuracy,
                6
            ),

        "per_language":
            per_language_summary,

        "predicted_language_counts":
            dict(
                predicted_language_counts
            ),

        "model_load_time_s":
            round(
                model_load_time_s,
                3
            ),

        "parameters_m":
            round(
                parameters_m,
                3
            ),

        "processor_latency":
            processor_stats,

        "model_latency":
            model_stats,

        "e2e_latency":
            e2e_stats,

        "model_request_service_rate_proxy_qps":
            (
                round(
                    1000.0
                    / model_stats[
                        "mean_ms"
                    ],
                    6
                )
                if model_stats[
                    "mean_ms"
                ]
                else None
            ),

        "e2e_request_service_rate_proxy_qps":
            (
                round(
                    1000.0
                    / e2e_stats[
                        "mean_ms"
                    ],
                    6
                )
                if e2e_stats[
                    "mean_ms"
                ]
                else None
            ),

        "torch_idle_allocated_mb":
            round(
                idle_allocated_mb,
                3
            ),

        "torch_peak_allocated_mb":
            round(
                peak_allocated_mb,
                3
            ),

        "memory_note":
            (
                "Final deployment VRAM "
                "comparison uses process-level "
                "NVML rather than this Torch "
                "allocator statistic."
            ),

        "latency_definition": {
            "model_latency":
                (
                    "synchronized "
                    "Whisper.detect_language() "
                    "after feature extraction "
                    "and H2D"
                ),

            "e2e_latency":
                (
                    "local audio read/decode + "
                    "feature extraction + H2D + "
                    "detect_language + "
                    "language-token mapping"
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
            datetime.now(
            ).astimezone(
            ).isoformat(),
    }

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.limit > 0:

        prediction_path = (
            RESULT_DIR
            / (
                "whisper_voxlingua_"
                "smoke_predictions.jsonl"
            )
        )

        summary_path = (
            RESULT_DIR
            / (
                "whisper_voxlingua_"
                "smoke_summary.json"
            )
        )

    else:

        prediction_path = (
            RESULT_DIR
            / (
                "whisper_voxlingua_"
                "predictions.jsonl"
            )
        )

        summary_path = (
            RESULT_DIR
            / (
                "whisper_voxlingua_"
                "summary.json"
            )
        )

    with open(
        prediction_path,
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
        summary_path,
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
        "===== SUMMARY ====="
    )

    print(
        "Accuracy:",
        f"{accuracy:.4f}"
    )

    print(
        "Saved predictions:",
        prediction_path
    )

    print(
        "Saved summary:",
        summary_path
    )


if __name__ == "__main__":
    main()
