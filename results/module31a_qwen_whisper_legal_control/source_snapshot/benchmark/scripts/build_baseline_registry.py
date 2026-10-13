import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/root/autodl-tmp/generalist-specialist")
FORMAL = ROOT / "benchmark/results/formal"

OUT_JSON = ROOT / "benchmark/results/baseline_registry.json"
OUT_CSV = ROOT / "benchmark/results/baseline_registry.csv"


EXPECTED_FILES = [
    "distilbert_sst2_summary.json",
    "resnet50_imagenette_summary.json",
    "paddleocr_iiit5k_summary.json",
    "whisper_small_librispeech_summary.json",

    "phi4_sst2_summary.json",
    "phi4_imagenette_summary.json",
    "phi4_iiit5k_summary.json",
    "phi4_librispeech_summary.json",

    "minicpm_sst2_summary.json",
    "minicpm_imagenette_summary.json",
    "minicpm_iiit5k_summary.json",
    "minicpm_librispeech_summary.json",

    "qwen_sst2_summary.json",
    "qwen_imagenette_summary.json",
    "qwen_iiit5k_summary.json",
    "qwen_librispeech_summary.json",
]


SPECIALIST_MODELS = {
    "DistilBERT-SST2",
    "ResNet50",
    "PP-OCRv6-medium-rec",
    "Whisper-small",
}


MODEL_KEYS = {
    "DistilBERT-SST2":
        "distilbert_sst2",

    "ResNet50":
        "resnet50",

    "PP-OCRv6-medium-rec":
        "ppocrv6_medium_rec",

    "Whisper-small":
        "whisper_small",

    "Phi-4-multimodal-instruct":
        "phi4_multimodal",

    "MiniCPM-o-4.5":
        "minicpm_o_4_5",

    "Qwen2.5-Omni-7B":
        "qwen2_5_omni_7b",
}


EXPECTED_PAIRS = {
    ("distilbert_sst2", "text_classification"),
    ("resnet50", "image_classification"),
    ("ppocrv6_medium_rec", "ocr"),
    ("whisper_small", "asr"),

    ("phi4_multimodal", "text_classification"),
    ("phi4_multimodal", "image_classification"),
    ("phi4_multimodal", "ocr"),
    ("phi4_multimodal", "asr"),

    ("minicpm_o_4_5", "text_classification"),
    ("minicpm_o_4_5", "image_classification"),
    ("minicpm_o_4_5", "ocr"),
    ("minicpm_o_4_5", "asr"),

    ("qwen2_5_omni_7b", "text_classification"),
    ("qwen2_5_omni_7b", "image_classification"),
    ("qwen2_5_omni_7b", "ocr"),
    ("qwen2_5_omni_7b", "asr"),
}


def canonical_task(raw):
    mapping = {
        "text_classification":
            "text_classification",

        "image_classification":
            "image_classification",

        "optical_character_recognition":
            "ocr",

        "ocr":
            "ocr",

        "automatic_speech_recognition":
            "asr",

        "asr":
            "asr",
    }

    if raw not in mapping:
        raise ValueError(
            f"Unknown task name: {raw}"
        )

    return mapping[raw]


def stat(block, key):
    if not isinstance(block, dict):
        return None

    return block.get(key)


def inverse_ms(mean_ms):
    if mean_ms is None:
        return None

    if mean_ms <= 0:
        return None

    return round(
        1000.0 / mean_ms,
        6
    )


def pick_artifact_size(d):
    # Preserve semantics. This field is only
    # a convenience value and is NOT equivalent
    # to runtime resident VRAM.

    if d.get(
        "checkpoint_safetensors_total_mb"
    ) is not None:
        return (
            d[
                "checkpoint_safetensors_total_mb"
            ],
            "checkpoint_safetensors_total"
        )

    if d.get(
        "checkpoint_size_mb"
    ) is not None:
        return (
            d["checkpoint_size_mb"],
            "checkpoint_size"
        )

    if d.get(
        "model_dir_size_mb"
    ) is not None:
        return (
            d["model_dir_size_mb"],
            "model_directory"
        )

    if d.get(
        "weight_size_mb"
    ) is not None:
        return (
            d["weight_size_mb"],
            "weight_file_or_model_weights"
        )

    return None, None


def get_latency_blocks(d, task):
    if task == "asr":

        all_block = d.get(
            "all_samples",
            {}
        )

        all_model = (
            all_block.get(
                "model_latency"
            )
            or
            all_block.get(
                "chat_latency"
            )
        )

        all_e2e = all_block.get(
            "e2e_latency"
        )

        fixed_block = (
            d.get(
                "fixed_latency_subset"
            )
            or
            d.get(
                "fixed_duration_latency"
            )
            or
            {}
        )

        fixed_model = (
            fixed_block.get(
                "model_latency"
            )
            or
            fixed_block.get(
                "chat_latency"
            )
        )

        fixed_e2e = fixed_block.get(
            "e2e_latency"
        )

        return (
            all_model,
            all_e2e,
            fixed_block,
            fixed_model,
            fixed_e2e,
        )

    all_model = (
        d.get("model_latency")
        or
        d.get("chat_latency")
    )

    all_e2e = d.get(
        "e2e_latency"
    )

    return (
        all_model,
        all_e2e,
        {},
        None,
        None,
    )


def get_asr_rtf(d):
    all_block = d.get(
        "all_samples",
        {}
    )

    fixed_block = (
        d.get(
            "fixed_latency_subset"
        )
        or
        d.get(
            "fixed_duration_latency"
        )
        or
        {}
    )

    all_model_dist = (
        all_block.get(
            "model_rtf"
        )
        or
        all_block.get(
            "model_rtf_distribution"
        )
        or
        all_block.get(
            "chat_rtf_distribution"
        )
        or
        {}
    )

    all_e2e_dist = (
        all_block.get(
            "e2e_rtf"
        )
        or
        all_block.get(
            "e2e_rtf_distribution"
        )
        or
        {}
    )

    fixed_model_dist = (
        fixed_block.get(
            "model_rtf"
        )
        or
        fixed_block.get(
            "model_rtf_distribution"
        )
        or
        fixed_block.get(
            "chat_rtf_distribution"
        )
        or
        {}
    )

    fixed_e2e_dist = (
        fixed_block.get(
            "e2e_rtf"
        )
        or
        fixed_block.get(
            "e2e_rtf_distribution"
        )
        or
        {}
    )

    aggregate_model_rtf = (
        all_block.get(
            "aggregate_model_rtf"
        )
    )

    if aggregate_model_rtf is None:
        aggregate_model_rtf = (
            all_block.get(
                "aggregate_chat_rtf"
            )
        )

    aggregate_e2e_rtf = (
        all_block.get(
            "aggregate_e2e_rtf"
        )
    )

    return {
        "all_model_rtf_mean":
            all_model_dist.get("mean"),

        "all_e2e_rtf_mean":
            all_e2e_dist.get("mean"),

        "aggregate_model_rtf":
            aggregate_model_rtf,

        "aggregate_e2e_rtf":
            aggregate_e2e_rtf,

        "fixed_model_rtf_mean":
            fixed_model_dist.get("mean"),

        "fixed_e2e_rtf_mean":
            fixed_e2e_dist.get("mean"),
    }


def build_record(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        d = json.load(f)

    model = d["model"]

    if model not in MODEL_KEYS:
        raise ValueError(
            f"Unknown model: {model}"
        )

    model_key = MODEL_KEYS[model]

    task = canonical_task(
        d["task"]
    )

    role = d.get(
        "model_role"
    )

    if role is None:
        role = (
            "specialist"
            if model in SPECIALIST_MODELS
            else "generalist"
        )

    (
        all_model_latency,
        all_e2e_latency,
        fixed_block,
        fixed_model_latency,
        fixed_e2e_latency,
    ) = get_latency_blocks(
        d,
        task
    )

    # ------------------------------------------
    # Quality
    # ------------------------------------------

    accuracy = d.get(
        "accuracy"
    )

    exact_match = (
        d.get(
            "exact_match_accuracy"
        )
    )

    if exact_match is None:
        exact_match = d.get(
            "exact_match"
        )

    cer = d.get(
        "cer"
    )

    wer = d.get(
        "wer"
    )

    if task in {
        "text_classification",
        "image_classification",
    }:
        quality_metric = "accuracy"
        quality_value = accuracy
        higher_is_better = True

    elif task == "ocr":
        quality_metric = "exact_match"
        quality_value = exact_match
        higher_is_better = True

    elif task == "asr":
        quality_metric = "wer"
        quality_value = wer
        higher_is_better = False

    else:
        raise ValueError(
            f"Unsupported task: {task}"
        )

    # ------------------------------------------
    # Primary latency:
    #
    # ASR uses fixed [4,6) s subset for
    # cross-model latency comparison.
    #
    # Other tasks use all benchmark samples.
    # ------------------------------------------

    if task == "asr":
        primary_scope = "fixed_[4,6)_seconds"
        primary_model_latency = (
            fixed_model_latency
        )
        primary_e2e_latency = (
            fixed_e2e_latency
        )
    else:
        primary_scope = "all_samples"
        primary_model_latency = (
            all_model_latency
        )
        primary_e2e_latency = (
            all_e2e_latency
        )

    # ------------------------------------------
    # Artifact/checkpoint size
    # ------------------------------------------

    (
        artifact_size_mb,
        artifact_size_semantics,
    ) = pick_artifact_size(d)

    # ------------------------------------------
    # Allocator memory
    # ------------------------------------------

    allocator_backend = None
    allocator_peak_mb = None

    if (
        d.get(
            "torch_peak_allocated_mb"
        )
        is not None
    ):
        allocator_backend = "torch"
        allocator_peak_mb = d[
            "torch_peak_allocated_mb"
        ]

    elif (
        d.get(
            "paddle_peak_allocated_mb"
        )
        is not None
    ):
        allocator_backend = "paddle"
        allocator_peak_mb = d[
            "paddle_peak_allocated_mb"
        ]

    # ------------------------------------------
    # Token statistics
    # ------------------------------------------

    input_token_block = d.get(
        "input_tokens",
        {}
    )

    output_token_block = d.get(
        "output_tokens",
        {}
    )

    # ------------------------------------------
    # ASR RTF
    # ------------------------------------------

    if task == "asr":
        rtf = get_asr_rtf(d)
    else:
        rtf = {
            "all_model_rtf_mean":
                None,

            "all_e2e_rtf_mean":
                None,

            "aggregate_model_rtf":
                None,

            "aggregate_e2e_rtf":
                None,

            "fixed_model_rtf_mean":
                None,

            "fixed_e2e_rtf_mean":
                None,
        }

    record = {
        "model_key":
            model_key,

        "model":
            model,

        "role":
            role,

        "task":
            task,

        "dataset":
            d.get("dataset"),

        "split":
            d.get("split"),

        "evaluation_mode":
            (
                d.get("evaluation_mode")
                or
                d.get("evaluation_setting")
            ),

        "precision":
            d.get("precision"),

        "batch_size":
            d.get("batch_size"),

        "num_samples":
            d.get("num_samples"),

        # ---------------- quality ----------------

        "quality_metric":
            quality_metric,

        "quality_value":
            quality_value,

        "quality_higher_is_better":
            higher_is_better,

        "accuracy":
            accuracy,

        "exact_match":
            exact_match,

        "cer":
            cer,

        "wer":
            wer,

        "mean_utterance_wer":
            d.get(
                "mean_utterance_wer"
            ),

        "valid_closed_set_output_rate":
            d.get(
                "valid_closed_set_output_rate"
            ),

        # ---------------- model ----------------

        "parameters_m":
            d.get("parameters_m"),

        "artifact_size_mb":
            artifact_size_mb,

        "artifact_size_semantics":
            artifact_size_semantics,

        "weight_size_mb":
            d.get("weight_size_mb"),

        "checkpoint_size_mb":
            d.get(
                "checkpoint_size_mb"
            ),

        "checkpoint_safetensors_total_mb":
            d.get(
                "checkpoint_safetensors_total_mb"
            ),

        "model_dir_size_mb":
            d.get(
                "model_dir_size_mb"
            ),

        "model_load_time_s":
            (
                d.get(
                    "model_load_time_s"
                )
                if d.get(
                    "model_load_time_s"
                )
                is not None
                else
                d.get("load_time_s")
            ),

        # ---------------- all latency ----------------

        "all_model_mean_ms":
            stat(
                all_model_latency,
                "mean_ms"
            ),

        "all_model_p50_ms":
            stat(
                all_model_latency,
                "p50_ms"
            ),

        "all_model_p95_ms":
            stat(
                all_model_latency,
                "p95_ms"
            ),

        "all_model_p99_ms":
            stat(
                all_model_latency,
                "p99_ms"
            ),

        "all_e2e_mean_ms":
            stat(
                all_e2e_latency,
                "mean_ms"
            ),

        "all_e2e_p50_ms":
            stat(
                all_e2e_latency,
                "p50_ms"
            ),

        "all_e2e_p95_ms":
            stat(
                all_e2e_latency,
                "p95_ms"
            ),

        "all_e2e_p99_ms":
            stat(
                all_e2e_latency,
                "p99_ms"
            ),

        # ---------------- ASR fixed ----------------

        "fixed_latency_num_samples":
            (
                fixed_block.get(
                    "num_samples"
                )
                if task == "asr"
                else None
            ),

        "fixed_model_mean_ms":
            stat(
                fixed_model_latency,
                "mean_ms"
            ),

        "fixed_model_p50_ms":
            stat(
                fixed_model_latency,
                "p50_ms"
            ),

        "fixed_model_p95_ms":
            stat(
                fixed_model_latency,
                "p95_ms"
            ),

        "fixed_model_p99_ms":
            stat(
                fixed_model_latency,
                "p99_ms"
            ),

        "fixed_e2e_mean_ms":
            stat(
                fixed_e2e_latency,
                "mean_ms"
            ),

        "fixed_e2e_p50_ms":
            stat(
                fixed_e2e_latency,
                "p50_ms"
            ),

        "fixed_e2e_p95_ms":
            stat(
                fixed_e2e_latency,
                "p95_ms"
            ),

        "fixed_e2e_p99_ms":
            stat(
                fixed_e2e_latency,
                "p99_ms"
            ),

        # ---------------- primary latency ----------------

        "primary_latency_scope":
            primary_scope,

        "primary_model_mean_ms":
            stat(
                primary_model_latency,
                "mean_ms"
            ),

        "primary_model_p95_ms":
            stat(
                primary_model_latency,
                "p95_ms"
            ),

        "primary_model_p99_ms":
            stat(
                primary_model_latency,
                "p99_ms"
            ),

        "primary_e2e_mean_ms":
            stat(
                primary_e2e_latency,
                "mean_ms"
            ),

        "primary_e2e_p95_ms":
            stat(
                primary_e2e_latency,
                "p95_ms"
            ),

        "primary_e2e_p99_ms":
            stat(
                primary_e2e_latency,
                "p99_ms"
            ),

        "all_e2e_service_rate_proxy_qps":
            inverse_ms(
                stat(
                    all_e2e_latency,
                    "mean_ms"
                )
            ),

        "primary_e2e_service_rate_proxy_qps":
            inverse_ms(
                stat(
                    primary_e2e_latency,
                    "mean_ms"
                )
            ),

        # ---------------- ASR RTF ----------------

        "all_model_rtf_mean":
            rtf[
                "all_model_rtf_mean"
            ],

        "all_e2e_rtf_mean":
            rtf[
                "all_e2e_rtf_mean"
            ],

        "aggregate_model_rtf":
            rtf[
                "aggregate_model_rtf"
            ],

        "aggregate_e2e_rtf":
            rtf[
                "aggregate_e2e_rtf"
            ],

        "fixed_model_rtf_mean":
            rtf[
                "fixed_model_rtf_mean"
            ],

        "fixed_e2e_rtf_mean":
            rtf[
                "fixed_e2e_rtf_mean"
            ],

        # ---------------- task sizes ----------------

        "total_audio_duration_s":
            d.get(
                "total_audio_duration_s"
            ),

        "corpus_reference_words":
            d.get(
                "corpus_reference_words"
            ),

        "corpus_word_errors":
            d.get(
                "corpus_word_errors"
            ),

        "total_reference_characters":
            (
                d.get(
                    "total_reference_characters"
                )
                if d.get(
                    "total_reference_characters"
                )
                is not None
                else
                d.get(
                    "corpus_reference_chars"
                )
            ),

        "total_character_errors":
            (
                d.get(
                    "total_character_errors"
                )
                if d.get(
                    "total_character_errors"
                )
                is not None
                else
                d.get(
                    "corpus_char_errors"
                )
            ),

        "input_tokens_mean":
            (
                input_token_block.get("mean")
                if isinstance(
                    input_token_block,
                    dict
                )
                else None
            ),

        "output_tokens_mean":
            (
                output_token_block.get("mean")
                if isinstance(
                    output_token_block,
                    dict
                )
                else None
            ),

        # ---------------- memory ----------------

        "allocator_backend":
            allocator_backend,

        "allocator_idle_mb":
            d.get(
                "torch_idle_allocated_mb"
            ),

        "allocator_peak_mb":
            allocator_peak_mb,

        "memory_comparable_across_frameworks":
            False,

        # ---------------- environment ----------------

        "device":
            d.get("device"),

        "gpu":
            d.get("gpu"),

        "torch_version":
            d.get(
                "torch_version"
            ),

        "transformers_version":
            d.get(
                "transformers_version"
            ),

        "paddle_version":
            d.get(
                "paddle_version"
            ),

        "paddleocr_version":
            d.get(
                "paddleocr_version"
            ),

        "timestamp":
            d.get("timestamp"),

        "source_summary":
            path.name,
    }

    return record


def main():
    missing = [
        name
        for name in EXPECTED_FILES
        if not (
            FORMAL / name
        ).exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing formal summaries:\n"
            + "\n".join(missing)
        )

    records = []

    for filename in EXPECTED_FILES:
        records.append(
            build_record(
                FORMAL / filename
            )
        )

    pairs = {
        (
            x["model_key"],
            x["task"],
        )
        for x in records
    }

    if pairs != EXPECTED_PAIRS:
        print(
            "Expected but missing:",
            sorted(
                EXPECTED_PAIRS
                - pairs
            )
        )

        print(
            "Unexpected:",
            sorted(
                pairs
                - EXPECTED_PAIRS
            )
        )

        raise RuntimeError(
            "Model-task coverage mismatch."
        )

    if len(records) != 16:
        raise RuntimeError(
            f"Expected 16 records, "
            f"got {len(records)}."
        )

    # ------------------------------------------
    # JSON registry
    # ------------------------------------------

    payload = {
        "schema_version":
            "baseline-registry-v1",

        "generated_at":
            datetime.now()
            .astimezone()
            .isoformat(),

        "num_records":
            len(records),

        "primary_latency_policy": {
            "text_classification":
                "all-sample E2E latency",

            "image_classification":
                "all-sample E2E latency",

            "ocr":
                "all-sample E2E latency",

            "asr":
                (
                    "fixed [4,6) second "
                    "subset E2E latency"
                ),
        },

        "notes": [
            (
                "Quality metrics retain their "
                "native task semantics: accuracy "
                "for classification, exact match "
                "plus CER for OCR, and WER for ASR."
            ),
            (
                "primary_e2e_* uses the fixed "
                "[4,6) second subset for ASR and "
                "all samples for other tasks."
            ),
            (
                "Service-rate proxy QPS is "
                "1000 / mean latency under "
                "sequential batch-1 execution; "
                "it is not load-tested throughput."
            ),
            (
                "Allocator memory values are "
                "framework-specific diagnostics "
                "and must not be used as the final "
                "cross-framework VRAM comparison."
            ),
            (
                "Artifact/checkpoint size is disk "
                "storage metadata and is not "
                "equivalent to resident VRAM."
            ),
            (
                "Unified process-level NVML VRAM "
                "profiling is required before "
                "resource-budget deployment "
                "experiments."
            ),
        ],

        "records":
            records,
    }

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ------------------------------------------
    # CSV
    # ------------------------------------------

    fieldnames = list(
        records[0].keys()
    )

    with open(
        OUT_CSV,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(
            records
        )

    # ------------------------------------------
    # Console validation
    # ------------------------------------------

    print(
        "===== Baseline Registry ====="
    )

    print(
        "Records:",
        len(records)
    )

    print(
        "Generalist rows:",
        sum(
            x["role"] == "generalist"
            for x in records
        )
    )

    print(
        "Specialist rows:",
        sum(
            x["role"] == "specialist"
            for x in records
        )
    )

    print()

    print(
        f"{'MODEL':24} "
        f"{'ROLE':11} "
        f"{'TASK':22} "
        f"{'QUALITY':16} "
        f"{'VALUE':>10} "
        f"{'PRIMARY E2E':>14}"
    )

    print(
        "-" * 104
    )

    for x in records:

        qv = x[
            "quality_value"
        ]

        lat = x[
            "primary_e2e_mean_ms"
        ]

        qv_text = (
            f"{qv:.6f}"
            if isinstance(
                qv,
                (int, float)
            )
            else "None"
        )

        lat_text = (
            f"{lat:.3f} ms"
            if isinstance(
                lat,
                (int, float)
            )
            else "None"
        )

        print(
            f"{x['model_key'][:24]:24} "
            f"{x['role']:11} "
            f"{x['task']:22} "
            f"{x['quality_metric']:16} "
            f"{qv_text:>10} "
            f"{lat_text:>14}"
        )

    print()
    print(
        "Saved JSON:",
        OUT_JSON
    )

    print(
        "Saved CSV :",
        OUT_CSV
    )


if __name__ == "__main__":
    main()
