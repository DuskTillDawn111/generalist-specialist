import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)


ROOT = Path("/root/autodl-tmp/generalist-specialist")

MODEL_DIR = ROOT / "models/whisper-small"

MANIFEST = (
    ROOT
    / "benchmark/manifests/asr.jsonl"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/asr_full.jsonl"
)

RESULT_DIR = (
    ROOT
    / "benchmark/results/formal"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "whisper_small_librispeech_summary.json"
)

PREDICTION_PATH = (
    RESULT_DIR
    / "whisper_small_librispeech_predictions.jsonl"
)


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def read_manifest(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def normalize_asr(text):
    """
    与当前 benchmark/evaluation/metrics.py
    的 ASR normalization 保持同一原则：

    - lowercase
    - 保留 apostrophe
    - 去除其他 punctuation
    - normalize whitespace
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9'\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def levenshtein_words(ref_words, hyp_words):
    n = len(ref_words)
    m = len(hyp_words)

    prev = list(range(m + 1))

    for i in range(1, n + 1):

        curr = [i] + [0] * m

        for j in range(1, m + 1):

            cost = (
                0
                if ref_words[i - 1]
                == hyp_words[j - 1]
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

    if len(values) == 0:
        return None

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


def rtf_stats(values):
    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(values) == 0:
        return None

    return {
        "mean":
            round(
                float(np.mean(values)),
                6
            ),

        "p50":
            round(
                percentile(values, 50),
                6
            ),

        "p95":
            round(
                percentile(values, 95),
                6
            ),

        "p99":
            round(
                percentile(values, 99),
                6
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


def load_audio(path):
    audio, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False
    )

    if audio.ndim == 2:
        audio = audio.mean(
            axis=1
        )

    if sr != 16000:
        raise ValueError(
            f"Unexpected sample rate "
            f"{sr} for {path}; "
            f"expected 16000 Hz."
        )

    return audio, sr


def prepare_inputs(
    processor,
    audio,
    sample_rate,
    duration_s,
    device
):
    """
    <= 30 s:
        Whisper short-form

    > 30 s:
        保留完整音频，
        交给 HF Whisper long-form generation。
    """

    if duration_s <= 30.0:

        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True
        )

        long_form = False

    else:

        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=False,
            padding="longest"
        )

        long_form = True

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    return inputs, long_form


def generate_transcription(
    model,
    inputs,
    long_form
):
    kwargs = {
        "language": "en",
        "task": "transcribe",
        "do_sample": False,
    }

    # >30 s 时明确使用 Whisper long-form timestamps。
    if long_form:
        kwargs["return_timestamps"] = True

    generated_ids = model.generate(
        **inputs,
        **kwargs
    )

    return generated_ids


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--warmup",
        type=int,
        default=3
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means evaluate all 200 samples"
    )

    parser.add_argument(
        "--latency-min-duration",
        type=float,
        default=4.0
    )

    parser.add_argument(
        "--latency-max-duration",
        type=float,
        default=6.0
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

    if args.limit > 0:
        samples = samples[
            :args.limit
        ]

    print(
        "Evaluation samples:",
        len(samples)
    )

    print(
        "Latency bucket:",
        f"[{args.latency_min_duration}, "
        f"{args.latency_max_duration}) s"
    )

    # --------------------------------
    # Processor
    # --------------------------------

    processor_start = (
        time.perf_counter()
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_DIR,
            local_files_only=True
        )
    )

    processor_load_time_s = (
        time.perf_counter()
        - processor_start
    )

    print(
        "Sampling rate:",
        processor.feature_extractor.sampling_rate
    )

    # --------------------------------
    # Model
    # --------------------------------

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
    model.to(device)

    # 清理旧 checkpoint 中的 forced_decoder_ids，
    # 正式实验统一显式使用 language/task。
    if hasattr(
        model.config,
        "forced_decoder_ids"
    ):
        model.config.forced_decoder_ids = None

    if hasattr(
        model.generation_config,
        "forced_decoder_ids"
    ):
        model.generation_config.forced_decoder_ids = None

    sync()

    model_load_time_s = (
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

    dtype = str(
        next(
            model.parameters()
        ).dtype
    ).replace(
        "torch.",
        ""
    )

    print(
        f"Parameters: "
        f"{parameters_m:.3f} M"
    )

    print(
        "Model dtype:",
        dtype
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

        duration = float(
            candidate["duration_s"]
        )

        if (
            candidate["id"]
            not in benchmark_ids
            and 4.0 <= duration < 6.0
        ):
            warmup_sample = candidate
            break

    if warmup_sample is None:
        raise RuntimeError(
            "Could not find a separate "
            "4-6 second warmup sample."
        )

    print(
        "Full-pipeline warmup runs:",
        args.warmup
    )

    print(
        "Warmup audio:",
        warmup_sample["input"]
    )

    print(
        "Warmup duration:",
        warmup_sample["duration_s"]
    )

    with torch.inference_mode():

        for _ in range(
            args.warmup
        ):

            audio, sr = load_audio(
                warmup_sample["input"]
            )

            inputs, long_form = (
                prepare_inputs(
                    processor,
                    audio,
                    sr,
                    float(
                        warmup_sample[
                            "duration_s"
                        ]
                    ),
                    device
                )
            )

            generated_ids = (
                generate_transcription(
                    model,
                    inputs,
                    long_form
                )
            )

            sync()

            _ = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            del inputs
            del generated_ids

    torch.cuda.reset_peak_memory_stats(
        device
    )

    # --------------------------------
    # Formal evaluation
    # --------------------------------

    predictions = []

    all_model_latencies = []
    all_e2e_latencies = []

    all_model_rtfs = []
    all_e2e_rtfs = []

    bucket_model_latencies = []
    bucket_e2e_latencies = []

    bucket_model_rtfs = []
    bucket_e2e_rtfs = []

    total_word_errors = 0
    total_reference_words = 0

    utterance_wers = []

    long_form_count = 0

    total_audio_duration_s = 0.0
    total_model_time_s = 0.0
    total_e2e_time_s = 0.0

    print()
    print(
        "Running evaluation..."
    )

    with torch.inference_mode():

        for i, sample in enumerate(
            samples,
            start=1
        ):

            duration_s = float(
                sample["duration_s"]
            )

            total_audio_duration_s += (
                duration_s
            )

            sync()

            e2e_start = (
                time.perf_counter()
            )

            # -------------------------
            # Local FLAC read/decode
            # -------------------------

            audio, sr = load_audio(
                sample["input"]
            )

            # -------------------------
            # Feature extraction + H2D
            # -------------------------

            inputs, long_form = (
                prepare_inputs(
                    processor,
                    audio,
                    sr,
                    duration_s,
                    device
                )
            )

            if long_form:
                long_form_count += 1

            sync()

            # -------------------------
            # Model generate latency
            # -------------------------

            model_start = (
                time.perf_counter()
            )

            generated_ids = (
                generate_transcription(
                    model,
                    inputs,
                    long_form
                )
            )

            sync()

            model_end = (
                time.perf_counter()
            )

            # -------------------------
            # Decode + normalization
            # -------------------------

            prediction = (
                processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
            )

            normalized_prediction = (
                normalize_asr(
                    prediction
                )
            )

            normalized_reference = (
                sample[
                    "normalized_label"
                ]
            )

            # 保证 benchmark normalization
            # 与 prediction 同一规则。
            normalized_reference = (
                normalize_asr(
                    normalized_reference
                )
            )

            e2e_end = (
                time.perf_counter()
            )

            # -------------------------
            # WER
            # -------------------------

            ref_words = (
                normalized_reference.split()
            )

            hyp_words = (
                normalized_prediction.split()
            )

            errors = (
                levenshtein_words(
                    ref_words,
                    hyp_words
                )
            )

            ref_word_count = len(
                ref_words
            )

            utterance_wer = (
                errors
                / ref_word_count
                if ref_word_count > 0
                else 0.0
            )

            total_word_errors += (
                errors
            )

            total_reference_words += (
                ref_word_count
            )

            utterance_wers.append(
                utterance_wer
            )

            # -------------------------
            # Timing
            # -------------------------

            model_latency_ms = (
                model_end
                - model_start
            ) * 1000

            e2e_latency_ms = (
                e2e_end
                - e2e_start
            ) * 1000

            model_time_s = (
                model_latency_ms
                / 1000.0
            )

            e2e_time_s = (
                e2e_latency_ms
                / 1000.0
            )

            total_model_time_s += (
                model_time_s
            )

            total_e2e_time_s += (
                e2e_time_s
            )

            model_rtf = (
                model_time_s
                / duration_s
            )

            e2e_rtf = (
                e2e_time_s
                / duration_s
            )

            all_model_latencies.append(
                model_latency_ms
            )

            all_e2e_latencies.append(
                e2e_latency_ms
            )

            all_model_rtfs.append(
                model_rtf
            )

            all_e2e_rtfs.append(
                e2e_rtf
            )

            in_latency_bucket = (
                args.latency_min_duration
                <= duration_s
                < args.latency_max_duration
            )

            if in_latency_bucket:

                bucket_model_latencies.append(
                    model_latency_ms
                )

                bucket_e2e_latencies.append(
                    e2e_latency_ms
                )

                bucket_model_rtfs.append(
                    model_rtf
                )

                bucket_e2e_rtfs.append(
                    e2e_rtf
                )

            output_tokens = int(
                generated_ids.shape[-1]
            )

            record = {
                "id":
                    sample["id"],

                "utterance_id":
                    sample[
                        "utterance_id"
                    ],

                "task":
                    "asr",

                "dataset":
                    "librispeech",

                "split":
                    "test-clean",

                "input":
                    sample["input"],

                "duration_s":
                    duration_s,

                "sample_rate":
                    int(
                        sample[
                            "sample_rate"
                        ]
                    ),

                "reference":
                    sample["label"],

                "normalized_reference":
                    normalized_reference,

                "prediction":
                    prediction,

                "normalized_prediction":
                    normalized_prediction,

                "word_errors":
                    errors,

                "reference_words":
                    ref_word_count,

                "wer":
                    round(
                        utterance_wer,
                        6
                    ),

                "output_tokens":
                    output_tokens,

                "long_form":
                    long_form,

                "latency_bucket":
                    in_latency_bucket,

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

                "model_rtf":
                    round(
                        model_rtf,
                        6
                    ),

                "e2e_rtf":
                    round(
                        e2e_rtf,
                        6
                    ),
            }

            predictions.append(
                record
            )

            print(
                f"[{i:03d}/"
                f"{len(samples):03d}] "
                f"dur={duration_s:5.2f}s "
                f"WER={utterance_wer:5.3f} "
                f"model={model_latency_ms:7.2f}ms "
                f"e2e={e2e_latency_ms:7.2f}ms "
                f"RTF={e2e_rtf:.4f} "
                f"{'LONG' if long_form else ''}"
            )

            del inputs
            del generated_ids

    # --------------------------------
    # Summary
    # --------------------------------

    corpus_wer = (
        total_word_errors
        / total_reference_words
        if total_reference_words > 0
        else 0.0
    )

    mean_utterance_wer = float(
        np.mean(
            utterance_wers
        )
    )

    all_model_stats = stats(
        all_model_latencies
    )

    all_e2e_stats = stats(
        all_e2e_latencies
    )

    bucket_model_stats = stats(
        bucket_model_latencies
    )

    bucket_e2e_stats = stats(
        bucket_e2e_latencies
    )

    aggregate_model_rtf = (
        total_model_time_s
        / total_audio_duration_s
    )

    aggregate_e2e_rtf = (
        total_e2e_time_s
        / total_audio_duration_s
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
            "automatic_speech_recognition",

        "dataset":
            "LibriSpeech",

        "split":
            "test-clean",

        "language":
            "en",

        "generation_task":
            "transcribe",

        "precision":
            dtype,

        "batch_size":
            1,

        "num_samples":
            len(samples),

        "total_audio_duration_s":
            round(
                total_audio_duration_s,
                3
            ),

        "long_form_samples":
            long_form_count,

        "corpus_word_errors":
            total_word_errors,

        "corpus_reference_words":
            total_reference_words,

        "wer":
            round(
                corpus_wer,
                6
            ),

        "mean_utterance_wer":
            round(
                mean_utterance_wer,
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

        "processor_load_time_s":
            round(
                processor_load_time_s,
                3
            ),

        "model_load_time_s":
            round(
                model_load_time_s,
                3
            ),

        "warmup_runs":
            args.warmup,

        "all_samples": {
            "model_latency":
                all_model_stats,

            "e2e_latency":
                all_e2e_stats,

            "model_rtf_distribution":
                rtf_stats(
                    all_model_rtfs
                ),

            "e2e_rtf_distribution":
                rtf_stats(
                    all_e2e_rtfs
                ),

            "aggregate_model_rtf":
                round(
                    aggregate_model_rtf,
                    6
                ),

            "aggregate_e2e_rtf":
                round(
                    aggregate_e2e_rtf,
                    6
                ),
        },

        "fixed_duration_latency": {
            "range":
                (
                    f"[{args.latency_min_duration}, "
                    f"{args.latency_max_duration}) seconds"
                ),

            "num_samples":
                len(
                    bucket_model_latencies
                ),

            "model_latency":
                bucket_model_stats,

            "e2e_latency":
                bucket_e2e_stats,

            "model_rtf_distribution":
                rtf_stats(
                    bucket_model_rtfs
                ),

            "e2e_rtf_distribution":
                rtf_stats(
                    bucket_e2e_rtfs
                ),
        },

        "torch_peak_allocated_mb":
            round(
                peak_allocated_mb,
                3
            ),

        "normalization":
            (
                "lowercase; remove punctuation "
                "except apostrophes; "
                "normalize whitespace"
            ),

        "latency_definition": {
            "model_latency":
                (
                    "Whisper generate only, "
                    "after audio decoding, "
                    "feature extraction and H2D"
                ),

            "e2e_latency":
                (
                    "local FLAC read/decode + "
                    "feature extraction + "
                    "host-to-device transfer + "
                    "Whisper generation + "
                    "text decoding + "
                    "ASR normalization"
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
        "Total audio:",
        round(
            total_audio_duration_s,
            3
        ),
        "s"
    )

    print(
        "Long-form samples:",
        long_form_count
    )

    print(
        "Corpus WER:",
        f"{corpus_wer:.4f}"
    )

    print(
        "Mean utterance WER:",
        f"{mean_utterance_wer:.4f}"
    )

    print()
    print(
        "All model latency:",
        all_model_stats
    )

    print(
        "All E2E latency:",
        all_e2e_stats
    )

    print(
        "Aggregate model RTF:",
        f"{aggregate_model_rtf:.6f}"
    )

    print(
        "Aggregate E2E RTF:",
        f"{aggregate_e2e_rtf:.6f}"
    )

    print()
    print(
        f"Fixed-duration "
        f"[{args.latency_min_duration}, "
        f"{args.latency_max_duration}) s "
        f"samples:",
        len(
            bucket_model_latencies
        )
    )

    print(
        "Bucket model latency:",
        bucket_model_stats
    )

    print(
        "Bucket E2E latency:",
        bucket_e2e_stats
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
