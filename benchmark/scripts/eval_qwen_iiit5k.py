import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import transformers

from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

from qwen_omni_utils import process_mm_info


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

MODEL_DIR = (
    ROOT / "models/qwen2.5-omni-7b"
)

MANIFEST = (
    ROOT / "benchmark/manifests/ocr.jsonl"
)

FULL_MANIFEST = (
    ROOT / "benchmark/manifests/ocr_full.jsonl"
)

PROMPT_FILE = (
    ROOT / "benchmark/prompts/task_prompts.json"
)

RESULT_DIR = (
    ROOT / "benchmark/results/formal"
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


def load_prompt():
    with open(
        PROMPT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        prompts = json.load(f)

    return prompts["ocr"]


def normalize_ocr(text):
    return re.sub(
        r"[^a-z0-9]",
        "",
        text.lower()
    )


def levenshtein_chars(
    reference,
    prediction
):
    prev = list(
        range(
            len(prediction) + 1
        )
    )

    for i, rc in enumerate(
        reference,
        start=1
    ):
        curr = [i] + (
            [0] * len(prediction)
        )

        for j, pc in enumerate(
            prediction,
            start=1
        ):
            cost = (
                0
                if rc == pc
                else 1
            )

            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )

        prev = curr

    return prev[-1]


def sync():
    torch.cuda.synchronize()


def latency_stats(values):
    arr = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "mean_ms":
            round(
                float(np.mean(arr)),
                3
            ),

        "p50_ms":
            round(
                float(
                    np.percentile(
                        arr,
                        50
                    )
                ),
                3
            ),

        "p95_ms":
            round(
                float(
                    np.percentile(
                        arr,
                        95
                    )
                ),
                3
            ),

        "p99_ms":
            round(
                float(
                    np.percentile(
                        arr,
                        99
                    )
                ),
                3
            ),

        "min_ms":
            round(
                float(np.min(arr)),
                3
            ),

        "max_ms":
            round(
                float(np.max(arr)),
                3
            ),
    }


def checkpoint_size_mb(path):
    total = 0

    for p in path.glob(
        "*.safetensors"
    ):
        total += p.stat().st_size

    return total / (1024 ** 2)


def run_one(
    model,
    processor,
    instruction,
    sample,
    max_new_tokens
):
    # ----------------------------------
    # E2E starts before image loading.
    #
    # process_mm_info handles local
    # image read/decode.
    # ----------------------------------

    sync()

    e2e_start = (
        time.perf_counter()
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["input"],
                },
                {
                    "type": "text",
                    "text": instruction,
                },
            ],
        }
    ]

    text = (
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    audios, images, videos = (
        process_mm_info(
            messages,
            use_audio_in_video=False,
        )
    )

    if images is None:
        raise RuntimeError(
            "No image returned by "
            "process_mm_info()."
        )

    if len(images) != 1:
        raise RuntimeError(
            "Expected exactly one image, "
            f"got {len(images)}."
        )

    image_width = None
    image_height = None

    if hasattr(
        images[0],
        "size"
    ):
        image_width = int(
            images[0].size[0]
        )

        image_height = int(
            images[0].size[1]
        )

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )

    if (
        "pixel_values" not in inputs
        or
        "image_grid_thw" not in inputs
    ):
        raise RuntimeError(
            "Vision processor outputs "
            "are missing."
        )

    pixel_values_shape = list(
        inputs[
            "pixel_values"
        ].shape
    )

    image_grid_thw = (
        inputs[
            "image_grid_thw"
        ]
        .detach()
        .cpu()
        .tolist()
    )

    inputs = inputs.to("cuda")

    input_length = int(
        inputs[
            "input_ids"
        ].shape[1]
    )

    # ----------------------------------
    # Model latency:
    # generate only after preprocessing
    # and H2D.
    #
    # Vision encoding is included.
    # ----------------------------------

    sync()

    model_start = (
        time.perf_counter()
    )

    output = model.generate(
        **inputs,
        thinker_max_new_tokens=max_new_tokens,
        do_sample=False,
        return_audio=False,
    )

    sync()

    model_end = (
        time.perf_counter()
    )

    if not isinstance(
        output,
        torch.Tensor
    ):
        raise TypeError(
            "Expected Tensor output, "
            f"got {type(output)}"
        )

    if output.shape[1] < input_length:
        raise RuntimeError(
            "Generated sequence shorter "
            "than input sequence."
        )

    generated_ids = (
        output[
            :,
            input_length:
        ]
    )

    response = (
        processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    )

    normalized = (
        normalize_ocr(
            response
        )
    )

    e2e_end = (
        time.perf_counter()
    )

    return {
        "raw_response":
            response,

        "normalized_response":
            normalized,

        "input_tokens":
            input_length,

        "output_tokens":
            int(
                generated_ids.shape[1]
            ),

        "image_width":
            image_width,

        "image_height":
            image_height,

        "pixel_values_shape":
            pixel_values_shape,

        "image_grid_thw":
            image_grid_thw,

        "model_latency_ms":
            (
                model_end
                - model_start
            ) * 1000,

        "e2e_latency_ms":
            (
                e2e_end
                - e2e_start
            ) * 1000,
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
        default=0
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    torch.cuda.set_device(0)

    print("Device: cuda:0")
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
    print(
        "Precision: bfloat16"
    )
    print(
        "enable_audio_output=False"
    )
    print(
        "return_audio=False"
    )
    print(
        "use_audio_in_video=False"
    )
    print(
        "Decoding: do_sample=False"
    )
    print(
        "Max new tokens:",
        args.max_new_tokens
    )

    # ----------------------------------
    # Prompt
    # ----------------------------------

    prompt_info = load_prompt()

    instruction = (
        prompt_info["prompt"]
    )

    print(
        "Prompt version:",
        prompt_info["version"]
    )

    # ----------------------------------
    # Dataset
    # ----------------------------------

    all_samples = read_manifest(
        MANIFEST
    )

    samples = list(
        all_samples
    )

    if args.limit > 0:
        samples = samples[
            :args.limit
        ]

    print(
        "Evaluation samples:",
        len(samples)
    )

    # ----------------------------------
    # Processor
    # ----------------------------------

    print(
        "Loading processor..."
    )

    processor = (
        Qwen2_5OmniProcessor
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
    )

    # ----------------------------------
    # Model
    # ----------------------------------

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(
        "Loading Qwen2.5-Omni-7B..."
    )

    load_start = (
        time.perf_counter()
    )

    model = (
        Qwen2_5OmniForConditionalGeneration
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            enable_audio_output=False,
        )
    )

    model.eval()

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

    actual_dtype = str(
        next(
            model.parameters()
        ).dtype
    ).replace(
        "torch.",
        ""
    )

    idle_allocated_mb = (
        torch.cuda.memory_allocated()
        / 1024**2
    )

    print(
        f"Parameters: "
        f"{parameters_m:.3f} M"
    )

    print(
        "Actual dtype:",
        actual_dtype
    )

    print(
        f"Model load time: "
        f"{model_load_time_s:.3f} s"
    )

    print(
        f"CUDA allocated after load: "
        f"{idle_allocated_mb:.3f} MB"
    )

    # ----------------------------------
    # Separate warmup sample
    # ----------------------------------

    benchmark_ids = {
        x["id"]
        for x in all_samples
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
            "No separate warmup "
            "sample found."
        )

    print(
        "Full-pipeline warmup runs:",
        args.warmup
    )

    print(
        "Warmup image:",
        warmup_sample["input"]
    )

    with torch.inference_mode():
        for _ in range(
            args.warmup
        ):
            _ = run_one(
                model=model,
                processor=processor,
                instruction=instruction,
                sample=warmup_sample,
                max_new_tokens=(
                    args.max_new_tokens
                ),
            )

    torch.cuda.reset_peak_memory_stats()

    # ----------------------------------
    # Evaluation
    # ----------------------------------

    predictions = []

    model_latencies = []
    e2e_latencies = []

    input_tokens = []
    output_tokens = []

    exact_correct = 0

    total_char_errors = 0
    total_reference_chars = 0

    print()
    print(
        "Running evaluation..."
    )

    with torch.inference_mode():

        for i, sample in enumerate(
            samples,
            start=1
        ):

            result = run_one(
                model=model,
                processor=processor,
                instruction=instruction,
                sample=sample,
                max_new_tokens=(
                    args.max_new_tokens
                ),
            )

            gt = normalize_ocr(
                sample["label"]
            )

            pred = result[
                "normalized_response"
            ]

            is_exact = (
                pred == gt
            )

            if is_exact:
                exact_correct += 1

            char_errors = (
                levenshtein_chars(
                    gt,
                    pred
                )
            )

            ref_chars = len(gt)

            total_char_errors += (
                char_errors
            )

            total_reference_chars += (
                ref_chars
            )

            sample_cer = (
                char_errors
                / ref_chars
                if ref_chars > 0
                else 0.0
            )

            model_ms = result[
                "model_latency_ms"
            ]

            e2e_ms = result[
                "e2e_latency_ms"
            ]

            model_latencies.append(
                model_ms
            )

            e2e_latencies.append(
                e2e_ms
            )

            input_tokens.append(
                result[
                    "input_tokens"
                ]
            )

            output_tokens.append(
                result[
                    "output_tokens"
                ]
            )

            predictions.append(
                {
                    "id":
                        sample["id"],

                    "task":
                        "ocr",

                    "dataset":
                        "IIIT5K",

                    "split":
                        "test",

                    "input":
                        sample["input"],

                    "reference":
                        sample["label"],

                    "normalized_reference":
                        gt,

                    "raw_response":
                        result[
                            "raw_response"
                        ],

                    "normalized_prediction":
                        pred,

                    "exact_match":
                        is_exact,

                    "char_errors":
                        char_errors,

                    "reference_chars":
                        ref_chars,

                    "cer":
                        round(
                            sample_cer,
                            6
                        ),

                    "input_tokens":
                        result[
                            "input_tokens"
                        ],

                    "output_tokens":
                        result[
                            "output_tokens"
                        ],

                    "processed_image_width":
                        result[
                            "image_width"
                        ],

                    "processed_image_height":
                        result[
                            "image_height"
                        ],

                    "pixel_values_shape":
                        result[
                            "pixel_values_shape"
                        ],

                    "image_grid_thw":
                        result[
                            "image_grid_thw"
                        ],

                    "model_latency_ms":
                        round(
                            model_ms,
                            3
                        ),

                    "e2e_latency_ms":
                        round(
                            e2e_ms,
                            3
                        ),
                }
            )

            print(
                f"[{i:03d}/"
                f"{len(samples):03d}] "
                f"GT={gt:<20} "
                f"Pred={pred[:20]:<20} "
                f"{'OK' if is_exact else 'WRONG'} "
                f"err={char_errors:2d} "
                f"in={result['input_tokens']:3d} "
                f"out={result['output_tokens']:2d} "
                f"model={model_ms:8.2f}ms "
                f"e2e={e2e_ms:8.2f}ms"
            )

    # ----------------------------------
    # Metrics
    # ----------------------------------

    exact_match = (
        exact_correct
        / len(samples)
    )

    corpus_cer = (
        total_char_errors
        / total_reference_chars
        if total_reference_chars > 0
        else 0.0
    )

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

    peak_allocated_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    summary = {
        "model":
            "Qwen2.5-Omni-7B",

        "task":
            "ocr",

        "dataset":
            "IIIT5K",

        "split":
            "test",

        "model_role":
            "generalist",

        "precision":
            actual_dtype,

        "batch_size":
            1,

        "audio_output_enabled":
            False,

        "generation": {
            "do_sample":
                False,

            "return_audio":
                False,

            "max_new_tokens":
                args.max_new_tokens,
        },

        "multimodal_processing": {
            "qwen_omni_utils":
                True,

            "qwen_omni_utils_version":
                "0.0.9",

            "use_audio_in_video":
                False,

            "required_vision_inputs": [
                "pixel_values",
                "image_grid_thw",
            ],
        },

        "prompt_version":
            prompt_info["version"],

        "prompt":
            instruction,

        "normalization":
            (
                "lowercase + remove all "
                "non-alphanumeric characters"
            ),

        "generated_token_extraction":
            (
                "output[:, input_length:] "
                "before batch_decode"
            ),

        "num_samples":
            len(samples),

        "exact_correct":
            exact_correct,

        "exact_match":
            round(
                exact_match,
                6
            ),

        "corpus_char_errors":
            total_char_errors,

        "corpus_reference_chars":
            total_reference_chars,

        "cer":
            round(
                corpus_cer,
                6
            ),

        "input_tokens": {
            "mean":
                round(
                    float(
                        np.mean(
                            input_tokens
                        )
                    ),
                    3
                ),

            "min":
                int(
                    min(input_tokens)
                ),

            "max":
                int(
                    max(input_tokens)
                ),
        },

        "output_tokens": {
            "total":
                int(
                    sum(output_tokens)
                ),

            "mean":
                round(
                    float(
                        np.mean(
                            output_tokens
                        )
                    ),
                    3
                ),

            "min":
                int(
                    min(output_tokens)
                ),

            "max":
                int(
                    max(output_tokens)
                ),
        },

        "parameters_m":
            round(
                parameters_m,
                3
            ),

        "checkpoint_safetensors_total_mb":
            round(
                checkpoint_size_mb(
                    MODEL_DIR
                ),
                3
            ),

        "checkpoint_note":
            (
                "Checkpoint disk size includes "
                "components not necessarily "
                "loaded when "
                "enable_audio_output=False."
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

        "model_request_service_rate_proxy_qps":
            round(
                1000.0
                / model_stats[
                    "mean_ms"
                ],
                3
            ),

        "e2e_request_service_rate_proxy_qps":
            round(
                1000.0
                / e2e_stats[
                    "mean_ms"
                ],
                3
            ),

        "throughput_note":
            (
                "Request QPS values are "
                "sequential batch-1 "
                "service-rate proxies "
                "(1000 / mean latency), "
                "not load-tested concurrent "
                "throughput."
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
                "Torch allocator memory is "
                "framework-specific auxiliary "
                "data; final deployment VRAM "
                "comparison will use "
                "process-level NVML."
            ),

        "latency_definition": {
            "model_latency":
                (
                    "synchronized "
                    "model.generate() after "
                    "image preprocessing and "
                    "H2D; includes vision "
                    "encoding and autoregressive "
                    "text generation"
                ),

            "e2e_latency":
                (
                    "message construction + "
                    "local image read/decode "
                    "inside process_mm_info + "
                    "chat template + multimodal "
                    "processor + H2D + "
                    "model.generate() + "
                    "new-token slicing + decode "
                    "+ OCR normalization"
                ),
        },

        "device":
            "cuda:0",

        "gpu":
            torch.cuda.get_device_name(0),

        "torch_version":
            torch.__version__,

        "transformers_version":
            transformers.__version__,

        "timestamp":
            datetime.now()
            .astimezone()
            .isoformat(),
    }

    # ----------------------------------
    # Save
    # ----------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.limit > 0:
        prediction_path = (
            RESULT_DIR
            / "qwen_iiit5k_smoke_predictions.jsonl"
        )

        summary_path = (
            RESULT_DIR
            / "qwen_iiit5k_smoke_summary.json"
        )

    else:
        prediction_path = (
            RESULT_DIR
            / "qwen_iiit5k_predictions.jsonl"
        )

        summary_path = (
            RESULT_DIR
            / "qwen_iiit5k_summary.json"
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
        "========== SUMMARY =========="
    )

    print(
        "Samples:",
        len(samples)
    )

    print(
        "Exact correct:",
        exact_correct
    )

    print(
        "Exact match:",
        f"{exact_match:.4f}"
    )

    print(
        "Corpus character errors:",
        total_char_errors
    )

    print(
        "Corpus reference chars:",
        total_reference_chars
    )

    print(
        "CER:",
        f"{corpus_cer:.4f}"
    )

    print(
        "Input tokens mean:",
        round(
            float(
                np.mean(
                    input_tokens
                )
            ),
            3
        )
    )

    print(
        "Output tokens total:",
        sum(output_tokens)
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
        "Model service-rate proxy:",
        summary[
            "model_request_service_rate_proxy_qps"
        ],
        "qps"
    )

    print(
        "E2E service-rate proxy:",
        summary[
            "e2e_request_service_rate_proxy_qps"
        ],
        "qps"
    )

    print()
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
