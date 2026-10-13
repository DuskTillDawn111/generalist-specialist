import sys
from pathlib import Path

import torch
from transformers import AutoModel


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

sys.path.insert(
    0,
    str(ROOT / "benchmark/scripts")
)

sys.path.insert(
    0,
    str(ROOT / "joint_deploy")
)


import eval_minicpm_sst2 as text_mod
import eval_minicpm_imagenette as image_mod
import eval_minicpm_iiit5k as ocr_mod
import eval_minicpm_librispeech as asr_mod

from worker_protocol import run_worker


MODEL_DIR = ROOT / "models/minicpm-o-4_5"


def instruction_from(info):
    if isinstance(info, str):
        return info

    for key in (
        "instruction",
        "prompt",
        "text",
    ):
        if key in info:
            return info[key]

    raise RuntimeError(
        f"Cannot locate instruction in: {info}"
    )


def response_from(result):
    if not isinstance(result, dict):
        raise RuntimeError(
            f"Unexpected result type: {type(result)}"
        )

    if "normalized_response" in result:
        return result["normalized_response"]

    if "raw_response" in result:
        return result["raw_response"]

    if "response" in result:
        return result["response"]

    raise RuntimeError(
        f"No response field. Keys={list(result.keys())}"
    )


def word_edit_distance(ref, hyp):
    n = len(ref)
    m = len(hyp)

    prev = list(range(m + 1))

    for i in range(1, n + 1):
        cur = [i] + [0] * m

        for j in range(1, m + 1):
            cost = (
                0
                if ref[i - 1] == hyp[j - 1]
                else 1
            )

            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )

        prev = cur

    return prev[m]


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable"
    )


device = torch.device(
    "cuda:0"
)

torch.cuda.set_device(
    device
)

torch.cuda.empty_cache()


# =====================================
# Load MiniCPM-o-4.5 exactly once
# =====================================

model = AutoModel.from_pretrained(
    MODEL_DIR,

    trust_remote_code=True,
    local_files_only=True,

    torch_dtype=torch.bfloat16,

    attn_implementation="sdpa",

    init_vision=True,
    init_audio=True,
    init_tts=False,

    low_cpu_mem_usage=True,
)

model.eval().cuda()

torch.cuda.synchronize()


# =====================================
# Formal benchmark prompts
# =====================================

text_instruction = instruction_from(
    text_mod.load_prompt()
)

image_instruction = instruction_from(
    image_mod.load_prompt()
)

ocr_instruction = instruction_from(
    ocr_mod.load_prompt()
)

asr_instruction = instruction_from(
    asr_mod.load_prompt()
)


def handler(request):
    task = request[
        "task_type"
    ]

    sample = request[
        "sample"
    ]

    # ---------------------------------
    # SST-2
    # ---------------------------------

    if task == "text":

        result = text_mod.chat_once(
            model=model,
            instruction=text_instruction,
            sentence=sample["input"],
            max_new_tokens=16,
        )

        prediction = (
            text_mod.normalize_label(
                response_from(result)
            )
        )

        reference = (
            sample["label"]
            .strip()
            .lower()
        )

        return {
            "model":
                "minicpm",

            "prediction":
                prediction,

            "reference":
                reference,

            "correct":
                prediction == reference,
        }

    # ---------------------------------
    # Imagenette
    # ---------------------------------

    if task == "image":

        result = image_mod.run_one(
            model=model,
            instruction=image_instruction,
            image_path=sample["input"],
            max_new_tokens=16,
        )

        prediction = (
            image_mod.normalize_label(
                response_from(result)
            )
        )

        reference = (
            sample["label"]
            .strip()
            .lower()
        )

        return {
            "model":
                "minicpm",

            "prediction":
                prediction,

            "reference":
                reference,

            "correct":
                prediction == reference,
        }

    # ---------------------------------
    # IIIT5K OCR
    # ---------------------------------

    if task == "ocr":

        result = ocr_mod.run_one(
            model=model,
            instruction=ocr_instruction,
            image_path=sample["input"],
            max_new_tokens=32,
        )

        prediction = (
            ocr_mod.normalize_ocr(
                response_from(result)
            )
        )

        reference = (
            ocr_mod.normalize_ocr(
                sample.get(
                    "normalized_label",
                    sample["label"]
                )
            )
        )

        return {
            "model":
                "minicpm",

            "prediction":
                prediction,

            "reference":
                reference,

            "correct":
                prediction == reference,
        }

    # ---------------------------------
    # LibriSpeech ASR
    # ---------------------------------

    if task == "asr":

        result = asr_mod.run_one(
            model=model,
            instruction=asr_instruction,
            sample=sample,
            max_new_tokens=256,
        )

        prediction = (
            asr_mod.normalize_asr(
                response_from(result)
            )
        )

        reference = (
            asr_mod.normalize_asr(
                sample.get(
                    "normalized_label",
                    sample["label"]
                )
            )
        )

        ref_words = (
            reference.split()
        )

        pred_words = (
            prediction.split()
        )

        errors = word_edit_distance(
            ref_words,
            pred_words
        )

        wer = (
            errors / len(ref_words)
            if ref_words
            else 0.0
        )

        return {
            "model":
                "minicpm",

            "prediction":
                prediction,

            "reference":
                reference,

            "word_errors":
                errors,

            "reference_words":
                len(ref_words),

            "wer":
                wer,
        }

    raise ValueError(
        f"Unsupported task: {task}"
    )


run_worker(
    worker_name="minicpm",
    handler=handler,
    ready_payload={
        "model":
            "MiniCPM-o-4.5",

        "tasks": [
            "text",
            "image",
            "ocr",
            "asr",
        ],
    }
)
