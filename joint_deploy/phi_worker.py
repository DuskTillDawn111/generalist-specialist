import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

sys.path.insert(
    0,
    str(
        ROOT / "benchmark/scripts"
    )
)

sys.path.insert(
    0,
    str(
        ROOT / "joint_deploy"
    )
)


import eval_phi4_sst2 as text_mod
import eval_phi4_imagenette as image_mod
import eval_phi4_iiit5k as ocr_mod
import eval_phi4_librispeech as asr_mod

from worker_protocol import run_worker


MODEL_DIR = text_mod.MODEL_DIR


def instruction_from(info):
    if isinstance(
        info,
        str
    ):
        return info

    for key in (
        "instruction",
        "prompt",
        "text",
    ):
        if key in info:
            return info[key]

    raise RuntimeError(
        f"Cannot locate instruction "
        f"in prompt info: {info}"
    )


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


processor = (
    AutoProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False
    )
)


model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa"
    )
)

model.eval()
model.to(
    device
)

model.config.use_cache = True

torch.cuda.synchronize()


text_instruction = (
    instruction_from(
        text_mod.load_task_prompt()
    )
)

image_instruction = (
    instruction_from(
        image_mod.load_task_prompt()
    )
)

ocr_instruction = (
    instruction_from(
        ocr_mod.load_task_prompt()
    )
)

asr_instruction = (
    instruction_from(
        asr_mod.load_task_prompt()
    )
)


image_prompt = (
    image_mod.build_chat_prompt(
        processor.tokenizer,
        image_instruction
    )
)

ocr_prompt = (
    ocr_mod.build_chat_prompt(
        processor.tokenizer,
        ocr_instruction
    )
)

asr_prompt = (
    asr_mod.build_chat_prompt(
        processor.tokenizer,
        asr_instruction
    )
)


def response_from(result):
    if not isinstance(result, dict):
        raise RuntimeError(
            f"Unexpected Phi result: {type(result)}"
        )

    if "response" in result:
        return result["response"]

    if "normalized_response" in result:
        return result["normalized_response"]

    raise RuntimeError(
        "Phi result contains neither "
        "'response' nor 'normalized_response'. "
        f"Keys: {list(result.keys())}"
    )
def handler(request):
    task = request[
        "task_type"
    ]

    sample = request[
        "sample"
    ]

    if task == "text":
        result = text_mod.generate_one(
            model=model,
            processor=processor,
            instruction=text_instruction,
            sentence=sample["input"],
            device=device,
            max_new_tokens=16
        )

        prediction = (
            text_mod
            .normalize_closed_label(
                response_from(
                    result
                )
            )
        )

        reference = (
            sample["label"]
            .strip()
            .lower()
        )

        return {
            "model":
                "phi4",

            "prediction":
                prediction,

            "reference":
                reference,

            "correct":
                prediction
                == reference,
        }

    if task == "image":
        result = image_mod.generate_one(
            model=model,
            processor=processor,
            prompt=image_prompt,
            image_path=sample["input"],
            device=device,
            max_new_tokens=16
        )

        prediction = (
            image_mod
            .normalize_closed_label(
                response_from(
                    result
                )
            )
        )

        reference = (
            sample["label"]
            .strip()
            .lower()
        )

        return {
            "model":
                "phi4",

            "prediction":
                prediction,

            "reference":
                reference,

            "correct":
                prediction
                == reference,
        }

    if task == "ocr":
        result = ocr_mod.generate_one(
            model=model,
            processor=processor,
            prompt=ocr_prompt,
            image_path=sample["input"],
            device=device,
            max_new_tokens=32
        )

        prediction = (
            ocr_mod.normalize_ocr(
                response_from(
                    result
                )
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
                "phi4",

            "prediction":
                prediction,

            "reference":
                reference,

            "correct":
                prediction
                == reference,
        }

    if task == "asr":
        result = asr_mod.generate_one(
            model=model,
            processor=processor,
            prompt=asr_prompt,
            audio_path=sample["input"],
            device=device,
            max_new_tokens=256
        )

        prediction = (
            asr_mod.normalize_asr(
                response_from(
                    result
                )
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

        errors = (
            asr_mod
            .levenshtein_words(
                ref_words,
                pred_words
            )
        )

        wer = (
            errors
            / len(ref_words)
            if ref_words
            else 0.0
        )

        return {
            "model":
                "phi4",

            "prediction":
                prediction,

            "reference":
                reference,

            "word_errors":
                errors,

            "reference_words":
                len(
                    ref_words
                ),

            "wer":
                wer,
        }

    raise ValueError(
        f"Unsupported task: {task}"
    )


run_worker(
    worker_name="phi4",
    handler=handler,
    ready_payload={
        "model":
            "Phi-4-multimodal-instruct",

        "tasks": [
            "text",
            "image",
            "ocr",
            "asr",
        ],
    }
)
