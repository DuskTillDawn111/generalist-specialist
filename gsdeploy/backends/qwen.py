import os
import os
import inspect
import sys
from pathlib import Path
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

ROOT = Path(os.environ.get("GS_ASSET_ROOT", ".")).resolve()
from gsdeploy.backends.tasks import qwen_sst2 as text_mod
from gsdeploy.backends.tasks import qwen_imagenette as image_mod
from gsdeploy.backends.tasks import qwen_iiit5k as ocr_mod
from gsdeploy.backends.tasks import qwen_librispeech as asr_mod
from gsdeploy.backends.tasks import qwen_boolq as text_qa_mod
from gsdeploy.backends.tasks import qwen_scienceqa as visual_qa_mod
from gsdeploy.backends.tasks import qwen_voxlingua as lid_mod
from gsdeploy.runtime.protocol import run_worker

MODEL_DIR = Path(os.environ["GS_MODEL_PATH"])


def instruction_from(info):
    if isinstance(info, str):
        return info
    for key in ("instruction", "prompt", "text"):
        if key in info:
            return info[key]
    raise RuntimeError(f"Cannot locate instruction in: {info}")


if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")
torch.cuda.set_device(torch.device("cuda:0"))
torch.cuda.empty_cache()
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    enable_audio_output=False,
)
model.eval()
torch.cuda.synchronize()
text_task_prompt = instruction_from(text_mod.load_prompt())
text_qa_task_prompt = instruction_from(text_qa_mod.load_prompt())
image_instruction = instruction_from(image_mod.load_prompt())
ocr_instruction = instruction_from(ocr_mod.load_prompt())
asr_instruction = instruction_from(asr_mod.load_prompt())
visual_qa_task_prompt = instruction_from(visual_qa_mod.load_prompt())
lid_instruction = instruction_from(lid_mod.load_prompt())
TEXT_RUN_PARAMS = set(inspect.signature(text_mod.run_one).parameters)


def run_text(sample):
    return text_mod.run_one(
        model=model,
        processor=processor,
        task_prompt=text_task_prompt,
        sample=sample,
        max_new_tokens=16,
    )


_PROFILE_CAPTURE_STARTED = False


def maybe_start_resource_profile(request):
    global _PROFILE_CAPTURE_STARTED
    if request.get("task_type") != "__profile_start__":
        return None
    if os.environ.get("QWEN_NSYS_CAPTURE", "0") != "1":
        return {"profile_started": False, "reason": "QWEN_NSYS_CAPTURE disabled"}
    if _PROFILE_CAPTURE_STARTED:
        return {"profile_started": True, "already_started": True}
    torch.cuda.synchronize()
    rc = torch.cuda.cudart().cudaProfilerStart()
    _PROFILE_CAPTURE_STARTED = True
    print(f"[PROFILE] cudaProfilerStart control rc={rc}", flush=True)
    return {"profile_started": True, "cuda_profiler_rc": str(rc)}


def handler(request):
    profile_control = maybe_start_resource_profile(request)
    if profile_control is not None:
        return profile_control
    task = request["task_type"]
    sample = request["sample"]
    if task == "text":
        result = run_text(sample)
        prediction = text_mod.normalize_label(result["normalized_response"])
        reference = sample["label"].strip().lower()
        return {
            "model": "qwen",
            "prediction": prediction,
            "reference": reference,
            "correct": prediction == reference,
        }
    if task == "text_qa":
        result = text_qa_mod.run_one(
            model=model,
            processor=processor,
            task_prompt=text_qa_task_prompt,
            sample=sample,
            max_new_tokens=16,
        )
        prediction = text_qa_mod.normalize_label(result["normalized_response"])
        reference = sample["label"].strip().lower()
        return {
            "model": "qwen",
            "prediction": prediction,
            "reference": reference,
            "correct": prediction == reference,
        }
    if task == "image":
        result = image_mod.run_one(
            model=model,
            processor=processor,
            instruction=image_instruction,
            sample=sample,
            max_new_tokens=16,
        )
        prediction = image_mod.normalize_label(result["normalized_response"])
        reference = sample["label"].strip().lower()
        return {
            "model": "qwen",
            "prediction": prediction,
            "reference": reference,
            "correct": prediction == reference,
        }
    if task == "ocr":
        result = ocr_mod.run_one(
            model=model,
            processor=processor,
            instruction=ocr_instruction,
            sample=sample,
            max_new_tokens=32,
        )
        prediction = ocr_mod.normalize_ocr(result["normalized_response"])
        reference = ocr_mod.normalize_ocr(
            sample.get("normalized_label", sample["label"])
        )
        return {
            "model": "qwen",
            "prediction": prediction,
            "reference": reference,
            "correct": prediction == reference,
        }
    if task == "asr":
        result = asr_mod.run_one(
            model=model,
            processor=processor,
            instruction=asr_instruction,
            sample=sample,
            max_new_tokens=256,
        )
        prediction = result["prediction"]
        reference = result["reference"]
        word_errors = int(result["word_errors"])
        reference_words = int(result["reference_words"])
        wer = word_errors / reference_words if reference_words else 0.0
        return {
            "model": "qwen",
            "prediction": prediction,
            "reference": reference,
            "word_errors": word_errors,
            "reference_words": reference_words,
            "wer": wer,
        }
    if task == "visual_qa":
        instruction = visual_qa_mod.build_instruction(visual_qa_task_prompt, sample)
        result = visual_qa_mod.image_base.run_one(
            model=model,
            processor=processor,
            instruction=instruction,
            sample=sample,
            max_new_tokens=1,
        )
        prediction = visual_qa_mod.normalize_option(result["raw_response"])
        reference = visual_qa_mod.normalize_option(sample["label"])
        valid_labels = {chr(ord("A") + i) for i in range(len(sample["choices"]))}
        return {
            "model": "qwen",
            "prediction": prediction,
            "reference": reference,
            "correct": prediction == reference,
            "valid_closed_set_output": prediction in valid_labels,
        }
    if task == "lid":
        result = lid_mod.audio_base.run_one(
            model=model,
            processor=processor,
            instruction=lid_instruction,
            sample=sample,
            max_new_tokens=16,
        )
        prediction = lid_mod.normalize_lang_code(result["raw_response"])
        reference = lid_mod.normalize_lang_code(sample["label"])
        return {
            "model": "qwen",
            "prediction": prediction,
            "reference": reference,
            "correct": prediction == reference,
        }
    raise ValueError(f"Unsupported task: {task}")


run_worker(
    worker_name="qwen",
    handler=handler,
    ready_payload={
        "model": "Qwen2.5-Omni-7B",
        "tasks": ["text", "image", "ocr", "asr", "visual_qa", "lid", "text_qa"],
    },
)
