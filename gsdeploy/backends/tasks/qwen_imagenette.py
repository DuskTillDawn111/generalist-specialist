import os
import json
import time
from pathlib import Path
import torch
from qwen_omni_utils import process_mm_info

PROMPT_FILE = Path(
    os.environ.get(
        "GS_PROMPT_FILE",
        str(Path(__file__).resolve().parents[3] / "configs/prompts.json"),
    )
)


def load_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts["image_classification"]


def normalize_label(text):
    return text.strip().lower()


def sync():
    torch.cuda.synchronize()


def run_one(model, processor, instruction, sample, max_new_tokens):
    sync()
    e2e_start = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["input"]},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    if images is None:
        raise RuntimeError("process_mm_info returned no image.")
    if len(images) != 1:
        raise RuntimeError(f"Expected exactly one image, got {len(images)}.")
    image_width = None
    image_height = None
    if hasattr(images[0], "size"):
        image_width = int(images[0].size[0])
        image_height = int(images[0].size[1])
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise RuntimeError("Vision inputs missing pixel_values/image_grid_thw.")
    pixel_values_shape = tuple(inputs["pixel_values"].shape)
    image_grid_thw = inputs["image_grid_thw"].detach().cpu().tolist()
    inputs = inputs.to("cuda")
    input_length = int(inputs["input_ids"].shape[1])
    sync()
    model_start = time.perf_counter()
    output = model.generate(
        **inputs,
        thinker_max_new_tokens=max_new_tokens,
        do_sample=False,
        return_audio=False,
    )
    sync()
    model_end = time.perf_counter()
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected Tensor output, got {type(output)}")
    if output.shape[1] < input_length:
        raise RuntimeError("Generated sequence shorter than input sequence.")
    generated_ids = output[:, input_length:]
    decoded = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    normalized = normalize_label(decoded)
    e2e_end = time.perf_counter()
    return {
        "raw_response": decoded,
        "normalized_response": normalized,
        "input_tokens": input_length,
        "output_tokens": int(generated_ids.shape[1]),
        "image_width": image_width,
        "image_height": image_height,
        "pixel_values_shape": list(pixel_values_shape),
        "image_grid_thw": image_grid_thw,
        "model_latency_ms": (model_end - model_start) * 1000,
        "e2e_latency_ms": (e2e_end - e2e_start) * 1000,
    }
