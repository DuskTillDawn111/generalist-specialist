import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

MODEL_DIR = (
    ROOT / "models/distilbert-sst2"
)


def marker(name):
    torch.cuda.synchronize()

    print(
        f"NVML_MARKER|{name}",
        flush=True
    )

    # Give external NVML sampler enough
    # time to observe a stable state.
    time.sleep(1.0)


print(
    "Loading tokenizer...",
    flush=True
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
)


print(
    "Loading model...",
    flush=True
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
)

model.eval()
model.cuda()

marker(
    "post_load"
)


text = (
    "The movie was surprisingly good "
    "and I enjoyed it."
)

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

inputs = {
    k: v.cuda()
    for k, v in inputs.items()
}


with torch.inference_mode():

    for _ in range(10):
        _ = model(
            **inputs
        )

torch.cuda.synchronize()

marker(
    "post_warmup"
)


print(
    "Running repeated inference...",
    flush=True
)

with torch.inference_mode():

    for _ in range(200):

        _ = model(
            **inputs
        )

        torch.cuda.synchronize()

marker(
    "post_inference"
)


print(
    "Probe complete.",
    flush=True
)
