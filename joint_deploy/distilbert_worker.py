import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

sys.path.insert(
    0,
    str(
        ROOT / "joint_deploy"
    )
)

from worker_protocol import run_worker


MODEL_DIR = (
    ROOT
    / "models/distilbert-sst2"
)


device = torch.device(
    "cuda:0"
)


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )
)


model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )
)

model.eval()
model.to(
    device
)

torch.cuda.synchronize()


@torch.inference_mode()
def handler(request):
    sample = request[
        "sample"
    ]

    inputs = tokenizer(
        sample["input"],
        return_tensors="pt",
        truncation=True
    )

    inputs = {
        k: v.to(device)
        for k, v
        in inputs.items()
    }

    logits = model(
        **inputs
    ).logits

    pred_id = int(
        logits.argmax(
            dim=-1
        ).item()
    )

    prediction = (
        "positive"
        if pred_id == 1
        else "negative"
    )

    reference = (
        sample["label"]
        .strip()
        .lower()
    )

    return {
        "model":
            "distilbert_sst2",

        "prediction":
            prediction,

        "reference":
            reference,

        "correct":
            prediction
            == reference,
    }


run_worker(
    "distilbert",
    handler,
    {
        "model":
            "DistilBERT-SST2",

        "tasks": [
            "text"
        ],
    }
)
