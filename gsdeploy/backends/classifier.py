"""Native FP32 SST-2 and Imagenette specialists, selected through configuration."""

import os
from pathlib import Path
import torch
from gsdeploy.runtime.protocol import run_worker

backend = os.environ["GS_CLASSIFIER"]
checkpoint = Path(os.environ["GS_MODEL_PATH"])
device = torch.device("cuda:0")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
labels = ["negative", "positive"]
image_labels = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]
if backend in {"roberta", "bart", "distilbert"}:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True
        )
        .eval()
        .float()
        .to(device)
    )
    if model.config.num_labels != 2:
        raise ValueError("A binary SST-2 checkpoint is required")
    tasks = ["text"]
elif backend in {"convnext", "resnet"}:
    from torchvision.models import (
        convnext_tiny,
        ConvNeXt_Tiny_Weights,
        resnet50,
        ResNet50_Weights,
    )

    weights = (
        ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        if backend == "convnext"
        else ResNet50_Weights.IMAGENET1K_V2
    )
    model = (
        convnext_tiny(weights=None) if backend == "convnext" else resnet50(weights=None)
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.eval().float().to(device)
    transform = weights.transforms()
    indices = torch.tensor(
        [weights.meta["categories"].index(label) for label in image_labels],
        device=device,
    )
    tasks = ["image"]
else:
    raise ValueError(f"Unknown classifier backend: {backend}")
torch.cuda.synchronize()


@torch.inference_mode()
def handler(request):
    sample = request["sample"]
    if request["task_type"] not in tasks:
        raise ValueError("Unadmitted classifier task")
    if tasks == ["text"]:
        inputs = tokenizer(sample["input"], return_tensors="pt", truncation=True).to(
            device
        )
        index = int(model(**inputs).logits.argmax(-1).item())
        prediction = labels[index]
    else:
        from PIL import Image

        with Image.open(sample["input"]) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        index = int(model(tensor).index_select(1, indices).argmax(-1).item())
        prediction = image_labels[index]
    torch.cuda.synchronize()
    reference = (
        sample["label"].strip().lower() if tasks == ["text"] else sample["label"]
    )
    correct = (
        index == int(sample["label_id"])
        if "label_id" in sample
        else prediction == reference
    )
    return {
        "prediction": prediction,
        "prediction_id": index,
        "reference": reference,
        "correct": correct,
    }


run_worker(backend, handler, {"tasks": tasks, "precision": "torch.float32"})
