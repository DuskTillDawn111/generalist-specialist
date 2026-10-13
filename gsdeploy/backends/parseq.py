"""Pinned PARSeq inference; checkpoint and installed STRHub come from the environment."""

import os
from pathlib import Path
import re
import torch
from PIL import Image
from strhub.models.utils import create_model
from strhub.data.module import SceneTextDataModule
from gsdeploy.runtime.protocol import run_worker

torch.set_num_threads(1)
torch.manual_seed(0)
model = create_model(os.environ.get("GS_PARSEQ_EXPERIMENT", "parseq"), pretrained=False)
weights = torch.load(
    Path(os.environ["GS_MODEL_PATH"]), map_location="cpu", weights_only=True
)
model.model.load_state_dict(weights, strict=True)
model.eval().float().to("cuda:0")
transform = SceneTextDataModule.get_transform(
    model.hparams.img_size, augment=False, rotation=0
)
torch.cuda.synchronize()


@torch.inference_mode()
def handler(request):
    if request["task_type"] != "ocr":
        raise ValueError("PARSeq only admits OCR")
    sample = request["sample"]
    with Image.open(sample["input"]) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to("cuda:0")
    logits = model(tensor)
    decoded, _ = model.tokenizer.decode(logits.softmax(-1))
    normalize = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    prediction = normalize(decoded[0])
    reference = normalize(sample.get("normalized_label", sample["label"]))
    torch.cuda.synchronize()
    return {
        "prediction": prediction,
        "reference": reference,
        "correct": prediction == reference,
    }


run_worker("parseq", handler, {"tasks": ["ocr"], "precision": "torch.float32"})
