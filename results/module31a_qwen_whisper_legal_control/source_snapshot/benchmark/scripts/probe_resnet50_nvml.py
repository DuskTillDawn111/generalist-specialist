import time
from pathlib import Path

import torch
from torchvision.models import resnet50


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

WEIGHT_PATH = (
    ROOT
    / "models/resnet50/resnet50-11ad3fa6.pth"
)


def marker(name):
    torch.cuda.synchronize()

    print(
        f"NVML_MARKER|{name}",
        flush=True
    )

    time.sleep(1.0)


print(
    "Loading ResNet50...",
    flush=True
)

model = resnet50(
    weights=None
)


state = torch.load(
    WEIGHT_PATH,
    map_location="cpu",
    weights_only=True,
)


if (
    isinstance(state, dict)
    and
    "state_dict" in state
):
    state = state[
        "state_dict"
    ]


model.load_state_dict(
    state
)

model.eval()
model.cuda()

marker(
    "post_load"
)


# Match the formal ResNet50 V2 inference
# tensor dimensions: batch=1, RGB, 224x224.
x = torch.zeros(
    1,
    3,
    224,
    224,
    dtype=torch.float32,
    device="cuda",
)


with torch.inference_mode():

    for _ in range(10):
        _ = model(x)

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
        _ = model(x)

torch.cuda.synchronize()

marker(
    "post_inference"
)


print(
    "Probe complete.",
    flush=True
)
