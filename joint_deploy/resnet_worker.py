import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
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


WEIGHT_PATH = (
    ROOT
    / "models/resnet50/"
    "resnet50-11ad3fa6.pth"
)


TARGET_NAMES = [
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


# --------------------------------------------------
# Exactly the same preprocessing / metadata
# as eval_resnet_imagenette.py
# --------------------------------------------------

weights = (
    ResNet50_Weights
    .IMAGENET1K_V2
)

preprocess = (
    weights.transforms()
)

categories = (
    weights.meta[
        "categories"
    ]
)


imagenet_indices = [
    categories.index(
        name
    )
    for name in TARGET_NAMES
]


index_tensor = torch.tensor(
    imagenet_indices,
    dtype=torch.long,
    device=device
)


# --------------------------------------------------
# Exactly the same model loading
# as eval_resnet_imagenette.py
# --------------------------------------------------

model = resnet50(
    weights=None
)


state_dict = torch.load(
    WEIGHT_PATH,
    map_location="cpu",
    weights_only=True
)


model.load_state_dict(
    state_dict,
    strict=True
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

    torch.cuda.synchronize()

    with Image.open(
        sample["input"]
    ) as img:

        img = img.convert(
            "RGB"
        )

        tensor = (
            preprocess(
                img
            )
            .unsqueeze(0)
            .to(
                device
            )
        )

    logits = model(
        tensor
    )

    # ----------------------------------------------
    # IMPORTANT:
    # 1000-class ImageNet logits
    #        ↓
    # Imagenette closed-set 10-class logits
    # ----------------------------------------------

    logits_10 = torch.index_select(
        logits,
        dim=1,
        index=index_tensor
    )


    pred_label_id = int(
        torch.argmax(
            logits_10,
            dim=1
        ).item()
    )


    pred_label = (
        TARGET_NAMES[
            pred_label_id
        ]
    )


    reference_id = int(
        sample[
            "label_id"
        ]
    )

    reference = (
        sample[
            "label"
        ]
    )


    is_correct = (
        pred_label_id
        == reference_id
    )

    torch.cuda.synchronize()


    return {
        "model":
            "resnet50",

        "prediction_id":
            pred_label_id,

        "prediction":
            pred_label,

        "reference_id":
            reference_id,

        "reference":
            reference,

        "imagenet_index":
            imagenet_indices[
                pred_label_id
            ],

        "correct":
            is_correct,
    }


run_worker(
    "resnet50",
    handler,
    {
        "model":
            "ResNet50",

        "evaluation_mode":
            "closed_set_10_class",

        "tasks": [
            "image"
        ],
    }
)
