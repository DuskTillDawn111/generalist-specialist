import sys
from pathlib import Path

import paddle
from paddleocr import (
    TextRecognition,
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


import eval_paddleocr_iiit5k as formal

from worker_protocol import run_worker


MODEL_DIR = (
    ROOT
    / "models/paddleocr/official_models/"
    "PP-OCRv6_medium_rec"
)


paddle.set_device(
    "gpu:0"
)


model = TextRecognition(
    model_name=
        "PP-OCRv6_medium_rec",

    model_dir=
        str(MODEL_DIR),

    device=
        "gpu:0"
)


paddle.device.synchronize()


def unpack_result(value):
    if isinstance(
        value,
        tuple
    ):
        text = value[0]

        score = (
            value[1]
            if len(value) > 1
            else None
        )

        return text, score

    if isinstance(
        value,
        dict
    ):
        text = (
            value.get("rec_text")
            or value.get("text")
            or value.get(
                "prediction"
            )
            or ""
        )

        score = (
            value.get("rec_score")
            or value.get("score")
        )

        return text, score

    return str(value), None


def handler(request):
    sample = request[
        "sample"
    ]

    image = (
        formal.load_image_bgr(
            sample["input"]
        )
    )

    output = model.predict(
        input=image,
        batch_size=1
    )

    extracted = (
        formal.extract_result(
            output
        )
    )

    prediction, score = (
        unpack_result(
            extracted
        )
    )

    prediction = (
        formal.normalize_ocr(
            prediction
        )
    )

    reference = (
        formal.normalize_ocr(
            sample.get(
                "normalized_label",
                sample["label"]
            )
        )
    )

    return {
        "model":
            "ppocrv6_medium_rec",

        "prediction":
            prediction,

        "reference":
            reference,

        "score":
            score,

        "correct":
            prediction
            == reference,
    }


run_worker(
    "paddleocr",
    handler,
    {
        "model":
            "PP-OCRv6-medium-rec",

        "tasks": [
            "ocr"
        ],
    }
)
