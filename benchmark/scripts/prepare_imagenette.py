import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path("/root/autodl-tmp/generalist-specialist")

VAL_DIR = (
    ROOT
    / "datasets/imagenette/raw/imagenette2-320/val"
)

MANIFEST_DIR = ROOT / "benchmark/manifests"

FULL_MANIFEST = (
    MANIFEST_DIR
    / "image_classification_full.jsonl"
)

DEV_MANIFEST = (
    MANIFEST_DIR
    / "image_classification.jsonl"
)

SEED = 42
SAMPLES_PER_CLASS = 20


# Imagenette 10 类
CLASSES = [
    ("n01440764", "tench"),
    ("n02102040", "English springer"),
    ("n02979186", "cassette player"),
    ("n03000684", "chain saw"),
    ("n03028079", "church"),
    ("n03394916", "French horn"),
    ("n03417042", "garbage truck"),
    ("n03425413", "gas pump"),
    ("n03445777", "golf ball"),
    ("n03888257", "parachute"),
]


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


def main():

    if not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {VAL_DIR}"
        )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Reading local Imagenette validation set...")
    print("Validation directory:", VAL_DIR)

    records = []
    class_records = {}

    global_index = 0

    for label_id, (wnid, label) in enumerate(CLASSES):

        class_dir = VAL_DIR / wnid

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        files = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in {".jpeg", ".jpg", ".png"}
        )

        print(
            f"{label_id}: "
            f"{label:<20} "
            f"{len(files)} images"
        )

        current = []

        for image_path in files:

            record = {
                "id": f"image_{global_index:06d}",
                "task": "image_classification",
                "dataset": "imagenette",
                "split": "validation",

                "input_type": "image",
                "input": str(image_path.resolve()),

                "label_id": label_id,
                "label": label,
                "wnid": wnid,
            }

            records.append(record)
            current.append(record)

            global_index += 1

        class_records[label_id] = current

    # 保存完整 validation
    save_jsonl(
        records,
        FULL_MANIFEST
    )

    # 每类固定抽 20 张，共 200 张
    rng = random.Random(SEED)

    dev_records = []

    for label_id, (_, label) in enumerate(CLASSES):

        candidates = class_records[label_id]

        if len(candidates) < SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"{label} only has "
                f"{len(candidates)} samples"
            )

        selected = rng.sample(
            candidates,
            SAMPLES_PER_CLASS
        )

        dev_records.extend(selected)

    # 保证顺序固定
    dev_records.sort(
        key=lambda x: x["id"]
    )

    save_jsonl(
        dev_records,
        DEV_MANIFEST
    )

    counts = Counter(
        r["label"]
        for r in dev_records
    )

    print()
    print("========== IMAGENETTE READY ==========")
    print("Full validation:", len(records))
    print("Benchmark subset:", len(dev_records))
    print("Seed:", SEED)
    print("Samples per class:", SAMPLES_PER_CLASS)

    print()
    print("Benchmark class distribution:")

    for _, label in CLASSES:
        print(
            f"  {label}: {counts[label]}"
        )

    print()
    print("Full manifest:", FULL_MANIFEST)
    print("Benchmark manifest:", DEV_MANIFEST)

    print()
    print("First 3 benchmark samples:")

    for record in dev_records[:3]:
        print(
            json.dumps(
                record,
                ensure_ascii=False
            )
        )


if __name__ == "__main__":
    main()
