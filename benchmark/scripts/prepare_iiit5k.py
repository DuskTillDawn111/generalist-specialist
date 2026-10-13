import json
import random
import re
from pathlib import Path


ROOT = Path("/root/autodl-tmp/generalist-specialist")

IMAGE_DIR = ROOT / "datasets/iiit5k/raw/IIIT5K/test"

LABEL_FILE = (
    ROOT
    / "datasets/iiit5k/raw/iiit5k_test_labels.txt"
)

MANIFEST_DIR = ROOT / "benchmark/manifests"

FULL_MANIFEST = MANIFEST_DIR / "ocr_full.jsonl"
DEV_MANIFEST = MANIFEST_DIR / "ocr.jsonl"

SEED = 42
DEV_SIZE = 200


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


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

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"IIIT5K test directory not found: {IMAGE_DIR}"
        )

    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"IIIT5K label file not found: {LABEL_FILE}"
        )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Reading lightweight IIIT5K label file...")

    records = []
    missing = []

    with open(
        LABEL_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line_no, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid line {line_no}: {line}"
                )

            original_path, label = parts

            # 只取图片文件名
            image_name = Path(original_path).name

            image_path = IMAGE_DIR / image_name

            if not image_path.exists():
                missing.append(str(image_path))
                continue

            normalized = normalize_text(label)

            record = {
                "id": f"ocr_{len(records):06d}",
                "task": "ocr",
                "dataset": "iiit5k",
                "split": "test",

                "input_type": "image",
                "input": str(image_path.resolve()),

                "label": label,
                "normalized_label": normalized,

                "word_length": len(normalized),
            }

            records.append(record)

    if missing:

        print()
        print("Missing image count:", len(missing))

        for path in missing[:10]:
            print("  ", path)

        raise RuntimeError(
            "Some IIIT5K images referenced by labels are missing."
        )

    print("Matched samples:", len(records))

    if len(records) != 3000:
        raise RuntimeError(
            f"Expected 3000 IIIT5K test samples, "
            f"but found {len(records)}"
        )

    # 完整测试集
    save_jsonl(
        records,
        FULL_MANIFEST
    )

    # 固定 seed 抽 200 条
    rng = random.Random(SEED)

    selected_indices = sorted(
        rng.sample(
            range(len(records)),
            DEV_SIZE
        )
    )

    dev_records = [
        records[i]
        for i in selected_indices
    ]

    save_jsonl(
        dev_records,
        DEV_MANIFEST
    )

    lengths = [
        r["word_length"]
        for r in dev_records
    ]

    print()
    print("========== IIIT5K READY ==========")

    print("Full test:", len(records))
    print("Benchmark subset:", len(dev_records))
    print("Seed:", SEED)

    print(
        "Word length range:",
        min(lengths),
        "-",
        max(lengths)
    )

    print(
        "Average word length:",
        round(
            sum(lengths) / len(lengths),
            2
        )
    )

    print()
    print("Full manifest:", FULL_MANIFEST)
    print("Benchmark manifest:", DEV_MANIFEST)

    print()
    print("First 5 benchmark samples:")

    for record in dev_records[:5]:
        print(
            json.dumps(
                record,
                ensure_ascii=False
            )
        )


if __name__ == "__main__":
    main()
