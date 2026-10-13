import json
import random
from pathlib import Path

from datasets import load_dataset


ROOT = Path("/root/autodl-tmp/generalist-specialist")

DATA_DIR = ROOT / "datasets/sst2"

MANIFEST_DIR = ROOT / "benchmark/manifests"

FULL_MANIFEST = MANIFEST_DIR / "text_classification_full.jsonl"
DEV_MANIFEST = MANIFEST_DIR / "text_classification.jsonl"

SEED = 42
DEV_SIZE = 200


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

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading SST-2...")

    dataset = load_dataset(
        "stanfordnlp/sst2",
        cache_dir=str(DATA_DIR / "cache")
    )

    validation = dataset["validation"]

    print(
        "Validation samples:",
        len(validation)
    )

    records = []

    for idx, sample in enumerate(validation):

        label_id = int(sample["label"])

        label_text = (
            "negative"
            if label_id == 0
            else "positive"
        )

        record = {
            "id": f"text_{idx:06d}",
            "task": "text_classification",
            "dataset": "sst2",
            "split": "validation",

            "input_type": "text",
            "input": sample["sentence"],

            "label_id": label_id,
            "label": label_text,
        }

        records.append(record)

    # 保存完整 validation
    save_jsonl(
        records,
        FULL_MANIFEST
    )

    # 固定随机种子抽取 200 条
    rng = random.Random(SEED)

    indices = list(
        range(len(records))
    )

    rng.shuffle(indices)

    selected_indices = sorted(
        indices[:DEV_SIZE]
    )

    dev_records = [
        records[i]
        for i in selected_indices
    ]

    save_jsonl(
        dev_records,
        DEV_MANIFEST
    )

    negative = sum(
        r["label_id"] == 0
        for r in dev_records
    )

    positive = sum(
        r["label_id"] == 1
        for r in dev_records
    )

    print()
    print("========== SST-2 READY ==========")
    print(
        "Full validation:",
        len(records)
    )
    print(
        "Benchmark subset:",
        len(dev_records)
    )
    print(
        "Negative:",
        negative
    )
    print(
        "Positive:",
        positive
    )
    print(
        "Seed:",
        SEED
    )

    print()
    print(
        "Full manifest:",
        FULL_MANIFEST
    )

    print(
        "Benchmark manifest:",
        DEV_MANIFEST
    )

    print()
    print("First 3 samples:")

    for r in dev_records[:3]:
        print(
            json.dumps(
                r,
                ensure_ascii=False
            )
        )


if __name__ == "__main__":
    main()
