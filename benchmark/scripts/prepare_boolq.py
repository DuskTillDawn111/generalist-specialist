import json
import random
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

RAW_FILE = (
    ROOT
    / "datasets/boolq/raw/BoolQ/val.jsonl"
)

MANIFEST_DIR = (
    ROOT / "benchmark/manifests"
)

FULL_MANIFEST = (
    MANIFEST_DIR / "text_qa_full.jsonl"
)

DEV_MANIFEST = (
    MANIFEST_DIR / "text_qa.jsonl"
)

SEED = 42
DEV_SIZE = 200


def read_jsonl(path):
    records = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def save_jsonl(records, path):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


def main():

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Missing BoolQ validation file: "
            f"{RAW_FILE}"
        )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_records = read_jsonl(
        RAW_FILE
    )

    print(
        "Raw validation samples:",
        len(raw_records)
    )

    records = []

    for local_idx, sample in enumerate(
        raw_records
    ):

        # Real BoolQ fields confirmed:
        # question / passage / idx / label
        question = str(
            sample["question"]
        ).strip()

        passage = str(
            sample["passage"]
        ).strip()

        answer = bool(
            sample["label"]
        )

        label_id = (
            1 if answer else 0
        )

        label = (
            "yes" if answer else "no"
        )

        # Keep the worker interface consistent:
        # sample["input"] is the text sent to
        # the task-specific prompt.
        input_text = (
            "Passage:\n"
            + passage
            + "\n\nQuestion:\n"
            + question
        )

        source_idx = int(
            sample["idx"]
        )

        record = {
            "id":
                f"textqa_{local_idx:06d}",

            "task":
                "text_qa",

            "dataset":
                "boolq",

            "split":
                "validation",

            "input_type":
                "text",

            "input":
                input_text,

            "question":
                question,

            "passage":
                passage,

            "source_idx":
                source_idx,

            "label_id":
                label_id,

            "label":
                label,
        }

        records.append(
            record
        )

    # ------------------------------
    # Full validation manifest
    # ------------------------------

    save_jsonl(
        records,
        FULL_MANIFEST
    )

    # ------------------------------
    # Fixed 200-sample benchmark
    # Same style as prepare_sst2.py
    # ------------------------------

    if len(records) < DEV_SIZE:
        raise RuntimeError(
            f"Need {DEV_SIZE} samples, "
            f"but only have {len(records)}"
        )

    rng = random.Random(
        SEED
    )

    indices = list(
        range(len(records))
    )

    rng.shuffle(
        indices
    )

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

    no_count = sum(
        r["label_id"] == 0
        for r in dev_records
    )

    yes_count = sum(
        r["label_id"] == 1
        for r in dev_records
    )

    print()
    print(
        "========== BOOLQ READY =========="
    )

    print(
        "Full validation:",
        len(records)
    )

    print(
        "Benchmark subset:",
        len(dev_records)
    )

    print(
        "No:",
        no_count
    )

    print(
        "Yes:",
        yes_count
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
    print(
        "First 3 benchmark samples:"
    )

    for r in dev_records[:3]:
        print(
            json.dumps(
                r,
                ensure_ascii=False
            )
        )


if __name__ == "__main__":
    main()
