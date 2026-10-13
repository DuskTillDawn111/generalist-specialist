import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/generalist-specialist")
M = ROOT / "benchmark/manifests"

FILES = {
    "text_classification": M / "text_classification.jsonl",
    "image_classification": M / "image_classification.jsonl",
    "ocr": M / "ocr.jsonl",
    "asr": M / "asr.jsonl",
}

EXPECTED = 200

for task, path in FILES.items():

    if not path.exists():
        raise FileNotFoundError(path)

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"{task}: {len(records)}")

    if len(records) != EXPECTED:
        raise RuntimeError(
            f"{task}: expected {EXPECTED}, got {len(records)}"
        )

    ids = [r["id"] for r in records]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"{task}: duplicate IDs found"
        )

    for r in records:

        if r["task"] != task:
            raise RuntimeError(
                f"Task mismatch: {r}"
            )

        if r["input_type"] != "text":

            p = Path(r["input"])

            if not p.exists():
                raise FileNotFoundError(
                    f"Missing input: {p}"
                )

print()
print("========== BENCHMARK VALID ==========")
print("Tasks: 4")
print("Samples per task: 200")
print("Total requests: 800")
