import json
import random
from pathlib import Path


ROOT = Path("/root/autodl-tmp/generalist-specialist")

MANIFEST_DIR = ROOT / "benchmark/manifests"
OUT_DIR = ROOT / "benchmark/workloads"

SEED = 42

FILES = {
    "text_classification":
        MANIFEST_DIR / "text_classification.jsonl",

    "image_classification":
        MANIFEST_DIR / "image_classification.jsonl",

    "ocr":
        MANIFEST_DIR / "ocr.jsonl",

    "asr":
        MANIFEST_DIR / "asr.jsonl",
}


def read_jsonl(path):

    data = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


DATA = {
    task: read_jsonl(path)
    for task, path in FILES.items()
}


def sample_task(task, n, rng):

    pool = DATA[task]

    return [
        rng.choice(pool)
        for _ in range(n)
    ]


def save_trace(name, requests):

    path = OUT_DIR / f"{name}.jsonl"

    with open(path, "w", encoding="utf-8") as f:

        for seq, (phase, item) in enumerate(requests):

            record = {
                "request_id":
                    f"{name}_{seq:06d}",

                "seq":
                    seq,

                "phase":
                    phase,

                "task":
                    item["task"],

                "sample_id":
                    item["id"],

                "input_type":
                    item["input_type"],

                "input":
                    item["input"],

                "label":
                    item["label"]
            }

            if "duration_s" in item:
                record["duration_s"] = item["duration_s"]

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        name,
        "->",
        len(requests),
        "requests"
    )


def stationary_trace(
    name,
    distribution,
    total=800
):

    rng = random.Random(
        f"{SEED}-{name}"
    )

    requests = []

    allocated = 0

    tasks = list(distribution.keys())

    for i, task in enumerate(tasks):

        if i == len(tasks) - 1:
            count = total - allocated
        else:
            count = round(
                total * distribution[task]
            )

        allocated += count

        samples = sample_task(
            task,
            count,
            rng
        )

        requests.extend(
            ("stationary", x)
            for x in samples
        )

    rng.shuffle(requests)

    save_trace(
        name,
        requests
    )


def dynamic_trace():

    rng = random.Random(
        f"{SEED}-dynamic"
    )

    phases = [

        (
            "phase1_balanced",
            {
                "text_classification": 50,
                "image_classification": 50,
                "ocr": 50,
                "asr": 50
            }
        ),

        (
            "phase2_image_heavy",
            {
                "text_classification": 20,
                "image_classification": 140,
                "ocr": 20,
                "asr": 20
            }
        ),

        (
            "phase3_asr_heavy",
            {
                "text_classification": 20,
                "image_classification": 20,
                "ocr": 20,
                "asr": 140
            }
        ),

        (
            "phase4_ocr_heavy",
            {
                "text_classification": 20,
                "image_classification": 20,
                "ocr": 140,
                "asr": 20
            }
        ),
    ]

    requests = []

    for phase_name, counts in phases:

        phase_requests = []

        for task, count in counts.items():

            samples = sample_task(
                task,
                count,
                rng
            )

            phase_requests.extend(
                (phase_name, x)
                for x in samples
            )

        rng.shuffle(
            phase_requests
        )

        requests.extend(
            phase_requests
        )

    save_trace(
        "dynamic_shift",
        requests
    )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    stationary_trace(
        "balanced",
        {
            "text_classification": 0.25,
            "image_classification": 0.25,
            "ocr": 0.25,
            "asr": 0.25
        }
    )

    stationary_trace(
        "text_heavy",
        {
            "text_classification": 0.55,
            "image_classification": 0.15,
            "ocr": 0.15,
            "asr": 0.15
        }
    )

    stationary_trace(
        "image_heavy",
        {
            "text_classification": 0.15,
            "image_classification": 0.55,
            "ocr": 0.15,
            "asr": 0.15
        }
    )

    stationary_trace(
        "ocr_heavy",
        {
            "text_classification": 0.15,
            "image_classification": 0.15,
            "ocr": 0.55,
            "asr": 0.15
        }
    )

    stationary_trace(
        "asr_heavy",
        {
            "text_classification": 0.15,
            "image_classification": 0.15,
            "ocr": 0.15,
            "asr": 0.55
        }
    )

    dynamic_trace()

    print()
    print("========== WORKLOADS READY ==========")


if __name__ == "__main__":
    main()
