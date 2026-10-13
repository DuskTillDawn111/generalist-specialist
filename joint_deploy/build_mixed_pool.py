import argparse
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

MANIFESTS = {
    "text": (
        ROOT
        / "benchmark/manifests/text_classification.jsonl"
    ),

    "image": (
        ROOT
        / "benchmark/manifests/image_classification.jsonl"
    ),

    "ocr": (
        ROOT
        / "benchmark/manifests/ocr.jsonl"
    ),

    "asr": (
        ROOT
        / "benchmark/manifests/asr.jsonl"
    ),

    "visual_qa": (
        ROOT
        / "benchmark/manifests/visual_qa.jsonl"
    ),

    "lid": (
        ROOT
        / "benchmark/manifests/lid.jsonl"
    ),
}


def read_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--per-task",
        type=int,
        default=200,
        help=(
            "Number of samples from each task. "
            "0 means use all available samples."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260918
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True
    )

    args = parser.parse_args()

    rng = random.Random(
        args.seed
    )

    pool = []

    print(
        "===== MIXED TASK POOL ====="
    )

    for task_type, path in (
        MANIFESTS.items()
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing manifest: {path}"
            )

        rows = read_jsonl(
            path
        )

        if args.per_task > 0:
            if len(rows) < args.per_task:
                raise RuntimeError(
                    f"{task_type}: requested "
                    f"{args.per_task}, but only "
                    f"{len(rows)} samples exist."
                )

            rows = rows[
                :args.per_task
            ]

        print(
            f"{task_type:>5}: "
            f"{len(rows)} samples "
            f"from {path.name}"
        )

        for local_index, sample in enumerate(
            rows
        ):
            original_id = sample.get(
                "id",
                f"{task_type}_{local_index:06d}"
            )

            pool.append(
                {
                    "pool_id":
                        f"{task_type}:{original_id}",

                    "task_type":
                        task_type,

                    "source_manifest":
                        path.name,

                    "sample":
                        sample,
                }
            )

    rng.shuffle(
        pool
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        for index, item in enumerate(
            pool
        ):
            item["arrival_index"] = index

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

    counts = Counter(
        x["task_type"]
        for x in pool
    )

    print()
    print(
        "Total tasks:",
        len(pool)
    )

    print(
        "Counts:",
        dict(counts)
    )

    print(
        "Seed:",
        args.seed
    )

    print(
        "Saved:",
        output_path
    )

    print()
    print(
        "First 10 tasks:"
    )

    for item in pool[:10]:
        print(
            f"{item['arrival_index']:03d} "
            f"{item['task_type']:>5} "
            f"{item['pool_id']}"
        )


if __name__ == "__main__":
    main()
