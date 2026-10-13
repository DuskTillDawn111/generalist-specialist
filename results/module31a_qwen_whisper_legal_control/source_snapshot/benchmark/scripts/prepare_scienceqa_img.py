import json
import random
import zipfile
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

PROBLEMS = (
    ROOT
    / "datasets/scienceqa/raw/problems.json"
)

IMAGE_ZIP = (
    ROOT
    / "datasets/scienceqa/raw/val.zip"
)

IMAGE_ROOT = (
    ROOT
    / "datasets/scienceqa/images/val"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/visual_qa_full.jsonl"
)

MANIFEST = (
    ROOT
    / "benchmark/manifests/visual_qa.jsonl"
)

SEED = 42
DEV_SIZE = 200


def answer_letter(label_id):
    return chr(
        ord("A")
        + int(label_id)
    )


def write_jsonl(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


def main():

    with PROBLEMS.open(
        "r",
        encoding="utf-8"
    ) as f:
        problems = json.load(f)

    with zipfile.ZipFile(
        IMAGE_ZIP
    ) as zf:

        archive_names = set(
            zf.namelist()
        )

        val_total = 0
        with_main_image = 0
        excluded_choice_images = 0
        missing_main_image = 0

        rows = []

        for source_id, item in (
            problems.items()
        ):

            if item.get("split") != "val":
                continue

            val_total += 1

            image_name = item.get(
                "image"
            )

            if not image_name:
                continue

            with_main_image += 1

            prefix = (
                f"val/{source_id}/"
            )

            main_archive_path = (
                prefix
                + image_name
            )

            choice_images = [
                name
                for name in archive_names
                if (
                    name.startswith(
                        prefix
                    )
                    and "/choice_" in name
                    and name.lower().endswith(
                        ".png"
                    )
                )
            ]

            if choice_images:
                excluded_choice_images += 1
                continue

            if (
                main_archive_path
                not in archive_names
            ):
                missing_main_image += 1
                continue

            output_dir = (
                IMAGE_ROOT
                / source_id
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_path = (
                output_dir
                / image_name
            )

            if not output_path.exists():

                with zf.open(
                    main_archive_path
                ) as src:

                    output_path.write_bytes(
                        src.read()
                    )

            choices = item.get(
                "choices"
            )

            answer = item.get(
                "answer"
            )

            if not isinstance(
                choices,
                list
            ):
                raise ValueError(
                    f"{source_id}: "
                    "choices is not a list"
                )

            if not isinstance(
                answer,
                int
            ):
                raise ValueError(
                    f"{source_id}: "
                    f"invalid answer={answer!r}"
                )

            if not (
                0
                <= answer
                < len(choices)
            ):
                raise ValueError(
                    f"{source_id}: "
                    f"answer={answer}, "
                    f"choices={len(choices)}"
                )

            rows.append(
                {
                    "id":
                        (
                            "visual_qa_"
                            f"{int(source_id):06d}"
                        ),

                    "source_id":
                        source_id,

                    "task":
                        "visual_qa",

                    "dataset":
                        "scienceqa",

                    "split":
                        "validation",

                    "input_type":
                        "image",

                    "input":
                        str(
                            output_path
                            .resolve()
                        ),

                    "question":
                        item[
                            "question"
                        ],

                    "choices":
                        choices,

                    "label_id":
                        int(answer),

                    "label":
                        answer_letter(
                            answer
                        ),

                    "hint":
                        item.get(
                            "hint",
                            ""
                        ),

                    "subject":
                        item.get(
                            "subject"
                        ),

                    "grade":
                        item.get(
                            "grade"
                        ),

                    "topic":
                        item.get(
                            "topic"
                        ),

                    "category":
                        item.get(
                            "category"
                        ),
                }
            )

    rows.sort(
        key=lambda x: int(
            x["source_id"]
        )
    )

    if len(rows) < DEV_SIZE:
        raise RuntimeError(
            f"Only {len(rows)} eligible "
            f"samples, need {DEV_SIZE}."
        )

    write_jsonl(
        FULL_MANIFEST,
        rows
    )

    rng = random.Random(
        SEED
    )

    indices = list(
        range(
            len(rows)
        )
    )

    rng.shuffle(
        indices
    )

    selected = [
        rows[i]
        for i in indices[
            :DEV_SIZE
        ]
    ]

    write_jsonl(
        MANIFEST,
        selected
    )

    print(
        "===== ScienceQA-IMG ====="
    )

    print(
        "validation total =",
        val_total
    )

    print(
        "with main image =",
        with_main_image
    )

    print(
        "excluded choice-image questions =",
        excluded_choice_images
    )

    print(
        "missing main image =",
        missing_main_image
    )

    print(
        "eligible single-image questions =",
        len(rows)
    )

    print(
        "formal subset =",
        len(selected)
    )

    print(
        "seed =",
        SEED
    )

    print(
        "full manifest =",
        FULL_MANIFEST
    )

    print(
        "formal manifest =",
        MANIFEST
    )

    print()
    print(
        "first formal sample:"
    )

    print(
        json.dumps(
            selected[0],
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
