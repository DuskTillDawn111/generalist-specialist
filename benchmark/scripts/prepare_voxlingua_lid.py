import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import soundfile as sf


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

ZIP_PATH = (
    ROOT
    / "datasets/voxlingua107/raw/dev.zip"
)

AUDIO_ROOT = (
    ROOT
    / "datasets/voxlingua107/dev"
)

FULL_MANIFEST = (
    ROOT
    / "benchmark/manifests/lid_full.jsonl"
)

MANIFEST = (
    ROOT
    / "benchmark/manifests/lid.jsonl"
)

SEED = 42
PER_LANGUAGE = 20

TARGET_LANGUAGES = [
    "ar",
    "de",
    "en",
    "es",
    "fa",
    "fr",
    "ja",
    "ru",
    "tr",
    "zh",
]

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "ja": "Japanese",
    "ru": "Russian",
    "tr": "Turkish",
    "zh": "Chinese",
}


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

    AUDIO_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    full_rows = []

    with zipfile.ZipFile(
        ZIP_PATH
    ) as zf:

        wav_names = sorted(
            name
            for name in zf.namelist()
            if name.lower().endswith(
                ".wav"
            )
        )

        for index, archive_name in enumerate(
            wav_names
        ):

            lang = archive_name.split(
                "/",
                1
            )[0]

            relative = Path(
                archive_name
            )

            output_path = (
                AUDIO_ROOT
                / relative
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            if not output_path.exists():

                with zf.open(
                    archive_name
                ) as src:

                    output_path.write_bytes(
                        src.read()
                    )

            info = sf.info(
                output_path
            )

            if int(
                info.samplerate
            ) != 16000:
                raise ValueError(
                    f"{archive_name}: "
                    f"unexpected sample rate "
                    f"{info.samplerate}"
                )

            full_rows.append(
                {
                    "id":
                        (
                            "lid_"
                            f"{index:06d}"
                        ),

                    "task":
                        "lid",

                    "dataset":
                        "voxlingua107",

                    "split":
                        "dev",

                    "input_type":
                        "audio",

                    "input":
                        str(
                            output_path
                            .resolve()
                        ),

                    "label":
                        lang,

                    "language_name":
                        LANGUAGE_NAMES.get(
                            lang
                        ),

                    "source_label":
                        lang,

                    "source_file":
                        archive_name,

                    "duration_s":
                        round(
                            float(
                                info.duration
                            ),
                            6
                        ),

                    "sample_rate":
                        int(
                            info.samplerate
                        ),
                }
            )

    write_jsonl(
        FULL_MANIFEST,
        full_rows
    )

    by_language = {}

    for row in full_rows:

        by_language.setdefault(
            row["label"],
            []
        ).append(
            row
        )

    missing = [
        lang
        for lang in TARGET_LANGUAGES
        if len(
            by_language.get(
                lang,
                []
            )
        ) < PER_LANGUAGE
    ]

    if missing:
        raise RuntimeError(
            "Not enough samples for: "
            f"{missing}"
        )

    rng = random.Random(
        SEED
    )

    selected = []

    for lang in TARGET_LANGUAGES:

        candidates = list(
            by_language[
                lang
            ]
        )

        rng.shuffle(
            candidates
        )

        selected.extend(
            candidates[
                :PER_LANGUAGE
            ]
        )

    # Avoid grouping requests by language
    # in the formal manifest.
    rng.shuffle(
        selected
    )

    write_jsonl(
        MANIFEST,
        selected
    )

    full_counts = Counter(
        row["label"]
        for row in full_rows
    )

    formal_counts = Counter(
        row["label"]
        for row in selected
    )

    print(
        "===== VoxLingua107 LID ====="
    )

    print(
        "full samples =",
        len(full_rows)
    )

    print(
        "full languages =",
        len(full_counts)
    )

    print(
        "formal samples =",
        len(selected)
    )

    print(
        "formal languages =",
        len(formal_counts)
    )

    print(
        "formal counts =",
        dict(
            sorted(
                formal_counts.items()
            )
        )
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
