import json
import random
import re
from pathlib import Path

import soundfile as sf


ROOT = Path("/root/autodl-tmp/generalist-specialist")

DATA_DIR = (
    ROOT
    / "datasets/librispeech/raw/LibriSpeech/test-clean"
)

MANIFEST_DIR = ROOT / "benchmark/manifests"

FULL_MANIFEST = MANIFEST_DIR / "asr_full.jsonl"
DEV_MANIFEST = MANIFEST_DIR / "asr.jsonl"

SEED = 42
DEV_SIZE = 200


def normalize_text(text):
    """
    ASR/WER 统一规范：
    1. lowercase
    2. 去标点
    3. 合并多余空格
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"LibriSpeech test-clean not found: "
            f"{DATA_DIR}"
        )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Scanning LibriSpeech test-clean...")

    records = []

    transcript_files = sorted(
        DATA_DIR.rglob("*.trans.txt")
    )

    print(
        "Transcript files:",
        len(transcript_files)
    )

    for transcript_file in transcript_files:

        with open(
            transcript_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                utt_id, text = line.split(
                    " ",
                    1
                )

                audio_path = (
                    transcript_file.parent
                    / f"{utt_id}.flac"
                )

                if not audio_path.exists():

                    raise FileNotFoundError(
                        f"Missing audio: "
                        f"{audio_path}"
                    )

                info = sf.info(
                    str(audio_path)
                )

                duration = (
                    info.frames
                    / info.samplerate
                )

                parts = utt_id.split("-")

                speaker_id = int(
                    parts[0]
                )

                chapter_id = int(
                    parts[1]
                )

                record = {

                    "id":
                        f"asr_{len(records):06d}",

                    "utterance_id":
                        utt_id,

                    "task":
                        "asr",

                    "dataset":
                        "librispeech",

                    "split":
                        "test-clean",

                    "input_type":
                        "audio",

                    "input":
                        str(
                            audio_path.resolve()
                        ),

                    "label":
                        text,

                    "normalized_label":
                        normalize_text(text),

                    "duration_s":
                        round(duration, 4),

                    "sample_rate":
                        info.samplerate,

                    "speaker_id":
                        speaker_id,

                    "chapter_id":
                        chapter_id,
                }

                records.append(
                    record
                )

    records.sort(
        key=lambda x: x[
            "utterance_id"
        ]
    )

    # 重新编号，确保顺序稳定
    for idx, record in enumerate(records):

        record["id"] = (
            f"asr_{idx:06d}"
        )

    # ===================================
    # 保存完整 test-clean
    # ===================================

    save_jsonl(
        records,
        FULL_MANIFEST
    )

    # ===================================
    # 固定随机种子抽 200 条
    # ===================================

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

    durations = [
        r["duration_s"]
        for r in dev_records
    ]

    print()
    print(
        "========== LIBRISPEECH READY =========="
    )

    print(
        "Full test-clean:",
        len(records)
    )

    print(
        "Benchmark subset:",
        len(dev_records)
    )

    print(
        "Seed:",
        SEED
    )

    print(
        "Duration range:",
        round(min(durations), 2),
        "-",
        round(max(durations), 2),
        "s"
    )

    print(
        "Average duration:",
        round(
            sum(durations)
            / len(durations),
            2
        ),
        "s"
    )

    print(
        "Unique speakers:",
        len(
            {
                r["speaker_id"]
                for r in dev_records
            }
        )
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

    for record in dev_records[:3]:

        print(
            json.dumps(
                record,
                ensure_ascii=False
            )
        )


if __name__ == "__main__":
    main()
