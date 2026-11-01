"""Freeze real-data calibration inputs independently of model outcomes."""

import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return {"path": str(path.resolve()), "sha256": digest(path), "samples": len(rows)}


def prepare(root, assets, archive):
    import pyarrow.parquet as pq
    import soundfile as sf

    root = Path(root).resolve()
    assets = Path(assets).resolve()
    archive = Path(archive).resolve()
    target = root / "samples"
    target.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(20261015)
    manifest = {"kind": "performance_calibration_not_quality_admission", "tasks": {}}
    # Equal answerable/unanswerable coverage, fixed before any candidate outputs.
    source = json.loads((root / "squad.json").read_text())
    groups = collections.defaultdict(list)
    for article in source["data"]:
        for paragraph in article["paragraphs"]:
            for q in paragraph["qas"]:
                groups[bool(q["is_impossible"])].append({
                    "id": q["id"], "task": "text", "dataset": "SQuAD2-dev",
                    "context": paragraph["context"], "question": q["question"],
                    "answers": [a["text"] for a in q["answers"]],
                    "label": "" if q["is_impossible"] else q["answers"][0]["text"],
                    "article": article["title"],
                })
    qa = []
    for group in groups.values():
        randomizer.shuffle(group)
        qa.extend(group[:32])
    randomizer.shuffle(qa)
    manifest["tasks"]["text"] = write(root / "text.jsonl", qa)
    # Length-stratified whole receipt OCR. Labels retain their source reading order.
    path = next((assets / "datasets/sroie").glob("*/test.parquet"))
    receipts = pq.read_table(path).to_pylist()
    receipts.sort(key=lambda row: len(" ".join(row["words"])))
    chosen = []
    for quartile in range(4):
        group = receipts[len(receipts) * quartile // 4:len(receipts) * (quartile + 1) // 4]
        randomizer.shuffle(group)
        chosen.extend(group[:8])
    ocr = []
    for row in chosen:
        image = target / ("sroie-" + row["key"] + ".jpg")
        image.write_bytes(row["image"]["bytes"])
        ocr.append({"id": row["key"], "task": "ocr", "dataset": "SROIE-test-pages",
                    "input": str(image), "input_sha256": digest(image),
                    "label": " ".join(row["words"]), "source_revision": path.parent.name})
    manifest["tasks"]["ocr"] = write(root / "ocr.jsonl", ocr)
    # Disjoint duration strata; sample at most one utterance per speaker/chapter
    # initially, then fill deterministically if a stratum has too few sources.
    path = next((assets / "datasets/librispeech-test-other").glob("*/0000.parquet"))
    groups = collections.defaultdict(list)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            with sf.SoundFile(io.BytesIO(row["audio"]["bytes"])) as audio:
                duration = len(audio) / audio.samplerate
            for low, high in [(4, 8), (8, 12), (12, 20), (20, 30)]:
                if low <= duration < high:
                    row["duration_s"] = duration
                    groups[(low, high)].append(row)
                    break
    asr = []
    for duration_bin, group in sorted(groups.items()):
        randomizer.shuffle(group)
        selected, used, other = [], set(), []
        for row in group:
            key = (row["speaker_id"], row["chapter_id"])
            if key not in used and len(selected) < 16:
                selected.append(row)
                used.add(key)
            else:
                other.append(row)
        selected.extend(other[:16 - len(selected)])
        for row in selected:
            audio = target / (row["id"] + ".flac")
            audio.write_bytes(row["audio"]["bytes"])
            asr.append({"id": row["id"], "task": "asr", "dataset": "LibriSpeech-test-other",
                        "input": str(audio), "input_sha256": digest(audio), "label": row["text"],
                        "duration_s": row["duration_s"], "duration_stratum": list(duration_bin),
                        "source_revision": path.parent.name, "speaker_id": row["speaker_id"]})
    manifest["tasks"]["asr"] = write(root / "asr.jsonl", asr)
    # Preserve the current ten language identities, introduce declared short clips.
    source = archive / "benchmark/manifests/lid.jsonl"
    groups = collections.defaultdict(list)
    for line in source.read_text().splitlines():
        row = json.loads(line)
        if Path(row["input"]).is_file() and row["duration_s"] >= 6:
            groups[row["label"]].append(row)
    lid = []
    for language, group in sorted(groups.items()):
        randomizer.shuffle(group)
        if len(group) < 8:
            raise ValueError(f"Insufficient real audio for {language}")
        for i, row in enumerate(group[:8]):
            duration = 3 if i < 4 else 6
            audio, rate = sf.read(row["input"], dtype="float32", frames=16000 * duration)
            assert rate == 16000
            path = target / (row["id"] + f"-{duration}s.wav")
            sf.write(path, audio, rate, subtype="PCM_16")
            lid.append({"id": row["id"] + f"-{duration}s", "task": "lid", "dataset": "VoxLingua107-dev-short",
                        "input": str(path), "input_sha256": digest(path), "label": language,
                        "duration_s": len(audio) / rate, "original_input": row["input"]})
    manifest["tasks"]["lid"] = write(root / "lid.jsonl", lid)
    write(root / "speech.jsonl", asr + lid)
    # Public ImageNet-A: use the full 1000 candidate labels; class coverage is
    # explicitly limited to the natural-adversarial dataset's 200 source classes.
    image_root = assets / "datasets/imagenet-a"
    meta = json.loads((root / "imagenet-a-meta.json").read_text())
    shard_root = image_root / meta["sha"]
    shards = sorted((shard_root / "test").glob("*.tar"))
    if len(shards) == 4:
        names = (shard_root / "classnames.txt").read_text().splitlines()
        classmap = json.loads((assets / "datasets/imagenet-class-index.json").read_text())
        # Official ImageNet-A folder-label to ImageNet-1K mapping, avoiding
        # ambiguous human-readable aliases such as robin/American robin.
        source_indices = json.loads((root / "imagenet-a-index-map.json").read_text())
        assert len(names) == len(source_indices) == 200
        by_class = collections.defaultdict(list)
        for shard in shards:
            with tarfile.open(shard) as tar:
                pending = {}
                for member in tar:
                    if not member.isfile():
                        continue
                    stem, extension = member.name.rsplit(".", 1)
                    if extension not in {"cls", "jpg", "jpeg", "png"}:
                        continue
                    value = tar.extractfile(member).read()
                    pending.setdefault(stem, {})[extension] = value
                    entry = pending[stem]
                    image_key = next((k for k in ("jpg", "jpeg", "png") if k in entry), None)
                    if "cls" in entry and image_key:
                        label = int(entry["cls"].decode())
                        score = hashlib.sha256(stem.encode()).hexdigest()
                        current = by_class[label]
                        if not current or score < current[0][0]:
                            by_class[label] = [(score, stem, entry[image_key])]
                        del pending[stem]
        images = []
        for label, group in sorted(by_class.items()):
            _, stem, data = group[0]
            wnid = classmap[str(source_indices[label])][0]
            path = target / ("imagenet-a-" + hashlib.sha256(stem.encode()).hexdigest()[:16] + ".jpg")
            path.write_bytes(data)
            images.append({"id": "imagenet-a-" + stem, "task": "image", "dataset": "ImageNet-A",
                           "input": str(path), "input_sha256": digest(path), "label": wnid,
                           "source_revision": meta["sha"], "source_class_index": label})
        manifest["tasks"]["image"] = write(root / "image.jsonl", images)
        manifest["tasks"]["image"]["scope"] = "ImageNet-A source classes, 1000-way predictions; not ImageNet1K validation"
    (root / "sample-manifests.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.assets, args.archive), indent=2))
