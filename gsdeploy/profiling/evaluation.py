"""Freeze held-out QA/image inputs and audit remaining quality-data blockers.

No model imports, CUDA or NVML calls. Outputs are data preparation, never model
admission. Source calibration files and historical metrics remain unchanged.
"""

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import io
import json
from pathlib import Path
import tarfile

from .prepare import digest, write


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def rank(identifier):
    return hashlib.sha256(("four-task-heldout-v1:" + identifier).encode()).hexdigest()


def select_qa(source, calibration, count=500):
    if count <= 0 or count % 2:
        raise ValueError("QA count must be positive and even")
    used_articles = {row["article"] for row in calibration}
    used_ids = {row["id"] for row in calibration}
    groups = {False: defaultdict(list), True: defaultdict(list)}
    seen = set()
    for article in source["data"]:
        if article["title"] in used_articles:
            continue
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for question in paragraph["qas"]:
                identifier = question["id"]
                if identifier in used_ids or identifier in seen:
                    raise ValueError("QA source contains duplicate/leaked IDs")
                seen.add(identifier)
                impossible = bool(question["is_impossible"])
                answers = [a["text"] for a in question["answers"]]
                if bool(answers) == impossible:
                    raise ValueError("QA answerability is inconsistent")
                row = {"id": identifier, "task": "text", "dataset": "SQuAD2-dev",
                       "article": article["title"], "context": context,
                       "question": question["question"], "answers": answers,
                       "label": "" if impossible else answers[0],
                       "context_character_stratum": sum(len(context) >= n for n in [500, 1000, 1500])}
                groups[impossible][(article["title"], row["context_character_stratum"])].append(row)
    selected = []
    for impossible, strata in groups.items():
        queues = [deque(sorted(rows, key=lambda row: rank(row["id"])))
                  for _, rows in sorted(strata.items())]
        if sum(len(q) for q in queues) < count // 2:
            raise ValueError("Insufficient article-disjoint QA samples")
        chosen = []
        while len(chosen) < count // 2:
            for queue in queues:
                if queue and len(chosen) < count // 2:
                    chosen.append(queue.popleft())
        selected.extend(chosen)
    return sorted(selected, key=lambda row: rank(row["id"]))


def audit_asr(source_path, calibration):
    import pyarrow.parquet as pq

    table = pq.read_table(source_path, columns=["id", "speaker_id"])
    rows = table.to_pylist()
    used = {str(row["speaker_id"]) for row in calibration}
    speakers = {str(row["speaker_id"]) for row in rows}
    available = speakers - used
    return {"source_rows": len(rows), "source_speakers": len(speakers),
            "calibration_speakers": len(used), "unused_speakers": sorted(available),
            "speaker_disjoint_available_rows": sum(str(r["speaker_id"]) in available for r in rows),
            "status": "NEEDS_UNUSED_SPEAKER_SOURCE" if not available else "NOT_YET_FROZEN",
            "source_sha256": digest(source_path),
            "next_source_candidate": "LibriSpeech dev-other; verify speaker and media disjointness before freezing",
            "limitation": "Independence is relative to this project's calibration, not guaranteed absence from model training."}


def select_asr(source_path, calibration, output, per_stratum=None):
    import pyarrow.parquet as pq
    import soundfile as sf

    used_speakers = {str(row["speaker_id"]) for row in calibration}
    used_hashes = {row["input_sha256"] for row in calibration}
    used_ids = {row["id"] for row in calibration}
    groups = defaultdict(lambda: defaultdict(list))
    for batch in pq.ParquetFile(source_path).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            if str(row["speaker_id"]) in used_speakers or row["id"] in used_ids:
                continue
            data = row["audio"]["bytes"]
            sha = hashlib.sha256(data).hexdigest()
            if sha in used_hashes:
                continue
            with sf.SoundFile(io.BytesIO(data)) as audio:
                duration = len(audio) / audio.samplerate
                rate = audio.samplerate
            if rate != 16000:
                raise ValueError("Unexpected LibriSpeech sample rate")
            for low, high in [(4, 8), (8, 12), (12, 20), (20, 30)]:
                if low <= duration < high:
                    row.update(duration_s=duration, duration_stratum=[low, high], input_sha256=sha)
                    groups[(low, high)][str(row["speaker_id"])].append(row)
                    break
    chosen = []
    quotas = [200, 150, 120, 30] if per_stratum is None else [per_stratum] * 4
    for stratum, quota in zip([(4, 8), (8, 12), (12, 20), (20, 30)], quotas):
        queues = [deque(sorted(rows, key=lambda r: rank(r["id"])))
                  for _, rows in sorted(groups[stratum].items())]
        if sum(len(q) for q in queues) < quota:
            raise ValueError("Insufficient unseen-speaker ASR samples in a duration stratum")
        current = []
        while len(current) < quota:
            for queue in queues:
                if queue and len(current) < quota:
                    current.append(queue.popleft())
        chosen.extend(current)
    if len({row["input_sha256"] for row in chosen}) != len(chosen):
        raise ValueError("Duplicate held-out ASR audio")
    output.mkdir(parents=True)
    rows = []
    for row in chosen:
        path = output / (row["input_sha256"] + ".flac")
        path.write_bytes(row["audio"]["bytes"])
        rows.append({"id": row["id"], "task": "asr", "dataset": "LibriSpeech-dev-other",
                     "input": str(path.resolve()), "input_sha256": row["input_sha256"],
                     "label": row["text"], "duration_s": row["duration_s"],
                     "duration_stratum": row["duration_stratum"], "speaker_id": row["speaker_id"],
                     "source_revision": source_path.parent.name})
    return rows


def select_ocr(parquet_path, native_root, calibration, output, per_stratum=50):
    import pyarrow.parquet as pq
    from .protocols import restore_sroie_reference, OCR_PROTOCOL

    provenance = json.loads((native_root / "source-provenance.json").read_text())
    if provenance["archive_sha256"] != native_root.name or digest(provenance["archive"]) != native_root.name:
        raise ValueError("Native OCR archive content mismatch")
    import zipfile
    # The archive digest anchors the annotation tree, not a self-declared hash file.
    with zipfile.ZipFile(provenance["archive"]) as archive:
        archive_blobs = {}
        for member in archive.infolist():
            if Path(member.filename).parent.name == "annotations" and member.filename.endswith(".txt"):
                raw = archive.read(member)
                archive_blobs[Path(member.filename).name] = hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    tree = json.loads((native_root / "source-tree.json").read_text())
    expected_blobs = {Path(item["path"]).name: item["oid"] for item in tree}
    if expected_blobs != archive_blobs:
        raise ValueError("Native annotation tree does not match verified archive")
    used_ids = {row["id"] for row in calibration}
    used_hashes = {row["input_sha256"] for row in calibration}
    valid, rejected, changed = [], [], []
    source_rows = pq.read_table(parquet_path).to_pylist()
    for row in source_rows:
        key = row["key"]
        path = native_root / (key + ".txt")
        try:
            raw = path.read_bytes()
            sha1 = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if sha1 != expected_blobs.get(path.name):
                raise ValueError("Native annotation Git blob mismatch")
            label = restore_sroie_reference(row, raw.decode("utf-8-sig"))
            if label != " ".join(row["words"]):
                changed.append(key)
            sha = hashlib.sha256(row["image"]["bytes"]).hexdigest()
            if key not in used_ids and sha not in used_hashes:
                row.update(label=label, input_sha256=sha, annotation_sha256=hashlib.sha256(raw).hexdigest())
                valid.append(row)
        except (OSError, UnicodeError, ValueError) as error:
            rejected.append({"id": key, "reason": str(error)})
    valid.sort(key=lambda r: (len(r["label"]), r["key"]))
    chosen = []
    for i in range(4):
        group = valid[len(valid) * i // 4:len(valid) * (i + 1) // 4]
        if len(group) < per_stratum:
            raise ValueError("Insufficient verified native OCR references")
        chosen.extend(sorted(group, key=lambda r: rank(r["key"]))[:per_stratum])
    if len({row["input_sha256"] for row in chosen}) != len(chosen):
        raise ValueError("Duplicate held-out OCR images")
    output.mkdir(parents=True)
    rows = []
    for row in chosen:
        path = output / (row["input_sha256"] + ".jpg")
        path.write_bytes(row["image"]["bytes"])
        rows.append({"id": row["key"], "task": "ocr", "dataset": "SROIE-test-pages",
                     "input": str(path.resolve()), "input_sha256": row["input_sha256"], "label": row["label"],
                     "source_revision": parquet_path.parent.name, "native_revision": native_root.name,
                     "annotation_sha256": row["annotation_sha256"], "protocol": OCR_PROTOCOL})
    return rows, {"native_revision": native_root.name, "source_rows": len(source_rows),
                  "verified_unseen_pages": len(valid), "restored_pages_in_source": len(changed),
                  "restored_page_ids": sorted(changed), "rejected_pages": rejected,
                  "geometry_and_text_prefix_verified": True,
                  "limitation": "Source line order retained; reading order and formatting policy still need an independent audit."}


def select_images(shards, calibration, classmap, source_indices, output, revision, per_class=2):
    if per_class < 1 or len(source_indices) != 200 or len(set(source_indices)) != 200:
        raise ValueError("Expected a unique official 200-to-1000 class mapping and positive sample count")
    used_ids = {r["id"] for r in calibration}
    used_hashes = {r["input_sha256"] for r in calibration}
    selected = defaultdict(list)
    for shard in shards:
        with tarfile.open(shard) as tar:
            pending = {}
            for member in tar:
                if not member.isfile() or "." not in member.name:
                    continue
                stem, extension = member.name.rsplit(".", 1)
                if extension not in {"cls", "jpg", "jpeg", "png"}:
                    continue
                entry = pending.setdefault(stem, {})
                entry[extension] = tar.extractfile(member).read()
                image_key = next((key for key in ["jpg", "jpeg", "png"] if key in entry), None)
                if "cls" not in entry or image_key is None:
                    continue
                label = int(entry["cls"].decode())
                if not 0 <= label < len(source_indices):
                    raise ValueError("Invalid ImageNet-A class index")
                data = entry[image_key]
                sha = hashlib.sha256(data).hexdigest()
                identifier = "imagenet-a-" + stem
                if identifier not in used_ids and sha not in used_hashes:
                    selected[label].append((rank(identifier), identifier, sha, data, image_key))
                    selected[label].sort(key=lambda item: item[0])
                    del selected[label][per_class:]
                del pending[stem]
            if pending:
                raise ValueError("Incomplete image/label pairs in shard")
    if set(selected) != set(range(200)) or any(len(v) != per_class for v in selected.values()):
        raise ValueError("Insufficient unseen images for every source class")
    identifiers, hashes = set(), set()
    rows = []
    # Check all selected inputs before writing any images.
    for values in selected.values():
        for _, identifier, sha, _, _ in values:
            if identifier in identifiers or sha in hashes:
                raise ValueError("Duplicate selected image ID or content")
            identifiers.add(identifier)
            hashes.add(sha)
    output.mkdir(parents=True)
    for label, values in sorted(selected.items()):
        for _, identifier, sha, data, extension in values:
            path = output / (sha + "." + extension)
            path.write_bytes(data)
            rows.append({"id": identifier, "task": "image", "dataset": "ImageNet-A",
                         "input": str(path.resolve()), "input_sha256": sha,
                         "label": classmap[str(source_indices[label])][0],
                         "source_class_index": label, "source_revision": revision})
    return rows


def prepare_evaluation(calibration, assets, output, asr_source=None, ocr_native=None):
    calibration, assets, output = map(lambda p: Path(p).resolve(), (calibration, assets, output))
    if output.exists():
        raise FileExistsError(output)
    declared = json.loads((calibration / "sample-manifests.json").read_text())["tasks"]
    samples = {}
    for task in ("text", "image", "ocr", "asr"):
        path = calibration / (task + ".jsonl")
        if digest(path) != declared[task]["sha256"]:
            raise ValueError(f"Changed calibration manifest: {task}")
        samples[task] = read_rows(path)
        for row in samples[task]:
            if task != "text" and digest(row["input"]) != row["input_sha256"]:
                raise ValueError(f"Changed calibration input: {row['id']}")
    qa_source = calibration / "squad.json"
    qa = select_qa(json.loads(qa_source.read_text()), samples["text"])
    asr_path = next((assets / "datasets/librispeech-test-other").glob("*/0000.parquet"))
    asr = audit_asr(asr_path, samples["asr"])
    meta_path = calibration / "imagenet-a-meta.json"
    meta = json.loads(meta_path.read_text())
    shards = sorted((assets / "datasets/imagenet-a" / meta["sha"] / "test").glob("*.tar"))
    if len(shards) != 4:
        raise ValueError("Expected all four pinned ImageNet-A shards")
    map_path = assets / "datasets/imagenet-class-index.json"
    index_path = calibration / "imagenet-a-index-map.json"
    output.mkdir(parents=True)
    images = select_images(shards, samples["image"], json.loads(map_path.read_text()),
                           json.loads(index_path.read_text()), output / "images", meta["sha"])
    entries = {"text": write(output / "text.jsonl", qa), "image": write(output / "image.jsonl", images)}
    extra_sources = []
    if asr_source is not None:
        asr_source = Path(asr_source).resolve()
        # A downloaded parquet must match its pinned LFS content hash.
        expected = "bcc1554c0d29e7e7f6902886ccdc089cc2cd3ce21b8396e62722bc7652cfcd06"
        if asr_source.parent.name != "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1" or digest(asr_source) != expected:
            raise ValueError("Unexpected dev-other source revision/content")
        speech = select_asr(asr_source, samples["asr"], output / "audio")
        entries["asr"] = write(output / "asr.jsonl", speech)
        extra_sources.append(asr_source)
    if ocr_native is not None:
        ocr_native = Path(ocr_native).resolve()
        if ocr_native.name != "41b3c746a20226fddc80d86d4b2a903d43b5be4f521dd1bbe759dbf8844745e2":
            raise ValueError("Unexpected native OCR source revision")
        ocr_path = next((assets / "datasets/sroie").glob("*/test.parquet"))
        pages, reference_audit = select_ocr(ocr_path, ocr_native, samples["ocr"], output / "pages")
        entries["ocr"] = write(output / "ocr.jsonl", pages)
        extra_sources.extend([ocr_path, ocr_native / "source-tree.json", ocr_native / "source-provenance.json"])
    report = {
        "schema": "gsdeploy.quality_inputs.v1", "status": "PARTIAL_DATA_PREPARED_NOT_ADMITTED",
        "gpu_started": False, "native_execution_ready": False, "utility": None,
        "calibration_manifest_sha256": digest(calibration / "sample-manifests.json"),
        "tasks": {
            "text": {**entries["text"], "status": "INPUTS_FROZEN_SCORING_REVIEW_PENDING",
                     "article_disjoint": True, "articles": sorted({r["article"] for r in qa}),
                     "answerable": sum(bool(r["answers"]) for r in qa),
                     "unanswerable": sum(not r["answers"] for r in qa),
                     "context_character_strata": dict(Counter(r["context_character_stratum"] for r in qa)),
                     "limitation": "Only six unused articles; balanced answerability and stratified contexts change natural weighting."},
            "image": {**entries["image"], "status": "INPUTS_FROZEN_SCORING_REVIEW_PENDING",
                      "id_and_content_disjoint": True, "source_classes": 200,
                      "per_source_class": 2, "candidate_labels": 1000,
                      "limitation": "Class-balanced ImageNet-A pressure test; not full ImageNet1K or its natural class weighting."},
            "ocr": {"status": "BLOCKED_MISSING_COMPLETE_NATIVE_TRANSCRIPTS", "samples": 0,
                    "root_cause": "Producer load_task1_2 uses split(',') then tokens[8], losing text after the first transcript comma.",
                    "producer_revision": "77c4c11490e3c4179ee22f86611569c71c04b5f9",
                    "producer_source": "https://github.com/jsdnrs/ICDAR2019-SROIE/blob/77c4c11490e3c4179ee22f86611569c71c04b5f9/to_jsonl.py",
                    "repair": "Use native eight-coordinate lines and split(',', 8); audit complete page reading order and ignored regions.",
                    "forbidden": "Do not infer gold text from model outputs or four-field entity labels."},
            "asr": asr,
        },
        "source_hashes": {str(p): digest(p) for p in [qa_source, meta_path, map_path, index_path, *shards]},
        "admission": "NO_INFERENCE_OR_MODEL_COMPARISON_PERFORMED",
    }
    if "asr" in entries:
        report["tasks"]["asr"] = {**entries["asr"], "status": "INPUTS_FROZEN_SCORING_REVIEW_PENDING",
                                   "dataset": "LibriSpeech-dev-other", "speaker_disjoint": True,
                                   "speakers": len({row["speaker_id"] for row in speech}),
                                   "duration_bins": [[4, 8], [8, 12], [12, 20], [20, 30]],
                                   "per_duration_bin": [200, 150, 120, 30], "prior_test_other_audit": asr,
                                   "limitation": "Duration/speaker stratified 4-30s dev-other with quotas 200/150/120/30; not natural duration weighting or test-other, no claim of training-data exclusion."}
    if "ocr" in entries:
        report["tasks"]["ocr"] = {**entries["ocr"], "status": "REFERENCES_RESTORED_SCORING_AUDIT_PENDING",
                                   "reference_audit": reference_audit,
                                   "source": "https://github.com/mindee/doctr/blob/main/doctr/datasets/sroie.py"}
    report["source_hashes"].update({str(p): digest(p) for p in extra_sources})
    if len(entries) == 4:
        report["status"] = "FOUR_TASK_INPUTS_FROZEN_PROTOCOL_REVIEW_PENDING_NOT_ADMITTED"
    (output / "evaluation-status.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asr-source", help="Verified unused dev-other parquet")
    parser.add_argument("--ocr-native", help="Verified native annotation directory")
    args = parser.parse_args()
    print(json.dumps(prepare_evaluation(args.calibration, args.assets, args.output, args.asr_source, args.ocr_native), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
