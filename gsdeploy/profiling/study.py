"""Freeze the compact four-task study without activating unmeasured models."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from .prepare import digest
from .runner import save


ACTIVE_TASKS = ("text", "image", "ocr", "asr")


def validate_study(study):
    if study.get("schema") != "gsdeploy.four_task_study.v1":
        raise ValueError("Unknown study schema")
    if tuple(study.get("active_tasks", [])) != ACTIVE_TASKS:
        raise ValueError("The compact study requires text/image/ocr/asr only")
    if set(study.get("tasks", {})) != set(ACTIVE_TASKS):
        raise ValueError("Study task definitions do not match the active tasks")
    if len(study.get("generalists", [])) != 3:
        raise ValueError("Screening requires all three generalist references")
    specialists = study.get("specialists", [])
    if len(specialists) != 4 or {s["slot"] for s in specialists} != set(ACTIVE_TASKS):
        raise ValueError("One specialist candidate per active task is required")
    for candidate in study["generalists"] + specialists:
        if not re.fullmatch(r"[0-9a-f]{40}", candidate.get("revision", "")):
            raise ValueError("Every candidate requires an immutable revision")
        if candidate.get("state") != "METADATA_SCREENED_NOT_ADMITTED":
            raise ValueError("This preparation entry cannot grant model admission")
    return study


def prepare_study(config, source, output):
    config, source, output = Path(config), Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    study = validate_study(json.loads(config.read_text()))
    declared = json.loads((source / "sample-manifests.json").read_text())["tasks"]
    manifests, counts = {}, {}
    for task in ACTIVE_TASKS:
        path = source / (task + ".jsonl")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != declared[task]["sha256"]:
            raise ValueError(f"Source manifest changed: {task}")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"Empty task manifest: {task}")
        expected = study["tasks"][task]["dataset"]
        for row in rows:
            if row.get("task") != task or row.get("dataset") != expected:
                raise ValueError(f"Wrong task or dataset protocol: {task}")
            if task == "text":
                if not all(key in row for key in ["context", "question", "answers"]):
                    raise ValueError("QA requires context/question/answers, not an old classifier input")
            else:
                if digest(row["input"]) != row["input_sha256"]:
                    raise ValueError(f"Input changed: {row['id']}")
        manifests[task] = raw
        counts[task] = len(rows)
    output.mkdir(parents=True)
    entries = {}
    for task, raw in manifests.items():
        path = output / (task + ".jsonl")
        path.write_bytes(raw)
        entries[task] = {"path": str(path), "sha256": digest(path), "samples": counts[task]}
    # The speech adapter's input stream must no longer include language ID.
    (output / "speech.jsonl").write_bytes(manifests["asr"])
    (output / "samples.jsonl").write_bytes(
        b"".join(manifests[t].rstrip(b"\r\n") + b"\n" for t in ACTIVE_TASKS)
    )
    save(output / "study.json", copy.deepcopy(study))
    save(output / "sample-manifests.json", {
        "kind": "performance_calibration_not_quality_admission",
        "source": str(source), "tasks": entries,
    })
    resident = study["memory_targets_gib"]["all_four_specialists_single_copy_sum"]
    replicas = study["memory_targets_gib"]["replicas_per_specialist_target"]
    result = {
        "status": "PREPARED_NOT_ADMITTED", "active_tasks": list(ACTIVE_TASKS),
        "sample_counts": counts, "excluded_tasks": study["excluded_tasks"],
        "candidate_generalists": len(study["generalists"]),
        "candidate_specialists": len(study["specialists"]),
        "gpu_started": False, "native_execution_ready": False, "utility": None,
        "design_only_safe_gib_all_specialists_with_replicas": resident * replicas * study["safe_factor"],
        "budget_gib": study["gpu_memory_mib"] / 1024,
        "remaining_requirements": [
            "Verify candidate compact inference implementations and actual residence/peak memory",
            "Repair OCR scoring and freeze task protocols and independent quality sets",
            "Measure strict quality and energy comparisons against all three generalists",
            "Measure replica/colocation capacity and whole-plan energy",
            "Freeze new-task normalization references and startup amortization",
            "Build a runtime catalog from verified evidence; never reuse old-task numbers",
        ],
    }
    save(output / "study-status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--source", required=True, help="Existing frozen calibration directory")
    parser.add_argument("--output", required=True, help="Fresh preparation directory")
    args = parser.parse_args()
    print(json.dumps(prepare_study(args.config, args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
