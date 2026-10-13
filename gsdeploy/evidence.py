"""Checkpoint fingerprints and recorded quality admission, without model imports."""

import hashlib
import json
import math
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checkpoint(path):
    """Hash all checkpoint files, excluding download cache and Git metadata."""
    root = Path(path)
    if root.is_file():
        files = {root.name: digest(root)}
    elif root.is_dir():
        files = {
            str(p.relative_to(root)): digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and not {".cache", ".git", "__pycache__"}.intersection(
                p.relative_to(root).parts
            )
        }
    else:
        raise ValueError(f"Checkpoint does not exist: {root}")
    if not files:
        raise ValueError(f"Checkpoint is empty: {root}")
    identity = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"files": files, "sha256": identity}


def verify_admission(config, winner, workers, manifest_path=None):
    if manifest_path is None:
        return {"verified": False, "reason": "NO_ADMISSION_MANIFEST", "pairs": []}
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported admission manifest schema")
    pairs = []
    identities = {}
    for model in winner["deployment"]:
        expected = manifest["models"][model]
        actual = checkpoint(workers[model]["env"]["GS_MODEL_PATH"])
        if actual != expected["checkpoint"]:
            raise ValueError(f"Checkpoint fingerprint mismatch: {model}")
        identities[model] = actual["sha256"]
        report_path = Path(expected["quality_report"])
        if not report_path.is_absolute():
            report_path = path.parent / report_path
        if digest(report_path) != expected["quality_report_sha256"]:
            raise ValueError(f"Quality report fingerprint mismatch: {model}")
        report = json.loads(report_path.read_text())
        if report["model"] != model or report["checkpoint_sha256"] != actual["sha256"]:
            raise ValueError(f"Quality report belongs to another checkpoint: {model}")
        for task, route in winner["routing"].items():
            if route.get(model, 0) <= 0:
                continue
            record = report["tasks"][task]
            rule = expected["admission_rules"][task]
            if not record.get("dataset") or not record.get("protocol"):
                raise ValueError(f"Missing quality dataset/protocol: {model}/{task}")
            value, threshold = record["value"], rule["threshold"]
            if any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (value, threshold)
            ):
                raise ValueError(f"Invalid quality measurement: {model}/{task}")
            direction = rule["direction"]
            if record["metric"] != rule["metric"] or direction not in {"min", "max"}:
                raise ValueError(f"Invalid admission rule: {model}/{task}")
            admitted = value >= threshold if direction == "min" else value <= threshold
            if not admitted:
                raise ValueError(f"Quality admission failed: {model}/{task}")
            pairs.append(
                {"model": model, "task": task, "measurement": record, "rule": rule}
            )
    return {
        "verified": True,
        "manifest_sha256": digest(path),
        "checkpoints": identities,
        "pairs": pairs,
        "scope": "checkpoint hashes and supplied quality records; quality inference not rerun",
    }


def admission_covers(raw):
    """Recheck coverage and recorded thresholds for offline evaluation."""
    evidence = raw.get("admission_evidence", {})
    if evidence.get("verified") is not True or not evidence.get("manifest_sha256"):
        return False
    needed = {
        (model, task)
        for task, route in raw["plan"]["routing"].items()
        for model, rate in route.items()
        if rate > 0
    }
    admitted = set()
    try:
        for pair in evidence["pairs"]:
            model, task = pair["model"], pair["task"]
            if not evidence["checkpoints"].get(model):
                return False
            measurement, rule = pair["measurement"], pair["rule"]
            value, threshold = measurement["value"], rule["threshold"]
            if any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (value, threshold)
            ):
                return False
            if (
                measurement["metric"] != rule["metric"]
                or not measurement.get("dataset")
                or not measurement.get("protocol")
            ):
                return False
            if rule["direction"] == "min" and value >= threshold:
                admitted.add((model, task))
            elif rule["direction"] == "max" and value <= threshold:
                admitted.add((model, task))
        return needed.issubset(admitted) and bool(needed)
    except (KeyError, TypeError):
        return False
