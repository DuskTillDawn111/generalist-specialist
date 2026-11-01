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


def checkpoint(path, selected_files=None):
    """Hash all checkpoint files, excluding download cache and Git metadata."""
    root = Path(path)
    if selected_files is not None:
        if not root.is_dir() or not selected_files or len(selected_files) != len(set(selected_files)):
            raise ValueError("Invalid selected checkpoint files")
        files = {}
        for relative in selected_files:
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                raise ValueError("Selected checkpoint file is outside the model root or unavailable")
            files[str(target.relative_to(root.resolve()))] = digest(target)
    elif root.is_file():
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
    if config.weights.get("Q", 0) > 0:
        return verify_quality_evidence(config, winner, workers, manifest_path)
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
        selected = workers[model]["env"].get("GS_CHECKPOINT_FILES_JSON")
        actual = checkpoint(workers[model]["env"]["GS_MODEL_PATH"], json.loads(selected) if selected else None)
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


def verify_quality_evidence(config, winner, workers, manifest_path=None):
    """New study identity check: accuracy is scored, never thresholded.

    Native evidence requires full checkpoint fingerprints. A configuration-only
    identity is useful for CPU prediction but cannot certify native inference.
    """
    from .quality import _document, _path, quality_digest

    if manifest_path is None:
        return {"verified": False, "reason": "NO_QUALITY_EVIDENCE_MANIFEST", "pairs": []}
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 2 or manifest.get("contract") != "quality_evidence_v1":
        raise ValueError("Six-term studies require quality_evidence_v1 manifest schema 2")
    quality = _document(config.catalog)
    if manifest.get("quality_sha256") != quality_digest(quality):
        raise ValueError("Quality catalog fingerprint mismatch")
    pairs, identities, reports = [], {}, []
    for model in winner["deployment"]:
        expected = manifest["models"][model]
        selected = workers[model]["env"].get("GS_CHECKPOINT_FILES_JSON")
        actual = checkpoint(workers[model]["env"]["GS_MODEL_PATH"], json.loads(selected) if selected else None)
        if actual != expected["checkpoint"]:
            raise ValueError(f"Checkpoint fingerprint mismatch: {model}")
        identities[model] = actual["sha256"]
        if not expected.get("quality_reports"):
            raise ValueError(f"Missing quality source reports: {model}")
        for report in expected["quality_reports"]:
            report_path = Path(report["path"])
            if not report_path.is_absolute():
                report_path = path.parent / report_path
            if digest(report_path) != report["sha256"]:
                raise ValueError(f"Quality report fingerprint mismatch: {model}")
            reports.append({"model": model, **report})
        for task, row in winner["routing"].items():
            if row.get(model, 0) <= 0:
                continue
            record = _path(quality, model, task)
            recorded_checkpoint = record.get("checkpoint_sha256")
            identity = record.get("checkpoint_identity", {})
            if recorded_checkpoint is None and identity.get("kind") == "checkpoint_files":
                recorded_checkpoint = identity.get("sha256")
            runtime_only = (
                recorded_checkpoint is None
                and identity.get("kind") == "configuration_manifest"
                and expected.get("runtime_checkpoint_scope") == "current_execution_only_not_retroactive_offline_hash"
                and expected.get("historical_quality_checkpoint_identities", {}).get(task) == identity
                and bool(expected.get("receipt_checks"))
                and all(r.get("current_sha256") and
                        (not r.get("historical_sha256") or
                         r["current_sha256"] == r["historical_sha256"])
                        for r in expected["receipt_checks"])
            )
            if recorded_checkpoint != actual["sha256"] and not runtime_only:
                raise ValueError(f"Quality/runtime checkpoint identity mismatch: {model}/{task}")
            pairs.append({"model": model, "task": task, "measurement": record,
                          "runtime_checkpoint_sha256": actual["sha256"],
                          "offline_full_checkpoint_equivalence_verified": not runtime_only})
    return {"verified": True, "contract": "quality_evidence_v1",
            "manifest_sha256": digest(path), "quality_sha256": quality_digest(quality),
            "checkpoints": identities, "pairs": pairs, "reports": reports,
            "scope": "current runtime checkpoint, source reports and frozen quality identity; configuration receipts do not retroactively prove offline full-weight identity; actual outputs must be rescored"}


def quality_covers(config, raw):
    """Recheck all routed identities without imposing accuracy thresholds."""
    from .quality import _document, _path, quality_digest

    evidence = raw.get("admission_evidence", {})
    try:
        quality = _document(config.catalog)
        if (evidence.get("verified") is not True
                or evidence.get("contract") != "quality_evidence_v1"
                or not evidence.get("manifest_sha256")
                or not evidence.get("reports")
                or evidence.get("quality_sha256") != quality_digest(quality)):
            return False
        needed = {(m, t) for t, row in raw["plan"]["routing"].items()
                  for m, q in row.items() if q > 0}
        covered = set()
        for pair in evidence["pairs"]:
            m, t = pair["model"], pair["task"]
            record = _path(quality, m, t)
            identity = record.get("checkpoint_identity", {})
            checkpoint_hash = record.get("checkpoint_sha256") or (
                identity.get("sha256") if identity.get("kind") == "checkpoint_files" else None)
            current_hash = evidence["checkpoints"].get(m)
            runtime_bound = (pair.get("runtime_checkpoint_sha256") == current_hash and bool(current_hash)
                             and identity.get("kind") == "configuration_manifest"
                             and pair.get("offline_full_checkpoint_equivalence_verified") is False)
            if (((checkpoint_hash and checkpoint_hash == current_hash) or runtime_bound)
                    and pair["measurement"] == record):
                covered.add((m, t))
        return bool(needed) and needed.issubset(covered)
    except (KeyError, TypeError, ValueError):
        return False
