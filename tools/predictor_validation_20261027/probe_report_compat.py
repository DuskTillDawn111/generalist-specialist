"""Read old sealed probe predictions without rewriting any frozen artifact.

The first sealer omitted two optional reporting group labels. Reconstruct only
those labels in memory from the frozen model lookup and frozen plan. The base
reporter and independent assessor still verify and assess the ORIGINAL sealed
prediction bytes. No timing, baseline, UNKNOWN reason, or forecast is changed.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path
from unittest.mock import patch

from gsdeploy.forecasting import ProfileProvider
from . import probe_validation as base
from .profiling import RESIDENTS, digest


GROUP_FIELDS = {"input_known_to_model", "timing_pattern"}


def report(sealed_dir, raw_path, output, *, failed_output=None):
    sealed_dir, output = Path(sealed_dir).resolve(), Path(output).resolve()
    sealing_path = sealed_dir / "sealing.json"
    sealing = base.read(sealing_path)
    original_prediction = base.read(sealing["prediction"]["path"])
    model = base.read(sealing["model"]["path"])
    plan = base.read(sealing["plan"]["path"])
    provider = ProfileProvider(model["document"], original_prediction["execution_context_sha256"])
    jobs = {job["id"]: job for job in plan["jobs"]}
    reconstructed = copy.deepcopy(original_prediction)
    added = collections.Counter()
    for row in reconstructed["requests"]:
        if "input_known_to_model" not in row:
            row["input_known_to_model"] = (RESIDENTS[row["alias"]], row["task"], row["pool_id"]) in provider.lookup
            added["input_known_to_model"] += 1
        if "timing_pattern" not in row:
            offsets = jobs[row["job_id"]]["start_offsets_ms"]
            row["timing_pattern"] = ",".join(a + "@" + str(t) for a, t in
                                               sorted(offsets.items(), key=lambda x: (x[1], x[0])))
            added["timing_pattern"] += 1
    stripped = copy.deepcopy(reconstructed)
    for original, row in zip(original_prediction["requests"], stripped["requests"]):
        for key in GROUP_FIELDS:
            if key not in original:
                row.pop(key, None)
    if digest(stripped) != digest(original_prediction):
        raise ValueError("Compatibility view changed sealed numerical or protocol data")
    failed = Path(failed_output).resolve() if failed_output else output.parent / "report-v1"
    history = {"path": str(failed), "exists": failed.exists(),
               "reason": "Original reporting adapter raised KeyError(input_known_to_model); no prediction changes"}
    if failed.is_dir():
        history["files"] = [base.receipt(path) for path in sorted(failed.rglob("*")) if path.is_file()]
    compatibility = {
        "schema": "gsdeploy.probe_report_metadata_compatibility.v1",
        "wrapper_source": base.receipt(__file__),
        "base_reporter_source": base.receipt(base.__file__),
        "sealing": base.receipt(sealing_path),
        "original_prediction": sealing["prediction"],
        "reconstructed_group_fields": dict(added), "reconstruction_scope": "Reporting group labels in memory only",
        "sealed_prediction_bytes_changed": False, "forecast_values_changed": False,
        "historical_failed_report": history,
        "seal_has_original_sealer_source_receipt": "sealer_source" in sealing,
    }
    original_read, original_save = base.read, base.save
    prediction_path = Path(sealing["prediction"]["path"]).resolve()

    def read_compat(path):
        # File receipts use base.receipt(), which is untouched. The independent
        # assessor reads original artifacts itself through validation._artifact.
        if Path(path).resolve() == prediction_path:
            return copy.deepcopy(reconstructed)
        return original_read(path)

    def save_report(path, value):
        if Path(path).name == "report.json":
            value["metadata_compatibility"] = compatibility
            value["unknown_reason_counts"] = dict(collections.Counter(
                row.get("unknown_reason") or "KNOWN" for row in original_prediction["requests"]))
        original_save(path, value)

    with patch.object(base, "read", read_compat), patch.object(base, "save", save_report):
        result = base.report(sealed_dir, raw_path, output)
    if base.receipt(prediction_path) != sealing["prediction"]:
        raise ValueError("Original sealed prediction changed during report generation")
    base.save(output / "compatibility-receipt.json", compatibility)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-dir", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failed-output")
    args = parser.parse_args(argv)
    result = report(args.sealed_dir, args.raw, args.output, failed_output=args.failed_output)
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()),
                      "validation_cases": result["validation_case_count"], "point_coverage": result["point_coverage"],
                      "GPU_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
