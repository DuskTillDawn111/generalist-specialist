"""Descriptive reanalysis of existing validation; never grants admission.

Aggregate latency bias is deliberately distinct from a percentile of request
errors. Old reports/decisions remain unchanged. All quantiles here use linear
interpolation. Inputs are already inspected development evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


def receipt(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def metrics(rows):
    observed = [r["observed_e2e_ms"] for r in rows]
    predicted = [r["predicted_e2e_ms"] for r in rows]
    if not observed or any(not math.isfinite(v) or v <= 0 for v in observed):
        raise ValueError("Empty cohort or invalid observed denominator")
    if any(not math.isfinite(v) or v < 0 for v in predicted):
        raise ValueError("Invalid prediction")
    errors = [abs(p - o) / o for p, o in zip(predicted, observed)]
    return {
        "n": len(rows),
        "observed_mean_ms": mean(observed),
        "predicted_mean_ms": mean(predicted),
        "mean_signed_relative_bias": mean(predicted) / mean(observed) - 1,
        "observed_p95_ms": quantile(observed, .95),
        "predicted_p95_ms": quantile(predicted, .95),
        "p95_signed_relative_bias": quantile(predicted, .95) / quantile(observed, .95) - 1,
        "request_mare": mean(errors),
        "request_error_p95": quantile(errors, .95),
    }


def review(path):
    original = json.loads(path.read_text())
    all_rows, runs, receipts = [], [], [receipt(path)]
    for run in original["runs"]:
        raw_name, expected_sha = run["source"].rsplit(":sha256:", 1)
        raw_path = Path(raw_name)
        raw_receipt = receipt(raw_path)
        if raw_receipt["sha256"] != expected_sha:
            raise ValueError(f"Changed raw source: {raw_path}")
        receipts.append(raw_receipt)
        raw = json.loads(raw_path.read_text())
        records = raw["request_records"]
        lookup = {r["request_id"]: r for r in records}
        rows = [dict(r) for b in run["batches"] for r in b.get("requests", [])]
        ids = [r["request_id"] for r in rows]
        if (len(lookup) != len(records) or len(set(ids)) != len(ids)
                or set(ids) != set(lookup) or run.get("unsupported")):
            raise ValueError("Incomplete, unsupported or duplicate cohort")
        for row in rows:
            record = lookup[row["request_id"]]
            if record["status"] != "completed":
                raise ValueError("Uncompleted request in cohort")
            row["task"] = record["task"]
        tasks = sorted({r["task"] for r in rows})
        runs.append({
            "run_id": run["run_id"],
            "execution_context_sha256": raw["execution_context_sha256"],
            "metrics": {t: metrics([r for r in rows if t == "ALL" or r["task"] == t])
                        for t in ["ALL", *tasks]},
            "original_request_latency_errors": run["latency_errors"],
            "conditional_fifo_energy": run.get("fifo_energy"),
        })
        all_rows.extend(rows)
    tasks = sorted({r["task"] for r in all_rows})
    return {
        "sources": receipts,
        "original_development_accuracy_passed": original["development_accuracy_passed"],
        "original_reasons": original["reasons"],
        "runs": runs,
        "pooled_descriptive_metrics": {
            t: metrics([r for r in all_rows if t == "ALL" or r["task"] == t])
            for t in ["ALL", *tasks]},
        "full_episode_energy_accuracy": None,
        "continuous_arrival_throughput_accuracy": None,
        "candidate_ranking_accuracy": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "gsdeploy.aggregate_accuracy_descriptive_review.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-hoc aggregate review of historical controlled FIFO batches, not fresh qualification",
        "quantile_method": "linear interpolation at (n-1)*q; old report quantiles retained separately",
        "script": receipt(Path(__file__)),
        "admission_enabled": False,
        "physical_predictor_qualified": False,
        "models": [review(p) for p in args.reports],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
