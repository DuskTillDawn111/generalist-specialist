"""Aggregate raw request evidence; never turn profiles into admission passes."""

import argparse
import collections
import csv
import json
from pathlib import Path
import re
import statistics
import string
import unicodedata

from .runner import quantile, save


def edit_distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def text(value):
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def speech_text(value):
    return text(value).translate(str.maketrans("", "", string.punctuation))


def qa_text(value):
    # T5's native vocabulary represents '<' as <unk>; skipping special tokens
    # decodes the training target '<No Answer>' as 'no answer>'. This is a
    # tokenizer-derived protocol alias, not a fitted rejection threshold.
    if text(value) in {"<no answer>", "no answer>"}:
        return ""
    value = text(value).translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def qa_score(prediction, references, success=True):
    if not success:
        return 0.0, 0.0
    prediction = qa_text(prediction)
    exact, f1 = [], []
    for reference in references or [""]:
        reference = qa_text(reference)
        exact.append(float(prediction == reference))
        a, b = prediction.split(), reference.split()
        if not a or not b:
            f1.append(float(a == b))
        else:
            overlap = sum((collections.Counter(a) & collections.Counter(b)).values())
            f1.append(2 * overlap / (len(a) + len(b)))
    return max(exact), max(f1)


def quality(task, records):
    # Repeated performance requests are not additional independent quality samples.
    unique = {}
    for row in records:
        unique.setdefault(row["sample_id"], row)
    rows = list(unique.values())
    result = {"unique_samples": len(rows), "scope": "calibration_diagnostic_not_admission"}
    if task == "text":
        scores = [qa_score(r["prediction"], r["answers"], r["success"]) for r in rows]
        result.update({"EM": statistics.mean(v[0] for v in scores), "F1": statistics.mean(v[1] for v in scores)})
        for name, group in [("answerable", [r for r in rows if r["answers"]]),
                            ("unanswerable", [r for r in rows if not r["answers"]])]:
            values = [qa_score(r["prediction"], r["answers"], r["success"]) for r in group]
            result[name] = {"samples": len(values), "EM": statistics.mean(v[0] for v in values) if values else None}
    elif task == "image":
        result["top1_accuracy"] = statistics.mean(bool(r["success"] and r["correct"]) for r in rows)
    elif task == "ocr":
        errors = sum(edit_distance(text(r["prediction"]) if r["success"] else "", text(r["reference"])) for r in rows)
        total = sum(len(text(r["reference"])) for r in rows)
        result.update({"CER": errors / total, "character_errors": errors, "reference_characters": total})
    elif task == "asr":
        errors = sum(edit_distance(speech_text(r["prediction"]).split() if r["success"] else [],
                                   speech_text(r["reference"]).split()) for r in rows)
        total = sum(len(speech_text(r["reference"]).split()) for r in rows)
        result.update({"WER": errors / total, "word_errors": errors, "reference_words": total})
    elif task == "lid":
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row["reference"]].append(bool(row["success"] and row["prediction"] == row["reference"]))
        per_language = {k: statistics.mean(v) for k, v in groups.items()}
        result.update({"macro_accuracy": statistics.mean(per_language.values()), "per_language_accuracy": per_language})
    result["failed_or_truncated_samples"] = sum(not r["success"] for r in rows)
    return result


def aggregate(root):
    root = Path(root)
    state = json.loads((root / "suite-status.json").read_text())
    profiles, rows = [], []
    for job in state["jobs"]:
        if job["status"] != "COMPLETE":
            continue
        profile = json.loads((Path(job["path"]) / "profile.json").read_text())
        profiles.append(profile)
        phases = sorted({r["task"] for r in profile["rounds"]})
        for phase in phases:
            rounds = [r for r in profile["rounds"] if r["task"] == phase]
            records = [json.loads(line) for r in rounds for line in Path(r["raw_records"]).read_text().splitlines()]
            service = [r["service_ms"] for r in records]
            client = [r["client_ms"] for r in records]
            wall = sum(r["wall_seconds"] for r in rounds)
            # Harmonize early counter-preferred runs from their preserved power
            # integrals. Never rewrite their raw profiles or telemetry.
            energy_values = [r["board_energy"]["power_integral_joules"] for r in rounds]
            joules = sum(energy_values) if all(v is not None for v in energy_values) else None
            counter_values = [r["board_energy"]["counter_joules"] for r in rounds]
            row = {
                "model": profile["model"]["id"], "replicas": profile["replicas"], "task": phase,
                "rounds": len(rounds), "requests": len(records), "successful": sum(r["success"] for r in records),
                "resident_mib": profile["resident_after_warmup_mib"], "peak_board_mib": profile["peak_board_used_mib"],
                "safe_memory_policy_passes": profile["peak_board_used_mib"] * profile["safe_factor"] <= profile["budget_mib"],
                "service_mean_ms": statistics.mean(service), "service_p50_ms": quantile(service, .5),
                "service_p95_ms": quantile(service, .95), "service_p99_ms": quantile(service, .99),
                "client_p95_ms": quantile(client, .95), "client_p99_ms": quantile(client, .99),
                "completed_rps": len(records) / wall, "successful_rps": sum(r["success"] for r in records) / wall,
                "board_joules_per_offered_request": joules / len(records) if joules is not None else None,
                "mean_board_watts": joules / wall if joules is not None else None,
                "startup_seconds": profile["startup_seconds"], "startup_joules": profile["startup_energy"]["power_integral_joules"],
                "energy_method": "nvml_board_power_integral",
                "counter_joules_per_request_diagnostic": sum(counter_values) / len(records)
                    if all(v is not None for v in counter_values) else None,
                "energy_coverage_valid": all(v is not None for v in energy_values) and not profile["telemetry_errors"],
                "quality": quality(phase, records) if "mix" not in phase else {
                    task: quality(task, [r for r in records if r["task"] == task])
                    for task in {r["task"] for r in records}
                },
                "profile_path": job["path"],
            }
            rows.append(row)
    for row in rows:
        baseline = next((r for r in rows if r["model"] == row["model"] and r["task"] == row["task"] and r["replicas"] == 1), None)
        row["throughput_scaling_vs_one"] = row["completed_rps"] / baseline["completed_rps"] if baseline else None
        row["per_replica_capacity_factor"] = row["throughput_scaling_vs_one"] / row["replicas"] if baseline else None
    result = {"status": state["status"], "profiles": rows,
              "excluded_or_failed_jobs": [j for j in state["jobs"] if j["status"] != "COMPLETE"],
              "admission": "NOT_CERTIFIED_AGAINST_THREE_GENERALISTS",
              "utility_integration": "standalone measurements only; not additive whole-plan energy or colocation evidence"}
    save(root / "summary.json", result)
    if rows:
        columns = [k for k in rows[0] if k != "quality"]
        with (root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    result = aggregate(args.root)
    print(json.dumps({"status": result["status"], "profile_rows": len(result["profiles"])}, indent=2))
