"""Prospectively seal fixed native probes and assess one independent run.

CPU only. Scheduled offsets are declared experimental inputs, not hidden online
future. One collected run remains one validation case. Strict exact-v1 UNKNOWN
coverage is retained, including unseen inputs and unsupported overlap contexts.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import time

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting import FullHorizonModel, ProfileProvider
from .profiling import PLAN_SCHEMA, RESIDENTS, digest
from .validation import (OBSERVATION_SCHEMA, PREDICTION_SCHEMA, SCHEMA,
                         assess, errors, percentile)


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, document):
    with Path(path).open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write("\n")


def _extract_model(value):
    for key in ("fitted_document", "document"):
        if isinstance(value.get(key), dict):
            return value[key]
    if value.get("schema") == "gsdeploy.segment_resource_profiles.v1":
        return value
    return None


def _jobs(plan):
    if plan.get("schema") != PLAN_SCHEMA or not plan.get("execution_context_sha256") or not plan.get("jobs"):
        raise ValueError("Nonempty independent native profile plan required")
    ids = set()
    for job in plan["jobs"]:
        if job.get("split") != "validation" or job.get("id") in ids:
            raise ValueError("Unique validation-only jobs required")
        ids.add(job["id"])
        if set(job["inputs"]) != set(job["start_offsets_ms"]) or set(job["inputs"]) - set(RESIDENTS):
            raise ValueError("Declared input and timing aliases must match")
    return ids


def _cohort(plan):
    return digest([{k: job[k] for k in ("id", "inputs", "start_offsets_ms", "split")} for job in plan["jobs"]])


def seal(model_path, training_raw_path, plan_path, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan, training = read(plan_path), read(training_raw_path)
    jobs = _jobs(plan)
    if training.get("schema") != "gsdeploy.native_phase_collection.v1" or training.get("status") != "completed":
        raise ValueError("Complete, separately identified native training run required")
    trained_jobs = {row["profile_job_id"] for row in training["request_records"] if row.get("split") == "training"}
    if not trained_jobs or trained_jobs & jobs:
        raise ValueError("Validation jobs must be disjoint from actual training jobs")
    if training["execution_context_sha256"] != plan["execution_context_sha256"]:
        raise ValueError("Training and independent plan execution context mismatch")
    original = read(model_path)
    document = _extract_model(original)
    provider, provider_reason = None, "EMPTY_OR_UNAVAILABLE_TRAINED_PROFILES"
    if document is not None:
        try:
            provider = ProfileProvider(document, plan["execution_context_sha256"])
            provider_reason = None
        except ValueError as exc:
            provider_reason = "INVALID_TRAINED_PROFILE:" + str(exc)
    # Freeze a model adapter with one auditable original model receipt. The
    # complete native provenance remains inside the source artifact.
    model_adapter = {"schema": "gsdeploy.frozen_probe_model.v1", "source": receipt(model_path),
                     "document": document, "provider_status": provider_reason or "READY"}
    model_ref_path = output / "model.json"
    save(model_ref_path, model_adapter)
    model_ref = receipt(model_ref_path)
    tasks = sorted({"ocr" if a == "ocr#0" else "text" for j in plan["jobs"] for a in j["inputs"]})
    protocol = {"weights": dict.fromkeys("QTLPME", 1 / 6), "utility_tie_tolerance": 0.,
                "required_categories": ["service", "p95"],
                "planned_cases": [{"id": "independent-native-run", "categories": ["service", "p95"],
                                   "expected_tasks": tasks}],
                "acceptance": {"service_mare": {"operator": "<=", "value": .2},
                               "p95_mare": {"operator": "<=", "value": .2}},
                "coverage_required": 1., "scope": "fixed offset probes; no full-horizon arrival/utility validation"}
    protocol_path = output / "protocol.json"
    save(protocol_path, protocol)
    frozen = time.perf_counter_ns()
    aliases = {a: {"family": family, "supported_tasks": ["ocr" if family == "ocr" else "text"]}
               for a, family in RESIDENTS.items()}
    predicted, job_reports = [], []
    sizes = {(r["task"], r["pool_id"]): r["input_size"] for r in plan["inputs"]}
    for job in plan["jobs"]:
        arrivals, queues, routes, metadata = [], {a: [] for a in RESIDENTS}, {}, []
        for alias, offset in job["start_offsets_ms"].items():
            task = "ocr" if alias == "ocr#0" else "text"
            rid = job["id"] + ":" + alias
            row = {"request_id": rid, "task": task, "pool_id": job["inputs"][alias],
                   "arrival_ns": round(offset * 1e6)}
            routes[rid] = alias
            (queues[alias] if row["arrival_ns"] == 0 else arrivals).append(row)
            profile = provider.lookup.get((RESIDENTS[alias], task, row["pool_id"])) if provider else None
            solo = sum(s["duration_ms"]["mean"] for s in profile["segments"]) if profile else None
            metadata.append({"request_id": rid, "task": task, "pool_id": row["pool_id"], "alias": alias,
                             "service_ms": None, "wait_ms": None, "e2e_ms": None, "solo_ms": solo,
                             "job_id": job["id"], "context": job["context"],
                             "validation_view": job.get("validation_view", "unspecified"),
                             "input_known_to_model": profile is not None,
                             "timing_pattern": ",".join(a + "@" + str(t) for a, t in
                                                        sorted(job["start_offsets_ms"].items(), key=lambda x: (x[1], x[0]))),
                             "input_size": sizes[(task, row["pool_id"])], "offset_ms": offset})
        reason, result = provider_reason, None
        if provider:
            try:
                last = max(round(x * 1e6) for x in job["start_offsets_ms"].values())
                deadline = last + 1
                result = FullHorizonModel(aliases, provider).forecast(
                    queues, 0, deadline, future_arrivals=arrivals,
                    route_future=lambda row, pending, now: routes[row["request_id"]],
                    require_energy=False, require_memory=False, drain_deadline_ns=deadline + 300_000_000_000)
                if result["incomplete_ids"]:
                    reason = "DRAIN_CENSORED"
            except (UnknownEvidence, ValueError) as exc:
                reason = type(exc).__name__ + ":" + str(exc)
        for row in metadata:
            finish = result["finishes"].get(row["request_id"]) if result else None
            if finish:
                arrival = round(row["offset_ms"] * 1e6)
                row.update(service_ms=(finish["finish_ns"] - finish["start_ns"]) / 1e6,
                           wait_ms=(finish["start_ns"] - arrival) / 1e6,
                           e2e_ms=(finish["finish_ns"] - arrival) / 1e6)
            row["unknown_reason"] = reason if row["service_ms"] is None else None
            predicted.append(row)
        job_reports.append({"job_id": job["id"], "context": job["context"], "unknown_reason": reason,
                            "request_count": len(metadata),
                            "known_count": sum(r["service_ms"] is not None for r in metadata)})
    p95 = {}
    for task in tasks:
        rows = [r for r in predicted if r["task"] == task]
        p95[task] = percentile([r["e2e_ms"] for r in rows]) if all(r["e2e_ms"] is not None for r in rows) else None
    created = time.perf_counter_ns()
    prediction = {"schema": PREDICTION_SCHEMA, "model_sha256": model_ref["sha256"],
                  "protocol_sha256": receipt(protocol_path)["sha256"],
                  "execution_context_sha256": plan["execution_context_sha256"],
                  "offered_cohort_sha256": _cohort(plan), "created_ns": created,
                  "origin_ns": frozen, "information_cutoff_ns": frozen,
                  "mode": "declared_plan", "schedule_declared_before_origin": True,
                  "actual_future_trace_used": False, "requests": predicted,
                  "p95_by_task": p95, "queue_samples": [], "candidates": [],
                  "scope": "Prospective fixed offset native probes; not online random-arrival queue validation",
                  "slowdown_reference": "Frozen training solo baseline, not independent validation solo ground truth"}
    save(output / "prediction.json", prediction)
    sealing = {"schema": "gsdeploy.native_probe_sealing.v1", "created_ns": created,
               "model": model_ref, "model_source": receipt(model_path),
               "training_raw": receipt(training_raw_path), "training_run_id": str(Path(training_raw_path).resolve().parent),
               "protocol": receipt(protocol_path), "plan": receipt(plan_path),
               "prediction": receipt(output / "prediction.json"), "frozen_ns": frozen,
               "sealer_source": receipt(__file__),
               "job_reports": job_reports, "GPU_started": False,
               "pending_categories": {"full_queue": "NOT_RUN", "delta_U": "NOT_RUN", "ranking": "NOT_RUN"}}
    save(output / "sealing.json", sealing)
    return sealing


def report(sealed_dir, raw_path, output):
    sealed_dir, output = Path(sealed_dir).resolve(), Path(output).resolve()
    sealing = read(sealed_dir / "sealing.json")
    for name in ("model", "model_source", "training_raw", "protocol", "plan", "prediction"):
        if receipt(sealing[name]["path"]) != sealing[name]:
            raise ValueError("Changed sealed " + name)
    raw, plan = read(raw_path), read(sealing["plan"]["path"])
    expected_jobs = _jobs(plan)
    if (raw.get("schema") != "gsdeploy.native_phase_collection.v1" or raw.get("status") != "completed"
            or raw.get("kind") != "measured" or raw.get("backend") != "native"
            or raw.get("performance_scope") != "unprofiled_with_phase_markers"
            or raw.get("errors") or raw.get("runtime_errors") or raw.get("telemetry_errors")):
        raise ValueError("Complete error-free independent measured run required")
    if raw.get("observation_start_ns", -1) <= sealing["created_ns"]:
        raise ValueError("Predictions must be sealed before independent observation starts")
    if raw["execution_context_sha256"] != plan["execution_context_sha256"]:
        raise ValueError("Independent run execution identity differs from sealed prediction")
    if receipt(raw["plan"]["path"]) != raw["plan"]:
        raise ValueError("Changed actual run plan")
    actual_plan = read(raw["plan"]["path"])
    actual_selected = [j for j in actual_plan["jobs"] if j["id"] in expected_jobs]
    if _cohort({"jobs": actual_selected}) != _cohort(plan):
        raise ValueError("Actual selected job ordering/inputs differs from sealed independent plan")
    batches = {b["job_id"]: b for b in raw["batches"]}
    if set(batches) != expected_jobs or len(batches) != len(raw["batches"]):
        raise ValueError("Independent run missing, duplicate, or unexpected jobs")
    predicted = read(sealing["prediction"]["path"])
    pred_map = {r["request_id"]: r for r in predicted["requests"]}
    rows, seen, native_ids, errors_by_group = [], set(), set(), collections.defaultdict(list)
    for native in raw["request_records"]:
        job_id, alias = native["profile_job_id"], native["model"]
        rid = job_id + ":" + alias
        if (rid not in pred_map or rid in seen or native["request_id"] in native_ids
                or native.get("split") != "validation" or native["status"] != "completed"):
            raise ValueError("Unexpected, repeated, nonvalidation or incomplete independent request")
        seen.add(rid)
        native_ids.add(native["request_id"])
        response = native["worker_response"]
        start, end, scheduled = native["observed_worker_start_time_ns"], response["end_perf_ns"], native["scheduled_time_ns"]
        if (start != response.get("start_perf_ns") or not scheduled <= start < end
                or native["pool_id"] != pred_map[rid]["pool_id"] or native["task"] != pred_map[rid]["task"]):
            raise ValueError("Actual observed timing or input mismatch")
        expected_scheduled = batches[job_id]["start_ns"] + round(pred_map[rid]["offset_ms"] * 1e6)
        if scheduled != expected_scheduled or native["request_id"] not in batches[job_id]["request_ids"]:
            raise ValueError("Scheduled input offset differs from sealed plan")
        row = {"request_id": rid, "task": native["task"], "pool_id": native["pool_id"],
               "service_ms": (end - start) / 1e6, "wait_ms": (start - scheduled) / 1e6,
               "e2e_ms": (end - scheduled) / 1e6, "solo_ms": pred_map[rid]["solo_ms"]}
        rows.append(row)
        pred = pred_map[rid]
        for group in ("all", "context:" + pred["context"], "view:" + pred["validation_view"],
                      "input_known_to_model:" + str(pred["input_known_to_model"]),
                      "timing_pattern:" + pred["timing_pattern"],
                      "input_size:" + str(pred["input_size"]), "offset_ms:" + str(pred["offset_ms"]),
                      "alias:" + alias):
            errors_by_group[group].append((pred["service_ms"], row["service_ms"]))
    if seen != set(pred_map):
        raise ValueError("Independent observation does not cover all planned requests")
    output.mkdir(parents=True, exist_ok=False)
    observation = {"schema": OBSERVATION_SCHEMA, "evidence_kind": "measured", "run_id": str(Path(raw_path).resolve().parent),
                   "execution_context_sha256": raw["execution_context_sha256"],
                   "offered_cohort_sha256": predicted["offered_cohort_sha256"],
                   "available_ns": time.perf_counter_ns(), "requests": rows, "queue_samples": [], "candidates": [],
                   "raw": receipt(raw_path), "slowdown_reference": predicted["slowdown_reference"]}
    save(output / "observation.json", observation)
    manifest = {"schema": SCHEMA, "model": {"artifact": sealing["model"], "frozen_ns": sealing["frozen_ns"],
                "training_sources": [{"run_id": sealing["training_run_id"], "artifact": sealing["training_raw"]}]},
                "protocol": {"artifact": sealing["protocol"], "frozen_ns": sealing["frozen_ns"]},
                "cases": [{"id": "independent-native-run", "split": "validation", "run_id": observation["run_id"],
                           "prediction": sealing["prediction"], "observation": receipt(output / "observation.json")}]}
    save(output / "manifest.json", manifest)
    assessment = assess(manifest, artifact_root=output)
    assessment["probe_groups"] = {key: {"requested": len(values), "known": sum(p is not None for p, _ in values),
                                        "service_error": errors([(p, o) for p, o in values if p is not None])}
                                  for key, values in sorted(errors_by_group.items())}
    assessment["scope"] = "One prospective independent run of controlled probes; not full arrival queue or final spill validation"
    assessment["pending_categories"] = sealing["pending_categories"]
    assessment["slowdown_reference"] = predicted["slowdown_reference"]
    save(output / "report.json", assessment)
    return assessment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("seal")
    for name in ("model", "training-raw", "plan", "output"):
        a.add_argument("--" + name, required=True)
    a = sub.add_parser("report")
    for name in ("sealed-dir", "raw", "output"):
        a.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    result = (seal(args.model, args.training_raw, args.plan, args.output) if args.command == "seal"
              else report(args.sealed_dir, args.raw, args.output))
    print(json.dumps({"output": str(Path(args.output).resolve()), "status": result.get("status", "SEALED"), "GPU_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
