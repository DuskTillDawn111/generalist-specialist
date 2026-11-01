"""Prospective fixed-probe validation of regularized statistical phase models.

No fitting and no GPU launches. Exact-input coverage remains separate from
full-domain coverage. Regularized point predictions and handler interpolation
are declared explicitly; neither is physical interference identification.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting import FullHorizonModel
from . import probe_validation as base
from .empirical_resources import EmpiricalResourceProvider
from .profiling import RESIDENTS, digest
from .statistical_profiles import StatisticalProvider, SCHEMA as MODEL_SCHEMA
from .validation import PREDICTION_SCHEMA, errors, percentile


class RecordedProvider:
    """Record prediction metadata without using any validation observations."""
    def __init__(self, provider):
        self.provider, self.calls = provider, []

    def plan(self, *args):
        return self.provider.plan(*args)

    def service(self, row, *args):
        result = self.provider.service(row, *args)
        self.calls.append({"request_id": row["request_id"], "phase": row["segment_id"],
                           "known": result.get("known") is True,
                           "interpolated": result.get("interpolated"),
                           "physically_identified": result.get("physically_identified", False),
                           "uncertainty_calibrated": result.get("uncertainty_calibrated", False),
                           "reason": result.get("reason")})
        return result

    def power(self, *args):
        return self.provider.power(*args)

    def memory(self, *args):
        return self.provider.memory(*args)


def _checked_source(ref):
    if not isinstance(ref, dict) or base.receipt(ref["path"]) != ref:
        raise ValueError("Changed statistical training/model source receipt")


def seal(model_path, training_raw_path, plan_path, output, *, resources_path=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan, training, document = base.read(plan_path), base.read(training_raw_path), base.read(model_path)
    if document.get("schema") != MODEL_SCHEMA:
        raise ValueError("Explicit regularized statistical semantic model required")
    jobs = base._jobs(plan)
    if (training.get("schema") != "gsdeploy.native_phase_collection.v1" or training.get("status") != "completed"
            or training.get("execution_context_sha256") != plan["execution_context_sha256"]):
        raise ValueError("Completed matching native training collection required")
    if any(row.get("split") != "training" for row in training["request_records"]):
        raise ValueError("Training-only raw required; no old validation refitting")
    trained_jobs = {r["profile_job_id"] for r in training["request_records"]}
    if not trained_jobs or trained_jobs & jobs:
        raise ValueError("Independent validation jobs must be disjoint from training jobs")
    training_ref = base.receipt(training_raw_path)
    for ref in document.get("source_receipts", []):
        _checked_source(ref)
    if training_ref not in document.get("source_receipts", []):
        raise ValueError("Statistical model must bind this exact training raw source")
    if (Path(plan_path).resolve().parent / "raw.json").exists():
        raise ValueError("Seal before independent collection, never after raw observations exist")
    provider = StatisticalProvider(document, plan["execution_context_sha256"])
    profiles = provider.profiles
    if not profiles:
        raise ValueError("Nonempty complete statistical input profiles required")
    resource, resource_ref = None, None
    if resources_path:
        resource_doc = base.read(resources_path)
        for ref in resource_doc["source_receipts"]:
            _checked_source(ref)
        if training_ref not in resource_doc["source_receipts"]:
            raise ValueError("Resources must bind the same training-only raw")
        resource = EmpiricalResourceProvider(resource_doc, plan["execution_context_sha256"])
        resource_ref = base.receipt(resources_path)
    sources = [base.receipt(__file__), base.receipt(base.__file__),
               base.receipt(Path(__file__).with_name("statistical_profiles.py")),
               base.receipt(Path(__file__).with_name("empirical_resources.py"))]
    output.mkdir(parents=True, exist_ok=False)
    model_adapter = {"schema": "gsdeploy.frozen_statistical_probe_model.v1", "source": base.receipt(model_path),
                     "resources": resource_ref, "document": document,
                     "regularization_dependent": True, "physically_identified": False}
    base.save(output / "model.json", model_adapter)
    tasks = sorted({"ocr" if a == "ocr#0" else "text" for job in plan["jobs"] for a in job["inputs"]})
    protocol = {"weights": dict.fromkeys("QTLPME", 1 / 6), "utility_tie_tolerance": 0.,
                "required_categories": ["service", "p95"],
                "planned_cases": [{"id": "independent-native-run", "categories": ["service", "p95"], "expected_tasks": tasks}],
                "acceptance": {"service_mare": {"operator": "<=", "value": .2},
                               "p95_mare": {"operator": "<=", "value": .2}},
                "known_input_gate": {"coverage": 1., "service_mare_max": .2, "p95_mare_max": .2,
                                     "unit": "One entire independent run; exact model input lookup defines subdomain",
                                     "unseen": "UNKNOWN, reported separately and retained in full-domain denominator"},
                "no_validation_tuning": True, "regularization_dependent": True, "physically_identified": False,
                "scope": "Declared-offset statistical service validation only; full queue/delta/ranking remain NOT_RUN"}
    base.save(output / "protocol.json", protocol)
    frozen = time.perf_counter_ns()
    aliases = {a: {"family": f, "supported_tasks": ["ocr" if f == "ocr" else "text"]} for a, f in RESIDENTS.items()}
    sizes = {(r["task"], r["pool_id"]): r["input_size"] for r in plan["inputs"]}
    predicted, job_reports = [], []
    for job in plan["jobs"]:
        queues, future, routes, rows = {a: [] for a in RESIDENTS}, [], {}, []
        timing = ",".join(a + "@" + str(t) for a, t in sorted(job["start_offsets_ms"].items(), key=lambda x: (x[1], x[0])))
        for alias, offset in job["start_offsets_ms"].items():
            task, rid = ("ocr" if alias == "ocr#0" else "text"), job["id"] + ":" + alias
            request = {"request_id": rid, "task": task, "pool_id": job["inputs"][alias], "arrival_ns": round(offset * 1e6)}
            (queues[alias] if request["arrival_ns"] == 0 else future).append(request)
            routes[rid] = alias
            profile = profiles.get((RESIDENTS[alias], task, request["pool_id"]))
            rows.append({"request_id": rid, "task": task, "pool_id": request["pool_id"], "alias": alias,
                         "job_id": job["id"], "context": job["context"], "validation_view": job.get("validation_view", "unspecified"),
                         "input_size": sizes[(task, request["pool_id"])], "offset_ms": offset, "timing_pattern": timing,
                         "input_known_to_model": profile is not None,
                         "solo_ms": sum(s["duration_ms"] for s in profile["segments"]) if profile else None,
                         "service_ms": None, "wait_ms": None, "e2e_ms": None,
                         "regularization_dependent": True, "physically_identified": False,
                         "interpolated": None, "uncertainty_calibrated": False})
        observed = RecordedProvider(provider)
        result, reason = None, None
        try:
            deadline = max(round(v * 1e6) for v in job["start_offsets_ms"].values()) + 1
            result = FullHorizonModel(aliases, observed, resource or observed, memory_provider=resource or observed).forecast(
                queues, 0, deadline, future_arrivals=future,
                route_future=lambda row, pending, now: routes[row["request_id"]],
                require_energy=False, require_memory=False, drain_deadline_ns=deadline + 300_000_000_000)
            if result["incomplete_ids"]:
                reason = "DRAIN_CENSORED"
        except UnknownEvidence as exc:
            reason = str(exc)
        for row in rows:
            finish = result["finishes"].get(row["request_id"]) if result else None
            calls = [c for c in observed.calls if c["request_id"] == row["request_id"]]
            if any(c["physically_identified"] or c["uncertainty_calibrated"] for c in calls):
                raise ValueError("Statistical adapter cannot declare physical identification/calibrated uncertainty")
            if finish:
                arrival = round(row["offset_ms"] * 1e6)
                row.update(service_ms=(finish["finish_ns"] - finish["start_ns"]) / 1e6,
                           wait_ms=(finish["start_ns"] - arrival) / 1e6,
                           e2e_ms=(finish["finish_ns"] - arrival) / 1e6,
                           interpolated=any(c["interpolated"] is True for c in calls))
            row.update(unknown_reason=reason if finish is None else None,
                       statistical_service_calls=len(calls), interpolated_call_count=sum(c["interpolated"] is True for c in calls))
            predicted.append(row)
        job_reports.append({"job_id": job["id"], "known": sum(r["service_ms"] is not None for r in rows),
                            "requested": len(rows), "unknown_reason": reason})
    p95 = {task: percentile([r["e2e_ms"] for r in predicted if r["task"] == task])
           if all(r["e2e_ms"] is not None for r in predicted if r["task"] == task) else None for task in tasks}
    created = time.perf_counter_ns()
    prediction = {"schema": PREDICTION_SCHEMA, "model_sha256": base.receipt(output / "model.json")["sha256"],
                  "protocol_sha256": base.receipt(output / "protocol.json")["sha256"],
                  "execution_context_sha256": plan["execution_context_sha256"], "offered_cohort_sha256": base._cohort(plan),
                  "created_ns": created, "origin_ns": frozen, "information_cutoff_ns": frozen,
                  "mode": "declared_plan", "schedule_declared_before_origin": True, "actual_future_trace_used": False,
                  "requests": predicted, "p95_by_task": p95, "queue_samples": [], "candidates": [],
                  "regularization_dependent": True, "physically_identified": False, "uncertainty_calibrated": False,
                  "slowdown_reference": "Training solo mean, not independent validation solo truth",
                  "scope": "Prospective regularized mixed semantic wall prediction; not kernel physical identification"}
    base.save(output / "prediction.json", prediction)
    sealing = {"schema": "gsdeploy.native_statistical_probe_sealing.v1", "created_ns": created, "frozen_ns": frozen,
               "model": base.receipt(output / "model.json"), "model_source": base.receipt(model_path),
               "training_raw": training_ref, "training_run_id": str(Path(training_raw_path).resolve().parent),
               "protocol": base.receipt(output / "protocol.json"), "plan": base.receipt(plan_path),
               "prediction": base.receipt(output / "prediction.json"), "resources": resource_ref,
               "source_receipts": sources, "job_reports": job_reports, "GPU_started": False,
               "pending_categories": {"full_queue": "NOT_RUN", "delta_U": "NOT_RUN", "ranking": "NOT_RUN"}}
    base.save(output / "sealing.json", sealing)
    return sealing


def report(sealed_dir, raw_path, output):
    sealed_dir, output = Path(sealed_dir).resolve(), Path(output).resolve()
    sealing = base.read(sealed_dir / "sealing.json")
    if sealing.get("schema") != "gsdeploy.native_statistical_probe_sealing.v1":
        raise ValueError("Explicit statistical probe sealing schema required")
    for ref in sealing["source_receipts"]:
        _checked_source(ref)
    if sealing.get("resources"):
        _checked_source(sealing["resources"])
    original_save = base.save
    prediction = base.read(sealing["prediction"]["path"])

    def save_report(path, assessment):
        if Path(path).name == "report.json":
            observed = {r["request_id"]: r for r in base.read(output / "observation.json")["requests"]}
            known_rows = [r for r in prediction["requests"] if r["input_known_to_model"]]
            unseen = [r for r in prediction["requests"] if not r["input_known_to_model"]]
            valid = [r for r in known_rows if r["service_ms"] is not None]
            service_error = errors([(r["service_ms"], observed[r["request_id"]]["service_ms"]) for r in valid])
            p95_pairs, task_coverage = [], {}
            for task in sorted({r["task"] for r in known_rows}):
                rows = [r for r in known_rows if r["task"] == task]
                complete = all(r["e2e_ms"] is not None for r in rows)
                actual = percentile([observed[r["request_id"]]["e2e_ms"] for r in rows])
                estimated = percentile([r["e2e_ms"] for r in rows]) if complete else None
                task_coverage[task] = {"requested": len(rows), "known": sum(r["e2e_ms"] is not None for r in rows),
                                       "observed_p95_ms": actual, "predicted_p95_ms": estimated}
                if estimated is not None:
                    p95_pairs.append((estimated, actual))
            tail_error = errors(p95_pairs)
            complete = bool(known_rows) and len(valid) == len(known_rows) and len(p95_pairs) == len(task_coverage)
            passed = (complete and service_error["mean_absolute_relative_error"] is not None
                      and service_error["mean_absolute_relative_error"] <= .2
                      and tail_error["mean_absolute_relative_error"] is not None
                      and tail_error["mean_absolute_relative_error"] <= .2)
            assessment["known_input_gate"] = {"status": "INCOMPLETE" if not complete else "PASS" if passed else "FAIL",
                "requested": len(known_rows), "known": len(valid), "coverage": len(valid) / len(known_rows) if known_rows else 0.,
                "service_error": service_error, "p95_error": tail_error, "per_task": task_coverage,
                "required_coverage": 1., "required_mare_max": .2, "independent_run_cases": 1,
                "scope": "Exact frozen input subdomain only; never substitutes for full-domain coverage"}
            assessment["unseen_input_diagnostics"] = {"requested": len(unseen),
                "known": sum(r["service_ms"] is not None for r in unseen),
                "unknown_reasons": dict(collections.Counter(r["unknown_reason"] for r in unseen))}
            assessment["statistical_prediction_semantics"] = {"regularization_dependent": True,
                "physically_identified": False, "uncertainty_calibrated": False,
                "interpolated_known_requests": sum(r["interpolated"] is True for r in valid),
                "interpolated_service_calls": sum(r["interpolated_call_count"] for r in prediction["requests"]),
                "source_receipts": sealing["source_receipts"], "resources": sealing.get("resources"),
                "no_validation_refit": True, "physical_optimality_certified": False}
            assessment["scope"] = "One prospective statistical mixed-phase heldout run; full-domain coverage retained; final four groups remain gated"
        original_save(path, assessment)
    with patch.object(base, "save", save_report):
        return base.report(sealed_dir, raw_path, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("seal")
    for name in ("model", "training-raw", "plan", "output"):
        a.add_argument("--" + name, required=True)
    a.add_argument("--resources")
    a = sub.add_parser("report")
    for name in ("sealed-dir", "raw", "output"):
        a.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    result = (seal(args.model, args.training_raw, args.plan, args.output, resources_path=args.resources)
              if args.command == "seal" else report(args.sealed_dir, args.raw, args.output))
    print(json.dumps({"status": result.get("status", "SEALED"), "known_input_gate": result.get("known_input_gate"),
                      "output": str(Path(args.output).resolve()), "GPU_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
