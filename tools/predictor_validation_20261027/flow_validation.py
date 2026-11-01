"""Prospective full-arrival native-flow forecast and independent assessment.

Fixed trace/routing plans are declared experiments. No actual future runtime
activity or outputs enter the sealer. Reports retain all offered requests and
compare worker-completion forecasts against actual parent-receipt E2E. This
tool never fits, launches a worker, or upgrades missing evidence to zero.
"""
from __future__ import annotations

import argparse
import collections
import copy
import itertools
import inspect
import json
import math
from pathlib import Path
import time

from gsdeploy.energy import integrate_power
from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting import FullHorizonModel, score_forecast
from . import probe_validation as base
from .empirical_resources import EmpiricalResourceProvider
from .profiling import digest
from .statistical_profiles import StatisticalProvider
from .validation import errors, percentile
from .flow_metrics import score_flow


PLAN_SCHEMA = "gsdeploy.native_flow_plan.v1"
RAW_SCHEMA = "gsdeploy.native_flow_collection.v1"


def _sources(document):
    return document.get("source_receipts", document.get("provenance", {}).get("sources", []))


def _verify(ref):
    if not isinstance(ref, dict) or base.receipt(ref["path"]) != ref:
        raise ValueError("Changed or missing source receipt")


def _provider(document, context):
    if document.get("schema") == "gsdeploy.background_task_statistical_progress.v1":
        from .background_statistical import BackgroundStatisticalProvider
        return BackgroundStatisticalProvider(document, context)
    return StatisticalProvider(document, context)


def _resources(document, context):
    if "task_conditional_definition" in document:
        from .background_resources import TaskConditionalResourceProvider
        return TaskConditionalResourceProvider(document, context)
    if "task_specific_resource_omission_audit" in document:
        from .background_resources import TaskAwareResourceProvider
        return TaskAwareResourceProvider(document, context)
    return EmpiricalResourceProvider(document, context)


def validate_plan(plan):
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("Explicit native full-flow plan schema required")
    if not plan.get("aliases") or not plan.get("trace") or not plan.get("candidates"):
        raise ValueError("Nonempty resident aliases, complete trace and finite candidates required")
    for field in ("window_seconds", "drain_seconds", "memory_budget_mib", "energy_reference_watts", "startup_amortization_seconds"):
        if type(plan[field]) not in (int, float) or not math.isfinite(plan[field]) or plan[field] <= 0:
            raise ValueError("Positive objective/window field required: " + field)
    if (not isinstance(plan.get("workload_rps"), dict) or not plan["workload_rps"]
            or set(plan.get("latency_reference_ms", {})) != set(plan["workload_rps"])
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                   for v in [*plan["workload_rps"].values(), *plan["latency_reference_ms"].values()])):
        raise ValueError("Positive offered rates and full latency references required")
    end = round(plan["window_seconds"] * 1e9)
    if any(type(v) is not int or not 0 <= v <= end for v in plan.get("queue_sample_offsets_ns", [])):
        raise ValueError("Queue sample offsets must belong to measurement window")
    if len(set(plan.get("queue_sample_offsets_ns", []))) != len(plan.get("queue_sample_offsets_ns", [])):
        raise ValueError("Unique queue sample offsets required")
    ids = set()
    for row in plan["trace"]:
        if (set(row) != {"request_id", "offset_ns", "task", "pool_id"}
                or not isinstance(row["request_id"], str) or not row["request_id"] or row["request_id"] in ids
                or type(row["offset_ns"]) is not int or not 0 <= row["offset_ns"] < end):
            raise ValueError("Unique public-input offered requests inside full window required")
        if row["task"] not in plan["workload_rps"] or not isinstance(row["pool_id"], str) or not row["pool_id"]:
            raise ValueError("Public task/input identity required")
        ids.add(row["request_id"])
    candidates = set()
    for candidate in plan["candidates"]:
        cid = candidate["id"]
        if not isinstance(cid, str) or not cid or cid in candidates or set(candidate["routes"]) != ids:
            raise ValueError("Unique candidates and full route coverage required")
        if not isinstance(candidate.get("execution_context_sha256"), str) or not candidate["execution_context_sha256"]:
            raise ValueError("Candidate base execution identity required")
        for field in ("predicted_startup_joules", "predicted_prephase_peak_mib"):
            v = candidate.get(field)
            if v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v < 0):
                raise ValueError("Finite nonnegative candidate resource estimate required")
        candidates.add(cid)
        for row in plan["trace"]:
            alias = candidate["routes"][row["request_id"]]
            if alias not in plan["aliases"] or row["task"] not in plan["aliases"][alias]["supported_tasks"]:
                raise ValueError("Unsupported candidate route")
    if plan["baseline_candidate_id"] not in candidates:
        raise ValueError("Declared baseline must belong to finite candidate domain")
    if "plan_sha256" in plan and digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise ValueError("Flow plan self hash mismatch")
    startup_policy = plan.get("startup_comparison_policy")
    if startup_policy is not None:
        if (startup_policy.get("kind") != "common_deployment_backend_omitted"
                or not startup_policy.get("group_id") or set(startup_policy.get("candidate_ids", [])) != candidates
                or len({c["execution_context_sha256"] for c in plan["candidates"]}) != 1):
            raise ValueError("Explicit common deployment/backend startup comparison declaration required")
    if (set(plan["weights"]) != set("QTLPME") or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in plan["weights"].values())
            or abs(sum(plan["weights"].values()) - 1) > 1e-12):
        raise ValueError("Six-term objective required")
    if (type(plan["memory_safe_factor"]) not in (int, float) or not math.isfinite(plan["memory_safe_factor"])
            or plan["memory_safe_factor"] < 1):
        raise ValueError("Original memory safety factor must be >=1")
    return plan


def cohort_sha(plan):
    return digest(plan["trace"])


def _score(plan, forecast, quality, *, startup_joules=None, historical_peak=None):
    horizon = plan["window_seconds"]
    # score_forecast accepts common past as joules normalized over H. Convert
    # the original startup amortization exactly: startupJ * H / amortization.
    amortized = None if startup_joules is None else startup_joules * horizon / plan["startup_amortization_seconds"]
    return score_forecast(forecast, [], plan["aliases"], plan["workload_rps"], plan["latency_reference_ms"], quality,
        phase_start_ns=0, now_ns=0, deadline_ns=round(horizon * 1e9),
        memory_budget_mib=plan["memory_budget_mib"], energy_reference_watts=plan["energy_reference_watts"],
        memory_safe_factor=plan["memory_safe_factor"], weights=plan["weights"],
        observed_peak_memory_mib=historical_peak, observed_board_energy_joules=amortized)


def _queue_samples(plan, forecast):
    result = []
    for offset in plan.get("queue_sample_offsets_ns", []):
        count = sum(row["arrival_ns"] <= offset and forecast["finishes"].get(rid, {}).get("finish_ns", float("inf")) > offset
                    for rid, row in forecast["cohort"].items())
        result.append({"offset_ns": offset, "depth": count})
    return result


def seal(plan_path, model_path, resources_path, quality_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = validate_plan(base.read(plan_path))
    model, resource_document, quality_document = base.read(model_path), base.read(resources_path), base.read(quality_path)
    quality = quality_document.get("loss_by_family_task", quality_document)
    quality_sources = quality_document.get("source_receipts", [])
    refs = {"plan": base.receipt(plan_path), "model": base.receipt(model_path),
            "resources": base.receipt(resources_path), "quality": base.receipt(quality_path)}
    training_sources = _sources(model)
    if not training_sources:
        raise ValueError("Model training source receipts required")
    execution_sources = [*plan.get("execution_sources", []),
                         *[plan[k] for k in ("runner_source", "context", "profile_plan", "catalog") if k in plan]]
    if plan.get("prephase_memory_evidence"):
        execution_sources.append(plan["prephase_memory_evidence"]["source"])
    for ref in [*training_sources, *_sources(resource_document), *execution_sources, *quality_sources]:
        _verify(ref)
    for ref in training_sources:
        if Path(ref["path"]).suffix == ".json":
            training = base.read(ref["path"])
            if training.get("kind") == "measured" and any(r.get("split") != "training" for r in training.get("request_records", [])):
                raise ValueError("Training-only model source observations required")
    predictor_paths = set(Path(inspect.getfile(FullHorizonModel)).parent.glob("*.py"))
    predictor_paths.update(Path(__file__).parent / name for name in (
        "statistical_profiles.py", "background_statistical.py", "empirical_resources.py", "background_resources.py"))
    predictor_sources = [base.receipt(p) for p in sorted(predictor_paths)]
    end = round(plan["window_seconds"] * 1e9)
    created_before = time.perf_counter_ns()
    predictions = []
    for candidate in plan["candidates"]:
        forecast, reason, metric = None, None, None
        try:
            provider = _provider(model, candidate["execution_context_sha256"])
            resource = _resources(resource_document, candidate["execution_context_sha256"])
            queues, future = {a: [] for a in plan["aliases"]}, []
            for row in plan["trace"]:
                r = {"request_id": row["request_id"], "task": row["task"], "pool_id": row["pool_id"], "arrival_ns": row["offset_ns"]}
                if r["arrival_ns"] == 0:
                    queues[candidate["routes"][r["request_id"]]].append(r)
                else:
                    future.append(r)
            forecast = FullHorizonModel(plan["aliases"], provider, resource, memory_provider=resource,
                resources=candidate.get("resource_configuration"), max_requests=plan.get("max_forecast_requests", 20000)).forecast(
                queues, 0, end, future_arrivals=future,
                route_future=lambda row, pending, now: candidate["routes"][row["request_id"]],
                require_energy=False, require_memory=False,
                drain_deadline_ns=end + round(plan["drain_seconds"] * 1e9))
            metric = _score(plan, forecast, quality,
                startup_joules=candidate.get("predicted_startup_joules"), historical_peak=candidate.get("predicted_prephase_peak_mib"))
        except (UnknownEvidence, ValueError) as exc:
            reason = type(exc).__name__ + ":" + str(exc)
        requests = []
        for row in plan["trace"]:
            finish = forecast["finishes"].get(row["request_id"]) if forecast else None
            requests.append({"request_id": row["request_id"], "task": row["task"], "pool_id": row["pool_id"],
                "alias": candidate["routes"][row["request_id"]],
                "service_ms": (finish["finish_ns"] - finish["start_ns"]) / 1e6 if finish else None,
                "wait_ms": (finish["start_ns"] - row["offset_ns"]) / 1e6 if finish else None,
                "e2e_ms": (finish["finish_ns"] - row["offset_ns"]) / 1e6 if finish else None,
                "finish_ns": finish["finish_ns"] if finish else None,
                "unknown_reason": reason if not finish else None})
        predictions.append({"candidate_id": candidate["id"], "requests": requests, "metrics": metric,
            "comparison_U": -metric["cost"] if metric and metric["cost"] is not None and
                (metric["absolute_utility_complete"] or plan.get("startup_comparison_policy")) else None,
            "comparison_scope": "Declared common startup omitted from predicted comparisons; actual measured startup differences retained"
                if plan.get("startup_comparison_policy") else "Absolute utility only",
            "queue_samples": _queue_samples(plan, forecast) if forecast else [], "unknown_reason": reason,
            "point_prediction_available": forecast is not None, "regularization_dependent": True,
            "physically_identified": False, "parent_receipt_delay_explicitly_modeled": False,
            "latency_scope": "Worker completion proxy forecast compared against actual parent receipt; missing transport tail remains forecast error"})
    output.mkdir(parents=True, exist_ok=False)
    value = {"schema": "gsdeploy.sealed_native_flow_predictions.v1", "created_ns": time.perf_counter_ns(),
        "information_cutoff_ns": created_before, "sources": refs, "training_sources": training_sources,
        "tool_source": base.receipt(__file__), "metric_source": base.receipt(score_flow.__code__.co_filename),
        "resource_sources": _sources(resource_document), "execution_sources": execution_sources,
        "quality_sources": quality_sources,
        "predictor_sources": predictor_sources,
        "cohort_sha256": cohort_sha(plan), "plan_sha256": refs["plan"]["sha256"],
        "candidates": predictions, "scope": "Declared full arrival trace and routes, no observed future activity",
        "GPU_started": False, "actual_future_trace_used": False,
        "gate_protocol": plan.get("validation_thresholds", {}), "phase": plan.get("phase", "predictor_validation")}
    base.save(output / "predictions.json", value)
    return value


def observed(plan, candidate, raw):
    """Recompute observed parent-receipt metrics and explicitly measured Q."""
    if raw.get("schema") != RAW_SCHEMA or raw.get("kind") != "measured":
        raise ValueError("Measured native full-flow schema required")
    if raw.get("candidate_id") != candidate["id"] or raw.get("execution_context_sha256") != candidate["execution_context_sha256"]:
        raise ValueError("Raw candidate/execution identity mismatch")
    if raw.get("cohort_sha256", raw.get("trace_sha256")) != cohort_sha(plan):
        raise ValueError("Unpaired offered arrival/input cohort")
    if "resource_configuration" in raw and raw["resource_configuration"] != candidate.get("resource_configuration"):
        raise ValueError("Raw resource configuration differs from frozen candidate")
    start = raw["phase_start_ns"]
    deadline = start + round(plan["window_seconds"] * 1e9)
    if raw["deadline_ns"] != deadline:
        raise ValueError("Actual measurement window differs from frozen plan")
    actual = {r["request_id"]: r for r in raw["request_records"]}
    if len(actual) != len(raw["request_records"]) or set(actual) != {r["request_id"] for r in plan["trace"]}:
        raise ValueError("Observed offered cohort has missing/foreign/duplicate requests")
    cohort, finishes, timing_rows = {}, {}, []
    for request in plan["trace"]:
        rid, offset = request["request_id"], request["offset_ns"]
        row = actual[rid]
        alias = candidate["routes"][rid]
        if (row["task"], row["pool_id"], row["model"], row["scheduled_time_ns"]) != (request["task"], request["pool_id"], alias, start + offset):
            raise ValueError("Actual route/input/scheduled arrival differs from frozen plan")
        cohort[rid] = {"request_id": rid, "task": request["task"], "pool_id": request["pool_id"], "alias": alias, "arrival_ns": offset}
        timing = {"request_id": rid, "task": request["task"], "pool_id": request["pool_id"], "status": row["status"],
                  "service_ms": None, "wait_ms": None, "e2e_ms": None, "response_tail_ms": None}
        if row["status"] == "completed":
            response = row["worker_response"]
            worker_start, worker_end, receipt_ns = row["observed_worker_start_time_ns"], response["end_perf_ns"], row["completion_time_ns"]
            if (worker_start != response["start_perf_ns"] or not start + offset <= worker_start < worker_end <= receipt_ns
                    or receipt_ns > deadline + round(plan["drain_seconds"] * 1e9)):
                raise ValueError("Actual parent receipt/worker timing order or bounded drain invalid")
            timing.update(service_ms=(worker_end - worker_start) / 1e6, wait_ms=(worker_start - start - offset) / 1e6,
                          e2e_ms=(receipt_ns - start - offset) / 1e6, response_tail_ms=(receipt_ns - worker_end) / 1e6)
            finishes[rid] = {**cohort[rid], "finish_ns": receipt_ns - start}
        timing_rows.append(timing)
    quality = raw.get("quality_evidence", {})
    quality_refs = quality.get("source_receipts", [])
    for ref in quality_refs:
        _verify(ref)
    # The helper uses original task-level numerator/denominator aggregation,
    # parent receipt latency, startup amortization and owned sampled memory.
    scored_raw = copy.deepcopy(raw)
    if quality.get("kind") != "measured_per_request" or not quality_refs:
        scored_raw["quality_evidence"] = {}
    diagnostic = score_flow(scored_raw, {**plan, "resident_aliases": list(plan["aliases"])})
    terms = diagnostic["terms"]
    complete = diagnostic["utility_complete"] and len(finishes) == len(cohort)
    reasons = list(diagnostic["unknown_or_ineligible_reasons"])
    if not quality_refs:
        reasons.append("UNKNOWN_ACTUAL_QUALITY_SOURCE_RECEIPTS")
    metric = {"terms": terms, "cost": -diagnostic["diagnostic_U"] if complete else None,
        "full_U": diagnostic["diagnostic_U"] if complete else None,
        "six_term_comparison_complete": complete, "absolute_utility_complete": complete,
        "cohort_size": len(cohort), "cohort_ids": sorted(cohort),
        "per_task": {t: {"request_count": v["offered"], "mean_ms": v["mean_e2e_ms"], "p95_ms": v["p95_e2e_ms"]}
                     for t, v in diagnostic["per_task"].items()},
        "incomplete_ids": sorted(set(cohort) - set(finishes)), "unknown_reasons": sorted(set(reasons)),
        "memory_budget_satisfied": diagnostic["memory"]["budget_satisfied"],
        "quality_scope": "Actual heldout per-task summed numerators/denominators; never mean request CER/WER",
        "E_scope": diagnostic["energy"]["run_scope"] + "; " + diagnostic["energy"]["startup_scope"],
        "original_metric_diagnostics": diagnostic}
    metric["eligible_for_gain_claim"] = (complete and raw.get("status") == "completed" and not raw.get("errors")
        and not raw.get("runtime_errors") and not raw.get("telemetry_errors")
        and not raw.get("qualification_issues") and raw.get("collection_complete", True) is True
        and raw.get("rho_admission_passed") is True
        and raw.get("stability", {}).get("stability_state_v2") in {"steady", "recoverable_congested"}
        and metric["memory_budget_satisfied"] is True)
    forecast = {"cohort": cohort, "finishes": finishes}
    return {"candidate_id": candidate["id"], "requests": timing_rows, "metrics": metric,
            "queue_samples": _queue_samples(plan, forecast),
            "telemetry_valid": diagnostic["terms"]["M"] is not None and diagnostic["energy"]["coverage_valid"],
            "raw_status": raw.get("status"), "missing_quality": terms["Q"] is None}


def report(sealed_dir, raw_paths, output):
    output, sealed_dir = Path(output).resolve(), Path(sealed_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    sealed = base.read(sealed_dir / "predictions.json")
    for ref in [*sealed["sources"].values(), *sealed["training_sources"], *sealed["resource_sources"],
                *sealed["execution_sources"], *sealed["quality_sources"], *sealed["predictor_sources"], sealed["tool_source"], sealed["metric_source"]]:
        _verify(ref)
    plan = validate_plan(base.read(sealed["sources"]["plan"]["path"]))
    raws = [base.read(path) for path in raw_paths]
    for raw in raws:
        if "plan" in raw and raw["plan"] != sealed["sources"]["plan"]:
            raise ValueError("Raw flow plan differs from sealed plan")
    by_id = {raw["candidate_id"]: raw for raw in raws}
    if len(by_id) != len(raws) or set(by_id) - {c["id"] for c in plan["candidates"]}:
        raise ValueError("Duplicate or foreign actual candidates")
    if any(raw["phase_start_ns"] <= sealed["created_ns"] for raw in raws):
        raise ValueError("Full-flow predictions must be sealed before actual measurement")
    predictions = {p["candidate_id"]: p for p in sealed["candidates"]}
    diagnostics, utilities, paired, ranks, eligible = [], {}, [], [], {}
    for candidate in plan["candidates"]:
        cid, prediction = candidate["id"], predictions[candidate["id"]]
        if cid not in by_id:
            diagnostics.append({"candidate_id": cid, "status": "NOT_RUN", "requested": len(plan["trace"]), "known": 0})
            continue
        result = observed(plan, candidate, by_id[cid])
        actual = {r["request_id"]: r for r in result["requests"]}
        timing_errors = {k: errors([(p[k + "_ms"], actual[p["request_id"]][k + "_ms"]) for p in prediction["requests"]
                                  if p[k + "_ms"] is not None and actual[p["request_id"]][k + "_ms"] is not None])
                         for k in ("service", "wait", "e2e")}
        queue_actual = {q["offset_ns"]: q["depth"] for q in result["queue_samples"]}
        queue_error = errors([(q["depth"], queue_actual[q["offset_ns"]]) for q in prediction["queue_samples"]])
        pmetric, ometric = prediction["metrics"], result["metrics"]
        p95_error = errors([(pmetric["per_task"][t]["p95_ms"], ometric["per_task"][t]["p95_ms"])
                           for t in plan["workload_rps"] if pmetric and t in pmetric["per_task"] and t in ometric["per_task"]
                           and pmetric["per_task"][t]["p95_ms"] is not None and ometric["per_task"][t]["p95_ms"] is not None])
        utilities[cid] = (prediction["comparison_U"], ometric["full_U"])
        eligible[cid] = ometric["eligible_for_gain_claim"]
        diagnostics.append({"candidate_id": cid, "status": "ASSESSED", "requested": len(plan["trace"]),
            "known": timing_errors["e2e"]["count"], "timing_errors": timing_errors,
            "queue_error": queue_error, "p95_error": p95_error, "prediction_metrics": pmetric,
            "predicted_comparison_U": prediction["comparison_U"], "comparison_scope": prediction["comparison_scope"],
            "observed_metrics": ometric, "telemetry_valid": result["telemetry_valid"],
            "parent_receipt_response_tail_ms": {"mean": sum(r["response_tail_ms"] for r in result["requests"] if r["response_tail_ms"] is not None) /
                max(1, sum(r["response_tail_ms"] is not None for r in result["requests"]))},
            "actual_incomplete_requests": sum(r["status"] != "completed" for r in result["requests"])})
    sign = lambda v: 0 if abs(v) <= plan.get("utility_tie_tolerance", 0.) else (1 if v > 0 else -1)
    baseline = utilities.get(plan["baseline_candidate_id"])
    if baseline and all(v is not None for v in baseline):
        for cid, values in utilities.items():
            if cid != plan["baseline_candidate_id"] and all(v is not None for v in values):
                p, o = values[0] - baseline[0], values[1] - baseline[1]
                paired.append({"candidate_id": cid, "predicted_delta_U": p, "observed_delta_U": o, "sign_correct": sign(p) == sign(o),
                    "actual_startup_delta_joules": next(d["observed_metrics"]["original_metric_diagnostics"]["energy"]["startup_joules"] for d in diagnostics if d["candidate_id"] == cid) -
                        next(d["observed_metrics"]["original_metric_diagnostics"]["energy"]["startup_joules"] for d in diagnostics if d["candidate_id"] == plan["baseline_candidate_id"]),
                    "eligible_for_gain_claim": eligible[cid] and eligible[plan["baseline_candidate_id"]]})
    for a, b in itertools.combinations(sorted(utilities), 2):
        if all(v is not None for v in (*utilities[a], *utilities[b])):
            ranks.append({"a": a, "b": b, "correct": sign(utilities[a][0] - utilities[b][0]) == sign(utilities[a][1] - utilities[b][1])})
    complete = (len(by_id) == len(plan["candidates"]) and all(d.get("known") == d["requested"] and
                d.get("observed_metrics", {}).get("six_term_comparison_complete") and d.get("prediction_metrics") and
                d["prediction_metrics"]["six_term_comparison_complete"] and d.get("predicted_comparison_U") is not None
                for d in diagnostics))
    value = {"schema": "gsdeploy.native_flow_validation_report.v1", "status": "ASSESSED" if complete else "INCOMPLETE",
        "sealed": base.receipt(sealed_dir / "predictions.json"), "raw_sources": [base.receipt(p) for p in raw_paths],
        "report_source": base.receipt(__file__), "candidates": diagnostics,
        "candidate_coverage": {"requested": len(plan["candidates"]), "observed": len(by_id)},
        "paired_deltas": paired, "paired_delta_coverage": {"requested": len(plan["candidates"]) - 1, "known": len(paired)},
        "paired_delta_error": errors([(r["predicted_delta_U"], r["observed_delta_U"]) for r in paired]),
        "delta_sign_accuracy": sum(r["sign_correct"] for r in paired) / len(paired) if paired else None,
        "candidate_pair_ranks": ranks, "rank_accuracy": sum(r["correct"] for r in ranks) / len(ranks) if ranks else None,
        "rank_coverage": {"requested": len(plan["candidates"]) * (len(plan["candidates"]) - 1) // 2, "known": len(ranks)},
        "uncertainty_calibrated": False, "physical_optimality_certified": False, "final_four_groups_enabled": False,
        "startup_comparison_policy": plan.get("startup_comparison_policy"),
        "predicted_absolute_U_coverage": {"requested": len(plan["candidates"]),
            "known": sum(p["metrics"] is not None and p["metrics"]["full_U"] is not None for p in sealed["candidates"])},
        "acceptance": "No automatic PASS or launch; frozen thresholds and full-domain evidence require explicit gate review",
        "scope": "Parent-receipt complete offered cohort, whole-board energy and sampled owned memory; one paired experimental block",
        "GPU_started": False}
    output.mkdir(parents=True, exist_ok=False)
    base.save(output / "report.json", value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("seal")
    for name in ("plan", "model", "resources", "quality", "output"):
        a.add_argument("--" + name, required=True)
    a = sub.add_parser("report")
    for name in ("sealed-dir", "output"):
        a.add_argument("--" + name, required=True)
    a.add_argument("--raw", action="append", required=True)
    args = parser.parse_args(argv)
    value = (seal(args.plan, args.model, args.resources, args.quality, args.output) if args.command == "seal"
             else report(args.sealed_dir, args.raw, args.output))
    print(json.dumps({"status": value.get("status", "SEALED"), "output": str(Path(args.output).resolve()), "GPU_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
