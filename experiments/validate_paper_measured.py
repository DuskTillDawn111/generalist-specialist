"""Bounded reuse of two native holdouts; no fitting, GPU launch or gate bypass.

The paper predictor is explicitly NOT RUN if matching training profiles are
absent. Observed validation timings are never converted into forecast inputs.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsdeploy.calibration_v2 import input_service_ms, replay_input_ids
from gsdeploy.config import SystemConfig
from gsdeploy.prediction import components, predict
from gsdeploy.routing import Router
from gsdeploy.trace import expand_routes, make_trace, trace_digest

BASE = ROOT / "artifacts/static-study-gpu-20261022"
CONFIG = BASE / "search-v2-v5-K8/system-config.json"
HOLDOUTS = [BASE / ("v2-holdout-runs/" + name + "-attempt1/run.json") for name in (
    "v2_holdout_A_g2_image075_ocr100", "v2_holdout_B_g2_image100_ocr075")]


def receipt(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest(), bytes=path.stat().st_size)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    return dict(mean_ms=math.fsum(values) / len(values),
                p95_ms=float(np.percentile(values, 95)), request_count=len(values))


def replay_ledger(config, deployment, routing, seed, horizon):
    """Build only from public trace/input selection and frozen old law."""
    trace = make_trace(config.workload, horizon, seed, config.arrival_process)
    aliases, routes = expand_routes(deployment, routing)
    router, clocks, rows = Router(routes), defaultdict(float), []
    for i, ((arrival, task), sample) in enumerate(zip(
            trace, replay_input_ids(config.catalog, trace, seed))):
        alias = router.choose(task)
        model = aliases[alias]
        ms = input_service_ms(config.catalog["calibrated_replay_v2"],
                              model, task, deployment[model], sample)
        ms *= deployment[model] / config.catalog["capacity_factors"][model][str(deployment[model])]
        start = max(arrival, clocks[alias])
        clocks[alias] = start + ms / 1000
        rows.append(dict(arrival_index=i, task=task, pool_id=sample, alias=alias,
                         arrival_seconds=arrival, predicted_service_ms=ms,
                         predicted_wait_ms=(start-arrival)*1000,
                         predicted_e2e_ms=(clocks[alias]-arrival)*1000))
    return trace, rows


def match_observations(expected, raw):
    """Reject lost/reordered/misbound requests, preserving the full cohort."""
    observed = sorted(raw["request_records"], key=lambda r: r["arrival_index"])
    if len(expected) != len(observed) or len(observed) != raw["measurement_window"]["offered_requests"]:
        raise ValueError("Offered cohort size changed")
    rows = []
    for e, o in zip(expected, observed):
        if (e["arrival_index"] != o["arrival_index"] or e["task"] != o["task"]
                or e["pool_id"] != o["pool_id"] or e["alias"] != o["model"]
                or abs(e["arrival_seconds"] - o["scheduled_offset_seconds"]) > 1e-9):
            raise ValueError("Request identity/trace/route differs")
        if o["status"] != "completed" or not o["worker_response"].get("success"):
            raise ValueError("Incomplete/failed request cannot be dropped")
        rows.append({**e, "request_id": o["request_id"],
            "observed_handler_ms": o["worker_latency_ms"],
            "observed_worker_queue_wait_ms": o["worker_queue_wait_ms"],
            "observed_e2e_ms": o["e2e_latency_ms"],
            "observed_schedule_delay_ms": o["schedule_delay_ms"]})
    return rows


def source_coverage(raw):
    """Field inventory of legacy records; no fabricated phase/READY timestamps."""
    rows = raw["request_records"]
    return dict(requests=len(rows),
        handler_start_end=sum(all(k in r["worker_response"] for k in ("start_ns", "end_ns")) for r in rows),
        parent_receipt=sum("completion_time_ns" in r for r in rows),
        ordered_segments=sum(bool(r.get("segments") or r["worker_response"].get("segments")) for r in rows),
        explicit_worker_ready=sum(any(k in r or k in r["worker_response"] for k in (
            "worker_next_ready_ns", "worker_next_ready_time_ns", "ready_time_ns")) for r in rows),
        ready_inferred_from_receipt=False)


def evaluate(config, deployment, routing, seed, horizon):
    return predict(config.catalog, deployment, routing, config.workload,
        config.gpu_memory_mib, config.latency_reference_ms, config.weights,
        horizon=horizon, seed=seed, arrival_process=config.arrival_process,
        drain_seconds=config.drain_seconds, energy_reference_watts=config.energy_reference_watts,
        startup_amortization_seconds=config.startup_amortization_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = SystemConfig.load(CONFIG)  # Original validation gate remains active.
    spec = config.catalog["calibrated_replay_v2"]
    raws = [json.loads(p.read_text()) for p in HOLDOUTS]
    deployment = raws[0]["plan"]["deployment"]
    if any(r["plan"]["deployment"] != deployment for r in raws):
        raise ValueError("One deployment required")
    sources = [receipt(CONFIG), receipt(Path(__file__))]
    conditions, task_rows, request_rows = [], [], []
    for label, raw, path in zip("AB", raws, HOLDOUTS):
        src = receipt(path)
        if not any(s["id"] == raw["experiment_job_id"] and s["sha256"] == src["sha256"] for s in spec["sources"]):
            raise ValueError("Original source binding changed")
        sources.append(src)
        if raw["backend"] != "native" or raw["status"] != "completed" or raw["errors"]:
            raise ValueError("Native complete run required")
        seed = raw["transport_config"]["random_seed"]
        horizon = raw["measurement_window"]["duration_seconds"]
        routing = raw["plan"]["routing"]
        trace, ledger = replay_ledger(config, deployment, routing, seed, horizon)
        rows = match_observations(ledger, raw)
        pred = evaluate(config, deployment, routing, seed, horizon)
        if not pred.get("eligible") or pred["trace_sha256"] != trace_digest(trace):
            raise ValueError("Old predictor refused frozen condition: " + str(pred.get("reason")))
        per_task = {}
        for task in sorted(config.workload):
            selected = [r for r in rows if r["task"] == task]
            measured = distribution([r["observed_e2e_ms"] for r in selected])
            derived = distribution([r["predicted_e2e_ms"] for r in selected])
            for key in ("mean_ms", "p95_ms"):
                if not math.isclose(derived[key], pred["per_task"][task][key], abs_tol=1e-7):
                    raise ValueError("Independent FIFO ledger does not match predictor")
            per_task[task] = measured
            task_rows.append(dict(condition=label, task=task, requests=len(selected),
                predicted_mean_ms=derived["mean_ms"], measured_mean_ms=measured["mean_ms"],
                predicted_p95_ms=derived["p95_ms"], measured_p95_ms=measured["p95_ms"],
                predicted_handler_mean_ms=math.fsum(r["predicted_service_ms"] for r in selected)/len(selected),
                measured_handler_mean_ms=math.fsum(r["observed_handler_ms"] for r in selected)/len(selected),
                measured_queue_wait_mean_ms=math.fsum(r["observed_worker_queue_wait_ms"] for r in selected)/len(selected)))
        archived = raw["evaluation"]["utility"]
        measured = components(per_task, config.workload, config.latency_reference_ms,
            archived["T_norm"], archived["M_norm"], config.weights,
            E=archived["E_norm"], Q=archived["Q_norm"])
        if not math.isclose(measured["U"], archived["U"], abs_tol=1e-12):
            raise ValueError("Measured six-term arithmetic differs")
        handler_expected = math.fsum(r["predicted_service_ms"] for r in rows)
        handler_observed = math.fsum(r["observed_handler_ms"] for r in rows)
        conditions.append(dict(condition=label, seed=seed, horizon_seconds=horizon,
            deployment=deployment, routing=routing, offered_requests=len(rows),
            predicted_U=pred["U"], measured_U=measured["U"],
            utility_absolute_error=abs(pred["U"]-measured["U"]),
            handler_relative_error=abs(handler_expected/handler_observed-1),
            mean_relative_error=abs(pred["L_norm"]/measured["L_norm"]-1),
            p95_relative_error=abs(pred["P_norm"]/measured["P_norm"]-1),
            trace_sha256=trace_digest(trace), coverage=source_coverage(raw),
            mapping_counts=dict(Counter(r["alias"] for r in rows))))
        request_rows.extend(dict(condition=label, **r) for r in rows)
        save(out / (label + "-old-prediction.json"), pred)
    # Cross the two public routes and seeds: two additional evaluations only.
    crossed = []
    for trace_index, raw in enumerate(raws):
        values = []
        for route_index, route_raw in enumerate(raws):
            if trace_index == route_index:
                p = json.loads((out / ("AB"[trace_index] + "-old-prediction.json")).read_text())
            else:
                p = evaluate(config, deployment, route_raw["plan"]["routing"],
                    raw["transport_config"]["random_seed"], raw["measurement_window"]["duration_seconds"])
            values.append(p["U"])
        crossed.append(dict(seed=raw["transport_config"]["random_seed"],
            route_A_U=values[0], route_B_U=values[1], A_minus_B_U=values[0]-values[1],
            source="old_predictor_only_no_matched_trace_measurement"))
    training = []
    for source in spec["sources"]:
        if source["id"] not in spec["training_ids"]:
            continue
        src = receipt(source["path"])
        if src["sha256"] != source["sha256"]:
            raise ValueError("Training source changed")
        sources.append(src)
        raw = json.loads(Path(source["path"]).read_text())
        training.append(dict(id=source["id"], deployment=raw["plan"]["deployment"],
            exact_target_deployment=raw["plan"]["deployment"] == deployment,
            coverage=source_coverage(raw)))
    # A bounded alternative-data check reads reviews, not 137MB raw captures.
    index_path = ROOT / "artifacts/predictor-public181-resource-training-v1-20261029-gpu01/report.json"
    sources.append(receipt(index_path))
    recent = json.loads(index_path.read_text())
    alternatives = []
    for attempt in recent["completed_attempts"][:2]:
        path = Path(attempt["training_review"]["path"])
        src = receipt(path)
        if src["sha256"] != attempt["training_review"]["sha256"]:
            raise ValueError("Alternative source review changed")
        sources.append(src)
        review = json.loads(path.read_text())
        alternatives.append(dict(attempt_id=attempt["attempt_id"], status=review["status"],
            cohort=review["cohort"], training_only=True,
            prior_history=review["histories"],
            independent_prediction_accuracy_validated=review["independent_prediction_accuracy_validated"],
            main_queue_provider_eligible=review["main_queue_provider_eligible"],
            final_gain_eligible=review["final_gain_eligible"],
            reason_not_used="Different four-owner execution; complete native prefixes are training evidence, not fitted segment/context/lifecycle laws"))
    try:
        hardware = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        hardware = dict(returncode=hardware.returncode, stdout=hardware.stdout, stderr=hardware.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        hardware = dict(error=type(exc).__name__ + ": " + str(exc))
    report = dict(schema="gsdeploy.paper_measured_validation.v1",
        status="PARTIAL_OLD_BASELINE_REPRODUCED_NEW_PROVIDER_NOT_RUN",
        new_gpu_measurements=0, fitted=False, optimizer_experiments_run=0,
        legacy_prediction_calls=4, conditions=conditions, crossed_routes=crossed,
        observed_condition_delta_U=conditions[0]["measured_U"]-conditions[1]["measured_U"],
        causal_route_gain_established=False,
        ranking_caveat="A/B use different arrival/input seeds; observed delta is a condition comparison, not a matched-trace causal route effect",
        paper_predictor=dict(status="NOT_RUN_MISSING_MATCHED_TRAINING_PROFILES",
            U=None, independent_accuracy_qualified=False, final_gain_eligible=False,
            missing=["ordered per-input segments under matched execution identity",
                     "identifiable full-active-context progress laws, including idle and overlap",
                     "distinct handler-end/parent-receipt/worker-READY lifecycle laws",
                     "context-bound whole-board power and all-resident memory laws"],
            interface_limitation="Exact per-input active-context lookup requires coverage beyond standalone tables; fitting a transferable law is separate work, not a schema conversion"),
        legacy_training_inventory=training, bounded_alternative_check=alternatives,
        hardware_access=hardware, elapsed_seconds=time.perf_counter()-started)
    save(out / "report.json", report)
    save(out / "source-manifest.json", sources)
    table(out / "per-task.csv", task_rows)
    table(out / "requests.csv", request_rows)
    table(out / "same-trace-routing.csv", crossed)
    print(json.dumps({k:report[k] for k in ("status", "legacy_prediction_calls", "new_gpu_measurements", "elapsed_seconds")}, indent=2))
    return 0  # Successful diagnostic; explicitly not an accuracy acceptance.


if __name__ == "__main__":
    raise SystemExit(main())
