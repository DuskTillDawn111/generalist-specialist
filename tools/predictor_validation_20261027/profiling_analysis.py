"""CPU preparation and strict conversion of externally observed native traces.

This module never imports CUDA or launches a worker. A submit or handler envelope
is not GPU evidence. All observed segments must partition actual START..END;
parallel kernels must have been reduced to nonoverlapping execution intervals by
the trace exporter, retaining its source receipt. Unknown evidence stays unknown.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from gsdeploy.forecasting.profiles import PROFILE_SCHEMA, context_key

TRACE_SCHEMA = "gsdeploy.observed_execution_segments.v1"
PLAN_SCHEMA = "gsdeploy.predictor_profile_plan.v1"
RESIDENTS = {"text#0": "text", "text#1": "text", "generalist_2#0": "generalist_2", "ocr#0": "ocr"}
CONTEXTS = {"S0": ["text#0"], "S1": ["text#1"], "G": ["generalist_2#0"],
            "O": ["ocr#0"], "SS": ["text#0", "text#1"],
            "SG": ["text#0", "generalist_2#0"],
            "SSG": ["text#0", "text#1", "generalist_2#0"],
            "SSGO": list(RESIDENTS)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _positive(value, name, *, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError("Invalid " + name)
    return value


def _ns(value):
    if type(value) is not int or value < 0:
        raise ValueError("Integer monotonic timestamp required")
    return value


def _receipt(value):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
            or not Path(value["path"]).is_file()
            or hashlib.sha256(Path(value["path"]).read_bytes()).hexdigest() != value["sha256"]):
        raise ValueError("Missing or changed observed evidence receipt")


def make_plan(inputs, execution_context_sha256, *, repetitions=3, offsets_ms=(0, 50, 200), seed=527):
    """Predeclare a length-stratified input holdout AND independent repeated trials.

    inputs contain public task/pool_id/input_size only. The final repeat for
    trained inputs is validation; entirely unseen inputs are a separate view.
    The current exact-input provider should return UNKNOWN on unseen inputs.
    """
    if type(repetitions) is not int or repetitions < 3 or not execution_context_sha256:
        raise ValueError("Execution identity and at least three repetitions required")
    pools, seen = {}, set()
    for row in inputs:
        if set(row) != {"task", "pool_id", "input_size"} or row["task"] not in {"text", "ocr"}:
            raise ValueError("Only public task/pool_id/input_size fields accepted")
        _positive(row["input_size"], "input size")
        key = row["task"], row["pool_id"]
        if key in seen or not isinstance(row["pool_id"], str) or not row["pool_id"]:
            raise ValueError("Unique nonempty input identities required")
        seen.add(key); pools.setdefault(row["task"], []).append(copy.deepcopy(row))
    if set(pools) != {"text", "ocr"} or any(len(v) < 4 for v in pools.values()):
        raise ValueError("At least four text and four OCR inputs required for disjoint holdout")
    assignments = {}
    for task, rows in pools.items():
        rows.sort(key=lambda r: (r["input_size"], r["pool_id"]))
        # Interleave sorted length ranges, avoiding a holdout solely of longest inputs.
        for index, row in enumerate(rows):
            assignments[(task, row["pool_id"])] = "unseen_input" if index % 4 == 3 else "known_input"
    offsets = tuple(offsets_ms)
    if not offsets or len(set(offsets)) != len(offsets) or 0 not in offsets:
        raise ValueError("Unique start offsets including zero required")
    for offset in offsets:
        _positive(offset, "start offset", zero=True)
    jobs = []
    for text in pools["text"]:
        for name, members in CONTEXTS.items():
            ocr = pools["ocr"][pools["text"].index(text) % len(pools["ocr"])]
            # O-only jobs deduplicate below; OCR independently covers all OCR inputs.
            pairs = [ocr] if name != "O" else pools["ocr"]
            for image in pairs:
                ids = {a: image["pool_id"] if a == "ocr#0" else text["pool_id"] for a in members}
                unseen = any(assignments[("ocr" if a == "ocr#0" else "text", pool)] == "unseen_input" for a, pool in ids.items())
                for offset in ((0,) if len(members) == 1 else offsets):
                    # Reverse lead/lag to expose asymmetric phase alignment.
                    orders = [members] if offset == 0 else [members, list(reversed(members))]
                    for order in orders:
                        for repeat in range(repetitions):
                            split = "validation" if unseen or repeat == repetitions - 1 else "training"
                            value = {"context": name, "inputs": ids,
                                     "start_offsets_ms": {a: index * offset for index, a in enumerate(order)},
                                     "repeat": repeat, "split": split,
                                     "validation_view": "unseen_input" if unseen else "independent_trial"}
                            value["id"] = digest(value); jobs.append(value)
    jobs = list({j["id"]: j for j in jobs}.values())
    random.Random(seed).shuffle(jobs)
    result = {"schema": PLAN_SCHEMA, "execution_context_sha256": execution_context_sha256,
              "resident_aliases": RESIDENTS, "inputs": sorted(inputs, key=lambda r: (r["task"], r["pool_id"])),
              "jobs": jobs, "seed": seed, "no_retries": True, "GPU_started": False,
              "input_assignments": [{"task": task, "pool_id": pool, "view": view}
                                    for (task, pool), view in sorted(assignments.items())],
              "required_observation": ["actual_worker_START_END", "nonoverlapping_observed_segments",
                                       "SM_DRAM_L2_normalization_evidence", "whole_board_power",
                                       "all_owned_resident_memory", "admission_wait"],
              "claim": "PLAN_ONLY_NOT_GPU_EVIDENCE_NOT_FULL_HORIZON_VALIDATION"}
    result["plan_sha256"] = digest(result)
    return result


def audit_trace(trace):
    """Fail closed on incomplete phase evidence, counter coverage or provenance."""
    issues = []
    if trace.get("schema") != TRACE_SCHEMA or trace.get("kind") != "measured" or trace.get("backend") != "native":
        issues.append("NATIVE_MEASURED_TRACE_REQUIRED")
    if trace.get("clock") != "host_monotonic_ns":
        issues.append("CALIBRATED_COMMON_MONOTONIC_CLOCK_REQUIRED")
    if trace.get("split") not in {"training", "validation"} or not trace.get("execution_context_sha256"):
        issues.append("SPLIT_AND_EXECUTION_ID_REQUIRED")
    try:
        if not trace.get("sources"):
            raise ValueError("No sources")
        for receipt in trace["sources"]:
            _receipt(receipt)
    except (ValueError, TypeError, KeyError):
        issues.append("OBSERVATION_SOURCE_RECEIPT_INVALID")
    seen = set()
    left, right = 0, -1
    try:
        window = trace["observation_window_ns"]
        left, right = _ns(window["start"]), _ns(window["end"])
        if right <= left:
            raise ValueError("Nonempty observation window required")
    except (ValueError, TypeError, KeyError):
        issues.append("UNKNOWN_OBSERVATION_WINDOW")
    for row in trace.get("requests", []):
        rid = row.get("request_id")
        try:
            if not isinstance(rid, str) or not rid or rid in seen or row["alias"] not in RESIDENTS:
                raise ValueError("Invalid request identity")
            seen.add(rid)
            if row["family"] != RESIDENTS[row["alias"]] or row["task"] != ("ocr" if row["alias"] == "ocr#0" else "text"):
                raise ValueError("Invalid request family/task")
            if not isinstance(row["pool_id"], str) or not row["pool_id"] or not isinstance(row["profile_id"], str) or not row["profile_id"]:
                raise ValueError("Missing profile/input identity")
            start, end = _ns(row["handler_start_ns"]), _ns(row["handler_end_ns"])
            if not left <= start < end <= right or row.get("start_basis") != "observed_worker_START":
                raise ValueError("Actual worker START required")
            if not row.get("segments"):
                raise ValueError("Observed segments required")
            previous, segment_ids = start, set()
            for segment in row["segments"]:
                a, b = _ns(segment["start_ns"]), _ns(segment["end_ns"])
                if a != previous or b <= a or b > end or segment["id"] in segment_ids:
                    raise ValueError("Nonoverlapping full execution partition required")
                previous = b; segment_ids.add(segment["id"])
                kind = segment["kind"]
                if kind not in {"host", "gpu", "sync", "cleanup"} or not segment.get("observation_basis"):
                    raise ValueError("Observed segment kind/basis required")
                if set(segment["demand"]) != {"sm", "dram", "l2"}:
                    raise ValueError("SM/DRAM/L2 demanded separately")
                for value in segment["demand"].values():
                    if value is None:
                        evidence = segment.get("demand_evidence")
                        explicit_unknown = (isinstance(evidence, str) and evidence.startswith("UNKNOWN:")) or (
                            isinstance(evidence, dict) and evidence.get("status") == "unknown" and bool(evidence.get("reason")))
                        if kind != "gpu" or not explicit_unknown:
                            raise ValueError("Unknown GPU demand needs explicit reason; never use zero")
                    elif _positive(value, "demand", zero=True) > 1 or (kind != "gpu" and value):
                        raise ValueError("Invalid normalized demand")
                if kind == "gpu" and (segment.get("observation_level") not in {"kernel", "phase"} or not segment.get("demand_evidence")):
                    raise ValueError("GPU observation and counter normalization evidence required")
            if previous != end:
                raise ValueError("Trace must cover complete handler")
            execution = row["execution"]
            if type(execution.get("batch_size")) is not int or execution["batch_size"] != 1 or execution.get("weight_reuse") is not False or not execution.get("backend"):
                raise ValueError("Only current independent-request executor supported")
            wait, grant = row.get("admission_wait_start_ns"), row.get("admission_granted_ns")
            if trace.get("admission_enabled"):
                if not _ns(wait) <= _ns(grant) <= start:
                    raise ValueError("Actual admission wait/grant required")
        except (ValueError, TypeError, KeyError):
            issues.append("UNKNOWN_COMPLETE_SEGMENT_TRACE:" + str(rid))
    if not seen:
        issues.append("NO_REQUESTS")
    # All resident allocations, including idle resident aliases, must be sampled.
    for name, field in (("power_samples", "watts"), ("memory_samples", "owned_used_mib")):
        samples = trace.get(name, [])
        try:
            times = [_ns(s["time_ns"]) for s in samples]
            if len(samples) < 2 or times != sorted(set(times)):
                raise ValueError("Ordered sample series required")
            for sample in samples:
                _positive(sample[field], field, zero=True)
                if name == "memory_samples" and (set(sample["owned_aliases"]) != set(RESIDENTS) or sample["foreign_pids"]):
                    raise ValueError("Owned-only all-resident evidence required")
            if times[0] > left or times[-1] < right:
                raise ValueError("Samples must bracket observed execution")
        except (ValueError, TypeError, KeyError):
            issues.append("UNKNOWN_" + name.upper() + "_COVERAGE")
    return {"known": not issues, "issues": sorted(set(issues)),
            "handler_overlap_is_kernel_overlap": False,
            "scope": "observed execution trace; not predictive validation or confidence guarantee"}


def _interpolate(samples, timestamp, field):
    for left, right in zip(samples, samples[1:]):
        if left["time_ns"] <= timestamp <= right["time_ns"]:
            fraction = (timestamp - left["time_ns"]) / (right["time_ns"] - left["time_ns"])
            return left[field] + fraction * (right[field] - left[field])
    raise ValueError("Uncovered telemetry timestamp")


def convert_training(traces):
    """Produce fit_progress input only from complete TRAINING evidence.

    Baselines require actual singleton activity in the unchanged resident set.
    Contexts missing an in-interval memory sample retain memory=None; interpolated
    board power is explicitly sample based, never kernel energy attribution.
    """
    traces = copy.deepcopy(traces)
    if not traces or any(t.get("split") != "training" for t in traces):
        raise ValueError("Training traces only; validation is never fitted")
    scopes = {t.get("performance_scope") for t in traces}
    if len(scopes) != 1:
        raise ValueError("Profiler-perturbed and natural timing traces cannot be pooled")
    execution_ids = {t.get("execution_context_sha256") for t in traces}
    if len(execution_ids) != 1:
        raise ValueError("Same execution context required")
    if len({t.get("id") for t in traces}) != len(traces) or any(not t.get("id") for t in traces):
        raise ValueError("Unique trace identities required")
    for trace in traces:
        audit = audit_trace(trace)
        if not audit["known"]:
            raise ValueError("UNKNOWN evidence: " + ",".join(audit["issues"]))
    profiles, baseline = {}, {}
    for trace in traces:
        for row in trace["requests"]:
            profile = {"id": row["profile_id"], "family": row["family"], "task": row["task"],
                       "pool_id": row["pool_id"], "execution": row["execution"],
                       "segments": [{"id": s["id"], "kind": s["kind"], "demand": s["demand"]} for s in row["segments"]]}
            if profile["id"] in profiles and profiles[profile["id"]] != profile:
                raise ValueError("Profile identity changed across observations")
            profiles[profile["id"]] = profile
            # Singleton means no other request active at ANY point of this handler.
            others = [r for r in trace["requests"] if r is not row]
            if not any(r["handler_start_ns"] < row["handler_end_ns"] and r["handler_end_ns"] > row["handler_start_ns"] for r in others):
                for segment in row["segments"]:
                    baseline.setdefault((row["profile_id"], segment["id"]), []).append((segment["end_ns"]-segment["start_ns"])/1e6)
    for profile in profiles.values():
        for segment in profile["segments"]:
            samples = baseline.get((profile["id"], segment["id"]), [])
            if len(samples) < 2:
                raise ValueError("UNKNOWN_SOLO_BASELINE:" + profile["id"] + ":" + segment["id"])
            segment["duration_ms"] = {"mean": statistics.mean(samples)}
    contexts, observations, telemetry = {}, [], {}
    for trace in traces:
        segments = [(r, s) for r in trace["requests"] for s in r["segments"]]
        cuts = sorted({trace["observation_window_ns"]["start"], trace["observation_window_ns"]["end"], *[s[k] for _, s in segments for k in ("start_ns", "end_ns")]})
        intervals = []
        for left, right in zip(cuts, cuts[1:]):
            active = [(r, s) for r, s in segments if s["start_ns"] <= left and right <= s["end_ns"]]
            if len({r["alias"] for r, _ in active}) != len(active):
                raise ValueError("Concurrent requests on same alias cannot match independent executor")
            members = {r["alias"]: {"profile_id": r["profile_id"], "segment_id": s["id"]} for r, s in active}
            key = context_key(members)
            contexts.setdefault(key, {"members": members, "scenarios": {"mean": {"progress": {}, "board_watts": None, "peak_memory_mib": None}}, "evidence": "training-observed:" + key})
            intervals.append((left, right, members))
            # Trapezoid integration is exact for the piecewise-linear sample model.
            internal = [s for s in trace["power_samples"] if left < s["time_ns"] < right]
            points = [(left, _interpolate(trace["power_samples"], left, "watts")), *[(s["time_ns"], s["watts"]) for s in internal], (right, _interpolate(trace["power_samples"], right, "watts"))]
            watt_ns = sum((b-a)*(pa+pb)/2 for (a,pa),(b,pb) in zip(points, points[1:]))
            peak = [s["owned_used_mib"] for s in trace["memory_samples"] if left <= s["time_ns"] < right]
            telemetry.setdefault(key, []).append((right-left, watt_ns, max(peak) if peak else None))
        for row, segment in segments:
            observation = {"id": trace["id"] + ":" + row["request_id"] + ":" + segment["id"],
                           "split": "training", "execution_context_sha256": trace["execution_context_sha256"],
                           "target_alias": row["alias"], "profile_id": row["profile_id"], "segment_id": segment["id"],
                           "intervals": [{"duration_ms": (b-a)/1e6, "members": members} for a,b,members in intervals if segment["start_ns"] <= a and b <= segment["end_ns"]]}
            observations.append(observation)
    for key, samples in telemetry.items():
        scenario = contexts[key]["scenarios"]["mean"]
        scenario["board_watts"] = sum(s[1] for s in samples)/sum(s[0] for s in samples)
        peaks = [s[2] for s in samples if s[2] is not None]
        # Missing samples in any interval prevent a complete context peak claim.
        scenario["peak_memory_mib"] = max(peaks) if len(peaks) == len(samples) else None
    document = {"schema": PROFILE_SCHEMA, "execution_context_sha256": traces[0]["execution_context_sha256"],
                "provenance": {"kind": "measured", "split": "training", "sources": sorted({r["sha256"] for t in traces for r in t["sources"]})},
                "profiles": sorted(profiles.values(), key=lambda p: p["id"]), "contexts": contexts}
    return {"document": document, "observations": observations,
            "conversion": {"trace_ids": [t["id"] for t in traces], "training_only": True,
                           "power_scope": "whole_board_piecewise_linear_sample_integral",
                           "memory_scope": "raw_owned_all_resident_sampled_peak_not_continuous_maximum",
                           "not_yet_fitted": True}}


def _source_receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def map_native_collection(raw_path, export_paths=()):
    """Map worker observations to MIXED-PHASE empirical execution profiles.

    The native inference range includes host launch work and transfers. A GPU
    kind denotes the observed native accelerator-call phase, not pure GPU busy
    time. Counter demand remains unknown; kernel tails are reported separately.
    Explicit worker-control gaps are known code-path wall intervals, never
    inferred by subtracting kernel duration from an NVTX envelope.
    """
    raw_path = Path(raw_path).resolve()
    raw = json.loads(raw_path.read_text())
    if raw.get("schema") != "gsdeploy.native_phase_collection.v1" or raw.get("status") != "completed" or raw.get("backend") != "native":
        raise ValueError("Complete native phase collection required")
    for name in ("context", "plan"):
        _receipt(raw[name])
    context = json.loads(Path(raw["context"]["path"]).read_text())
    plan = json.loads(Path(raw["plan"]["path"]).read_text())
    if plan.get("execution_context_sha256") != raw["execution_context_sha256"]:
        raise ValueError("Native plan execution identity mismatch")
    planned_jobs = {job["id"]: job for job in plan["jobs"]}
    if len(planned_jobs) != len(plan["jobs"]):
        raise ValueError("Duplicate planned job identity")
    for source in context.get("sources", []):
        _receipt(source)
    if "execution_sources" in context:
        for source in context["execution_sources"]:
            _receipt(source["frozen"])
        if context.get("sources") != [s["frozen"] for s in context["execution_sources"]]:
            raise ValueError("Native snapshot source list mismatch")
        identity_content = {k:v for k,v in context.items() if k not in {"sources","execution_sources"}}
        identity_content["source_content"] = [{"original_path":s["original_path"],"sha256":s["frozen"]["sha256"]} for s in context["execution_sources"]]
        expected_identity = digest(identity_content)
    else:
        expected_identity = digest(context)  # historical collection without snapshots
    if expected_identity != raw["execution_context_sha256"]:
        raise ValueError("Native context hash mismatch")
    if raw.get("telemetry_errors") or raw.get("runtime_errors") or raw.get("errors"):
        raise ValueError("Collection errors prevent training")
    sources = [_source_receipt(raw_path), raw["context"], raw["plan"]]
    exported_phases = {}
    unattributed_count = 0
    for path in export_paths:
        exported = json.loads(Path(path).read_text())
        if exported.get("schema") != "gsdeploy.nsys_observed_phases.v1":
            raise ValueError("Explicit observed Nsight phase export required")
        sources.append(_source_receipt(path))
        for phase in exported["phases"]:
            exported_phases.setdefault((phase["request_id"], phase["segment_id"]), []).append(phase)
        unattributed_count += len(exported.get("unattributed_kernels", []))
    by_id = {r["request_id"]: r for r in raw["request_records"]}
    if len(by_id) != len(raw["request_records"]):
        raise ValueError("Duplicate native request id")
    traces, reports = [], []
    samples = raw["board_samples"]
    used_requests, used_jobs = set(), set()
    for batch in raw["batches"]:
        if batch["job_id"] not in planned_jobs or batch["job_id"] in used_jobs:
            raise ValueError("Unknown or repeated native batch")
        used_jobs.add(batch["job_id"])
        planned = planned_jobs[batch["job_id"]]
        actual_rows = [by_id[rid] for rid in batch["request_ids"]]
        if len({r["model"] for r in actual_rows}) != len(actual_rows) or {r["model"]:r["pool_id"] for r in actual_rows} != planned["inputs"]:
            raise ValueError("Actual batch input aliases differ from frozen plan")
        for row in actual_rows:
            if row["request_id"] in used_requests:
                raise ValueError("Native request reused across batches")
            used_requests.add(row["request_id"])
            expected = planned["split"]
            actual = row["split"]
            if actual != expected and not (actual == "diagnostic_validation" and expected == "validation"):
                raise ValueError("Native split differs from frozen plan")
        requests = []
        splits = {by_id[rid]["split"] for rid in batch["request_ids"]}
        if len(splits) != 1 or not splits <= {"training", "validation", "diagnostic_validation"}:
            raise ValueError("Explicit homogeneous batch split required")
        for rid in batch["request_ids"]:
            row = by_id[rid]; response = row.get("worker_response", {})
            if row.get("status") != "completed":
                raise ValueError("Complete native request required")
            start = row.get("observed_worker_start_time_ns")
            end = response.get("end_perf_ns")
            if start != response.get("start_perf_ns") or type(start) is not int or type(end) is not int:
                raise ValueError("Actual START event and RESULT monotonic anchors must agree")
            stages = response.get("observed_stages", [])
            if not stages:
                raise ValueError("Observed native phase boundaries required")
            segments, cursor = [], start
            for index, stage in enumerate(stages):
                left, right = stage["start_ns"], stage["end_ns"]
                if not cursor <= left < right <= end:
                    raise ValueError("Native phases are not ordered within handler")
                if left > cursor:
                    segments.append({"id": "worker_control_"+str(index), "kind": "host",
                                     "start_ns": cursor, "end_ns": left,
                                     "demand": dict(sm=0.,dram=0.,l2=0.),
                                     "observation_basis": "Observed START/PHASE wall boundaries surrounding known emit/NVTX/control code; not kernel subtraction"})
                segment = {"id": stage["id"], "kind": stage["kind"], "start_ns": left, "end_ns": right,
                           "observation_basis": "Observed worker PHASE begin/end for explicit native code call; mixed wall phase"}
                segment["demand"] = dict.fromkeys(("sm","dram","l2"), None if stage["kind"] == "gpu" else 0.)
                if stage["kind"] == "gpu":
                    segment.update(observation_level="phase", demand_evidence="UNKNOWN:ERR_NVGPUCTRPERM")
                if stage["kind"] == "sync":
                    segment["scope_note"] = "Host synchronization wait; demand=0 is not evidence GPU is idle; asynchronous tails reported separately"
                # These extra semantics are preserved in mapped evidence, while the
                # strict provider document contains its supported schema only.
                segments.append(segment); cursor = right
                for phase in exported_phases.get((rid,stage["id"]), []):
                    valid = phase.get("clock_mapping_valid") is True
                    kernels = phase.get("kernels", [])
                    tails = [k for k in kernels if valid and k.get("end_ns",0) > right]
                    reports.append({"request_id":rid, "segment_id":stage["id"], "clock_mapping_valid":valid,
                                    "phase_kind":stage["kind"], "associated_kernel_count":len(kernels),
                                    "kernel_union_busy_ns":phase.get("kernel_union_busy_ns"),
                                    "kernel_duration_sum_ns":phase.get("kernel_duration_sum_ns"),
                                    "async_tail_kernel_count":len(tails) if valid else None,
                                    "max_async_tail_ns":max((k["end_ns"]-right for k in tails),default=0) if valid else None,
                                    "kernel_busy_not_phase_duration":True})
            if cursor < end:
                segments.append({"id":"worker_control_end", "kind":"host", "start_ns":cursor, "end_ns":end,
                                 "demand":dict(sm=0.,dram=0.,l2=0.),
                                 "observation_basis":"Observed last PHASE end to worker RESULT end around known control/metadata code"})
            alias = row["model"]; family=RESIDENTS[alias]
            profile_identity = {"family":family,"task":row["task"],"pool_id":row["pool_id"],
                                "execution_context_sha256":raw["execution_context_sha256"],
                                "segments":[(v["id"],v["kind"]) for v in segments]}
            request = {"request_id":rid,"alias":alias,"family":family,"task":row["task"],"pool_id":row["pool_id"],
                       "profile_id":digest(profile_identity),"execution":context["execution"],
                       "handler_start_ns":start,"handler_end_ns":end,"start_basis":"observed_worker_START","segments":segments}
            requests.append(request)
        # Nearest outside samples establish bracketing; do not interpolate owned
        # memory into short phases that have no actual in-phase sample.
        left,right=batch["start_ns"],batch["end_ns"]
        before=[s for s in samples if s["time_ns"] <= left]
        after=[s for s in samples if s["time_ns"] >= right]
        selected=([before[-1]] if before else [])+[s for s in samples if left < s["time_ns"] < right]+([after[0]] if after else [])
        trace={"schema":TRACE_SCHEMA,"kind":"measured","backend":"native","id":batch["job_id"],
               "split":"training" if splits=={"training"} else "validation", "clock":"host_monotonic_ns",
               "execution_context_sha256":raw["execution_context_sha256"],"sources":sources,
               "observation_window_ns":{"start":left,"end":right},"admission_enabled":False,"requests":requests,
               "power_samples":[{"time_ns":s["time_ns"],"watts":s["watts"]} for s in selected],
               "memory_samples":[{k:s[k] for k in ("time_ns","owned_used_mib","owned_aliases","foreign_pids")} for s in selected],
               "scope":"Mixed native phases calibrated empirically; not pure-kernel interference or resource-causal attribution",
               "performance_scope":raw.get("performance_scope")}
        traces.append(trace)
    return {"traces":traces,"mapping_report":{"trace_count":len(traces),"training_traces":sum(t["split"]=="training" for t in traces),
                "validation_traces":sum(t["split"]=="validation" for t in traces),"kernel_phase_reports":reports,
                "unattributed_kernel_count":unattributed_count,"kernel_evidence_available":bool(export_paths),
                "demand_classification":"UNKNOWN_COUNTERS_UNAVAILABLE",
                "async_tail_scope":"Kernels attributed by launch phase can end after that phase; tails are not relabelled as idle host time",
                "validation_scope":"Collection interleaved validation is diagnostic only; final holdout must follow frozen training model"}}


def fit_native_collection(raw_path, export_paths=()):
    from gsdeploy.forecasting.profiles import fit_progress
    mapped = map_native_collection(raw_path, export_paths)
    training = [t for t in mapped["traces"] if t["split"] == "training"]
    result = {"mapping_report":mapped["mapping_report"], "trace_audits":[{"id":t["id"],**audit_trace(t)} for t in mapped["traces"]],
              "known":False,"GPU_started":False,"physical_optimality_certified":False}
    try:
        converted=convert_training(training)
        result["conversion"]=converted["conversion"]
        result["training_input"]=converted
        fitted=fit_progress(converted["document"],converted["observations"])
        result["fitted_document"]=fitted
        result["known"]=True
        result["fit_diagnostics"]=fitted["fit_diagnostics"]
        result["known_context_count"]=len(fitted["contexts"])
        profile_map={p["id"]:p for p in fitted["profiles"]}
        per_kind={}
        for group in fitted["fit_diagnostics"]["groups"]:
            profile=profile_map[group["profile_id"]]
            kind=next(s["kind"] for s in profile["segments"] if s["id"]==group["segment_id"])
            bucket=per_kind.setdefault(kind,{"groups":0,"accepted":0,"rank_deficient":0,"reconstruction_rejected":0})
            bucket["groups"]+=1; bucket["accepted"]+=int(group["accepted"])
            bucket["rank_deficient"]+=int(group["matrix_rank"]<group["columns"])
            bucket["reconstruction_rejected"]+=int(group["matrix_rank"]==group["columns"] and not group["accepted"])
        for bucket in per_kind.values():
            bucket["acceptance_rate"]=bucket["accepted"]/bucket["groups"]
        before=converted["document"]["contexts"]; after=fitted["contexts"]
        possible = math.prod(1+sum(len(p["segments"]) for p in fitted["profiles"] if p["family"]==family) for family in RESIDENTS.values())
        result["reliability_audit"]={"per_kind":per_kind,
            "observed_context_count":len(before), "fitted_supported_context_count":len(after),
            "observed_contexts_dropped_by_fit":len(set(before)-set(after)),
            "fitted_contexts_missing_memory":sum(any(s["peak_memory_mib"] is None for s in c["scenarios"].values()) for c in after.values()),
            "observed_contexts_missing_memory":sum(any(s["peak_memory_mib"] is None for s in c["scenarios"].values()) for c in before.values()),
            "selected_profile_cartesian_context_count":possible,
            "unobserved_cartesian_context_count":possible-len(before),
            "unobserved_scope":"Count only: idle or one selected-profile segment per resident alias. Unseen combinations are not inferred or declared feasible."}
        result["scope"]="Training work reconstruction only; unsupported contexts remain UNKNOWN, holdout accuracy not yet established"
    except ValueError as exc:
        result["reason"]=str(exc)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "audit", "convert", "map-native", "fit-native"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text())
    if args.command == "plan": result=make_plan(**payload)
    elif args.command == "audit": result=audit_trace(payload)
    elif args.command == "convert": result=convert_training(payload)
    elif args.command == "map-native": result=map_native_collection(**payload)
    else: result=fit_native_collection(**payload)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
    return 0 if result.get("known", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
