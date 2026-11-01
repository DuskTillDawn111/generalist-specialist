"""Deployment-local empirical shared-board resources from declared development runs.

Host-visible pending contexts are observational proxies, not causal GPU rates.
Power is a single whole-board quantity for an active-family state. No per-alias
sum of board power, missing-state interpolation, or reserved-seed fitting occurs.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

SCHEMA = "gsdeploy.coupled_shared_board_resources.v1"
MANIFEST_SCHEMA = "gsdeploy.coupled_resource_training_manifest.v1"
ALLOWED_SEEDS = (113, 114, 201, 202, 203, 204)
PHASES = {113: "development", 114: "development", 201: "calibration",
          202: "calibration", 203: "development", 204: "development"}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def deployment_identity(deployment):
    if not isinstance(deployment, dict) or not deployment or any(
        not isinstance(family, str) or not family or type(count) is not int or count <= 0
        for family, count in deployment.items()
    ):
        raise ValueError("Explicit positive integer deployment required")
    return dict(sorted(deployment.items()))


def normalize_counts(deployment, counts):
    deployment = deployment_identity(deployment)
    if not isinstance(counts, dict) or set(counts) - set(deployment):
        raise ValueError("Activity contains an unknown family")
    result = {family: counts.get(family, 0) for family in deployment}
    if any(type(n) is not int or not 0 <= n <= deployment[f] for f, n in result.items()):
        raise ValueError("Active alias counts must lie within the resident deployment")
    return result


def active_counts_from_snapshot(aliases, snapshot, exclude_alias=None):
    if set(aliases) != set(snapshot):
        raise ValueError("Exact resident alias snapshot required")
    deployment = dict(Counter(spec["family"] for spec in aliases.values()))
    counts = dict.fromkeys(deployment, 0)
    for alias, state in snapshot.items():
        if alias != exclude_alias and state.get("outstanding"):
            counts[aliases[alias]["family"]] += 1
    return normalize_counts(deployment, counts)


def make_manifest(sources):
    """Bind an explicit source list. Seed and phase checked before fit/row extraction."""
    result = []
    for source in sources:
        seed, phase = source["seed"], source["phase"]
        if seed not in ALLOWED_SEEDS or PHASES[seed] != phase:
            raise ValueError("Reserved or incorrectly declared training seed")
        receipt = sha(source["path"])
        if source.get("sha256") and source["sha256"] != receipt["sha256"]:
            raise ValueError("Source SHA changed")
        result.append({**receipt, "seed": seed, "phase": phase})
    if not result or len({s["sha256"] for s in result}) != len(result):
        raise ValueError("Nonempty unique training sources required")
    return {"schema": MANIFEST_SCHEMA, "sources": result,
            "allowed_seeds": list(ALLOWED_SEEDS), "reserved_seeds_used": False,
            "claim": "NEW_EXPLORATORY_DEVELOPMENT_FIT_NOT_INDEPENDENT_VALIDATION"}


class PowerIntegral:
    """Piecewise-linear sampled whole-board integral with the original 2s gap rule."""
    def __init__(self, samples):
        points = sorted((s["time_ns"], s["watts"]) for s in samples)
        if len(points) < 2 or any(not finite(w) or w < 0 for _, w in points):
            raise ValueError("Finite nonnegative power samples required")
        self.points, self.times = points, [p[0] for p in points]

    def integrate(self, start, end):
        if end <= start or self.times[0] > start or self.times[-1] < end:
            return None
        energy = 0.0
        begin = max(0, bisect_right(self.times, start) - 1)
        for i in range(begin, len(self.points) - 1):
            a, pa = self.points[i]; b, pb = self.points[i + 1]
            if a >= end:
                break
            left, right = max(start, a), min(end, b)
            if right <= left:
                continue
            if b <= a or b - a > 2_000_000_000:
                return None
            pl = pa + (pb - pa) * (left - a) / (b - a)
            pr = pa + (pb - pa) * (right - a) / (b - a)
            energy += (pl + pr) * 0.5 * (right - left) / 1e9
        return energy


def _run_rows(raw, source):
    seed = raw.get("transport_config", {}).get("random_seed")
    phase = raw.get("experiment_phase")
    if seed != source["seed"] or seed not in ALLOWED_SEEDS:
        raise ValueError("Reserved/mismatched raw seed")
    if source["phase"] != PHASES[seed] or phase not in (None, source["phase"]):
        raise ValueError("Raw training phase mismatch")
    if raw.get("backend") != "native" or raw.get("status") != "completed":
        raise ValueError("Only complete native training is permitted")
    contract = raw["policy_contract"]; aliases = contract["aliases"]
    deployment = deployment_identity(contract["deployment"])
    if dict(Counter(a["family"] for a in aliases.values())) != deployment:
        raise ValueError("Alias/deployment identity mismatch")
    energy = raw["energy_evidence"]; window = raw["measurement_window"]
    if energy.get("scope") != "whole_gpu_board" or energy.get("kind") != "measured" or energy.get("coverage_valid") is not True:
        raise ValueError("Covered measured whole-board energy required for training")
    integral = PowerIntegral(energy["samples"])
    start, end = window["start_time_ns"], window["end_time_ns"]
    joules = integral.integrate(start, end)
    if joules is None or not math.isclose(joules, energy["run_joules"], rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("Original measurement energy integral changed")
    services, events = [], defaultdict(list)
    events[start]; events[end]
    for row in raw["request_records"]:
        if row.get("status") != "completed" or row["model"] not in aliases:
            raise ValueError("Incomplete or unknown alias training request")
        service = row["worker_response"].get("service_ms")
        if not finite(service) or service <= 0:
            raise ValueError("Positive actual worker service required")
        alias = row["model"]
        snapshot = row["routing_decision"]["snapshot"]
        counts = active_counts_from_snapshot(aliases, snapshot, exclude_alias=alias)
        services.append({"family": aliases[alias]["family"], "task": row["task"],
                         "pool_id": row["pool_id"], "other_active_counts": counts,
                         "service_ms": float(service), "seed": seed,
                         "raw_sha256": source["sha256"]})
        a, b = row["enqueued_time_ns"], row["completion_time_ns"]
        if not finite(a) or not finite(b) or b < a:
            raise ValueError("Host enqueue/completion intervals required")
        if b > start and a < end:
            events[max(start, a)].append((alias, 1))
            events[min(end, b)].append((alias, -1))
    outstanding = Counter(); powers = []
    times = sorted(events)
    for i, t in enumerate(times[:-1]):
        for alias, change in events[t]:
            outstanding[alias] += change
            if outstanding[alias] < 0:
                raise ValueError("Negative reconstructed pending work")
        right = times[i + 1]
        if right <= t:
            continue
        counts = dict.fromkeys(deployment, 0)
        for alias, n in outstanding.items():
            if n:
                counts[aliases[alias]["family"]] += 1
        counts = normalize_counts(deployment, counts)
        value = integral.integrate(t, right)
        if value is None:
            raise ValueError("Measurement subinterval lacks power coverage")
        powers.append({"active_counts": counts, "seconds": (right - t) / 1e9,
                       "joules": value, "seed": seed, "raw_sha256": source["sha256"]})
    if not math.isclose(math.fsum(p["joules"] for p in powers), joules, rel_tol=1e-9, abs_tol=1e-5):
        raise ValueError("Shared-board partition must reproduce original energy once")
    latest = max(r["completion_time_ns"] for r in raw["request_records"])
    full_end = max(end, latest)
    whole_j = integral.integrate(start, full_end)
    startup_begin = energy.get("startup_begin_time_ns")
    startup_j = integral.integrate(startup_begin, start) if finite(startup_begin) else None
    memory = raw.get("memory_evidence", {})
    peak = memory.get("safe_memory_mib") if memory.get("measured") and memory.get("resource_valid") else None
    diagnostics = {"source": source, "deployment": deployment,
        "window_seconds": (end-start)/1e9, "window_joules": joules,
        "startup_joules": startup_j, "whole_cohort_joules": whole_j,
        "drain_joules": whole_j-joules if whole_j is not None else None,
        "whole_cohort_energy_known": whole_j is not None, "drain_seconds": (full_end-end)/1e9,
        "safe_memory_mib": peak, "window_energy_partition_reproduced": True,
        "formal_original_E_excludes_drain": True}
    return deployment, services, powers, diagnostics


def fit_manifest(manifest, *, minimum_power_seconds=0.25, minimum_service_observations=2,
                 input_prior=16.0):
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("reserved_seeds_used") is not False:
        raise ValueError("Explicit unreserved source manifest required")
    if not finite(minimum_power_seconds) or minimum_power_seconds <= 0 or type(minimum_service_observations) is not int or minimum_service_observations <= 0 or not finite(input_prior) or input_prior <= 0:
        raise ValueError("Positive evidence thresholds and shrinkage prior required")
    sources = manifest.get("sources", [])
    if not sources or len({s["sha256"] for s in sources}) != len(sources):
        raise ValueError("Unique explicit training runs required")
    # Validate all declarations before opening any raw document.
    if any(s["seed"] not in ALLOWED_SEEDS or PHASES[s["seed"]] != s["phase"] for s in sources):
        raise ValueError("Reserved/mismatched training declaration")
    groups = {}; diagnostics = []
    for source in sources:
        receipt = sha(source["path"])
        if receipt != {k: source[k] for k in ("path", "sha256")}:
            raise ValueError("Training source SHA/path changed")
        raw = json.loads(Path(source["path"]).read_bytes())
        deployment, services, powers, audit = _run_rows(raw, source)
        group = groups.setdefault(key(deployment), {"deployment": deployment, "services": [], "power": [], "runs": []})
        group["services"].extend(services); group["power"].extend(powers); group["runs"].append(audit)
        diagnostics.append(audit)
    deployed = {}
    for dk, group in groups.items():
        strata = {"task_context": defaultdict(list), "input_context": defaultdict(list)}
        for row in group["services"]:
            context = row["other_active_counts"]
            strata["task_context"][key([row["family"], row["task"], context])].append(row)
            strata["input_context"][key([row["family"], row["task"], row["pool_id"], context])].append(row)
        power = defaultdict(list)
        for row in group["power"]:
            power[key(row["active_counts"])].append(row)
        deployed[dk] = {"deployment": group["deployment"], "service_strata": {
            name: {k: {"values_ms": [r["service_ms"] for r in rows],
                         "runs": sorted({r["raw_sha256"] for r in rows}),
                         "seeds": sorted({r["seed"] for r in rows})}
                   for k, rows in table.items()} for name, table in strata.items()},
            "power_states": {k: {"seconds": math.fsum(r["seconds"] for r in rows),
                "joules": math.fsum(r["joules"] for r in rows),
                "segments": [{"watts": r["joules"]/r["seconds"], "seconds": r["seconds"]} for r in rows],
                "runs": sorted({r["raw_sha256"] for r in rows}),
                "seeds": sorted({r["seed"] for r in rows})} for k, rows in power.items()},
            "runs": group["runs"]}
    return {"schema": SCHEMA, "training_manifest": manifest, "training_seeds": sorted({s["seed"] for s in sources}),
        "deployments": deployed, "run_diagnostics": diagnostics,
        "minimum_power_seconds": minimum_power_seconds,
        "minimum_service_observations": minimum_service_observations, "input_prior": input_prior,
        "power_scope": "one_whole_gpu_board_per_shared_activity_state",
        "context_semantics": "Host-visible alias pending counts by resident family; not actual GPU kernel activity",
        "quality_used": False, "reserved_seeds_used": False, "causal_externality_identified": False,
        "limitations": ["Historical selected-path conditional association; no causal intervention certificate.",
            "Exact deployment and supported active-count contexts only; unseen states remain UNKNOWN.",
            "Correlated repeated inputs and power samples are not independent experimental replicates.",
            "Empirical memory range is neither a lower/upper physical bound nor a route-invariant constant.",
            "Startup and complete-cohort drain energy are diagnostics; original formal E definition is unchanged."]}


def weighted_quantile(values, quantile):
    if not finite(quantile) or not 0 < quantile < 1:
        raise ValueError("Quantile must lie strictly between zero and one")
    values = sorted(values); total = math.fsum(w for _, w in values); mass = 0.0
    for value, weight in values:
        mass += weight
        if mass >= total * quantile:
            return value
    return values[-1][0]


class ResourceModel:
    def __init__(self, document):
        if document.get("schema") != SCHEMA or document.get("reserved_seeds_used") is not False or document.get("quality_used") is not False or any(s not in ALLOWED_SEEDS for s in document.get("training_seeds", [])):
            raise ValueError("Unsupported or reserved-contaminated resource model")
        self.document = document

    def _deployment(self, deployment):
        return self.document["deployments"].get(key(deployment_identity(deployment)))

    def predict_power(self, deployment, active_counts):
        counts = normalize_counts(deployment, active_counts); group = self._deployment(deployment)
        row = group["power_states"].get(key(counts)) if group else None
        if not row or row["seconds"] < self.document["minimum_power_seconds"]:
            return {"status": "UNKNOWN_ACTIVITY_CONTEXT", "watts": None, "p90_watts": None,
                    "active_counts": counts, "scope": "whole_gpu_board", "support": row}
        return {"status": "EMPIRICAL_SHARED_BOARD_PROXY", "watts": row["joules"]/row["seconds"],
                "p90_watts": weighted_quantile([(s["watts"], s["seconds"]) for s in row["segments"]], .9),
                "active_counts": counts, "scope": "whole_gpu_board",
                "support": {k: row[k] for k in ("seconds", "runs", "seeds")}, "causal": False}

    def energy_for_timeline(self, deployment, segments):
        parts = []
        for segment in segments:
            duration = segment["duration_ms"]
            if not finite(duration) or duration < 0:
                raise ValueError("Finite nonnegative segment duration required")
            if duration == 0:
                continue
            prediction = self.predict_power(deployment, segment["active_counts"])
            parts.append({"duration_ms": duration, "prediction": prediction,
                          "joules": prediction["watts"] * duration / 1000 if prediction["watts"] is not None else None})
        known = bool(parts) and all(p["joules"] is not None for p in parts)
        return {"status": "EMPIRICAL_SHARED_BOARD_PROXY" if known else "UNKNOWN_ACTIVITY_CONTEXT",
                "joules": math.fsum(p["joules"] for p in parts) if known else None,
                "segments": parts, "scope": "whole_gpu_board_counted_once_per_segment"}

    def predict_service(self, deployment, family, task, pool_id, other_active_counts, quantile=.5):
        context = normalize_counts(deployment, other_active_counts); group = self._deployment(deployment)
        parent = group["service_strata"]["task_context"].get(key([family, task, context])) if group else None
        child = group["service_strata"]["input_context"].get(key([family, task, pool_id, context])) if group else None
        if not parent or len(parent["values_ms"]) < self.document["minimum_service_observations"]:
            return {"status": "UNKNOWN_SERVICE_CONTEXT", "mean_ms": None, "quantile_ms": None,
                    "conservative_ms": None, "support": None, "context": context}
        weight = len(child["values_ms"])/(len(child["values_ms"])+self.document["input_prior"]) if child else 0.0
        values = [(x, (1-weight)/len(parent["values_ms"])) for x in parent["values_ms"]]
        if child:
            values += [(x, weight/len(child["values_ms"])) for x in child["values_ms"]]
        mean = math.fsum(x*w for x, w in values)
        qvalue = weighted_quantile(values, quantile)
        return {"status": "EMPIRICAL_CONTEXT_SERVICE_PROXY", "mean_ms": mean,
                "quantile_ms": qvalue, "conservative_ms": max(mean, qvalue),
                "support": {"task_context": len(parent["values_ms"]), "input_context": len(child["values_ms"]) if child else 0,
                            "runs": parent["runs"], "seeds": parent["seeds"]},
                "context": context, "causal": False}

    def predict_residual(self, deployment, family, task, pool_id, other_active_counts, age_ms, quantile=.5):
        if not finite(age_ms) or age_ms < 0:
            raise ValueError("Finite nonnegative service age required")
        context = normalize_counts(deployment, other_active_counts); group = self._deployment(deployment)
        parent = group["service_strata"]["task_context"].get(key([family, task, context])) if group else None
        child = group["service_strata"]["input_context"].get(key([family, task, pool_id, context])) if group else None
        empty = {"status": "UNKNOWN_RESIDUAL_SUPPORT", "mean_ms": None, "quantile_ms": None,
                 "conservative_ms": None, "age_ms": age_ms, "context": context,
                 "support": None, "tail_extrapolation": False}
        if not parent or len(parent["values_ms"]) < self.document["minimum_service_observations"]:
            return empty
        weight = len(child["values_ms"])/(len(child["values_ms"])+self.document["input_prior"]) if child else 0.0
        values = [(x-age_ms, (1-weight)/len(parent["values_ms"])) for x in parent["values_ms"] if x > age_ms]
        if child:
            values += [(x-age_ms, weight/len(child["values_ms"])) for x in child["values_ms"] if x > age_ms]
        survivors = sum(x > age_ms for x in parent["values_ms"])
        if survivors < self.document["minimum_service_observations"] or not values:
            empty["support"] = {"task_context_survivors": survivors, "task_context": len(parent["values_ms"])}
            return empty
        mass = math.fsum(w for _, w in values)
        mean = math.fsum(x*w for x,w in values)/mass; qvalue = weighted_quantile(values,quantile)
        return {"status": "EMPIRICAL_CONTEXT_RESIDUAL_PROXY", "mean_ms": mean,
                "quantile_ms": qvalue, "conservative_ms": max(mean,qvalue),
                "age_ms": age_ms, "context": context, "tail_extrapolation": False,
                "support": {"task_context_survivors": survivors, "task_context": len(parent["values_ms"]),
                            "runs": parent["runs"], "seeds": parent["seeds"]}, "causal": False}

    def service_externality(self, deployment, family, task, pool_id, before_other_counts, after_other_counts):
        before = self.predict_service(deployment, family, task, pool_id, before_other_counts)
        after = self.predict_service(deployment, family, task, pool_id, after_other_counts)
        known = before["mean_ms"] is not None and after["mean_ms"] is not None
        return {"status": "OBSERVATIONAL_CONTEXT_ASSOCIATION" if known else "UNKNOWN_CONTEXT_EXTERNALITY",
                "before": before, "after": after,
                "service_ratio": after["mean_ms"]/before["mean_ms"] if known else None,
                "delta_service_ms": after["mean_ms"]-before["mean_ms"] if known else None,
                "causal": False, "counterfactual_certified": False}

    def predict_memory(self, deployment):
        group = self._deployment(deployment)
        values = [r["safe_memory_mib"] for r in group["runs"] if finite(r["safe_memory_mib"])] if group else []
        return {"status": "EMPIRICAL_DEPLOYMENT_RANGE" if values else "UNKNOWN_MEMORY",
                "safe_memory_mib_mean": statistics.fmean(values) if values else None,
                "observed_min_mib": min(values) if values else None, "observed_max_mib": max(values) if values else None,
                "physical_lower_bound_mib": None, "physical_upper_bound_mib": None,
                "same_residency_route_delta_mib": None, "route_invariant": False}


CONTROLLED_SCHEMA = "gsdeploy.coupled_controlled_resources.v1"


def _count_active(aliases, active, exclude_alias=None):
    if set(active)-set(aliases):
        raise ValueError("Active alias outside deployment")
    result = dict.fromkeys({spec["family"] for spec in aliases.values()}, 0)
    for alias in active:
        if alias != exclude_alias:
            result[aliases[alias]["family"]] += 1
    return result


def _controlled_partition(rows, target):
    """Actual worker-interval overlap, independently of routing snapshots."""
    start, end = target["start_unix_ns"], target["end_unix_ns"]
    endpoints = sorted({start,end,*[max(start,min(end,p)) for r in rows if r is not target
                                  for p in (r["start_unix_ns"],r["end_unix_ns"])]})
    segments = defaultdict(float)
    for left,right in zip(endpoints,endpoints[1:]):
        if right <= left:
            continue
        counts = dict.fromkeys(target["deployment"],0)
        for other in rows:
            if other is not target and other["start_unix_ns"] < right and other["end_unix_ns"] > left:
                counts[other["family"]] += 1
        segments[key(counts)] += (right-left)/1e6
    return dict(segments)


def _fit_progress(rows, contexts, training_seeds):
    import numpy as np
    from scipy.optimize import nnls
    selected = [r for r in rows if r["seed"] in training_seeds]
    baselines = defaultdict(list)
    for row in selected:
        if all(not any(json.loads(k).values()) for k in row["durations_ms"]):
            baselines[row["pool_id"]].append(row["service_ms"])
    baseline = {pid:statistics.median(values) for pid,values in baselines.items()}
    usable = [r for r in selected if r["pool_id"] in baseline]
    matrix = np.asarray([[r["durations_ms"].get(c,0)/baseline[r["pool_id"]] for c in contexts] for r in usable],dtype=float)
    if not usable or not contexts:
        return {"status":"UNKNOWN_NO_SOLO_INPUT_BASELINE","progress":None,"baselines_ms":baseline,"rows":len(usable),"rank":0,"columns":len(contexts)}
    rank = int(np.linalg.matrix_rank(matrix)); singular = np.linalg.svd(matrix,compute_uv=False)
    scales = np.linalg.norm(matrix,axis=0)
    design = {"rows":len(usable),"columns":len(contexts),"rank":rank,"full_column_rank":rank==len(contexts),
              "singular_values":singular.tolist(),"condition_number":float(singular[0]/singular[-1]) if rank==len(contexts) else None,
              "input_baselines":len(baseline),"missing_solo_input_rows":len(selected)-len(usable)}
    if rank != len(contexts) or (scales<=0).any():
        return {"status":"UNKNOWN_RANK_DEFICIENT","progress":None,"baselines_ms":baseline,"design":design}
    normalized = matrix/scales
    beta,residual = nnls(normalized,np.ones(len(usable))); beta = beta/scales
    if not np.isfinite(beta).all() or (beta<=0).any():
        return {"status":"UNKNOWN_NONPOSITIVE_PROGRESS","progress":None,"baselines_ms":baseline,"design":design}
    errors = matrix@beta-1
    return {"status":"FULL_COLUMN_RANK_EMPIRICAL_PROGRESS_PROXY","progress":dict(zip(contexts,beta.tolist())),
            "baselines_ms":baseline,"baseline_distributions_ms":dict(baselines),"design":design,
            "fit_work_errors":{"mean_absolute_relative_error":float(np.mean(abs(errors))),
                               "max_absolute_relative_error":float(np.max(abs(errors))),"rmse":float(np.sqrt(np.mean(errors**2)))},
            "normalized_nnls_residual":float(residual),"causal_certificate":False}


def fit_controlled(path):
    """Fit progress from actual changing overlap, never batch latency as a rate."""
    path = Path(path).resolve(); receipt = sha(path); raw = json.loads(path.read_bytes())
    if raw.get("schema") not in ("gsdeploy.coupled_controlled_calibration.v1","gsdeploy.controlled_calibration.v1"):
        raise ValueError("Explicit new controlled calibration schema required")
    if raw.get("status") not in ("completed","COMPLETE") or raw.get("errors",[]):
        raise ValueError("Complete successful controlled calibration required")
    if raw.get("backend") != "native" or raw.get("calibration_usable") is not True:
        raise ValueError("Usable measured native calibration required")
    deployment = deployment_identity(raw["deployment"]); aliases = raw["aliases"]
    if dict(Counter(a["family"] for a in aliases.values())) != deployment:
        raise ValueError("Controlled alias/deployment mismatch")
    batches = raw["batches"]
    if {b["seed"] for b in batches} != {301,302}:
        raise ValueError("Only declared new calibration seeds301/302 permitted")
    records = raw["request_records"]; by_id = {r["request_id"]:r for r in records}
    if len(by_id) != len(records) or sorted(rid for b in batches for rid in b["request_ids"]) != sorted(by_id):
        raise ValueError("Each calibration request must belong to exactly one batch")
    services = []; per_batch = []; diagnostics = []
    for batch in batches:
        rows = []
        if batch.get("all_completed") is not True or len(set(batch["aliases"])) != len(batch["aliases"]):
            raise ValueError("Each isolated declared batch must have completed")
        for rid in batch["request_ids"]:
            row = by_id[rid]; interval = row["actual_service_interval"]
            if row.get("status") != "completed" or row["task"] != "text" or row["model"] not in aliases or row["pool_id"] != batch["pool_id"]:
                raise ValueError("Controlled request identity/status mismatch")
            service = row["worker_response"]["service_ms"]
            a,b = interval["start_unix_ns"],interval["end_unix_ns"]
            am,bm = interval["start_time_ns"],interval["end_time_ns"]
            if not finite(service) or service<=0 or not (a<b and am<bm) or not math.isclose((b-a)/1e6,service,rel_tol=1e-5,abs_tol=.005):
                raise ValueError("Controlled worker interval/service mismatch")
            rows.append({"request_id":rid,"family":aliases[row["model"]]["family"],"alias":row["model"],
                "task":"text","pool_id":row["pool_id"],"service_ms":service,"seed":batch["seed"],
                "start_unix_ns":a,"end_unix_ns":b,"start_time_ns":am,"end_time_ns":bm,
                "deployment":deployment,"batch_id":batch["id"]})
        if sorted(r["alias"] for r in rows) != sorted(batch["aliases"]):
            raise ValueError("Batch aliases must exactly match its measured requests")
        for row in rows:
            row["durations_ms"] = _controlled_partition(rows,row)
            if not math.isclose(math.fsum(row["durations_ms"].values()),row["service_ms"],rel_tol=1e-5,abs_tol=.005):
                raise ValueError("Controlled overlap partition fails service reproduction")
        services.extend(rows); per_batch.append(rows)
    family_models = {}
    for family in sorted({r["family"] for r in services}):
        rows = [r for r in services if r["family"]==family]
        contexts = sorted({k for r in rows for k in r["durations_ms"]})
        fitted = _fit_progress(rows,contexts,{301,302}); folds = []
        for held in (301,302):
            trained = _fit_progress(rows,contexts,{301,302}-{held}); held_rows = [r for r in rows if r["seed"]==held]
            errors = []; unknown = 0
            for row in held_rows:
                work = trained.get("baselines_ms",{}).get(row["pool_id"])
                if trained.get("progress") is None or work is None:
                    unknown += 1; continue
                predicted = math.fsum(duration*trained["progress"][context] for context,duration in row["durations_ms"].items())
                errors.append(predicted/work-1)
            folds.append({"held_seed":held,"training_seeds":sorted({301,302}-{held}),"fit":trained,
                "heldout_rows":len(held_rows),"unknown_rows":unknown,
                "held_work_mean_absolute_relative_error":statistics.fmean(abs(x) for x in errors) if errors else None,
                "held_work_max_absolute_relative_error":max(map(abs,errors)) if errors else None,
                "held_work_mean_signed_relative_error":statistics.fmean(errors) if errors else None})
        alias_errors=defaultdict(list)
        for row in rows:
            baseline=fitted.get("baselines_ms",{}).get(row["pool_id"])
            if baseline and fitted.get("progress"):
                predicted=math.fsum(d*fitted["progress"][c] for c,d in row["durations_ms"].items())
                alias_errors[row["alias"]].append(predicted/baseline-1)
        family_models[family] = {"contexts":contexts,"fit":fitted,"leave_seed_out":folds,
            "replica_residual_diagnostics":{alias:{"rows":len(values),"mean_signed_work_error":statistics.fmean(values),
                "mean_absolute_work_error":statistics.fmean(abs(x) for x in values)} for alias,values in alias_errors.items()},
            "same_family_replica_exchangeability_assumed":True,
            "scope":"same-input work scale, actual worker overlap durations, NNLS relative progress by concurrent family counts"}
    energy = raw["energy_evidence"]
    if energy.get("scope") != "whole_gpu_board" or energy.get("kind") != "measured" or energy.get("coverage_valid") is not True:
        raise ValueError("Controlled power must have whole-board scope")
    integral = PowerIntegral(energy["samples"])
    # Use actual service intervals for shared power states, including idle gaps.
    start = min(r["start_time_ns"] for r in services); end = max(r["end_time_ns"] for r in services)
    endpoints = sorted({start,end,*[p for r in services for p in (r["start_time_ns"],r["end_time_ns"])]})
    grouped_power = defaultdict(list)
    for left,right in zip(endpoints,endpoints[1:]):
        counts = dict.fromkeys(deployment,0)
        for row in services:
            if row["start_time_ns"] < right and row["end_time_ns"] > left:
                counts[row["family"]] += 1
        counts = normalize_counts(deployment,counts); joules = integral.integrate(left,right)
        if joules is None:
            raise ValueError("Controlled whole-board power lacks complete coverage")
        grouped_power[key(counts)].append({"seconds":(right-left)/1e9,"joules":joules})
    power_states = {c:{"seconds":math.fsum(x["seconds"] for x in values),"joules":math.fsum(x["joules"] for x in values),
        "segments":[{"seconds":x["seconds"],"watts":x["joules"]/x["seconds"]} for x in values]}
        for c,values in grouped_power.items()}
    partition = math.fsum(p["joules"] for p in power_states.values()); measured = integral.integrate(start,end)
    if measured is None or not math.isclose(partition,measured,rel_tol=1e-9,abs_tol=1e-5):
        raise ValueError("Controlled shared-board energy partition must reproduce exactly once")
    return {"schema":CONTROLLED_SCHEMA,"source":receipt,"deployment":deployment,"training_seeds":[301,302],
        "training_phase":"new_controlled_calibration","reserved_seeds_used":False,"quality_used":False,
        "request_count":len(records),"batch_count":len(batches),"family_models":family_models,
        "power_states":power_states,"power_scope":"one_whole_gpu_board_per_actual_service_activity_state",
        "power_partition":{"start_time_ns":start,"end_time_ns":end,"joules":partition,"measured_joules":measured,"reproduced":True},
        "memory_evidence":raw.get("memory_evidence"),"independent_validation":False,"causal_certificate":False,
        "limits":["Controlled in-training overlap supports a conditional shared-rate model, not an unqualified causal or optimality proof.",
            "Progress is fit to changing interval overlap; fixed batch mean latency is never directly interpreted as a dynamic rate.",
            "Fresh-seed predictive checks and native workflow validation remain necessary.",
            "Text-only active states; background/image/OCR/ASR work or unseen concurrency states remain UNKNOWN.",
            "Input solo baselines and shared rates are empirical; within-input differences and host/worker timing alignment can cause error."]}


class ControlledProvider:
    def __init__(self,document,deployment):
        if document.get("schema") != CONTROLLED_SCHEMA or document.get("reserved_seeds_used") is not False or document.get("training_seeds") != [301,302]:
            raise ValueError("Unsupported controlled provider")
        self.document = document; self.deployment = deployment_identity(deployment)
        if self.deployment != document["deployment"]:
            raise ValueError("Controlled provider exact deployment mismatch")

    def _validate(self,aliases):
        if dict(Counter(a["family"] for a in aliases.values())) != self.deployment:
            raise ValueError("Provider alias deployment mismatch")

    def _state(self,request,active,aliases):
        self._validate(aliases)
        if request.get("task") != "text" or any(r.get("task") != "text" for r in active.values()):
            return None,None,None
        family = aliases[request["alias"]]["family"]
        model = self.document["family_models"].get(family)
        fit = model.get("fit") if model else None
        context = key(_count_active(aliases,active,request["alias"]))
        progress = fit.get("progress",{}).get(context) if fit and fit.get("progress") else None
        baselines = fit.get("baseline_distributions_ms",{}).get(request["pool_id"]) if fit else None
        return progress,baselines,context

    def service(self,request,active,aliases,scenario="mean"):
        progress,baselines,context = self._state(request,active,aliases)
        if not progress or not baselines:
            return {"known":False,"context_supported":False,"service_ms":None,"status":"UNKNOWN_CONTROLLED_CONTEXT_OR_INPUT"}
        if scenario not in ("mean","p90"):
            raise ValueError("Controlled provider scenario must be mean or p90")
        baseline = statistics.fmean(baselines) if scenario=="mean" else weighted_quantile([(x,1) for x in baselines],.9)
        return {"known":True,"context_supported":True,"service_ms":baseline/progress,
                "status":"CONTROLLED_OVERLAP_PROGRESS_PROXY","progress_multiplier":progress,
                "baseline_work_ms":baseline,"other_active_counts":json.loads(context),"causal_certificate":False}

    def residual(self,request,age_ms,active,aliases,scenario="mean"):
        if not finite(age_ms) or age_ms<0:
            raise ValueError("Finite nonnegative inferred service age required")
        progress,baselines,context = self._state(request,active,aliases)
        if not progress or not baselines:
            return {"known":False,"tail_supported":False,"remaining_ms":None,"status":"UNKNOWN_CONTROLLED_RESIDUAL_CONTEXT"}
        durations = [b/progress for b in baselines]; survivors = [t-age_ms for t in durations if t>age_ms]
        if not survivors:
            return {"known":False,"tail_supported":False,"remaining_ms":None,"status":"UNKNOWN_CONTROLLED_RESIDUAL_TAIL",
                    "max_observed_constant_context_ms":max(durations),"inferred_age_ms":age_ms}
        value = statistics.fmean(survivors) if scenario=="mean" else weighted_quantile([(x,1) for x in survivors],.9)
        return {"known":True,"tail_supported":True,"remaining_ms":value,"status":"CONTROLLED_SUPPORTED_CONSTANT_CONTEXT_RESIDUAL_PROXY",
                "survivors":len(survivors),"inferred_age_ms":age_ms,"tail_extrapolation":False,
                "limitation":"Host-inferred age and constant past context proxy; no reconstructed historical work certificate"}

    def power(self,active,aliases,scenario="mean"):
        self._validate(aliases)
        if any(r.get("task") != "text" for r in active.values()):
            return {"known":False,"watts":None,"status":"UNKNOWN_BACKGROUND_POWER"}
        counts = _count_active(aliases,active)
        row = self.document["power_states"].get(key(counts))
        if not row or row["seconds"] < .25:
            return {"known":False,"watts":None,"status":"UNKNOWN_CONTROLLED_POWER_CONTEXT"}
        watts = row["joules"]/row["seconds"] if scenario=="mean" else weighted_quantile([(s["watts"],s["seconds"]) for s in row["segments"]],.9)
        return {"known":True,"watts":watts,"status":"CONTROLLED_SHARED_BOARD_PROXY","scope":"whole_gpu_board",
                "active_counts":counts,"support_seconds":row["seconds"]}


class HistoricalProvider:
    """Compatibility adapter; activity interpretation remains historical pending."""
    def __init__(self,document,deployment):
        self.model=ResourceModel(document); self.deployment=deployment_identity(deployment)

    def service(self,request,active,aliases,scenario="mean"):
        value=self.model.predict_service(self.deployment,aliases[request["alias"]]["family"],request["task"],request["pool_id"],
            _count_active(aliases,active,request["alias"]),.9 if scenario=="p90" else .5)
        ms=value["mean_ms"] if scenario=="mean" else value["quantile_ms"]
        return {"known":ms is not None,"context_supported":ms is not None,"service_ms":ms,"evidence":value,
                "historical_pending_association_only":True}

    def residual(self,request,age_ms,active,aliases,scenario="mean"):
        value=self.model.predict_residual(self.deployment,aliases[request["alias"]]["family"],request["task"],request["pool_id"],
            _count_active(aliases,active,request["alias"]),age_ms,.9 if scenario=="p90" else .5)
        ms=value["mean_ms"] if scenario=="mean" else value["quantile_ms"]
        return {"known":ms is not None,"tail_supported":ms is not None,"remaining_ms":ms,"evidence":value}

    def power(self,active,aliases,scenario="mean"):
        value=self.model.predict_power(self.deployment,_count_active(aliases,active))
        watts=value["watts"] if scenario=="mean" else value["p90_watts"]
        return {"known":watts is not None,"watts":watts,"evidence":value}


def load_provider(document,deployment):
    if document.get("schema")==CONTROLLED_SCHEMA:
        return ControlledProvider(document,deployment)
    if document.get("schema")==SCHEMA:
        return HistoricalProvider(document,deployment)
    raise ValueError("Unsupported shared-resource provider schema")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    model = fit_manifest(json.loads(args.manifest.read_bytes()))
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "resources.json"
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"resource_model": sha(path), "training_runs": len(model["run_diagnostics"]),
                      "training_seeds": model["training_seeds"], "deployments": len(model["deployments"])}))


if __name__ == "__main__":
    main()
