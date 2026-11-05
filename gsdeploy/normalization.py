"""Versioned, fixed-reference six-term costs. Never infer missing references."""

import hashlib
import json
import math

SCHEMA = "six_term_reference_v2"
KEY = "utility_normalization"


def contract(data):
    return data.get(KEY)


def is_v2(data):
    return contract(data) is not None


def identity(data):
    value = contract(data)
    return (hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
            if value is not None else None)


def finite(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def validate(data, rates, refs, horizon, budget):
    spec = contract(data)
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise ValueError("Explicit six_term_reference_v2 reference contract required")
    if spec.get("status") not in {"pending", "frozen", "synthetic_fixture"}:
        raise ValueError("Unknown normalization reference status")
    if not spec.get("provenance") or set(spec.get("tasks", {})) != set(rates):
        raise ValueError("Normalization needs provenance and exactly the active tasks")
    if spec.get("workload_rps") != rates or spec.get("horizon_seconds") != horizon:
        raise ValueError("Normalization reference workload/horizon mismatch")
    if spec.get("memory_budget_mib") != budget:
        raise ValueError("Normalization reference memory budget mismatch")
    pending = spec["status"] == "pending"
    for task, row in spec["tasks"].items():
        a, z, d = (row.get(k) for k in ("best_loss", "worst_loss", "deadline_ms"))
        if a is not None or z is not None or not pending:
            if not finite(a) or not finite(z) or z <= a:
                raise ValueError("Quality anchors require finite worst_loss > best_loss")
        if d is not None or not pending:
            if not finite(d, True) or d != refs[task]:
                raise ValueError("Deadline must match latency_reference_ms")
        if not row.get("provenance"):
            raise ValueError("Task reference provenance required")
    energy = spec.get("episode_reference_joules")
    if energy is not None or not pending:
        if not finite(energy, True):
            raise ValueError("Positive fixed reference episode energy required")
    if not pending and not spec.get("reference_receipts"):
        raise ValueError("Frozen references require source receipts")
    if spec["status"] == "frozen":
        if (not spec.get("measurement_context")
                or spec["measurement_context"] != data.get("measurement_context")
                or spec.get("input_identity") != data.get("quality", {}).get("identity")):
            raise ValueError("Frozen references must bind GPU/checkpoints and input/protocol identity")
        k = spec.get("deadline_multiplier")
        if not finite(k, True) or not spec.get("episode_protocol"):
            raise ValueError("Frozen references require a deadline multiplier and full episode protocol")
        for row in spec["tasks"].values():
            mean = row.get("reference_mean_e2e_ms")
            if not finite(mean, True) or not math.isclose(row["deadline_ms"], k*mean, rel_tol=1e-12):
                raise ValueError("Frozen deadline must equal k times measured reference mean e2e")
    return spec


def deadlines(data, fallback):
    spec = contract(data)
    return ({t: spec["tasks"][t].get("deadline_ms") for t in fallback}
            if spec is not None else fallback)


def quality_loss(data, task, value):
    if value is None or not is_v2(data):
        return value
    row = contract(data)["tasks"][task]
    a, z = row.get("best_loss"), row.get("worst_loss")
    return (value-a)/(z-a) if a is not None and z is not None and z > a else None


def quality_floor(data, rates):
    """Native losses >=0, hence v2 costs >= -a/(z-a), including unseen routes."""
    if not is_v2(data):
        return 0.0
    total = math.fsum(rates.values())
    values = [quality_loss(data, t, 0.0) for t in rates]
    return (math.fsum(rates[t]/total*v for t, v in zip(rates, values))
            if all(v is not None for v in values) else -math.inf)


def ready(data):
    spec = contract(data)
    return spec is None or spec.get("status") in {"frozen", "synthetic_fixture"}


def require_ready(data, rates, refs, horizon, budget):
    if is_v2(data):
        validate(data, rates, refs, horizon, budget)
        if not ready(data):
            raise ValueError("NORMALIZATION_REFERENCES_PENDING: freeze independent references before optimization")


def measured_episode(evidence, window):
    """Reconstruct full-episode joules from telemetry, not cached sums."""
    from .energy import integrate_power

    start = evidence.get("startup_begin_time_ns")
    end = window.get("drain_end_time_ns")
    valid = (type(start) is int and type(end) is int
             and start < window["start_time_ns"] <= window["end_time_ns"] <= end
             and not window.get("aborted") and not evidence.get("sampling_errors"))
    joules = integrate_power(evidence.get("samples", []), start, end) if valid else None
    return {**evidence, "episode_joules": joules,
            "episode_coverage_valid": joules is not None,
            "episode_start_time_ns": start, "episode_end_time_ns": end}


def protocol_matches(data, transport, priming_policy):
    """A denominator measured with another warmup/drain policy is incompatible."""
    spec = contract(data)
    protocol = spec.get("episode_protocol") if spec is not None else None
    if protocol is None:
        return spec is None or spec.get("status") != "frozen"
    return protocol == observed_protocol(transport, priming_policy)


def observed_protocol(transport, priming_policy):
    return dict(warmup_seconds=transport.get("warmup_seconds"),
        priming_policy=priming_policy, drain_seconds=transport.get("drain_timeout_seconds"),
        arrival_process=transport.get("arrival_process"))


def deadline_loss(latencies, refs):
    """latencies contains (task, response_ms, success); denominator is offered."""
    rows = list(latencies)
    if not rows or any(refs.get(t) is None for t, _, _ in rows):
        return None
    good = sum(ok and finite(ms) and ms <= refs[t] for t, ms, ok in rows)
    return 1-good/len(rows)


def episode_cost(data, evidence):
    ref = contract(data).get("episode_reference_joules")
    if (ref is None or not isinstance(evidence, dict)
            or evidence.get("scope") != "whole_gpu_board"
            or evidence.get("kind") not in {"measured", "synthetic_fixture", "predicted_episode"}
            or not evidence.get("episode_coverage_valid")
            or not finite(evidence.get("episode_joules"))):
        return None
    spec = contract(data)
    if spec.get("status") == "frozen" and evidence.get("episode_protocol") != spec.get("episode_protocol"):
        return None
    return evidence["episode_joules"]/ref


def raw_peak(memory):
    """Require raw owned observations, never recover a peak by dividing a proxy."""
    samples = [s for s in [memory.get("before"), memory.get("after"),
                          *memory.get("samples", [])] if isinstance(s, dict)]
    values = [s.get("owned_used_mib") for s in samples]
    return (max(values) if memory.get("measured") and memory.get("episode_coverage_valid")
            and values and all(finite(v) for v in values) else None)
