"""Measured-request evaluation with explicit memory evidence and cohort semantics."""

from types import SimpleNamespace
from .prediction import components
from .stability import DEFAULT_THRESHOLDS, reconstruct, classify
from .evidence import admission_covers
from .energy import energy_norm, energy_key
from .trace import make_trace, trace_digest, expand_routes
from .routing import Router
from .metrics import summarize


def evaluate_run(config, raw):
    window = raw.get("measurement_window")
    if window and (
        window.get("aborted")
        or window.get("drain_end_time_ns", 0) < window["end_time_ns"]
    ):
        return {
            "stability": {"stability_state_v2": "invalid_observation"},
            "stability_evidence": {
                "reason": "WINDOW_ABORTED",
                "offered_requests": len(raw.get("request_records", [])),
            },
            "utility": {"U": None, "eligible": False, "reason": "WINDOW_ABORTED"},
        }
    evidence = reconstruct(
        raw,
        {
            alias: SimpleNamespace(
                service_time_ms=config.catalog["models"][family]["service_time_ms"]
            )
            for alias, family in raw["replica_to_family"].items()
        },
        DEFAULT_THRESHOLDS,
    )
    stability = classify(evidence, DEFAULT_THRESHOLDS)
    rates = config.workload
    aliases, routes = expand_routes(raw["plan"]["deployment"], raw["plan"]["routing"])
    measurement = SimpleNamespace(
        measurement_seconds=window["duration_seconds"],
        workload_rps=rates,
        routing_rps=routes,
        deployment={alias: 1 for alias in aliases},
    )
    # Recompute derived metrics from the saved cohort, not cached summary values.
    replayed_tasks, replayed_system, _ = summarize(
        raw["request_records"], measurement, window
    )
    per_task = {}
    for task in rates:
        metrics = replayed_tasks[task]
        if (
            metrics["mean_e2e_latency_ms"] is None
            or metrics["p95_e2e_latency_ms"] is None
        ):
            return {
                "stability": stability,
                "stability_evidence": evidence,
                "utility": {
                    "U": None,
                    "eligible": False,
                    "reason": "MISSING_TASK_OBSERVATIONS",
                },
            }
        per_task[task] = {
            "mean_ms": metrics["mean_e2e_latency_ms"],
            "p95_ms": metrics["p95_e2e_latency_ms"],
        }
    system = replayed_system
    seconds = raw["measurement_window"]["duration_seconds"]
    offered = len(raw.get("request_records", []))
    if offered == 0:
        return {
            "stability": stability,
            "stability_evidence": evidence,
            "utility": {"U": None, "eligible": False, "reason": "EMPTY_COHORT"},
        }
    throughput_loss = 1 - system["completed_within_window"] / offered
    memory = raw["memory_evidence"]
    energy = raw.get("energy_evidence", {})
    expected_key = energy_key(
        raw["plan"]["deployment"],
        raw["plan"]["routing"],
        rates,
        seconds,
        raw["transport_config"]["random_seed"],
        raw["transport_config"]["arrival_process"],
        catalog=config.catalog,
    )
    E = energy_norm(
        energy,
        seconds,
        config.energy_reference_watts,
        config.startup_amortization_seconds,
    )
    if energy.get("profile_key") != expected_key:
        E = None
    arithmetic = components(
        per_task,
        rates,
        config.latency_reference_ms,
        throughput_loss,
        memory["safe_memory_mib"] / config.gpu_memory_mib,
        config.weights,
        E,
    )
    diagnostic = arithmetic.pop("U")
    transport = raw["transport_config"]
    trace = make_trace(
        rates, seconds, transport["random_seed"], transport["arrival_process"]
    )
    records = raw.get("request_records", [])
    aliases, routes = expand_routes(raw["plan"]["deployment"], raw["plan"]["routing"])
    router = Router(routes)
    window = raw["measurement_window"]
    cohort_valid = len(records) == len(trace) and window.get(
        "trace_sha256"
    ) == trace_digest(trace)
    for i, ((offset, task), record) in enumerate(zip(trace, records)):
        expected_alias = router.choose(task)
        cohort_valid &= (
            record.get("task") == task
            and record.get("model") == expected_alias
            and record.get("request_id") == f"measurement:{i}"
            and record.get("scheduled_time_ns")
            == window["start_time_ns"] + round(offset * 1e9)
        )
    try:
        config.validate_plan(raw["plan"]["deployment"], raw["plan"]["routing"])
        constraints_valid = True
    except ValueError:
        constraints_valid = False
    quality_verified = admission_covers(raw)
    eligible = (
        raw.get("status") == "completed"
        and raw["backend"] == "native"
        and memory["measured"]
        and memory["resource_valid"]
        and stability["stability_state_v2"] in {"steady", "recoverable_congested"}
        and quality_verified
        and cohort_valid
        and constraints_valid
        and system["completion_ratio"] == 1
        and raw["measurement_window"].get("offered_requests") == offered
        and (
            "E" not in config.weights
            or (E is not None and energy.get("kind") == "measured")
        )
    )
    return {
        "stability": stability,
        "stability_evidence": evidence,
        "utility": {
            **arithmetic,
            "U": diagnostic if eligible else None,
            "diagnostic_U": diagnostic,
            "eligible": bool(eligible),
            "memory_scope": memory["scope"],
            "quality_admission_verified": quality_verified,
            "cohort_verified": bool(cohort_valid),
            "constraints_verified": constraints_valid,
            "energy_verified": E is not None and energy.get("kind") == "measured",
        },
    }
