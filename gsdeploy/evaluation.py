"""Measured-request evaluation with explicit memory evidence and cohort semantics."""

from types import SimpleNamespace
from .prediction import components
from .stability import DEFAULT_THRESHOLDS, reconstruct, classify
from .evidence import admission_covers


def evaluate_run(config, raw):
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
    per_task = {}
    for task in rates:
        metrics = raw["per_task_metrics"][task]
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
    system = raw["system_metrics"]
    seconds = raw["measurement_window"]["duration_seconds"]
    throughput_loss = 1 - min(
        1.0,
        system["completion_ratio"],
        system["completed_within_window"] / (sum(rates.values()) * seconds),
    )
    memory = raw["memory_evidence"]
    arithmetic = components(
        per_task,
        rates,
        config.latency_reference_ms,
        throughput_loss,
        memory["safe_memory_mib"] / config.gpu_memory_mib,
        config.weights,
    )
    diagnostic = arithmetic.pop("U")
    quality_verified = admission_covers(raw)
    eligible = (
        raw.get("status") == "completed"
        and raw["backend"] == "native"
        and memory["measured"]
        and memory["resource_valid"]
        and stability["stability_state_v2"] in {"steady", "recoverable_congested"}
        and quality_verified
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
        },
    }
