"""Measured-request evaluation with explicit memory evidence and cohort semantics."""

from types import SimpleNamespace
from .prediction import components
from .stability import DEFAULT_THRESHOLDS, reconstruct, classify
from .evidence import admission_covers, quality_covers
from .quality import measure_quality
from .energy import energy_norm, energy_key
from .trace import make_trace, trace_digest, expand_routes
from .routing import Router
from .metrics import summarize
from . import normalization as norm


def evaluate_run(config, raw):
    result = _evaluate_run(config, raw)
    if norm.is_v2(config.catalog):
        result["utility"].update(utility_schema=norm.SCHEMA,
            normalization_sha256=norm.identity(config.catalog),
            reference_status=norm.contract(config.catalog)["status"])
    return result


def _evaluate_run(config, raw):
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
        resource_configuration=raw.get("resource_configuration"),
    )
    E = energy_norm(
        energy,
        seconds,
        config.energy_reference_watts,
        config.startup_amortization_seconds,
    )
    if energy.get("profile_key") != expected_key:
        E = None
    M = memory["safe_memory_mib"] / config.gpu_memory_mib
    episode = None
    if norm.is_v2(config.catalog):
        norm.validate(config.catalog, rates, config.latency_reference_ms,
                      seconds, config.gpu_memory_mib)
        throughput_loss = norm.deadline_loss([
            (r["task"], (r["completion_time_ns"]-r["scheduled_time_ns"])/1e6
             if r.get("completion_time_ns") is not None else None,
             r.get("status") == "completed" and r.get("worker_response", {}).get("success") is True)
            for r in raw["request_records"]
        ], norm.deadlines(config.catalog, config.latency_reference_ms))
        peak = norm.raw_peak(memory)
        M = peak/config.gpu_memory_mib if peak is not None else None
        episode = norm.measured_episode(energy, window)
        episode["episode_protocol"] = norm.observed_protocol(raw["transport_config"], raw.get("priming_policy"))
        E = norm.episode_cost(config.catalog, episode)
        if (energy.get("profile_key") != expected_key
                or not norm.protocol_matches(config.catalog, raw["transport_config"], raw.get("priming_policy"))):
            E = None
    six_term = config.weights.get("Q", 0) > 0
    Q, quality_evidence = (
        measure_quality(config.catalog, raw, rates) if six_term
        else (None, {"status": "not_required", "kind": "historical_admission"})
    )
    arithmetic = components(
        per_task,
        rates,
        config.latency_reference_ms,
        throughput_loss,
        M,
        config.weights,
        E,
        Q,
        data=config.catalog,
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
        validation = config.validate_plan(raw["plan"]["deployment"], raw["plan"]["routing"])
        constraints_valid = validation["constraints_valid"]
    except ValueError:
        constraints_valid = False
    quality_verified = quality_covers(config, raw) if six_term else admission_covers(raw)
    # The new CPU packet is not a validated native execution protocol. A saved
    # record must not gain eligibility merely by adding a resource config/key.
    resource_contract_valid = raw.get("resource_configuration") is None
    eligible = (
        raw.get("status") == "completed"
        and raw["backend"] == "native"
        and memory["measured"]
        and memory["resource_valid"]
        and stability["stability_state_v2"] in {"steady", "recoverable_congested"}
        and quality_verified
        and arithmetic["utility_complete"]
        and cohort_valid
        and constraints_valid
        and resource_contract_valid
        and (not norm.is_v2(config.catalog) or norm.contract(config.catalog)["status"] == "frozen")
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
            "quality_identity_verified": quality_verified,
            "quality_evidence": quality_evidence,
            "cohort_verified": bool(cohort_valid),
            "constraints_verified": constraints_valid,
            "energy_verified": E is not None and energy.get("kind") == "measured",
            **({"episode_energy_evidence": episode} if episode is not None else {}),
            **({"resource_contract_verified": False,
                "resource_contract_reason": "RESOURCE_NATIVE_PROTOCOL_NOT_VALIDATED"}
               if not resource_contract_valid else {}),
        },
    }
