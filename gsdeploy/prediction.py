"""Finite-cohort FIFO surrogate using exactly the runtime router and arrivals."""

import math
import numpy as np
from .routing import Router
from .trace import expand_routes, make_trace, trace_digest
from .energy import energy_key, energy_norm
from .performance import resolve_service
from .quality import predict_quality
from . import normalization as norm


def components(task_stats, rates, refs, T, M, weights, E=None, Q=None, *, data=None):
    v2 = data is not None and norm.is_v2(data)
    refs = norm.deadlines(data, refs) if v2 else refs
    refs_known = all(refs.get(t) is not None for t in rates)
    total = math.fsum(rates.values())
    L = math.fsum(
        rates[t] / total * task_stats[t]["mean_ms"] / refs[t] for t in sorted(rates)
    ) if refs_known else None
    P = math.fsum(
        rates[t] / total * task_stats[t]["p95_ms"] / refs[t] for t in sorted(rates)
    ) if refs_known else None
    terms = {"T": T, "L": L, "P": P, "M": M}
    if weights.get("Q", 0) > 0:
        terms = {"Q": Q, **terms}
    if "E" in weights:
        terms["E"] = E
    complete = all(v is not None and math.isfinite(v) for v in terms.values())
    if v2:
        complete = complete and norm.ready(data)
    contributions = {
        f"{k}_contribution": -weights[k] * v if v is not None else None
        for k, v in terms.items()
    }
    diagnostic = math.fsum(v for v in contributions.values() if v is not None)
    return {
        **({"normalization_sha256": norm.identity(data),
            "reference_status": norm.contract(data)["status"]} if v2 else {}),
        "Q_norm": None,
        "Q_contribution": 0.0,
        **{f"{k}_norm": v for k, v in terms.items()},
        **contributions,
        "U": diagnostic if complete else None,
        "diagnostic_partial_U": diagnostic,
        "utility_complete": complete,
        "missing_utility_terms": [k for k, v in terms.items() if v is None],
        "utility_schema": (
            norm.SCHEMA if v2 else "six_term_quality_v1" if weights.get("Q", 0) > 0
            else "five_term" if "E" in weights else "legacy_four_term"
        ),
    }


def predict(
    data,
    x,
    routing,
    rates,
    budget,
    refs,
    weights,
    multiplier=1.0,
    phase=None,
    horizon=120.0,
    seed=42,
    arrival_process="deterministic_periodic",
    drain_seconds=60.0,
    energy_reference_watts=300.0,
    startup_amortization_seconds=None,
    trace=None,
    _calibration_validation=False,
):
    if norm.is_v2(data):
        norm.validate(data, rates, refs, horizon, budget)
    if data.get("performance_mode") == "paper_profiled_fifo_v1":
        from .paper_forecast import predict as whole_predict

        if _calibration_validation:
            raise ValueError("Legacy calibration bypass cannot qualify the paper FIFO predictor")
        return whole_predict(data, x, routing, rates, budget, refs, weights,
            multiplier=multiplier, horizon=horizon, seed=seed, arrival_process=arrival_process,
            drain_seconds=drain_seconds, energy_reference_watts=energy_reference_watts,
            startup_amortization_seconds=startup_amortization_seconds, trace=trace)
    if data.get("performance_mode") == "replay_transfer_v1":
        contract = data.get("replay_transfer_v1", {}).get("scoring_contract", {})
        supplied = {"weights": weights, "latency_reference_ms": refs, "gpu_memory_mib": budget,
                    "drain_seconds": drain_seconds, "energy_reference_watts": energy_reference_watts,
                    "startup_amortization_seconds": startup_amortization_seconds, "arrival_process": arrival_process}
        if multiplier <= 0 or any(contract.get(k) != v for k, v in supplied.items()):
            return {"U": None, "diagnostic_partial_U": None, "eligible": False,
                    "constraints_eligible": False, "performance_unknown": True,
                    "reason": "TRANSFER_SCORING_CONTRACT_CHANGED"}
    # Historical phase arguments cannot change task identity or generate per-route arrivals.
    trace = (
        make_trace(rates, horizon, seed, arrival_process) if trace is None else trace
    )
    service, performance_evidence = resolve_service(
        data, x, routing, rates, horizon, seed, arrival_process,
        _allow_unvalidated_calibration=_calibration_validation,
    )
    if service is None:
        return dict(
            U=None,
            diagnostic_partial_U=None,
            eligible=False,
            constraints_eligible=False,
            performance_unknown=True,
            reason=performance_evidence["reason"],
            performance_evidence=performance_evidence,
            offered_requests=len(trace),
            trace_sha256=trace_digest(trace),
        )
    aliases, routes = expand_routes(x, routing)
    router = Router(routes)
    input_ids = None
    if data.get("performance_mode") in {"calibrated_replay_v2", "replay_transfer_v1"}:
        from .calibration_v2 import input_service_ms, replay_input_ids

        input_ids = replay_input_ids(data, trace, seed)
    clocks = {a: 0.0 for a in aliases}  # runtime drains warmup before the formal window
    stats = {t: [] for t in rates}
    inwindow = late = 0
    deadline_rows = []
    for request_index, (offset, task) in enumerate(trace):
        sample_id = input_ids[request_index] if input_ids is not None else None
        alias = router.choose(task)
        m = aliases[alias]
        svc = (
            (input_service_ms(data[data["performance_mode"]], m, task, x[m], sample_id)
             if input_ids is not None else service[m][task])
            / 1000
            * x[m]
            / data["capacity_factors"][m][str(x[m])]
            * multiplier
        )
        finish = max(offset, clocks[alias]) + svc
        clocks[alias] = finish
        stats[task].append((finish - offset) * 1000)
        deadline_rows.append((task, (finish-offset)*1000, finish <= horizon+drain_seconds))
        inwindow += finish <= horizon
        late += finish > horizon + drain_seconds
    taskstats = {
        t: {
            "mean_ms": float(np.mean(v)),
            "p95_ms": float(np.percentile(v, 95)),
            "request_count": len(v),
        }
        for t, v in sorted(stats.items())
        if v
    }
    rho = {
        m: math.fsum(
            routing[t].get(m, 0) * service.get(m, {}).get(t, 0) / 1000 * multiplier
            for t in rates
        )
        / data["capacity_factors"][m][str(n)]
        for m, n in x.items()
    }
    mem = data["safe_factor"] * math.fsum(
        data["memory_table_mib"][m][str(n)] for m, n in x.items()
    )
    profile = data.get("colocation_profiles", {}).get(performance_evidence.get("key"), {})
    measured_memory = profile.get("safe_memory_mib")
    memory_scope = "standalone_residency_sum_proxy"
    if (data.get("memory_mode") == "profiled" and data.get("performance_mode") == "profiled" and type(measured_memory) in (int, float)
            and math.isfinite(measured_memory) and measured_memory > 0):
        mem = float(measured_memory)
        memory_scope = "exact_condition_measured_safe_memory"
    memory_unknown = data.get("memory_mode") == "profiled" and memory_scope != "exact_condition_measured_safe_memory"
    auxiliary = None
    if data.get("performance_mode") in {"calibrated_surrogate", "calibrated_replay_v2", "replay_transfer_v1"}:
        if data.get("performance_mode") == "replay_transfer_v1":
            from .replay_transfer import transfer_auxiliary as surrogate_auxiliary
        elif data.get("performance_mode") == "calibrated_replay_v2":
            from .calibration_v2 import replay_auxiliary as surrogate_auxiliary
        else:
            from .calibration import surrogate_auxiliary

        auxiliary = surrogate_auxiliary(data, x, routing, rates, horizon, energy_reference_watts,
                                        startup_amortization_seconds,
                                        _allow_unvalidated=_calibration_validation)
        mem = auxiliary["safe_memory_mib"]
        memory_unknown = mem is None
        memory_scope = "calibrated_memory_transfer_prediction"
    base = {
        "rho": rho,
        "max_rho": max(rho.values(), default=0),
        "safe_memory_mib": mem,
        "memory_prediction_scope": memory_scope,
        "per_task": taskstats,
        "offered_requests": len(trace),
        "trace_sha256": trace_digest(trace),
        "within_window_requests": int(inwindow),
        "end_backlog_requests": len(trace) - int(inwindow),
        "incomplete_after_drain": int(late),
        "predicted_within_window_QPS": inwindow / horizon,
        "operational_rho_08": max(rho.values(), default=0) <= 0.8 + 1e-8,
        "prediction_scope": performance_evidence["scope"],
        "performance_evidence": performance_evidence,
    }
    if memory_unknown:
        return {**base, "safe_memory_mib": None, "U": None, "diagnostic_partial_U": None,
                "eligible": False, "constraints_eligible": False, "memory_unknown": True,
                "reason": "MISSING_PLAN_MEMORY"}
    if len(taskstats) != len(rates):
        return {
            **base,
            "U": None,
            "diagnostic_partial_U": None,
            "eligible": False,
            "constraints_eligible": False,
            "reason": "MISSING_TASK_OBSERVATIONS",
        }
    admissible = all(
        t in data["models"][m]["supported_tasks"] and m in x
        for t, row in routing.items()
        for m, q in row.items()
        if q > 0
    )
    conservation = set(routing) == set(rates) and all(
        math.isclose(math.fsum(routing[t].values()), q, rel_tol=1e-7, abs_tol=1e-9)
        for t, q in rates.items()
    )
    valid = (
        admissible
        and conservation
        and mem <= budget + 1e-6
        and base["operational_rho_08"]
        and not late
    )
    key = energy_key(
        x, routing, rates, horizon, seed, arrival_process, multiplier, catalog=data
    )
    evidence = data.get("energy_profiles", {}).get(key)
    E = energy_norm(
        evidence, horizon, energy_reference_watts, startup_amortization_seconds
    )
    if auxiliary is not None:
        E = auxiliary["energy_norm"]
        evidence = auxiliary["evidence"]
    Q, quality_evidence = (
        predict_quality(data, x, routing, rates, trace, seed)
        if weights.get("Q", 0) > 0
        else (None, {"status": "not_required", "kind": "historical_admission"})
    )
    T = 1 - inwindow / len(trace)
    M = mem / budget
    if norm.is_v2(data):
        T = norm.deadline_loss(deadline_rows, norm.deadlines(data, refs))
        # Legacy average-power calibration cannot predict a complete episode.
        E = norm.episode_cost(data, evidence)
        if auxiliary is not None:
            E = None
        if memory_scope == "standalone_residency_sum_proxy":
            M = (mem / data["safe_factor"]) / budget
        else:
            M = None  # No raw-peak model supplied by these legacy adapters.
    arithmetic = components(taskstats, rates, refs, T, M, weights, E, Q, data=data)
    missing = []
    if weights.get("Q", 0) > 0 and Q is None:
        missing.append("QUALITY")
    if "E" in weights and E is None:
        missing.append("ENERGY")
    if M is None:
        missing.append("RAW_PEAK_MEMORY")
    if T is None or arithmetic["L_norm"] is None:
        missing.append("DEADLINE_REFERENCES")
    if norm.is_v2(data) and not norm.ready(data):
        missing.append("FROZEN_NORMALIZATION_REFERENCES")
    complete = arithmetic["utility_complete"]
    merged = [v for vs in stats.values() for v in vs]
    return {
        **base,
        **arithmetic,
        "U": arithmetic["U"] if valid and not _calibration_validation else None,
        "constraints_eligible": bool(valid),
        "eligible": bool(valid and complete and not _calibration_validation),
        "diagnostic_only": bool(_calibration_validation),
        **({"performance_unknown": not complete} if norm.is_v2(data) else {}),
        "reason": (
            "OK"
            if valid and complete
            else ("MISSING_" + "_AND_".join(missing) if valid else "CONSTRAINT_OR_DRAIN_FAILURE")
        ),
        "quality_capability": admissible,
        "quality_evidence": quality_evidence,
        "route_conservation": conservation,
        "energy_profile_key": key,
        "energy_evidence": evidence,
        "pooled_request_P95_ms": float(np.percentile(merged, 95)),
        "pooled_request_mean_ms": float(np.mean(merged)),
    }


def predict_scenarios(d, x, r, rates, budget, refs, weights, **context):
    context.pop("multiplier", None)
    scenarios = [
        predict(d, x, r, rates, budget, refs, weights, multiplier=s, **context)
        for s in (1.0, 1.25, 2.0)
    ]
    nominal = scenarios[0]
    values = [p["U"] for p in scenarios]
    return {
        **nominal,
        "scenario_U_values": values,
        "scenario_diagnostic_partial_U": [p["diagnostic_partial_U"] for p in scenarios],
        "scenario_delay_costs": [
            (
                -math.fsum(p[f"{k}_contribution"] for k in "TLP")
                if all(p.get(f"{k}_contribution") is not None for k in "TLP")
                else None
            )
            for p in scenarios
        ],
        "scenario_constraint_valid": [p["constraints_eligible"] for p in scenarios],
        "scenario_performance_unknown": [p.get("performance_unknown", False) for p in scenarios],
        "stress_all_eligible": all(p["eligible"] for p in scenarios),
        "stress_constraints_valid": all(p["constraints_eligible"] for p in scenarios),
        "scenario_U_low": min(values) if all(v is not None for v in values) else None,
        "scenario_U_high": (
            max(v for v in values if v is not None)
            if any(v is not None for v in values)
            else None
        ),
        "scenario_definition": (
            "same cohort; explicitly supplied phase/lifecycle scenarios; unsupported stresses remain UNKNOWN"
            if d.get("performance_mode") == "paper_profiled_fifo_v1" else
            "same cohort; service multipliers 1/1.25/2, not arrival-rate stress"
        ),
        "uncertainty": "HIGH_UNMEASURED_COLOCATION_AND_SERVICE_TAIL",
        "scenario_bounds_are_statistical": False,
    }
