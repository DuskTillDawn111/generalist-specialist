"""Finite-cohort FIFO surrogate using exactly the runtime router and arrivals."""

import math
import numpy as np
from .routing import Router
from .trace import expand_routes, make_trace, trace_digest
from .energy import energy_key, energy_norm


def components(task_stats, rates, refs, T, M, weights, E=None):
    total = math.fsum(rates.values())
    L = math.fsum(
        rates[t] / total * task_stats[t]["mean_ms"] / refs[t] for t in sorted(rates)
    )
    P = math.fsum(
        rates[t] / total * task_stats[t]["p95_ms"] / refs[t] for t in sorted(rates)
    )
    terms = {"T": T, "L": L, "P": P, "M": M}
    if "E" in weights:
        terms["E"] = E
    complete = all(v is not None for v in terms.values())
    contributions = {
        f"{k}_contribution": -weights[k] * v if v is not None else None
        for k, v in terms.items()
    }
    diagnostic = math.fsum(v for v in contributions.values() if v is not None)
    return {
        "Q_norm": None,
        "Q_contribution": 0.0,
        **{f"{k}_norm": v for k, v in terms.items()},
        **contributions,
        "U": diagnostic if complete else None,
        "diagnostic_partial_U": diagnostic,
        "utility_complete": complete,
        "utility_schema": "five_term" if "E" in weights else "legacy_four_term",
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
):
    # Historical phase arguments cannot change task identity or generate per-route arrivals.
    trace = (
        make_trace(rates, horizon, seed, arrival_process) if trace is None else trace
    )
    aliases, routes = expand_routes(x, routing)
    router = Router(routes)
    clocks = {a: 0.0 for a in aliases}  # runtime drains warmup before the formal window
    stats = {t: [] for t in rates}
    inwindow = late = 0
    for offset, task in trace:
        alias = router.choose(task)
        m = aliases[alias]
        svc = (
            data["models"][m]["service_time_ms"][task]
            / 1000
            * x[m]
            / data["capacity_factors"][m][str(x[m])]
            * multiplier
        )
        finish = max(offset, clocks[alias]) + svc
        clocks[alias] = finish
        stats[task].append((finish - offset) * 1000)
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
            routing[t].get(m, 0)
            * data["models"][m]["service_time_ms"].get(t, 0)
            / 1000
            * multiplier
            for t in rates
        )
        / data["capacity_factors"][m][str(n)]
        for m, n in x.items()
    }
    mem = data["safe_factor"] * math.fsum(
        data["memory_table_mib"][m][str(n)] for m, n in x.items()
    )
    base = {
        "rho": rho,
        "max_rho": max(rho.values(), default=0),
        "safe_memory_mib": mem,
        "per_task": taskstats,
        "offered_requests": len(trace),
        "trace_sha256": trace_digest(trace),
        "within_window_requests": int(inwindow),
        "end_backlog_requests": len(trace) - int(inwindow),
        "incomplete_after_drain": int(late),
        "predicted_within_window_QPS": inwindow / horizon,
        "operational_rho_08": max(rho.values(), default=0) <= 0.8 + 1e-8,
        "prediction_scope": "constant-service independent FIFO replicas; colocated interference and service tails unmeasured",
    }
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
    arithmetic = components(
        taskstats, rates, refs, 1 - inwindow / len(trace), mem / budget, weights, E
    )
    complete = arithmetic["utility_complete"]
    merged = [v for vs in stats.values() for v in vs]
    return {
        **base,
        **arithmetic,
        "U": arithmetic["U"] if valid else None,
        "constraints_eligible": bool(valid),
        "eligible": bool(valid and complete),
        "reason": (
            "OK"
            if valid and complete
            else ("MISSING_ENERGY" if valid else "CONSTRAINT_OR_DRAIN_FAILURE")
        ),
        "quality_capability": admissible,
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
        "scenario_constraint_valid": [p["constraints_eligible"] for p in scenarios],
        "stress_all_eligible": all(p["eligible"] for p in scenarios),
        "stress_constraints_valid": all(p["constraints_eligible"] for p in scenarios),
        "scenario_U_low": min(values) if all(v is not None for v in values) else None,
        "scenario_U_high": (
            max(v for v in values if v is not None)
            if any(v is not None for v in values)
            else None
        ),
        "scenario_definition": "same cohort; service multipliers 1/1.25/2, not arrival-rate stress",
        "uncertainty": "HIGH_UNMEASURED_COLOCATION_AND_SERVICE_TAIL",
        "scenario_bounds_are_statistical": False,
    }
