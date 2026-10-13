"""Periodic FIFO predictor and shared normalized utility, in milliseconds."""

import math
import numpy as np


def components(task_stats, rates, refs, T, M, weights):
    total = sum(rates.values())
    L = sum((rates[t] / total * task_stats[t]["mean_ms"] / refs[t] for t in rates))
    P = sum((rates[t] / total * task_stats[t]["p95_ms"] / refs[t] for t in rates))
    c = {
        "Q_contribution": 0.0,
        "T_contribution": -weights["T"] * T,
        "L_contribution": -weights["L"] * L,
        "P_contribution": -weights["P"] * P,
        "M_contribution": -weights["M"] * M,
    }
    return {
        "Q_norm": None,
        "T_norm": T,
        "L_norm": L,
        "P_norm": P,
        "M_norm": M,
        **c,
        "U": sum(c.values()),
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
    phase=0.0,
    horizon=120.0,
):
    rho = {
        m: sum(
            (
                routing.get(t, {}).get(m, 0)
                * data["models"][m]["service_time_ms"][t]
                / 1000
                * multiplier
                for t in rates
                if t in data["models"][m]["supported_tasks"]
            )
        )
        / data["capacity_factors"][m][str(n)]
        for (m, n) in x.items()
    }
    mem = data["safe_factor"] * sum(
        (data["memory_table_mib"][m][str(n)] for (m, n) in x.items())
    )
    admissible = all(
        (
            t in data["models"][m]["supported_tasks"] and m in x
            for (t, rr) in routing.items()
            for (m, q) in rr.items()
            if q > 1e-09
        )
    )
    conservation = all(
        (abs(sum(routing.get(t, {}).values()) - q) < 1e-06 for (t, q) in rates.items())
    )
    stats = {t: [] for t in rates}
    inwindow = 0
    offered = 0
    backlog = 0
    late = 0
    for m, n in x.items():
        arrival = []
        tasks = []
        svc = []
        replica = []
        for ti, t in enumerate(rates):
            q = routing.get(t, {}).get(m, 0)
            if q <= 1e-09:
                continue
            a = np.arange(-10 * q + phase * (ti + 1), horizon * q, 1.0) / q
            a = a[(a >= -10) & (a < horizon)]
            arrival.extend(a)
            tasks.extend([t] * len(a))
            svc.extend(
                [
                    data["models"][m]["service_time_ms"][t]
                    / 1000
                    * n
                    / data["capacity_factors"][m][str(n)]
                    * multiplier
                ]
                * len(a)
            )
            replica.extend(np.arange(len(a)) % n)
        if not arrival:
            continue
        arrival = np.array(arrival)
        tasks = np.array(tasks)
        svc = np.array(svc)
        replica = np.array(replica)
        for ri in range(n):
            ix = np.flatnonzero(replica == ri)
            ix = ix[np.argsort(arrival[ix], kind="stable")]
            a = arrival[ix]
            s = svc[ix]
            tt = tasks[ix]
            c = np.cumsum(s)
            finish = c + np.maximum.accumulate(a - (c - s))
            use = a >= 0
            lat = (finish - a)[use] * 1000
            tt = tt[use]
            f = finish[use]
            offered += len(f)
            inwindow += int(sum(f <= horizon))
            backlog += int(sum(f > horizon))
            late += int(sum(f > horizon + 60))
            for t in set(tt):
                stats[t].extend(lat[tt == t])
    taskstats = {
        t: {
            "mean_ms": float(np.mean(v)),
            "p95_ms": float(np.percentile(v, 95)),
            "request_count": len(v),
        }
        for (t, v) in stats.items()
        if v
    }
    if len(taskstats) != len(rates):
        return {"U": None, "eligible": False, "reason": "MISSING_TASK", "rho": rho}
    merged = [v for vs in stats.values() for v in vs]
    T = 1 - min(
        1.0,
        (offered - late) / max(1, offered),
        inwindow / (sum(rates.values()) * horizon),
    )
    norms = components(taskstats, rates, refs, T, mem / budget, weights)
    eligible = (
        admissible
        and conservation
        and (mem <= budget + 1e-06)
        and (max(rho.values(), default=0) < 1)
        and (late == 0)
    )
    return {
        **norms,
        "U": norms["U"] if eligible else None,
        "completed_subset_diagnostic_U_NOT_sortable": (
            norms["U"] if not eligible else None
        ),
        "eligible": eligible,
        "rho": rho,
        "max_rho": max(rho.values(), default=0),
        "operational_rho_08": max(rho.values(), default=0) <= 0.8 + 1e-08,
        "safe_memory_mib": mem,
        "per_task": taskstats,
        "pooled_request_P95_ms": float(np.percentile(merged, 95)),
        "pooled_request_mean_ms": float(np.mean(merged)),
        "predicted_within_window_QPS": inwindow / horizon,
        "end_backlog_requests": backlog,
        "incomplete_after_60s_drain": late,
        "quality_capability": admissible,
        "route_conservation": conservation,
        "stability_scope": "nominal long-run demand<1 plus finite-window drain; sample/interference stability unverified",
    }


def predict_scenarios(d, x, r, rates, budget, refs, weights):
    preds = [
        predict(d, x, r, rates, budget, refs, weights, 1.0, ph) for ph in [0.0, 0.37]
    ]
    nominal = preds[1]
    stress = preds + [
        predict(d, x, r, rates, budget, refs, weights, s, 0.37) for s in [1.25, 2.0]
    ]
    vals = [p["U"] for p in stress if p["U"] is not None]
    return {
        **nominal,
        "scenario_U_values": [p["U"] for p in stress],
        "stress_all_eligible": all((p["eligible"] for p in stress)),
        "scenario_U_low": min(vals) if len(vals) == len(stress) else None,
        "scenario_U_high": max(vals) if vals else None,
        "uncertainty": "HIGH_UNMEASURED_COLOCATION_AND_SERVICE_TAIL",
        "scenario_bounds_are_statistical": False,
    }
