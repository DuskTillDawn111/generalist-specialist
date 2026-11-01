"""Full-cohort six-term arithmetic; missing evidence never becomes zero."""

from __future__ import annotations

import math
import statistics

from ..prediction import components
from .workload import number


def percentile(values, q=.95):
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def _expected_quality_totals(entry, task, pool_id):
    """Training-predicted *per request* units, aggregated before division.

    A family/task ratio alone cannot mix OCR/ASR paths with unequal reference
    lengths correctly. Input-conditioned estimates may override a declared
    family fallback; neither interface accepts future validation outputs.
    """
    if isinstance(entry, dict) and 'by_input' in entry:
        entry = entry['by_input'].get(pool_id, entry.get('fallback'))
    if isinstance(entry, dict):
        if (entry.get('basis') != 'expected_per_request_totals' or not entry.get('evidence')
                or type(entry.get('numerator')) not in (int, float)
                or type(entry.get('denominator')) not in (int, float)):
            return None
        numerator, denominator = entry['numerator'], entry['denominator']
        if (not math.isfinite(numerator) or not math.isfinite(denominator)
                or numerator < 0 or denominator <= 0
                or task in {'text', 'image'} and (denominator != 1 or numerator > 1)):
            return None
        return numerator, denominator
    if (task in {'text', 'image'} and type(entry) in (int, float)
            and math.isfinite(entry) and 0 <= entry <= 1):
        return entry, 1.
    return None


def score_forecast(forecast, completed, aliases, rates, refs, quality_loss, *,
                   phase_start_ns, now_ns, deadline_ns, memory_budget_mib,
                   energy_reference_watts, weights=None, observed_peak_memory_mib=None,
                   observed_board_energy_joules=None, memory_safe_factor=1.):
    weights = {k: 1 / 6 for k in "QTLPME"} if weights is None else weights
    if set(weights) != set("QTLPME"):
        raise ValueError("Six explicit objective terms required")
    for weight in weights.values():
        number(weight, "objective weight")
    if abs(sum(weights.values()) - 1.) > 1e-12:
        raise ValueError("Objective weights must sum to one")
    if (any(type(v) is not int for v in (phase_start_ns, now_ns, deadline_ns))
            or not phase_start_ns <= now_ns <= deadline_ns or deadline_ns <= phase_start_ns):
        raise ValueError("Valid measurement times required")
    if not rates or set(refs) != set(rates):
        raise ValueError("Complete offered rates and latency references required")
    for rate in rates.values():
        number(rate, "offered rate", positive=True)
    for ref in refs.values():
        number(ref, "latency reference", positive=True)
    number(memory_budget_mib, "memory budget", positive=True)
    number(memory_safe_factor, "memory safety factor", positive=True)
    if memory_safe_factor < 1:
        raise ValueError("Memory safety factor must be >=1")
    number(energy_reference_watts, "energy reference", positive=True)
    if observed_peak_memory_mib is not None:
        number(observed_peak_memory_mib, "observed historical memory peak")
    if observed_board_energy_joules is not None:
        number(observed_board_energy_joules, "observed historical board energy")
    def validate_identity(row, expected_id=None):
        if (not isinstance(row.get('request_id'), str) or not row['request_id']
                or expected_id is not None and row['request_id'] != expected_id
                or type(row.get('arrival_ns')) is not int
                or not phase_start_ns <= row['arrival_ns'] < deadline_ns):
            raise ValueError('Invalid or outside-measurement scoring cohort identity/arrival')
        if (row.get('task') not in rates or row.get('alias') not in aliases
                or row['task'] not in aliases[row['alias']]['supported_tasks']):
            raise ValueError('Unknown task or unsupported route in scoring cohort')
    history = {}
    for row in completed:
        validate_identity(row)
        rid = row["request_id"]
        finish = row.get("completion_ns", row.get("completion_time_ns"))
        if rid in history or type(finish) is not int or not row["arrival_ns"] <= finish <= now_ns:
            raise ValueError("Duplicate/invalid/future completion feedback")
        history[rid] = {**row, "finish_ns": finish}
    cohort = forecast["cohort"]
    for rid, row in cohort.items():
        validate_identity(row, rid)
    if set(history) & set(cohort):
        raise ValueError("Completed/pending identity overlap")
    expected = set(history) | set(cohort)
    if set(forecast["finishes"]) - set(cohort):
        raise ValueError("Foreign forecast completion")
    for rid, row in forecast['finishes'].items():
        validate_identity(row, rid)
        source = cohort[rid]
        if any(row.get(key) != source.get(key) for key in ('request_id', 'task', 'alias', 'pool_id', 'arrival_ns')):
            raise ValueError('Forecast completion public identity differs from its cohort')
        finish = row.get('finish_ns')
        if type(finish) is not int or finish < max(now_ns, row['arrival_ns']):
            raise ValueError('Forecast completion requires integer time at or after visible now/arrival')
    incomplete = sorted(set(cohort) - set(forecast["finishes"]))
    rows = [*history.values(), *forecast["finishes"].values()]
    all_rows = [*history.values(), *cohort.values()]
    total = sum(rates.values())
    per_task, reasons, quality_totals = {}, [], {}
    Q, L, P = 0., 0., 0.
    for task, rate in rates.items():
        offered = [r for r in all_rows if r["task"] == task]
        done = [r for r in rows if r["task"] == task]
        alpha = rate / total
        if not offered:
            Q = L = P = None
            reasons.append("NO_COHORT_OBSERVATION_FOR_TASK:" + task)
            continue
        quality = [_expected_quality_totals(
            quality_loss.get(aliases[r['alias']]['family'], {}).get(task), task, r.get('pool_id'))
            for r in offered]
        if any(v is None for v in quality):
            Q = None
            reasons.append("UNKNOWN_AGGREGATE_QUALITY:" + task)
            quality_totals[task] = None
        else:
            numerator = sum(v[0] for v in quality)
            denominator = sum(v[1] for v in quality)
            quality_totals[task] = dict(numerator=numerator, denominator=denominator,
                                       requests=len(quality), basis='sum_expected_units_before_division')
            if Q is not None:
                Q += alpha * numerator / denominator
        if len(done) != len(offered):
            L = P = None
            reasons.append("DRAIN_CENSORED_LATENCY:" + task)
            continue
        latencies = [(r["finish_ns"] - r["arrival_ns"]) / 1e6 for r in done]
        if any(not math.isfinite(v) or v < 0 for v in latencies):
            raise ValueError("Invalid predicted latency")
        mean, tail = statistics.mean(latencies), percentile(latencies)
        per_task[task] = {"request_count": len(offered), "mean_ms": mean, "p95_ms": tail}
        if L is not None:
            L += alpha * mean / refs[task]
        if P is not None:
            P += alpha * tail / refs[task]
    T = 1 - sum(r["finish_ns"] <= deadline_ns for r in rows) / len(expected) if expected else None
    future_peak = forecast.get("peak_memory_mib")
    M = None
    if future_peak is None or observed_peak_memory_mib is None:
        reasons.append("UNKNOWN_WHOLE_PHASE_PEAK_MEMORY")
    else:
        number(future_peak, "predicted future peak memory")
        M = memory_safe_factor * max(observed_peak_memory_mib, future_peak) / memory_budget_mib
    horizon = (deadline_ns - phase_start_ns) / 1e9
    remaining_energy = forecast.get("board_energy_joules")
    E = None
    if remaining_energy is None:
        reasons.append("UNKNOWN_REMAINING_BOARD_ENERGY")
    else:
        number(remaining_energy, "predicted remaining energy")
        E = (remaining_energy + (observed_board_energy_joules or 0.)) / (energy_reference_watts * horizon)
    terms = dict(Q=Q, T=T, L=L, P=P, M=M, E=E)
    comparable = all(value is not None for value in terms.values())
    # Unobserved past energy is identical and additive in both branches, so
    # comparison is valid; an absolute U must still remain unknown.
    absolute = comparable and observed_board_energy_joules is not None
    metric = components(per_task, rates, refs, T, M, weights, E=E, Q=Q) if L is not None and P is not None else None
    cost = sum(weights[k] * v for k, v in terms.items()) if comparable else None
    return {"terms": terms, "cost": cost, "full_U": -cost if absolute else None,
            "six_term_comparison_complete": comparable, "absolute_utility_complete": absolute,
            "cohort_size": len(expected), "cohort_ids": sorted(expected),
            "task_counts": {t: sum(r["task"] == t for r in all_rows) for t in rates},
            "per_task": per_task, "unknown_reasons": reasons, "incomplete_ids": incomplete,
            "safe_memory_mib": memory_safe_factor * max(observed_peak_memory_mib, future_peak) if M is not None else None,
            "memory_safe_factor": memory_safe_factor,
            "memory_budget_satisfied": M <= 1 if M is not None else None,
            "E_scope": "remaining window plus observed past, if available; missing common past omitted from comparison only",
            "quality_scope": "frozen training expected numerator/denominator per offered request; summed by task before division; text/image scalar losses use denominator one; no future outputs",
            "quality_totals_by_task": quality_totals,
            "metric_components": metric, "physical_six_term_improvement_certified": False}


def assess_prediction_pairs(rows):
    """Offline held-out delta/ranking diagnostics; never an online label input.

    observed_delta_U and predicted_delta_U both mean spill minus no-spill.
    This reports errors, NOT a confidence interval or a runtime error bound.
    """
    if not rows:
        raise ValueError("Held-out paired predictions required")
    seen, errors, correct = set(), [], 0
    for row in rows:
        if set(row) != {"id", "split", "predicted_delta_U", "observed_delta_U"}:
            raise ValueError("Explicit held-out paired delta fields required")
        if row["id"] in seen or row["split"] != "validation":
            raise ValueError("Independent unique validation pairs required")
        seen.add(row["id"])
        for key in ("predicted_delta_U", "observed_delta_U"):
            if type(row[key]) not in (int, float) or not math.isfinite(row[key]):
                raise ValueError("Finite paired deltas required")
        errors.append(row["observed_delta_U"] - row["predicted_delta_U"])
        correct += (row["predicted_delta_U"] > 0) == (row["observed_delta_U"] > 0)
    return {"count": len(rows), "delta_mean_absolute_error": statistics.mean(map(abs, errors)),
            "delta_bias_observed_minus_predicted": statistics.mean(errors),
            "positive_gain_classification_accuracy": correct / len(rows),
            "delta_absolute_error_p95": percentile([abs(v) for v in errors]),
            "confidence_bound_calibrated": False}
