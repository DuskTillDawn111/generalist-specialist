"""Reconstruct submission backlog from existing request records, without inference."""

from collections import defaultdict
import math
import numpy as np

DEFAULT_THRESHOLDS = dict(
    completion_ratio_min=0.98,
    timeline_bins=20,
    tail_fraction=0.25,
    tail_slope_min_requests_per_second=0.02,
    tail_growth_min_requests=5.0,
    tail_increasing_fraction=0.6,
    tail_end_min_requests=5,
    warning_peak_backlog_requests=30.0,
    warning_peak_service_seconds=5.0,
    warning_queue_wait_ms=1000.0,
    warning_queue_episode_min_requests=10,
    warning_queue_episode_fraction=0.05,
    warning_drain_seconds=5.0,
)
WARNING_KEYS = (
    "warning_peak_backlog_requests",
    "warning_peak_service_seconds",
    "warning_queue_wait_ms",
    "warning_drain_seconds",
)


def scaled_thresholds(factor):
    return {
        k: v * factor if k in WARNING_KEYS else v
        for (k, v) in DEFAULT_THRESHOLDS.items()
    }


def timeline(records, start, end, observed_end, catalog, bins, arrival_field):
    """Exact event integration; simultaneous events are summed before measuring peaks.

    Completion means a successful request response in the recorded cohort. Late
    responses after drain timeout do not change the frozen completion status.
    Actual submission is host send-start, not a worker ACK. Unsubmitted incomplete
    requests are absent from submitted B(t), but retained in incomplete counts.
    """
    events = defaultdict(lambda: [0, 0.0])
    missing_times, missing_paths = ([], set())
    for r in records:
        arrival = r.get(arrival_field)
        if arrival is None:
            if r["status"] == "completed":
                missing_times.append(r["request_id"])
            continue
        if not start <= arrival <= observed_end:
            raise ValueError("Arrival outside observed cohort interval")
        s = catalog.get(r.get("model"))
        s0 = s.service_time_ms.get(r.get("task")) if s is not None else None
        valid = s0 is not None and math.isfinite(s0) and (s0 > 0)
        if not valid:
            missing_paths.add((r.get("task"), r.get("model")))
        demand = s0 / 1000.0 if valid else 0.0
        events[arrival][0] += 1
        events[arrival][1] += demand
        if r["status"] == "completed":
            finish = r.get("completion_time_ns")
            if finish is None:
                missing_times.append(r["request_id"])
                continue
            if not arrival <= finish <= observed_end:
                raise ValueError(
                    "Completion precedes submission or exceeds observation cutoff"
                )
            events[finish][0] -= 1
            events[finish][1] -= demand
    valid_b, valid_w = (not missing_times, not missing_times and (not missing_paths))
    ordered = sorted(events.items())
    b = 0
    w = 0.0
    peak_b = peak_w = measurement_peak_b = measurement_peak_w = 0.0
    for t, (db, dw) in ordered:
        b += db
        w += dw
        if b < 0 or w < -1e-06:
            raise ValueError("Negative reconstructed backlog")
        peak_b, peak_w = (max(peak_b, b), max(peak_w, w))
        if t <= end:
            measurement_peak_b, measurement_peak_w = (
                max(measurement_peak_b, b),
                max(measurement_peak_w, w),
            )
    final_b, final_w = (b, max(0.0, w))
    b, w, cursor, last = (0, 0.0, 0, start)
    summaries = []
    for i in range(bins):
        left = start + round((end - start) * i / bins)
        edge = start + round((end - start) * (i + 1) / bins)
        area_b = area_w = 0.0
        while cursor < len(ordered) and ordered[cursor][0] <= edge:
            t, (db, dw) = ordered[cursor]
            area_b += b * (t - last)
            area_w += w * (t - last)
            b += db
            w += dw
            last = t
            cursor += 1
        area_b += b * (edge - last)
        area_w += w * (edge - last)
        last = edge
        summaries.append(
            dict(
                offset_seconds=(edge - start) / 1000000000.0,
                mean_backlog_requests=area_b / (edge - left) if valid_b else None,
                end_backlog_requests=b if valid_b else None,
                mean_service_demand_backlog=(
                    max(0.0, area_w / (edge - left)) if valid_w else None
                ),
                end_service_demand_backlog=max(0.0, w) if valid_w else None,
            )
        )
    return dict(
        peak_backlog_requests=int(peak_b) if valid_b else None,
        end_backlog_requests=b if valid_b else None,
        final_backlog_after_drain=final_b if valid_b else None,
        peak_service_demand_backlog=peak_w if valid_w else None,
        end_service_demand_backlog=max(0.0, w) if valid_w else None,
        final_service_demand_backlog_after_drain=final_w if valid_w else None,
        measurement_peak_backlog_requests=int(measurement_peak_b) if valid_b else None,
        measurement_peak_service_demand_backlog=measurement_peak_w if valid_w else None,
        missing_timestamps=missing_times,
        missing_service_paths=[list(p) for p in sorted(missing_paths)],
        bins=summaries,
        arrival_timestamp_field=arrival_field,
    )


def reconstruct(data, catalog, thresholds):
    if set(thresholds) != set(DEFAULT_THRESHOLDS):
        raise ValueError("Explicit complete threshold configuration required")
    records = data["request_records"]
    window = data.get("measurement_window")
    if not window:
        raise ValueError("Cannot reconstruct without measurement window")
    start, end, stop = (
        window[k] for k in ("start_time_ns", "end_time_ns", "drain_end_time_ns")
    )
    if not start < end <= stop:
        raise ValueError("Invalid observation interval")
    submitted = timeline(
        records,
        start,
        end,
        stop,
        catalog,
        thresholds["timeline_bins"],
        "actual_submit_time_ns",
    )
    offered = timeline(
        records,
        start,
        end,
        stop,
        catalog,
        thresholds["timeline_bins"],
        "scheduled_time_ns",
    )
    n_tail = max(
        3, math.ceil(thresholds["timeline_bins"] * thresholds["tail_fraction"])
    )
    tail = submitted["bins"][-n_tail:]
    if any((b["mean_backlog_requests"] is None for b in tail)):
        slope = growth = increasing = None
    else:
        ts = np.array([b["offset_seconds"] for b in tail])
        ys = np.array([b["mean_backlog_requests"] for b in tail])
        slope = float(
            np.dot(ts - ts.mean(), ys - ys.mean()) / np.sum((ts - ts.mean()) ** 2)
        )
        growth = float(ys[-1] - ys[0])
        increasing = float(np.mean(np.diff(ys) > 0))
    queues = [
        r["worker_queue_wait_ms"]
        for r in records
        if r.get("worker_queue_wait_ms") is not None
    ]
    incomplete = sum((r["status"] not in ("completed", "error") for r in records))
    completed = sum((r["status"] == "completed" for r in records))
    errors = sum(
        (
            r["status"] == "error"
            for kind in (
                "request_records",
                "warmup_request_records",
                "priming_request_records",
            )
            for r in data.get(kind, [])
        )
    )
    ratio = completed / len(records) if records else None
    saved_ratio = (data.get("system_metrics") or {}).get("completion_ratio")
    if (
        saved_ratio is not None
        and ratio is not None
        and (not math.isclose(ratio, saved_ratio, abs_tol=1e-12))
    ):
        raise ValueError("Completion accounting disagrees with saved result")
    timed_out = any(
        ("drain timeout" in str(r.get("error", "")).lower() for r in records)
    )
    return dict(
        **{
            k: v
            for (k, v) in submitted.items()
            if k not in ("bins", "arrival_timestamp_field")
        },
        backlog_timeline=submitted["bins"],
        offered_backlog_diagnostic=offered,
        backlog_definition="B(t)=actual host send-start submissions minus successful recorded completions; not scheduled arrivals or worker ACKs.",
        service_demand_backlog_definition="Sum full standalone s0 of each submitted, not-yet-completed request; standalone service-seconds, not remaining GPU work.",
        peak_interval="entire observed cohort including bounded drain; measurement-only peaks also saved",
        end_definition="value at measurement window end; separate final values are after drain",
        tail_backlog_slope=slope,
        tail_backlog_slope_units="requests/second",
        tail_backlog_growth=growth,
        tail_increasing_fraction=increasing,
        tail_slope_definition="OLS slope of exact-time-integrated B bin means over final configured fraction; descriptive trend, not a performance predictor.",
        drain_duration=window.get("drain_seconds"),
        drain_timeout=timed_out,
        configured_drain_timeout=data.get("drain_timeout_seconds"),
        completion_ratio=ratio,
        incomplete_requests=incomplete,
        completed_requests=completed,
        offered_requests=len(records),
        submitted_requests=sum(
            (r.get("actual_submit_time_ns") is not None for r in records)
        ),
        worker_or_request_errors=errors,
        run_errors=data.get("errors", []),
        run_status=data.get("status"),
        unclean_shutdown=any(
            (
                v.get("forced")
                or v.get("exit_code") != 0
                or (not v.get("bye_received"))
                for v in data.get("shutdown", {}).values()
            )
        ),
        queue_wait_observations=len(queues),
        peak_queue_wait_ms=max(queues) if queues else None,
        queue_episode_definition="Warning requires both a configured absolute count and fraction of offered requests above the queue-wait threshold; it does not assert these observations are contiguous.",
        queue_wait_values_ms=queues,
        clock="same-run perf_counter_ns; no cross-clock timestamp subtraction",
    )


def classify(evidence, thresholds):
    e, t = (evidence, thresholds)
    hard, warning = ([], [])
    if e["incomplete_requests"] > 0:
        hard.append("incomplete_requests")
    if e["drain_timeout"]:
        hard.append("drain_timeout")
    if (
        e["completion_ratio"] is None
        or e["completion_ratio"] < t["completion_ratio_min"]
    ):
        hard.append("completion_ratio_missing_or_low")
    if e["worker_or_request_errors"] or e["run_errors"]:
        hard.append("worker_or_run_error")
    if e["unclean_shutdown"] or e["run_status"] not in (
        "completed",
        "completed_with_failures",
    ):
        hard.append("unclean_or_failed_run")
    if e["peak_backlog_requests"] is None:
        hard.append("insufficient_submission_timestamps")
    if (
        e["tail_backlog_slope"] is not None
        and e["tail_backlog_slope"] >= t["tail_slope_min_requests_per_second"]
        and (e["tail_backlog_growth"] >= t["tail_growth_min_requests"])
        and (e["tail_increasing_fraction"] >= t["tail_increasing_fraction"])
        and (e["end_backlog_requests"] >= t["tail_end_min_requests"])
    ):
        hard.append("sustained_positive_tail_growth_with_nontrivial_end_backlog")
    for key, threshold, reason in (
        (
            "peak_backlog_requests",
            "warning_peak_backlog_requests",
            "large_submitted_backlog_peak",
        ),
        (
            "peak_service_demand_backlog",
            "warning_peak_service_seconds",
            "large_service_demand_backlog_peak",
        ),
        ("drain_duration", "warning_drain_seconds", "large_but_completed_drain"),
    ):
        if e[key] is not None and e[key] >= t[threshold]:
            warning.append(reason)
    count = sum((v >= t["warning_queue_wait_ms"] for v in e["queue_wait_values_ms"]))
    episode_min = max(
        t["warning_queue_episode_min_requests"],
        math.ceil(t["warning_queue_episode_fraction"] * e["offered_requests"]),
    )
    if count >= episode_min:
        warning.append("large_queue_wait_episode")
    recovered = e["completed_requests"] == e["offered_requests"] and (
        not e["drain_timeout"]
    )
    if not recovered and (not hard):
        hard.append("not_fully_recovered")
    state = "unstable" if hard else "recoverable_congested" if warning else "steady"
    return dict(
        stability_state_v2=state,
        stability_reason_v2="; ".join(
            hard or warning or ["complete_without_congestion_warning"]
        ),
        hard_reasons=hard,
        congestion_warning_reasons=warning,
        queue_wait_episode_requests=count,
        queue_wait_episode_required_requests=episode_min,
        final_recovery_confirmed=recovered,
        thresholds=dict(t),
    )
