"""Metrics for requests scheduled in the measurement cohort, with explicit drain semantics."""

from collections import Counter


def percentile(xs, p):
    if not xs:
        return None
    values = sorted(xs)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def mean(values):
    return sum(values) / len(values) if values else None


def annotate_records(records):
    for row in records:
        scheduled, submitted, completed = (
            row.get(k)
            for k in (
                "scheduled_time_ns",
                "actual_submit_time_ns",
                "completion_time_ns",
            )
        )
        row["schedule_delay_ms"] = (
            (submitted - scheduled) / 1000000.0 if submitted is not None else None
        )
        success = row["status"] == "completed"
        row["e2e_latency_ms"] = (completed - scheduled) / 1000000.0 if success else None
        row["runtime_latency_ms"] = (
            (completed - submitted) / 1000000.0 if success else None
        )


def summarize(records, config, window):
    annotate_records(records)
    duration = config.measurement_seconds
    end = window["end_time_ns"]
    tasks = set(config.workload_rps) | set(config.routing_rps)
    per_task = {}
    for task in sorted(tasks):
        rows = [r for r in records if r["task"] == task]
        completed = [r for r in rows if r["status"] == "completed"]
        failures = sum((r["status"] == "error" for r in rows))
        in_window = sum((r["completion_time_ns"] <= end for r in completed))
        counts = Counter((r["model"] for r in rows))
        models = sorted(set(config.routing_rps.get(task, {})) | set(counts))
        metrics = {
            "offered_requests": len(rows),
            "completed_requests": len(completed),
            "failed_requests": failures,
            "incomplete_requests": len(rows) - len(completed) - failures,
            "target_arrival_rate_rps": config.workload_rps.get(task, 0),
            "actual_offered_rate_rps": len(rows) / duration,
            "achieved_completion_throughput_rps": len(completed) / duration,
            "completed_within_window": in_window,
            "completed_during_drain": len(completed) - in_window,
            "within_window_completion_throughput_rps": in_window / duration,
            "target_routing_rps": config.routing_rps.get(task, {}),
            "actual_routed_request_count": {m: counts[m] for m in models},
            "actual_routing_fraction": {
                m: counts[m] / len(rows) if rows else None for m in models
            },
            "actual_routed_rate_rps": {m: counts[m] / duration for m in models},
        }
        for metric in (
            "e2e_latency_ms",
            "runtime_latency_ms",
            "worker_latency_ms",
            "schedule_delay_ms",
            "worker_queue_wait_ms",
        ):
            population = rows if metric == "schedule_delay_ms" else completed
            values = [r[metric] for r in population if r.get(metric) is not None]
            metrics[f"mean_{metric}"] = mean(values)
            metrics[f"p95_{metric}"] = percentile(values, 95)
            metrics[f"{metric}_observations"] = len(values)
            if metric == "e2e_latency_ms":
                metrics[f"p50_{metric}"] = percentile(values, 50)
                metrics[f"p99_{metric}"] = percentile(values, 99)
        per_task[task] = metrics
    completed = [r for r in records if r["status"] == "completed"]
    in_window = sum((r["completion_time_ns"] <= end for r in completed))
    elapsed = (
        window.get("drain_end_time_ns", window["end_time_ns"]) - window["start_time_ns"]
    ) / 1000000000.0
    system = {
        "offered_requests": len(records),
        "completed_requests": len(completed),
        "failed_requests": sum((r["status"] == "error" for r in records)),
        "incomplete_requests": sum(
            (r["status"] not in ("completed", "error") for r in records)
        ),
        "offered_total_rps": sum(config.workload_rps.values()),
        "actual_offered_total_rps": len(records) / duration,
        "achieved_total_throughput_rps": len(completed) / duration,
        "within_window_completion_throughput_rps": in_window / duration,
        "completed_within_window": in_window,
        "completed_during_drain": len(completed) - in_window,
        "completion_ratio": len(completed) / len(records) if records else None,
        "completion_throughput_including_drain_time_rps": (
            len(completed) / elapsed if elapsed else None
        ),
        "completed_request_weighted_mean_e2e_latency_ms": mean(
            [r["e2e_latency_ms"] for r in completed]
        ),
    }
    model_metrics = {}
    for model, count in config.deployment.items():
        if count == 0:
            continue
        rows = [r for r in records if r["model"] == model]
        received = [
            r
            for r in rows
            if r.get("submit_write_completed_time_ns") is not None
            or r.get("worker_response") is not None
        ]
        successful = [r for r in rows if r["status"] == "completed"]
        worker_times = [
            r["worker_latency_ms"]
            for r in successful
            if r.get("worker_latency_ms") is not None
        ]
        model_metrics[model] = {
            "instances": count,
            "routed_requests": len(rows),
            "received_requests": len(received),
            "completed_requests": len(successful),
            "received_count_definition": "successful host pipe writes or worker responses; not a separate worker ACK",
            "per_task_received_counts": dict(Counter((r["task"] for r in received))),
            "per_task_completed_counts": dict(Counter((r["task"] for r in successful))),
            "observed_worker_handler_time_ms": (
                sum(worker_times) if worker_times else None
            ),
            "worker_latency_observations": len(worker_times),
        }
    return (per_task, system, model_metrics)
