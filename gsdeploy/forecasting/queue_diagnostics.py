"""Queue explanations, not a new stability proof or an extra utility penalty.

Work is remaining solo-profile milliseconds; demand-weighted work has units
normalized-resource*ms, not bytes, instructions, or measured GPU throughput.
Missing resource demand remains unknown. No change to FIFO or rho acceptance.
"""

from __future__ import annotations

import copy
import heapq

from ..forecast_errors import UnknownEvidence
from ..runtime.admission import admission_available
from .workload import number


def remaining_work(plan, cursor=None):
    """Sum solo baseline work, preserving observed in-flight progress."""
    if cursor is None:
        index, fraction = 0, 1.
    else:
        index, fraction = cursor["segment_index"], cursor["remaining_fraction"]
        if type(index) is not int or not 0 <= index < len(plan["segments"]):
            raise ValueError("Invalid work cursor")
        number(fraction, "remaining fraction")
        if fraction > 1:
            raise ValueError("Remaining fraction exceeds one")
    solo_ms, gpu_ms = 0., 0.
    resource = dict.fromkeys(("sm", "dram", "l2"), 0.)
    for i, segment in enumerate(plan["segments"][index:], start=index):
        ms = number(segment["duration_ms"], "solo duration", positive=True) * (fraction if i == index else 1.)
        solo_ms += ms
        if segment["kind"] != "gpu":
            continue
        gpu_ms += ms
        for name in resource:
            value = segment["demand"].get(name)
            if value is None:
                resource[name] = None
            elif resource[name] is not None:
                value = number(value, "normalized resource demand")
                if value > 1:
                    raise ValueError("Normalized resource demand exceeds one")
                resource[name] += ms * value
    return {"solo_work_ms": solo_ms, "gpu_solo_work_ms": gpu_ms,
            "normalized_resource_work_ms": resource}


def workload_envelopes(rows, costs, scales_ns=(100_000_000, 1_000_000_000, 10_000_000_000)):
    """Exact maximum offered work in half-open (t-window,t] trace windows.

    Empirical scenario envelopes only: no deterministic arrival/service bound.
    Unknown input cost invalidates the work envelope, not the count envelope.
    """
    ordered = sorted(rows, key=lambda r: (r["arrival_ns"], r["request_id"]))
    result = []
    known = all(costs.get(r["request_id"]) is not None for r in ordered)
    for scale in scales_ns:
        if type(scale) is not int or scale <= 0:
            raise ValueError("Positive integer envelope window required")
        left, max_count, max_work, work = 0, 0, 0., 0.
        for right, row in enumerate(ordered):
            if known:
                work += number(costs[row["request_id"]], "offered solo work")
            while ordered[left]["arrival_ns"] <= row["arrival_ns"] - scale:
                if known:
                    work -= costs[ordered[left]["request_id"]]
                left += 1
            max_count = max(max_count, right - left + 1)
            max_work = max(max_work, work)
        result.append({"window_ns": scale, "max_arrival_count": max_count,
                       "max_offered_solo_work_ms": max_work if known else None})
    return result


class QueueDiagnostics:
    def __init__(self, aliases, provider, scenario, resources, now_ns):
        self.aliases, self.provider, self.scenario = aliases, provider, scenario
        self.resources, self.now_ns = resources, now_ns
        self.cache, self.costs = {}, {}
        self.total_work = 0.
        self.resource_work = dict.fromkeys(("sm", "dram", "l2"), 0.)
        self.waiting_ids, self.waiting_arrivals = set(), []
        self.max_waiting, self.max_oldest_ms, self.max_work_ms = 0, 0., 0.
        self.max_resource_work = dict.fromkeys(("sm", "dram", "l2"), 0.)
        self.hol_ns, self.unknown, self.episodes = 0., set(), []
        self.episode_start = None

    def register(self, row):
        """O(segments) once per profile; O(1) per admitted request."""
        rid = row["request_id"]
        self.waiting_ids.add(rid)
        heapq.heappush(self.waiting_arrivals, (row["arrival_ns"], rid))
        key = row["alias"], row["task"], row["pool_id"]
        try:
            if key not in self.cache:
                plan = self.provider.plan(row, self.aliases, self.scenario)
                self.cache[key] = (plan, remaining_work(plan))
            plan, full = self.cache[key]
            self.costs[rid] = full["solo_work_ms"]
            residual = remaining_work(plan, row["forecast_cursor"]) if "forecast_cursor" in row else full
            self.total_work += residual["solo_work_ms"]
            for name, value in residual["normalized_resource_work_ms"].items():
                if value is None:
                    self.resource_work[name] = None
                elif self.resource_work[name] is not None:
                    self.resource_work[name] += value
        except (UnknownEvidence, AttributeError) as exc:
            self.unknown.add(str(exc) if isinstance(exc, UnknownEvidence) else "NO_ORDERED_SOLO_PROFILE")
            self.costs[rid] = None

    def mark_started(self, rows):
        for row in rows:
            self.waiting_ids.discard(row["request_id"])

    def observe(self, time_ns, delta_ns, pending, active, plans, progress, permitted, waiting, rates=None, batch_members=None):
        # Queue lengths and the oldest waiting head are sufficient: no walk of
        # all pending requests at each completion. FIFO preserves arrival order.
        batch_members = batch_members or {}
        count = sum(len(rows) for rows in pending.values()) - sum(len(batch_members.get(a, [r])) for a, r in active.items())
        while self.waiting_arrivals and self.waiting_arrivals[0][1] not in self.waiting_ids:
            heapq.heappop(self.waiting_arrivals)
        if self.waiting_arrivals:
            self.max_oldest_ms = max(self.max_oldest_ms, (time_ns + delta_ns - self.waiting_arrivals[0][0]) / 1e6)
        self.max_waiting = max(self.max_waiting, count)
        if not self.unknown:
            self.max_work_ms = max(self.max_work_ms, self.total_work)
        for name, value in self.resource_work.items():
            if self.unknown or value is None:
                self.max_resource_work[name] = None
            elif self.max_resource_work[name] is not None:
                self.max_resource_work[name] = max(self.max_resource_work[name], value)
        # Consume exactly the active segments' solo work, independently of the
        # elapsed shared-service time. Stalled work consumes zero.
        for alias, rate in (rates or {}).items():
            segment = plans[alias]["segments"][progress[alias][0]]
            completed_fraction = min(progress[alias][1], delta_ns / (rate * 1e6))
            ms = completed_fraction * segment["duration_ms"]
            self.total_work = max(0., self.total_work - ms)
            if segment["kind"] == "gpu":
                for name, demand in segment["demand"].items():
                    if demand is not None and self.resource_work[name] is not None:
                        self.resource_work[name] = max(0., self.resource_work[name] - ms * demand)
        if count and self.episode_start is None:
            self.episode_start = time_ns
        elif not count and self.episode_start is not None:
            self.episodes.append({"backlog_start_ns": self.episode_start, "queue_clear_ns": time_ns,
                                  "recovery_ms": (time_ns - self.episode_start) / 1e6})
            self.episode_start = None
        if waiting and self.resources is not None:
            head = waiting[0][0]
            if (not admission_available(self.resources, permitted.values(), head)
                    and any(admission_available(self.resources, permitted.values(), a) for a, _ in list(waiting)[1:])):
                self.hol_ns += delta_ns

    def result(self, cohort):
        return {"max_waiting_requests": self.max_waiting, "max_oldest_waiting_age_ms": self.max_oldest_ms,
                "max_remaining_solo_work_ms": self.max_work_ms if not self.unknown else None,
                "max_normalized_resource_work_ms": dict(self.max_resource_work),
                "strict_fifo_hol_blocked_ms": self.hol_ns / 1e6,
                "queue_recovery_episodes": copy.deepcopy(self.episodes),
                "unrecovered_backlog_since_ns": self.episode_start,
                "initial_backlog_left_censored": bool(self.episodes and self.episodes[0]["backlog_start_ns"] == self.now_ns
                                                       or self.episode_start == self.now_ns),
                "offered_work_envelopes": workload_envelopes(list(cohort.values()), self.costs),
                "unknown_reasons": sorted(self.unknown),
                "scope": "Scenario diagnostics; solo work is not contention-adjusted completion time",
                "stability_rule_changed": False, "calibrated_long_term_risk": False}
