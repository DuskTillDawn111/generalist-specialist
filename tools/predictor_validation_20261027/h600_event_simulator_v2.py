"""Full remaining-horizon arrival, FIFO, admission and segment event replay.

Segment boundaries are prediction events, not new runtime preemption actions.
Only the existing nonpreemptive complete-request admission rule is simulated.
"""

from __future__ import annotations

from collections import deque
import copy
import math

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import validate_resources
from gsdeploy.runtime.admission import admission_available
from gsdeploy.forecasting.workload import number


class FullHorizonModel:
    def __init__(self, aliases, service_provider, power_provider=None, *, memory_provider=None,
                 resources=None, max_events=100000, max_requests=20000):
        if not aliases:
            raise ValueError("Resident aliases required")
        for value in (max_events, max_requests):
            if type(value) is not int or value < 1:
                raise ValueError("Positive integer forecast budgets required")
        self.aliases = copy.deepcopy(aliases)
        self.service_provider = service_provider
        self.power_provider = power_provider if power_provider is not None else service_provider
        self.memory_provider = memory_provider if memory_provider is not None else service_provider
        self.resources = validate_resources(resources, aliases) if resources is not None else None
        self.gated = resources is not None and resources["mode"] == "request_admission"
        self.max_events, self.max_requests = max_events, max_requests

    def forecast(self, queues, now_ns, deadline_ns, scenario="mean", *, future_arrivals=(),
                 route_future=None, admission_state=None, drain_deadline_ns=None,
                 require_energy=True, require_memory=False, trace_request_id=None):
        if (type(now_ns) is not int or type(deadline_ns) is not int or now_ns > deadline_ns
                or (drain_deadline_ns is not None and (type(drain_deadline_ns) is not int or drain_deadline_ns < deadline_ns))):
            raise ValueError("Explicit integer measurement/drain times required")
        if set(queues) - set(self.aliases):
            raise ValueError("Unknown queue alias")
        pending = {a: deque() for a in self.aliases}
        cohort, seen = {}, set()
        future = sorted(copy.deepcopy(list(future_arrivals)),
                        key=lambda r: (r["arrival_ns"], r["task"], r["request_id"]))
        if sum(map(len, queues.values())) + len(future) > self.max_requests:
            raise UnknownEvidence("FULL_COHORT_BUDGET_EXCEEDED_NO_TRUNCATION")

        def check(row, alias=None, *, is_future=False):
            allowed = {"request_id", "task", "pool_id", "arrival_ns", "alias", "head_age_ms",
                       "forecast_cursor", "observed_pending"}
            if set(row) - allowed:
                raise ValueError("Forecast requests accept timing/input/progress only, no labels or outputs")
            if (not isinstance(row.get("request_id"), str) or not row["request_id"]
                    or row["request_id"] in seen or type(row.get("arrival_ns")) is not int):
                raise ValueError("Invalid/duplicate request identity or arrival")
            if is_future:
                if not now_ns < row["arrival_ns"] < deadline_ns:
                    raise ValueError("Future arrivals must be inside the remaining measurement window")
            elif row["arrival_ns"] > now_ns:
                raise ValueError("Future request in observed queues")
            if alias is not None and (alias not in self.aliases or row["task"] not in self.aliases[alias]["supported_tasks"]):
                raise ValueError("Unsupported request route")
            if not isinstance(row.get("pool_id"), str) or not row["pool_id"]:
                raise ValueError("Public input identity required")
            number(row.get("head_age_ms", 0.), "observed handler age")
            if "observed_pending" in row and type(row["observed_pending"]) is not bool:
                raise ValueError("Explicit observed pending boolean required")
            seen.add(row["request_id"])

        for alias, rows in queues.items():
            for index, row in enumerate(copy.deepcopy(list(rows))):
                check(row, alias)
                if index and ("forecast_cursor" in row or row.get("head_age_ms", 0.) > 0):
                    raise ValueError("Only the executing FIFO head may have progress/age")
                row["alias"] = alias
                pending[alias].append(row)
                cohort[row["request_id"]] = row
        for row in future:
            check(row, is_future=True)
        if future and route_future is None:
            raise ValueError("An explicit continuation routing policy is required")

        permitted, waiting = {}, deque()
        if self.gated:
            if admission_state is None and any(r.get("observed_pending") for rows in pending.values() for r in rows):
                raise UnknownEvidence("OBSERVED_HOST_ADMISSION_STATE_REQUIRED")
            state = admission_state or {"active": {}, "waiting": []}
            if state.get("closed", False):
                raise UnknownEvidence("HOST_ADMISSION_CLOSED")
            permitted = copy.deepcopy(state["active"])
            waiting = deque(tuple(ticket) for ticket in state["waiting"])
            flat = {r["request_id"]: a for a, rows in pending.items() for r in rows}
            tickets = [rid for _, rid in waiting]
            if (len(set(tickets)) != len(tickets) or set(tickets) & set(permitted)
                    or len({a for a, _ in waiting}) != len(waiting)
                    or any(flat.get(rid) != a for rid, a in permitted.items())
                    or any(flat.get(rid) != a for a, rid in waiting)):
                raise ValueError("Admission snapshot inconsistent with queued requests/senders")
            for alias in self.aliases:
                active_aliases = [a for a in permitted.values()]
                # Removing one existing permit must make that permit admissible.
                if alias in active_aliases:
                    active_aliases.remove(alias)
                    if not admission_available(self.resources, active_aliases, alias):
                        raise ValueError("Observed admission exceeds configured capacity")
                rows = list(pending[alias])
                granted = [r["request_id"] in permitted for r in rows]
                if any(granted[i] and not granted[i-1] for i in range(1, len(granted))):
                    raise ValueError("Granted worker requests must form a FIFO prefix")
                ticket = next((rid for a, rid in waiting if a == alias), None)
                if ticket is not None and next((r["request_id"] for r in rows if r["request_id"] not in permitted), None) != ticket:
                    raise ValueError("Waiting sender ticket must be first unsubmitted request")
                if rows and ("forecast_cursor" in rows[0] or rows[0].get("head_age_ms", 0.) > 0) and rows[0]["request_id"] not in permitted:
                    raise ValueError("Executing progress requires an observed admission permit")

        active, progress, plans, starts, service_time, service_by_kind = {}, {}, {}, {}, {}, {}
        finishes, transitions, evidence = {}, [], []
        clock, energy, peak = 0., 0., 0.
        energy_known, memory_known = True, True
        arrival_index, events = 0, 0
        window_end = deadline_ns - now_ns
        drain_end = math.inf if drain_deadline_ns is None else drain_deadline_ns - now_ns

        def refill_tickets():
            """One sender per alias; strict FIFO has head-of-line blocking."""
            if not self.gated:
                return
            busy = {a for a, _ in waiting}
            offers = []
            for alias, rows in pending.items():
                if alias not in busy:
                    row = next((r for r in rows if r["request_id"] not in permitted), None)
                    if row:
                        offers.append((max(now_ns + round(clock), row["arrival_ns"]), alias, row["request_id"]))
            for _, alias, rid in sorted(offers):
                waiting.append((alias, rid))

        def grant():
            refill_tickets()
            while waiting:
                alias, rid = waiting[0]
                if not admission_available(self.resources, permitted.values(), alias):
                    break
                waiting.popleft()
                permitted[rid] = alias
                refill_tickets()

        def set_segment(alias, index, fraction):
            row = active[alias]
            segment = plans[alias]["segments"][index]
            row.update(profile_id=plans[alias]["profile_id"], segment_id=segment["id"],
                       segment_kind=segment["kind"], segment_duration_ms=segment["duration_ms"],
                       resource_demand=segment["demand"])
            progress[alias] = [index, fraction]

        def start_ready():
            if self.gated:
                grant()
            for alias, rows in pending.items():
                if alias in active or not rows or (self.gated and rows[0]["request_id"] not in permitted):
                    continue
                row = rows[0]
                number(row.get("head_age_ms", 0.), "observed handler age")
                active[alias] = row
                starts[row["request_id"]] = now_ns + round(clock)
                service_time[row["request_id"]] = 0.
                service_by_kind[row["request_id"]] = {}
                provider = self.service_provider
                if hasattr(provider, "plan"):
                    plans[alias] = provider.plan(row, self.aliases, scenario)
                    execution = plans[alias].get("execution", {})
                    if execution.get("batch_size", 1) != 1 or execution.get("weight_reuse", False):
                        raise UnknownEvidence("JOINT_BATCH_OR_MERGED_WEIGHT_EXECUTOR_REQUIRED")
                    cursor = row.get("forecast_cursor")
                    if cursor is None and row.get("head_age_ms", 0) > 0:
                        raise UnknownEvidence("OBSERVED_SEGMENT_PROGRESS_REQUIRED:" + alias)
                    index, fraction = 0, 1.
                    if cursor is not None:
                        if (set(cursor) != {"profile_id", "segment_index", "remaining_fraction", "observed_at_ns", "evidence"}
                                or cursor["profile_id"] != plans[alias]["profile_id"]
                                or type(cursor["observed_at_ns"]) is not int
                                or cursor["observed_at_ns"] != now_ns
                                or not isinstance(cursor["evidence"], str) or not cursor["evidence"]):
                            raise ValueError("Explicit current conditional segment progress evidence required")
                        index, fraction = cursor["segment_index"], cursor["remaining_fraction"]
                        if type(index) is not int or not 0 <= index < len(plans[alias]["segments"]):
                            raise ValueError("Invalid observed segment index")
                        number(fraction, "remaining segment fraction", positive=True)
                        if fraction > 1:
                            raise ValueError("Remaining segment fraction must be <=1")
                    set_segment(alias, index, fraction)
                else:
                    # Legacy support is explicit whole-handler progress, not
                    # fabricated kernel detail or measured GPU start state.
                    plans[alias] = {"profile_id": "legacy:" + alias,
                                    "segments": [{"id": "handler", "kind": "host", "duration_ms": 1.,
                                                  "demand": {"sm": 0., "dram": 0., "l2": 0.}}]}
                    set_segment(alias, 0, 1.)

        start_ready()
        # Initialize legacy conditional residuals after ALL heads are active.
        if not hasattr(self.service_provider, "plan"):
            for alias, row in active.items():
                age = number(row.get("head_age_ms", 0.), "observed handler age")
                if age:
                    service = self.service_provider.service(row, active, self.aliases, scenario)
                    residual = self.service_provider.residual(row, age, active, self.aliases, scenario)
                    if (not service.get("known") or not service.get("context_supported")
                            or not residual.get("known") or not residual.get("tail_supported")
                            or type(service.get("service_ms")) not in (int, float)
                            or type(residual.get("remaining_ms")) not in (int, float)
                            or not math.isfinite(service["service_ms"]) or service["service_ms"] <= 0
                            or not math.isfinite(residual["remaining_ms"]) or residual["remaining_ms"] <= 0):
                        raise UnknownEvidence("UNKNOWN_CONDITIONAL_RESIDUAL_TAIL:" + alias)
                    progress[alias][1] = residual["remaining_ms"] / service["service_ms"]
                    evidence.append({"alias": alias, "kind": "LEGACY_HOST_AGE_CONDITIONAL_PROXY", "evidence": residual})

        while active or arrival_index < len(future) or any(pending.values()) or clock < window_end:
            events += 1
            if events > self.max_events:
                raise UnknownEvidence("FULL_HORIZON_EVENT_BUDGET_EXCEEDED_NO_TRUNCATION")
            rates = {}
            for alias, row in active.items():
                value = self.service_provider.service(row, active, self.aliases, scenario)
                ms = value.get("service_ms")
                if (value.get("known") is not True or value.get("context_supported") is not True
                        or type(ms) not in (int, float) or not math.isfinite(ms) or ms <= 0):
                    raise UnknownEvidence("UNKNOWN_SHARED_SEGMENT_SERVICE:" + alias)
                rates[alias] = ms
            durations = {a: progress[a][1] * rates[a] * 1e6 for a in active}
            arrival_time = future[arrival_index]["arrival_ns"] - now_ns if arrival_index < len(future) else math.inf
            completion_times = {a: clock + duration for a, duration in durations.items()}
            boundary = min(min(completion_times.values(), default=math.inf), arrival_time,
                           window_end if clock < window_end else math.inf, drain_end)
            if not math.isfinite(boundary) or boundary < clock:
                raise UnknownEvidence("ADMISSION_DEADLOCK_OR_INVALID_EVENT")
            delta = boundary - clock
            power = self.power_provider.power(active, self.aliases, scenario) if clock < window_end else None
            if power is not None:
                watts = power.get("watts")
                if power.get("known") is not True or type(watts) not in (int, float) or not math.isfinite(watts) or watts < 0:
                    energy_known = False
                    if require_energy:
                        raise UnknownEvidence("UNKNOWN_WHOLE_BOARD_POWER_CONTEXT")
                else:
                    energy += watts * min(delta, window_end - clock) / 1e9
            method = getattr(self.memory_provider, "memory", None)
            memory = method(active, self.aliases, scenario) if method else {"known": False}
            memory_mib = memory.get("peak_memory_mib")
            if (memory.get("known") is not True or type(memory_mib) not in (int, float)
                    or not math.isfinite(memory_mib) or memory_mib < 0):
                memory_known = False
                if require_memory:
                    raise UnknownEvidence("UNKNOWN_WHOLE_BOARD_PEAK_MEMORY_CONTEXT")
            else:
                peak = max(peak, memory_mib)
            transition = {"time_ns": now_ns + round(clock), "until_ns": now_ns + round(boundary),
                          "active_requests": {a: r["request_id"] for a, r in active.items()},
                          "segments": {a: r["segment_id"] for a, r in active.items()}, "service_ms": rates,
                          "resource_demands": {a: r["resource_demand"] for a, r in active.items()},
                          "board_watts": power.get("watts") if power and power.get("known") else None,
                          "peak_memory_mib": memory_mib if memory.get("known") else None,
                          "admission_waiting": list(waiting), "repriced_aliases": sorted(active)}
            if trace_request_id is None or trace_request_id in transition["active_requests"].values():
                transitions.append(transition)
            done = []
            for alias, row in active.items():
                progress[alias][1] = max(0., progress[alias][1] - delta / (rates[alias] * 1e6))
                service_time[row["request_id"]] += delta / 1e6
                breakdown = service_by_kind[row["request_id"]]
                kind = row["segment_kind"]
                breakdown[kind] = breakdown.get(kind, 0.) + delta / 1e6
                # Preserve the selected completion event even if subtracting
                # large rounded clocks leaves a sub-ULP positive residual.
                if completion_times[alias] <= boundary or progress[alias][1] <= 1e-10:
                    done.append(alias)
            clock = boundary
            for alias in done:
                index = progress[alias][0] + 1
                if index < len(plans[alias]["segments"]):
                    set_segment(alias, index, 1.)
                else:
                    row = pending[alias].popleft()
                    rid = row["request_id"]
                    finishes[rid] = {**row, "finish_ns": now_ns + round(clock), "start_ns": starts[rid],
                                     "predicted_remaining_service_ms": service_time[rid]}
                    finishes[rid]["remaining_service_by_kind_ms"] = service_by_kind[rid]
                    active.pop(alias)
                    progress.pop(alias)
                    permitted.pop(rid, None)
            transition["completed_aliases"] = sorted(a for a in done if a not in active)
            while arrival_index < len(future) and future[arrival_index]["arrival_ns"] - now_ns <= clock:
                row = future[arrival_index]
                alias = route_future(copy.deepcopy(row), {a: list(q) for a, q in pending.items()}, now_ns + round(clock))
                if alias not in self.aliases or row["task"] not in self.aliases[alias]["supported_tasks"]:
                    raise ValueError("Unsupported continuation route")
                row["alias"] = alias
                pending[alias].append(row)
                cohort[row["request_id"]] = row
                arrival_index += 1
            if clock >= drain_end:
                break
            start_ready()
        incomplete = sorted(set(cohort) - set(finishes))
        return {"finishes": finishes, "cohort": cohort, "incomplete_ids": incomplete,
                "board_energy_joules": energy if energy_known else None, "energy_known": energy_known,
                "peak_memory_mib": peak if memory_known else None, "memory_known": memory_known,
                "transitions": transitions, "initial_residual_evidence": evidence,
                "transition_count": events,
                "trace_scope": "full" if trace_request_id is None else "current-request active intervals only; all workload events simulated",
                "drain_finish_ns": now_ns + round(clock), "future_arrival_count": len(future),
                "scope": "Full remaining arrival horizon, ordered segment progress and nonpreemptive FIFO execution",
                "admission_scope": "observed initial FIFO tickets; deterministic alias ordering for simultaneous predicted sender offers",
                "physical_six_term_improvement_certified": False}
