"""Full remaining-horizon arrival, FIFO, admission and segment event replay.

Segment boundaries are prediction events, not new runtime preemption actions.
Only the existing nonpreemptive complete-request admission rule is simulated.
"""

from __future__ import annotations

from collections import deque
import copy
import math

from ..forecast_errors import UnknownEvidence
from ..resources import validate_resources
from ..runtime.admission import admission_available
from ..runtime.batching import select_batch
from .workload import number
from .queue_diagnostics import QueueDiagnostics


class FullHorizonModel:
    def __init__(self, aliases, service_provider, power_provider=None, *, memory_provider=None,
                 resources=None, lifecycle_provider=None, batch_configurations=None,
                 max_events=100000, max_requests=20000):
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
        self.lifecycle_provider = lifecycle_provider
        if lifecycle_provider is not None:
            if (not getattr(service_provider, "execution_context_sha256", None)
                    or lifecycle_provider.execution_context_sha256 != service_provider.execution_context_sha256):
                raise ValueError("Service and lifecycle execution identities differ")
        self.gated = resources is not None and resources["mode"] == "request_admission"
        self.batch_configurations = dict(batch_configurations or {})
        if set(self.batch_configurations) - set(aliases):
            raise ValueError("Unknown batch owner")
        if self.batch_configurations and self.gated:
            raise ValueError("Per-request admission with shared batches requires a matching runtime gate")
        if self.batch_configurations and not hasattr(service_provider, "plan_batch"):
            raise ValueError("Batch formation requires a matching batch execution profile provider")
        self.max_events, self.max_requests = max_events, max_requests

    def forecast(self, queues, now_ns, deadline_ns, scenario="mean", *, future_arrivals=(),
                 route_future=None, admission_state=None, drain_deadline_ns=None,
                 require_energy=True, require_memory=False, trace_request_id=None,
                 handoff_state=()):
        if (type(now_ns) is not int or type(deadline_ns) is not int or now_ns > deadline_ns
                or (drain_deadline_ns is not None and (type(drain_deadline_ns) is not int or drain_deadline_ns < deadline_ns))):
            raise ValueError("Explicit integer measurement/drain times required")
        if set(queues) - set(self.aliases):
            raise ValueError("Unknown queue alias")
        pending = {a: deque() for a in self.aliases}
        cohort, seen = {}, set()
        future = sorted(copy.deepcopy(list(future_arrivals)),
                        key=lambda r: (r["arrival_ns"], r["task"], r["request_id"]))
        handoffs = copy.deepcopy(list(handoff_state))
        if sum(map(len, queues.values())) + len(future) + len(handoffs) > self.max_requests:
            raise UnknownEvidence("FULL_COHORT_BUDGET_EXCEEDED_NO_TRUNCATION")

        def check(row, alias=None, *, is_future=False):
            allowed = {"request_id", "task", "pool_id", "arrival_ns", "alias", "head_age_ms",
                       "forecast_cursor", "observed_pending", "batch_ready_ns", "batch_ready_evidence"}
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
            if "batch_ready_ns" in row or "batch_ready_evidence" in row:
                if (is_future or alias not in self.batch_configurations
                        or type(row.get("batch_ready_ns")) is not int
                        or not row["arrival_ns"] <= row["batch_ready_ns"] <= now_ns
                        or not isinstance(row.get("batch_ready_evidence"), str) or not row["batch_ready_evidence"]):
                    raise ValueError("Current observed batch readiness evidence required")
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
        if handoffs and self.lifecycle_provider is None:
            raise UnknownEvidence("INITIAL_HANDOFF_REQUIRES_CONDITIONAL_LIFECYCLE_EVIDENCE")
        for handoff in handoffs:
            row = {k: handoff[k] for k in ("request_id", "task", "pool_id", "arrival_ns", "alias")}
            check(row, row["alias"])
            if row["alias"] in self.batch_configurations:
                raise UnknownEvidence("INITIAL_BATCH_HANDOFF_MEMBERSHIP_REQUIRED")
            if (handoff.get("observed_at_ns") != now_ns or not handoff.get("evidence")):
                raise ValueError("Current coherent handoff observation evidence required")
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
            flat.update({h["request_id"]: h["alias"] for h in handoffs})
            if self.lifecycle_provider is not None and set(permitted) - set(flat):
                raise UnknownEvidence("INITIAL_POST_HANDLER_HANDOFF_STATE_REQUIRED")
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
        batch_members, batch_wakes, batch_audit = {}, {}, []
        preparing, prepared, preparation_events = {}, {}, []
        for alias, rows in pending.items():
            if alias not in self.batch_configurations:
                continue
            for row in rows:
                if "batch_ready_ns" in row:
                    prepared[row["request_id"]] = row["batch_ready_ns"]
                elif row["arrival_ns"] < now_ns:
                    raise UnknownEvidence("OBSERVED_BATCH_PREPARATION_STATE_REQUIRED")
        finishes, transitions, evidence = {}, [], []
        receipts, releases, ready_times = {}, {}, {}
        lifecycle_events = []
        clock, energy, peak = 0., 0., 0.
        energy_known, memory_known = True, True
        arrival_index, events = 0, 0
        diagnostics = QueueDiagnostics(self.aliases, self.service_provider, scenario, self.resources, now_ns)
        if self.batch_configurations:
            diagnostics.unknown.add("BATCH_REMAINING_SOLO_WORK_IS_NOT_ADDITIVE_SERVICE")
        for rows in pending.values():
            for row in rows:
                diagnostics.register(row)
        window_end = deadline_ns - now_ns
        drain_end = math.inf if drain_deadline_ns is None else drain_deadline_ns - now_ns

        for handoff in handoffs:
            row = {k: handoff[k] for k in ("request_id", "task", "pool_id", "arrival_ns", "alias")}
            alias, rid = row["alias"], row["request_id"]
            observed = {k: handoff[k] for k in ("handler_end_ns", "parent_receipt_ns", "permit_release_ns", "worker_next_ready_ns")}
            residual = self.lifecycle_provider.residual_handoff(row, self.aliases, scenario, now_ns, observed,
                boundary_observation_status=handoff.get("boundary_observation_status"))
            evidence.append({"request_id": rid, "kind": "CONDITIONAL_POST_HANDLER_RESIDUAL", "evidence": residual})
            if observed["parent_receipt_ns"] is None:
                cohort[rid] = row
                receipt_at = residual["parent_receipt_remaining_ms"] * 1e6
                receipts[rid] = {"at": receipt_at, "record": {**row, "handler_end_ns": observed["handler_end_ns"],
                    "start_ns": handoff.get("start_ns"), "finish_ns": now_ns + round(receipt_at),
                    "predicted_remaining_service_ms": 0., "remaining_service_by_kind_ms": {}}}
            if observed["worker_next_ready_ns"] is None:
                if alias in ready_times:
                    raise ValueError("Multiple unready handoffs for one owner")
                ready_times[alias] = residual["worker_next_ready_remaining_ms"] * 1e6
            if self.gated:
                held = rid in permitted
                if held != (observed["permit_release_ns"] is None):
                    raise ValueError("Handoff release and observed gate state disagree")
                if held:
                    releases[rid] = residual["permit_release_remaining_ms"] * 1e6

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
                if (alias in active or alias in preparing or not rows or alias in ready_times
                        or (self.gated and rows[0]["request_id"] not in permitted)):
                    continue
                row = rows[0]
                selected_members, batch_plan = None, None
                if alias in self.batch_configurations:
                    if any("forecast_cursor" in r or r.get("head_age_ms", 0) > 0 for r in rows):
                        raise UnknownEvidence("OBSERVED_INFLIGHT_BATCH_STATE_REQUIRED")
                    # Match the single-owner runtime: dispatch a ready batch
                    # first; otherwise prepare the next FIFO request. Owner
                    # preparation cannot overlap its own generate/cleanup.
                    while True:
                        ready_rows = []
                        for candidate in rows:
                            if candidate["request_id"] not in prepared:
                                break
                            ready_rows.append({**self.service_provider.scheduling_row(candidate),
                                               "ready_ns": prepared[candidate["request_id"]]})
                        selection = select_batch(ready_rows, now_ns + round(clock), self.batch_configurations[alias])
                        if selection["status"] in {"READY", "UNSUPPORTED"}:
                            break
                        if len(ready_rows) == len(rows):
                            break
                        candidate = rows[len(ready_rows)]
                        metadata = self.service_provider.scheduling_row(candidate)
                        delay = metadata["input_ready_delay_ns"]
                        at = clock + delay
                        preparation_events.append({"kind": "prepare_start", "alias": alias,
                                                   "request_id": candidate["request_id"], "time_ns": now_ns + round(clock)})
                        if delay:
                            preparing[alias] = {"row": candidate, "at": at, "start": clock}
                            batch_wakes.pop(alias, None)
                            break
                        prepared[candidate["request_id"]] = now_ns + round(at)
                        preparation_events.append({"kind": "prepare_ready", "alias": alias,
                                                   "request_id": candidate["request_id"], "time_ns": now_ns + round(at)})
                    if alias in preparing:
                        continue
                    if selection["status"] == "UNSUPPORTED":
                        raise UnknownEvidence(selection["reason"])
                    if selection["status"] != "READY":
                        if selection["next_event_ns"] is None:
                            raise UnknownEvidence("BATCH_PREPARATION_NO_NEXT_EVENT")
                        batch_wakes[alias] = selection["next_event_ns"] - now_ns
                        continue
                    batch_wakes.pop(alias, None)
                    selected_members = list(rows)[:selection["batch_size"]]
                    row, batch_plan = self.service_provider.plan_batch(selected_members, selection, self.aliases, scenario)
                    batch_members[alias] = selected_members
                    batch_audit.append({**selection, "alias": alias, "dispatch_ns": now_ns + round(clock)})
                number(row.get("head_age_ms", 0.), "observed handler age")
                active[alias] = row
                diagnostics.mark_started(selected_members if selected_members is not None else [row])
                starts[row["request_id"]] = now_ns + round(clock)
                service_time[row["request_id"]] = 0.
                service_by_kind[row["request_id"]] = {}
                provider = self.service_provider
                if hasattr(provider, "plan"):
                    plans[alias] = batch_plan if batch_plan is not None else provider.plan(row, self.aliases, scenario)
                    execution = plans[alias].get("execution", {})
                    if (selected_members is None
                            and (execution.get("batch_size", 1) != 1 or execution.get("weight_reuse", False))):
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

        while (active or preparing or receipts or releases or ready_times or arrival_index < len(future)
               or any(pending.values()) or clock < window_end):
            events += 1
            if events > self.max_events:
                raise UnknownEvidence("FULL_HORIZON_EVENT_BUDGET_EXCEEDED_NO_TRUNCATION")
            rates = {}
            for alias, row in active.items():
                value = self.service_provider.service(row, active, self.aliases, scenario)
                ms = value.get("service_ms")
                if (value.get("known") is True and value.get("context_supported") is True
                        and value.get("stalled") is True and ms is None):
                    rates[alias] = math.inf
                    continue
                if (value.get("known") is not True or value.get("context_supported") is not True
                        or type(ms) not in (int, float) or not math.isfinite(ms) or ms <= 0):
                    raise UnknownEvidence("UNKNOWN_SHARED_SEGMENT_SERVICE:" + alias + ":" + value.get("reason", "UNSUPPORTED_SERVICE"))
                rates[alias] = ms
            durations = {a: progress[a][1] * rates[a] * 1e6 for a in active}
            arrival_time = future[arrival_index]["arrival_ns"] - now_ns if arrival_index < len(future) else math.inf
            boundary = min(clock + min(durations.values(), default=math.inf), arrival_time,
                           min((v["at"] for v in receipts.values()), default=math.inf),
                           min(releases.values(), default=math.inf),
                           min(ready_times.values(), default=math.inf),
                           min(batch_wakes.values(), default=math.inf),
                           min((p["at"] for p in preparing.values()), default=math.inf),
                           window_end if clock < window_end else math.inf, drain_end)
            if not math.isfinite(boundary) or boundary < clock:
                raise UnknownEvidence("ADMISSION_DEADLOCK_OR_INVALID_EVENT")
            delta = boundary - clock
            diagnostics.observe(now_ns + round(clock), delta, pending, active, plans, progress, permitted, waiting, rates, batch_members)
            if clock >= window_end:
                power = None
            elif preparing:
                method = getattr(self.power_provider, "power_preparing", None)
                power = method(preparing, active, self.aliases, scenario) if method else {"known": False}
            else:
                power = self.power_provider.power(active, self.aliases, scenario)
            if power is not None:
                watts = power.get("watts")
                if power.get("known") is not True or type(watts) not in (int, float) or not math.isfinite(watts) or watts < 0:
                    energy_known = False
                    if require_energy:
                        raise UnknownEvidence("UNKNOWN_WHOLE_BOARD_POWER_CONTEXT")
                else:
                    energy += watts * min(delta, window_end - clock) / 1e9
            if preparing:
                method = getattr(self.memory_provider, "memory_preparing", None)
                memory = method(preparing, active, self.aliases, scenario) if method else {"known": False}
            else:
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
                          "active_batch_members": {a: [r["request_id"] for r in members] for a, members in batch_members.items()},
                          "preparing_requests": {a: p["row"]["request_id"] for a, p in preparing.items()},
                          "segments": {a: r["segment_id"] for a, r in active.items()},
                          "service_ms": {a: ms if math.isfinite(ms) else None for a, ms in rates.items()},
                          "stalled_aliases": sorted(a for a, ms in rates.items() if not math.isfinite(ms)),
                          "resource_demands": {a: r["resource_demand"] for a, r in active.items()},
                          "board_watts": power.get("watts") if power and power.get("known") else None,
                          "peak_memory_mib": memory_mib if memory.get("known") else None,
                          "admission_waiting": list(waiting), "repriced_aliases": sorted(active)}
            if (trace_request_id is None or trace_request_id in transition["active_requests"].values()
                    or any(trace_request_id in ids for ids in transition["active_batch_members"].values())):
                transitions.append(transition)
            done = []
            for alias, row in active.items():
                progress[alias][1] = max(0., progress[alias][1] - delta / (rates[alias] * 1e6))
                service_time[row["request_id"]] += delta / 1e6
                breakdown = service_by_kind[row["request_id"]]
                kind = row["segment_kind"]
                breakdown[kind] = breakdown.get(kind, 0.) + delta / 1e6
                if math.isfinite(rates[alias]) and boundary >= clock + durations[alias]:
                    done.append(alias)
            clock = boundary
            for alias in done:
                index = progress[alias][0] + 1
                if index < len(plans[alias]["segments"]):
                    set_segment(alias, index, 1.)
                else:
                    row = active[alias]
                    members = batch_members.pop(alias, None)
                    if members is None:
                        members = [pending[alias].popleft()]
                    else:
                        for member in members:
                            if pending[alias].popleft()["request_id"] != member["request_id"]:
                                raise ValueError("FIFO batch membership changed during execution")
                            prepared.pop(member["request_id"], None)
                    rid = row["request_id"]
                    end_ns = now_ns + round(clock)
                    if self.lifecycle_provider is None:
                        delays = dict(parent_receipt_ms=0., permit_release_ms=0., worker_next_ready_ms=0.)
                    else:
                        delays = self.lifecycle_provider.delays(row, self.aliases, scenario)
                        for key in ("parent_receipt_ms", "worker_next_ready_ms") + (("permit_release_ms",) if self.gated else ()):
                            if delays.get(key) is None:
                                raise UnknownEvidence("UNKNOWN_POST_HANDLER_LIFECYCLE:" + key + ":" + alias)
                            number(delays[key], key)
                        if self.gated and delays["permit_release_ms"] < delays["parent_receipt_ms"]:
                            raise ValueError("Permit release precedes parent receipt")
                    receipt_at = clock + delays["parent_receipt_ms"] * 1e6
                    for member in members:
                        record = {**member, "finish_ns": now_ns + round(receipt_at), "handler_end_ns": end_ns,
                                  "start_ns": starts[rid], "predicted_remaining_service_ms": service_time[rid],
                                  "remaining_service_by_kind_ms": service_by_kind[rid]}
                        if alias in self.batch_configurations:
                            record.update(batch_input_key=row["pool_id"], batch_delivery_scope="whole batch return")
                        receipts[member["request_id"]] = {"at": receipt_at, "record": record}
                    if self.gated:
                        releases[rid] = clock + delays["permit_release_ms"] * 1e6
                    ready_times[alias] = clock + delays["worker_next_ready_ms"] * 1e6
                    lifecycle_events.append({"kind": "handler_end", "request_id": rid, "alias": alias, "time_ns": end_ns})
                    active.pop(alias)
                    progress.pop(alias)
            for rid, handoff in list(receipts.items()):
                if handoff["at"] <= clock:
                    finishes[rid] = handoff["record"]
                    lifecycle_events.append({"kind": "parent_receipt", "request_id": rid,
                                             "alias": handoff["record"]["alias"], "time_ns": now_ns + round(handoff["at"])})
                    receipts.pop(rid)
            for rid, at in list(releases.items()):
                if at <= clock:
                    alias = permitted.pop(rid)
                    lifecycle_events.append({"kind": "permit_release", "request_id": rid, "alias": alias,
                                             "time_ns": now_ns + round(at)})
                    releases.pop(rid)
            for alias, at in list(ready_times.items()):
                if at <= clock:
                    lifecycle_events.append({"kind": "worker_next_ready", "alias": alias, "time_ns": now_ns + round(at)})
                    ready_times.pop(alias)
            for alias, prep in list(preparing.items()):
                if prep["at"] <= clock:
                    rid = prep["row"]["request_id"]
                    prepared[rid] = now_ns + round(prep["at"])
                    preparation_events.append({"kind": "prepare_ready", "alias": alias, "request_id": rid,
                                               "time_ns": now_ns + round(prep["at"])})
                    preparing.pop(alias)
            transition["completed_aliases"] = sorted(a for a in done if a not in active)
            while arrival_index < len(future) and future[arrival_index]["arrival_ns"] - now_ns <= clock:
                row = future[arrival_index]
                alias = route_future(copy.deepcopy(row), {a: list(q) for a, q in pending.items()}, now_ns + round(clock))
                if alias not in self.aliases or row["task"] not in self.aliases[alias]["supported_tasks"]:
                    raise ValueError("Unsupported continuation route")
                row["alias"] = alias
                pending[alias].append(row)
                cohort[row["request_id"]] = row
                diagnostics.register(row)
                arrival_index += 1
            if clock >= drain_end:
                break
            start_ready()
        incomplete = sorted(set(cohort) - set(finishes))
        diagnostics.observe(now_ns + round(clock), 0, pending, active, plans, progress, permitted, waiting, batch_members=batch_members)
        return {"finishes": finishes, "cohort": cohort, "incomplete_ids": incomplete,
                "queue_diagnostics": diagnostics.result(cohort),
                "lifecycle_events": lifecycle_events,
                "batch_execution": batch_audit,
                "batch_preparation_events": preparation_events,
                "batch_preparation_scope": "serial owner; batch dispatch before next prepare; explicit preprocessing duration; power/memory require matching evidence",
                "lifecycle_scope": ("explicit receipt/release/worker-ready evidence" if self.lifecycle_provider is not None
                                    else "legacy instantaneous handler-to-receipt/release/ready assumption"),
                "board_energy_joules": energy if energy_known else None, "energy_known": energy_known,
                "peak_memory_mib": peak if memory_known else None, "memory_known": memory_known,
                "transitions": transitions, "initial_residual_evidence": evidence,
                "transition_count": events,
                "trace_scope": "full" if trace_request_id is None else "current-request active intervals only; all workload events simulated",
                "drain_finish_ns": now_ns + round(clock), "future_arrival_count": len(future),
                "scope": "Full remaining arrival horizon, ordered segment progress and nonpreemptive FIFO execution",
                "admission_scope": "observed initial FIFO tickets; deterministic alias ordering for simultaneous predicted sender offers",
                "physical_six_term_improvement_certified": False}
