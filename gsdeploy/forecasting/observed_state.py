"""Reconcile forecasts to visible host observations without inferring GPU work.

This stores observed facts, never predicted completion times. Delayed worker
events become visible at their reader receipt time, not their earlier worker
timestamp. Frozen evaluation fixes an observation cutoff for the whole state.
"""

from __future__ import annotations

import copy

from ..resources import identity


STATE_SCHEMA = "gsdeploy.visible_host_state.v2"
FEEDBACK_SCHEMA = "gsdeploy.visible_feedback.v2"
CLOCK = "host_perf_counter_ns"


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError("Explicit " + name + " required")
    return value


def _ns(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("Nonnegative integer " + name + " required")
    return value


def _digest(state):
    return identity({k: value for k, value in state.items() if k != "state_sha256"})


def new_observed_state(aliases, execution_context_sha256, *, mode="online",
                       configuration_sha256=None, frozen_cutoff_ns=None):
    """Create an empty observer, requiring a complete first gate history.

    A previous model's predicted progress is deliberately not a bootstrap
    source. An existing process needs a complete v2 history or an independently
    verified checkpoint before it can be reconciled.
    """
    _text(execution_context_sha256, "execution identity")
    if not isinstance(aliases, dict) or not aliases:
        raise ValueError("Resident aliases required")
    if any(not isinstance(alias, str) or not alias for alias in aliases):
        raise ValueError("Explicit alias identities required")
    if mode not in {"online", "frozen"}:
        raise ValueError("Observer mode must be online or frozen")
    if mode == "frozen":
        _ns(frozen_cutoff_ns, "frozen observation cutoff")
    elif frozen_cutoff_ns is not None:
        raise ValueError("Online observation cannot have a frozen cutoff")
    if configuration_sha256 is not None:
        _text(configuration_sha256, "resource identity")
    state = {"schema": STATE_SCHEMA, "execution_context_sha256": execution_context_sha256,
             "configuration_sha256": configuration_sha256, "clock": CLOCK, "mode": mode,
             "frozen_cutoff_ns": frozen_cutoff_ns, "observed_at_ns": None,
             "aliases": sorted(aliases), "admission_snapshot": None,
             "last_feedback_sequence": 0, "last_feedback_received_ns": None,
             "last_arrival_sequence": 0, "last_arrival_observed_ns": None,
             "last_arrivals": {}, "requests": {},
             "unknowns": ["KERNEL_CURSOR_UNOBSERVED", "INITIAL_HANDOFF_RESIDUAL_UNMODELLED"]}
    state["state_sha256"] = _digest(state)
    return state


def feedback_event(sequence, kind, payload, received_time_ns, execution_context_sha256):
    """Wrap a collector observation in the consumer's ordered feedback feed.

    ``sequence`` is the application's persistent feed sequence, not a native
    kernel order or a claim that the worker protocol has a global sequence.
    """
    return {"schema": FEEDBACK_SCHEMA, "sequence": sequence, "kind": kind,
            "payload": copy.deepcopy(payload), "received_time_ns": received_time_ns,
            "clock": CLOCK, "execution_context_sha256": execution_context_sha256}


def feedback_from_runtime_records(records, execution_context_sha256, now_ns, *, since_sequence=0):
    """Adapt visible START/PHASE/RESULT/WORKER_READY records to one feed.

    The caller supplies a consistent snapshot of runtime records. Sequence is
    deterministic reader-receipt order, with tied observations ordered by
    request identity and boundary. A later snapshot must retain the earlier
    cohort; use this adapter with one immutable request registry, not a sliding
    window that removes already consumed observations. Standard workers report
    their start timestamp in RESULT; it becomes visible at response receipt
    when there was no separate START observation.
    """
    _ns(now_ns, "record visibility cutoff")
    _text(execution_context_sha256, "execution identity")
    if type(since_sequence) is not int or since_sequence < 0:
        raise ValueError("Nonnegative feedback cursor required")
    visible = []
    for record in records:
        rid, alias = record["request_id"], record["model"]
        execution_events = record.get("execution_events", [])
        for index, event in enumerate(execution_events):
            received = _ns(event.get("received_time_ns"), "execution event reader receipt")
            if received > now_ns:
                continue
            if "event" in event:
                kind = "PHASE"
                fields = ("request_id", "perf_ns", "segment_id", "kind", "event")
            else:
                kind = "START"
                fields = ("request_id", "perf_ns")
            payload = {key: event[key] for key in fields}
            if payload["request_id"] != rid:
                raise ValueError("Execution event belongs to another request")
            payload["alias"] = alias
            visible.append((received, rid, 0, index, kind, payload))
        receipt = record.get("response_reader_event_enqueue_time_ns")
        if receipt is not None and _ns(receipt, "response reader receipt") <= now_ns:
            response = record.get("worker_response", {})
            start, end = response.get("start_perf_ns"), response.get("end_perf_ns")
            if end is not None:
                if start is not None and not any("event" not in event for event in execution_events):
                    # This timestamp is observed only once RESULT arrives;
                    # never backdate its visibility to the handler start.
                    payload = {"request_id": rid, "alias": alias, "perf_ns": start}
                    visible.append((receipt, rid, 0, -1, "START", payload))
                payload = {"request_id": rid, "alias": alias, "perf_ns": end}
                visible.append((receipt, rid, 1, 0, "HANDLER_END", payload))
        ready = record.get("previous_handler_next_request_ready")
        if ready is not None and _ns(ready.get("received_time_ns"), "readiness reader receipt") <= now_ns:
            fields = ("request_id", "perf_ns", "clock", "boundary", "scope")
            payload = {key: ready[key] for key in fields}
            if payload["request_id"] != rid:
                raise ValueError("Readiness event belongs to another request")
            payload["alias"] = alias
            visible.append((ready["received_time_ns"], rid, 2, 0, "WORKER_READY", payload))
    visible.sort(key=lambda row: row[:4])
    if since_sequence > len(visible):
        raise ValueError("Feedback registry lost already consumed events")
    return [feedback_event(sequence, kind, payload, received, execution_context_sha256)
            for sequence, (received, _, _, _, kind, payload) in enumerate(visible, start=1)
            if sequence > since_sequence]


def _request(state, rid, alias):
    _text(rid, "request identity")
    if alias not in state["aliases"]:
        raise ValueError("Unknown observation alias")
    if rid not in state["requests"]:
        state["requests"][rid] = {"request_id": rid, "alias": alias,
                                 "kernel_cursor": None, "kernel_cursor_status": "UNKNOWN",
                                 "handler_start_ns": None, "handler_end_ns": None,
                                 "worker_next_ready_ns": None, "phases": {}, "phase_order": []}
    row = state["requests"][rid]
    if row["alias"] != alias:
        raise ValueError("Request observation changed alias")
    return row


def _apply_gate(state, snapshot, now_ns):
    previous = state["admission_snapshot"]
    if (snapshot.get("event_contract") != "gsdeploy.admission_events.v2"
            or snapshot.get("clock") != CLOCK
            or snapshot.get("configuration_sha256") != state["configuration_sha256"]):
        raise ValueError("Admission clock/contract/resource identity mismatch")
    observed = _ns(snapshot.get("observed_time_ns"), "gate observation time")
    if observed > now_ns or (previous is not None and observed < previous["observed_time_ns"]):
        raise ValueError("Future or backward gate observation")
    sequence = previous["last_sequence"] if previous is not None else 0
    if snapshot.get("events_after_sequence") != sequence:
        raise ValueError("Admission event cursor gap or duplicate")
    active = copy.deepcopy(previous["active"]) if previous is not None else {}
    waiting = [tuple(t) for t in previous["waiting"]] if previous is not None else []
    closed = previous["closed"] if previous is not None else False
    previous_time = previous["last_event_time_ns"] if previous is not None else 0
    peak = previous["peak_active"] if previous is not None else 0
    for event in snapshot["events"]:
        event_ns = _ns(event.get("time_ns"), "gate event time")
        if (type(event.get("sequence")) is not int or event["sequence"] != sequence + 1
                or not previous_time <= event_ns <= observed):
            raise ValueError("Gate sequence gap/duplicate or clock violation")
        kind, rid, alias = event["kind"], event["request_id"], event["alias"]
        row = _request(state, rid, alias) if rid is not None else None
        ticket = (alias, rid)
        if kind == "enqueue":
            if closed or rid in active or any(r == rid for _, r in waiting) or "admission_enqueue_ns" in row:
                raise ValueError("Invalid gate enqueue")
            waiting.append(ticket)
            row["admission_enqueue_ns"] = event_ns
        elif kind == "grant":
            if closed or not waiting or waiting.pop(0) != ticket or rid in active:
                raise ValueError("Invalid FIFO grant")
            active[rid] = alias
            row["admission_grant_ns"] = event_ns
        elif kind == "release":
            if active.get(rid) != alias:
                raise ValueError("Gate release without matching permit")
            active.pop(rid)
            row["admission_release_ns"] = event_ns
            row["admission_release_reason"] = event["reason"]
        elif kind == "cancel":
            if ticket not in waiting:
                raise ValueError("Gate cancel without queued ticket")
            waiting.remove(ticket)
            row["admission_cancel_ns"] = event_ns
        elif kind == "reject":
            if not closed:
                raise ValueError("Gate reject before close")
        elif kind == "close":
            if closed or rid is not None or alias is not None:
                raise ValueError("Invalid gate close")
            closed = True
        else:
            raise ValueError("Unknown gate event kind")
        peak = max(peak, len(active))
        if event["active_count_after"] != len(active) or event["waiting_count_after"] != len(waiting):
            raise ValueError("Gate post-transition counts disagree")
        sequence, previous_time = event["sequence"], event_ns
    if (sequence != snapshot["last_sequence"] or active != snapshot["active"]
            or waiting != [tuple(t) for t in snapshot["waiting"]]
            or closed != snapshot["closed"] or peak != snapshot["peak_active"]):
        raise ValueError("Admission replay differs from current snapshot")
    state["admission_snapshot"] = {
        "active": active, "waiting": waiting, "closed": closed, "peak_active": peak,
        "event_contract": "gsdeploy.admission_events.v2", "clock": CLOCK,
        "configuration_sha256": state["configuration_sha256"], "last_sequence": sequence,
        "observed_time_ns": observed, "last_event_time_ns": previous_time,
        "events_after_sequence": sequence, "events": [],
    }


def _apply_feedback(state, event, now_ns):
    required = {"schema", "sequence", "kind", "payload", "received_time_ns", "clock", "execution_context_sha256"}
    if (set(event) != required or event["schema"] != FEEDBACK_SCHEMA or event["clock"] != CLOCK
            or event["execution_context_sha256"] != state["execution_context_sha256"]
            or type(event["sequence"]) is not int
            or event["sequence"] != state["last_feedback_sequence"] + 1):
        raise ValueError("Feedback sequence/clock/execution identity mismatch")
    received = _ns(event["received_time_ns"], "feedback visibility time")
    previous_received = state["last_feedback_received_ns"]
    if received > now_ns or (previous_received is not None and received < previous_received):
        raise ValueError("Future or backward feedback visibility")
    payload, kind = event["payload"], event["kind"]
    timestamp = _ns(payload.get("perf_ns"), "worker event time")
    if timestamp > received:
        raise ValueError("Worker event is not in the verified host clock")
    row = _request(state, payload["request_id"], payload["alias"])
    if row.get("last_worker_event_ns") is not None and timestamp < row["last_worker_event_ns"]:
        raise ValueError("Per-request worker event clock moved backward")
    if kind == "START":
        if row["handler_start_ns"] is not None or row["handler_end_ns"] is not None:
            raise ValueError("Duplicate/out-of-order handler START")
        row["handler_start_ns"] = timestamp
    elif kind == "PHASE":
        if row["handler_start_ns"] is None or timestamp < row["handler_start_ns"] or row["handler_end_ns"] is not None:
            raise ValueError("PHASE outside observed handler")
        key = _text(payload.get("segment_id"), "phase identity")
        _text(payload.get("kind"), "phase kind")
        edge = payload.get("event")
        phase = row["phases"].get(key)
        if edge == "start":
            if phase is not None:
                raise ValueError("Duplicate phase start")
            row["phases"][key] = {"kind": payload["kind"], "start_ns": timestamp, "end_ns": None}
            row["phase_order"].append(key)
        elif edge == "end":
            if phase is None or phase["end_ns"] is not None or phase["kind"] != payload["kind"] or timestamp < phase["start_ns"]:
                raise ValueError("Unmatched/duplicate phase end")
            phase["end_ns"] = timestamp
        else:
            raise ValueError("Unknown phase edge")
    elif kind == "HANDLER_END":
        if (row["handler_start_ns"] is None or row["handler_end_ns"] is not None
                or timestamp < row["handler_start_ns"]
                or any(p["end_ns"] is None or p["end_ns"] > timestamp for p in row["phases"].values())):
            raise ValueError("Invalid handler END chronology")
        row["handler_end_ns"] = timestamp
        row["parent_receipt_ns"] = received
    elif kind == "WORKER_READY":
        if (payload.get("clock") != CLOCK
                or payload.get("boundary") != "previous_handler_next_request_ready"
                or payload.get("scope") != "after_response_flush_before_next_read"
                or row["worker_next_ready_ns"] is not None or row["handler_end_ns"] is None
                or timestamp < row["handler_end_ns"]):
            raise ValueError("Invalid explicit worker readiness boundary")
        row["worker_next_ready_ns"] = timestamp
    else:
        raise ValueError("Unsupported visible feedback kind")
    row["last_visible_feedback_ns"] = received
    row["last_worker_event_ns"] = timestamp
    state["last_feedback_sequence"] = event["sequence"]
    state["last_feedback_received_ns"] = received


def advance_observed_state(previous, now_ns, *, admission_snapshot=None, feedback=(), arrivals=(), mode=None):
    """Return a new immutable-by-convention state after visible observations.

    All streams must be incremental, gap-free, ordered, and no later than now.
    Public arrivals accept no labels or model outputs. They have their own
    sequence so an empty interval cannot silently discard arrivals. ``mode``
    may be supplied as a guard; it cannot change an existing state's mode.
    """
    _ns(now_ns, "observation time")
    if (previous.get("schema") != STATE_SCHEMA or previous.get("clock") != CLOCK
            or previous.get("state_sha256") != _digest(previous)):
        raise ValueError("Invalid or mutated observer checkpoint")
    if mode is not None and mode != previous["mode"]:
        raise ValueError("Cannot change observer online/frozen mode")
    if previous["observed_at_ns"] is not None and now_ns < previous["observed_at_ns"]:
        raise ValueError("Observer time moved backward")
    if previous["mode"] == "frozen" and now_ns > previous["frozen_cutoff_ns"]:
        raise ValueError("Frozen prospective evaluation cannot consume later feedback")
    state = copy.deepcopy(previous)
    for arrival in arrivals:
        required = {"sequence", "request_id", "alias", "task", "pool_id", "arrival_ns", "observed_time_ns"}
        if set(arrival) != required or type(arrival["sequence"]) is not int or arrival["sequence"] != state["last_arrival_sequence"] + 1:
            raise ValueError("Arrival stream sequence gap/duplicate or invalid fields")
        arrived = _ns(arrival["arrival_ns"], "public arrival time")
        observed = _ns(arrival["observed_time_ns"], "public arrival visibility time")
        if (not arrived <= observed <= now_ns or (state["last_arrival_observed_ns"] is not None
                                                and observed < state["last_arrival_observed_ns"])):
            raise ValueError("Future or unobservable arrival")
        _text(arrival["task"], "task")
        _text(arrival["pool_id"], "public input identity")
        row = _request(state, arrival["request_id"], arrival["alias"])
        if "arrival_ns" in row:
            raise ValueError("Duplicate observed request arrival")
        row.update({key: arrival[key] for key in ("task", "pool_id", "arrival_ns")})
        old = state["last_arrivals"].get(arrival["task"])
        if old is not None and arrived < old["arrival_ns"]:
            raise ValueError("Per-task visible arrival order moved backward")
        state["last_arrivals"][arrival["task"]] = {
            "request_id": arrival["request_id"], "arrival_ns": arrived, "observed_time_ns": observed,
            "pool_id": arrival["pool_id"], "alias": arrival["alias"]}
        state["last_arrival_sequence"] = arrival["sequence"]
        state["last_arrival_observed_ns"] = observed
    if admission_snapshot is not None:
        if state["configuration_sha256"] is None:
            raise ValueError("Gate observations require an explicit resource identity")
        _apply_gate(state, admission_snapshot, now_ns)
    for event in feedback:
        _apply_feedback(state, event, now_ns)
    # Cross-stream release may be observed before the delayed RESULT feed is
    # supplied. Check the relationship once both facts are actually visible.
    for row in state["requests"].values():
        if (row.get("admission_release_reason") == "worker_result" and row.get("parent_receipt_ns") is not None
                and row["admission_release_ns"] < row["parent_receipt_ns"]):
            raise ValueError("Permit release precedes observed RESULT receipt")
    state["observed_at_ns"] = now_ns
    state["state_sha256"] = _digest(state)
    return state


def forecast_snapshot_from_observed(state, aliases, execution_context_sha256, now_ns, tasks):
    """Expose verified observed facts to the forecast entry, without progress.

    The caller owns a complete arrival feed from the phase start. Missing tasks
    have no observed arrival; this is NOT permission to bootstrap mid-run with
    an incomplete registry. State must be reconciled at this decision time.
    """
    if (state.get("schema") != STATE_SCHEMA or state.get("state_sha256") != _digest(state)
            or state.get("execution_context_sha256") != execution_context_sha256
            or state.get("aliases") != sorted(aliases) or state.get("observed_at_ns") != now_ns):
        raise ValueError("Visible state does not match forecast execution/aliases/decision time")
    if state["mode"] == "frozen" and now_ns > state["frozen_cutoff_ns"]:
        raise ValueError("Frozen forecast cannot consume later observations")
    if state["admission_snapshot"] is not None and state["admission_snapshot"]["observed_time_ns"] != now_ns:
        raise ValueError("Stale admission snapshot cannot become current by advancing observer time")
    return {"admission": copy.deepcopy(state["admission_snapshot"]),
            "observed_last_arrivals_ns": {task: state["last_arrivals"].get(task, {}).get("arrival_ns") for task in tasks},
            "visible_state_sha256": state["state_sha256"],
            "kernel_cursor_status": "UNKNOWN_UNLESS_SEPARATELY_OBSERVED"}


def validate_handoffs_against_observed(handoffs, state):
    """Explicit initial residual inputs cannot contradict observed facts."""
    for handoff in handoffs:
        row = state["requests"].get(handoff["request_id"])
        if row is None:
            raise ValueError("Handoff lacks corresponding visible request")
        for key in ("alias", "task", "pool_id", "arrival_ns", "handler_end_ns", "parent_receipt_ns",
                    "worker_next_ready_ns"):
            if row.get(key) is not None and handoff.get(key) != row[key]:
                raise ValueError("Handoff contradicts visible observed boundary: " + key)
        if row.get("admission_release_ns") is not None and handoff.get("permit_release_ns") != row["admission_release_ns"]:
            raise ValueError("Handoff contradicts observed permit release")
