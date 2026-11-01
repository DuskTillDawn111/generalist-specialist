"""Nonpreemptive host admission for complete requests, not CUDA kernel control.

Permits are acquired before transport submission and returned on RESULT, ERROR,
send failure or worker exit. Waiting remains host queue time. Strict FIFO
admission is deterministic relative to actual sender arrival order; it does not
claim a globally optimal or future-aware execution schedule.
"""

from __future__ import annotations

from collections import deque
import copy
import threading
import time

from ..resources import identity, validate_resources


def admission_available(resources, active_aliases, alias):
    """Pure capacity rule shared by the runtime gate and CPU event simulator."""
    active_aliases = tuple(active_aliases)
    if active_aliases.count(alias) >= resources["allocations"][alias]["max_inflight"]:
        return False
    return all(sum(a in group["aliases"] for a in active_aliases) < group["max_active"]
               for group in resources["groups"].values() if alias in group["aliases"])


class RequestAdmission:
    def __init__(self, resources):
        resources = validate_resources(resources, resources["allocations"])
        if resources["mode"] != "request_admission":
            raise ValueError("Explicit request-admission configuration required")
        self.resources = resources
        self.configuration_sha256 = identity(resources)
        self.condition = threading.Condition()
        self.waiting = deque()
        self.active = {}
        self.closed = False
        self.peak_active = 0
        self.events = []
        self._request_events = {}

    def _event(self, kind, alias=None, request_id=None, reason=None):
        """Append at the state transition, while holding ``condition``.

        No callback takes the deployment record lock here: senders acquire the
        gate independently and collectors can hold that lock while releasing.
        """
        event = {"sequence": len(self.events) + 1, "kind": kind,
                 "time_ns": time.perf_counter_ns(), "alias": alias,
                 "request_id": request_id, "reason": reason,
                 "active_count_after": len(self.active),
                 "waiting_count_after": len(self.waiting)}
        self.events.append(event)
        if request_id is not None:
            self._request_events.setdefault(request_id, []).append(event)
        return dict(event)

    def request_events(self, request_id):
        """Exact gate transitions, independently of record update timing."""
        with self.condition:
            return [dict(event) for event in self._request_events.get(request_id, ())]

    def _available(self, alias):
        return admission_available(self.resources, self.active.values(), alias)

    def acquire(self, alias, request_id):
        """Block a sender without holding the runtime record lock."""
        if alias not in self.resources["allocations"] or not isinstance(request_id, str) or not request_id:
            raise ValueError("Known alias and nonempty request id required")
        with self.condition:
            if request_id in self.active or any(t[1] == request_id for t in self.waiting):
                raise ValueError("Duplicate admission request id")
            if self.closed:
                self._event("reject", alias, request_id, "gate_closed")
                return False
            ticket = (alias, request_id)
            self.waiting.append(ticket)
            self._event("enqueue", alias, request_id)
            self.condition.notify_all()
            try:
                self.condition.wait_for(lambda: self.closed or
                                        (self.waiting[0] == ticket and self._available(alias)))
                if self.closed:
                    self.waiting.remove(ticket)
                    self._event("cancel", alias, request_id, "gate_closed")
                    return False
                self.waiting.remove(ticket)
                self.active[request_id] = alias
                self.peak_active = max(self.peak_active, len(self.active))
                self._event("grant", alias, request_id)
                return True
            finally:
                if ticket in self.waiting:
                    self.waiting.remove(ticket)
                    self._event("cancel", alias, request_id, "acquire_exception")
                self.condition.notify_all()

    def release(self, request_id, reason="caller_release"):
        with self.condition:
            alias = self.active.pop(request_id, None)
            event = self._event("release", alias, request_id, reason) if alias is not None else None
            self.condition.notify_all()
            return event

    def release_alias(self, alias, reason="alias_release"):
        with self.condition:
            events = []
            for rid in [r for r, a in self.active.items() if a == alias]:
                self.active.pop(rid)
                events.append(self._event("release", alias, rid, reason))
            self.condition.notify_all()
            return events

    def close(self):
        with self.condition:
            if not self.closed:
                self.closed = True
                self._event("close", reason="runtime_shutdown")
            self.condition.notify_all()

    def snapshot(self, since_sequence=0):
        """Atomically capture gate state and events after a consumer cursor.

        ``since_sequence=0`` includes all events, preserving the evidence export
        default. Rolling predictors can pass the last consumed sequence.
        """
        with self.condition:
            if (isinstance(since_sequence, bool) or not isinstance(since_sequence, int)
                    or not 0 <= since_sequence <= len(self.events)):
                raise ValueError("Event cursor must be an existing sequence or zero")
            return {"active": dict(self.active), "waiting": list(self.waiting),
                    "closed": self.closed, "peak_active": self.peak_active,
                    "event_contract": "gsdeploy.admission_events.v2",
                    "configuration_sha256": self.configuration_sha256,
                    "clock": "host_perf_counter_ns", "last_sequence": len(self.events),
                    "events_after_sequence": since_sequence,
                    "observed_time_ns": time.perf_counter_ns(),
                    "events": [dict(event) for event in self.events[since_sequence:]],
                    "scope": "Host complete-request admission only; no GPU resource isolation"}


def admission_for_registry(registry):
    """Every worker must agree on one complete admission contract."""
    selected = [entry.get("resource_binding", {}) for entry in registry.values()
                if entry.get("resource_binding", {}).get("mode") == "request_admission"]
    if not selected:
        return None
    if len(selected) != len(registry):
        raise ValueError("Admission configuration must cover every resident worker")
    resources = copy.deepcopy(selected[0]["admission_configuration"])
    resources = validate_resources(resources, registry)
    for alias, entry in registry.items():
        binding = entry["resource_binding"]
        if (binding["alias"] != alias or binding["admission_configuration"] != resources
                or binding["configuration_sha256"] != identity(resources)):
            raise ValueError("Conflicting worker admission configurations")
    return RequestAdmission(resources)
