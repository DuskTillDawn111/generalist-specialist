"""Training-bound observations of three distinct post-handler boundaries.

Missing worker-readiness or gate-release evidence remains unknown. In
particular, a RESULT reader timestamp is never substituted for permit release.
"""

from __future__ import annotations

import copy
import statistics

from ..forecast_errors import UnknownEvidence
from .profiles import _provenance
from .workload import number


LIFECYCLE_SCHEMA = "gsdeploy.handler_lifecycle_profiles.v2"
OBSERVATION_SCHEMA = "gsdeploy.handler_lifecycle_observation.v2"
DELAY_FIELDS = ("parent_receipt_ms", "permit_release_ms", "worker_next_ready_ms")


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError("Explicit " + name + " required")
    return value


def _timestamp(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("Nonnegative integer " + name + " required")
    return value


def _delays(value, permit_scope):
    if permit_scope not in {"request_admission", "not_applicable"}:
        raise ValueError("Explicit permit scope required")
    if set(value) != set(DELAY_FIELDS):
        raise ValueError("Three distinct lifecycle delay fields required")
    for key in DELAY_FIELDS:
        if value[key] is not None:
            number(value[key], key)
    receipt, release = value["parent_receipt_ms"], value["permit_release_ms"]
    if release is not None and (permit_scope == "not_applicable" or receipt is None or release < receipt):
        raise ValueError("Owned permit release must follow parent receipt")


def _gate_releases(snapshot):
    """Replay a complete critical-section event stream before using releases."""
    if (snapshot.get("event_contract") != "gsdeploy.admission_events.v2"
            or snapshot.get("clock") != "host_perf_counter_ns"
            or snapshot.get("events_after_sequence") != 0):
        raise ValueError("Full v2 admission event stream required; old receipt is not release")
    active, waiting, releases = {}, [], {}
    previous_sequence, previous_time, closed = 0, -1, False
    for event in snapshot["events"]:
        timestamp = _timestamp(event["time_ns"], "gate event timestamp")
        if type(event["sequence"]) is not int or event["sequence"] != previous_sequence + 1 or timestamp < previous_time:
            raise ValueError("Gate event sequence/clock order mismatch")
        previous_sequence, previous_time = event["sequence"], timestamp
        kind, rid, alias = event["kind"], event["request_id"], event["alias"]
        ticket = (alias, rid)
        if kind == "enqueue":
            if closed or rid in active or any(r == rid for _, r in waiting):
                raise ValueError("Invalid gate enqueue history")
            waiting.append(ticket)
        elif kind == "grant":
            if closed or not waiting or waiting[0] != ticket:
                raise ValueError("Invalid gate FIFO grant history")
            waiting.pop(0)
            active[rid] = alias
        elif kind == "release":
            if rid in releases or active.get(rid) != alias:
                raise ValueError("Duplicate or unowned lifecycle permit release")
            active.pop(rid)
            releases[rid] = event
        elif kind == "cancel":
            if ticket not in waiting:
                raise ValueError("Gate cancellation lacks queued ticket")
            waiting.remove(ticket)
        elif kind == "close":
            if closed:
                raise ValueError("Duplicate gate close event")
            closed = True
        elif kind == "reject":
            if not closed:
                raise ValueError("Gate rejection precedes closure")
        else:
            raise ValueError("Unknown gate event kind")
        if (event["active_count_after"] != len(active)
                or event["waiting_count_after"] != len(waiting)):
            raise ValueError("Gate post-transition counts mismatch")
    if (previous_sequence != snapshot["last_sequence"] or active != snapshot["active"]
            or waiting != [tuple(ticket) for ticket in snapshot["waiting"]] or closed != snapshot["closed"]
            or _timestamp(snapshot["observed_time_ns"], "snapshot observation") < previous_time):
        raise ValueError("Incomplete or inconsistent admission event stream")
    return releases


class LifecycleProvider:
    """Exact family/task/input lookup under one execution identity.

    Scenario entries are joint vectors, not independently selected quantiles.
    ``known`` covers only boundaries applicable to the declared permit scope.
    Per-input means are an explicit model assumption, not a scheduler bound.
    """

    def __init__(self, document, execution_context_sha256, *, allow_cpu_fixture=False):
        _text(execution_context_sha256, "execution context")
        required = {"schema", "execution_context_sha256", "provenance", "profiles"}
        if not isinstance(document, dict) or set(document) != required:
            raise ValueError("Invalid lifecycle profile fields")
        if (document["schema"] != LIFECYCLE_SCHEMA
                or document["execution_context_sha256"] != execution_context_sha256):
            raise ValueError("Lifecycle execution context mismatch")
        _provenance(document["provenance"])
        if document["provenance"]["kind"] == "cpu_fixture" and not allow_cpu_fixture:
            raise ValueError("CPU fixture is not physical lifecycle evidence")
        self.document = copy.deepcopy(document)
        self.execution_context_sha256 = execution_context_sha256
        self.physical_evidence = document["provenance"]["kind"] == "measured"
        self.lookup = {}
        if not isinstance(self.document["profiles"], list) or not self.document["profiles"]:
            raise ValueError("Nonempty lifecycle profiles required")
        for profile in self.document["profiles"]:
            required = {"family", "task", "pool_id", "permit_scope", "scenarios", "evidence"}
            if required - set(profile) or set(profile) - required - {"joint_observations"}:
                raise ValueError("Invalid lifecycle profile")
            key = tuple(_text(profile[k], k) for k in ("family", "task", "pool_id"))
            if key in self.lookup:
                raise ValueError("Duplicate lifecycle profile")
            if not isinstance(profile["scenarios"], dict) or not profile["scenarios"]:
                raise ValueError("Explicit lifecycle scenarios required")
            if (not isinstance(profile["evidence"], list) or not profile["evidence"]
                    or any(not isinstance(e, str) or not e for e in profile["evidence"])):
                raise ValueError("Lifecycle evidence identities required")
            for scenario, delays in profile["scenarios"].items():
                _text(scenario, "scenario")
                _delays(delays, profile["permit_scope"])
            if "joint_observations" in profile:
                if not isinstance(profile["joint_observations"], list):
                    raise ValueError("Joint lifecycle observations must be a list")
                seen = set()
                for observation in profile["joint_observations"]:
                    if set(observation) != {"id", "delays"}:
                        raise ValueError("Joint lifecycle observation identity/delays required")
                    oid = _text(observation["id"], "joint observation identity")
                    if oid in seen or oid not in profile["evidence"]:
                        raise ValueError("Duplicate or unbound joint lifecycle observation")
                    seen.add(oid)
                    _delays(observation["delays"], profile["permit_scope"])
            self.lookup[key] = profile

    def delays(self, row, aliases, scenario):
        alias = row["alias"]
        profile = self.lookup.get((aliases[alias]["family"], row["task"], row["pool_id"]))
        if profile is None:
            raise UnknownEvidence("UNKNOWN_HANDLER_LIFECYCLE_INPUT:" + alias + ":" + row["pool_id"])
        if scenario not in profile["scenarios"]:
            raise UnknownEvidence("UNKNOWN_HANDLER_LIFECYCLE_SCENARIO:" + scenario)
        delays = copy.deepcopy(profile["scenarios"][scenario])
        required = ["parent_receipt_ms", "worker_next_ready_ms"]
        if profile["permit_scope"] == "request_admission":
            required.append("permit_release_ms")
        return {**delays, "known": all(delays[k] is not None for k in required),
                "permit_scope": profile["permit_scope"], "evidence": list(profile["evidence"]),
                "scope": "training_bound_post_handler_lifecycle"}

    def residual_handoff(self, row, aliases, scenario, now_ns, observed, *, boundary_observation_status=None):
        """Empirical joint mean residual, conditional on confirmed event pattern.

        Unknown boundary state (including a readiness announcement still in
        transit) is not a survival observation. Exact observed boundary times
        validate chronology, but samples are conditioned on the coarser pattern
        of happened/pending at this age, not on continuous exact-time equality.
        """
        now = _timestamp(now_ns, "handoff observation time")
        fields = {"parent_receipt_ns": "parent_receipt_ms", "permit_release_ns": "permit_release_ms",
                  "worker_next_ready_ns": "worker_next_ready_ms"}
        if not isinstance(observed, dict) or set(observed) != {"handler_end_ns", *fields}:
            raise ValueError("Explicit observed handoff boundary timestamps required")
        end = _timestamp(observed["handler_end_ns"], "observed handler end")
        if end > now:
            raise ValueError("Handler end is in the observation future")
        alias = row["alias"]
        profile = self.lookup.get((aliases[alias]["family"], row["task"], row["pool_id"]))
        if profile is None:
            raise UnknownEvidence("UNKNOWN_HANDLER_LIFECYCLE_INPUT:" + alias + ":" + row["pool_id"])
        if scenario != "mean":
            raise UnknownEvidence("UNKNOWN_CONDITIONAL_HANDOFF_SCENARIO:" + scenario)
        if (not isinstance(boundary_observation_status, dict)
                or set(boundary_observation_status) != set(fields)):
            raise UnknownEvidence("CONFIRMED_HANDOFF_BOUNDARY_STATE_REQUIRED")
        applicable = list(fields)
        if profile["permit_scope"] == "not_applicable":
            applicable.remove("permit_release_ns")
            if observed["permit_release_ns"] is not None or boundary_observation_status["permit_release_ns"] != "not_applicable":
                raise ValueError("Shared execution cannot have an owned permit boundary")
        for boundary in applicable:
            status, timestamp = boundary_observation_status[boundary], observed[boundary]
            if status == "unknown":
                raise UnknownEvidence("UNKNOWN_HANDOFF_BOUNDARY_STATE:" + boundary)
            if status == "observed":
                if not end <= _timestamp(timestamp, "observed handoff boundary") <= now:
                    raise ValueError("Observed handoff boundary outside handler-end/current time")
            elif status == "confirmed_pending":
                if timestamp is not None:
                    raise ValueError("Confirmed-pending boundary already has a timestamp")
            else:
                raise ValueError("Unknown handoff observation status")
        if (observed["permit_release_ns"] is not None
                and (observed["parent_receipt_ns"] is None
                     or observed["permit_release_ns"] < observed["parent_receipt_ns"])):
            raise ValueError("Observed permit release must follow observed parent receipt")
        joint = profile.get("joint_observations")
        if not joint:
            raise UnknownEvidence("JOINT_HANDOFF_TRAINING_OBSERVATIONS_REQUIRED")
        age_ms = (now - end) / 1e6
        selected = []
        for sample in joint:
            delays = sample["delays"]
            if any(delays[fields[boundary]] is None for boundary in applicable):
                continue
            if all((delays[fields[boundary]] <= age_ms if boundary_observation_status[boundary] == "observed"
                    else delays[fields[boundary]] > age_ms) for boundary in applicable):
                selected.append(sample)
        if len(selected) < 2:
            raise UnknownEvidence("INSUFFICIENT_CONDITIONAL_JOINT_HANDOFF_SUPPORT")
        remaining = {key.replace("_ms", "_remaining_ms"): None for key in DELAY_FIELDS}
        for boundary in applicable:
            field = fields[boundary]
            remaining[field.replace("_ms", "_remaining_ms")] = (
                0. if boundary_observation_status[boundary] == "observed" else
                statistics.mean(sample["delays"][field] - age_ms for sample in selected))
        return {**remaining, "known": True, "permit_scope": profile["permit_scope"],
                "evidence": [sample["id"] for sample in selected],
                "condition_evidence": {"handler_age_ms": age_ms, "observed": copy.deepcopy(observed),
                    "boundary_observation_status": dict(boundary_observation_status),
                    "joint_support_count": len(selected),
                    "conditioning": "confirmed event pattern at age; exact observed boundary times not regression features"},
                "scope": "training_empirical_joint_conditional_mean_handoff_residual",
                "calibrated_conditional_distribution": False}


def export_training_observations(raw, aliases, *, execution_context_sha256,
                                 clock_contract, source, worker_ready_events=None):
    """Export explicit observations; never infer readiness from next START.

    Input wrapper must declare training/kind/context/admission_enabled and
    request_records. ``clock_contract`` asserts externally verified common
    host perf-counter origin; device/Unix clocks require prior explicit mapping.
    Readiness events must mark the previous request id and the boundary's
    meaning, rather than opportunistically assigning the next request's START.
    """
    _text(source, "source identity")
    _text(execution_context_sha256, "execution context")
    if (raw.get("split") != "training" or raw.get("not_for_training", False) is not False
            or raw.get("execution_context_sha256") != execution_context_sha256
            or raw.get("kind") not in {"measured", "cpu_fixture"}
            or type(raw.get("admission_enabled")) is not bool):
        raise ValueError("Explicit training-only lifecycle source/context/permit mode required")
    if (not isinstance(clock_contract, dict)
            or set(clock_contract) != {"handler", "runtime", "worker_next_ready", "same_host_verified", "source"}
            or any(clock_contract[k] != "host_perf_counter_ns" for k in ("handler", "runtime", "worker_next_ready"))
            or clock_contract["same_host_verified"] is not True):
        raise ValueError("Verified common host perf-counter clock contract required")
    _text(clock_contract["source"], "clock verification source")
    if not isinstance(raw.get("request_records"), list) or not raw["request_records"]:
        raise ValueError("Nonempty training request records required")
    snapshot = raw.get("request_admission", raw.get("admission_snapshot_after"))
    releases = {}
    if raw["admission_enabled"] and snapshot is not None:
        releases = _gate_releases(snapshot)
    ready = {}
    for event in worker_ready_events or []:
        if (set(event) != {"request_id", "alias", "time_ns", "clock", "boundary", "source"}
                or event["clock"] != "host_perf_counter_ns"
                or event["boundary"] != "previous_handler_next_request_ready"):
            raise ValueError("Explicit worker readiness event required")
        _text(event["source"], "readiness source")
        _timestamp(event["time_ns"], "readiness timestamp")
        if event["request_id"] in ready:
            raise ValueError("Duplicate worker readiness event")
        ready[event["request_id"]] = event
    observations, seen = [], set()
    for row in raw["request_records"]:
        if row.get("split") != "training" or row.get("status") != "completed":
            raise ValueError("Completed training-only request required")
        rid = _text(row.get("request_id"), "request identity")
        if rid in seen:
            raise ValueError("Duplicate lifecycle request identity")
        seen.add(rid)
        alias = row["model"]
        if alias not in aliases:
            raise ValueError("Unknown lifecycle alias")
        response = row.get("worker_response", {})
        end = _timestamp(response.get("end_perf_ns"), "handler end")
        receipt = _timestamp(row.get("response_reader_event_enqueue_time_ns"), "reader event enqueue")
        if row.get("completion_time_ns") != receipt or receipt < end:
            raise ValueError("Handler/parent receipt chronology mismatch")
        release = releases.get(rid)
        if release is not None:
            if release["alias"] != alias or release["reason"] != "worker_result":
                raise ValueError("Completed request must release on its own worker RESULT")
            if release["time_ns"] < receipt:
                raise ValueError("Permit release precedes reader receipt")
            if row.get("admission_released_time_ns", release["time_ns"]) != release["time_ns"]:
                raise ValueError("Record and gate release timestamps disagree")
        elif row.get("admission_released_time_ns") is not None:
            raise ValueError("Record release lacks authoritative gate event")
        next_ready = ready.get(rid)
        recorded_ready = row.get("previous_handler_next_request_ready")
        if recorded_ready is not None:
            if (recorded_ready.get("request_id") != rid
                    or recorded_ready.get("clock") != "host_perf_counter_ns"
                    or recorded_ready.get("boundary") != "previous_handler_next_request_ready"
                    or recorded_ready.get("scope") != "after_response_flush_before_next_read"
                    or type(recorded_ready.get("pid")) is not int or recorded_ready["pid"] <= 0):
                raise ValueError("Invalid recorded worker readiness boundary")
            timestamp = _timestamp(recorded_ready.get("perf_ns"), "recorded readiness timestamp")
            received = _timestamp(recorded_ready.get("received_time_ns"), "recorded readiness reader timestamp")
            if timestamp > received:
                raise ValueError("Readiness reader receipt precedes worker boundary")
            if next_ready is not None and (next_ready["time_ns"] != timestamp or next_ready["alias"] != alias):
                raise ValueError("External and recorded readiness boundaries disagree")
            next_ready = {"request_id": rid, "alias": alias, "time_ns": timestamp, "source": source + ":WORKER_READY"}
        if next_ready is not None and (next_ready["alias"] != alias or next_ready["time_ns"] < end):
            raise ValueError("Worker readiness identity/chronology mismatch")
        timestamps = {"handler_end_ns": end, "parent_receipt_ns": receipt,
                      "permit_release_ns": release["time_ns"] if release is not None else None,
                      "worker_next_ready_ns": next_ready["time_ns"] if next_ready is not None else None}
        delays = {"parent_receipt_ms": (receipt - end) / 1e6,
                  "permit_release_ms": (release["time_ns"] - end) / 1e6 if release is not None else None,
                  "worker_next_ready_ms": (next_ready["time_ns"] - end) / 1e6 if next_ready is not None else None}
        scope = "request_admission" if raw["admission_enabled"] else "not_applicable"
        _delays(delays, scope)
        observations.append({"schema": OBSERVATION_SCHEMA, "id": source + ":" + rid,
                             "split": "training", "kind": raw["kind"],
                             "execution_context_sha256": execution_context_sha256,
                             "family": aliases[alias]["family"], "task": row["task"], "pool_id": row["pool_id"],
                             "permit_scope": scope, "clock": "host_perf_counter_ns",
                             "timestamps": timestamps, "delays": delays,
                             "source": source, "clock_source": clock_contract["source"],
                             "ready_source": next_ready["source"] if next_ready is not None else None})
    if set(ready) - seen:
        raise ValueError("Readiness event does not match training cohort")
    return observations


def fit_lifecycle(observations, execution_context_sha256, *, minimum_observations=2):
    """Fit explicit input-conditioned mean vectors, retaining missing evidence.

    A missing boundary in any member leaves that boundary unknown. No marginal
    percentile vector is manufactured or advertised as a joint tail scenario.
    """
    _text(execution_context_sha256, "execution context")
    if type(minimum_observations) is not int or minimum_observations < 2:
        raise ValueError("At least two lifecycle observations required")
    grouped, seen, kinds, sources = {}, set(), set(), set()
    for observation in observations:
        if (observation.get("schema") != OBSERVATION_SCHEMA or observation.get("split") != "training"
                or observation.get("execution_context_sha256") != execution_context_sha256
                or observation.get("clock") != "host_perf_counter_ns"
                or observation.get("kind") not in {"measured", "cpu_fixture"}):
            raise ValueError("Training-only, same-context host lifecycle observations required")
        oid = _text(observation.get("id"), "observation identity")
        if oid in seen:
            raise ValueError("Duplicate lifecycle observation")
        seen.add(oid)
        kinds.add(observation["kind"])
        sources.add(_text(observation.get("source"), "training source"))
        _delays(observation["delays"], observation["permit_scope"])
        timestamps = observation.get("timestamps", {})
        if set(timestamps) != {"handler_end_ns", "parent_receipt_ns", "permit_release_ns", "worker_next_ready_ns"}:
            raise ValueError("Observed lifecycle boundaries required for fitting")
        end = _timestamp(timestamps["handler_end_ns"], "observed handler end")
        for field, timestamp_field in zip(DELAY_FIELDS, ("parent_receipt_ns", "permit_release_ns", "worker_next_ready_ns")):
            timestamp = timestamps[timestamp_field]
            delay = observation["delays"][field]
            if timestamp is None:
                if delay is not None:
                    raise ValueError("Delay lacks observed lifecycle boundary")
            elif (_timestamp(timestamp, "observed lifecycle boundary") < end
                  or delay != (timestamp - end) / 1e6):
                raise ValueError("Lifecycle delay differs from observed boundary")
        key = tuple(_text(observation.get(k), k) for k in ("family", "task", "pool_id", "permit_scope"))
        grouped.setdefault(key, []).append(observation)
    if len(kinds) != 1 or not grouped:
        raise ValueError("Nonempty homogeneous physical/CPU lifecycle evidence required")
    profiles = []
    for key, rows in sorted(grouped.items()):
        if len(rows) < minimum_observations:
            raise UnknownEvidence("INSUFFICIENT_LIFECYCLE_TRAINING_OBSERVATIONS:" + ":".join(key))
        means = {}
        for field in DELAY_FIELDS:
            values = [r["delays"][field] for r in rows]
            means[field] = statistics.mean(values) if all(v is not None for v in values) else None
        profiles.append({"family": key[0], "task": key[1], "pool_id": key[2], "permit_scope": key[3],
                         "scenarios": {"mean": means}, "evidence": [r["id"] for r in rows],
                         "joint_observations": [{"id": r["id"], "delays": copy.deepcopy(r["delays"])} for r in rows]})
    document = {"schema": LIFECYCLE_SCHEMA, "execution_context_sha256": execution_context_sha256,
                "provenance": {"kind": next(iter(kinds)), "split": "training", "sources": sorted(sources)},
                "profiles": profiles}
    LifecycleProvider(document, execution_context_sha256, allow_cpu_fixture=True)
    return document
