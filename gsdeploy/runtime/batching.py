"""Deterministic FIFO batch formation shared by execution and prediction.

Rows are already tokenized public requests. No future output length is used.
This module decides a batch; the caller must actually execute it through a
batch backend. It grants neither GPU resource isolation nor request permits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..resources import identity


@dataclass(frozen=True)
class BatchConfiguration:
    max_batch_size: int
    max_padded_tokens: int
    max_wait_ns: int
    length_buckets: tuple = ()

    def __post_init__(self):
        for name in ("max_batch_size", "max_padded_tokens", "max_wait_ns"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "max_wait_ns" else 1):
                raise ValueError("Invalid batch configuration: " + name)
        if (not isinstance(self.length_buckets, tuple)
                or any(type(v) is not int or v < 1 for v in self.length_buckets)
                or tuple(sorted(set(self.length_buckets))) != self.length_buckets):
            raise ValueError("Length buckets must be strictly increasing positive integers")


def select_batch(rows, now_ns, configuration):
    """Select one compatible FIFO prefix or wait until head's absolute limit.

    Buckets prevent incompatible padding; they do not silently bypass the FIFO
    head. max_padded_tokens constrains actual padded prefill tokens. This is
    complete-request batching, not preemptive decode or KV-cache migration.
    """
    if type(now_ns) is not int:
        raise ValueError("Integer batch decision time required")
    rows = list(rows)
    if not rows:
        return {"status": "EMPTY", "members": [], "next_event_ns": None}
    seen = set()
    for row in rows:
        for key in ("request_id", "owner", "task", "execution_key"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError("Explicit batch identity required: " + key)
        if (type(row.get("ready_ns")) is not int or row["ready_ns"] < 0
                or type(row.get("input_tokens")) is not int or row["input_tokens"] < 1
                or row["request_id"] in seen):
            raise ValueError("Invalid token count, duplicate identity or ready time")
        seen.add(row["request_id"])
    first = rows[0]
    if first["ready_ns"] > now_ns:
        return {"status": "WAIT", "members": [], "next_event_ns": first["ready_ns"]}
    if first["input_tokens"] > configuration.max_padded_tokens:
        return {"status": "UNSUPPORTED", "reason": "PREFILL_MICROBATCH_EXECUTOR_REQUIRED",
                "members": [], "next_event_ns": None}

    def key(row):
        bucket = next((b for b in configuration.length_buckets if row["input_tokens"] <= b), None)
        # Above the last bucket is its own compatible overflow bucket, still
        # subject to the exact padded-token budget.
        return row["owner"], row["task"], row["execution_key"], bucket

    members, max_tokens, blocked = [], 0, False
    for row in rows:
        if row["ready_ns"] > now_ns:
            break
        proposed_max = max(max_tokens, row["input_tokens"])
        if key(row) != key(first) or proposed_max * (len(members) + 1) > configuration.max_padded_tokens:
            blocked = True
            break
        members.append(row)
        max_tokens = proposed_max
        if len(members) == configuration.max_batch_size:
            break
    dispatch_by = first["ready_ns"] + configuration.max_wait_ns
    if len(members) < configuration.max_batch_size and not blocked and now_ns < dispatch_by:
        next_ready = next((r["ready_ns"] for r in rows if r["ready_ns"] > now_ns), dispatch_by)
        return {"status": "WAIT", "members": [], "next_event_ns": min(dispatch_by, next_ready),
                "formation_wait_so_far_ns": now_ns - first["ready_ns"]}
    useful = sum(r["input_tokens"] for r in members)
    padded = max_tokens * len(members)
    return {"status": "READY", "members": [r["request_id"] for r in members], "next_event_ns": None,
            "batch_id": identity({"members": [r["request_id"] for r in members], "config": asdict(configuration)}),
            "batch_size": len(members), "useful_input_tokens": useful, "padded_input_tokens": padded,
            "padding_tokens": padded - useful, "padding_fraction": (padded - useful) / padded,
            "formation_wait_ns": {r["request_id"]: now_ns - r["ready_ns"] for r in members},
            "dispatch_reason": "full" if len(members) == configuration.max_batch_size else "fifo_barrier" if blocked else "timeout",
            "execution_scope": "same-owner complete-request batch; no implicit microbatch/weight/cache discount"}


def execute_selected_batch(selection, rows, backend):
    """Execute the exact planned batch once, retaining per-request identity.

    Backend.run_batch accepts samples and returns one result per input, in
    input order. Individual delivery occurs after the entire call returns.
    """
    if selection.get("status") != "READY":
        raise ValueError("Only a READY selection can execute")
    index = {r["request_id"]: r for r in rows}
    selected = [index[rid] for rid in selection["members"]]
    if len({(r["owner"], r["task"], r["execution_key"]) for r in selected}) != 1:
        raise ValueError("Batch execution identities differ")
    result = backend.run_batch([r["sample"] for r in selected])
    if not isinstance(result, (list, tuple)) or len(result) != len(selected):
        raise ValueError("Backend result cardinality differs from planned batch")
    return [{"request_id": row["request_id"], "batch_id": selection["batch_id"], "result": value}
            for row, value in zip(selected, result)]
