"""Bind real batch formation to configuration-specific execution profiles.

No linear batch speedup, cache discount, padding discount, or inferred output
length. A missing exact batch profile remains UNKNOWN. KernelResourceProvider
can supply the ordered plan; independent measurements supply power and memory.
"""
from __future__ import annotations

import copy

from ..forecast_errors import UnknownEvidence
from ..resources import identity


def batch_input_key(rows, selection):
    return "batch:" + identity({"inputs": [(r["task"], r["pool_id"]) for r in rows],
                                "padded_input_tokens": selection["padded_input_tokens"],
                                "batch_size": selection["batch_size"],
                                "execution_keys": [r["execution_key"] for r in rows]})


class BatchedProfileProvider:
    def __init__(self, provider, input_metadata, *, execution_context_sha256):
        if provider.execution_context_sha256 != execution_context_sha256:
            raise ValueError("Batch and service execution identities differ")
        self.provider = provider
        self.execution_context_sha256 = execution_context_sha256
        self.physical_evidence = provider.physical_evidence
        self.input_metadata = copy.deepcopy(input_metadata)
        for task, pools in self.input_metadata.items():
            for pool, row in pools.items():
                if (set(row) != {"input_tokens", "execution_key", "input_ready_delay_ns", "evidence"}
                        or type(row["input_tokens"]) is not int or row["input_tokens"] < 1
                        or type(row["input_ready_delay_ns"]) is not int or row["input_ready_delay_ns"] < 0
                        or not isinstance(row["execution_key"], str) or not row["execution_key"]
                        or not isinstance(row["evidence"], str) or not row["evidence"]):
                    raise ValueError("Explicit public token/config/readiness evidence required")

    def scheduling_row(self, row):
        metadata = self.input_metadata.get(row["task"], {}).get(row["pool_id"])
        if metadata is None:
            raise UnknownEvidence("UNKNOWN_BATCH_INPUT_METADATA:" + row["pool_id"])
        # Historical field spelling retained in the serialized input contract:
        # input_ready_delay_ns is a PREPARATION DURATION, not a promise that
        # readiness occurs at arrival+duration. FullHorizonModel serializes
        # preparation against the same owner's generate and cleanup.
        return {**row, **metadata, "owner": row["alias"],
                "ready_ns": row.get("batch_ready_ns", row["arrival_ns"] + metadata["input_ready_delay_ns"])}

    def plan_batch(self, rows, selection, aliases, scenario):
        public_rows = [self.scheduling_row(r) for r in rows]
        key = batch_input_key(public_rows, selection)
        representative = {**rows[0], "pool_id": key}
        plan = self.provider.plan(representative, aliases, scenario)
        if plan["execution"]["batch_size"] != selection["batch_size"]:
            raise ValueError("Batch profile size differs from selected batch")
        return representative, plan

    def plan(self, *args):
        # Used only for solo-work diagnostics. Exact singleton profiles are
        # distinct from batch profiles; unavailable solo work stays unknown.
        return self.provider.plan(*args)

    def service(self, *args): return self.provider.service(*args)
    def power(self, *args): return self.provider.power(*args)
    def memory(self, *args): return self.provider.memory(*args)

    def power_preparing(self, preparing, active, aliases, scenario):
        method = getattr(self.provider, "power_preparing", None)
        return method(preparing, active, aliases, scenario) if method else dict(known=False, watts=None)

    def memory_preparing(self, preparing, active, aliases, scenario):
        method = getattr(self.provider, "memory_preparing", None)
        return method(preparing, active, aliases, scenario) if method else dict(known=False, peak_memory_mib=None)
