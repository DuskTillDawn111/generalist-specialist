"""Evidence-bound, ordered phase/kernel profiles and calibrated progress.

USHER motivates configuration-dependent resource demands; Roomie motivates
sequence alignment. We replay ordered segments and reprice at every boundary.
We do NOT import Roomie's disputed formula, assume occupancy is throughput, or
infer multi-way slowdown by adding pairwise slowdowns. Context progress must
be supported by observations under this exact execution configuration.
"""

from __future__ import annotations

import copy
import math
import statistics

import numpy as np
from scipy.optimize import nnls

from ..forecast_errors import UnknownEvidence
from ..resources import identity
from .workload import number


PROFILE_SCHEMA = "gsdeploy.segment_resource_profiles.v1"


def context_key(active):
    """Include input/config-specific profile and currently overlapping segment."""
    return identity(sorted((alias, row["profile_id"], row["segment_id"])
                           for alias, row in active.items()))


def preparation_context_key(preparing, active):
    return identity({"active": sorted((a, r["profile_id"], r["segment_id"]) for a, r in active.items()),
                     "preparing": sorted((a, p["row"]["task"], p["row"]["pool_id"]) for a, p in preparing.items())})


def _provenance(value):
    if (not isinstance(value, dict) or set(value) != {"kind", "sources", "split"}
            or value["kind"] not in {"measured", "cpu_fixture"}
            or value["split"] != "training" or not isinstance(value["sources"], list)
            or not value["sources"] or any(not isinstance(s, str) or not s for s in value["sources"])):
        raise ValueError("Explicit training evidence provenance required")


class ProfileProvider:
    """Lookup profiles and full-context calibrated speeds, memory and power.

    Each profile is bound to family/task/pool_id and an execution signature
    (batch size, backend, weight-sharing mode, etc.). The enclosing context
    hash binds those signatures to the actual candidate execution. Cache
    reuse has no implicit discount: a merged backend needs its own profile.
    """

    def __init__(self, document, execution_context_sha256, *, allow_cpu_fixture=False):
        self.document = copy.deepcopy(document)
        required = {"schema", "execution_context_sha256", "provenance", "profiles", "contexts"}
        if set(document) - required - {"fit_diagnostics", "preparation_contexts", "context_interpolation"} or required - set(document):
            raise ValueError("Invalid segment profile document fields")
        if document["schema"] != PROFILE_SCHEMA or document["execution_context_sha256"] != execution_context_sha256:
            raise ValueError("Resource execution context mismatch")
        if not isinstance(execution_context_sha256, str) or not execution_context_sha256:
            raise ValueError("Explicit resource execution identity required")
        _provenance(document["provenance"])
        if document["provenance"]["kind"] == "cpu_fixture" and not allow_cpu_fixture:
            raise ValueError("CPU fixture is not physical profiling evidence")
        self.execution_context_sha256 = execution_context_sha256
        self.physical_evidence = document["provenance"]["kind"] == "measured"
        self.profiles, self.lookup = {}, {}
        for profile in self.document["profiles"]:
            if set(profile) != {"id", "family", "task", "pool_id", "execution", "segments"}:
                raise ValueError("Invalid profile fields")
            if any(not isinstance(profile[k], str) or not profile[k] for k in ("id", "family", "task", "pool_id")):
                raise ValueError("Explicit profile identities required")
            if not isinstance(profile["execution"], dict) or not profile["execution"]:
                raise ValueError("Explicit execution configuration required")
            if (type(profile["execution"].get("batch_size")) is not int or profile["execution"]["batch_size"] < 1
                    or not isinstance(profile["execution"].get("backend"), str) or not profile["execution"]["backend"]
                    or type(profile["execution"].get("weight_reuse")) is not bool):
                raise ValueError("Execution requires integer batch_size, backend and weight_reuse boolean")
            key = (profile["family"], profile["task"], profile["pool_id"])
            if profile["id"] in self.profiles or key in self.lookup or not profile["segments"]:
                raise ValueError("Duplicate or empty execution profile")
            segment_ids = set()
            for segment in profile["segments"]:
                if set(segment) != {"id", "kind", "duration_ms", "demand"}:
                    raise ValueError("Explicit segment id/kind/durations/resource demand required")
                if (not isinstance(segment["id"], str) or not segment["id"] or segment["id"] in segment_ids
                        or segment["kind"] not in {"host", "gpu", "sync", "cleanup"}):
                    raise ValueError("Invalid segment identity/kind")
                segment_ids.add(segment["id"])
                if not isinstance(segment["duration_ms"], dict) or not segment["duration_ms"]:
                    raise ValueError("Named duration scenarios required")
                for duration in segment["duration_ms"].values():
                    number(duration, "segment baseline duration", positive=True)
                if set(segment["demand"]) != {"sm", "dram", "l2"}:
                    raise ValueError("Separate SM, DRAM and L2 demand required")
                for demand in segment["demand"].values():
                    if demand is None and segment["kind"] == "gpu":
                        continue
                    if number(demand, "normalized resource demand") > 1:
                        raise ValueError("Resource demand fractions must be <=1")
                if segment["kind"] != "gpu" and any(segment["demand"].values()):
                    raise ValueError("Non-GPU segments cannot claim GPU resource demand")
            self.profiles[profile["id"]] = profile
            self.lookup[key] = profile
        if not self.profiles:
            raise ValueError("Nonempty execution profiles required")
        if not isinstance(document["contexts"], dict):
            raise ValueError("Explicit calibrated context table required")
        for key, context in self.document["contexts"].items():
            if set(context) != {"members", "scenarios", "evidence"}:
                raise ValueError("Invalid calibrated context fields")
            if not isinstance(context["members"], dict):
                raise ValueError("Context members required, including explicit idle context")
            for alias, member in context["members"].items():
                if (not isinstance(alias, str) or not alias or set(member) != {"profile_id", "segment_id"}
                        or member["profile_id"] not in self.profiles
                        or member["segment_id"] not in {s["id"] for s in self.profiles[member["profile_id"]]["segments"]}):
                    raise ValueError("Invalid calibrated context member")
            if key != context_key(context["members"]) or not context["scenarios"]:
                raise ValueError("Context identity/scenarios invalid")
            if not isinstance(context["evidence"], str) or not context["evidence"]:
                raise ValueError("Context evidence identity required")
            for scenario in context["scenarios"].values():
                if set(scenario) != {"progress", "board_watts", "peak_memory_mib"}:
                    raise ValueError("Explicit progress/board power/peak memory fields required")
                if set(scenario["progress"]) != set(context["members"]):
                    raise ValueError("Progress must cover every context member")
                for speed in scenario["progress"].values():
                    number(speed, "segment progress", positive=True)
                for term in ("board_watts", "peak_memory_mib"):
                    if scenario[term] is not None:
                        number(scenario[term], term)
        for key, context in self.document.get("preparation_contexts", {}).items():
            if (set(context) != {"active", "preparing", "scenarios", "evidence"}
                    or not context["preparing"] or not isinstance(context["evidence"], str) or not context["evidence"]):
                raise ValueError("Explicit preparation power/memory context required")
            active, preparing = context["active"], context["preparing"]
            if set(active) & set(preparing):
                raise ValueError("One owner cannot prepare and generate simultaneously")
            for alias, member in active.items():
                if (set(member) != {"profile_id", "segment_id"} or member["profile_id"] not in self.profiles
                        or member["segment_id"] not in {s["id"] for s in self.profiles[member["profile_id"]]["segments"]}):
                    raise ValueError("Unknown active preparation context segment")
            for alias, member in preparing.items():
                if (set(member) != {"family", "task", "pool_id"}
                        or any(not isinstance(v, str) or not v for v in member.values())):
                    raise ValueError("Preparation family/task/input identity required")
            wrapped = {a: {"row": r} for a, r in preparing.items()}
            if key != preparation_context_key(wrapped, active) or not context["scenarios"]:
                raise ValueError("Preparation context identity/scenarios mismatch")
            for scenario in context["scenarios"].values():
                if set(scenario) != {"board_watts", "peak_memory_mib"}:
                    raise ValueError("Preparation context has independent board power and owned memory")
                for name, value in scenario.items():
                    if value is not None:
                        number(value, name)

        self.context_law = None
        if 'context_interpolation' in document:
            from .context_law import ConvexContextLaw
            self.context_law = ConvexContextLaw(self.document, self.profiles)

    def plan(self, row, aliases, scenario):
        alias = row["alias"]
        profile = self.lookup.get((aliases[alias]["family"], row["task"], row["pool_id"]))
        if profile is None:
            raise UnknownEvidence("UNKNOWN_INPUT_EXECUTION_PROFILE:" + alias + ":" + row["pool_id"])
        segments = []
        for segment in profile["segments"]:
            if scenario not in segment["duration_ms"]:
                raise UnknownEvidence("UNKNOWN_SEGMENT_DURATION_SCENARIO:" + scenario)
            segments.append({**copy.deepcopy(segment), "duration_ms": segment["duration_ms"][scenario]})
        return {"profile_id": profile["id"], "execution": copy.deepcopy(profile["execution"]), "segments": segments}

    def _context(self, active, aliases, scenario):
        for alias, row in active.items():
            profile = self.profiles.get(row.get("profile_id"))
            if profile is None or (profile["family"], profile["task"], profile["pool_id"]) != (
                    aliases[alias]["family"], row["task"], row["pool_id"]):
                raise ValueError("Active profile does not match request identity")
        key = context_key(active)
        context = self.document["contexts"].get(key)
        # Missing fields/scenarios of an observed context remain UNKNOWN.
        # Interpolation is explicit and only fills entirely unseen contexts.
        if context is None and self.context_law is not None:
            return self.context_law.evaluate(active, scenario, key)
        if context is None or scenario not in context["scenarios"]:
            raise UnknownEvidence("UNKNOWN_ALIGNED_SEGMENT_CONTEXT:" + key + ":" + scenario)
        return context["scenarios"][scenario], context["evidence"]

    def service(self, row, active, aliases, scenario):
        context, evidence = self._context(active, aliases, scenario)
        return {"known": True, "context_supported": True,
                "service_ms": row["segment_duration_ms"] / context["progress"][row["alias"]],
                "evidence": evidence, "scope": "calibrated_ordered_segment_progress"}

    def power(self, active, aliases, scenario):
        context, evidence = self._context(active, aliases, scenario)
        return {"known": context["board_watts"] is not None, "watts": context["board_watts"],
                "evidence": evidence, "scope": "whole_board_once"}

    def memory(self, active, aliases, scenario):
        context, evidence = self._context(active, aliases, scenario)
        return {"known": context["peak_memory_mib"] is not None, "peak_memory_mib": context["peak_memory_mib"],
                "evidence": evidence, "scope": "raw_owned_worker_context_peak_including_idle_residents"}

    def _preparation_context(self, preparing, active, aliases, scenario):
        key = preparation_context_key(preparing, active)
        context = self.document.get("preparation_contexts", {}).get(key)
        if context is None or scenario not in context["scenarios"]:
            return None, None
        for alias, member in context["preparing"].items():
            if (alias not in aliases or member["family"] != aliases[alias]["family"]
                    or member["task"] not in aliases[alias]["supported_tasks"]):
                raise ValueError("Preparation context differs from resident execution identity")
        return context["scenarios"][scenario], context["evidence"]

    def power_preparing(self, preparing, active, aliases, scenario):
        value, evidence = self._preparation_context(preparing, active, aliases, scenario)
        watts = value["board_watts"] if value else None
        return dict(known=watts is not None, watts=watts, evidence=evidence, scope="whole_board_with_owner_preparation")

    def memory_preparing(self, preparing, active, aliases, scenario):
        value, evidence = self._preparation_context(preparing, active, aliases, scenario)
        memory = value["peak_memory_mib"] if value else None
        return dict(known=memory is not None, peak_memory_mib=memory, evidence=evidence, scope="owned_workers_with_owner_preparation")

    def demand_summary(self, profile_id):
        """Candidate ordering aid only; never an admission or pruning proof."""
        profile = self.profiles[profile_id]
        gpu = [s for s in profile["segments"] if s["kind"] == "gpu"]
        if any(v is None for s in gpu for v in s["demand"].values()):
            return {"classification": "unknown", "sm": None, "dram": None, "l2": None,
                    "reason": "GPU_RESOURCE_COUNTER_EVIDENCE_UNAVAILABLE"}
        if not gpu:
            return {"classification": "host", "sm": 0., "dram": 0., "l2": 0.}
        durations = [s["duration_ms"].get("mean") for s in gpu]
        if any(d is None for d in durations):
            raise UnknownEvidence("MEAN_SCENARIO_REQUIRED_FOR_DEMAND_SUMMARY")
        total = sum(durations)
        demand = {name: sum(d * s["demand"][name] for d, s in zip(durations, gpu)) / total
                  for name in ("sm", "dram", "l2")}
        demand["classification"] = ("compute" if demand["sm"] > 1.2 * demand["dram"] else
                                    "memory_bandwidth" if demand["dram"] > 1.2 * demand["sm"] else "balanced")
        return demand


def fit_progress(document, observations, *, minimum_observations=2, max_relative_work_error=.15):
    """Fit mean progress from COMPLETE segment traces with changing overlap.

    For target segment j: sum(dt_context * speed_context / solo_ms_j)=1.
    No ridge guesses for unidentifiable contexts: rank-deficient fits stay
    UNKNOWN. Observations must be training-only and must bind the candidate.
    Power/memory and non-mean scenarios are never manufactured by this fit.
    """
    if type(minimum_observations) is not int or minimum_observations < 2:
        raise ValueError("At least two independent complete observations required")
    number(max_relative_work_error, "maximum relative work error")
    result = copy.deepcopy(document)
    # Empty progress slots are allowed only in the fitting input, not provider.
    if result.get("schema") != PROFILE_SCHEMA:
        raise ValueError("Explicit segment profile schema required")
    _provenance(result["provenance"])
    profiles = {p["id"]: p for p in result["profiles"]}
    grouped, seen = {}, set()
    for row in observations:
        if set(row) != {"id", "split", "execution_context_sha256", "target_alias", "profile_id", "segment_id", "intervals"}:
            raise ValueError("Complete segment training observation fields required")
        if (not isinstance(row["id"], str) or not row["id"] or row["id"] in seen
                or row["split"] != "training" or row["execution_context_sha256"] != result["execution_context_sha256"]):
            raise ValueError("Duplicate, nontraining or wrong-context observation")
        seen.add(row["id"])
        profile = profiles.get(row["profile_id"])
        segment = next((s for s in profile["segments"] if s["id"] == row["segment_id"]), None) if profile else None
        if segment is None or "mean" not in segment["duration_ms"] or not row["intervals"]:
            raise ValueError("Unknown training target segment")
        durations = {}
        for interval in row["intervals"]:
            if set(interval) != {"duration_ms", "members"}:
                raise ValueError("Interval duration and observed members required")
            number(interval["duration_ms"], "observed interval", positive=True)
            key = context_key(interval["members"])
            if key not in result["contexts"] or result["contexts"][key]["members"] != interval["members"]:
                raise ValueError("Unknown training overlap context")
            if interval["members"].get(row["target_alias"]) != {"profile_id": row["profile_id"], "segment_id": row["segment_id"]}:
                raise ValueError("Target missing from observed overlap")
            durations[key] = durations.get(key, 0.) + interval["duration_ms"] / segment["duration_ms"]["mean"]
        grouped.setdefault((row["target_alias"], row["profile_id"], row["segment_id"]), []).append((row["id"], durations))
    diagnostics = []
    # Clear all fitted mean speeds to prevent stale fits surviving bad evidence.
    for context in result["contexts"].values():
        if "mean" in context["scenarios"]:
            context["scenarios"]["mean"]["progress"] = {}
    for (alias, profile_id, segment_id), rows in sorted(grouped.items()):
        keys = sorted({key for _, durations in rows for key in durations})
        matrix = np.array([[durations.get(key, 0.) for key in keys] for _, durations in rows])
        rank = int(np.linalg.matrix_rank(matrix))
        diagnostic = {"alias": alias, "profile_id": profile_id, "segment_id": segment_id,
                      "observation_ids": [rid for rid, _ in rows], "contexts": keys,
                      "matrix_rank": rank, "columns": len(keys), "accepted": False}
        if len(rows) >= minimum_observations and rank == len(keys):
            speeds, _ = nnls(matrix, np.ones(len(rows)))
            errors = abs(matrix @ speeds - 1.)
            diagnostic["max_relative_work_error"] = float(max(errors))
            diagnostic["mean_relative_work_error"] = float(statistics.mean(errors))
            if all(math.isfinite(s) and s > 0 for s in speeds) and max(errors) <= max_relative_work_error:
                diagnostic["accepted"] = True
                for key, speed in zip(keys, speeds):
                    result["contexts"][key]["scenarios"]["mean"]["progress"][alias] = float(speed)
        diagnostics.append(diagnostic)
    for context in result["contexts"].values():
        scenario = context["scenarios"].get("mean")
        if scenario is not None and set(scenario["progress"]) != set(context["members"]):
            del context["scenarios"]["mean"]
    # Drop unsupported scenarios/contexts; providers report missing as UNKNOWN.
    result["contexts"] = {k: c for k, c in result["contexts"].items() if c["scenarios"]}
    result["fit_diagnostics"] = {"groups": diagnostics, "scope": "training work reconstruction only; not holdout or confidence bounds"}
    ProfileProvider(result, result["execution_context_sha256"], allow_cpu_fixture=True)
    return result
