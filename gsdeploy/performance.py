"""Shared independent or frozen whole-plan colocation service evidence.

Profiles describe family-normalized service times, before n/g replica scaling.
No unmeasured slowdown, traffic dependence, or GPU capacity is invented.
"""

import math
from .energy import energy_key


def profile_key(data, deployment, routing, rates, horizon, seed, arrival_process):
    # Deliberately exclude profile contents: this identifies the measurement
    # contract. Energy identity separately includes the selected profile contents.
    base = {
        k: data[k]
        for k in ("models", "capacity_factors", "memory_table_mib", "safe_factor")
    }
    return energy_key(
        deployment, routing, rates, horizon, seed, arrival_process, catalog=base
    )


def resolve_service(
    data,
    deployment,
    routing,
    rates,
    horizon=120.0,
    seed=42,
    arrival_process="deterministic_periodic",
):
    mode = data.get("performance_mode", "independent")
    if mode == "independent":
        return {m: dict(data["models"][m]["service_time_ms"]) for m in deployment}, {
            "mode": mode,
            "kind": "standalone_replica_surrogate",
            "scope": "constant-service independent FIFO replicas; colocation and service tails unmeasured",
        }
    key = profile_key(data, deployment, routing, rates, horizon, seed, arrival_process)
    profile = data.get("colocation_profiles", {}).get(key)
    if not valid_profile(profile):
        return None, {"mode": mode, "key": key, "reason": "MISSING_COLOCATION_PROFILE"}
    service = profile["service_time_ms"]
    for t, row in routing.items():
        for m, q in row.items():
            if q > 0 and t not in service.get(m, {}):
                return None, {
                    "mode": mode,
                    "key": key,
                    "reason": "INCOMPLETE_COLOCATION_PROFILE",
                }
    return service, {
        "mode": mode,
        "key": key,
        "kind": profile["kind"],
        "provenance": profile.get("provenance"),
        "scope": "frozen whole-plan colocation service profile; service tails unmeasured",
    }


def valid_profile(profile):
    if not isinstance(profile, dict) or profile.get("kind") not in {
        "measured",
        "synthetic_fixture",
    }:
        return False
    if (
        profile.get("scope") != "whole_gpu_colocation"
        or profile.get("coverage_valid") is not True
    ):
        return False
    service = profile.get("service_time_ms")
    return isinstance(service, dict) and all(
        isinstance(row, dict)
        and all(
            not isinstance(v, bool)
            and isinstance(v, (int, float))
            and math.isfinite(v)
            and v > 0
            for v in row.values()
        )
        for row in service.values()
    )


def service_lower_ms(data, model, task):
    if data.get("performance_mode", "independent") == "independent":
        return data["models"][model]["service_time_ms"][task]
    # Partial measurements do not establish a universal minimum service time
    # for unknown combinations. Zero safely relaxes both latency and capacity.
    # It is a mathematical bound, never a predicted service time or score.
    return 0.0
