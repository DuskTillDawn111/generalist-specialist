"""Whole-plan energy records, tied to the actual planning contract.

No additive per-model power assumption or implicit missing=zero conversion.
"""

import hashlib
import json
import math


def energy_key(
    deployment,
    routing,
    rates,
    horizon,
    seed,
    arrival_process,
    multiplier=1.0,
    catalog=None,
):
    fingerprint = (
        None
        if catalog is None
        else {
            k: catalog[k]
            for k in ("models", "capacity_factors", "memory_table_mib", "safe_factor")
        }
    )
    payload = [
        deployment,
        {
            t: {m: float(q) for m, q in row.items() if q > 0}
            for t, row in routing.items()
        },
        {t: float(q) for t, q in rates.items()},
        float(horizon),
        seed,
        arrival_process,
        float(multiplier),
        fingerprint,
    ]
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if catalog is not None and catalog.get("performance_mode") == "profiled":
        # Stress scales the nominal service profile; bind that same evidence
        # into every scenario's energy identity, including non-unit stress.
        service_contract = list(payload)
        service_contract[6] = 1.0
        service_key = hashlib.sha256(
            json.dumps(service_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        payload.extend(
            ["profiled", catalog.get("colocation_profiles", {}).get(service_key)]
        )
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return key


def energy_norm(evidence, horizon, reference_watts, amortization=None):
    if not isinstance(evidence, dict) or evidence.get("scope") != "whole_gpu_board":
        return None
    if evidence.get("kind") not in {
        "measured",
        "synthetic_fixture",
    } or not evidence.get("coverage_valid"):
        return None
    values = [evidence.get("run_joules"), evidence.get("startup_joules")]
    if any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or v < 0
        for v in values
    ):
        return None
    return (
        values[0] / horizon + values[1] / (amortization or horizon)
    ) / reference_watts


def integrate_power(samples, start_ns, end_ns, max_gap_seconds=2.0):
    """Trapezoidal integral with interpolation, full coverage and gap checks."""
    points = sorted((s["time_ns"], s["watts"]) for s in samples)
    if end_ns <= start_ns or len(points) < 2:
        return None
    if any(not math.isfinite(w) or w < 0 for _, w in points):
        return None
    if points[0][0] > start_ns or points[-1][0] < end_ns:
        return None
    energy = 0.0
    for (a, pa), (b, pb) in zip(points, points[1:]):
        left, right = max(a, start_ns), min(b, end_ns)
        if right <= left:
            continue
        if b <= a or (b - a) / 1e9 > max_gap_seconds:
            return None
        pleft = pa + (pb - pa) * (left - a) / (b - a)
        pright = pa + (pb - pa) * (right - a) / (b - a)
        energy += (pleft + pright) / 2 * (right - left) / 1e9
    return energy
