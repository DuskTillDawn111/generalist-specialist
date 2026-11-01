"""Whole-plan energy records, tied to the actual planning contract.

No additive per-model power assumption or implicit missing=zero conversion.
"""

import hashlib
import json
import math


def deployed_measurement_identity(catalog, deployment, routing):
    """Bind the actual deployment and its input/protocol identities across pools.

    Quality values and offline predictions are not physical configuration. Their
    immutable input/reference bindings and checkpoint/protocol receipts are.
    Global catalog hashes and source-directory labels are intentionally omitted.
    """
    active_tasks = {t for t, row in routing.items() if any(q > 0 for q in row.values())}
    quality = catalog.get("quality", {})
    identity = quality.get("identity", {})
    paths = quality.get("paths", {})
    context = catalog.get("measurement_context")
    if isinstance(context, dict) and isinstance(context.get("runtime_checkpoint_sha256"), dict):
        context = {**context, "runtime_checkpoint_sha256": {
            m: context["runtime_checkpoint_sha256"][m]
            for m in sorted(deployment) if m in context["runtime_checkpoint_sha256"]
        }}
    primed_paths = {}
    # Largest-three priming exercises every supported active workload task on
    # each resident worker, including paths with no measured-window routing.
    for model in sorted(deployment):
        for task in sorted(active_tasks):
            if task not in catalog["models"][model]["supported_tasks"]:
                continue
            record = paths.get(model, {}).get(task, {})
            checkpoint = record.get("checkpoint_identity", {})
            primed_paths.setdefault(model, {})[task] = {
                "checkpoint_sha256": record.get("checkpoint_sha256"),
                "checkpoint_identity": {k: checkpoint[k] for k in ("kind", "sha256") if k in checkpoint},
                # Legacy aggregate fixtures may provide only a shared protocol
                # identity; real retained paths bind the task's own protocol.
                "protocol_sha256": record.get("protocol_sha256", identity.get("protocol_sha256")),
                "reference_sha256": record.get("reference_sha256"),
            }
    return {
        "schema": "deployed_measurement_contract_v3",
        "models": {m: catalog["models"][m] for m in sorted(deployment)},
        "capacity_factors": {m: {str(n): catalog["capacity_factors"][m][str(n)]}
                             for m, n in sorted(deployment.items())},
        "memory_table_mib": {m: {str(n): catalog["memory_table_mib"][m][str(n)]}
                             for m, n in sorted(deployment.items())},
        "safe_factor": catalog["safe_factor"],
        "measurement_context": context,
        "task_replica_profiles": {
            m: {t: {str(n): catalog.get("task_replica_profiles", {}).get(m, {}).get(t, {}).get(str(n))}
                for t in sorted(active_tasks) if routing[t].get(m, 0) > 0}
            for m, n in sorted(deployment.items())
        },
        "quality_inputs": {
            "input_pool_sha256": identity.get("input_pool_sha256"),
            "reference_sha256": identity.get("reference_sha256"),
            "input_order": {t: quality.get("input_order", {}).get(t) for t in sorted(active_tasks)},
        },
        "primed_checkpoint_protocols": primed_paths,
    }


def energy_key(
    deployment,
    routing,
    rates,
    horizon,
    seed,
    arrival_process,
    multiplier=1.0,
    catalog=None,
    resource_configuration=None,
):
    fingerprint = (
        None
        if catalog is None
        else {
            k: catalog[k]
            for k in ("models", "capacity_factors", "memory_table_mib", "safe_factor")
        }
    )
    if catalog is not None and catalog.get("capability_contract") == "verified_executable_v1":
        # New studies identify a measurement by deployed configurations, not by
        # unrelated alternatives in the search pool. Keep historical keys intact.
        fingerprint = deployed_measurement_identity(catalog, deployment, routing)
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
    if resource_configuration is not None:
        # An explicit resource mode is a new physical experiment identity.
        # Never borrow whole-board energy from the old unbound deployment.
        key = hashlib.sha256(json.dumps(
            [key, "resource_configuration_v1", resource_configuration],
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
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
