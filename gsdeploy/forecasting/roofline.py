"""Operator-level arithmetic-intensity candidates, independent of counters.

Useful when performance counters are restricted, but deliberately a structural
proxy. Declared tensor traffic is not measured DRAM traffic: L2 reuse, spills,
fusion, quantization and launch overhead can change the bottleneck.
"""
from __future__ import annotations

import math

from .workload import number


def classify_operator_work(configuration, hardware, *, separation_ratio=1.2):
    required = ("configuration_sha256", "hardware_id", "operation_evidence", "precision_signature",
                "flops", "declared_traffic_bytes", "traffic_assumption")
    if any(configuration.get(k) is None for k in required):
        return dict(classification="unknown", reason="OPERATOR_WORK_EVIDENCE_REQUIRED", measured=False)
    if (configuration["hardware_id"] != hardware.get("hardware_id")
            or configuration["precision_signature"] != hardware.get("precision_signature")
            or not hardware.get("peak_evidence") or not configuration["operation_evidence"]
            or not configuration["traffic_assumption"] or not configuration["configuration_sha256"]):
        return dict(classification="unknown", reason="CONFIGURATION_HARDWARE_OR_PRECISION_MISMATCH", measured=False)
    if hardware.get("peak_flops_per_second") is None or hardware.get("peak_dram_bytes_per_second") is None:
        return dict(classification="unknown", reason="MATCHED_PRECISION_PEAK_EVIDENCE_REQUIRED", measured=False)
    ratio = number(separation_ratio, "classification ratio", positive=True)
    if ratio <= 1:
        raise ValueError("Classification ratio must exceed one")
    flops = number(configuration["flops"], "operator FLOPs")
    traffic = number(configuration["declared_traffic_bytes"], "declared tensor traffic", positive=True)
    peak_compute = number(hardware["peak_flops_per_second"], "matched-precision compute peak", positive=True)
    peak_memory = number(hardware["peak_dram_bytes_per_second"], "DRAM peak", positive=True)
    compute_seconds, memory_seconds = flops / peak_compute, traffic / peak_memory
    classification = ("compute_proxy" if compute_seconds > ratio * memory_seconds else
                      "bandwidth_proxy" if memory_seconds > ratio * compute_seconds else "mixed_proxy")
    return dict(classification=classification, arithmetic_intensity_flops_per_byte=flops / traffic,
                device_balance_flops_per_byte=peak_compute / peak_memory,
                ideal_compute_seconds=compute_seconds, ideal_traffic_seconds=memory_seconds,
                roofline_lower_seconds=max(compute_seconds, memory_seconds),
                configuration_sha256=configuration["configuration_sha256"], hardware_id=configuration["hardware_id"],
                traffic_assumption=configuration["traffic_assumption"], measured=False,
                calibrated=False, pruning_allowed=False, throughput_prediction_qualified=False,
                evidence=[configuration["operation_evidence"], hardware["peak_evidence"]])


def complementary_operator_candidates(configurations, hardware):
    """Order complementary structural pairs; never delete the other pairs."""
    classified = [(c, classify_operator_work(c, hardware)) for c in configurations]
    pairs = []
    for i, (left, a) in enumerate(classified):
        for right, b in classified[i + 1:]:
            pairs.append(dict(configurations=[left.get("configuration_sha256"), right.get("configuration_sha256")],
                              classifications=[a["classification"], b["classification"]],
                              complementary_proxy=set((a["classification"], b["classification"])) == {"compute_proxy", "bandwidth_proxy"},
                              pruning_allowed=False, physical_interference_reduction_proven=False))
    return sorted(pairs, key=lambda p: not p["complementary_proxy"])
