"""CPU-only preregistration of gated mechanism and paired four-group studies.

There is deliberately no execute command. This writes a finite review domain,
local seed audit/reservations, and NOT_RUN pair descriptors. It never converts
missing queue/P95/delta/ranking evidence into permission to launch native work.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from gsdeploy.stability import DEFAULT_THRESHOLDS
from .profiling import RESIDENTS, digest


DEV_SEEDS = (627101, 627102, 627103, 627104)
CONFIRM_SEEDS = (627201, 627202, 627203, 627204, 627205, 627206, 627207, 627208)
ROOT = Path(__file__).resolve().parents[2]


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def seed_audit(root, development=DEV_SEEDS, confirmation=CONFIRM_SEEDS):
    """Search seed-named JSON fields, not milliseconds, request IDs or labels.

    rg limits content output to filenames. Only hit files are parsed for exact
    integer/list seed values, preserving multiline seed arrays. No weights or
    binary formats are opened. Scope is local repository metadata, not remote
    jobs or an external reservation service.
    """
    root = Path(root).resolve()
    proposed = set(development) | set(confirmation)
    if len(proposed) != len(development) + len(confirmation):
        raise ValueError("Development and confirmation seeds must be unique and disjoint")
    wanted = proposed | {501, 502}
    alternatives = "|".join(str(s) for s in sorted(wanted))
    field = r'"[^"\r\n]*seed[^"\r\n]*"[ \t\r\n]*:[ \t\r\n]*'
    pattern = field + r'(?:' + alternatives + r')(?:[^0-9]|$)|' + field + r'\[[0-9, \t\r\n]*\b(?:' + alternatives + r')\b[0-9, \t\r\n]*\]'
    roots = [name for name in ("configs", "artifacts", "tools") if (root / name).is_dir()]
    if not roots:
        raise ValueError("Repository metadata roots required")
    command = ["rg", "-l", "--multiline", "--glob", "*.json", "--glob", "!**/source-snapshot/**",
               "--glob", "!**/source-qa/**", "--glob", "!**/site-packages/**", pattern, *roots]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=180)
    if result.returncode not in (0, 1):
        raise RuntimeError("Seed audit search failed: " + result.stderr)
    records = []
    value_pattern = re.compile(field + r'([0-9]+|\[[0-9, \t\r\n]*\])')
    for name in sorted(result.stdout.splitlines()):
        path = root / name
        text = path.read_text()
        for match in value_pattern.finditer(text):
            value = json.loads(match.group(1))
            seeds = value if isinstance(value, list) else [value]
            hits = sorted(set(seeds) & wanted)
            if hits:
                records.append({"source": receipt(path), "line": text.count("\n", 0, match.start()) + 1,
                                "matched_field": match.group(0), "seeds": hits})
    used = {seed for record in records for seed in record["seeds"]}
    conflicts = sorted(proposed & used)
    return {"schema": "gsdeploy.local_seed_audit.v1", "created_perf_ns": time.perf_counter_ns(),
            "scope": "Local configs/artifacts/tools JSON seed-named integer/list fields; binaries/weights excluded",
            "search_command": command, "records": records,
            "proposed_development_seeds": list(development), "proposed_confirmation_seeds": list(confirmation),
            "proposed_conflicts": conflicts,
            "new_seeds_status": "CONFLICT" if conflicts else "NO_PRIOR_LOCAL_SEED_FIELD_FOUND",
            "historical_501_502": {"source_field_found": {str(s): s in used for s in (501, 502)},
                                    "reuse": False,
                                    "interpretation": "Existing resource-study CPU draft and NOT_RUN packet; retain historical placeholder status"},
            "limits": ["Does not establish absence from external jobs or untracked non-JSON logs",
                       "A local protocol reservation is not a GPU execution or globally atomic reservation"]}


def candidate_domain():
    aliases = list(RESIDENTS)
    resources = [{"id": "shared", "mode": "shared", "allocations": dict.fromkeys(aliases, {})}]
    resources += [{"id": "admission_max" + str(limit), "mode": "request_admission",
                   "allocations": {a: {"max_inflight": 1} for a in aliases},
                   "groups": {"all_residents": {"aliases": aliases, "max_active": limit}}}
                  for limit in (1, 2)]
    policies = [{"id": "no_spill_full_horizon", "allow_spill": False,
                 "kind": "full_horizon_specialist_only", "parameters": {}}]
    policies += [{"id": "no_spill_swrr_" + str(numerator), "allow_spill": False,
                  "kind": "specialist_swrr", "parameters": {"text#0_fraction": numerator / 4,
                                                              "text#1_fraction": 1 - numerator / 4}}
                 for numerator in (1, 2, 3)]
    policies += [{"id": "allow_spill_cap_" + str(denominator), "allow_spill": True,
                  "kind": "full_horizon_utility", "parameters": {"exposure_fraction_cap": 1 / denominator,
                    "exposure_rate_rps": 1.85 / denominator, "exposure_burst": 1,
                    "utility_margin": 0., "cooldown_ms": 30000., "drift_ratio_limit": 2.}}
                 for denominator in (16, 8, 4)]
    candidates = []
    for resource in resources:
        for policy in policies:
            row = {"deployment": {"text": 2, "generalist_2": 1, "ocr": 1},
                   "resource": resource, "routing": policy,
                   "backend": "independent_native_observed_workers_batch1_no_weight_reuse",
                   "batch_size": 1, "weight_reuse": False}
            row["id"] = digest(row)
            candidates.append(row)
    return candidates


def prepare(output, *, root=ROOT, predictor_report=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite future protocol")
    audit = seed_audit(root)
    if audit["proposed_conflicts"]:
        raise ValueError("Proposed local seeds already used/reserved: " + str(audit["proposed_conflicts"]))
    objective_path = root / "configs/resource_spillover_objective_20261024.json"
    design_path = root / "configs/resource_spillover_study_20261024.json"
    objective, design = json.loads(objective_path.read_text()), json.loads(design_path.read_text())
    system_path = (design_path.parent / design["system_config"]).resolve()
    system = json.loads(system_path.read_text())
    catalog_path = Path(system["catalog"])
    if not catalog_path.is_absolute():
        catalog_path = system_path.parent / catalog_path
    catalog = json.loads(catalog_path.read_text())
    prediction = json.loads(Path(predictor_report).read_text()) if predictor_report else None
    candidates = candidate_domain()
    groups = {"original_no_spill": [c["id"] for c in candidates if c["resource"]["id"] == "shared" and not c["routing"]["allow_spill"]],
              "original_allow_spill": [c["id"] for c in candidates if c["resource"]["id"] == "shared" and c["routing"]["allow_spill"]],
              "optimized_no_spill": [c["id"] for c in candidates if not c["routing"]["allow_spill"]],
              "optimized_allow_spill": [c["id"] for c in candidates if c["routing"]["allow_spill"]]}
    blocked = ["PROSPECTIVE_SERVICE_P95_VALIDATION_NOT_PASSED", "FULL_ARRIVAL_QUEUE_VALIDATION_NOT_RUN",
               "PAIRED_DELTA_U_SIGN_AND_CANDIDATE_RANK_VALIDATION_NOT_RUN", "RESOURCE_CONTEXT_PROFILES_REQUIRED",
               "RUNTIME_DECISION_OVERHEAD_AND_OBSERVATION_SUPPORT_REQUIRED", "DEVELOPMENT_SELECTION_NOT_RUN"]
    protocol = {
        "schema": "gsdeploy.gated_future_paired_protocol.v1", "status": "NOT_RUN", "created_perf_ns": time.perf_counter_ns(),
        "GPU_started": False, "native_execution_enabled": False, "automatic_launch_supported": False,
        "sources": {"objective": receipt(objective_path), "design": receipt(design_path), "system": receipt(system_path),
                    "catalog": receipt(catalog_path), "protocol_generator": receipt(__file__),
                    "stability_implementation": receipt(root / "gsdeploy/stability.py"),
                    "memory_implementation": receipt(root / "gsdeploy/runtime/runner.py")},
        "objective": objective, "workload_rps": design["workload_rps"], "latency_reference_ms": design["latency_reference_ms"],
        "measurement": {"warmup_seconds": 30, "window_seconds": 600, "bounded_drain_seconds": 300,
                        "arrival_process": "poisson", "startup_amortization_seconds": system["startup_amortization_seconds"],
                        "energy_reference_watts": system["energy_reference_watts"]},
        "resource_and_stability": {
            "gpu_memory_mib": system["gpu_memory_mib"], "catalog_safe_factor": catalog["safe_factor"],
            "safe_memory_formula": "catalog.safe_factor * max(before.owned_used_mib,after.owned_used_mib,*monitor.samples.owned_used_mib)",
            "M_formula": "safe_memory_mib / gpu_memory_mib",
            "memory_scope": "Owned workers including idle residents; raw sample peak, factor applied once; no foreign processes",
            "unmeasured_memory": "INVALID_RESOURCE_EVIDENCE; no nominal substitute for final GPU gain claim",
            "rho_max": .8, "rho_policy": "Retain existing per-worker system admission under matching resource-specific evidence",
            "stability_thresholds": DEFAULT_THRESHOLDS,
            "eligible_stability_states": ["steady", "recoverable_congested"],
            "no_new_accuracy_threshold": True, "no_new_stability_exception": True},
        "candidate_domain": candidates, "candidate_count": len(candidates), "group_candidate_ids": groups,
        "resource_modes": {"active_candidate_modes": ["shared", "request_admission_max1", "request_admission_max2"],
                           "MPS": "DISABLED_NO_VERIFIED_PLATFORM_ADAPTER",
                           "green_context": "DISABLED_NO_VERIFIED_PLATFORM_ADAPTER",
                           "joint_batch": "DISABLED_NO_REAL_EXECUTOR", "weight_submatrix_cache": "DISABLED_NO_REAL_EXECUTOR",
                           "assumed_batch_or_cache_discount": 0.,
                           "admission_semantics": "Nonpreemptive whole-request sender FIFO, all four resident aliases covered; not reserved GPU resources"},
        "pairing": {
            "cohort": "One declared arrival+public-input trace per phase/seed; identical bytes for every candidate and four confirmation groups",
            "identity": ["phase", "seed", "offered_trace_sha256", "public_input_cohort_sha256", "warmup_trace_sha256"],
            "arrival_clock": "Scheduled arrivals remain fixed regardless of runtime delay; no drop/pace-to-completion",
            "background_routes": design["background_routing"],
            "predictor_visibility": "Known distribution/public pools and observed prefix only; actual future trace opaque to online routing",
            "order": "Development: deterministic balanced reversal by seed; confirmation: four cyclic orders repeated twice across eight seeds",
            "priming": "Same declared priming policy, inputs and physical resident set in every branch; complete before warmup",
            "failure_policy": "No selective retry/drop; retain failures and incomplete cohorts; protocol amendment before any new seed"},
        "development": {
            "phase": "development", "seeds": list(DEV_SEEDS), "seed_status": "LOCALLY_RESERVED_BY_THIS_PROTOCOL_NOT_RUN",
            "finite_candidates": len(candidates), "planned_candidate_seed_pairs": len(candidates) * len(DEV_SEEDS),
            "selection": "Evaluate the same finite execution domain fairly in both routing branches; maximize mean complete six-term U across all four development seeds",
            "tie_break": "Lexicographic candidate id after exact mean U tie; failures cannot be omitted",
            "fairness": "Same model fitting/profile/CPU search budgets and all shared/admission resource choices for no-spill and allow-spill; no-spill also gets SWRR fractions and full-horizon specialist-only policy",
            "matched_ablation": "Within each resource point compare allow/no spill on identical development traces before attributing gains to resource optimization",
            "freeze_before_confirmation": ["four winning candidate IDs", "model bytes", "source/context bytes", "all routing parameters", "complete development report"]},
        "confirmation": {
            "phase": "confirmation", "seeds": list(CONFIRM_SEEDS), "seed_status": "LOCALLY_RESERVED_BLINDED_NOT_RUN",
            "groups": list(groups), "planned_pairs": 8, "planned_group_seed_runs": 32,
            "primary_estimand": "U(optimized_allow_spill_winner)-U(optimized_no_spill_winner), paired within seed",
            "secondary_estimands": ["original allow minus original no-spill", "optimized no-spill minus original no-spill", "optimized allow minus original allow"],
            "independence": "No confirmation seed/output used for training, resource selection, threshold tuning or model repair",
            "positive_result": "Report all eight paired deltas and their mean; do not force a spill or label incomplete/unstable/resource-invalid runs as positive",
            "uncertainty": "Seed is independent unit; descriptive paired seed bootstrap 95% CI with seed resampling, not per-request pseudo-replication",
            "inferential_claim": "No net-gain claim unless all primary paired runs valid and predeclared paired-seed CI lower bound >0; otherwise report inconclusive/nonpositive",
            "CI_method": {"resamples": 10000, "percentiles": [2.5, 97.5], "rng_namespace": "confirmation-analysis-bootstrap-v1", "unit": "seed"},
            "rerun_policy": "No early stopping, seed replacement or post-confirmation tuning; any new model starts a fresh prospective confirmation protocol"},
        "gates": {"service_and_probe_P95": {"status": prediction["status"] if prediction else "NOT_RUN",
                    "report": receipt(predictor_report) if predictor_report else None, "required": "PASS with 100% declared request and task coverage"},
                  "full_queue_and_full_window_P95": {"status": "NOT_RUN", "required": "Independent full-arrival rollout validation for each eligible execution context"},
                  "paired_delta_and_ranking": {"status": "NOT_RUN", "required": "Frozen-protocol independent six-term delta sign and candidate ranking acceptance"},
                  "mechanism_backend": {"status": "NOT_RUN", "required": "Actual configuration binding and overhead/numerical checks; no simulated backend discounts"}},
        "launch_blockers": blocked, "executable_jobs": [],
        "release_rule": "All relevant gates PASS and source-bound concrete packet prepared; this protocol has no launcher and never promotes itself",
        "scope": "Finite fixed-residency mechanism and spill study only; not global deployment optimum or physical DFS certificate"}
    output.mkdir(parents=True, exist_ok=False)
    save(output / "seed-audit.json", audit)
    reservation = {"schema": "gsdeploy.local_future_seed_reservation.v1", "status": "RESERVED_NOT_RUN",
                   "development_seeds": list(DEV_SEEDS), "confirmation_seeds": list(CONFIRM_SEEDS),
                   "excluded_historical_seeds": [501, 502], "audit": receipt(output / "seed-audit.json"),
                   "GPU_started": False, "automatic_launch_supported": False}
    save(output / "seed-reservation.json", reservation)
    save(output / "protocol.json", protocol)
    descriptors = [{"phase": "confirmation", "seed": seed, "group": group, "candidate_id": None,
                    "status": "NOT_RUN", "launch_blockers": blocked, "offered_trace_sha256": None,
                    "public_input_cohort_sha256": None, "U": None}
                   for seed in CONFIRM_SEEDS for group in groups]
    save(output / "four-group-descriptors.json", {"status": "NOT_RUN", "descriptors": descriptors,
                                                  "executable_jobs": [], "GPU_started": False})
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--predictor-report")
    args = parser.parse_args(argv)
    protocol = prepare(args.output, root=args.root, predictor_report=args.predictor_report)
    print(json.dumps({"status": protocol["status"], "candidates": protocol["candidate_count"],
                      "executable_jobs": 0, "GPU_started": False, "output": str(Path(args.output).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
