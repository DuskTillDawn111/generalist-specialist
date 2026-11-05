"""CPU preparation and evidence inventory for the C1-Lite V2 task.

This module never starts workers or converts old total timings into stage laws.
The preparation packet lists unresolved collection prerequisites explicitly.
"""

import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess

from . import normalization as norm


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def hardware_status():
    command = shutil.which("nvidia-smi")
    result = dict(command=command, device_nodes=sorted(str(p) for p in Path("/dev").glob("nvidia*")),
                  usable=False, workers_started=False)
    if command is None:
        return dict(result, reason="NVIDIA_SMI_NOT_FOUND")
    try:
        proc = subprocess.run([command, "-L"], capture_output=True, text=True, timeout=5)
        result.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
                      usable=proc.returncode == 0 and bool(result["device_nodes"]))
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["reason"] = str(exc)
    return result


def inventory_run(path):
    raw = json.loads(Path(path).read_text())
    rows = raw.get("request_records", [])
    def ready(row):
        response = row.get("worker_response", {})
        return any(k in row or k in response for k in
                   ("worker_next_ready_ns", "worker_next_ready_time_ns", "ready_time_ns"))
    stages = sum(bool(r.get("segments") or r.get("worker_response", {}).get("segments")) for r in rows)
    ready_count = sum(ready(r) for r in rows)
    reasons = []
    if not rows:
        reasons.append("NO_REQUEST_RECORDS")
    if stages != len(rows):
        reasons.append("COMPLETE_ORDERED_STAGES_NOT_IN_LEGACY_FIELDS")
    if ready_count != len(rows):
        reasons.append("EXPLICIT_READY_NOT_IN_LEGACY_FIELDS")
    # Even presence of all fields would need the actual normalizer and
    # execution/clock/roster validation. Inventory never grants eligibility.
    return dict(**receipt(path), deployment=raw.get("plan", {}).get("deployment"),
        requests=len(rows), ordered_stage_fields=stages, explicit_ready_fields=ready_count,
        new_stage_capture_validation="NOT_PERFORMED_BY_FIELD_INVENTORY",
        matching_training_qualified=False, rejection_reasons=reasons,
        energy_scope=raw.get("energy_evidence", {}).get("scope"),
        role="historical_already_inspected_not_blind_confirmation")


def build_packet(config, *, hardware=None, historical_runs=()):
    c, data = config, config.catalog
    if not norm.is_v2(data):
        raise ValueError("C1_QUALIFICATION_REQUIRES_CURRENT_REFERENCE_V2")
    norm.validate(data, c.workload, c.latency_reference_ms, c.horizon_seconds, c.gpu_memory_mib)
    models = ("generalist_2", "image", "ocr")
    if any(m not in data["models"] for m in models) or "ocr" not in c.workload:
        raise ValueError("TARGETED_G2_IMAGE_OCR_DOMAIN_REQUIRED")
    deployment = dict.fromkeys(models, 1)
    for m in models:
        if "1" not in data["capacity_factors"][m]:
            raise ValueError("TARGET_DEPLOYMENT_STATE_UNSUPPORTED")
    scenarios = []
    def add(name, x, image_share, ocr_share, *, load="original", inputs="frozen_original"):
        shares = {t: {"generalist_2": 1.} for t in c.workload}
        for task, share in (("image", image_share), ("ocr", ocr_share)):
            if task not in shares:
                continue
            shares[task] = {"generalist_2": 1.-share, task: share}
        scenarios.append(dict(id=name, deployment=x, routing_shares=shares, load_class=load,
            workload_rps=copy_rates(c.workload) if load == "original" else None,
            input_stratum=inputs, input_manifest=None if inputs != "frozen_original" else data["quality"]["identity"],
            execution_context_sha256=None, execution_ready=False,
            prerequisites=["MATCHED_EXECUTION_STAGE_TEMPLATE", "COMPLETE_NATIVE_CAPTURE_AND_EPISODE"]+
                (["MEASURE_SAFE_CAPACITY_AND_DECLARE_LOAD"] if load != "original" else [])+
                (["FREEZE_OBSERVED_INPUT_DIFFICULTY_STRATA"] if inputs != "frozen_original" else [])))
    for image_share, ocr_share in itertools.product((0., 1.), repeat=2):
        add(f"pure-image{image_share:g}-ocr{ocr_share:g}", deployment.copy(), image_share, ocr_share)
    for share in (.25, .5, .75):
        add(f"ocr-internal-{share:g}", deployment.copy(), 1., share)
    add("image-ocr-joint-internal", deployment.copy(), .5, .5)
    add("remove-idle-ocr", {"generalist_2": 1, "image": 1}, 1., 0.)
    if "2" in data["capacity_factors"]["ocr"]:
        add("ocr-two-replicas", {**deployment, "ocr": 2}, 1., 1.)
    add("capacity-selected-ocr-hot", deployment.copy(), 1., .5, load="capacity_selected_hotspot")
    add("observed-harder-ocr-inputs", deployment.copy(), 1., .5, inputs="predeclared_harder_ocr")
    hw = hardware if hardware is not None else hardware_status()
    reference = norm.contract(data)
    missing = []
    if not hw.get("usable"):
        missing.append("GPU_UNAVAILABLE")
    if not norm.ready(data):
        missing.append("NORMALIZATION_REFERENCES_PENDING")
    missing.extend(["MATCHED_TRAINING_STAGE_AND_LIFECYCLE_PROFILES_REQUIRED",
                    "CAPTURE_PRODUCER_AND_STAGE_TEMPLATE_BINDING_REQUIRED",
                    "CAPACITY_AND_INPUT_STRATA_NOT_YET_MEASURED"])
    return dict(schema="gsdeploy.c1_qualification_packet.v1", status="PREPARED_NOT_EXECUTED",
        hardware=hw, blockers=missing, utility_schema=norm.SCHEMA,
        reference_sha256=norm.identity(data), reference_status=reference["status"],
        reference_pending_tasks=[t for t, row in reference["tasks"].items() if row.get("deadline_ms") is None],
        episode_reference_joules=reference.get("episode_reference_joules"),
        historical_inventory=[inventory_run(p) for p in historical_runs], scenarios=scenarios,
        data_splits=dict(training_arrival_seeds=[2026110301, 2026110307, 2026110311],
            confirmation_arrival_seeds=[2026110319, 2026110323],
            roles_declared_before_collection=True, confirmation_used_for_selection=False,
            seeds_are_reserved_not_collected=True),
        budget=dict(pilot_max_runs=3, pilot_arrival_seconds=30., pilot_max_episode_seconds=300.,
                    pilot_total_gpu_seconds=900., qualification_total_gpu_seconds=3600.,
                    qualification_subset_must_fit_budget=True,
                    full_matrix_is_staged_not_cartesian_execution=True),
        measurement_protocol=dict(qualification_horizon_seconds=c.horizon_seconds,
            drain_seconds=c.drain_seconds, minimum_independent_training_runs_per_context=2,
            complete_cohort=True, no_handler_to_gpu_stage_inference=True,
            total_power_is_not_sum_of_isolated_models=True, replica_energy_monotonicity_assumed=False,
            pilot_U=None, pilot_reason="short_pilot_does_not_match_formal_reference_horizon"),
        qualification_decision=dict(status="NOT_RUN", service_accuracy=None, fifo_accuracy=None,
            episode_energy_accuracy=None, paired_route_delta_U=None, ranking=None,
            thresholds="DECLARE_FROM_DEVELOPMENT_DATA_BEFORE_CONFIRMATION",
            native_eligible=False, independent_accuracy_qualified=False, final_gain_eligible=False),
        next_steps=["Bind an actual complete-stage capture producer and matched solo template",
            "Run bounded capacity/clock/cohort/energy pilot only when prerequisites hold",
            "Choose a budget-fitting subset; freeze reference and training/confirmation manifests",
            "Normalize/fit training only; verify held-out services, FIFO, episode energy and paired route ranking"],
        downstream_commands=["python experiments/normalize_paper_capture.py --help",
            "python experiments/fit_paper_predictor.py --help",
            "python -m gsdeploy.reference_freeze --help"],
        collector_implemented_by_this_packet=False, GPU_started=False, algorithm_comparison_qualified=False)


def copy_rates(rates):
    return {t: float(q) for t, q in rates.items()}


def paired_route_evidence(pairs, minimum_gain=0., confidence=.95):
    """Paired independent-run interval; never treat requests as repetitions.

    Caller must supply preselected fixed route comparisons. This checks
    provenance/identity and reports a t interval, not universal accuracy.
    """
    from scipy.stats import t
    if (not math.isfinite(minimum_gain) or minimum_gain < 0
            or not 0 < confidence < 1 or len(pairs) < 2):
        raise ValueError("INDEPENDENT_PAIRED_RUNS_AND_VALID_MARGIN_REQUIRED")
    seeds, differences = set(), []
    identity = None
    for row in pairs:
        if (row.get("kind") != "measured" or row.get("split") != "confirmation"
                or row.get("used_for_selection") is not False
                or not row.get("execution_sha256") or not row.get("normalization_sha256")
                or not row.get("trace_sha256") or not row.get("comparison_sha256")
                or row.get("seed") in seeds):
            raise ValueError("FIXED_MATCHED_INDEPENDENT_CONFIRMATION_REQUIRED")
        current = tuple(row[k] for k in ("execution_sha256", "normalization_sha256", "comparison_sha256"))
        if identity is not None and current != identity:
            raise ValueError("PAIRED_COMPARISON_IDENTITY_CHANGED")
        identity = current; seeds.add(row["seed"])
        values = row.get("candidate_U"), row.get("baseline_U")
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError("COMPLETE_FINITE_UTILITY_REQUIRED")
        differences.append(values[0]-values[1])
    mean = math.fsum(differences)/len(differences)
    std = math.sqrt(math.fsum((v-mean)**2 for v in differences)/(len(differences)-1))
    margin = float(t.ppf((1+confidence)/2, len(differences)-1))*std/math.sqrt(len(differences))
    return dict(pairs=len(pairs), mean_delta_U=mean, interval=[mean-margin, mean+margin],
                confidence=confidence, minimum_gain=minimum_gain,
                stable_gain_supported=mean-margin > minimum_gain,
                scope="fixed preselected paired conditions; source claims still require raw-record validation",
                physical_qualification_granted=False)
