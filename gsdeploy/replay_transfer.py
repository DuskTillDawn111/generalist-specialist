"""Zero-refit replay transfer with a new workload and independent validation.

The historical v2 spec is never edited or resealed. This separate envelope
copies its numerical parameters exactly and records a new validation identity.
"""
import copy
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from .calibration import DEFAULT_THRESHOLDS, _baseline, _baseline_identity, _deployment_key, _features, _import_source
from .calibration_v2 import input_service_ms, replay_input_ids, replay_path_statistics
from .evidence import digest
from .quality import quality_digest

MODE = "replay_transfer_v1"
SCHEMA = "gsdeploy.replay_transfer.v1"
PARAMETERS = ("service_multiplier", "power_coefficients", "startup_joules_per_worker", "memory_multiplier",
              "training_deployment_memory", "input_baselines", "input_order", "baseline_identities",
              "quality_document_sha256", "safe_factor", "measurement_context")
SCORING = ("workload", "weights", "latency_reference_ms", "gpu_memory_mib", "energy_reference_watts",
           "startup_amortization_seconds", "utility_schema", "drain_seconds", "arrival_process")


def seal(value, field="sha256"):
    return quality_digest({k: v for k, v in value.items() if k != field})


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


@lru_cache(maxsize=32)
def _verified_file(path, expected, stat_identity):
    # Reuse only while OS file identity/size/mtime/ctime all remain identical.
    # The expected SHA is part of the key; every new/changed file is rehashed.
    if digest(path) != expected:
        raise ValueError("TRANSFER_SOURCE_RECEIPT_CHANGED")
    return json.loads(Path(path).read_text())


def read_receipt(source):
    path = Path(source["path"])
    if not path.is_absolute():
        raise ValueError("TRANSFER_SOURCE_RECEIPT_CHANGED")
    info = path.stat()
    identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    return _verified_file(str(path), source["sha256"], identity)


def code_bindings():
    root = Path(__file__).resolve().parent
    names = ("replay_transfer.py", "calibration_v2.py", "calibration.py", "config.py", "performance.py",
             "prediction.py", "quality.py", "trace.py", "routing.py", "surrogate_bounds.py", "optimizer.py",
             "milp.py", "search_session.py")
    return [{"path": "gsdeploy/" + name, "sha256": digest(root / name)} for name in names]


def _contract(spec):
    return seal({k: v for k, v in spec.items() if k not in {"status", "version_sha256", "validation_protocol", "validation_report"}})


def prepare_transfer(source_config_path, workload, *, source_manifest_path=None,
                     development_seeds=(51,), validation_seeds=(61, 62), confirmation_seeds=(71, 72, 73)):
    """Create TRANSFER_UNVALIDATED config. No new observations or fitting."""
    source = receipt(source_config_path)
    original = read_receipt(source)
    from .calibration_v2 import validated_replay_spec
    from .config import SystemConfig

    old = SystemConfig(**original)
    basis = validated_replay_spec(old.catalog, old.workload)
    seeds = [list(x) for x in (development_seeds, validation_seeds, confirmation_seeds)]
    if len(seeds[1]) != 2 or len(set(sum(seeds, []))) != sum(map(len, seeds)):
        raise ValueError("TRANSFER_SEEDS_MUST_BE_DISJOINT_WITH_TWO_HOLDOUTS")
    if set(workload) != set(old.workload) or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in workload.values()):
        raise ValueError("TRANSFER_REQUIRES_SAME_TASKS_AND_POSITIVE_RATES")
    config = copy.deepcopy(original)
    config["workload"] = dict(workload)
    spec = {"schema": SCHEMA, "status": "TRANSFER_UNVALIDATED", "basis_config": source,
            "basis_version_sha256": basis["version_sha256"], "basis_status": basis["status"],
            "basis_workload": copy.deepcopy(old.workload), "workload": dict(workload),
            "source_manifest": receipt(source_manifest_path) if source_manifest_path else None,
            "code_bindings": code_bindings(), "thresholds": dict(DEFAULT_THRESHOLDS),
            "seed_roles": dict(zip(("development", "validation", "confirmation"), seeds)),
            "scoring_contract": {k: copy.deepcopy(config.get(k)) for k in SCORING},
            "parameters_fitted_on_new_runs": False,
            "scope": "Zero-refit frozen v2 mathematics; new workload validation required; no physical global optimum claim",
            **{k: copy.deepcopy(basis[k]) for k in PARAMETERS}}
    spec["version_sha256"] = seal(spec, "version_sha256")
    config["catalog"][MODE] = spec
    config["catalog"]["performance_mode"] = config["catalog"]["memory_mode"] = MODE
    SystemConfig(**config)
    return config


def _spec_transfer(data):
    spec = data.get(MODE, {})
    if (spec.get("schema") != SCHEMA or spec.get("version_sha256") != seal(spec, "version_sha256")
            or spec.get("parameters_fitted_on_new_runs") is not False
            or spec.get("thresholds") != DEFAULT_THRESHOLDS or spec.get("code_bindings") != code_bindings()):
        raise ValueError("TRANSFER_SPEC_OR_CODE_CHANGED")
    original = read_receipt(spec["basis_config"])
    if spec.get("source_manifest") is not None:
        manifest = read_receipt(spec["source_manifest"])
        source_name = Path(spec["basis_config"]["path"]).name
        if manifest.get("files", {}).get(source_name) != spec["basis_config"]["sha256"]:
            raise ValueError("TRANSFER_BASIS_MANIFEST_MISMATCH")
    from .calibration_v2 import validated_replay_spec
    basis = validated_replay_spec(original["catalog"], original["workload"])
    if (basis["version_sha256"] != spec.get("basis_version_sha256") or spec.get("basis_workload") != original["workload"]
            or spec.get("basis_status") != basis["status"] or any(spec.get(k) != basis[k] for k in PARAMETERS)):
        raise ValueError("TRANSFER_NUMERICAL_BASIS_CHANGED")
    return spec


def check_scoring(config):
    """Bind the full new scoring configuration, beyond a catalog-only gate."""
    spec = config.catalog[MODE]
    if any(getattr(config, k, None) != v for k, v in spec["scoring_contract"].items()):
        raise ValueError("TRANSFER_SCORING_CONTRACT_CHANGED")


def _checked_report(spec):
    protocol, report = read_receipt(spec["validation_protocol"]), read_receipt(spec["validation_report"])
    if (protocol.get("sha256") != seal(protocol) or report.get("sha256") != seal(report)
            or protocol.get("schema") != "gsdeploy.replay_transfer_protocol.v1"
            or report.get("schema") != "gsdeploy.replay_transfer_validation.v1"
            or report.get("status") != "VALIDATED_ON_NEW_WORKLOAD"
            or protocol.get("transfer_contract_sha256") != _contract(spec)
            or report.get("transfer_contract_sha256") != _contract(spec)
            or report.get("protocol_sha256") != protocol["sha256"]
            or report.get("parameters_fitted_on_new_runs") is not False
            or report.get("code_bindings") != spec["code_bindings"]
            or protocol.get("thresholds") != spec["thresholds"]):
        raise ValueError("TRANSFER_VALIDATION_REPORT_OR_PROTOCOL_CHANGED")
    if protocol.get("code_bindings") != spec["code_bindings"]:
        raise ValueError("TRANSFER_PROTOCOL_CODE_CHANGED")
    conditions = report.get("conditions", [])
    if len(conditions) != 2 or [c["id"] for c in conditions] != protocol["validation_ids"]:
        raise ValueError("TRANSFER_VALIDATION_ROSTER_CHANGED")
    if len({c["executed_measurement_id"] for c in conditions}) != 2:
        raise ValueError("TRANSFER_REPEATED_EXECUTION")
    packet = read_receipt(protocol["packet"])
    if [j["id"] for j in packet["jobs"]] != protocol["validation_ids"]:
        raise ValueError("TRANSFER_VALIDATION_PACKET_CHANGED")
    for condition in conditions:
        read_receipt(condition["source"])
        if (condition.get("passed") is not True or set(condition["errors"]) != set(DEFAULT_THRESHOLDS)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v > DEFAULT_THRESHOLDS[k]
                       or v < 0 for k, v in condition["errors"].items())):
            raise ValueError("TRANSFER_VALIDATION_THRESHOLDS_FAILED")
    a, b = conditions
    ordering = ((a["prediction"]["U"] - b["prediction"]["U"]) * (a["measured"]["U"] - b["measured"]["U"]) >= 0)
    if report.get("pair_ordering_consistent") is not True or not ordering:
        raise ValueError("TRANSFER_VALIDATION_RANKING_FAILED")
    return report


def validated_transfer_spec(data, rates, *, _allow_unvalidated=False):
    spec = _spec_transfer(data)
    if spec.get("status") != "VALIDATED_ON_NEW_WORKLOAD":
        if not (_allow_unvalidated and spec.get("status") == "TRANSFER_UNVALIDATED"):
            raise ValueError("TRANSFER_NOT_VALIDATED:" + str(spec.get("status")))
    else:
        _checked_report(spec)
    if (data.get("performance_mode") != MODE or data.get("memory_mode") != MODE
            or spec["workload"] != rates or spec["quality_document_sha256"] != quality_digest(data.get("quality", {}))
            or spec["input_order"] != data.get("quality", {}).get("input_order")
            or spec["safe_factor"] != data.get("safe_factor") or spec["measurement_context"] != data.get("measurement_context")
            or any(spec["baseline_identities"].get(m) != identity for m, identity in _baseline_identity(data).items())):
        raise ValueError("TRANSFER_INPUT_CONTEXT_OR_WORKLOAD_CHANGED")
    return spec


def transfer_service(data, deployment, routing, rates, *, _allow_unvalidated=False):
    try:
        spec = validated_transfer_spec(data, rates, _allow_unvalidated=_allow_unvalidated)
        service = {m: {} for m in deployment}
        for task, row in routing.items():
            for model, q in row.items():
                if q > 0:
                    _baseline(data, model, task, deployment[model])
                    service[model][task] = replay_path_statistics(spec, model, task, deployment[model])["mean_ms"]
        return service, {"mode": MODE, "kind": "zero_refit_replay_transfer", "version_sha256": spec["version_sha256"],
            "basis_version_sha256": spec["basis_version_sha256"], "validation_status": spec["status"],
            "diagnostic_only": bool(_allow_unvalidated), "hardware_capacity_certified": False,
            "scope": "Unchanged frozen per-input FIFO mathematics; separately bound new workload validation"}
    except (KeyError, TypeError, ValueError) as exc:
        return None, {"mode": MODE, "kind": "unknown", "reason": str(exc)}


def transfer_auxiliary(data, deployment, routing, rates, horizon, energy_reference_watts, amortization=None,
                       *, _allow_unvalidated=False):
    service, evidence = transfer_service(data, deployment, routing, rates, _allow_unvalidated=_allow_unvalidated)
    if service is None:
        return {"energy_norm": None, "safe_memory_mib": None, "evidence": evidence}
    spec = data[MODE]
    watts = float(np.dot(_features(data, deployment, routing), spec["power_coefficients"]))
    startup = spec["startup_joules_per_worker"] * sum(deployment.values())
    additive = data["safe_factor"] * math.fsum(data["memory_table_mib"][m][str(n)] for m, n in deployment.items())
    memory = max(additive * spec["memory_multiplier"], spec["training_deployment_memory"].get(_deployment_key(deployment), 0))
    return {"energy_norm": (watts + startup / (amortization or horizon)) / energy_reference_watts,
            "safe_memory_mib": memory, "evidence": {**evidence, "measured": False, "watts": watts,
            "startup_joules": startup, "coverage_valid": False, "memory_is_hardware_bound": False}}


def declare_validation(config, packet_path):
    """Bind exactly two future native validation jobs before looking at runs."""
    from .static_study import validate_packet
    check_scoring(config)
    spec = validated_transfer_spec(config.catalog, config.workload, _allow_unvalidated=True)
    if spec["status"] != "TRANSFER_UNVALIDATED":
        raise ValueError("TRANSFER_DECLARE_REQUIRES_UNVALIDATED")
    source, packet = receipt(packet_path), json.loads(Path(packet_path).read_text())
    if not validate_packet(packet)["valid"] or len(packet["jobs"]) != 2:
        raise ValueError("TRANSFER_REQUIRES_TWO_BOUND_VALIDATION_JOBS")
    jobs = packet["jobs"]
    if [j["seed"] for j in jobs] != spec["seed_roles"]["validation"]:
        raise ValueError("TRANSFER_VALIDATION_SEEDS_CHANGED")
    for job in jobs:
        if (job["phase"] != "calibration" or job.get("purpose") != "replay_transfer_independent_validation"
                or job["identity"]["workload"] != config.workload or job["identity"]["horizon_seconds"] != 600):
            raise ValueError("TRANSFER_VALIDATION_JOB_CONTRACT_CHANGED")
    if any(packet["config"].get(k) != spec["scoring_contract"][k] for k in SCORING):
        raise ValueError("TRANSFER_VALIDATION_SCORING_CONTRACT_CHANGED")
    if quality_digest(packet["config"]["catalog"]["quality"]) != spec["quality_document_sha256"]:
        raise ValueError("TRANSFER_VALIDATION_QUALITY_CHANGED")
    protocol = {"schema": "gsdeploy.replay_transfer_protocol.v1", "transfer_contract_sha256": _contract(spec),
        "packet": source, "validation_ids": [j["id"] for j in jobs], "validation_seeds": spec["seed_roles"]["validation"],
        "thresholds": dict(DEFAULT_THRESHOLDS), "code_bindings": spec["code_bindings"],
        "basis_version_sha256": spec["basis_version_sha256"], "parameters_fitted_on_new_runs": False,
        "accepted_runtime_versions": sorted({j["identity"]["runtime_code_sha256"] for j in jobs}),
        "scope": "Two predeclared new-load holdouts; neither development nor final confirmation"}
    protocol["sha256"] = seal(protocol)
    return protocol


def predict_validation(config, deployment, routing, *, horizon=600, seed=61):
    """Diagnostic only: no formal U or eligible status before validation."""
    from .prediction import predict
    check_scoring(config)
    return predict(config.catalog, deployment, routing, config.workload, config.gpu_memory_mib,
        config.latency_reference_ms, config.weights, horizon=horizon, seed=seed, arrival_process=config.arrival_process,
        drain_seconds=config.drain_seconds, energy_reference_watts=config.energy_reference_watts,
        startup_amortization_seconds=config.startup_amortization_seconds, _calibration_validation=True)


def validate_transfer(config, protocol, sources):
    """Strictly import two unseen runs; compute errors only, never fit parameters."""
    from .routing import Router
    from .trace import expand_routes, make_trace
    spec = validated_transfer_spec(config.catalog, config.workload, _allow_unvalidated=True)
    if (protocol.get("sha256") != seal(protocol) or protocol["transfer_contract_sha256"] != _contract(spec)
            or protocol.get("code_bindings") != code_bindings()):
        raise ValueError("TRANSFER_PROTOCOL_CHANGED")
    packet = read_receipt(protocol["packet"])
    if declare_validation(config, protocol["packet"]["path"]) != protocol:
        raise ValueError("TRANSFER_PROTOCOL_REBUILD_MISMATCH")
    by_id = {s["id"]: s for s in sources}
    if len(by_id) != len(sources) or set(by_id) != set(protocol["validation_ids"]):
        raise ValueError("TRANSFER_SOURCE_ROSTER_CHANGED")
    conditions, actual_ids = [], set()
    for key in protocol["validation_ids"]:
        source = by_id[key]
        if source.get("packet_path") != protocol["packet"]["path"] or source.get("packet_sha256") != protocol["packet"]["sha256"]:
            raise ValueError("TRANSFER_SOURCE_PACKET_CHANGED")
        raw, measured, _ = _import_source(source, set(protocol["accepted_runtime_versions"]))
        if (raw.get("experiment_job_id") != key or raw.get("experiment_phase") != "calibration"
                or raw["transport_config"]["random_seed"] != spec["seed_roles"]["validation"][len(conditions)]
                or raw["measurement_window"]["duration_seconds"] != 600):
            raise ValueError("TRANSFER_ACTUAL_VALIDATION_IDENTITY_CHANGED")
        if raw["executed_measurement_id"] in actual_ids:
            raise ValueError("TRANSFER_REPEATED_EXECUTION")
        actual_ids.add(raw["executed_measurement_id"])
        native = raw["config"]
        if (any(native.get(k) != spec["scoring_contract"][k] for k in SCORING)
                or _baseline_identity(native["catalog"]) != _baseline_identity(config.catalog)
                or quality_digest(native["catalog"]["quality"]) != spec["quality_document_sha256"]
                or native["catalog"]["safe_factor"] != spec["safe_factor"]
                or native["catalog"].get("capability_contract") != config.catalog.get("capability_contract")):
            raise ValueError("TRANSFER_ACTUAL_SCORING_OR_BASELINE_CHANGED")
        context = native["catalog"]["measurement_context"]
        if context["gpu_uuid"] != spec["measurement_context"]["gpu_uuid"]:
            raise ValueError("TRANSFER_GPU_CONTEXT_CHANGED")
        for m in raw["plan"]["deployment"]:
            if context["runtime_checkpoint_sha256"].get(m) != spec["measurement_context"]["runtime_checkpoint_sha256"].get(m):
                raise ValueError("TRANSFER_RUNTIME_CHECKPOINT_CHANGED:" + m)
        x, rows = raw["plan"]["deployment"], raw["plan"]["routing"]
        duration, seed = raw["measurement_window"]["duration_seconds"], raw["transport_config"]["random_seed"]
        pred = predict_validation(config, x, rows, horizon=duration, seed=seed)
        trace = make_trace(config.workload, duration, seed, config.arrival_process)
        aliases, routes = expand_routes(x, rows)
        router, expected_service = Router(routes), 0.
        for (_, task), sample in zip(trace, replay_input_ids(config.catalog, trace, seed)):
            m = aliases[router.choose(task)]
            expected_service += input_service_ms(spec, m, task, x[m], sample)
        actual_service = math.fsum(r["worker_latency_ms"] for r in raw["request_records"])
        predicted_u = pred.get("diagnostic_partial_U") if pred.get("constraints_eligible") and pred.get("utility_complete") else None
        def relative(a, b):
            return None if a is None or b is None else abs(a-b) / max(abs(b), 1e-9)
        def absolute(a, b):
            return None if a is None or b is None else abs(a-b)
        errors = {"service_relative_error": relative(expected_service, actual_service),
            "energy_relative_error": relative(pred.get("E_norm"), measured.get("E_norm")),
            "memory_relative_error": relative(pred.get("M_norm"), measured.get("M_norm")),
            "mean_relative_error": relative(pred.get("L_norm"), measured.get("L_norm")),
            "p95_relative_error": relative(pred.get("P_norm"), measured.get("P_norm")),
            "quality_absolute_error": absolute(pred.get("Q_norm"), measured.get("Q_norm")),
            "utility_absolute_error": absolute(predicted_u, measured.get("U"))}
        passed = all(v is not None and v <= DEFAULT_THRESHOLDS[k] for k, v in errors.items())
        conditions.append({"id": key, "seed": seed, "source": {"path": source["path"], "sha256": source["sha256"]},
            "executed_measurement_id": raw["executed_measurement_id"],
            "prediction": {"U": predicted_u, **{k+"_norm": pred.get(k+"_norm") for k in "QTLPME"}},
            "measured": {k: measured.get(k) for k in ["U"] + [t+"_norm" for t in "QTLPME"]},
            "errors": errors, "passed": bool(passed)})
    a, b = conditions
    ordering = (all(c["prediction"]["U"] is not None for c in conditions)
        and (a["prediction"]["U"] - b["prediction"]["U"]) * (a["measured"]["U"] - b["measured"]["U"]) >= 0)
    report = {"schema": "gsdeploy.replay_transfer_validation.v1", "transfer_contract_sha256": _contract(spec),
        "protocol_sha256": protocol["sha256"], "basis_version_sha256": spec["basis_version_sha256"],
        "code_bindings": spec["code_bindings"], "status": "VALIDATED_ON_NEW_WORKLOAD" if ordering and all(c["passed"] for c in conditions) else "TRANSFER_VALIDATION_FAILED",
        "parameters_fitted_on_new_runs": False, "conditions": conditions, "pair_ordering_consistent": bool(ordering),
        "gpu_started": False, "scope": "Independent new-workload transfer validation only; no physical all-pool optimality"}
    report["sha256"] = seal(report)
    return report


def activate_transfer(config, protocol_path, report_path):
    """Bind a persisted passing report. A failed report remains failed."""
    data = config.as_dict()
    spec = data["catalog"][MODE]
    spec["validation_protocol"], spec["validation_report"] = receipt(protocol_path), receipt(report_path)
    report = read_receipt(spec["validation_report"])
    spec["status"] = report.get("status", "TRANSFER_VALIDATION_FAILED")
    spec["version_sha256"] = seal(spec, "version_sha256")
    if spec["status"] == "VALIDATED_ON_NEW_WORKLOAD":
        validated_transfer_spec(data["catalog"], data["workload"])
    else:
        _spec_transfer(data["catalog"])
    return data
