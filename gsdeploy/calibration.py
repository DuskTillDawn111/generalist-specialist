"""Small, explicit mixed-deployment surrogate fitted to native calibration runs.

This model transfers standalone service curves. A finite-domain certificate is
about this frozen mathematical predictor, never about every unmeasured GPU run.
Whole-board power is regressed directly; it is not a sum of model powers.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from .evidence import digest
from .quality import quality_digest

SCHEMA = "gsdeploy.calibrated_surrogate.v1"
DEFAULT_THRESHOLDS = {"service_relative_error": .30, "energy_relative_error": .25,
                      "memory_relative_error": .25, "mean_relative_error": .35,
                      "p95_relative_error": .40, "quality_absolute_error": .05,
                      "utility_absolute_error": .10}


def _number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _baseline(data, model, task, n):
    entry = data.get("task_replica_profiles", {}).get(model, {}).get(task, {}).get(str(n), {})
    value = entry.get("mean_service_ms")
    if not _number(value, positive=True):
        raise ValueError(f"MISSING_TASK_REPLICA_PROFILE:{model}/{task}/{n}")
    if not math.isclose(data["capacity_factors"][model][str(n)], n):
        raise ValueError("Calibrated transfer requires task-replica capacity factor n")
    return float(value)


def _baseline_identity(data):
    # The fitted model is reusable across subsets of the same frozen pool.
    return {m: quality_digest({"model": spec,
                "states": data["task_replica_profiles"].get(m),
                "memory": data["memory_table_mib"][m],
                "capacity": data["capacity_factors"][m]})
            for m, spec in data["models"].items()}


def _features(data, deployment, routing):
    demand = math.fsum(q * _baseline(data, m, task, deployment[m]) / 1000
                       for task, row in routing.items() for m, q in row.items() if q > 0)
    return np.asarray([1.0, demand, float(sum(deployment.values()))])


def _deployment_key(deployment):
    return quality_digest(deployment)


def _spec(data):
    spec = data.get("calibrated_surrogate", {})
    if spec.get("schema") != SCHEMA:
        raise ValueError("MISSING_CALIBRATED_SURROGATE")
    payload = {k: v for k, v in spec.items() if k != "version_sha256"}
    if spec.get("version_sha256") != quality_digest(payload):
        raise ValueError("CALIBRATED_SURROGATE_VERSION_MISMATCH")
    if (not _number(spec.get("service_multiplier"), positive=True)
            or not _number(spec.get("memory_multiplier"), positive=True)
            or spec["memory_multiplier"] < 1
            or not _number(spec.get("startup_joules_per_worker"))
            or not isinstance(spec.get("power_coefficients"), list)
            or len(spec["power_coefficients"]) != 3
            or not all(_number(v) for v in spec["power_coefficients"])):
        raise ValueError("INVALID_CALIBRATED_SURROGATE_PARAMETERS")
    return spec


def validated_spec(data, rates, *, _allow_unvalidated=False):
    """One gate for scoring and mathematical bounds on a frozen predictor.

    A failed/unchecked predictor may be inspected, but cannot supply a formal
    score or search bound. The private override is used only while calculating
    predeclared held-out diagnostics, before a validation decision exists.
    """
    spec = _spec(data)
    if spec.get("status") != "VALIDATED_ON_HELDOUT_CONDITIONS" and not (
            _allow_unvalidated and spec.get("status") == "FITTED_UNVALIDATED"):
        raise ValueError("CALIBRATED_SURROGATE_NOT_VALIDATED:" + str(spec.get("status")))
    for model, current in _baseline_identity(data).items():
        if current != spec["baseline_identities"].get(model):
            raise ValueError("CALIBRATED_BASELINE_IDENTITY_MISMATCH:" + model)
    if (data.get("quality", {}).get("identity") != spec.get("quality_identity")
            or quality_digest(data.get("quality", {})) != spec.get("quality_document_sha256")):
        raise ValueError("CALIBRATED_QUALITY_INPUT_IDENTITY_MISMATCH")
    if (data.get("measurement_context") or {}) != spec.get("measurement_context"):
        raise ValueError("CALIBRATED_MEASUREMENT_CONTEXT_MISMATCH")
    if rates != spec["workload"]:
        raise ValueError("CALIBRATED_WORKLOAD_OUTSIDE_DECLARED_SCOPE")
    return spec


def declare_protocol(base_config, packet, training_labels, validation_labels, thresholds=None):
    """Declare a data-independent split before observing held-out outcomes."""
    labels = {label: job["id"] for job in packet["jobs"] for label in job["labels"]}
    train = [labels[label] for label in training_labels]
    holdout = [labels[label] for label in validation_labels]
    if len(train) != len(set(train)) or len(holdout) != len(set(holdout)) or set(train) & set(holdout):
        raise ValueError("Protocol conditions must be unique and disjoint")
    limits = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    protocol = {"schema": "gsdeploy.calibration_protocol.v1",
        "base_problem_sha256": quality_digest(base_config.as_dict()),
        "training_ids": train, "validation_ids": holdout,
        "training_labels": list(training_labels), "validation_labels": list(validation_labels),
        "thresholds": limits, "feature_hierarchy": ["board_intercept", "baseline_demand", "worker_count"],
        "max_parameters_rule": "at most n_training/2 and full column rank; holdout unused for selection",
        "service_rule": "one global ratio of sums, baseline matched to each observed path-state request",
        "memory_rule": "max(additive standalone residency times >=1 global factor, seen deployment peak)",
        "scope": "predictor finite-domain optimum; physical all-combination optimum not certified"}
    protocol["sha256"] = quality_digest(protocol)
    return protocol


def surrogate_service(data, deployment, routing, rates, *, _allow_unvalidated=False):
    """Return family-normalized means and explicit interpolation/extrapolation scope."""
    try:
        spec = validated_spec(data, rates, _allow_unvalidated=_allow_unvalidated)
        service, extrapolated = {m: {} for m in deployment}, []
        covered = set(tuple(v) for v in spec["training_path_states"])
        for task, row in routing.items():
            for model, q in row.items():
                if q <= 0:
                    continue
                n = deployment[model]
                service[model][task] = _baseline(data, model, task, n) * spec["service_multiplier"]
                if (model, task, n) not in covered:
                    extrapolated.append([model, task, n])
        return service, {"mode": "calibrated_surrogate", "kind": "calibrated_prediction",
            "scope": "standalone task/replica means transferred with one fitted global multiplier; mixed tails unmeasured",
            "version_sha256": spec["version_sha256"], "validation_status": spec["status"],
            "extrapolated_path_states": extrapolated,
            "hardware_capacity_certified": False,
            "diagnostic_only": bool(_allow_unvalidated),
            "optimality_scope": "frozen predictor finite domain only"}
    except (KeyError, TypeError, ValueError) as exc:
        return None, {"mode": "calibrated_surrogate", "reason": str(exc), "kind": "unknown"}


def surrogate_auxiliary(data, deployment, routing, rates, horizon, energy_reference_watts, amortization=None,
                        *, _allow_unvalidated=False):
    service, evidence = surrogate_service(data, deployment, routing, rates, _allow_unvalidated=_allow_unvalidated)
    if service is None:
        return {"energy_norm": None, "safe_memory_mib": None, "evidence": evidence}
    spec = _spec(data)
    features = _features(data, deployment, routing)
    watts = float(np.dot(features, np.asarray(spec["power_coefficients"])))
    startup = spec["startup_joules_per_worker"] * sum(deployment.values())
    additive = data["safe_factor"] * math.fsum(data["memory_table_mib"][m][str(n)] for m, n in deployment.items())
    memory = max(additive * spec["memory_multiplier"],
                 spec["training_deployment_memory"].get(_deployment_key(deployment), 0.0))
    return {"energy_norm": (watts + startup / (amortization or horizon)) / energy_reference_watts,
            "safe_memory_mib": memory,
            "evidence": {**evidence, "scope": "whole-board nonnegative power regression, startup-per-worker and memory transfer predictions",
                         "watts": watts, "startup_joules": startup,
                         "run_joules": watts * horizon, "coverage_valid": False,
                         "scope_kind": "whole_gpu_board", "measured": False,
                         "power_features": features.tolist(),
                         "power_extrapolation": any(features[i] < spec["training_feature_min"][i] - 1e-9 or
                                                    features[i] > spec["training_feature_max"][i] + 1e-9 for i in (1, 2)),
                         "memory_is_hardware_bound": False}}


def _import_source(source, accepted_runtime_versions, accepted_development_phases=None):
    """Verify explicit historical packet revision, without rewriting its bindings.

    Current source files may have changed since execution. Historical acceptance
    requires the caller to name the stored packet/hash and accepted runtime hash;
    this does not run validate_packet against a new revision or disguise it as
    unchanged. Actual measured utility is recomputed under the current evaluator.
    """
    path, packet_path = Path(source["path"]), Path(source["packet_path"])
    if digest(path) != source["sha256"] or digest(packet_path) != source["packet_sha256"]:
        raise ValueError("Calibration source/packet file fingerprint mismatch")
    raw, packet = json.loads(path.read_text()), json.loads(packet_path.read_text())
    from .static_study import validate_packet
    validation = validate_packet(packet)
    if not validation["valid"]:
        raise ValueError("Historical execution packet/snapshot bindings changed: " + repr(validation["changed"]))
    relocation = packet.get("source_relocation")
    if relocation:
        original_path = Path(relocation["source_packet"])
        if digest(original_path) != relocation["source_packet_sha256"]:
            raise ValueError("Original execution packet revision changed")
        original = json.loads(original_path.read_text())
        if [j["id"] for j in packet["jobs"]] != [j["id"] for j in original["jobs"]]:
            raise ValueError("Relocated snapshot changed original job identities")
        for relocated_job, original_job in zip(packet["jobs"], original["jobs"]):
            # Only a storage-level file-binding digest may change. Roles,
            # phases, labels/purposes and every physical decision stay original.
            def semantic_job(value):
                result = copy.deepcopy(value)
                result.get("provenance", {}).pop("frozen_files_sha256", None)
                return result
            if semantic_job(relocated_job) != semantic_job(original_job):
                raise ValueError("Relocation changed original job semantics/phase/purpose")
        original_bindings = {b["path"]: b["sha256"] for b in original["bindings"]}
        if len(original_bindings) != len(original["bindings"]):
            raise ValueError("Original execution packet repeats file bindings")
        relocated_paths = {}
        snapshot_paths = set()
        for mapping in relocation["mapping"]:
            if (original_bindings.get(mapping["original"]) != mapping["sha256"]
                    or mapping["original"] in relocated_paths
                    or mapping["snapshot"] in snapshot_paths
                    or digest(mapping["snapshot"]) != mapping["sha256"]):
                raise ValueError("Relocated execution source differs from original bound bytes")
            relocated_paths[mapping["original"]] = mapping["snapshot"]
            snapshot_paths.add(mapping["snapshot"])
        expected_bindings = {relocated_paths.get(path, path): sha for path, sha in original_bindings.items()}
        actual_bindings = {b["path"]: b["sha256"] for b in packet["bindings"]}
        if (len(expected_bindings) != len(original_bindings)
                or len(actual_bindings) != len(packet["bindings"])
                or actual_bindings != expected_bindings):
            raise ValueError("Relocation dropped, added or changed original file bindings")
    job = next((j for j in packet["jobs"] if j["id"] == raw.get("experiment_job_id")), None)
    phases = {"calibration"} if accepted_development_phases is None else set(accepted_development_phases)
    if not phases or phases - {"calibration", "adaptive"}:
        raise ValueError("Only explicit calibration/adaptive development phases can be accepted")
    if not job or job.get("phase") not in phases or job.get("phase") == "confirmation":
        raise ValueError("Only explicitly bound development calibration runs may fit the predictor")
    if job.get("phase") == "adaptive" and source.get("purpose") != "adaptive_development":
        raise ValueError("Adaptive evidence requires explicit purpose=adaptive_development")
    identity = raw.get("executed_measurement_identity", {})
    if quality_digest(identity) != raw.get("executed_measurement_id"):
        raise ValueError("Invalid executed measurement identity")
    expected = job["identity"]
    if expected["runtime_code_sha256"] not in accepted_runtime_versions:
        raise ValueError("Historical runtime revision was not explicitly accepted")
    if quality_digest(expected) != job["id"]:
        raise ValueError("Invalid prepared measurement identity")
    for key, value in expected.items():
        if key not in {"hardware_contract", "workers"} and identity.get(key) != value:
            raise ValueError("Executed measurement differs from prepared job: " + key)
    actual_hardware = identity.get("hardware_contract", {})
    expected_hardware = expected["hardware_contract"]
    if ({k: v for k, v in actual_hardware.items() if k != "device_uuid"}
            != {k: v for k, v in expected_hardware.items() if k != "device_uuid"}):
        raise ValueError("Executed hardware contract differs from prepared job")
    from .config import SystemConfig
    from .evaluation import evaluate_run
    config = SystemConfig(**raw["config"])
    transport = raw["transport_config"]
    preflight = raw.get("preflight", {})
    gpu = preflight.get("gpu", {})
    devices = {worker.get("env", {}).get("CUDA_VISIBLE_DEVICES", "0")
               for worker in expected["workers"].values()}
    if (expected_hardware.get("single_gpu") is not True
            or actual_hardware.get("single_gpu") is not True
            or actual_hardware.get("memory_budget_mib") != config.gpu_memory_mib
            or preflight.get("ready") is not True or preflight.get("gpu_checked") is not True
            or not _number(gpu.get("memory_mib"), positive=True)
            or gpu["memory_mib"] < config.gpu_memory_mib
            or not isinstance(preflight.get("device"), str)
            or "," in preflight["device"] or not preflight["device"]
            or preflight["device"] == "-1" or devices != {preflight["device"]}):
        raise ValueError("Executed GPU count/memory budget differs from configuration/preflight")
    declared_uuid = expected_hardware.get("device_uuid")
    if declared_uuid not in {None, "UNBOUND_UNTIL_EXECUTION", gpu.get("uuid")}:
        raise ValueError("Executed GPU UUID differs from explicitly declared hardware")
    if (raw.get("status") != "completed" or raw.get("backend") != "native"
            or raw["plan"]["deployment"] != expected["deployment"]
            or raw["plan"]["routing"] != expected["routing"]
            or config.workload != expected["workload"]
            or config.horizon_seconds != expected["horizon_seconds"]
            or config.random_seed != expected["seed"]
            or config.arrival_process != expected["arrival_process"]
            or transport.get("measurement_seconds") != expected["horizon_seconds"]
            or transport.get("warmup_seconds") != expected["warmup_seconds"]
            or transport.get("drain_timeout_seconds") != expected["drain_seconds"]
            or transport.get("random_seed") != expected["seed"]
            or transport.get("arrival_process") != expected["arrival_process"]
            or raw.get("priming_policy") != expected["warmup_policy"]
            or raw.get("priming_size_metric") != expected["warmup_size_metric"]
            or raw.get("measurement_window", {}).get("trace_sha256") != expected["trace_sha256"]):
        raise ValueError("Calibration raw transport differs from frozen source job")
    for model, worker in expected["workers"].items():
        current = copy.deepcopy(identity["workers"][model])
        current.get("env", {}).pop("GS_CHECKPOINT_FILES_JSON", None)
        if current != worker:
            raise ValueError("Calibration worker identity changed")
    utility = evaluate_run(config, raw)["utility"]
    if not utility.get("eligible") or raw.get("energy_evidence", {}).get("kind") != "measured":
        raise ValueError("Incomplete or invalid measured calibration utility")
    context = config.catalog.get("measurement_context") or {}
    if (identity.get("runtime_checkpoint_sha256") != context.get("runtime_checkpoint_sha256")
            or identity.get("hardware_contract", {}).get("device_uuid") != raw.get("preflight", {}).get("gpu", {}).get("uuid")):
        raise ValueError("Actual runtime checkpoint/GPU differs from measured execution identity")
    return raw, utility, {**source, "runtime_code_sha256": expected["runtime_code_sha256"],
                          "accepted_development_phases": sorted(phases),
                          "source_revision_policy": "explicit historical acceptance; original packet not mutated"}


def fit_calibrated_surrogate(base_config, sources, training_ids, validation_ids,
                             accepted_runtime_versions, thresholds=None, protocol=None,
                             accepted_development_phases=None):
    """Fit <=3 power coefficients and one service multiplier; holdout stays unused.

    IDs and acceptance thresholds must be declared before inspecting outcomes.
    Six training and two held-out conditions are recommended; at least four
    training and two validation conditions are required. Confirmation runs cannot
    be relabeled silently. This function never starts a GPU or model process.
    """
    limits = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    if set(limits) != set(DEFAULT_THRESHOLDS) or not all(_number(v, positive=True) for v in limits.values()):
        raise ValueError("Explicit positive validation thresholds required for all metrics")
    if protocol is not None:
        payload = {k: v for k, v in protocol.items() if k != "sha256"}
        if (protocol.get("sha256") != quality_digest(payload)
                or protocol.get("base_problem_sha256") != quality_digest(base_config.as_dict())
                or protocol.get("training_ids") != list(training_ids)
                or protocol.get("validation_ids") != list(validation_ids)
                or protocol.get("thresholds") != limits):
            raise ValueError("Fitting differs from declared calibration protocol")
    if (len(training_ids) < 4 or len(validation_ids) < 2
            or len(set(training_ids)) != len(training_ids)
            or len(set(validation_ids)) != len(validation_ids)
            or set(training_ids) & set(validation_ids)):
        raise ValueError("Need disjoint >=4 training and >=2 held-out calibration IDs")
    by_id = {s["id"]: s for s in sources}
    if len(by_id) != len(sources) or set(training_ids) | set(validation_ids) != set(by_id):
        raise ValueError("Calibration IDs must be unique and match declared split")
    imported = {key: _import_source(by_id[key], set(accepted_runtime_versions), accepted_development_phases)
                for key in by_id}
    executed_ids = [v[0]["executed_measurement_id"] for v in imported.values()]
    if len(set(executed_ids)) != len(executed_ids):
        raise ValueError("Repeated execution identity cannot supply separate train/held-out conditions")
    config = copy.deepcopy(base_config.as_dict())
    data = config["catalog"]
    gpu, weights = set(), {}
    for raw, utility, binding in imported.values():
        context = raw["config"]["catalog"].get("measurement_context", {})
        uuid = raw.get("preflight", {}).get("gpu", {}).get("uuid")
        if not uuid or context.get("gpu_uuid") != uuid:
            raise ValueError("Calibration GPU identity inconsistent")
        gpu.add(uuid)
        if raw["config"]["workload"] != config["workload"]:
            raise ValueError("Workload changed across calibration/base problem")
        for field in ("weights", "latency_reference_ms", "gpu_memory_mib", "energy_reference_watts",
                      "startup_amortization_seconds", "utility_schema", "drain_seconds"):
            if raw["config"].get(field) != config.get(field):
                raise ValueError("Calibration scoring contract differs from base problem: " + field)
        raw_quality = raw["config"]["catalog"].get("quality", {})
        if quality_digest(raw_quality) != quality_digest(data.get("quality", {})):
            raise ValueError("Calibration quality/input/reference identity differs from base problem")
        for m in raw["plan"]["deployment"]:
            value = context.get("runtime_checkpoint_sha256", {}).get(m)
            if not value or (m in weights and weights[m] != value):
                raise ValueError("Calibration weights missing or changed")
            weights[m] = value
    if len(gpu) != 1:
        raise ValueError("Calibrations on different GPUs cannot share one predictor")
    features, power, service_num, service_den = [], [], 0.0, 0.0
    startup, worker_counts, trained_paths, deployment_memory = [], [], set(), {}
    memory_multiplier = 1.0
    for key in training_ids:
        raw, utility, _ = imported[key]
        x, routing = raw["plan"]["deployment"], raw["plan"]["routing"]
        features.append(_features(data, x, routing))
        energy = raw["energy_evidence"]
        power.append(energy["run_joules"] / raw["measurement_window"]["duration_seconds"])
        startup.append(energy["startup_joules"])
        worker_counts.append(sum(x.values()))
        for record in raw["request_records"]:
            m, task = raw["replica_to_family"][record["model"]], record["task"]
            observed = record.get("worker_latency_ms")
            if not _number(observed, positive=True):
                raise ValueError("Missing positive actual worker service time")
            service_num += observed
            service_den += _baseline(data, m, task, x[m])
            trained_paths.add((m, task, x[m]))
        additive = data["safe_factor"] * math.fsum(data["memory_table_mib"][m][str(n)] for m, n in x.items())
        measured = raw["memory_evidence"]["safe_memory_mib"]
        memory_multiplier = max(memory_multiplier, measured / additive)
        deployment_memory[_deployment_key(x)] = max(deployment_memory.get(_deployment_key(x), 0.0), measured)
    matrix = np.asarray(features)
    # Predeclared hierarchy limits parameters to n_train/2 and identifiable
    # columns; no held-out value participates in feature/model selection.
    columns = [0]
    for candidate in (1, 2):
        proposed = columns + [candidate]
        if len(proposed) * 2 <= len(training_ids) and np.linalg.matrix_rank(matrix[:, proposed]) == len(proposed):
            columns = proposed
    coefficients = np.zeros(3)
    coefficients[columns], residual = nnls(matrix[:, columns], np.asarray(power))
    spec = {"schema": SCHEMA, "status": "FITTED_UNVALIDATED",
            "baseline_identities": _baseline_identity(data),
            "input_pool_sha256": data["quality"]["identity"]["input_pool_sha256"],
            "quality_identity": copy.deepcopy(data["quality"]["identity"]),
            "quality_document_sha256": quality_digest(data["quality"]),
            "gpu_uuid": next(iter(gpu)), "workload": config["workload"],
            "service_multiplier": service_num / service_den,
            "service_fit_rule": "sum(actual worker service ms)/sum(path-state baseline mean ms for each routed request)",
            "power_coefficients": coefficients.tolist(),
            "power_feature_names": ["board_intercept_watts", "baseline_busy_seconds_per_second", "resident_worker_count"],
            "selected_power_columns": columns, "training_power_residual_l2": float(residual),
            "training_feature_min": matrix.min(axis=0).tolist(), "training_feature_max": matrix.max(axis=0).tolist(),
            "startup_joules_per_worker": math.fsum(startup) / sum(worker_counts),
            "startup_scope": "pooled load/prime/warmup energy per worker proxy, not an independently measured physical constant",
            "memory_multiplier": memory_multiplier, "training_deployment_memory": deployment_memory,
            "training_path_states": [list(v) for v in sorted(trained_paths)],
            "training_ids": list(training_ids), "validation_ids": list(validation_ids),
            "protocol": protocol,
            "baseline_hardware_scope": "standalone hardware transfer baseline; equal hardware/performance is not assumed",
            "accepted_runtime_versions": sorted(set(accepted_runtime_versions)),
            "accepted_development_phases": sorted({"calibration"} if accepted_development_phases is None else accepted_development_phases),
            "sources": [imported[key][2] for key in sorted(imported)],
            "thresholds": limits, "validation": None,
            "scope": "calibrated transfer predictor; finite-domain model optimum only; unmeasured GPU optimum unproven"}
    def seal():
        spec["version_sha256"] = quality_digest({k: v for k, v in spec.items() if k != "version_sha256"})
        data["calibrated_surrogate"] = spec
    data["performance_mode"] = data["memory_mode"] = "calibrated_surrogate"
    data["measurement_context"] = {"gpu_uuid": spec["gpu_uuid"], "runtime_checkpoint_sha256": weights,
        "historical_quality_checkpoint_equivalence": "receipt_scope_only; actual run outputs rescored"}
    spec["measurement_context"] = copy.deepcopy(data["measurement_context"])
    seal()
    from .prediction import predict
    comparisons, passed = [], True
    for key in validation_ids:
        raw, measured, _ = imported[key]
        x, routing = raw["plan"]["deployment"], raw["plan"]["routing"]
        duration = raw["measurement_window"]["duration_seconds"]
        pred = predict(data, x, routing, config["workload"], config["gpu_memory_mib"], config["latency_reference_ms"],
                       config["weights"], horizon=duration, seed=raw["transport_config"]["random_seed"],
                       arrival_process=config["arrival_process"], drain_seconds=config["drain_seconds"],
                       energy_reference_watts=config["energy_reference_watts"],
                       startup_amortization_seconds=config.get("startup_amortization_seconds"),
                       _calibration_validation=True)
        # This score is a held-out diagnostic, never a formal eligible result.
        pred["U"] = pred.get("diagnostic_partial_U") if pred.get("utility_complete") and pred.get("constraints_eligible") else None
        expected_service = math.fsum(_baseline(data, raw["replica_to_family"][r["model"]], r["task"],
                                               x[raw["replica_to_family"][r["model"]]])
                                     * spec["service_multiplier"] for r in raw["request_records"])
        actual_service = math.fsum(r["worker_latency_ms"] for r in raw["request_records"])
        relative = lambda a, b: abs(a - b) / max(abs(b), 1e-9)
        errors = {"service_relative_error": relative(expected_service, actual_service),
                  "energy_relative_error": relative(pred.get("E_norm", 0) or 0, measured["E_norm"]),
                  "memory_relative_error": relative(pred.get("M_norm", 0) or 0, measured["M_norm"]),
                  "mean_relative_error": relative(pred.get("L_norm", 0) or 0, measured["L_norm"]),
                  "p95_relative_error": relative(pred.get("P_norm", 0) or 0, measured["P_norm"]),
                  "quality_absolute_error": abs((pred.get("Q_norm", 0) or 0) - measured["Q_norm"]),
                  "utility_absolute_error": abs((pred.get("U", 0) or 0) - measured["U"])}
        condition_passed = pred.get("constraints_eligible") is True and pred.get("utility_complete") is True and all(errors[k] <= limits[k] for k in limits)
        passed &= condition_passed
        comparisons.append({"id": key, "prediction": {k: pred.get(k) for k in ("U", "Q_norm", "T_norm", "L_norm", "P_norm", "M_norm", "E_norm")},
                            "measured": {k: measured.get(k) for k in ("U", "Q_norm", "T_norm", "L_norm", "P_norm", "M_norm", "E_norm")},
                            "errors": errors, "passed": bool(condition_passed)})
        from .metrics import mean, percentile
        comparisons[-1]["per_task_latency_ms"] = {
            task: {"predicted_mean": pred.get("per_task", {}).get(task, {}).get("mean_ms"),
                   "predicted_p95": pred.get("per_task", {}).get(task, {}).get("p95_ms"),
                   "measured_mean": mean([(r["completion_time_ns"] - r["scheduled_time_ns"]) / 1e6
                                          for r in raw["request_records"] if r["task"] == task]),
                   "measured_p95": percentile([(r["completion_time_ns"] - r["scheduled_time_ns"]) / 1e6
                                               for r in raw["request_records"] if r["task"] == task], 95)}
            for task in config["workload"]}
    rank_comparisons = []
    for i, left in enumerate(comparisons):
        for right in comparisons[i + 1:]:
            measured_delta = left["measured"]["U"] - right["measured"]["U"]
            a, b = left["prediction"]["U"], right["prediction"]["U"]
            match = a is not None and b is not None and (a - b) * measured_delta >= 0
            rank_comparisons.append({"left": left["id"], "right": right["id"],
                                     "measured_U_delta": measured_delta, "ordering_consistent": bool(match)})
            passed &= bool(match)
    spec["validation"] = {"conditions": comparisons, "ranking": rank_comparisons,
                          "note": "Two held-out conditions do not certify unseen combinations or stochastic significance"}
    spec["status"] = "VALIDATED_ON_HELDOUT_CONDITIONS" if passed else "VALIDATION_FAILED"
    seal()
    return config, {"schema": SCHEMA, "version_sha256": spec["version_sha256"], "status": spec["status"],
                    "training_conditions": len(training_ids), "validation_conditions": len(validation_ids),
                    "validation": spec["validation"], "gpu_started": False, "scope": spec["scope"]}
