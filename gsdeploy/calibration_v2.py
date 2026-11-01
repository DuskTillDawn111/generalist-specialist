"""Explicit per-input replay experiment; independent from the v1 predictor.

All fitting is CPU-only and requires a predeclared protocol. Old v1 holdouts
are development data; new seed-46/47 conditions supply independent validation.
No request service from a validation run is used to construct its prediction.
"""
import copy
import json
import math
from pathlib import Path
import random
from statistics import median

import numpy as np
from scipy.optimize import nnls

from .calibration import (DEFAULT_THRESHOLDS, _baseline, _baseline_identity,
                          _deployment_key, _features, _import_source, _number)
from .evidence import digest
from .quality import quality_digest

MODE = "calibrated_replay_v2"
SCHEMA = "gsdeploy.calibrated_replay.v2"
BASELINE_SCHEMA = "gsdeploy.frozen_input_service.v2"
PROTOCOL_SCHEMA = "gsdeploy.calibration_protocol.v2"
RULES = {
    "baseline": "median of successful standalone service_ms for each (model,task,n,sample_id), across frozen rounds/replicas",
    "missing": "UNKNOWN; no aggregate fallback and no validation-worker-time substitution",
    "identity": "raw request/protocol hashes remain separate from quality receipts; historical missing hashes require profile+job+frozen input binding; protocol equivalence unverified",
    "sequence": "shuffle frozen task input_order with Random(f'{seed}:samples:{task}'); consume before Router; cycle per task",
    "service_fit": "sum(development actual worker_ms)/sum(matching frozen per-input median_ms)",
    "rho": "sum(q * equally-weighted full-input-pool median service_ms * global_ratio / 1000) / n",
    "power": "v1 original standalone mean features [intercept, sum(q*baseline_mean_ms/1000), worker_count]; NNLS, full rank, <=n_development/2, <=3 columns",
    "memory": "max(additive standalone safe residency * learned multiplier>=1, development deployment observed peak)",
    "scope": "frozen finite-domain predictor only; physical all-combination optimum unproven",
}


def _sealed(value, field="sha256"):
    return quality_digest({k: v for k, v in value.items() if k != field})


def predictor_code_bindings():
    """Declare the mathematical implementation revision without importing GPU."""
    root = Path(__file__).resolve().parent
    names = ("calibration_v2.py", "calibration.py", "prediction.py", "performance.py", "config.py",
             "quality.py", "trace.py", "routing.py")
    return [{"path": "gsdeploy/" + name, "sha256": digest(root / name)} for name in names]


def freeze_input_baselines(base_config, runtime_inputs_path=None):
    """Read standalone files only; never read any open-loop calibration run."""
    data, tables, sources, missing = base_config.catalog, {}, {}, []
    runtime_samples = {}
    runtime_source = None
    if runtime_inputs_path is not None:
        path = Path(runtime_inputs_path).resolve()
        sha = digest(path)
        if sha != data["quality"]["identity"]["input_pool_sha256"]:
            raise ValueError("Runtime input pool fingerprint differs from frozen quality")
        sources[str(path)] = sha
        runtime_source = {"path": str(path), "sha256": sha}
        for line in path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                key = (item["task_type"], item["pool_id"])
                if key in runtime_samples:
                    raise ValueError("Repeated runtime input identity")
                runtime_samples[key] = item["sample"]
        for task in base_config.workload:
            if [sample for offered_task, sample in runtime_samples if offered_task == task] != data["quality"]["input_order"][task]:
                raise ValueError("Runtime sample order differs from frozen quality input order")
    for model, tasks in data.get("task_replica_profiles", {}).items():
        for task, states in tasks.items():
            if model not in data["models"] or task not in base_config.workload:
                continue
            path_quality = data["quality"]["paths"][model][task]
            quality_protocol = path_quality.get("protocol_sha256", data["quality"]["identity"]["protocol_sha256"])
            for n, entry in states.items():
                _baseline(data, model, task, int(n))
                observations, raw_protocols = {}, set()
                historical = None
                files = entry.get("raw_service_samples", [])
                for filename in files:
                    path = Path(filename).resolve()
                    sha = digest(path)
                    if str(path) in sources and sources[str(path)] != sha:
                        raise ValueError("Standalone source changed while freezing")
                    sources[str(path)] = sha
                    for line in path.read_text().splitlines():
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if record.get("success") is not True:
                            continue
                        sample = record.get("sample_id")
                        if (record.get("task") != task or not sample
                                or not _number(record.get("service_ms"), positive=True)):
                            raise ValueError(f"Invalid standalone input service identity: {model}/{task}/{n}")
                        if record.get("protocol_sha256") and record.get("request_sha256"):
                            raw_protocols.add(record["protocol_sha256"])
                            identity = {"identity_kind": "raw_request_observation",
                                "request_sha256": record["request_sha256"],
                                "raw_protocol_sha256": record["protocol_sha256"],
                                "request_protocol_equivalence_to_current_execution_verified": False}
                        else:
                            if historical is None:
                                historical = _historical_input_bindings(data, model, task, entry,
                                    runtime_samples, runtime_source, sources)
                            if sample not in historical:
                                continue  # Unverifiable input linkage remains UNKNOWN.
                            identity = historical[sample]
                        group = observations.setdefault(sample, {"values": [], "identity": identity})
                        if group["identity"] != identity:
                            raise ValueError("Same sample has changed request identity")
                        group["values"].append(record["service_ms"])
                samples = {sample: {"median_service_ms": median(group["values"]),
                                    "observations": len(group["values"]),
                                    **group["identity"]}
                           for sample, group in observations.items()}
                needed = set(data["quality"]["input_order"][task])
                absent = sorted(needed - set(samples))
                if absent:
                    missing.append({"model": model, "task": task, "n": int(n), "sample_ids": absent})
                tables.setdefault(model, {}).setdefault(task, {})[str(n)] = {
                    "samples": samples, "quality_protocol_receipt_sha256": quality_protocol,
                    "observed_raw_protocol_sha256": sorted(raw_protocols),
                    "scope": "historical standalone input transfer; current request/protocol equivalence unverified; actual confirmation outputs must be rescored",
                    "checkpoint_identity": copy.deepcopy(path_quality.get("checkpoint_identity", path_quality.get("checkpoint_sha256"))),
                    "raw_sources": [str(Path(f).resolve()) for f in files]}
    frozen = {"schema": BASELINE_SCHEMA, "base_problem_sha256": quality_digest(base_config.as_dict()),
              "quality_document_sha256": quality_digest(data["quality"]),
              "baseline_identities": _baseline_identity(data), "input_baselines": tables,
              "sources": [{"path": path, "sha256": sha} for path, sha in sorted(sources.items())],
              "missing_paths": missing, "rules": copy.deepcopy(RULES)}
    frozen["sha256"] = _sealed(frozen)
    return frozen


def _historical_input_bindings(data, model, task, entry, runtime_samples, runtime_source, sources):
    """Bind missing raw hashes through existing profile/job/media receipts.

    This never manufactures a request hash or asserts protocol equivalence.
    No linkage to the frozen runtime input means the sample stays unknown.
    """
    if not runtime_source or not entry.get("source"):
        return {}
    quality_path = data["quality"]["paths"][model][task]
    worker_path = Path(quality_path["source"]).resolve()
    job_path, profile_path = worker_path.parent / "job.json", Path(entry["source"]).resolve()
    if not job_path.is_file() or not worker_path.is_file() or not profile_path.is_file():
        return {}
    worker, job, profile = [json.loads(path.read_text()) for path in (worker_path, job_path, profile_path)]
    if (Path(job.get("spec", "")).resolve() != worker_path or profile.get("model") != worker
            or profile.get("status") != "COMPLETE"):
        return {}
    bindings = []
    for path in (worker_path, job_path, profile_path):
        sources[str(path)] = digest(path)
        bindings.append({"path": str(path), "sha256": sources[str(path)]})
    resources = worker.get("resources", {})
    source = resources.get("inference_source")
    if source:
        path = Path(source).resolve()
        sha = digest(path)
        if sha != resources.get("inference_source_sha256"):
            raise ValueError("Historical inference source fingerprint changed")
        sources[str(path)] = sha
        bindings.append({"path": str(path), "sha256": sha})
    results = {}
    samples = job.get("samples", [])
    if not isinstance(samples, list) or len({sample.get("id") for sample in samples}) != len(samples):
        return {}
    for sample in samples:
        runtime = runtime_samples.get((task, sample.get("id")))
        if not runtime or sample.get("task") != task:
            continue
        # Require identical frozen input payload, beyond a matching sample ID.
        if sample != runtime or not sample.get("input_sha256") or not isinstance(sample.get("input"), str):
            continue
        media = Path(sample["input"]).resolve()
        sha = digest(media)
        if sha != sample["input_sha256"]:
            raise ValueError("Historical input media changed")
        sources[str(media)] = sha
        results[sample["id"]] = {"identity_kind": "historical_profile_input_binding",
            "source_binding": {"receipts": bindings, "runtime_input_pool": runtime_source,
                "sample_sha256": quality_digest(sample), "input": {"path": str(media), "sha256": sha}},
            "request_sha256": None, "raw_protocol_sha256": None,
            "request_protocol_equivalence_to_current_execution_verified": False}
    return results


def _check_baselines(frozen, base_config=None, *, recheck_files=False):
    if (frozen.get("schema") != BASELINE_SCHEMA or frozen.get("sha256") != _sealed(frozen)
            or frozen.get("rules") != RULES):
        raise ValueError("Invalid frozen v2 input baseline identity/rules")
    if base_config is not None and (frozen["base_problem_sha256"] != quality_digest(base_config.as_dict())
            or frozen["quality_document_sha256"] != quality_digest(base_config.catalog["quality"])
            or frozen["baseline_identities"] != _baseline_identity(base_config.catalog)):
        raise ValueError("Frozen input baseline differs from declared base problem")
    if recheck_files:
        for source in frozen["sources"]:
            if digest(source["path"]) != source["sha256"]:
                raise ValueError("Frozen standalone input service file changed")


def declare_replay_protocol(base_config, frozen_baselines, development, holdouts, legacy_v1_protocol):
    """Bind per-condition historical packets before the two new holdouts run.

    Each condition is {id, packet_path, packet_sha256, purpose}. Paths of future
    run files are deliberately absent. Adaptive development is opt-in by role.
    """
    from .static_study import validate_packet
    _check_baselines(frozen_baselines, base_config, recheck_files=True)
    legacy_ids = legacy_v1_protocol["training_ids"] + legacy_v1_protocol["validation_ids"]
    if (len(legacy_ids) != 8 or len(set(legacy_ids)) != 8
            or legacy_v1_protocol.get("sha256") != _sealed(legacy_v1_protocol)):
        raise ValueError("Need the unchanged original eight-condition v1 protocol")
    conditions = copy.deepcopy(list(development) + list(holdouts))
    holdout_ids = {condition["id"] for condition in holdouts}
    ids = [condition["id"] for condition in conditions]
    if (len(set(ids)) != len(ids) or len(holdouts) != 2 or len(development) < 8
            or not set(legacy_ids).issubset(c["id"] for c in development)):
        raise ValueError("Need unique disjoint development (including original eight) and two new holdouts")
    seeds, versions = [], set()
    for condition in conditions:
        path = Path(condition["packet_path"]).resolve()
        if digest(path) != condition["packet_sha256"]:
            raise ValueError("Declared v2 source packet fingerprint changed")
        packet = json.loads(path.read_text())
        if not validate_packet(packet)["valid"]:
            raise ValueError("Declared v2 historical packet bindings changed")
        job = next(j for j in packet["jobs"] if j["id"] == condition["id"])
        if job["phase"] not in {"calibration", "adaptive"}:
            raise ValueError("Formal confirmation must never become v2 development/holdout")
        seed = job["identity"]["seed"]
        if seed in {43, 44, 45}:
            raise ValueError("Seeds 43/44/45 are reserved for formal confirmation")
        if condition["id"] in holdout_ids:
            if (condition.get("purpose") != "v2_independent_holdout" or job["phase"] != "calibration"
                    or job.get("purpose") != "v2_predeclared_heldout_validation"
                    or job["identity"]["horizon_seconds"] != 600):
                raise ValueError("V2 independent holdouts require explicit purpose and H600")
            seeds.append(seed)
        elif condition["id"] in legacy_ids:
            if condition.get("purpose") != "legacy_v1_development":
                raise ValueError("Original v1 holdouts must explicitly become development")
            receipt = legacy_v1_protocol.get("execution_packet_provenance")
            if receipt and condition["packet_sha256"] != receipt["sha256"]:
                raise ValueError("Original v1 conditions must retain their predeclared execution packet revision")
        elif condition.get("purpose") != "adaptive_development" or job["phase"] != "adaptive":
            raise ValueError("Additional development requires explicit adaptive evidence")
        versions.add(job["identity"]["runtime_code_sha256"])
        condition["packet_path"] = str(path)
    if seeds != [46, 47]:
        raise ValueError("New v2 independent holdouts must use seeds 46 and 47")
    protocol = {"schema": PROTOCOL_SCHEMA, "base_problem_sha256": quality_digest(base_config.as_dict()),
        "frozen_baselines_sha256": frozen_baselines["sha256"], "rules": copy.deepcopy(RULES),
        "predictor_code_bindings": predictor_code_bindings(),
        "development_ids": [c["id"] for c in development], "validation_ids": [c["id"] for c in holdouts],
        "conditions": conditions, "thresholds": dict(DEFAULT_THRESHOLDS),
        "legacy_v1_protocol": copy.deepcopy(legacy_v1_protocol),
        "legacy_v1_protocol_sha256": legacy_v1_protocol["sha256"], "legacy_development_ids": legacy_ids,
        "old_holdout_policy": "Original eight conditions, including viewed v1 holdouts, are development only",
        "holdout_seeds": [46, 47], "holdout_horizon_seconds": 600,
        "formal_confirmation_seeds_reserved": [43, 44, 45], "accepted_runtime_versions": sorted(versions)}
    protocol["sha256"] = _sealed(protocol)
    return protocol


def _spec_v2(data):
    spec = data.get(MODE, {})
    if (spec.get("schema") != SCHEMA or spec.get("version_sha256") != _sealed(spec, "version_sha256")
            or spec.get("rules") != RULES):
        raise ValueError("MISSING_OR_CHANGED_V2_REPLAY_SPEC")
    if (not _number(spec.get("service_multiplier"), positive=True)
            or not _number(spec.get("memory_multiplier"), positive=True) or spec["memory_multiplier"] < 1
            or not _number(spec.get("startup_joules_per_worker"))
            or not isinstance(spec.get("power_coefficients"), list)
            or len(spec["power_coefficients"]) != 3
            or not all(_number(v) for v in spec["power_coefficients"])):
        raise ValueError("INVALID_V2_REPLAY_PARAMETERS")
    frozen = {**spec["frozen_baseline_contract"], "input_baselines": spec["input_baselines"],
              "sources": spec["baseline_sources"]}
    protocol = spec["protocol"]
    if (spec["frozen_baselines_sha256"] != _sealed(frozen)
            or protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("sha256") != _sealed(protocol)
            or protocol.get("frozen_baselines_sha256") != spec["frozen_baselines_sha256"]
            or protocol.get("base_problem_sha256") != frozen["base_problem_sha256"]):
        raise ValueError("V2_REPLAY_FROZEN_BASELINE_OR_PROTOCOL_CHANGED")
    return spec


def validated_replay_spec(data, rates, *, _allow_unvalidated=False):
    spec = _spec_v2(data)
    if spec.get("status") != "VALIDATED_ON_HELDOUT_CONDITIONS" and not (
            _allow_unvalidated and spec.get("status") == "FITTED_UNVALIDATED"):
        raise ValueError("V2_REPLAY_NOT_VALIDATED:" + str(spec.get("status")))
    if data.get("performance_mode") != MODE or data.get("memory_mode") != MODE:
        raise ValueError("V2_REPLAY_EXPLICIT_MODES_REQUIRED")
    if (spec["quality_document_sha256"] != quality_digest(data.get("quality", {}))
            or spec["input_order"] != data.get("quality", {}).get("input_order")
            or spec["safe_factor"] != data.get("safe_factor")
            or spec["measurement_context"] != data.get("measurement_context") or spec["workload"] != rates):
        raise ValueError("V2_REPLAY_INPUT_CONTEXT_OR_WORKLOAD_CHANGED")
    for model, identity in _baseline_identity(data).items():
        if spec["baseline_identities"].get(model) != identity:
            raise ValueError("V2_REPLAY_BASELINE_CHANGED:" + model)
    return spec


def input_service_ms(spec, model, task, n, sample_id):
    """Already-scaled per-worker service; caller holds a validated v2 spec."""
    try:
        entry = spec["input_baselines"][model][task][str(n)]["samples"][sample_id]
        value = entry["median_service_ms"]
        if (not _number(value, positive=True) or entry.get("identity_kind") not in {
                "raw_request_observation", "historical_profile_input_binding"}
                or (entry["identity_kind"] == "raw_request_observation" and
                    (not entry.get("request_sha256") or not entry.get("raw_protocol_sha256")))
                or (entry["identity_kind"] == "historical_profile_input_binding" and not entry.get("source_binding"))):
            raise ValueError("Invalid median/input fingerprint")
        return value * spec["service_multiplier"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"MISSING_V2_INPUT_SERVICE:{model}/{task}/{n}/{sample_id}") from exc


def replay_path_statistics(spec, model, task, n):
    """Pool mean for rho and cross-input minimum bound; no FIFO mean-as-bound."""
    ids = spec["input_order"][task]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("INVALID_V2_INPUT_ORDER")
    values = [input_service_ms(spec, model, task, n, sample) for sample in ids]
    return {"minimum_ms": min(values), "mean_ms": math.fsum(values) / len(values)}


def service_lower_ms(data, rates, model, task, n):
    """Validated same-predictor mathematical bound; None means unknown, not zero."""
    try:
        return replay_path_statistics(validated_replay_spec(data, rates), model, task, n)["minimum_ms"]
    except (KeyError, TypeError, ValueError):
        return None


def replay_input_ids(data, trace, seed):
    """Select route-independent samples before the caller invokes Router."""
    orders, indices = {}, {}
    for task, ids in data["quality"]["input_order"].items():
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("INVALID_V2_INPUT_ORDER")
        orders[task] = list(ids)
        random.Random(f"{seed}:samples:{task}").shuffle(orders[task])
        indices[task] = 0
    selected = []
    for _, task in trace:
        selected.append(orders[task][indices[task] % len(orders[task])])
        indices[task] += 1
    return selected


def replay_service(data, deployment, routing, rates, *, _allow_unvalidated=False):
    try:
        spec = validated_replay_spec(data, rates, _allow_unvalidated=_allow_unvalidated)
        service = {m: {} for m in deployment}
        for task, row in routing.items():
            for model, q in row.items():
                if q > 0:
                    _baseline(data, model, task, deployment[model])
                    service[model][task] = replay_path_statistics(spec, model, task, deployment[model])["mean_ms"]
        return service, {"mode": MODE, "kind": "calibrated_per_input_prediction",
            "version_sha256": spec["version_sha256"], "validation_status": spec["status"],
            "scope": "frozen per-input standalone medians replayed through shared FIFO; global development ratio",
            "rho_scope": RULES["rho"], "diagnostic_only": bool(_allow_unvalidated),
            "hardware_capacity_certified": False, "optimality_scope": "frozen predictor finite domain only"}
    except (KeyError, TypeError, ValueError) as exc:
        return None, {"mode": MODE, "kind": "unknown", "reason": str(exc)}


def replay_auxiliary(data, deployment, routing, rates, horizon, energy_reference_watts, amortization=None,
                     *, _allow_unvalidated=False):
    service, evidence = replay_service(data, deployment, routing, rates, _allow_unvalidated=_allow_unvalidated)
    if service is None:
        return {"energy_norm": None, "safe_memory_mib": None, "evidence": evidence}
    spec = data[MODE]
    features = _features(data, deployment, routing)
    watts = float(np.dot(features, np.asarray(spec["power_coefficients"])))
    startup = spec["startup_joules_per_worker"] * sum(deployment.values())
    additive = data["safe_factor"] * math.fsum(data["memory_table_mib"][m][str(n)] for m, n in deployment.items())
    memory = max(additive * spec["memory_multiplier"], spec["training_deployment_memory"].get(_deployment_key(deployment), 0))
    return {"energy_norm": (watts + startup / (amortization or horizon)) / energy_reference_watts,
        "safe_memory_mib": memory, "evidence": {**evidence, "scope": "whole-board NNLS power/startup/memory transfer prediction",
            "watts": watts, "startup_joules": startup, "run_joules": watts * horizon,
            "power_features": features.tolist(), "measured": False, "coverage_valid": False,
            "memory_is_hardware_bound": False}}


def fit_calibrated_replay(base_config, frozen_baselines, sources, protocol):
    """Fit declared development only; preserve failure on new independent seeds."""
    _check_baselines(frozen_baselines, base_config, recheck_files=True)
    if (protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("sha256") != _sealed(protocol)
            or protocol.get("base_problem_sha256") != quality_digest(base_config.as_dict())
            or protocol.get("frozen_baselines_sha256") != frozen_baselines["sha256"]
            or protocol.get("rules") != RULES or protocol.get("thresholds") != DEFAULT_THRESHOLDS
            or protocol.get("holdout_seeds") != [46, 47]
            or protocol.get("formal_confirmation_seeds_reserved") != [43, 44, 45]):
        raise ValueError("V2 fit differs from predeclared protocol")
    declarations = {condition["id"]: condition for condition in protocol["conditions"]}
    rebuilt_protocol = declare_replay_protocol(base_config, frozen_baselines,
        [declarations[key] for key in protocol["development_ids"]],
        [declarations[key] for key in protocol["validation_ids"]], protocol["legacy_v1_protocol"])
    if rebuilt_protocol != protocol:
        raise ValueError("V2 protocol does not match its frozen packet conditions")
    development, holdout = protocol["development_ids"], protocol["validation_ids"]
    expected = {c["id"]: c for c in protocol["conditions"]}
    by_id = {s["id"]: s for s in sources}
    if (len(by_id) != len(sources) or set(by_id) != set(expected)
            or len(set(development + holdout)) != len(development + holdout)
            or len(development) < 8 or len(holdout) != 2):
        raise ValueError("V2 sources must exactly match unique disjoint declared conditions")
    imported = {}
    for key, source in by_id.items():
        if any(source.get(field) != value for field, value in expected[key].items()):
            raise ValueError("V2 source packet/purpose differs from declared binding")
        imported[key] = _import_source(source, set(protocol["accepted_runtime_versions"]), {"calibration", "adaptive"})
    if len({raw["executed_measurement_id"] for raw, _, _ in imported.values()}) != len(imported):
        raise ValueError("Repeated actual condition cannot be an independent v2 observation")
    config = copy.deepcopy(base_config.as_dict())
    data, gpu, checkpoints = config["catalog"], set(), {}
    for key, (raw, _, _) in imported.items():
        native = raw["config"]
        if (quality_digest(native["catalog"]["quality"]) != quality_digest(data["quality"])
                or native["workload"] != config["workload"]
                or native.get("arrival_process") != config.get("arrival_process")):
            raise ValueError("V2 calibration input/workload changed")
        if (native["catalog"].get("safe_factor") != data.get("safe_factor")
                or native["catalog"].get("capability_contract") != data.get("capability_contract")
                or _baseline_identity(native["catalog"]) != _baseline_identity(data)):
            raise ValueError("V2 calibration baseline/capacity/memory/safe-factor contract changed")
        for field in ("weights", "latency_reference_ms", "gpu_memory_mib", "energy_reference_watts",
                      "startup_amortization_seconds", "utility_schema", "drain_seconds"):
            if native.get(field) != config.get(field):
                raise ValueError("V2 scoring contract changed: " + field)
        context = native["catalog"]["measurement_context"]
        if context.get("gpu_uuid") != raw["preflight"]["gpu"]["uuid"]:
            raise ValueError("V2 execution GPU context differs")
        gpu.add(context["gpu_uuid"])
        for model in raw["plan"]["deployment"]:
            value = context["runtime_checkpoint_sha256"][model]
            if not value or (model in checkpoints and checkpoints[model] != value):
                raise ValueError("V2 execution checkpoint differs")
            checkpoints[model] = value
        seed = raw["transport_config"]["random_seed"]
        if seed in {43, 44, 45} or (key in holdout and
                (seed != [46, 47][holdout.index(key)] or raw["measurement_window"]["duration_seconds"] != 600)):
            raise ValueError("V2 actual split seed/horizon differs")
    if len(gpu) != 1:
        raise ValueError("V2 calibration cannot combine GPUs")
    spec = {"schema": SCHEMA, "status": "FITTED_UNVALIDATED", "rules": copy.deepcopy(RULES), "gpu_uuid": next(iter(gpu)),
        "baseline_identities": _baseline_identity(data), "quality_document_sha256": quality_digest(data["quality"]),
        "input_baselines": copy.deepcopy(frozen_baselines["input_baselines"]),
        "input_order": copy.deepcopy(data["quality"]["input_order"]), "frozen_baselines_sha256": frozen_baselines["sha256"],
        "frozen_baseline_contract": {k: copy.deepcopy(v) for k, v in frozen_baselines.items()
                                     if k not in {"input_baselines", "sources", "sha256"}},
        "baseline_sources": copy.deepcopy(frozen_baselines["sources"]), "protocol": copy.deepcopy(protocol),
        "workload": config["workload"], "safe_factor": data["safe_factor"], "development_ids": development, "validation_ids": holdout,
        "training_ids": development, "training_deployment_memory": {}, "memory_multiplier": 1.0,
        "sources": [imported[key][2] for key in sorted(imported)], "scope": RULES["scope"]}
    service_num = service_den = 0.0
    features, powers, startups, counts = [], [], [], []
    spec["service_multiplier"] = 1.0
    for key in development:
        raw, _, _ = imported[key]
        x, routing = raw["plan"]["deployment"], raw["plan"]["routing"]
        features.append(_features(data, x, routing))
        powers.append(raw["energy_evidence"]["run_joules"] / raw["measurement_window"]["duration_seconds"])
        startups.append(raw["energy_evidence"]["startup_joules"])
        counts.append(sum(x.values()))
        for record in raw["request_records"]:
            if not _number(record.get("worker_latency_ms"), positive=True):
                raise ValueError("Missing positive development worker service")
            model = raw["replica_to_family"][record["model"]]
            service_num += record["worker_latency_ms"]
            service_den += input_service_ms(spec, model, record["task"], x[model], record["pool_id"])
        additive = data["safe_factor"] * math.fsum(data["memory_table_mib"][m][str(n)] for m, n in x.items())
        measured_memory = raw["memory_evidence"]["safe_memory_mib"]
        spec["memory_multiplier"] = max(spec["memory_multiplier"], measured_memory / additive)
        deployment_key = _deployment_key(x)
        spec["training_deployment_memory"][deployment_key] = max(spec["training_deployment_memory"].get(deployment_key, 0), measured_memory)
    matrix, columns = np.asarray(features), [0]
    for candidate in (1, 2):
        proposed = columns + [candidate]
        if len(proposed) * 2 <= len(development) and np.linalg.matrix_rank(matrix[:, proposed]) == len(proposed):
            columns = proposed
    coefficients = np.zeros(3)
    coefficients[columns], residual = nnls(matrix[:, columns], np.asarray(powers))
    spec.update(service_multiplier=service_num / service_den, power_coefficients=coefficients.tolist(),
                startup_joules_per_worker=math.fsum(startups) / sum(counts),
                selected_power_columns=columns, training_power_residual_l2=float(residual),
                training_feature_min=matrix.min(axis=0).tolist(), training_feature_max=matrix.max(axis=0).tolist())
    data["performance_mode"] = data["memory_mode"] = MODE
    data["measurement_context"] = {"gpu_uuid": next(iter(gpu)), "runtime_checkpoint_sha256": checkpoints,
        "historical_quality_checkpoint_equivalence": "receipt_scope_only; actual run outputs rescored"}
    spec["measurement_context"] = copy.deepcopy(data["measurement_context"])
    data[MODE] = spec
    def seal():
        spec["version_sha256"] = _sealed(spec, "version_sha256")
    seal()
    from .prediction import predict
    from .trace import expand_routes, make_trace
    from .routing import Router
    comparisons, passed = [], True
    for key in holdout:
        raw, measured, _ = imported[key]
        x, routing = raw["plan"]["deployment"], raw["plan"]["routing"]
        duration, seed = raw["measurement_window"]["duration_seconds"], raw["transport_config"]["random_seed"]
        trace = make_trace(config["workload"], duration, seed, config["arrival_process"])
        pred = predict(data, x, routing, config["workload"], config["gpu_memory_mib"], config["latency_reference_ms"],
            config["weights"], horizon=duration, seed=seed, arrival_process=config["arrival_process"],
            drain_seconds=config["drain_seconds"], energy_reference_watts=config["energy_reference_watts"],
            startup_amortization_seconds=config.get("startup_amortization_seconds"), _calibration_validation=True)
        diagnostic_u = pred.get("diagnostic_partial_U") if pred.get("constraints_eligible") and pred.get("utility_complete") else None
        # Derive predicted services from the route-independent input replay,
        # never from validation records' worker_ms or their model assignments.
        aliases, routes = expand_routes(x, routing)
        router, expected_service = Router(routes), 0.0
        try:
            for (_, task), sample in zip(trace, replay_input_ids(data, trace, seed)):
                model = aliases[router.choose(task)]
                expected_service += input_service_ms(spec, model, task, x[model], sample)
        except (KeyError, TypeError, ValueError):
            expected_service = None
        actual_service = math.fsum(record["worker_latency_ms"] for record in raw["request_records"])
        def relative(a, b):
            return None if a is None or b is None else abs(a - b) / max(abs(b), 1e-9)
        def absolute(a, b):
            return None if a is None or b is None else abs(a - b)
        errors = {"service_relative_error": relative(expected_service, actual_service),
            "energy_relative_error": relative(pred.get("E_norm"), measured.get("E_norm")),
            "memory_relative_error": relative(pred.get("M_norm"), measured.get("M_norm")),
            "mean_relative_error": relative(pred.get("L_norm"), measured.get("L_norm")),
            "p95_relative_error": relative(pred.get("P_norm"), measured.get("P_norm")),
            "quality_absolute_error": absolute(pred.get("Q_norm"), measured.get("Q_norm")),
            "utility_absolute_error": absolute(diagnostic_u, measured.get("U"))}
        condition_passed = (pred.get("constraints_eligible") is True and pred.get("utility_complete") is True
            and all(value is not None and value <= protocol["thresholds"][name] for name, value in errors.items()))
        passed &= condition_passed
        comparisons.append({"id": key, "seed": seed, "prediction": {"U": diagnostic_u,
            **{term + "_norm": pred.get(term + "_norm") for term in "QTLPME"}},
            "measured": {field: measured.get(field) for field in ["U"] + [term + "_norm" for term in "QTLPME"]},
            "predicted_per_task": pred.get("per_task"), "errors": errors, "passed": bool(condition_passed)})
    left, right = comparisons
    a, b = left["prediction"]["U"], right["prediction"]["U"]
    ordering = a is not None and b is not None and (a - b) * (left["measured"]["U"] - right["measured"]["U"]) >= 0
    passed &= ordering
    spec["validation"] = {"conditions": comparisons, "pair_ordering_consistent": bool(ordering),
        "scope": "Two new heldout conditions; no physical all-pool optimality or statistical coverage claim"}
    spec["status"] = "VALIDATED_ON_HELDOUT_CONDITIONS" if passed else "VALIDATION_FAILED"
    seal()
    return config, {"schema": SCHEMA, "status": spec["status"], "version_sha256": spec["version_sha256"],
        "development_conditions": len(development), "validation_conditions": len(holdout),
        "validation": spec["validation"], "gpu_started": False, "scope": spec["scope"]}
