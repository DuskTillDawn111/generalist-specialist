"""Prepare static-study calibration and confirmation packets without GPU work.

Job identities bind deployment, traffic, seed, input, workers and code. They are
independent of the set of unused alternatives in a model pool. Native execution
uses the six-term evidence contract; no historical absolute quality gate applies.
"""
import copy
import itertools
import json
from pathlib import Path

from .config import SystemConfig
from .evidence import checkpoint, digest
from .io import save
from .quality import quality_digest
from .trace import make_trace, trace_digest
from .energy import deployed_measurement_identity


def read(path):
    return json.loads(Path(path).read_text())


def _file(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def _bindings(inputs, root):
    files = {str(p.resolve()): _file(p) for p in Path(inputs).rglob("*") if p.is_file()}
    for source in (root / "gsdeploy", root / "scripts"):
        for path in source.rglob("*.py"):
            files[str(path.resolve())] = _file(path)
    sources = read(Path(inputs) / "source-bindings.json")
    for source, expected in sources.items():
        path = Path(source).resolve()
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Changed frozen source binding: " + str(path))
        files[str(path)] = {"path": str(path), "sha256": expected}
    return list(files.values())


def validate_packet(packet):
    """Read-only validation: no model imports, GPU queries or subprocesses."""
    changed = [b["path"] for b in packet["bindings"]
               if not Path(b["path"]).is_file() or digest(b["path"]) != b["sha256"]]
    for job in packet["jobs"]:
        if quality_digest(job["identity"]) != job["id"]:
            changed.append("job identity: " + job["id"])
        identity = job["identity"]
        if (job["plan"]["deployment"] != identity["deployment"]
                or job["plan"]["routing"] != identity["routing"] or job["seed"] != identity["seed"]
                or {m: packet["workers"][m] for m in identity["deployment"]} != identity["workers"]):
            changed.append("job payload: " + job["id"])
        config = copy.deepcopy(packet["config"])
        config["horizon_seconds"], config["random_seed"] = identity["horizon_seconds"], identity["seed"]
        if quality_digest(config) != job["provenance"]["source_problem_sha256"]:
            changed.append("job problem: " + job["id"])
    if quality_digest(packet["bindings"]) != packet["frozen_files_sha256"]:
        changed.append("file binding set")
    return {"valid": not changed, "changed": changed, "gpu_started": False,
            "jobs": len(packet["jobs"]), "scope": "frozen files and job identity only; GPU readiness unverified"}


def _candidate_calibrations(config):
    """Predeclared coverage proposals; no utility-based winner selection."""
    models = config.catalog["models"]
    generals = sorted(m for m, s in models.items() if s["role"] == "generalist")
    specialists = {t: sorted(m for m, s in models.items() if s["role"] == "specialist"
                            and t in s["supported_tasks"]) for t in config.workload}
    for g in generals:
        yield f"shared_{g}", {g: 1}, {t: {g: q} for t, q in config.workload.items()}
    if all(specialists.values()):
        route = {t: {ms[0]: config.workload[t]} for t, ms in specialists.items()}
        yield "specialists_shared_board", {m: 1 for r in route.values() for m in r}, route
    # Pair each generalist with one specialist. Adjacent fractions include a
    # directly paired no-split/split condition at the identical static workload.
    for task in ("text", "asr", "ocr", "image"):
        for g, specialist in itertools.product(generals, specialists.get(task, [])):
            for share in (1.0, .75, .5, .25):
                route = {t: {g: q} for t, q in config.workload.items()}
                rate = config.workload[task]
                route[task] = {g: rate * (1 - share), specialist: rate * share}
                yield f"mixed_{g}_{specialist}_specialist_share_{share:g}", {g: 1, specialist: 1}, route


def _job(config, workers, plan, label, phase, seed, bindings, warmup_seconds):
    deployment, routing = plan["deployment"], plan["routing"]
    root = Path(__file__).resolve().parents[1]
    runtime_sources = [b for b in bindings if
        str(root / "gsdeploy/runtime") + "/" in b["path"]
        or Path(b["path"]).parent == root / "gsdeploy"
        or b["path"] == str(root / "scripts/static_study_worker.py")]
    for model in deployment:
        command = workers[model]["command"]
        if "--source-root" in command:
            prefix = str(Path(command[command.index("--source-root") + 1]).resolve()) + "/"
            runtime_sources.extend(b for b in bindings if b["path"].startswith(prefix) and b["path"].endswith(".py"))
    runtime_sources = sorted({b["path"]: b for b in runtime_sources}.values(), key=lambda b: b["path"])
    identity = {
        "schema": "gsdeploy.static_measurement.v1", "deployment": deployment, "routing": routing,
        "workload": config.workload, "horizon_seconds": config.horizon_seconds,
        "seed": seed, "arrival_process": config.arrival_process,
        "drain_seconds": config.drain_seconds, "warmup_seconds": warmup_seconds,
        "warmup_policy": "largest_three_per_task",
        "warmup_size_metric": "media_file_bytes_or_text_utf8_bytes_v1",
        "deployed_configuration": deployed_measurement_identity(config.catalog, deployment, routing),
        "hardware_contract": {"single_gpu": True, "memory_budget_mib": config.gpu_memory_mib,
                              "device_uuid": config.catalog.get("measurement_context", {}).get("gpu_uuid", "UNBOUND_UNTIL_EXECUTION")},
        "workers": {m: workers[m] for m in sorted(deployment)},
        "input_pool_sha256": config.catalog["quality"]["identity"]["input_pool_sha256"],
        "reference_sha256": config.catalog["quality"]["identity"]["reference_sha256"],
        "runtime_code_sha256": quality_digest(runtime_sources),
        "trace_sha256": trace_digest(make_trace(config.workload, config.horizon_seconds, seed, config.arrival_process)),
    }
    return {"id": quality_digest(identity), "identity": identity, "labels": [label],
            "phase": phase, "plan": plan, "seed": seed, "status": "PREPARED_NOT_EXECUTED",
            "provenance": {"source_problem_sha256": quality_digest({**config.as_dict(), "random_seed": seed}),
                           "frozen_files_sha256": quality_digest(bindings)},
            "gpu_started": False, "native_full_checkpoint_binding": "PENDING"}


def confirmation_candidates(config):
    """Re-solve strong competitors in the identical frozen MILP domain."""
    from .milp import JointMilpOptimizer
    from .system import DeploymentSystem

    def solve(**constraints):
        c = config
        result = JointMilpOptimizer(c.catalog, c.workload, c.gpu_memory_mib,
            c.latency_reference_ms, c.weights, search_max_states=c.search_max_states,
            search_policy=c.search_policy, milp_max_iterations=c.milp_max_iterations,
            milp_time_limit_seconds=c.milp_time_limit_seconds, milp_max_columns=c.milp_max_columns,
            milp_absolute_gap=c.milp_absolute_gap, milp_capacity_witness=c.milp_capacity_witness,
            route_pair_denominators_by_task=c.route_pair_denominators_by_task,
            **DeploymentSystem(c).context()).solve(K=c.route_grid, **constraints)
        return result

    best = solve()
    winner = best.get("winner")
    if not winner or winner.get("prediction", {}).get("U") is None:
        return {"status": "WAITING_GPU_CALIBRATION", "search": best, "candidates": []}
    alternatives = [("free_search_winner", best),
        ("strongest_alternative_deployment", solve(exclude_deployments=[winner["deployment"]])),
        ("strongest_alternative_route", solve(deployment=winner["deployment"], exclude_plans=[winner])),
        ("best_no_split", solve(no_split=True))]
    return {"status": "READY_FOR_CONFIRMATION" if best.get("optimality_certified") else "INCUMBENT_ONLY_UNRESOLVED_GAP",
            "candidates": [{"label": label, "search": result, "plan": result["winner"]}
                           for label, result in alternatives if result.get("winner")],
            "adaptive_slot": {"reserved": 1, "selection": "highest valid unresolved upper bound; no fabricated fifth candidate"}}


def prepare_experiments(inputs, output, root=None, config_path=None):
    inputs, output = Path(inputs).resolve(), Path(output).resolve()
    root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    config = SystemConfig.load(config_path or inputs / "system-config.json")
    contract, workers = read(inputs / "study-contract.json"), read(inputs / "worker-registry.json")
    bindings = _bindings(inputs, root)
    if config_path:
        bindings.append(_file(config_path))
    binding_hash = quality_digest(bindings)
    horizon = contract["calibration_horizon_seconds"]
    from dataclasses import replace
    calibration_config = replace(config, horizon_seconds=horizon)
    proposals, skips = [], []
    for label, deployment, routing in _candidate_calibrations(calibration_config):
        try:
            diagnostic = calibration_config.validate_plan(deployment, routing)
            if not diagnostic["constraints_valid"]:
                raise ValueError("Unknown standalone transfer capacity")
            proposals.append((label, {"deployment": deployment, "routing": routing}, diagnostic))
        except ValueError as exc:
            skips.append({"label": label, "reason": str(exc)})
    # Prefer shared multi-task workers, all-specialist board, and paired routes.
    selected = [p for p in proposals if p[0].startswith("shared_") or p[0] == "specialists_shared_board"]
    for task in ("text", "asr"):
        group = sorted((p for p in proposals if p[0].startswith("mixed_")
                        and any(config.catalog["models"][m]["role"] == "specialist"
                                and task in config.catalog["models"][m]["supported_tasks"]
                                for m in p[1]["deployment"])),
                       key=lambda p: ("generalist_2" not in p[0], p[0]))
        if group:
            model = next(m for m in group[0][1]["deployment"] if config.catalog["models"][m]["role"] == "generalist")
            pair = [p for p in group if model in p[1]["deployment"]]
            unsplit = next((p for p in pair if p[0].endswith("share_1")), None)
            split = next((p for p in pair if p[0].endswith("share_0.75")), None)
            selected.extend(p for p in (unsplit, split) if p is not None)
    for task in ("ocr", "image"):
        proposal = next((p for p in proposals if "generalist_2" in p[0]
                         and p[0].endswith("share_0.75") and task in p[1]["deployment"]), None)
        if proposal is not None and len(selected) < 8:
            selected.append(proposal)
    for proposal in proposals:
        if len(selected) >= 8:
            break
        if proposal not in selected:
            selected.append(proposal)
    selected = selected[:8]
    jobs = {}
    for label, plan, proxy in selected:
        job = _job(calibration_config, workers, plan, label, "calibration", contract["seeds"]["development"][0],
                   bindings, contract["warmup"]["seconds"])
        job["proxy_constraints"] = proxy
        if job["id"] in jobs:
            jobs[job["id"]]["labels"].append(label)
        else:
            jobs[job["id"]] = job
    # Calibration absence is decisive: standalone evidence cannot support a
    # six-term winner, regardless of how attractive partial scores look.
    phase6 = {"status": "WAITING_GPU_CALIBRATION", "candidates": [], "adaptive_slots_reserved": 1,
              "reason": "No frozen mixed service/whole-board energy evaluator; no real six-term winner claimed"}
    if config.catalog.get("performance_mode") == "profiled" and config.catalog.get("energy_profiles"):
        phase6 = confirmation_candidates(config)
        for item in phase6["candidates"]:
            for seed in contract["seeds"]["confirmation"]:
                job = _job(config, workers, item["plan"], item["label"], "confirmation", seed,
                           bindings, contract["warmup"]["seconds"])
                if job["id"] in jobs:
                    jobs[job["id"]]["labels"].append(item["label"])
                else:
                    jobs[job["id"]] = job
    packet = {"schema": "gsdeploy.static_experiments.v1", "status": "CPU_PREPARED_GPU_NOT_STARTED",
              "inputs": str(inputs), "config": config.as_dict(), "workers": workers,
              "bindings": bindings, "frozen_files_sha256": binding_hash,
              "gpu_started": False, "jobs": list(jobs.values()),
              "calibration": {"status": "READY_FOR_FUTURE_GPU" if len(jobs) >= 6 else "INSUFFICIENT_VALID_CALIBRATION_PROPOSALS",
                              "target_conditions": "6-8", "selected": len(selected), "skipped": skips,
                              "horizon_seconds": horizon, "formal_horizon_seconds": config.horizon_seconds,
                              "selection": "predeclared coverage proposals passing standalone proxy constraints; no utility ranking",
                              "measurement_scope": "Mixed capacity, peak memory, tails and energy remain unverified"},
              "confirmation": phase6,
              "execution": {"policy": "Explicit --execute only; serial jobs; no launch during prepare/validate",
                            "full_checkpoint_binding": "CPU helper bind_native_job available; not performed during prepare",
                            "warmup_policy": "largest_three_per_task",
                            "warmup_runtime_support_required": True}}
    save(output / "experiment-packet.json", packet)
    return packet


def bind_native_job(packet, job, output):
    """CPU-only full checkpoint/report binding, potentially reads large weights.

    Write a distinct native config; never mutate the frozen input bundle. This
    step deliberately happens only when explicitly invoked, not at prepare time.
    """
    config = copy.deepcopy(packet["config"])
    workers = copy.deepcopy(packet["workers"])
    source_quality = config["catalog"]["quality"]
    models = {}
    for model in job["plan"]["deployment"]:
        worker = workers[model]
        model_path = Path(worker["env"]["GS_MODEL_PATH"]).resolve()
        inventory = worker["checkpoint"]
        if model_path != Path(inventory["path"]).resolve() or not inventory.get("ready"):
            raise ValueError("Runtime checkpoint does not match frozen inventory: " + model)
        selected_files = None
        if model == "ocr":
            selected_files = sorted({str(p.resolve().relative_to(model_path))
                for item in inventory["files"] for p in Path(item["path"]).parent.rglob("*")
                if p.is_file() and not {".cache", ".git", "__pycache__"}.intersection(p.parts)})
            worker["env"]["GS_CHECKPOINT_FILES_JSON"] = json.dumps(selected_files)
        actual = checkpoint(model_path, selected_files)
        receipt_checks = []
        for item in inventory["files"]:
            # Snapshot files can legitimately symlink a shared immutable blob.
            # Bind the lexical inventory name and hash its dereferenced bytes.
            path = Path(item["path"]).absolute()
            if not path.is_relative_to(model_path) or path.stat().st_size != item["size"]:
                raise ValueError("Checkpoint inventory changed: " + str(path))
            actual_sha = actual["files"].get(str(path.relative_to(model_path)))
            historic_sha = item.get("historical_sha256")
            if historic_sha and actual_sha != historic_sha:
                raise ValueError("Checkpoint historical receipt hash mismatch: " + str(path))
            receipt_checks.append({"path": str(path), "current_sha256": actual_sha,
                                   "resolved_target": str(path.resolve()),
                                   "historical_sha256": historic_sha, "historical_hash_verified": bool(historic_sha)})
        reports = {}
        historical_identities = {}
        for task, record in source_quality["paths"][model].items():
            historical_identities[task] = record.get("checkpoint_identity", record.get("checkpoint_sha256"))
            identity = record.get("checkpoint_identity", {})
            if identity.get("kind") == "configuration_manifest":
                original = read(identity["source"])
                receipt = {"checkpoint": original["resources"]["checkpoint"], "execution": original["execution"],
                           "adapter": original.get("adapter"), "peft_adapter": original["execution"].get("peft_adapter")}
                if quality_digest(receipt) != identity["sha256"] or original["execution"] != worker["execution"]:
                    raise ValueError("Historical quality configuration binding changed: " + model + "/" + task)
            for score in record.get("samples", {}).values():
                path = score.get("source")
                if path:
                    reports[path] = {"path": path, "sha256": digest(path)}
        models[model] = {"checkpoint": actual, "quality_reports": list(reports.values()),
                         "historical_quality_checkpoint_identities": historical_identities,
                         "runtime_checkpoint_scope": "current_execution_only_not_retroactive_offline_hash",
                         "receipt_checks": receipt_checks}
    config["horizon_seconds"] = job["identity"]["horizon_seconds"]
    config["random_seed"] = job["seed"]
    config["catalog"]["measurement_context"] = {
        **(config["catalog"].get("measurement_context") or {}),
        "runtime_checkpoint_sha256": {m: value["checkpoint"]["sha256"] for m, value in models.items()},
        "historical_quality_checkpoint_equivalence": "receipt_scope_only; actual run outputs rescored",
    }
    manifest = {"schema_version": 2, "contract": "quality_evidence_v1",
                "quality_sha256": quality_digest(source_quality), "models": models}
    output = Path(output)
    save(output / "native-config.json", config)
    save(output / "quality-evidence.json", manifest)
    save(output / "worker-registry.json", workers)
    return config, output / "quality-evidence.json", workers


def execute_job(packet, job_id, output, *, execute=False):
    if not execute:
        raise ValueError("Native execution requires explicit execute=True")
    check = validate_packet(packet)
    if not check["valid"]:
        raise ValueError("Changed frozen experiment inputs: " + repr(check["changed"]))
    job = next(j for j in packet["jobs"] if j["id"] == job_id)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Experiment output already exists: " + str(output))
    # This low-level runtime entry collects calibration evidence. It does not
    # promote the transfer proxy to a certified optimum or reuse an old gate.
    from .runtime.runner import run
    import inspect
    if "priming_policy" not in inspect.signature(run).parameters:
        raise ValueError("Runtime lacks frozen largest-three priming protocol; update runtime before GPU execution")
    config, manifest, workers = bind_native_job(packet, job, output / "native-binding")
    # A measurement-only collection may have no matching frozen profile yet.
    # Use the declared standalone transfer proxy only to prepare/monitor the
    # run; never reuse this collection configuration as an optimality proof.
    frozen_mode = config["catalog"].get("performance_mode", "independent")
    if frozen_mode in {"profiled", "calibrated_surrogate", "calibrated_replay_v2"}:
        config["catalog"]["performance_mode"] = "task_replica_proxy"
    config["catalog"]["memory_mode"] = "standalone_proxy"
    import subprocess
    device = next(iter(workers.values()))["env"].get("CUDA_VISIBLE_DEVICES", "0")
    uuid = subprocess.check_output(["nvidia-smi", "-i", device, "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=5).strip()
    if not uuid or "\n" in uuid:
        raise ValueError("Expected one GPU UUID for native measurement identity")
    config["catalog"]["measurement_context"]["gpu_uuid"] = uuid
    save(output / "collection-config.json", config)
    raw = run(SystemConfig(**config), job["plan"],
              read(Path(packet["inputs"]) / "study-contract.json")["runtime_input_pool"],
              backend="native", workers=workers,
              duration=job["identity"]["horizon_seconds"], seed=job["seed"],
              arrival_process=config["arrival_process"], warmup=job["identity"]["warmup_seconds"],
              drain_timeout=config["drain_seconds"], admission=str(manifest),
              priming_policy=job["identity"]["warmup_policy"])
    raw["experiment_job_id"] = job_id
    raw["experiment_phase"] = job["phase"]
    if packet.get("confirmation_sha256"):
        raw["confirmation_packet_sha256"] = packet["confirmation_sha256"]
    raw["collection_contract"] = {"frozen_performance_mode": frozen_mode,
                                  "collection_performance_mode": config["catalog"].get("performance_mode"),
                                  "optimality_transferred": False,
                                  "reason": "Exact unmeasured conditions are collected without inventing a profile"}
    actual_identity = copy.deepcopy(job["identity"])
    actual_identity["hardware_contract"]["device_uuid"] = raw.get("preflight", {}).get("gpu", {}).get("uuid")
    actual_identity["runtime_checkpoint_sha256"] = config["catalog"]["measurement_context"]["runtime_checkpoint_sha256"]
    actual_identity["workers"] = {m: workers[m] for m in job["plan"]["deployment"]}
    raw["executed_measurement_identity"] = actual_identity
    raw["executed_measurement_id"] = quality_digest(actual_identity)
    raw["measurement_claim"] = "calibration observation; no optimality claim" if job["phase"] == "calibration" else "independent confirmation observation"
    save(output / "run.json", raw)
    return raw


def freeze_evaluator(packet, run_paths, output):
    """Import exact measured conditions; never extrapolate eight calibration runs.

    Mean service is a declared FIFO predictor, not a measured latency tail.
    Different H/seed/route conditions remain unknown. Repeated measurements of
    an identical condition are pooled; memory uses their maximum safe residency.
    """
    import math
    from .energy import energy_key
    from .performance import profile_key
    from .evaluation import evaluate_run

    valid = validate_packet(packet)
    if not valid["valid"]:
        raise ValueError("Changed packet bindings: " + repr(valid["changed"]))
    config = copy.deepcopy(packet["config"])
    catalog = config["catalog"]
    jobs = {j["id"]: j for j in packet["jobs"]}
    imported, runtime_hashes, uuids = [], {}, set()
    for path in run_paths:
        path = Path(path).resolve()
        raw = read(path)
        job = jobs.get(raw.get("experiment_job_id"))
        if not job or raw.get("backend") != "native" or raw.get("status") != "completed":
            raise ValueError("Only completed native runs from this packet can be imported: " + str(path))
        if job.get("phase") != "calibration":
            raise ValueError("Independent confirmation/adaptive runs cannot fit the evaluator; explicitly reclassify a new development study first")
        expected = job["identity"]
        actual_identity = raw.get("executed_measurement_identity", {})
        if not actual_identity or raw.get("executed_measurement_id") != quality_digest(actual_identity):
            raise ValueError("Executed measurement identity missing or invalid: " + str(path))
        for field, value in expected.items():
            if field not in {"hardware_contract", "workers"} and actual_identity.get(field) != value:
                raise ValueError("Executed identity differs from prepared condition: " + field)
        if {k: v for k, v in actual_identity.get("hardware_contract", {}).items() if k != "device_uuid"} != {
                k: v for k, v in expected["hardware_contract"].items() if k != "device_uuid"}:
            raise ValueError("Executed hardware contract changed")
        for model, expected_worker in expected["workers"].items():
            current_worker = copy.deepcopy(actual_identity["workers"][model])
            current_worker.get("env", {}).pop("GS_CHECKPOINT_FILES_JSON", None)
            if current_worker != expected_worker:
                raise ValueError("Actual worker differs from frozen worker: " + model)
        source_config = SystemConfig(**raw["config"])
        if (raw["plan"]["deployment"] != expected["deployment"]
                or raw["plan"]["routing"] != expected["routing"]
                or source_config.workload != expected["workload"]
                or source_config.horizon_seconds != expected["horizon_seconds"]
                or source_config.random_seed != expected["seed"]
                or source_config.arrival_process != expected["arrival_process"]
                or raw.get("transport_config", {}).get("random_seed") != expected["seed"]
                or raw.get("transport_config", {}).get("arrival_process") != expected["arrival_process"]
                or raw.get("measurement_window", {}).get("trace_sha256") != expected["trace_sha256"]
                or raw.get("transport_config", {}).get("warmup_seconds") != expected["warmup_seconds"]
                or raw.get("transport_config", {}).get("measurement_seconds") != expected["horizon_seconds"]
                or raw.get("transport_config", {}).get("drain_timeout_seconds") != expected["drain_seconds"]
                or raw.get("priming_policy") != expected["warmup_policy"]
                or raw.get("priming_size_metric") != expected["warmup_size_metric"]
                or quality_digest(source_config.catalog["quality"]) != quality_digest(catalog["quality"])):
            raise ValueError("Run differs from frozen measurement condition: " + str(path))
        evaluation = evaluate_run(source_config, raw)
        if not evaluation["utility"].get("eligible"):
            raise ValueError("Native run is not valid for complete measured utility: " + str(path))
        energy = raw.get("energy_evidence", {})
        if energy.get("kind") != "measured" or not energy.get("coverage_valid"):
            raise ValueError("Missing measured whole-board energy: " + str(path))
        context = source_config.catalog.get("measurement_context", {})
        gpu = raw.get("preflight", {}).get("gpu", {}).get("uuid")
        if not gpu or gpu != context.get("gpu_uuid"):
            raise ValueError("Actual GPU UUID is not bound consistently: " + str(path))
        uuids.add(gpu)
        for model in expected["deployment"]:
            checkpoint_hash = context.get("runtime_checkpoint_sha256", {}).get(model)
            if not checkpoint_hash or (model in runtime_hashes and runtime_hashes[model] != checkpoint_hash):
                raise ValueError("Runtime checkpoint missing or changed across calibration: " + model)
            runtime_hashes[model] = checkpoint_hash
        imported.append((job, raw, _file(path), evaluation))
    if not imported or len(uuids) != 1:
        raise ValueError("Freeze requires nonempty measured runs on one declared GPU")
    catalog["performance_mode"] = "profiled"
    catalog["memory_mode"] = "profiled"
    catalog["measurement_context"] = {"gpu_uuid": next(iter(uuids)), "runtime_checkpoint_sha256": runtime_hashes,
        "historical_quality_checkpoint_equivalence": "receipt_scope_only; actual run outputs rescored"}
    catalog["colocation_profiles"], catalog["energy_profiles"] = {}, {}
    groups = {}
    for item in imported:
        job, raw, binding, evaluation = item
        identity = job["identity"]
        key = profile_key(catalog, identity["deployment"], identity["routing"], identity["workload"],
                          identity["horizon_seconds"], identity["seed"], identity["arrival_process"])
        groups.setdefault(key, []).append(item)
    for key, items in groups.items():
        first = items[0][0]["identity"]
        service_samples = {}
        for job, raw, binding, evaluation in items:
            for row in raw["request_records"]:
                model = raw["replica_to_family"][row["model"]]
                value = row.get("worker_latency_ms")
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Missing actual per-request worker service time")
                service_samples.setdefault(model, {}).setdefault(row["task"], []).append(value)
        service = {m: {t: math.fsum(values) / len(values)
                       * catalog["capacity_factors"][m][str(first["deployment"][m])]
                       / first["deployment"][m] for t, values in rows.items()}
                   for m, rows in service_samples.items()}
        for task, row in first["routing"].items():
            if any(q > 0 and task not in service.get(m, {}) for m, q in row.items()):
                raise ValueError("Measured condition lacks a routed service path")
        bindings = [item[2] for item in items]
        profile = {"kind": "measured", "scope": "whole_gpu_colocation", "coverage_valid": True,
                   "service_time_ms": service,
                   "safe_memory_mib": max(item[1]["memory_evidence"]["safe_memory_mib"] for item in items),
                   "provenance": bindings, "replicate_count": len(items),
                   "prediction_scope": "exact-condition measured worker means in constant-service FIFO; no measured tail model or extrapolation"}
        catalog["colocation_profiles"][key] = profile
        ekey = energy_key(first["deployment"], first["routing"], first["workload"], first["horizon_seconds"],
                          first["seed"], first["arrival_process"], catalog=catalog)
        catalog["energy_profiles"][ekey] = {"kind": "measured", "scope": "whole_gpu_board", "coverage_valid": True,
            "run_joules": math.fsum(item[1]["energy_evidence"]["run_joules"] for item in items) / len(items),
            "startup_joules": math.fsum(item[1]["energy_evidence"]["startup_joules"] for item in items) / len(items),
            "provenance": bindings, "profile_key": ekey, "replicate_count": len(items)}
    revision = quality_digest({"catalog": catalog, "runs": [item[2] for item in imported]})
    report = {"schema": "gsdeploy.frozen_exact_evaluator.v1", "version_sha256": revision,
              "status": "EXACT_MEASURED_CONDITIONS_ONLY", "gpu_started": False,
              "conditions": len(groups), "runs": [item[2] for item in imported],
              "scope": "exact measured plan/route/H/seed only; constant-mean FIFO latency and offline quality replay remain explicit predictors",
              "unknown_policy": "unmeasured conditions return unknown, remain in gap, and enter a deduplicated measurement queue",
              "cache_policy": "instantiate a new DeploymentSystem from this config; discard all prior cuts, caches and certificates",
              "formal_horizon_seconds": config["horizon_seconds"],
              "calibration_does_not_transfer_to_formal_window": True}
    output = Path(output)
    save(output / "system-config.json", config)
    save(output / "evaluator-freeze.json", report)
    return report


def measurement_queue(config, plans, workers, bindings, seed=None, warmup_seconds=30):
    """Return unique exact conditions needing evidence, with honest unknowns."""
    from .prediction import predict
    jobs, evaluated = {}, []
    for plan in plans:
        result = predict(config.catalog, plan["deployment"], plan["routing"], config.workload,
                         config.gpu_memory_mib, config.latency_reference_ms, config.weights,
                         horizon=config.horizon_seconds, seed=config.random_seed if seed is None else seed,
                         arrival_process=config.arrival_process, drain_seconds=config.drain_seconds,
                         energy_reference_watts=config.energy_reference_watts,
                         startup_amortization_seconds=config.startup_amortization_seconds)
        state = "scored" if result.get("eligible") else (
            "needs_measurement" if result.get("performance_unknown") or result.get("memory_unknown") or "MISSING" in result.get("reason", "")
            else "invalid_under_frozen_model")
        evaluated.append({"plan": plan, "state": state, "prediction": result})
        if state == "needs_measurement":
            job = _job(config, workers, plan, "unresolved_search_candidate", "calibration", config.random_seed if seed is None else seed,
                       bindings, warmup_seconds)
            job["measurement_role"] = "adaptive_development; not independent confirmation"
            jobs.setdefault(job["id"], job)
    return {"evaluated": evaluated, "jobs": list(jobs.values()), "gpu_started": False,
            "scope": "queue preparation only; no upper-bound claim from prediction ranking"}


def queue_packet(packet, config, plans, extra_bindings=()):
    """Produce a validate/run-compatible derived packet for unmeasured plans."""
    if not validate_packet(packet)["valid"]:
        raise ValueError("Source packet bindings changed")
    bindings = list({b["path"]: b for b in [*packet["bindings"], *extra_bindings]}.values())
    result = measurement_queue(config, plans, packet["workers"], bindings)
    return {**copy.deepcopy(packet), "status": "CPU_PREPARED_ADAPTIVE_GPU_NOT_STARTED",
            "config": config.as_dict(), "bindings": bindings, "frozen_files_sha256": quality_digest(bindings),
            "jobs": result["jobs"], "queue_evaluated": result["evaluated"], "gpu_started": False,
            "confirmation": {"status": "WAITING_REQUESTED_MEASUREMENTS", "candidates": []}}
