"""Independent paired confirmation of a frozen deployment search, without refitting.

Preparation and verification are CPU-only. Native execution is explicit, serial,
and scoped to this sealed confirmation packet. Physical optimality is not claimed.
"""
import copy
from dataclasses import replace
import math
from pathlib import Path

from .config import SystemConfig
from .io import save
from .quality import quality_digest
from .static_study import _bindings, _file, _job, execute_job, read, validate_packet

SCHEMA = "gsdeploy.independent_confirmation.v1"
SEEDS = (43, 44, 45)
HORIZON = 1800
CANDIDATE_LABELS = (
    "free_search_winner", "strongest_alternative_deployment",
    "strongest_alternative_route", "best_no_split",
    "best_generalist_only", "best_specialist_only",
)


def _seal(packet):
    return quality_digest({k: v for k, v in packet.items() if k != "confirmation_sha256"})


def _validated(config):
    if config.catalog.get("performance_mode") == "calibrated_replay_v2":
        from .calibration_v2 import validated_replay_spec
        spec = validated_replay_spec(config.catalog, config.workload)
    else:
        from .calibration import validated_spec
        spec = validated_spec(config.catalog, config.workload)
    if config.horizon_seconds != HORIZON or config.random_seed != 42:
        raise ValueError("Search must use the frozen development seed 42 and formal H1800")
    return spec


def _plan(value):
    return {"deployment": {m: n for m, n in value["deployment"].items() if n > 0},
            "routing": {t: {m: q for m, q in row.items() if q > 0}
                        for t, row in value["routing"].items()}}


def _check_worker_source_coverage(workers, bindings):
    """The native launcher and its wire protocol may live outside the snapshot."""
    covered = {b["path"] for b in bindings}
    for worker in workers.values():
        for argument in worker["command"]:
            if Path(argument).name != "static_study_worker.py":
                continue
            script = Path(argument).resolve()
            protocol = script.parents[1] / "gsdeploy/runtime/protocol.py"
            if not {str(script), str(protocol)} <= covered:
                raise ValueError("Actual native launcher/protocol is not bound to the confirmation packet")


def _prepared_runtime_hash(packet, deployment):
    """Rebuild the runtime identity within the original confirmation snapshot."""
    bindings = packet["bindings"]
    root = Path(packet["identity_runtime_root"])
    selected = [b for b in bindings if str(root / "gsdeploy/runtime") + "/" in b["path"]
                or Path(b["path"]).parent == root / "gsdeploy"
                or b["path"] == str(root / "scripts/static_study_worker.py")]
    for model in deployment:
        command = packet["workers"][model]["command"]
        if "--source-root" in command:
            prefix = str(Path(command[command.index("--source-root") + 1]).resolve()) + "/"
            selected.extend(b for b in bindings if b["path"].startswith(prefix) and b["path"].endswith(".py"))
    return quality_digest(sorted({b["path"]: b for b in selected}.values(), key=lambda b: b["path"]))


def _check_search(config, bundle):
    if bundle.get("config_sha256") != quality_digest(config.as_dict()):
        raise ValueError("Frozen search does not bind the validated configuration")
    records = bundle.get("candidates", [])
    mapping = {row["label"]: row["search"] for row in records}
    if len(mapping) != len(records) or set(mapping) != set(CANDIDATE_LABELS):
        raise ValueError("Search must explicitly record all six comparison roles")
    winner = mapping["free_search_winner"].get("winner")
    if not winner:
        raise ValueError("Independent confirmation requires a frozen search incumbent")
    winner = _plan(winner)
    for label, result in mapping.items():
        expected_problem = config.as_dict()
        if label in {"best_generalist_only", "best_specialist_only"}:
            role = "generalist" if label == "best_generalist_only" else "specialist"
            selected = {m for m, s in config.catalog["models"].items() if s["role"] == role}
            for field in ("models", "capacity_factors", "memory_table_mib"):
                expected_problem["catalog"][field] = {
                    m: v for m, v in expected_problem["catalog"][field].items() if m in selected}
        if result.get("problem") != expected_problem:
            raise ValueError("Search problem differs from the frozen role domain: " + label)
        candidate = result.get("winner")
        if candidate is None:
            continue  # Preserve infeasible/limited/unknown domains without inventing a control.
        plan = _plan(candidate)
        if not config.validate_plan(**plan)["constraints_valid"]:
            raise ValueError("Candidate has unknown constraints: " + label)
        if label == "strongest_alternative_deployment" and plan["deployment"] == winner["deployment"]:
            raise ValueError("Alternative deployment repeats the winner deployment")
        if label == "strongest_alternative_route" and (
                plan["deployment"] != winner["deployment"] or plan == winner):
            raise ValueError("Alternative route must differ on the same deployment")
        if label == "best_no_split" and any(len(row) > 1 for row in plan["routing"].values()):
            raise ValueError("No-split control contains a split route")
        if label in {"best_generalist_only", "best_specialist_only"}:
            role = "generalist" if label == "best_generalist_only" else "specialist"
            if any(config.catalog["models"][m]["role"] != role for m in plan["deployment"]):
                raise ValueError("Pure control contains the other model role")
    return mapping


def build_confirmation_packet(source_packet_path, config_path, search_results_path, output=None, *, root=None,
                              checkpoint_bindings_path=None):
    """Seal six candidate roles and their common-seed blocks; never start a GPU.

    Search JSON is {config_sha256, candidates: [{label, search: full_result}]}.
    Every role is retained, including a null winner with its unresolved domain.
    Equal physical plans are run once per seed and keep all comparison labels.
    """
    source_packet_path, config_path, search_results_path = map(
        lambda p: Path(p).resolve(), (source_packet_path, config_path, search_results_path))
    source = read(source_packet_path)
    if not validate_packet(source)["valid"]:
        raise ValueError("Source packet bindings changed")
    config = SystemConfig.load(config_path)
    spec = _validated(config)
    search = read(search_results_path)
    roles = _check_search(config, search)
    inputs = Path(source["inputs"]).resolve()
    contract = read(inputs / "study-contract.json")
    if (contract["formal_horizon_seconds"] != HORIZON
            or contract["seeds"]["confirmation"] != list(SEEDS)
            or contract["seeds"]["development"] != [42]
            or contract["workload"] != config.workload):
        raise ValueError("Confirmation differs from the frozen study contract")
    if quality_digest(source["config"]["catalog"]["quality"]) != quality_digest(config.catalog["quality"]):
        raise ValueError("Confirmation changed the frozen input/reference/quality contract")
    workers = copy.deepcopy(source["workers"])
    root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    bindings = _bindings(inputs, root)
    bindings = list({b["path"]: b for b in [*bindings, *map(_file,
        (source_packet_path, config_path, search_results_path))]}.values())
    _check_worker_source_coverage(workers, bindings)
    checkpoint_hashes = dict(config.catalog["measurement_context"]["runtime_checkpoint_sha256"])
    fitted_models = set(checkpoint_hashes)
    checkpoint_manifests = {}
    if checkpoint_bindings_path is not None:
        checkpoint_bindings_path = Path(checkpoint_bindings_path).resolve()
        report = read(checkpoint_bindings_path)
        bindings.append(_file(checkpoint_bindings_path))
        for model, record in report["models"].items():
            if model not in config.catalog["models"]:
                continue
            value = record["checkpoint"]["sha256"]
            manifest_path = Path(record["manifest"])
            manifest = _file(manifest_path if manifest_path.is_absolute() else root / manifest_path)
            if (manifest["sha256"] != record["manifest_sha256"]
                    or read(manifest["path"])["models"][model]["checkpoint"]["sha256"] != value):
                raise ValueError("CPU checkpoint binding receipt changed: " + model)
            if model in checkpoint_hashes and checkpoint_hashes[model] != value:
                raise ValueError("CPU checkpoint binding conflicts with fitted identity: " + model)
            checkpoint_hashes[model] = value
            checkpoint_manifests[model] = manifest
            bindings.append(manifest)
        bindings = list({b["path"]: b for b in bindings}.values())
    candidates, role_records = {}, []
    for label in CANDIDATE_LABELS:
        result = roles[label]
        record = {"label": label, "status": result.get("status"),
                  "optimality_certified": bool(result.get("optimality_certified")),
                  "plan_key": None, "search": result}
        if result.get("winner"):
            plan = _plan(result["winner"])
            missing = set(plan["deployment"]) - set(checkpoint_hashes)
            if missing:
                raise ValueError("WAITING_CONFIRMATION_CHECKPOINT_BINDING: " + ",".join(sorted(missing)))
            key = quality_digest(plan)
            record["plan_key"] = key
            candidates.setdefault(key, {"plan_key": key, "plan": plan, "labels": []})["labels"].append(label)
        role_records.append(record)
    jobs = []
    for seed in SEEDS:  # Same trace/input order across every candidate in a seed block.
        for key, candidate in candidates.items():
            job = _job(config, workers, candidate["plan"], candidate["labels"][0],
                       "confirmation", seed, bindings, contract["warmup"]["seconds"])
            job["labels"] = candidate["labels"]
            job["candidate_key"] = key
            job["purpose"] = "independent_confirmation_not_training"
            job["identity"]["runtime_checkpoint_sha256"] = {
                m: checkpoint_hashes[m] for m in candidate["plan"]["deployment"]}
            job["id"] = quality_digest(job["identity"])
            jobs.append(job)
    packet = {"schema": SCHEMA, "status": "CPU_PREPARED_INDEPENDENT_CONFIRMATION",
        "inputs": str(inputs), "config": config.as_dict(), "workers": workers,
        "bindings": bindings, "frozen_files_sha256": quality_digest(bindings),
        "source_packet": _file(source_packet_path), "validated_config": _file(config_path),
        "identity_runtime_root": str(Path(__file__).resolve().parents[1]),
        "frozen_search": _file(search_results_path), "surrogate_version_sha256": spec["version_sha256"],
        "winner_label": "free_search_winner", "candidate_roles": role_records,
        "execution_checkpoint_contract": {
            "runtime_checkpoint_sha256": checkpoint_hashes,
            "additional_cpu_bound_models": sorted(set(checkpoint_hashes) - fitted_models),
            "receipt": _file(checkpoint_bindings_path) if checkpoint_bindings_path is not None else None,
            "manifest_receipts": checkpoint_manifests,
            "scope": "execution identity only; extra CPU-bound models have no mixed performance calibration claim"},
        "candidates": list(candidates.values()), "jobs": jobs, "gpu_started": False,
        "confirmation": {"seeds": list(SEEDS), "horizon_seconds": HORIZON,
            "warmup_seconds": contract["warmup"]["seconds"], "phase": "confirmation",
            "order": "seed blocks, frozen candidate order within each block",
            "search_certified": bool(roles["free_search_winner"].get("optimality_certified")),
            "scope": "independent paired measurements of frozen candidates; physical all-combination optimum unproven",
            "training_forbidden": True}}
    packet["confirmation_sha256"] = _seal(packet)
    check = validate_confirmation_packet(packet)
    if not check["valid"]:
        raise ValueError("Invalid constructed confirmation packet: " + repr(check["changed"]))
    if output is not None:
        save(output, packet)
    return packet


def validate_confirmation_packet(packet):
    """Validate the complete purpose-bearing manifest, file bindings and blocks."""
    changed = []
    try:
        if packet.get("schema") != SCHEMA or packet.get("confirmation_sha256") != _seal(packet):
            changed.append("confirmation manifest identity")
        if packet.get("identity_runtime_root") != str(Path(__file__).resolve().parents[1]):
            changed.append("confirmation runtime must use its prepared source snapshot")
        config = SystemConfig(**packet["config"])
        spec = _validated(config)
        if spec["version_sha256"] != packet["surrogate_version_sha256"]:
            changed.append("surrogate version")
        changed.extend(validate_packet(packet)["changed"])
        _check_worker_source_coverage(packet["workers"], packet["bindings"])
        if read(packet["validated_config"]["path"]) != packet["config"]:
            changed.append("embedded configuration differs from frozen artifact")
        source = read(packet["source_packet"]["path"])
        if source["workers"] != packet["workers"] or source["inputs"] != packet["inputs"]:
            changed.append("source workers/inputs changed")
        contract = read(Path(packet["inputs"]) / "study-contract.json")
        if (contract["formal_horizon_seconds"] != HORIZON or contract["seeds"]["confirmation"] != list(SEEDS)
                or packet["confirmation"]["warmup_seconds"] != contract["warmup"]["seconds"]):
            changed.append("bound formal study/priming contract")
        roles = _check_search(config, {"config_sha256": quality_digest(packet["config"]),
            "candidates": [{"label": r["label"], "search": r["search"]} for r in packet["candidate_roles"]]})
        if roles != _check_search(config, read(packet["frozen_search"]["path"])):
            changed.append("embedded candidates differ from frozen search artifact")
        candidates = {c["plan_key"]: c for c in packet["candidates"]}
        role_plans = {quality_digest(_plan(result["winner"])) for result in roles.values() if result.get("winner")}
        if set(candidates) != role_plans:
            changed.append("missing or unexpected candidate plan")
        expected_jobs = [(key, seed) for seed in SEEDS for key in candidates]
        if len(candidates) != len(packet["candidates"]) or [
                (j.get("candidate_key"), j.get("seed")) for j in packet["jobs"]] != expected_jobs:
            changed.append("confirmation candidates/seed blocks")
        expected_labels = {key: [label for label in CANDIDATE_LABELS if roles[label].get("winner")
            and quality_digest(_plan(roles[label]["winner"])) == key] for key in candidates}
        for key, candidate in candidates.items():
            if key != quality_digest(candidate["plan"]) or candidate["labels"] != expected_labels[key]:
                changed.append("candidate plan or role labels")
        for job in packet["jobs"]:
            candidate = candidates[job["candidate_key"]]
            if (job["phase"] != "confirmation" or job.get("purpose") != "independent_confirmation_not_training"
                    or job["seed"] not in SEEDS or job["identity"]["horizon_seconds"] != HORIZON
                    or job["plan"] != candidate["plan"] or job["labels"] != candidate["labels"]):
                changed.append("independent job purpose/condition: " + job["id"])
            if job["identity"].get("runtime_checkpoint_sha256") != {
                    m: packet["execution_checkpoint_contract"]["runtime_checkpoint_sha256"][m]
                    for m in job["plan"]["deployment"]}:
                changed.append("resident execution checkpoint contract")
            rebuilt = _job(config, packet["workers"], candidate["plan"], candidate["labels"][0],
                           "confirmation", job["seed"], packet["bindings"], contract["warmup"]["seconds"])
            rebuilt["identity"]["runtime_code_sha256"] = _prepared_runtime_hash(packet, candidate["plan"]["deployment"])
            rebuilt["identity"]["runtime_checkpoint_sha256"] = job["identity"]["runtime_checkpoint_sha256"]
            if rebuilt["identity"] != job["identity"] or quality_digest(rebuilt["identity"]) != job["id"]:
                changed.append("rebuilt frozen confirmation identity: " + job["id"])
        checkpoint_contract = packet["execution_checkpoint_contract"]
        fitted_hashes = config.catalog["measurement_context"]["runtime_checkpoint_sha256"]
        if any(checkpoint_contract["runtime_checkpoint_sha256"].get(m) != h for m, h in fitted_hashes.items()):
            changed.append("execution checkpoint differs from fitted measurement context")
        if checkpoint_contract.get("receipt") is not None and checkpoint_contract["receipt"] not in packet["bindings"]:
            changed.append("unbound CPU checkpoint receipt")
        expected_hashes = dict(fitted_hashes)
        if checkpoint_contract.get("receipt") is not None:
            for model, record in read(checkpoint_contract["receipt"]["path"])["models"].items():
                if model not in config.catalog["models"]:
                    continue
                value = record["checkpoint"]["sha256"]
                receipt = checkpoint_contract["manifest_receipts"][model]
                if (receipt not in packet["bindings"] or receipt["sha256"] != record["manifest_sha256"]
                        or read(receipt["path"])["models"][model]["checkpoint"]["sha256"] != value
                        or (model in fitted_hashes and fitted_hashes[model] != value)):
                    changed.append("invalid CPU checkpoint manifest: " + model)
                expected_hashes[model] = value
        if checkpoint_contract["runtime_checkpoint_sha256"] != expected_hashes:
            changed.append("execution checkpoint union differs from bound receipts")
        if packet["confirmation"]["seeds"] != list(SEEDS) or packet["confirmation"]["horizon_seconds"] != HORIZON:
            changed.append("formal confirmation contract")
        for receipt_key in ("source_packet", "validated_config", "frozen_search"):
            if packet[receipt_key] not in packet["bindings"]:
                changed.append("unbound " + receipt_key)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        changed.append(str(exc))
    return {"valid": not changed, "changed": changed, "jobs": len(packet.get("jobs", [])),
            "gpu_started": False, "scope": "CPU-only independent confirmation manifest and file validation"}


def verify_confirmation_run(packet, job, run_path):
    """Read-only identity validation and measured re-evaluation of one native run."""
    check = validate_confirmation_packet(packet)
    if not check["valid"]:
        raise ValueError("Invalid independent confirmation packet: " + repr(check["changed"]))
    if not any(j == job for j in packet["jobs"]):
        raise ValueError("Job is not an unchanged member of this confirmation packet")
    raw = read(run_path)
    expected, actual = job["identity"], raw.get("executed_measurement_identity", {})
    if (raw.get("experiment_job_id") != job["id"] or raw.get("backend") != "native"
            or raw.get("status") != "completed"
            or raw.get("experiment_phase") != "confirmation"
            or raw.get("confirmation_packet_sha256") != packet["confirmation_sha256"]
            or raw.get("executed_measurement_id") != quality_digest(actual)):
        raise ValueError("Run is not this completed native confirmation job")
    for field, value in expected.items():
        if field not in {"hardware_contract", "workers"} and actual.get(field) != value:
            raise ValueError("Executed confirmation identity differs: " + field)
    expected_hardware = expected["hardware_contract"]
    if actual.get("hardware_contract") != expected_hardware:
        raise ValueError("Executed confirmation hardware differs from the validated GPU contract")
    for model, worker in expected["workers"].items():
        current = copy.deepcopy(actual.get("workers", {}).get(model, {}))
        current.get("env", {}).pop("GS_CHECKPOINT_FILES_JSON", None)
        if current != worker:
            raise ValueError("Executed confirmation worker differs: " + model)
    if set(actual.get("workers", {})) != set(expected["workers"]):
        raise ValueError("Executed confirmation worker set differs")
    config = SystemConfig(**raw["config"])
    frozen = SystemConfig(**packet["config"])
    effective = replace(frozen, random_seed=job["seed"])
    # Collection switches predictor modes and scopes runtime hashes to residents.
    observed_config, expected_config = config.as_dict(), effective.as_dict()
    if (config.catalog.get("performance_mode") != "task_replica_proxy"
            or config.catalog.get("memory_mode") != "standalone_proxy"):
        raise ValueError("Confirmation collection must use the declared standalone transfer proxy")
    for value in (observed_config, expected_config):
        for key in ("performance_mode", "memory_mode", "measurement_context"):
            value["catalog"].pop(key, None)
    if observed_config != expected_config or raw.get("plan") != job["plan"]:
        raise ValueError("Confirmation raw scoring/plan differs from the frozen condition")
    context = config.catalog.get("measurement_context") or {}
    expected_context = copy.deepcopy(frozen.catalog["measurement_context"])
    hashes = {m: packet["execution_checkpoint_contract"]["runtime_checkpoint_sha256"][m]
              for m in job["plan"]["deployment"]}
    expected_context["runtime_checkpoint_sha256"] = hashes
    gpu = raw.get("preflight", {}).get("gpu", {})
    preflight = raw.get("preflight", {})
    if (context != expected_context or actual.get("runtime_checkpoint_sha256") != hashes
            or gpu.get("uuid") != expected_hardware["device_uuid"]
            or preflight.get("ready") is not True or preflight.get("gpu_checked") is not True
            or {w.get("env", {}).get("CUDA_VISIBLE_DEVICES", "0") for w in expected["workers"].values()} != {preflight.get("device")}
            or type(gpu.get("memory_mib")) not in (int, float)
            or not math.isfinite(gpu["memory_mib"])
            or gpu["memory_mib"] < frozen.gpu_memory_mib):
        raise ValueError("Confirmation actual GPU/checkpoint does not match the frozen predictor")
    transport = raw.get("transport_config", {})
    for field, value in {"random_seed": job["seed"], "measurement_seconds": HORIZON,
        "arrival_process": expected["arrival_process"], "warmup_seconds": expected["warmup_seconds"],
        "drain_timeout_seconds": expected["drain_seconds"]}.items():
        if transport.get(field) != value:
            raise ValueError("Confirmation transport differs: " + field)
    window = raw.get("measurement_window", {})
    if (window.get("trace_sha256") != expected["trace_sha256"] or window.get("duration_seconds") != HORIZON
            or raw.get("priming_policy") != expected["warmup_policy"]
            or raw.get("priming_size_metric") != expected["warmup_size_metric"]):
        raise ValueError("Confirmation cohort/priming differs")
    from .evaluation import evaluate_run
    utility = evaluate_run(config, raw)["utility"]
    if not utility.get("eligible") or not utility.get("energy_verified"):
        raise ValueError("Confirmation did not yield complete measured utility")
    return raw, utility


def run_confirmation_job(packet, job_id, output_root, *, execute=False):
    """Execute or strictly resume one seed-specific job; never overwrite an attempt."""
    if not execute:
        raise ValueError("Independent native confirmation requires explicit execute=True")
    check = validate_confirmation_packet(packet)
    if not check["valid"]:
        raise ValueError("Invalid confirmation packet: " + repr(check["changed"]))
    job = next((j for j in packet["jobs"] if j["id"] == job_id), None)
    if job is None:
        raise ValueError("Unknown confirmation job")
    output = Path(output_root) / (job["labels"][0] + "-seed" + str(job["seed"]) + "-" + job["id"][:12])
    run_path = output / "run.json"
    resumed = run_path.is_file()
    if resumed:
        raw, utility = verify_confirmation_run(packet, job, run_path)
    else:
        if output.exists():
            raise FileExistsError("Incomplete confirmation attempt requires separate review: " + str(output))
        execute_job(packet, job_id, output, execute=True)
        raw, utility = verify_confirmation_run(packet, job, run_path)
    return {"job_id": job_id, "seed": job["seed"], "candidate_key": job["candidate_key"],
            "labels": job["labels"], "resumed": resumed, "path": str(run_path.resolve()),
            "status": raw["status"], "utility": utility}
