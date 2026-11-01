"""Contract-driven native confirmation, separate from the historical study.

Preparation/reporting are CPU-only. Execution uses an explicitly bound frozen
runtime and a declared collection proxy; new transfer validation is mandatory.
"""
import copy
from dataclasses import replace
import importlib
import math
from pathlib import Path
import sys
import types
import uuid

from .config import SystemConfig
from .confirmation import _check_worker_source_coverage, _plan
from .evidence import digest
from .io import save
from .quality import quality_digest
from .static_study import _bindings, _file, read, validate_packet
from .study_reporting import measured_summary, paired_comparisons

SCHEMA = "gsdeploy.spillover_confirmation.v1"
CONTRACT_SCHEMA = "gsdeploy.spillover_confirmation_contract.v1"
ROLE_NAMES = ("free_search_winner", "best_global_no_split", "best_same_deployment_no_split",
              "strongest_alternative_deployment", "best_one_specialist_max")


def seal(packet):
    return quality_digest({k: v for k, v in packet.items() if k != "confirmation_sha256"})


class Runtime:
    """Keep the selected execution snapshot private and unmodified."""
    def __init__(self, root):
        self.prefix = "_spillover_runtime_" + uuid.uuid4().hex
        package = types.ModuleType(self.prefix)
        package.__path__ = [str(Path(root) / "gsdeploy")]
        package.__package__ = self.prefix
        sys.modules[self.prefix] = package
        try:
            self.study = importlib.import_module(self.prefix + ".static_study")
        except BaseException:
            self.close()
            raise

    def close(self):
        for name in tuple(sys.modules):
            if name == self.prefix or name.startswith(self.prefix + "."):
                del sys.modules[name]

    def __enter__(self):
        return self.study

    def __exit__(self, *args):
        self.close()


def _gate(config, contract):
    from .replay_transfer import check_scoring, validated_transfer_spec
    spec = validated_transfer_spec(config.catalog, config.workload)
    check_scoring(config)
    if contract.get("schema") != CONTRACT_SCHEMA or not contract.get("study_id"):
        raise ValueError("Expected a named predeclared spillover confirmation contract")
    seeds = contract["seed_roles"]
    flat = [seed for role in ("development", "validation", "confirmation") for seed in seeds[role]]
    if (seeds != spec["seed_roles"] or len(flat) != len(set(flat))
            or any(type(seed) is not int for seed in flat) or len(seeds["confirmation"]) < 2
            or config.random_seed not in seeds["development"]):
        raise ValueError("Confirmation/development/validation seeds must match the transfer and remain disjoint")
    if (contract["workload"] != config.workload or contract["horizon_seconds"] != config.horizon_seconds
            or type(contract["horizon_seconds"]) not in (int, float) or contract["horizon_seconds"] <= 0
            or type(contract["warmup_seconds"]) not in (int, float) or contract["warmup_seconds"] < 0
            or not math.isfinite(contract["horizon_seconds"]) or not math.isfinite(contract["warmup_seconds"])
            or contract.get("training_forbidden") is not True
            or not isinstance(contract.get("purpose"), str) or not contract["purpose"]
            or contract["purpose"] in {"calibration", "replay_transfer_independent_validation"}
            or contract.get("required_roles") != list(ROLE_NAMES)
            or contract.get("winner_label") != "free_search_winner"):
        raise ValueError("Changed workload, time, purpose or candidate-role contract")
    if contract.get("target_task") not in config.workload:
        raise ValueError("Missing target task")
    return spec


def _roles(config, bundle, contract):
    if (bundle.get("schema") != "gsdeploy.spillover_confirmation_candidates.v1"
            or bundle.get("config_sha256") != quality_digest(config.as_dict())
            or bundle.get("study_contract_sha256") != quality_digest(contract)
            or [row["label"] for row in bundle["candidates"]] != list(ROLE_NAMES)):
        raise ValueError("Candidate roster/config/contract is not frozen")
    records, unique = [], {}
    winner = bundle["candidates"][0].get("plan")
    if winner is None:
        raise ValueError("A frozen free-search incumbent is required")
    winner = _plan(winner)
    for row in bundle["candidates"]:
        label, value = row["label"], row.get("plan")
        record = copy.deepcopy(row)
        record["plan_key"] = None
        if value is not None:
            plan = _plan(value)
            if not config.validate_plan(**plan)["constraints_valid"]:
                raise ValueError("Unknown candidate constraints: " + label)
            if label in {"best_global_no_split", "best_same_deployment_no_split"} and any(
                    len(paths) != 1 for paths in plan["routing"].values()):
                raise ValueError("No-split control contains a split route")
            if label == "best_same_deployment_no_split" and plan["deployment"] != winner["deployment"]:
                raise ValueError("Same-deployment control changed deployment")
            if label == "strongest_alternative_deployment" and plan["deployment"] == winner["deployment"]:
                raise ValueError("Alternative deployment repeats the winner")
            if label == "best_one_specialist_max" and sum(n for m,n in plan["deployment"].items()
                    if config.catalog["models"][m]["role"] == "specialist"
                    and contract["target_task"] in config.catalog["models"][m]["supported_tasks"]) > 1:
                raise ValueError("One-specialist control exceeds one target-specialist replica")
            key = quality_digest(plan)
            record.update(plan=plan, plan_key=key)
            unique.setdefault(key, dict(plan_key=key, plan=plan, labels=[]))["labels"].append(label)
        elif not isinstance(row.get("status"),str) or not any(token in row["status"]
                for token in ("INFEASIBLE","UNKNOWN","BUDGET","LIMIT","NO_ELIGIBLE")):
            raise ValueError("Absent control must preserve explicit unresolved/infeasible status")
        records.append(record)
    return records, list(unique.values())


def _checkpoint_contract(config, checkpoint_report, runtime_root):
    hashes = dict(config.catalog["measurement_context"]["runtime_checkpoint_sha256"])
    receipts = []
    if checkpoint_report is not None:
        receipts.append(_file(checkpoint_report))
        for model, record in read(checkpoint_report)["models"].items():
            if model not in config.catalog["models"]:
                continue
            path = Path(record["manifest"])
            bound = _file(path if path.is_absolute() else runtime_root / path)
            value = record["checkpoint"]["sha256"]
            if (bound["sha256"] != record["manifest_sha256"]
                    or read(bound["path"])["models"][model]["checkpoint"]["sha256"] != value
                    or model in hashes and hashes[model] != value):
                raise ValueError("Changed or conflicting checkpoint receipt: " + model)
            hashes[model] = value
            receipts.append(bound)
    return hashes, receipts


def _job(study, config, workers, candidate, seed, bindings, contract, hashes):
    job = study._job(config, workers, candidate["plan"], candidate["labels"][0], "confirmation", seed,
                     bindings, contract["warmup_seconds"])
    job.update(candidate_key=candidate["plan_key"], labels=candidate["labels"], purpose=contract["purpose"])
    job["identity"].update(confirmation_purpose=contract["purpose"],
        study_contract_sha256=quality_digest(contract), scoring_config_sha256=quality_digest(config.as_dict()),
        runtime_checkpoint_sha256={m:hashes[m] for m in candidate["plan"]["deployment"]})
    job["id"] = quality_digest(job["identity"])
    return job


def prepare(source_packet_path, config_path, candidates_path, contract_path, runtime_root, output=None,
            *, checkpoint_report=None):
    """Require independently validated transfer before sealing final candidates."""
    paths = [Path(p).resolve() for p in (source_packet_path, config_path, candidates_path, contract_path)]
    source, raw, bundle, contract = map(read, paths)
    if not validate_packet(source)["valid"]:
        raise ValueError("Source packet changed")
    config = SystemConfig(**raw)
    spec = _gate(config, contract)
    roles, candidates = _roles(config, bundle, contract)
    if source["config"]["catalog"]["quality"] != config.catalog["quality"]:
        raise ValueError("Confirmation input/reference/quality differs from source")
    runtime_root = Path(runtime_root).resolve()
    inputs = Path(source["inputs"]).resolve()
    bindings = [*source["bindings"], *_bindings(inputs, runtime_root),
                *_bindings(inputs,Path(__file__).resolve().parents[1]), *map(_file, paths), _file(__file__)]
    hashes, checkpoint_receipts = _checkpoint_contract(config, checkpoint_report, runtime_root)
    bindings.extend(checkpoint_receipts)
    bindings = list({row["path"]:row for row in bindings}.values())
    _check_worker_source_coverage(source["workers"], bindings)
    missing = {m for c in candidates for m in c["plan"]["deployment"]} - set(hashes)
    if missing:
        raise ValueError("WAITING_CONFIRMATION_CHECKPOINT_BINDING: " + ",".join(sorted(missing)))
    with Runtime(runtime_root) as study:
        jobs = [_job(study, config, source["workers"], candidate, seed, bindings, contract, hashes)
                for seed in contract["seed_roles"]["confirmation"] for candidate in candidates]
    packet = dict(schema=SCHEMA, status="CPU_PREPARED_SPILLOVER_CONFIRMATION", inputs=str(inputs),
        config=config.as_dict(), workers=copy.deepcopy(source["workers"]), bindings=bindings,
        frozen_files_sha256=quality_digest(bindings), source_packet=_file(paths[0]),
        validated_config=_file(paths[1]), frozen_candidates=_file(paths[2]), study_contract=_file(paths[3]),
        contract=contract, runtime_root=str(runtime_root), transfer_version_sha256=spec["version_sha256"],
        candidate_roles=roles, candidates=candidates, jobs=jobs,
        execution_checkpoint_contract={"runtime_checkpoint_sha256":hashes,"receipts":checkpoint_receipts},
        collection_conversion={"performance_mode":"task_replica_proxy","memory_mode":"standalone_proxy",
                               "scope":"Measurement only; original transfer/scoring contract remains frozen"},
        gpu_started=False, physical_global_optimality_certified=False)
    packet["confirmation_sha256"] = seal(packet)
    check = validate(packet)
    if not check["valid"]:
        raise ValueError("Invalid confirmation packet: " + repr(check["changed"]))
    if output is not None:
        if Path(output).exists():
            raise FileExistsError(output)
        save(output, packet)
    return packet


def validate(packet):
    changed=[]
    try:
        if packet.get("schema") != SCHEMA or packet.get("confirmation_sha256") != seal(packet):
            raise ValueError("Packet seal changed")
        if packet.get("physical_global_optimality_certified") is not False:
            raise ValueError("Physical optimum claim is unsupported")
        check=validate_packet(packet)
        if not check["valid"]:
            raise ValueError("Packet files/jobs changed: " + repr(check["changed"]))
        for key in ("source_packet","validated_config","frozen_candidates","study_contract"):
            if packet[key] not in packet["bindings"] or _file(packet[key]["path"]) != packet[key]:
                raise ValueError("Unbound receipt: " + key)
        if read(packet["validated_config"]["path"]) != packet["config"] or read(packet["study_contract"]["path"]) != packet["contract"]:
            raise ValueError("Embedded config/contract changed")
        config=SystemConfig(**packet["config"]); spec=_gate(config,packet["contract"])
        if spec["version_sha256"] != packet["transfer_version_sha256"]:
            raise ValueError("Transfer version changed")
        roles,candidates=_roles(config,read(packet["frozen_candidates"]["path"]),packet["contract"])
        if roles != packet["candidate_roles"] or candidates != packet["candidates"]:
            raise ValueError("Candidate identity/roles changed")
        source=read(packet["source_packet"]["path"])
        if source["workers"] != packet["workers"] or source["inputs"] != packet["inputs"]:
            raise ValueError("Source workers/inputs changed")
        if source["config"]["catalog"]["quality"] != config.catalog["quality"]:
            raise ValueError("Source quality/input contract changed")
        _check_worker_source_coverage(packet["workers"],packet["bindings"])
        hashes=packet["execution_checkpoint_contract"]["runtime_checkpoint_sha256"]
        checkpoint_receipts=packet["execution_checkpoint_contract"]["receipts"]
        for row in checkpoint_receipts:
            if row not in packet["bindings"] or _file(row["path"])!=row:
                raise ValueError("Unbound checkpoint receipt")
        expected_hashes=dict(config.catalog["measurement_context"]["runtime_checkpoint_sha256"])
        for row in checkpoint_receipts:
            value=read(row["path"])
            for model,record in value.get("models",{}).items():
                if model not in config.catalog["models"] or "checkpoint" not in record:
                    continue
                current=record["checkpoint"]["sha256"]
                if model in expected_hashes and expected_hashes[model]!=current:
                    raise ValueError("Conflicting execution checkpoint")
                expected_hashes[model]=current
        if expected_hashes!=hashes:
            raise ValueError("Execution checkpoint union changed")
        with Runtime(packet["runtime_root"]) as study:
            expected=[_job(study,config,packet["workers"],c,seed,packet["bindings"],packet["contract"],hashes)
                      for seed in packet["contract"]["seed_roles"]["confirmation"] for c in candidates]
        if expected != packet["jobs"]:
            raise ValueError("Rebuilt confirmation job/purpose differs")
        # All execution Python bytes must be bound, even if a resealed packet
        # points at another runtime folder.
        covered={r["path"]:r["sha256"] for r in packet["bindings"]}
        for folder in ("gsdeploy","scripts"):
            for p in (Path(packet["runtime_root"])/folder).rglob("*.py"):
                if covered.get(str(p.resolve())) != digest(p):
                    raise ValueError("Unbound execution source")
        if packet["collection_conversion"]["performance_mode"] != "task_replica_proxy" or packet["collection_conversion"]["memory_mode"] != "standalone_proxy":
            raise ValueError("Collection mode changed")
    except (ValueError,KeyError,TypeError,OSError) as exc:
        changed.append(str(exc))
    return dict(valid=not changed,changed=changed,jobs=len(packet.get("jobs",[])),gpu_started=False)


def verify_run(packet, job, run_path):
    check=validate(packet)
    if not check["valid"] or job not in packet["jobs"]:
        raise ValueError("Invalid packet/job: " + repr(check["changed"]))
    raw=read(run_path); actual=copy.deepcopy(raw.get("executed_measurement_identity",{}))
    if (raw.get("status")!="completed" or raw.get("backend")!="native" or raw.get("experiment_phase")!="confirmation"
            or raw.get("experiment_job_id")!=job["id"] or raw.get("confirmation_packet_sha256")!=packet["confirmation_sha256"]
            or raw.get("executed_measurement_id")!=quality_digest(actual) or raw.get("plan")!=job["plan"]):
        raise ValueError("Not this completed native confirmation")
    for worker in actual.get("workers",{}).values():
        worker.get("env",{}).pop("GS_CHECKPOINT_FILES_JSON",None)
    if actual!=job["identity"]:
        raise ValueError("Executed confirmation purpose/identity changed")
    if read(Path(run_path).parent/"spillover-collection-receipt.json") != {**_collection_receipt(packet),"run":_file(run_path)}:
        raise ValueError("Missing declared collection conversion receipt")
    frozen=SystemConfig(**packet["config"]); config=SystemConfig(**raw["config"])
    expected=replace(frozen,random_seed=job["seed"]).as_dict()
    expected["catalog"].update(performance_mode="task_replica_proxy",memory_mode="standalone_proxy")
    expected["catalog"]["measurement_context"]["runtime_checkpoint_sha256"]=job["identity"]["runtime_checkpoint_sha256"]
    expected["catalog"]["measurement_context"]["historical_quality_checkpoint_equivalence"]="receipt_scope_only; actual run outputs rescored"
    if config.as_dict()!=expected:
        raise ValueError("Measured collection/scoring config changed")
    identity=job["identity"];preflight=raw.get("preflight",{});gpu=preflight.get("gpu",{})
    if (preflight.get("ready") is not True or preflight.get("gpu_checked") is not True
            or gpu.get("uuid")!=identity["hardware_contract"]["device_uuid"]
            or type(gpu.get("memory_mib")) not in (int,float) or not math.isfinite(gpu["memory_mib"])
            or gpu["memory_mib"]<frozen.gpu_memory_mib
            or {w.get("env",{}).get("CUDA_VISIBLE_DEVICES","0") for w in identity["workers"].values()}!={preflight.get("device")}):
        raise ValueError("GPU preflight differs from contract")
    for key,value in dict(random_seed=job["seed"],measurement_seconds=identity["horizon_seconds"],
        arrival_process=identity["arrival_process"],warmup_seconds=identity["warmup_seconds"],drain_timeout_seconds=identity["drain_seconds"]).items():
        if raw.get("transport_config",{}).get(key)!=value:
            raise ValueError("Transport contract changed: "+key)
    window=raw.get("measurement_window",{})
    if (window.get("duration_seconds")!=identity["horizon_seconds"] or window.get("trace_sha256")!=identity["trace_sha256"]
            or raw.get("priming_policy")!=identity["warmup_policy"] or raw.get("priming_size_metric")!=identity["warmup_size_metric"]):
        raise ValueError("Cohort/priming contract changed")
    from .evaluation import evaluate_run
    utility=evaluate_run(config,raw)["utility"]
    if not utility.get("eligible") or not utility.get("energy_verified") or type(utility.get("U")) not in (int,float) or not math.isfinite(utility["U"]):
        raise ValueError("Missing/ineligible measured utility; never substitute zero")
    return raw,utility


def _collection_receipt(packet):
    return dict(original_config_sha256=quality_digest(packet["config"]),
                frozen_performance_mode=packet["config"]["catalog"]["performance_mode"],
                collection_performance_mode="task_replica_proxy",collection_memory_mode="standalone_proxy",
                confirmation_sha256=packet["confirmation_sha256"],runtime_root=packet["runtime_root"],
                purpose=packet["contract"]["purpose"],optimality_transferred=False)


def run(packet, job_id, output, *, execute=False):
    if not execute:
        raise ValueError("Explicit execute=True required")
    check=validate(packet)
    if not check["valid"]:
        raise ValueError(check["changed"])
    job=next(j for j in packet["jobs"] if j["id"]==job_id)
    output=Path(output)
    if (output/"run.json").is_file():
        return verify_run(packet,job,output/"run.json")
    # v5 does not recognize replay_transfer_v1. Declare the collection proxy
    # before calling its untouched executor, preserving original job identity
    # and packet seal. This is never reused as a search/validation config.
    collection=copy.deepcopy(packet)
    collection["config"]["catalog"].update(performance_mode="task_replica_proxy",memory_mode="standalone_proxy")
    for item in collection["jobs"]:
        item["provenance"]["source_problem_sha256"]=quality_digest({**collection["config"],"random_seed":item["seed"]})
    with Runtime(packet["runtime_root"]) as study:
        study.execute_job(collection,job_id,output,execute=True)
    # The immutable executor output is never overwritten. A separate receipt
    # binds its exact bytes to the original configuration and declared delta.
    save(output/"spillover-collection-receipt.json",{**_collection_receipt(packet),"run":_file(output/"run.json")})
    return verify_run(packet,job,output/"run.json")


def report(packet_path, run_paths, output):
    packet=read(packet_path);check=validate(packet)
    if not check["valid"]:
        raise ValueError(check["changed"])
    jobs={j["id"]:j for j in packet["jobs"]};rows=[];seen=set()
    for path in run_paths:
        value=read(path);key=value.get("experiment_job_id")
        if key not in jobs or key in seen:
            raise ValueError("Unexpected or duplicate confirmation job")
        raw,utility=verify_run(packet,jobs[key],path);seen.add(key);job=jobs[key]
        rows.append(dict(job_id=key,plan_key=job["candidate_key"],labels=job["labels"],seed=job["seed"],
                         **_file(path),collection_receipt=_file(Path(path).parent/"spillover-collection-receipt.json"),
                         utility=utility,errors=raw.get("errors",[])))
    selected=next(r["plan_key"] for r in packet["candidate_roles"] if r["label"]==packet["contract"]["winner_label"])
    missing=sorted(set(jobs)-seen)
    result=dict(schema="gsdeploy.spillover_confirmation_report.v1",packet=_file(packet_path),
        status="ALL_DECLARED_JOBS_VERIFIED" if not missing else "INCOMPLETE_CONFIRMATION",completed_jobs=len(seen),
        planned_jobs=len(jobs),missing_job_ids=missing,runs=measured_summary(rows,packet["config"]["weights"]),
        paired=paired_comparisons(rows,selected,packet["contract"]["seed_roles"]["confirmation"],
                                  planned_plan_keys=[c["plan_key"] for c in packet["candidates"]]),
        candidate_roles=packet["candidate_roles"],training_forbidden=True,gpu_started=False,
        physical_global_optimality_certified=False,scope="Only declared candidates, workload and seed blocks")
    if Path(output).exists():
        raise FileExistsError(output)
    save(output,result)
    return result
