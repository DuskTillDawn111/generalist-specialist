"""Isolated controlled shared-GPU batches; never a 600-second utility trial.

The CLI prepares an append-only packet without touching CUDA. Only an explicit
execute invocation launches the exact resident deployment. Each batch drains
fully before the next randomized context/input condition is admitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.spillover_capacity_20261022.exposure import SOURCE, frozen_api
from tools.queue_refined_20261023 import admission, protocol
from tools.queue_refined_20261023.native_runner import frozen_runtime

SCHEMA = "gsdeploy.coupled_controlled_calibration.v1"
PACKET_SCHEMA = "gsdeploy.coupled_controlled_calibration_packet.v1"
PRIMARY = {"text": 2, "generalist_2": 1, "ocr": 1}
SEEDS = (301, 302)
CONTEXTS = {
    "S": ("text#0",),
    "G": ("generalist_2#0",),
    "SS": ("text#0", "text#1"),
    "SG": ("text#0", "generalist_2#0"),
    "SSG": ("text#0", "text#1", "generalist_2#0"),
}
DEFAULT_SIMULATION = ROOT / "artifacts/queue-aware-study-20261023/cpu-closed-loop-v1/frozen-simulation-contract.json"


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def text_inputs(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    items = sorted((row for row in rows if row["task_type"] == "text"), key=lambda row: row["pool_id"])
    if len(items) != 32 or len({row["pool_id"] for row in items}) != 32:
        raise ValueError("Exactly 32 distinct frozen text inputs required")
    return items


def schedule(pool_ids):
    if len(pool_ids) != 32 or len(set(pool_ids)) != 32:
        raise ValueError("Exactly 32 distinct pool IDs required")
    result = []
    for seed in SEEDS:
        conditions = [(context, pool) for context in CONTEXTS for pool in sorted(pool_ids)]
        random.Random(f"coupled-controlled-v1:{seed}").shuffle(conditions)
        for index, (context, pool) in enumerate(conditions):
            aliases = list(CONTEXTS[context])
            random.Random(f"coupled-controlled-v1:{seed}:{index}:aliases").shuffle(aliases)
            identity = {"seed": seed, "index": index, "context": context,
                        "pool_id": pool, "aliases": aliases}
            result.append({"id": digest(identity), **identity})
    return result


def interval(row):
    """Worker end and handler service define a measured, not predicted interval."""
    response = row.get("worker_response", {})
    end, service = response.get("end_ns"), response.get("service_ms")
    anchor_perf, anchor_unix = row.get("completion_time_ns"), row.get("completion_unix_ns")
    if (type(end) is not int or type(service) not in (int, float)
            or not math.isfinite(service) or service <= 0
            or type(anchor_perf) is not int or type(anchor_unix) is not int):
        return None
    start = end - round(service * 1e6)
    offset = anchor_unix - anchor_perf
    return {"start_time_ns": start-offset, "end_time_ns": end-offset,
            "start_unix_ns": start, "end_unix_ns": end,
            "basis": "worker end_ns minus service_ms; unix to perf by own host completion anchor",
            "scope": "completed handler service, not observed GPU kernel execution"}


def prepare(base_packet, output, simulation_contract=DEFAULT_SIMULATION):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    base_path, simulation_path = Path(base_packet).resolve(), Path(simulation_contract).resolve()
    base, sim = json.loads(base_path.read_text()), json.loads(simulation_path.read_text())
    if sim.get("deployment") != PRIMARY:
        raise ValueError("Exact PRIMARY simulation deployment required for unchanged preflight")
    samples = Path(base["inputs"]) / "runtime-inputs.jsonl"
    items = text_inputs(samples)
    if receipt(samples)["sha256"] != base["config"]["catalog"]["quality"]["identity"]["input_pool_sha256"]:
        raise ValueError("Frozen input fingerprint changed")
    templates = [job for job in base["jobs"] if job["plan"]["deployment"] == PRIMARY]
    if not templates:
        raise ValueError("No PRIMARY base binding job")
    sources = [receipt(__file__), receipt(ROOT / "tools/spillover_capacity_20261022/exposure.py")]
    sources += [receipt(p) for p in sorted((ROOT / "tools/queue_refined_20261023").glob("*.py"))]
    sources += [receipt(p) for p in sorted(SOURCE.rglob("*.py"))]
    bindings = [receipt(base_path), receipt(simulation_path), receipt(samples), *base.get("bindings", [])]
    for source in bindings:
        if receipt(source["path"]) != source:
            raise ValueError("Source binding changed: " + source["path"])
    rate = base["config"]["workload"]["text"]
    contract = protocol.make_policy_contract(PRIMARY, sim["service_ms"], sim["aggregate_quality_loss"],
        base["config"]["weights"], base["config"]["latency_reference_ms"], strategy="queue_quality",
        models=base["config"]["catalog"]["models"], text_rate=rate,
        sources={r["path"]: r["sha256"] for r in sources},
        guard={"schema": "queue_aware_token_bucket_v1", "task": "text", "family": "generalist_2",
               "rate_rps": rate/8, "burst": 1})
    batches = schedule([item["pool_id"] for item in items])
    packet = {"schema": PACKET_SCHEMA, "base_packet": receipt(base_path),
              "input_pool": receipt(samples), "sources": sources, "bindings": bindings,
              "deployment": PRIMARY, "policy_contract_for_preflight_only": contract,
              "binding_template": copy.deepcopy(templates[0]), "batches": batches,
              "seeds": list(SEEDS), "batch_schedule_sha256": digest(batches),
              "planned_batches": len(batches),
              "planned_text_requests": sum(len(batch["aliases"]) for batch in batches),
              "priming_policy": "largest_three_per_supported_active_task_per_resident_alias",
              "no_retries": True, "GPU_started": False,
              "claim": "CONTROLLED_CALIBRATION_TRAINING_NOT_600S_UTILITY_NOT_STABILITY_VALIDATION",
              "ordering": "Two declared random condition permutations; no future arrival process",
              "batch_isolation": "Same pool input on every active alias; drain every batch before next",
              "nominal_preflight_scope": "Unchanged identity/resources/paths and nominal original proxy gate; not a capacity proof for calibration bursts"}
    output.mkdir(parents=True)
    save(output / "experiment-packet.json", packet)
    return packet


def validate_packet(packet):
    if packet.get("schema") != PACKET_SCHEMA or packet.get("deployment") != PRIMARY or packet.get("seeds") != list(SEEDS):
        raise ValueError("Unknown controlled calibration packet")
    for source in packet["sources"] + packet["bindings"]:
        if receipt(source["path"]) != source:
            raise ValueError("Frozen calibration source changed: " + source["path"])
    items = text_inputs(packet["input_pool"]["path"])
    expected = schedule([item["pool_id"] for item in items])
    if packet["batches"] != expected or packet["batch_schedule_sha256"] != digest(expected):
        raise ValueError("Controlled batch schedule changed")
    if packet.get("no_retries") is not True:
        raise ValueError("Explicit no-retry protocol required")
    protocol.validate_contract(packet["policy_contract_for_preflight_only"])
    return True


def collect(system, samples, contract, batches, *, backend="mock", workers=None,
            admission_path=None, startup_timeout=300., drain_timeout=300., idle_seconds=3.,
            progress=None):
    original = frozen_runtime()
    plan = protocol.nominal_plan(contract)
    aliases, routes = original.expand_routes(PRIMARY, plan["routing"])
    config = original.MeasurementConfig(deployment={a: 1 for a in aliases},
        workload_rps=dict(system.workload), routing_rps=routes, warmup_seconds=0,
        measurement_seconds=1., random_seed=SEEDS[0], arrival_process="deterministic_periodic",
        startup_timeout_seconds=startup_timeout, drain_timeout_seconds=drain_timeout,
        routing_source="isolated_controlled_calibration_not_arrival_trace")
    config.validate({a: system.catalog["models"][family]["supported_tasks"] for a, family in aliases.items()})
    registry = original.registry_for(system, aliases, backend, workers or {})
    for worker in registry.values():
        worker["env"].update({key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    raw = {"schema": SCHEMA, "backend": backend, "status": "starting", "errors": [],
           "deployment": copy.deepcopy(PRIMARY), "replica_to_family": aliases,
           "aliases": contract["aliases"], "worker_registry": registry,
           "policy_contract": copy.deepcopy(contract), "request_records": [],
           "priming_request_records": [], "batches": [], "idle_windows": [],
           "measurement_window": None, "memory_evidence": {}, "energy_evidence": {},
           "preflight": None, "shutdown": {}, "seeds": list(SEEDS),
           "calibration_phase_seeds_are_arrival_seeds": False,
           "original_stability_evaluated": False, "U": None,
           "claim": "CONTROLLED_CALIBRATION_TRAINING_NOT_600S_UTILITY_NOT_STABILITY_VALIDATION"}
    runtime = original.DeploymentRuntime(config, registry)
    monitor = power = None
    start = end = None
    try:
        if backend == "native":
            checks = admission.preflight(system, plan, samples, workers, contract,
                                         admission=admission_path, require_gpu=True)
            raw["preflight"] = checks
            if not checks["ready"]:
                raise ValueError("Unchanged native preflight failed: " + repr(checks["issues"]))
        else:
            checks = None
        device = checks["device"] if checks else "0"
        monitor = original.MemoryMonitor(runtime, system.gpu_memory_mib, enabled=backend == "native", device=device)
        power = original.PowerMonitor(enabled=backend == "native", device=device)
        power.sample(); power.start()
        runtime.start(); monitor.sample(); monitor.start(); monitor.check()
        pool = original.SamplePool(samples, list(system.workload), SEEDS[0])
        by_pool = {item["pool_id"]: item for item in text_inputs(samples)}
        for alias, family in aliases.items():
            for task in system.catalog["models"][family]["supported_tasks"]:
                if system.workload.get(task, 0) <= 0:
                    continue
                for item in original.largest_inputs(pool, task):
                    row = original.record("priming", len(raw["priming_request_records"]), task, alias, item, time.perf_counter_ns())
                    raw["priming_request_records"].append(row)
                    runtime.enqueue(row, item["sample"])
        runtime.drain(raw["priming_request_records"], drain_timeout)
        if any(row["status"] != "completed" for row in raw["priming_request_records"]):
            raise RuntimeError("Controlled priming failed")
        for position in ("before",):
            a = time.perf_counter_ns()
            original.wait_until(a+round(idle_seconds*1e9), monitor)
            raw["idle_windows"].append({"position": position, "start_time_ns": a, "end_time_ns": time.perf_counter_ns()})
        start = time.perf_counter_ns()
        for batch in batches:
            monitor.check()
            rows = []; item = by_pool[batch["pool_id"]]
            a = time.perf_counter_ns()
            with runtime.lock:
                for alias in batch["aliases"]:
                    row = original.record("controlled", len(raw["request_records"]), "text", alias, item, a)
                    row.update(calibration_batch_id=batch["id"], calibration_context=batch["context"],
                               calibration_seed=batch["seed"])
                    rows.append(row); raw["request_records"].append(row)
                    runtime.enqueue(row, item["sample"])
            runtime.drain(rows, drain_timeout)
            b = time.perf_counter_ns()
            evidence = {**copy.deepcopy(batch), "request_ids": [row["request_id"] for row in rows],
                        "start_time_ns": a, "end_time_ns": b,
                        "all_completed": all(row["status"] == "completed" for row in rows)}
            raw["batches"].append(evidence)
            if not evidence["all_completed"]:
                raise RuntimeError("Controlled batch failed: " + batch["id"])
            if len(raw["batches"]) % 16 == 0 or len(raw["batches"]) == len(batches):
                event = {"event": "controlled_calibration_progress", "completed_batches": len(raw["batches"]),
                         "planned_batches": len(batches), "completed_requests": len(raw["request_records"]),
                         "planned_requests": sum(len(item["aliases"]) for item in batches),
                         "elapsed_seconds": (b-start)/1e9, "seed": batch["seed"]}
                if progress is None:
                    print(json.dumps(event), flush=True)
                else:
                    progress(event)
        end = time.perf_counter_ns()
        a = end
        original.wait_until(a+round(idle_seconds*1e9), monitor)
        raw["idle_windows"].append({"position": "after", "start_time_ns": a, "end_time_ns": time.perf_counter_ns()})
        monitor.sample(); monitor.check()
        raw["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        raw["status"] = "failed"; raw["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if monitor:
            monitor.close()
        if power:
            power.close()
        try:
            raw["shutdown"] = runtime.close()
        except Exception as exc:
            raw["errors"].append("shutdown: " + str(exc)); raw["status"] = "failed"
        raw["errors"].extend(runtime.errors)
        if runtime.errors or any(s["forced"] or s["exit_code"] != 0 or not s["bye_received"] for s in raw["shutdown"].values()):
            raw["status"] = "failed"
        for row in raw["request_records"] + raw["priming_request_records"]:
            if row["status"] in {"planned", "queued", "submitted"}:
                row.update(status="incomplete", error="controlled collection ended before completion")
            row["actual_service_interval"] = interval(row)
        original.annotate_records(raw["request_records"])
    samples_power = power.samples if power else []
    joules = original.integrate_power(samples_power, start, end) if start is not None and end is not None else None
    raw["measurement_window"] = ({"start_time_ns": start, "end_time_ns": end,
                                  "duration_seconds": (end-start)/1e9,
                                  "scope": "actual isolated calibration elapsed time, not 600 seconds"}
                                 if start is not None and end is not None else None)
    raw["energy_evidence"] = {"kind": "measured" if backend == "native" else "unmeasured",
        "scope": "whole_gpu_board", "samples": samples_power, "run_joules": joules,
        "coverage_valid": backend == "native" and power is not None and not power.failed and joules is not None,
        "sampling_errors": power.errors if power else [],
        "scope_note": "Calibration full elapsed span; preserves samples through priming and final idle; no utility E"}
    memory = monitor.samples if monitor else []
    peak = max((sample["owned_used_mib"] for sample in memory), default=None)
    safe = peak*system.catalog["safe_factor"] if peak is not None else None
    memory_ok = (backend == "native" and monitor is not None and not monitor.failed and safe is not None
                 and safe <= system.gpu_memory_mib and all(not sample["foreign_pids"] for sample in memory))
    raw["memory_evidence"] = {"measured": backend == "native" and bool(memory),
        "samples": memory, "safe_memory_mib": safe, "resource_valid": bool(memory_ok),
        "scope": "whole resident owned-process sampled peak including all-path priming"}
    raw["calibration_usable"] = (raw["status"] == "completed" and backend == "native"
        and raw["energy_evidence"]["coverage_valid"] and bool(memory_ok)
        and all(row["actual_service_interval"] is not None for row in raw["request_records"]))
    return raw


def execute(packet, output, *, execute=False):
    if not execute:
        raise ValueError("Explicit execute=True required to launch GPU workers")
    validate_packet(packet)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    raw = None
    try:
        original = frozen_runtime(); package = original.__package__.split(".runtime")[0]
        static = importlib.import_module(package + ".static_study")
        config_module = importlib.import_module(package + ".config")
        base = json.loads(Path(packet["base_packet"]["path"]).read_text())
        job = copy.deepcopy(packet["binding_template"])
        job["seed"] = SEEDS[0]
        config, manifest, workers = static.bind_native_job(base, job, output / "native-binding")
        config["catalog"]["performance_mode"] = "task_replica_proxy"
        config["catalog"]["memory_mode"] = "standalone_proxy"
        for worker in workers.values():
            worker.setdefault("env", {}).update({key: "1" for key in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
        raw = collect(config_module.SystemConfig(**config), packet["input_pool"]["path"],
                      packet["policy_contract_for_preflight_only"], packet["batches"],
                      backend="native", workers=workers, admission_path=str(manifest))
    except (Exception, KeyboardInterrupt) as exc:
        raw = {"schema": SCHEMA, "status": "failed", "backend": "native", "calibration_usable": False,
               "errors": [f"{type(exc).__name__}: {exc}"], "request_records": [], "batches": [], "U": None}
    raw.update(source_packet_digest=digest(packet), source_receipts=packet["sources"],
               input_pool=packet["input_pool"], no_retries=True)
    save(output / "calibration.json", raw)
    return raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--base-packet", type=Path, required=True)
    prep.add_argument("--simulation-contract", type=Path, default=DEFAULT_SIMULATION)
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("execute"); run.add_argument("--packet", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True); run.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.base_packet, args.output, args.simulation_contract)
        print(json.dumps({"packet": receipt(args.output / "experiment-packet.json"),
                          "batches": result["planned_batches"], "requests": result["planned_text_requests"], "GPU_started": False}))
    else:
        packet = json.loads(args.packet.read_text())
        result = execute(packet, args.output, execute=args.execute)
        print(json.dumps({"status": result["status"], "calibration_usable": result["calibration_usable"],
                          "raw": receipt(args.output / "calibration.json")}))
        raise SystemExit(0 if result["calibration_usable"] else 1)
