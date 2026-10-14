"""Run a supplied plan using independent workers and measured request cohorts."""

import json
import math
import os
from pathlib import Path
import random
import sys
import time

from .deployment import DeploymentRuntime
from .schema import MeasurementConfig
from .telemetry import MemoryMonitor
from .workload import SamplePool, periodic_arrivals
from .preflight import preflight
from ..evaluation import evaluate_run
from ..metrics import annotate_records, summarize
from ..routing import Router
from ..trace import expand_routes, make_trace, trace_digest
from .telemetry import PowerMonitor
from ..energy import integrate_power, energy_key


def record(phase, index, task, worker, item, scheduled):
    return {
        "request_id": f"{phase}:{index}",
        "phase": phase,
        "arrival_index": index,
        "task": task,
        "model": worker,
        "pool_id": item["pool_id"],
        "sample_id": item["sample"].get("id"),
        "scheduled_time_ns": scheduled,
        "actual_submit_time_ns": None,
        "completion_time_ns": None,
        "worker_latency_ms": None,
        "worker_queue_wait_ms": None,
        "status": "planned",
    }


def wait_until(deadline, monitor):
    while True:
        remaining = (deadline - time.perf_counter_ns()) / 1e9
        if remaining <= 0:
            return
        monitor.check()
        time.sleep(min(remaining, 0.01))


def window(
    runtime, config, pool_path, phase, duration, records, monitor, metadata=None
):
    pool = SamplePool(pool_path, list(config.workload_rps), config.random_seed)
    router = Router(config.routing_rps)
    trace = make_trace(
        config.workload_rps, duration, config.random_seed, config.arrival_process
    )
    payloads = []
    for index, (offset, task) in enumerate(trace):
        item = pool.next(task)
        row = record(phase, index, task, router.choose(task), item, 0)
        row["scheduled_offset_seconds"] = offset
        records.append(row)
        payloads.append(item["sample"])
    start, unix = time.perf_counter_ns(), time.time_ns()
    end = start + round(duration * 1e9)
    metadata = {} if metadata is None else metadata
    metadata.update(
        start_time_ns=start,
        start_unix_ns=unix,
        end_time_ns=end,
        duration_seconds=duration,
        offered_requests=len(trace),
        trace_sha256=trace_digest(trace),
        cohort="all requests scheduled in [start,end), registered before dispatch",
    )
    for row in records:
        row["scheduled_time_ns"] = start + round(row["scheduled_offset_seconds"] * 1e9)
    for row, sample in zip(records, payloads):
        wait_until(row["scheduled_time_ns"], monitor)
        runtime.enqueue(row, sample)
    wait_until(end, monitor)
    metadata["arrival_loop_end_time_ns"] = time.perf_counter_ns()
    runtime.drain(records, config.drain_timeout_seconds)
    metadata["drain_end_time_ns"] = time.perf_counter_ns()
    metadata["drain_seconds"] = (metadata["drain_end_time_ns"] - end) / 1e9
    return metadata


def registry_for(system, aliases, backend, workers):
    project = str(Path(__file__).resolve().parents[2])
    registry = {}
    for alias, family in aliases.items():
        if backend == "mock":
            entry = {
                "command": [sys.executable, "-m", "gsdeploy.backends.mock"],
                "env": {
                    "GS_TASKS": json.dumps(
                        system.catalog["models"][family]["supported_tasks"]
                    )
                },
            }
        else:
            if family not in workers:
                raise ValueError(f"No worker registered for {family}")
            entry = dict(workers[family])
            entry["env"] = dict(entry.get("env", {}))
        command = entry.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(v, str) or not v for v in command)
        ):
            raise ValueError(f"Invalid worker command for {family}")
        entry["env"] = {
            **entry.get("env", {}),
            "PYTHONPATH": project
            + os.pathsep
            + entry["env"].get("PYTHONPATH", os.environ.get("PYTHONPATH", "")),
        }
        entry.setdefault("cwd", project)
        if backend != "mock":
            entry["env"].setdefault("CUDA_VISIBLE_DEVICES", "0")
        registry[alias] = entry
    return registry


def run(
    system,
    winner,
    samples,
    backend="mock",
    workers=None,
    duration=30.0,
    warmup=1.0,
    seed=42,
    arrival_process="deterministic_periodic",
    startup_timeout=300.0,
    drain_timeout=60.0,
    admission=None,
):
    if backend not in {"mock", "native"}:
        raise ValueError("backend must be mock or native")
    if arrival_process not in {"deterministic_periodic", "poisson"}:
        raise ValueError("Unknown arrival process")
    aliases, routes = expand_routes(winner["deployment"], winner["routing"])
    config = MeasurementConfig(
        deployment={a: 1 for a in aliases},
        workload_rps=dict(system.workload),
        routing_rps=routes,
        warmup_seconds=warmup,
        measurement_seconds=duration,
        arrival_process=arrival_process,
        random_seed=seed,
        startup_timeout_seconds=startup_timeout,
        drain_timeout_seconds=drain_timeout,
    )
    config.validate(
        {a: system.catalog["models"][m]["supported_tasks"] for a, m in aliases.items()}
    )
    pool = SamplePool(samples, list(system.workload), seed)
    registry = registry_for(system, aliases, backend, workers or {})
    checks = None
    if backend == "native":
        checks = preflight(
            system, winner, samples, workers or {}, admission, require_gpu=True
        )
        if not checks["ready"]:
            raise ValueError("Native preflight failed: " + "; ".join(checks["issues"]))
    elif admission is not None:
        raise ValueError("Admission evidence applies to native runs only")
    runtime = DeploymentRuntime(config, registry)
    monitor = MemoryMonitor(
        runtime,
        system.gpu_memory_mib,
        enabled=backend == "native",
        device=checks["device"] if checks else "0",
    )
    raw = {
        "schema_version": 1,
        "backend": backend,
        "config": system.as_dict(),
        "transport_config": config.as_dict(),
        "replica_to_family": aliases,
        "plan": winner,
        "status": "starting",
        "errors": [],
        "request_records": [],
        "warmup_request_records": [],
        "priming_request_records": [],
        "measurement_window": None,
        "preflight": checks,
        "worker_registry": registry,
        "admission_evidence": (
            checks["admission_evidence"]
            if checks
            else {"verified": False, "reason": "MOCK_BACKEND", "pairs": []}
        ),
        "memory_evidence": {},
        "per_task_metrics": {},
        "system_metrics": None,
    }
    before = after = None
    power = PowerMonitor(
        enabled=backend == "native", device=checks["device"] if checks else "0"
    )
    power.sample()
    startup_begin = time.perf_counter_ns()
    power.start()
    try:
        runtime.start()
        for task, route in routes.items():
            for worker in route:
                item = pool.next(task)
                row = record(
                    "priming",
                    len(raw["priming_request_records"]),
                    task,
                    worker,
                    item,
                    time.perf_counter_ns(),
                )
                raw["priming_request_records"].append(row)
                runtime.enqueue(row, item["sample"])
        runtime.drain(raw["priming_request_records"], drain_timeout)
        if any(r["status"] != "completed" for r in raw["priming_request_records"]):
            raise RuntimeError("Priming failed")
        if warmup > 0:
            raw["warmup_window"] = window(
                runtime,
                config,
                samples,
                "warmup",
                warmup,
                raw["warmup_request_records"],
                monitor,
            )
            if any(r["status"] != "completed" for r in raw["warmup_request_records"]):
                raise RuntimeError("Warmup failed")
        before = monitor.sample()
        monitor.start()
        monitor.check()
        raw["measurement_window"] = {}
        window(
            runtime,
            config,
            samples,
            "measurement",
            duration,
            raw["request_records"],
            monitor,
            raw["measurement_window"],
        )
        after = monitor.sample()
        monitor.check()
        raw["status"] = (
            "completed"
            if all(r["status"] == "completed" for r in raw["request_records"])
            else "completed_with_failures"
        )
    except (Exception, KeyboardInterrupt) as exc:
        raw["status"] = "failed"
        raw["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if (
            raw["measurement_window"]
            and "drain_end_time_ns" not in raw["measurement_window"]
        ):
            raw["measurement_window"].update(
                aborted=True, drain_end_time_ns=time.perf_counter_ns()
            )
        monitor.close()
        power.close()
        try:
            raw["shutdown"] = runtime.close()
        except Exception as exc:
            raw["shutdown"] = {}
            raw["errors"].append(f"shutdown: {exc}")
            raw["status"] = "failed"
        raw["errors"].extend(runtime.errors)
        if runtime.errors or any(
            v["forced"] or v["exit_code"] != 0 or not v["bye_received"]
            for v in raw["shutdown"].values()
        ):
            raw["status"] = "failed"
        for key in (
            "request_records",
            "warmup_request_records",
            "priming_request_records",
        ):
            for row in raw[key]:
                if row["status"] in {"planned", "queued", "submitted"}:
                    row.update(status="incomplete", error="run ended before response")
            annotate_records(raw[key])
    measured = before is not None and after is not None and not monitor.failed
    nominal = system.catalog["safe_factor"] * sum(
        system.catalog["memory_table_mib"][m][str(n)]
        for m, n in winner["deployment"].items()
    )
    safe = (
        system.catalog["safe_factor"]
        * max(before["owned_used_mib"], after["owned_used_mib"])
        if measured
        else nominal
    )
    raw["memory_evidence"] = {
        "measured": measured,
        "safe_memory_mib": safe,
        "resource_valid": measured
        and safe <= system.gpu_memory_mib
        and all(
            not s["foreign_pids"] and s["owned_used_mib"] <= system.gpu_memory_mib
            for s in monitor.samples
        ),
        "scope": (
            "owned-process idle memory and sampled peak"
            if measured
            else "catalog planning surrogate; diagnostic only"
        ),
        "samples": monitor.samples,
        "before": before,
        "after": after,
    }
    w = raw["measurement_window"]
    run_j = (
        integrate_power(power.samples, w["start_time_ns"], w["end_time_ns"])
        if w
        else None
    )
    once_j = (
        integrate_power(power.samples, startup_begin, w["start_time_ns"]) if w else None
    )
    raw["energy_evidence"] = {
        "kind": "measured" if backend == "native" else "unmeasured",
        "scope": "whole_gpu_board",
        "coverage_valid": backend == "native"
        and not power.failed
        and run_j is not None
        and once_j is not None,
        "run_joules": run_j,
        "startup_joules": once_j,
        "samples": power.samples,
        "sampling_errors": power.errors,
        "startup_begin_time_ns": startup_begin,
        "scope_note": "total GPU board energy including load/prime/warmup; excludes CPU and bounded post-window drain",
        "profile_key": energy_key(
            winner["deployment"],
            winner["routing"],
            system.workload,
            duration,
            seed,
            arrival_process,
            catalog=system.catalog,
        ),
    }
    if raw["measurement_window"]:
        raw["per_task_metrics"], raw["system_metrics"], raw["model_metrics"] = (
            summarize(raw["request_records"], config, raw["measurement_window"])
        )
        raw["evaluation"] = evaluate_run(system, raw)
    return raw
