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
from ..evaluation import evaluate_run
from ..metrics import annotate_records, summarize
from ..routing import Router


def poisson_arrivals(rates, duration, seed):
    import heapq

    rng = {task: random.Random(f"{seed}:arrival:{task}") for task in rates}
    events = [(rng[t].expovariate(q), t) for t, q in rates.items()]
    heapq.heapify(events)
    while events:
        offset, task = heapq.heappop(events)
        if offset >= duration:
            break
        yield offset, task
        heapq.heappush(events, (offset + rng[task].expovariate(rates[task]), task))


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


def window(runtime, config, pool_path, phase, duration, records, monitor):
    pool = SamplePool(pool_path, list(config.workload_rps), config.random_seed)
    router = Router(config.routing_rps)
    start, unix = time.perf_counter_ns(), time.time_ns()
    end = start + round(duration * 1e9)
    arrivals = (
        periodic_arrivals
        if config.arrival_process == "deterministic_periodic"
        else poisson_arrivals
    )
    for index, (offset, task) in enumerate(
        arrivals(config.workload_rps, duration, config.random_seed)
    ):
        item = pool.next(task)
        row = record(
            phase, index, task, router.choose(task), item, start + round(offset * 1e9)
        )
        records.append(row)
        wait_until(row["scheduled_time_ns"], monitor)
        runtime.enqueue(row, item["sample"])
    wait_until(end, monitor)
    arrival_end = time.perf_counter_ns()
    runtime.drain(records, config.drain_timeout_seconds)
    drain_end = time.perf_counter_ns()
    return {
        "start_time_ns": start,
        "start_unix_ns": unix,
        "end_time_ns": end,
        "duration_seconds": duration,
        "arrival_loop_end_time_ns": arrival_end,
        "drain_end_time_ns": drain_end,
        "drain_seconds": (drain_end - end) / 1e9,
        "cohort": "requests scheduled in [start,end); completed up to bounded drain",
    }


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
            "PYTHONPATH": project + os.pathsep + os.environ.get("PYTHONPATH", ""),
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
    quality_admission_confirmed=False,
):
    if backend not in {"mock", "native"}:
        raise ValueError("backend must be mock or native")
    if arrival_process not in {"deterministic_periodic", "poisson"}:
        raise ValueError("Unknown arrival process")
    aliases = {f"{m}#{i}": m for m, n in winner["deployment"].items() for i in range(n)}
    routes = {
        t: {
            f"{m}#{i}": q / winner["deployment"][m]
            for m, q in row.items()
            if q > 0
            for i in range(winner["deployment"][m])
        }
        for t, row in winner["routing"].items()
    }
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
    runtime = DeploymentRuntime(
        config, registry_for(system, aliases, backend, workers or {})
    )
    monitor = MemoryMonitor(runtime, system.gpu_memory_mib, enabled=backend == "native")
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
        "quality_admission_confirmed": bool(quality_admission_confirmed),
        "memory_evidence": {},
        "per_task_metrics": {},
        "system_metrics": None,
    }
    before = after = None
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
        raw["measurement_window"] = window(
            runtime,
            config,
            samples,
            "measurement",
            duration,
            raw["request_records"],
            monitor,
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
        monitor.close()
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
    if raw["measurement_window"]:
        raw["per_task_metrics"], raw["system_metrics"], raw["model_metrics"] = (
            summarize(raw["request_records"], config, raw["measurement_window"])
        )
        raw["evaluation"] = evaluate_run(system, raw)
    return raw
