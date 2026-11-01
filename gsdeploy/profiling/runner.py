"""Isolated same-GPU replicas, complete request cohorts and raw telemetry."""

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import statistics
import time
import traceback

from .telemetry import BoardMonitor


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker(connection, spec, resources):
    try:
        import torch
        if spec.get("adapter") == "compact":
            from ..backends.compact_native import ProfileAdapter as Adapter
        else:
            from .adapters import Adapter

        torch.set_num_threads(1)
        torch.manual_seed(20261015)
        print("Loading", spec["id"], "PID", os.getpid(), flush=True)
        adapter = Adapter(spec, resources)
        print("Ready", spec["id"], "PID", os.getpid(), flush=True)
        connection.send({
            "status": "READY", "pid": os.getpid(), "memory": adapter.memory(),
            "parameters": adapter.parameter_count, "parameter_bytes": adapter.parameter_bytes,
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "memory_optimizations": getattr(getattr(adapter, 'executor', None), 'memory_audit', {}),
        })
        while True:
            message = connection.recv()
            if message["command"] == "STOP":
                return
            if message["command"] == "RESET":
                torch.cuda.reset_peak_memory_stats()
                connection.send({"status": "RESET", "memory": adapter.memory()})
                continue
            try:
                result = adapter.run(message["sample"])
                result.update({"status": "RESULT", "request_id": message["request_id"], "memory": adapter.memory()})
                connection.send(result)
            except Exception as exc:
                connection.send({
                    "status": "REQUEST_FAILED", "request_id": message["request_id"],
                    "sample_id": message["sample"]["id"], "task": message["sample"]["task"],
                    "success": False, "error": type(exc).__name__, "traceback": traceback.format_exc(),
                })
                return
    except Exception as exc:
        connection.send({"status": "LOAD_FAILED", "error": type(exc).__name__, "traceback": traceback.format_exc()})


def receive(connection, process, timeout=600):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if connection.poll(0.2):
            return connection.recv()
        if not process.is_alive():
            raise RuntimeError(f"Worker exited: {process.exitcode}")
    raise TimeoutError("Worker response timeout")


def quantile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    left = math.floor(index)
    return ordered[left] + (ordered[math.ceil(index)] - ordered[left]) * (index - left)


def summary(records, seconds, energy):
    values = [r["service_ms"] for r in records if "service_ms" in r]
    successful = sum(r.get("success", False) for r in records)
    return {
        "offered_requests": len(records), "completed_requests": len(records),
        "successful_requests": successful, "failed_requests": len(records) - successful,
        "wall_seconds": seconds,
        "completed_rps": len(records) / seconds,
        "successful_rps": successful / seconds,
        "service_mean_ms": statistics.mean(values) if values else None,
        "service_p50_ms": quantile(values, .5) if values else None,
        "service_p95_ms": quantile(values, .95) if values else None,
        "service_p99_ms": quantile(values, .99) if values else None,
        "mean_client_ms": statistics.mean(r["client_ms"] for r in records),
        "board_energy": energy,
        "joules_per_offered_request": energy["joules"] / len(records) if energy["joules"] is not None else None,
        "mean_board_watts": energy["joules"] / seconds if energy["joules"] is not None else None,
        "torch_peak_allocated_mib_sum": sum(
            max(r.get("memory", {}).get("peak_allocated_mib", 0) for r in records if r["replica"] == i)
            for i in {r["replica"] for r in records}
        ),
    }


def run_profile(spec, resources, samples, replicas, output, rounds=3, minimum_seconds=30, minimum_requests=64,
                warmup_samples=None, exact_cohort=False):
    ctx = mp.get_context("spawn")
    output = Path(output)
    if (output / "profile.json").exists():
        raise FileExistsError(f"Profile output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    monitor = BoardMonitor()
    baseline = monitor.sample()
    if baseline["processes"]:
        raise RuntimeError("GPU is occupied before profile start")
    monitor.start()
    startup_start = monitor.sample()
    processes, connections, ready, warmups = [], [], [], []
    result = {
        "schema_version": 1, "kind": "measured", "scope": "standalone_same_model_replicas",
        "model": spec, "replicas": replicas, "dtype": spec.get("execution", {}).get("dtype", "bfloat16"), "batch_size_per_replica": 1,
        "cpu_offload": False, "quantization": spec.get("execution", {}).get("quantization"),
        "minimum_seconds": minimum_seconds, "minimum_requests": minimum_requests,
        "manifest_sha256": hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(),
        "filesystem_cache": "uncontrolled; repeated checkpoint reads may be cached",
        "workload": "saturated closed loop, one outstanding request per replica; no serving queue offered-load claim",
        "rounds": [], "status": "RUNNING",
        "gpu_uuid": monitor.nv.nvmlDeviceGetUUID(monitor.handle),
        "hardware": {"gpu_name": monitor.nv.nvmlDeviceGetName(monitor.handle),
                     "gpu_memory_mib": monitor.nv.nvmlDeviceGetMemoryInfo(monitor.handle).total / 2**20,
                     "power_limit_watts": monitor.nv.nvmlDeviceGetPowerManagementLimit(monitor.handle) / 1000,
                     "driver_version": monitor.nv.nvmlSystemGetDriverVersion()},
    }
    save(output / "profile.json", result)
    stage = 'loading'
    failure_detail = None
    try:
        for i in range(replicas):
            stage = 'loading'
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(child, spec, resources), name=f"profile-replica-{i}")
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
            message = receive(parent, process)
            ready.append(message)
            save(output / "startup.json", ready)
            if message["status"] != "READY":
                raise RuntimeError("Model failed to load; see startup.json")
            prime = samples[:3] if warmup_samples is None else warmup_samples
            stage = 'warmup'
            for j, sample in enumerate(prime):
                parent.send({"command": "INFER", "sample": sample, "request_id": f"warmup-{i}-{j}"})
                response = receive(parent, process)
                warmups.append(response)
                save(output / "warmup.json", warmups)
                if response["status"] != "RESULT" or response.get('success') is False:
                    failure_detail = {key: response.get(key) for key in
                                      ['sample_id', 'task', 'status', 'failure_reason', 'finish_reason', 'error']}
                    raise RuntimeError("Warmup failed; see warmup.json")
        stage = 'memory_screen'
        resident = monitor.sample()
        result["resident_after_warmup_mib"] = resident["used_mib"]
        result["safe_factor"] = 1.110854
        result["safe_memory_mib"] = resident["used_mib"] * result["safe_factor"]
        result["budget_mib"] = 49140
        if result["safe_memory_mib"] > result["budget_mib"]:
            raise RuntimeError("Warm resident memory exceeds safety-adjusted GPU budget")
        result["startup_energy"] = monitor.window(startup_start, resident)
        result["startup_seconds"] = (resident["time_ns"] - startup_start["time_ns"]) / 1e9
        phases = [(task, [s for s in samples if s["task"] == task])
                  for task in sorted({s["task"] for s in samples})]
        if spec["kind"] == "speech" and {s["task"] for s in samples} == {"asr", "lid"}:
            phases.append(("asr_lid_mix_4_to_5", samples))
        for task, task_samples in phases:
            stage = 'measurement'
            for repetition in range(rounds):
                order = list(task_samples)
                random.Random(20261015 + repetition).shuffle(order)
                for connection, process in zip(connections, processes):
                    connection.send({"command": "RESET"})
                    receive(connection, process)
                before = monitor.sample()
                start = time.monotonic()
                submitted, records, outstanding = 0, [], {}
                cohort_path = output / f"{task}-round{repetition}.jsonl"

                def dispatch(index):
                    nonlocal submitted
                    request_id = f"{task}-{repetition}-{submitted}"
                    sample = order[submitted % len(order)]
                    outstanding[index] = (request_id, time.perf_counter_ns())
                    connections[index].send({"command": "INFER", "sample": sample, "request_id": request_id})
                    submitted += 1

                for index in range(replicas):
                    dispatch(index)
                with cohort_path.open("w") as stream:
                    while outstanding:
                        progressed = False
                        for index in list(outstanding):
                            if not connections[index].poll(0.01):
                                if not processes[index].is_alive():
                                    raise RuntimeError("Worker died with requests outstanding")
                                if (time.perf_counter_ns() - outstanding[index][1]) / 1e9 > 300:
                                    raise TimeoutError("Request exceeded 300 seconds")
                                continue
                            response = connections[index].recv()
                            request_id, sent = outstanding.pop(index)
                            response.update({"replica": index, "client_ms": (time.perf_counter_ns() - sent) / 1e6})
                            if response.get("request_id") != request_id:
                                raise RuntimeError("Request identity mismatch")
                            records.append(response)
                            stream.write(json.dumps(response, ensure_ascii=False) + "\n")
                            stream.flush()
                            if monitor.samples and max(v['used_mib'] for v in monitor.samples) * result['safe_factor'] > result['budget_mib']:
                                raise RuntimeError('Peak board memory exceeds safety-adjusted GPU budget')
                            if response["status"] != "RESULT":
                                raise RuntimeError("Request failure; see raw cohort records")
                            progressed = True
                            again = submitted < len(order) if exact_cohort else (
                                submitted < max(minimum_requests, len(order)) or time.monotonic() - start < minimum_seconds)
                            if again:
                                dispatch(index)
                        if not progressed:
                            time.sleep(.001)
                after = monitor.sample()
                assert len(records) == submitted
                seconds = (after["time_ns"] - before["time_ns"]) / 1e9
                row = summary(records, seconds, monitor.window(before, after))
                row.update({"task": task, "repetition": repetition, "raw_records": str(cohort_path)})
                result["rounds"].append(row)
                result["peak_board_used_mib"] = max(v["used_mib"] for v in monitor.samples)
                result["status"] = "RUNNING"
                save(output / "profile.json", result)
                print(spec["id"], "replicas", replicas, task, "round", repetition,
                      "mean_ms", round(row["service_mean_ms"], 2), "rps", round(row["completed_rps"], 3), flush=True)
        result["status"] = "COMPLETE"
    except Exception as exc:
        result["status"] = "FAILED"
        result['failure_stage'] = stage
        result['failure_detail'] = failure_detail
        result["error"] = type(exc).__name__ + ": " + str(exc)
        result["traceback"] = traceback.format_exc()
    except KeyboardInterrupt:
        result["status"] = "INTERRUPTED"
        result["error"] = "Interrupted; partial measurements retained, not a completed cohort"
        raise
    finally:
        for connection, process in zip(connections, processes):
            if process.is_alive():
                try:
                    connection.send({"command": "STOP"})
                except (BrokenPipeError, OSError):
                    pass
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            connection.close()
        monitor.close()
        result["telemetry_errors"] = monitor.errors
        result["ready"] = ready
        if monitor.samples:
            result["peak_board_used_mib"] = max(v['used_mib'] for v in monitor.samples)
        save(output / "telemetry.json", monitor.samples)
        save(output / "profile.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--minimum-seconds", type=float, default=30)
    parser.add_argument("--minimum-requests", type=int, default=64)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    samples = [json.loads(line) for line in Path(args.samples).read_text().splitlines() if line.strip()]
    result = run_profile(spec["model"], spec["resources"], samples, args.replicas, args.output,
                         args.rounds, args.minimum_seconds, args.minimum_requests)
    raise SystemExit(0 if result["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
