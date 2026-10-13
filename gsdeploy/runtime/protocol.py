import json
import sys
import time
import traceback
import os
import ctypes
import ctypes.util


def emit(kind, payload):
    print(kind + "|" + json.dumps(payload, ensure_ascii=False), flush=True)


def _start_cuda_profiler():
    if os.environ.get("SPECIALIST_NSYS_CAPTURE", "0") != "1":
        raise RuntimeError("SPECIALIST_NSYS_CAPTURE disabled")
    candidates = []
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    candidates.extend(["libcudart.so", "libcudart.so.13", "libcudart.so.12"])
    last_error = None
    cudart = None
    for name in candidates:
        try:
            cudart = ctypes.CDLL(name)
            break
        except OSError as e:
            last_error = e
    if cudart is None:
        raise RuntimeError("Could not load libcudart: " + repr(last_error))
    sync_rc = cudart.cudaDeviceSynchronize()
    rc = cudart.cudaProfilerStart()
    return {
        "profile_control": True,
        "cuda_sync_rc": int(sync_rc),
        "cuda_profiler_start_rc": int(rc),
    }


def run_worker(worker_name, handler, ready_payload=None):
    payload = {"worker": worker_name, "pid": __import__("os").getpid()}
    if ready_payload:
        payload.update(ready_payload)
    emit("READY", payload)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("op") == "shutdown":
                emit("BYE", {"worker": worker_name})
                break
            if request.get("task_type") == "__profile_start__":
                start_ns = time.time_ns()
                result = _start_cuda_profiler()
                end_ns = time.time_ns()
                output = {
                    "worker": worker_name,
                    "request_id": request["request_id"],
                    "pool_id": request["pool_id"],
                    "arrival_index": request["arrival_index"],
                    "task_type": request["task_type"],
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "worker_latency_ms": (end_ns - start_ns) / 1000000.0,
                }
                output.update(result)
                emit("RESULT", output)
                continue
            start_ns = time.time_ns()
            result = handler(request)
            end_ns = time.time_ns()
            output = {
                "worker": worker_name,
                "request_id": request["request_id"],
                "pool_id": request["pool_id"],
                "arrival_index": request["arrival_index"],
                "task_type": request["task_type"],
                "submit_ns": request.get("submit_ns"),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "worker_latency_ms": (end_ns - start_ns) / 1000000.0,
            }
            submit_ns = request.get("submit_ns")
            if submit_ns is not None:
                output["queue_wait_ms"] = (start_ns - submit_ns) / 1000000.0
            output.update(result)
            emit("RESULT", output)
        except Exception as e:
            emit(
                "ERROR",
                {
                    "worker": worker_name,
                    "request_id": (
                        request.get("request_id") if "request" in locals() else None
                    ),
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                },
            )


__all__ = ["emit", "run_worker"]
