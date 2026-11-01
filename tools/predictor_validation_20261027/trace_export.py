"""Export genuine CUPTI kernel intervals; NVTX provides labels, not GPU evidence.

Worker anchors are explicit phase boundaries from time.perf_counter_ns(). Each
anchor has request_id, segment_id, kind, pid, start_ns, end_ns. Timestamp mapping
is fitted independently per process and its residual is reported, never assumed
identical to the Nsight relative clock. Missing counters remain None.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import subprocess

from .nvtx_interval_index_v2 import NVTXRangeIndex

SCHEMA = "gsdeploy.nsys_observed_phases.v1"


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def export_sqlite(report, destination):
    report, destination = Path(report).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    command = ["/usr/local/bin/nsys", "export", "--type", "sqlite", "--output", str(destination), str(report)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    record = {"command": command, "returncode": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr, "source": receipt(report)}
    if result.returncode != 0 or not destination.is_file():
        raise RuntimeError(json.dumps(record))
    record["output"] = receipt(destination)
    return record


def linux_pid(global_id):
    return (int(global_id) >> 24) & 0xFFFFFF


def anchors_from_worker_lines(lines):
    """Pair emitted PHASE boundaries using observed READY/START worker PID."""
    pid, request_pids, starts, anchors = None, {}, {}, []
    for line in lines:
        if "|" not in line:
            continue
        event, payload = line.rstrip("\n").split("|", 1)
        if event not in {"READY", "START", "PHASE"}:
            continue
        row = json.loads(payload)
        if event == "READY":
            pid = row["pid"]
        elif event == "START":
            request_pids[row["request_id"]] = row["pid"]
        else:
            key = row["request_id"], row["kind"], row["segment_id"]
            if row["event"] == "start":
                if key in starts:
                    raise ValueError("Duplicate phase start")
                starts[key] = row["perf_ns"]
            elif row["event"] == "end":
                if key not in starts:
                    raise ValueError("Phase end without start")
                worker_pid = request_pids.get(key[0], pid)
                if worker_pid is None:
                    raise ValueError("Missing observed worker PID")
                anchors.append({"pid": worker_pid, "request_id": key[0], "kind": key[1],
                                "segment_id": key[2], "start_ns": starts.pop(key),
                                "end_ns": row["perf_ns"]})
    if starts:
        raise ValueError("Incomplete phase boundaries")
    return anchors


def _stage_label(label):
    if not label.startswith("GS_STAGE:"):
        return None
    fields = label[len("GS_STAGE:"):].rsplit(":", 2)
    if len(fields) != 3 or any(not field for field in fields):
        raise ValueError("Malformed GS_STAGE label: " + label)
    return tuple(fields)  # request, kind, segment


def parse_sqlite(path, anchors=(), *, max_anchor_skew_ns=5_000_000):
    path = Path(path).resolve()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
    required = {"StringIds", "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME"}
    if not required <= tables:
        raise ValueError("Missing Nsight CUDA/NVTX tables: " + repr(sorted(required - tables)))
    strings = {r["id"]: r["value"] for r in connection.execute("select * from StringIds")}
    ranges = []
    for row in connection.execute("select * from NVTX_EVENTS where end is not null order by start,end"):
        label = row["text"] or strings.get(row["textId"], "")
        stage = _stage_label(label)
        ranges.append({"label": label, "nsys_start_ns": row["start"], "nsys_end_ns": row["end"],
                       "global_tid": row["globalTid"], "pid": linux_pid(row["globalTid"]),
                       "stage": stage, "kernels": []})
    launches = {}
    for row in connection.execute("select * from CUPTI_ACTIVITY_KIND_RUNTIME"):
        if row["correlationId"] is not None:
            # correlation identifiers are process-scoped.
            launches[(linux_pid(row["globalTid"]), row["correlationId"])] = dict(row)
    # Launch correlation is process-scoped; NVTX ownership is thread-scoped.
    # A balanced containment index avoids scanning every range for each kernel.
    range_index = NVTXRangeIndex(ranges)
    unattributed = []
    for row in connection.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL order by start,end"):
        columns = set(row.keys())
        def dimensions(prefix):
            names = [prefix + axis for axis in ('X', 'Y', 'Z')]
            return [row[name] for name in names] if all(name in columns and row[name] is not None for name in names) else None
        cluster_raw = dimensions('cluster')
        kernel = {"name": strings.get(row["demangledName"], ""),
                  "nsys_start_ns": row["start"], "nsys_end_ns": row["end"],
                  "pid": linux_pid(row["globalPid"]), "correlation_id": row["correlationId"],
                  "device": row["deviceId"], "stream": row["streamId"],
                  "registers_per_thread": row["registersPerThread"],
                  "static_shared_bytes": row["staticSharedMemory"],
                  "dynamic_shared_bytes": row["dynamicSharedMemory"],
                  "block": dimensions('block'), "grid": dimensions('grid'),
                  "cluster": None if cluster_raw == [0, 0, 0] else cluster_raw,
                  "cluster_raw": cluster_raw,
                  "context_id": row['contextId'] if 'contextId' in columns else None,
                  "green_context_id": row['greenContextId'] if 'greenContextId' in columns else None,
                  "local_memory_per_thread_bytes": row['localMemoryPerThread'] if 'localMemoryPerThread' in columns else None,
                  "allocated_registers_per_block": None, "allocated_shared_bytes": None,
                  "allocation_evidence": None,
                  "static_metadata_scope": "Observed CUPTI launch attributes; architecture-rounded allocation and device-limit evidence are separate and remain unknown",
                  "demand": None, "demand_evidence": None,
                  "observation_basis": "CUPTI_ACTIVITY_KIND_KERNEL"}
        launch = launches.get((kernel["pid"], row["correlationId"]))
        owner = None if launch is None else range_index.owner_for_launch(
            launch["globalTid"], launch["start"], launch["end"])
        if owner is not None:
            owner["kernels"].append(kernel)
        else:
            unattributed.append(kernel)
    connection.close()
    anchor_map = {}
    for anchor in anchors:
        key = (anchor["pid"], anchor["request_id"], anchor["kind"], anchor["segment_id"])
        if key in anchor_map or not anchor["start_ns"] < anchor["end_ns"]:
            raise ValueError("Duplicate or invalid worker anchor")
        anchor_map[key] = anchor
    matched, process_offsets = [], {}
    for item in ranges:
        if item["stage"] is None:
            continue
        request_id, kind, segment_id = item["stage"]
        anchor = anchor_map.get((item["pid"], request_id, kind, segment_id))
        if anchor:
            for edge in ("start", "end"):
                process_offsets.setdefault(item["pid"], []).append(anchor[edge + "_ns"] - item["nsys_" + edge + "_ns"])
            matched.append((item, anchor))
    mappings = {}
    for pid, offsets in process_offsets.items():
        offset = round(statistics.median(offsets))
        residual = max(abs(value - offset) for value in offsets)
        mappings[pid] = {"method": "median_worker_phase_boundary_offset", "offset_ns": offset,
                         "max_boundary_residual_ns": residual, "boundary_count": len(offsets),
                         "valid": residual <= max_anchor_skew_ns,
                         "uncertainty_note": "Includes NVTX push/pop and worker timestamp placement; no drift estimate"}
    phases = []
    for item, anchor in matched:
        mapping = mappings[item["pid"]]
        kernels = item["kernels"]
        if mapping["valid"]:
            for kernel in kernels:
                kernel["start_ns"] = kernel["nsys_start_ns"] + mapping["offset_ns"]
                kernel["end_ns"] = kernel["nsys_end_ns"] + mapping["offset_ns"]
        # Sweep-line union preserves real GPU concurrency and does not count it twice.
        events = {}
        for kernel in kernels:
            for edge, delta in (("nsys_start_ns", 1), ("nsys_end_ns", -1)):
                events[kernel[edge]] = events.get(kernel[edge], 0) + delta
        count, previous, busy_ns = 0, None, 0
        for timestamp, delta in sorted(events.items()):
            if count and previous is not None:
                busy_ns += timestamp - previous
            count += delta
            previous = timestamp
        phases.append({**anchor, "nsys_start_ns": item["nsys_start_ns"], "nsys_end_ns": item["nsys_end_ns"],
                       "phase_scope": "Explicit native call range; may include host work, transfers, CUDA launches and waits",
                       "kernel_count": len(kernels), "kernel_union_busy_ns": busy_ns,
                       "kernel_duration_sum_ns": sum(k["nsys_end_ns"]-k["nsys_start_ns"] for k in kernels),
                       "kernels": kernels, "clock_mapping_valid": mapping["valid"],
                       "unclassified_gap_note": "NVTX phase duration minus kernels is not automatically CPU work",
                       "demand": None, "demand_evidence": None})
    return {"schema": SCHEMA, "sources": [receipt(path)],
            "nsys_clock": "relative_ns_requires_explicit_worker_mapping",
            "clock_mappings": {str(k): v for k, v in mappings.items()},
            "phases": phases, "nvtx_ranges": ranges, "unattributed_kernels": unattributed,
            "unmatched_anchor_count": len(anchor_map)-len(matched),
            "claim": "Measured CUDA kernel timings; no SM/DRAM/L2 demand or unlabelled host-phase attribution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    anchors = json.loads(args.anchors.read_text()) if args.anchors else []
    result = parse_sqlite(args.sqlite, anchors)
    if args.anchors:
        result["sources"].append(receipt(args.anchors))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
