"""Strict all-task native mapper, independent of historical text-only audits.

G2 image/ASR language-forward semantics are observed exactly like text. Media
preprocessing remains in the mixed prepare interval, never labelled GPU-only.
New collections must explicitly bind per-alias tasks for background jobs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from .profiling_analysis import digest
from .semantic_profiles import parse_semantic_request

RESIDENTS = {"text#0":"text","text#1":"text","generalist_2#0":"generalist_2","ocr#0":"ocr"}
ALLOWED = {"text":{"text"},"generalist_2":{"text","image","asr","ocr"},"ocr":{"ocr"}}
TRACE_SCHEMA = "gsdeploy.observed_background_execution.v1"


def receipt(path):
    path = Path(path).resolve()
    return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def checked_receipt(value):
    if not isinstance(value,dict) or set(value)!={"path","sha256"} or receipt(value["path"])!=value:
        raise ValueError("Changed or invalid source receipt")
    return Path(value["path"])


def ns(value):
    if type(value) is not int or value<0:
        raise ValueError("Integer monotonic timestamp required")
    return value


def positive(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError("Finite nonnegative telemetry required")
    return value


def _events(row,response,pid):
    events = row.get("execution_events",[])
    start = response["start_perf_ns"]
    starts = [e for e in events if "event" not in e and e.get("perf_ns")==start and e.get("pid")==pid]
    if len(starts)!=1 or starts[0].get("request_id")!=row["request_id"]:
        raise ValueError("Actual START event/PID missing or inconsistent")
    observed = response["observed_stages"]+response.get("semantic_forward_observations",{}).get("records",[])
    keys = set()
    for stage in observed:
        key = stage["id"],stage["kind"]
        if key in keys:
            raise ValueError("Duplicate observed phase identity")
        keys.add(key)
        for edge in ("start","end"):
            matching = [e for e in events if e.get("request_id")==row["request_id"] and
                e.get("segment_id")==stage["id"] and e.get("kind")==stage["kind"] and e.get("event")==edge]
            if len(matching)!=1 or matching[0].get("perf_ns")!=stage[edge+"_ns"]:
                raise ValueError("Observed phase does not match emitted boundary")
            expected = f"GS_STAGE:{row['request_id']}:{stage['kind']}:{stage['id']}"
            if matching[0].get("nvtx_name")!=expected:
                raise ValueError("Phase NVTX/request identity mismatch")


def map_background_collection(raw_path):
    raw_path = Path(raw_path).resolve()
    raw = json.loads(raw_path.read_text())
    if (raw.get("schema")!="gsdeploy.native_phase_collection.v1" or raw.get("kind")!="measured"
            or raw.get("backend")!="native" or raw.get("status")!="completed"):
        raise ValueError("Complete measured native collection required")
    if any(raw.get(k) for k in ("errors","runtime_errors","telemetry_errors")):
        raise ValueError("Collection errors prevent training")
    context = json.loads(checked_receipt(raw["context"]).read_text())
    plan = json.loads(checked_receipt(raw["plan"]).read_text())
    executed = json.loads(checked_receipt(raw["executed_jobs"]).read_text())
    if context.get("residents")!=RESIDENTS or plan.get("resident_aliases")!=RESIDENTS:
        raise ValueError("Exact frozen residency required")
    snapshots = context.get("execution_sources",[])
    if not snapshots or context.get("sources")!=[s["frozen"] for s in snapshots]:
        raise ValueError("Explicit consistent frozen source snapshot required")
    original_paths = set()
    for source in snapshots:
        checked_receipt(source["frozen"])
        if source["original_path"] in original_paths:
            raise ValueError("Duplicate original snapshot source")
        original_paths.add(source["original_path"])
    content = {k:v for k,v in context.items() if k not in {"sources","execution_sources"}}
    content["source_content"] = [{"original_path":s["original_path"],"sha256":s["frozen"]["sha256"]} for s in snapshots]
    identity = digest(content)
    if identity!=raw["execution_context_sha256"] or identity!=plan["execution_context_sha256"]:
        raise ValueError("Snapshot execution identity mismatch")
    plan_content = {k:v for k,v in plan.items() if k!="plan_sha256"}
    if plan.get("plan_sha256")!=digest(plan_content):
        raise ValueError("Frozen plan self hash mismatch")
    jobs = {j["id"]:j for j in plan["jobs"]}
    executed_jobs = {j["id"]:j for j in executed["jobs"]}
    if len(jobs)!=len(plan["jobs"]) or len(executed_jobs)!=len(executed["jobs"]):
        raise ValueError("Duplicate job identities")
    if any(key not in jobs or jobs[key]!=job for key,job in executed_jobs.items()):
        raise ValueError("Executed jobs differ from frozen plan")
    selected_inputs = {(r["task"],r["pool_id"]) for r in plan["inputs"]}
    ready = raw["ready"]
    if set(ready)!=set(RESIDENTS) or len({r["pid"] for r in ready.values()})!=len(RESIDENTS):
        raise ValueError("Every resident must have distinct actual READY PID")
    for alias,item in ready.items():
        family = RESIDENTS[alias]
        if type(item["pid"]) is not int or item["pid"]<=0 or item.get("worker")!=family:
            raise ValueError("READY resident identity mismatch")
        binding = context["worker_bindings"][family]
        if item.get("spec_sha256")!=binding["spec_sha256"]:
            raise ValueError("READY worker spec differs from frozen binding")
        if not ALLOWED[family]>=set(item["tasks"]):
            raise ValueError("Worker advertises tasks outside allowed family")
    samples = raw["board_samples"]
    if len(samples)<2:
        raise ValueError("Bracketing whole-board telemetry required")
    previous = -1
    pids = {str(v["pid"]) for v in ready.values()}
    for sample in samples:
        timestamp = ns(sample["time_ns"])
        if timestamp<=previous:
            raise ValueError("Telemetry timestamps not strictly increasing")
        previous = timestamp
        positive(sample["watts"]);positive(sample["owned_used_mib"])
        if sample.get("foreign_pids") or set(sample["owned_aliases"])!=set(RESIDENTS):
            raise ValueError("Telemetry must cover all owned residents without foreign processes")
        processes = sample["processes_mib"]
        if set(map(str,processes))!=pids:
            raise ValueError("Actual owned telemetry PID mismatch")
        total = sum(positive(v) for v in processes.values())
        if abs(total-sample["owned_used_mib"])>1e-6:
            raise ValueError("Owned memory sum mismatch")
    if not samples[0]["time_ns"]<=ns(raw["observation_start_ns"])<=ns(raw["observation_end_ns"])<=samples[-1]["time_ns"]:
        raise ValueError("Whole collection telemetry not bracketed")
    records = {r["request_id"]:r for r in raw["request_records"]}
    if len(records)!=len(raw["request_records"]):
        raise ValueError("Duplicate request identities")
    sources = [receipt(raw_path),raw["context"],raw["plan"],raw["executed_jobs"]]
    traces,used_jobs,used_requests = [],set(),set()
    for batch in raw["batches"]:
        job_id = batch["job_id"]
        if job_id in used_jobs or job_id not in executed_jobs:
            raise ValueError("Repeated or unexecuted batch")
        used_jobs.add(job_id);job = jobs[job_id]
        left,right = ns(batch["start_ns"]),ns(batch["end_ns"])
        if not raw["observation_start_ns"]<=left<right<=raw["observation_end_ns"]:
            raise ValueError("Batch outside actual observation window")
        ids = batch["request_ids"]
        if len(set(ids))!=len(ids) or used_requests.intersection(ids):
            raise ValueError("Repeated request in batches")
        used_requests.update(ids)
        actual = [records[rid] for rid in ids]
        if len({r["model"] for r in actual})!=len(actual) or {r["model"]:r["pool_id"] for r in actual}!=job["inputs"]:
            raise ValueError("Batch inputs differ from planned aliases")
        requests = []
        split = "training" if job["split"]=="training" else "validation"
        for row in actual:
            alias = row["model"];family = RESIDENTS[alias];task = row["task"]
            default = "ocr" if family=="ocr" else "text"
            expected_task = job.get("tasks",{}).get(alias,default)
            if task!=expected_task or task not in ALLOWED[family] or task not in ready[alias]["tasks"]:
                raise ValueError("Task differs from explicit job/family contract")
            if (task,row["pool_id"]) not in selected_inputs:
                raise ValueError("Task/input not in frozen public pool")
            if row.get("profile_job_id")!=job_id or row.get("split")!=job["split"] or row.get("status")!="completed":
                raise ValueError("Request job/split/status mismatch")
            response = row["worker_response"]
            if response.get("request_id")!=row["request_id"] or response.get("task_type")!=task or response.get("task")!=task:
                raise ValueError("Response request/task identity mismatch")
            start,end = ns(response["start_perf_ns"]),ns(response["end_perf_ns"])
            if start!=row["observed_worker_start_time_ns"] or not left<=start<end<=right:
                raise ValueError("Actual START/RESULT anchors mismatch")
            stages = response.get("observed_stages",[])
            if not stages:
                raise ValueError("Observed native stages required")
            cursor,segments,stage_ids = start,[],set()
            for stage in stages:
                a,b = ns(stage["start_ns"]),ns(stage["end_ns"])
                if not cursor<=a<b<=end or stage["id"] in stage_ids:
                    raise ValueError("Native stages out of order/duplicate")
                stage_ids.add(stage["id"])
                if stage["kind"] not in {"gpu","host","sync","cleanup"}:
                    raise ValueError("Unknown explicit native call kind")
                if cursor<a:
                    segments.append({"id":"control_"+str(len(segments)),"kind":"mixed","start_ns":cursor,"end_ns":a})
                segments.append({"id":stage["id"],"kind":stage["kind"],"start_ns":a,"end_ns":b})
                cursor = b
            if cursor<end:
                segments.append({"id":"control_end","kind":"mixed","start_ns":cursor,"end_ns":end})
            _events(row,response,ready[alias]["pid"])
            semantic = parse_semantic_request({"request_id":row["request_id"],"alias":alias,"family":family,"task":task,
                "pool_id":row["pool_id"],"handler_start_ns":start,"handler_end_ns":end},response)
            request = {**semantic,"segments":segments,"semantic_segments":semantic["segments"],
                "execution":context["execution"],"start_basis":"observed_worker_START"}
            for segment in request["segments"]:
                segment.update(demand={"sm":None,"dram":None,"l2":None},
                    observation_basis="Actual explicit native call host boundaries; mixed wall, not kernel-only")
            requests.append(request)
        before = [s for s in samples if s["time_ns"]<=left]
        after = [s for s in samples if s["time_ns"]>=right]
        if not before or not after:
            raise ValueError("Batch telemetry not bracketed")
        selected = [before[-1]]+[s for s in samples if left<s["time_ns"]<right]+[after[0]]
        traces.append({"schema":TRACE_SCHEMA,"kind":"measured","backend":"native","id":job_id,"split":split,
            "clock":"host_monotonic_ns","execution_context_sha256":identity,"sources":sources,
            "observation_window_ns":{"start":left,"end":right},"admission_enabled":False,"requests":requests,
            "power_samples":[{"time_ns":s["time_ns"],"watts":s["watts"]} for s in selected],
            "memory_samples":[{k:s[k] for k in ("time_ns","owned_used_mib","owned_aliases","foreign_pids")} for s in selected],
            "performance_scope":raw.get("performance_scope"),"scope":"All-task native mixed-phase observations; counters unknown"})
    if used_requests!=set(records) or used_jobs!=set(executed_jobs):
        raise ValueError("Unmapped request or executed job")
    return {"traces":traces,"mapping_report":{"training_traces":sum(t["split"]=="training" for t in traces),
        "validation_traces":sum(t["split"]=="validation" for t in traces),"trace_count":len(traces),
        "tasks":sorted({r["task"] for t in traces for r in t["requests"]}),"snapshot_verified":True,
        "actual_START_PHASE_events_verified":True,"all_resident_telemetry_verified":True,"kernel_evidence_available":False}}


def background_semantic_traces(raw_path):
    mapped = map_background_collection(raw_path)
    traces = copy.deepcopy(mapped["traces"])
    for trace in traces:
        for row in trace["requests"]:
            row["segments"] = row.pop("semantic_segments")
    return traces,mapped["mapping_report"]


semantic_traces = background_semantic_traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    value = map_background_collection(args.raw)
    with args.output.open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")


if __name__=="__main__":main()
