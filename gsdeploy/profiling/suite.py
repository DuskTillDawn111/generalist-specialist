"""Sequential GPU measurement suite with evidence-based replica domains."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .runner import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--existing-qa", default="qa-n1-attempt2")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    kinds = {"qa": "text", "image": "image", "ocr": "ocr", "speech": "speech"}
    state = {"status": "RUNNING", "started_epoch": time.time(), "jobs": [], "gpu_runs_are_sequential": True}
    results = {}
    if args.retry_failed:
        state = json.loads((root / "suite-status.json").read_text())
        if state["status"] == "RUNNING":
            raise RuntimeError("Existing suite must finish before retrying failed jobs")
        state.setdefault("history", []).extend(
            j for j in state["jobs"]
            if j["status"] not in {"COMPLETE", "EXCLUDED_BY_MEASURED_RESIDENT_MEMORY_POLICY"}
        )
        state["jobs"] = [j for j in state["jobs"]
                         if j["status"] in {"COMPLETE", "EXCLUDED_BY_MEASURED_RESIDENT_MEMORY_POLICY"}]
        state["status"] = "RUNNING"
        for job in state["jobs"]:
            if job["replicas"] == 1 and job["status"] == "COMPLETE":
                results[job["kind"]] = json.loads((Path(job["path"]) / "profile.json").read_text())

    def run(kind, replicas):
        name = f"{kind}-n{replicas}" + args.suffix
        output = root / "runs" / name
        command = [sys.executable, "-u", "-m", "gsdeploy.profiling.runner",
                   "--spec", str(root / (kind + "-spec.json")),
                   "--samples", str(root / (kinds[kind] + ".jsonl")),
                   "--output", str(output), "--replicas", str(replicas)]
        job = {"kind": kind, "replicas": replicas, "status": "RUNNING", "path": str(output)}
        state["jobs"].append(job)
        save(root / "suite-status.json", state)
        with (root / (name + ".log")).open("w") as log:
            execution = subprocess.run(command, stdout=log, stderr=log)
        profile_path = output / "profile.json"
        result = json.loads(profile_path.read_text()) if profile_path.exists() else {
            "status": "FAILED", "error": f"Runner exited {execution.returncode} before writing profile"
        }
        job["status"] = result["status"]
        job["error"] = result.get("error")
        save(root / "suite-status.json", state)
        return result

    if not args.retry_failed:
        existing = root / "runs" / args.existing_qa / "profile.json"
        print("Waiting for existing QA measurement", flush=True)
        while True:
            result = json.loads(existing.read_text())
            if result["status"] != "RUNNING":
                break
            time.sleep(1)
        results["qa"] = result
        state["jobs"].append({"kind": "qa", "replicas": 1, "status": result["status"], "path": str(existing.parent)})
        save(root / "suite-status.json", state)
    for kind in ["qa", "image", "ocr", "speech"]:
        if kind in results:
            continue
        print("Starting", kind, "replicas 1", flush=True)
        results[kind] = run(kind, 1)
    for kind, result in results.items():
        if result["status"] != "COMPLETE":
            state["jobs"].append({"kind": kind, "replicas": 2, "status": "BLOCKED_BY_SINGLE_REPLICA_FAILURE"})
            continue
        # Conservative deployment policy: every replica reserves its measured
        # warmed resident footprint with the project's existing safety factor.
        resident = result["resident_after_warmup_mib"]
        for replicas in [2, 3]:
            if any(j["kind"] == kind and j["replicas"] == replicas for j in state["jobs"]):
                continue
            required = resident * replicas * result["safe_factor"]
            if required > result["budget_mib"]:
                state["jobs"].append({
                    "kind": kind, "replicas": replicas, "status": "EXCLUDED_BY_MEASURED_RESIDENT_MEMORY_POLICY",
                    "estimated_safe_mib": required, "budget_mib": result["budget_mib"],
                    "note": "Derived conservative domain exclusion, not an observed multi-replica OOM",
                })
                save(root / "suite-status.json", state)
            else:
                print("Starting", kind, "replicas", replicas, flush=True)
                run(kind, replicas)
    state["status"] = "COMPLETE" if all(
        job["status"] in {"COMPLETE", "EXCLUDED_BY_MEASURED_RESIDENT_MEMORY_POLICY"}
        for job in state["jobs"]
    ) else "INCOMPLETE"
    state["finished_epoch"] = time.time()
    save(root / "suite-status.json", state)


if __name__ == "__main__":
    main()
