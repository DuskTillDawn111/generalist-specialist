"""Serial, resumable execution of a frozen static-study GPU packet.

The frozen runtime (not the current development tree) collects every run.
Completed outputs are never overwritten. A failed/invalid run stops the queue
for review; evidence validity is separate from subprocess success.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def summary(path):
    raw = read(path)
    utility = raw.get("evaluation", {}).get("utility", {})
    return {"status": raw.get("status"), "eligible": utility.get("eligible", False),
            "U": utility.get("U"), "diagnostic_U": utility.get("diagnostic_U"),
            "energy_verified": utility.get("energy_verified"),
            "quality_identity_verified": utility.get("quality_identity_verified"),
            "errors": raw.get("errors", []), "output": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-first", type=Path)
    parser.add_argument("--existing-pid", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required for native GPU runs")
    packet = read(args.packet)
    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / "queue-progress.json"
    state = {"schema": "gsdeploy.serial_gpu_queue.v1", "packet": str(args.packet.resolve()),
             "runner": str(args.runner.resolve()), "gpu_started": True,
             "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "status": "RUNNING", "jobs": []}
    write(progress_path, state)
    for index, job in enumerate(packet["jobs"]):
        output = args.output / (job["labels"][0] + "-attempt1")
        if index == 0 and args.existing_first:
            output = args.existing_first
            while not (output / "run.json").is_file():
                try:
                    import os
                    os.kill(args.existing_pid, 0)
                except (ProcessLookupError, TypeError):
                    raise RuntimeError("Existing first experiment exited without run.json")
                print(json.dumps({"waiting_existing_job": job["labels"][0]}, ensure_ascii=False), flush=True)
                time.sleep(20)
        row = {"job_id": job["id"], "labels": job["labels"], "output": str(output)}
        state["current"] = row
        write(progress_path, state)
        if not (output / "run.json").is_file():
            if output.exists():
                raise FileExistsError("Incomplete output requires a new attempt directory: " + str(output))
            log = args.output / (job["labels"][0] + "-attempt1.log")
            command = [sys.executable, "-u", str(args.runner.resolve()), "run", "--packet",
                       str(args.packet.resolve()), "--job", job["id"], "--output",
                       str(output.resolve()), "--execute"]
            print(json.dumps({"starting": row, "command": command}, ensure_ascii=False), flush=True)
            with log.open("x") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                row["pid"] = process.pid
                while process.poll() is None:
                    write(progress_path, state)
                    time.sleep(20)
                    print(json.dumps({"running": job["labels"][0], "pid": process.pid}, ensure_ascii=False), flush=True)
                row["exit_code"] = process.returncode
            if process.returncode or not (output / "run.json").is_file():
                row["status"] = "EXECUTION_ERROR"
                state["jobs"].append(row)
                state["status"] = "STOPPED_FOR_REVIEW"
                write(progress_path, state)
                raise RuntimeError("Native execution failed; see " + str(log))
        row.update(summary(output / "run.json"))
        state["jobs"].append(row)
        print(json.dumps({"finished": row}, ensure_ascii=False), flush=True)
        if row["status"] != "completed" or not row["eligible"]:
            state["status"] = "STOPPED_FOR_REVIEW"
            write(progress_path, state)
            raise RuntimeError("Experiment did not yield valid complete measured utility")
    state["status"] = "CALIBRATION_COMPLETE"
    state.pop("current", None)
    write(progress_path, state)
    print(json.dumps({"status": state["status"], "completed": len(state["jobs"])}), flush=True)


if __name__ == "__main__":
    main()
