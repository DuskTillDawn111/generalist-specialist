"""Serial execution/resume of a sealed independent confirmation packet."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsdeploy.confirmation import validate_confirmation_packet, verify_confirmation_run
from gsdeploy.static_study import read


def publish(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Native GPU confirmation requires explicit --execute")
    packet = read(args.packet)
    validation = validate_confirmation_packet(packet)
    if not validation["valid"]:
        raise ValueError("Invalid confirmation packet: " + repr(validation["changed"]))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = args.output / "queue-progress.json"
        state = {"schema": "gsdeploy.confirmation_queue.v1", "packet": str(args.packet.resolve()),
                 "confirmation_sha256": packet["confirmation_sha256"], "gpu_started": True,
                 "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "status": "RUNNING", "jobs": []}
        for job in packet["jobs"]:
            stem = job["labels"][0] + "-seed" + str(job["seed"]) + "-" + job["id"][:12]
            output = args.output / stem
            raw_path = output / "run.json"
            row = {"job_id": job["id"], "seed": job["seed"], "labels": job["labels"],
                   "candidate_key": job["candidate_key"], "path": str(raw_path.resolve())}
            state["current"] = row
            publish(progress, state)
            try:
                if not raw_path.exists():
                    if output.exists():
                        raise FileExistsError("Incomplete attempt requires review: " + str(output))
                    command = [sys.executable, "-u", str(ROOT / "scripts/confirm_static_study.py"),
                               "run", "--packet", str(args.packet.resolve()), "--job", job["id"],
                               "--output-root", str(args.output.resolve()), "--execute"]
                    # A new log revision preserves failed execution logs too.
                    revision = 1
                    log = args.output / (stem + "-launch1.log")
                    while log.exists():
                        revision += 1
                        log = args.output / (stem + "-launch" + str(revision) + ".log")
                    with log.open("x") as stream:
                        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                        row.update({"pid": process.pid, "log": str(log.resolve())})
                        while process.poll() is None:
                            publish(progress, state)
                            print(json.dumps({"running": stem, "pid": process.pid}), flush=True)
                            time.sleep(20)
                    row["exit_code"] = process.returncode
                    if process.returncode:
                        raise RuntimeError("Confirmation execution failed: " + str(log))
                raw, utility = verify_confirmation_run(packet, job, raw_path)
                row.update({"status": raw["status"], "eligible": utility["eligible"], "U": utility["U"]})
                state["jobs"].append(row)
                print(json.dumps({"verified": row}), flush=True)
            except Exception as exc:
                state["status"] = "STOPPED_FOR_REVIEW"
                row["error"] = str(exc)
                publish(progress, state)
                raise
        state["status"] = "ALL_FROZEN_CONFIRMATIONS_VERIFIED"
        state.pop("current", None)
        publish(progress, state)
        print(json.dumps({"status": state["status"], "completed": len(state["jobs"])}), flush=True)


if __name__ == "__main__":
    main()
