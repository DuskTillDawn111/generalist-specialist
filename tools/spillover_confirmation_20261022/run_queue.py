#!/usr/bin/env python3
"""Serial queue for the explicit frozen spillover confirmation CLI.

Without --execute this only checks the packet and bound CLI (CPU only).
Execution creates immutable attempts/logs; --resume revalidates every existing
raw run through that CLI's strict report. Incomplete attempts stop for review;
the queue never silently retries, overwrites, or selects among repeated runs.

CPU preflight:
  python run_queue.py --confirmation-source-root FROZEN_ROOT --packet PACKET
Native queue (only the orchestrator should invoke this): add
  --output NEW_QUEUE_DIRECTORY --execute
Resume the same directory: add --resume. Each invocation writes a new immutable
launch directory and reimports existing evidence before proceeding.

CPU protocol verification uses tests.test_spillover_confirmation fixtures with
the transfer gate/evaluation explicitly mocked, substitutes native collection
with write_native, and exercises new execution, strict-report resume, raw SHA
tampering and incomplete-attempt rejection. This is not GPU evidence.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

SCHEMA = "gsdeploy.spillover_confirmation_queue.v1"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def publish(path, value):
    """Replace progress atomically; never apply this to raw evidence/logs."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".queue-progress-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_cli(root, packet):
    cli = root / "scripts" / "spillover_confirmation.py"
    bound = {row["path"]: row["sha256"] for row in packet["bindings"]}
    paths = sorted({*root.joinpath("gsdeploy").rglob("*.py"), *root.joinpath("scripts").rglob("*.py")})
    if not cli.is_file() or not (root / "gsdeploy" / "spillover_confirmation.py").is_file():
        raise ValueError("Explicit confirmation source must contain its CLI and protocol module")
    for path in paths:
        row = receipt(path)
        if bound.get(row["path"]) != row["sha256"]:
            raise ValueError("Unbound or changed confirmation source: " + str(path))
    return cli


def cpu_cli(cli, arguments, log=None):
    command = [sys.executable, "-u", str(cli), *map(str, arguments)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if log is not None:
        with Path(log).open("x") as stream:
            stream.write(completed.stdout)
            stream.write(completed.stderr)
    if completed.returncode:
        raise RuntimeError("Frozen CLI failed (%s): %s" % (arguments[0], completed.stderr[-4000:]))
    try:
        return json.loads(completed.stdout)
    except ValueError as exc:
        raise ValueError("Frozen CLI did not return JSON: " + str(arguments[0])) from exc


def validate(cli, packet_path, packet, log=None):
    result = cpu_cli(cli, ["validate", "--packet", packet_path], log)
    if result.get("valid") is not True or result.get("jobs") != len(packet["jobs"]):
        raise ValueError("Full frozen packet validation failed: " + repr(result))
    return result


def strict_report(cli, packet_path, packet, run_paths, output, log, expected_jobs):
    if output.exists():
        raise FileExistsError(output)
    result = cpu_cli(cli, ["report", "--packet", packet_path, "--runs", *run_paths, "--output", output], log)
    report = read(output)
    expected_jobs = set(expected_jobs)
    missing = sorted({job["id"] for job in packet["jobs"]} - expected_jobs)
    rows = report.get("runs", [])
    if (report.get("schema") != "gsdeploy.spillover_confirmation_report.v1"
            or report.get("packet") != receipt(packet_path)
            or report.get("completed_jobs") != len(expected_jobs)
            or report.get("planned_jobs") != len(packet["jobs"])
            or report.get("missing_job_ids") != missing
            or report.get("status") != ("INCOMPLETE_CONFIRMATION" if missing else "ALL_DECLARED_JOBS_VERIFIED")
            or len(rows) != len(expected_jobs) or {row["job_id"] for row in rows} != expected_jobs
            or {row["path"] for row in rows} != {str(Path(path).resolve()) for path in run_paths}
            or report.get("training_forbidden") is not True
            or result.get("completed_jobs") != report["completed_jobs"]):
        raise ValueError("Strict report roster/packet differs from requested runs")
    for row in rows:
        if {k: row[k] for k in ("path", "sha256")} != receipt(row["path"]):
            raise ValueError("Raw run changed during strict report")
        sidecar = row.get("collection_receipt", {})
        if sidecar != receipt(Path(row["path"]).parent / "spillover-collection-receipt.json"):
            raise ValueError("Collection receipt changed during strict report")
    return report


def native(cli, packet_path, job, attempt, log, progress, state):
    command = [sys.executable, "-u", str(cli), "run", "--packet", str(packet_path),
               "--job", job["id"], "--output", str(attempt), "--execute"]
    if attempt.exists() or log.exists():
        raise FileExistsError("Attempt/log already exists")
    process = None
    with log.open("x") as stream:
        try:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state["gpu_started"] = True
            state["current"].update(pid=process.pid, started_utc=utc(), log=str(log), attempt=str(attempt))
            publish(progress, state)
            while process.poll() is None:
                state["updated_utc"] = utc()
                publish(progress, state)
                time.sleep(5)
            state["current"]["exit_code"] = process.returncode
            if process.returncode:
                raise RuntimeError("Native job failed; inspect " + str(log))
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


def existing_run(directory):
    """Accept exactly one complete attempt; never rank or combine retries."""
    if not directory.exists():
        return None
    attempts = sorted(directory.glob("attempt-*"))
    raw = sorted(directory.glob("attempt-*/run.json"))
    directories = [path for path in attempts if path.is_dir()]
    logs = [path for path in attempts if path.is_file() and path.suffix == ".log"]
    if len(raw) != 1 or len(directories) != 1 or len(logs) != 1:
        raise ValueError("Incomplete or repeated attempt requires review: " + str(directory))
    if logs[0].stem != directories[0].name:
        raise ValueError("Attempt/log roster mismatch: " + str(directory))
    return raw[0]


def execute(args, packet, cli):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = args.output / "queue-progress.json"
        packet_receipt = receipt(args.packet)
        cli_receipt = receipt(cli)
        previous = read(progress) if progress.exists() else None
        if previous is not None:
            if not args.resume:
                raise FileExistsError("Queue exists; explicit --resume required")
            if previous.get("packet") != packet_receipt or previous.get("confirmation_cli") != cli_receipt:
                raise ValueError("Resume packet or frozen CLI changed")
        elif any(path.name != ".queue.lock" for path in args.output.iterdir()):
            raise ValueError("Output contains files without a matching queue-progress receipt")
        launch = args.output / "launches" / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
        launch.mkdir(parents=True, exist_ok=False)
        state = dict(schema=SCHEMA, packet=packet_receipt, confirmation_sha256=packet["confirmation_sha256"],
            confirmation_cli=cli_receipt, queue_source=receipt(__file__), started_utc=utc(),
            launch=str(launch), previous_progress_sha256=receipt(progress)["sha256"] if previous else None,
            status="VALIDATING", gpu_started=False, jobs=[], planned_jobs=len(packet["jobs"]))
        publish(progress, state)
        try:
            state["validation"] = validate(cli, args.packet, packet, launch / "validate.log")
            state["status"] = "RUNNING"
            for job in packet["jobs"]:
                if receipt(args.packet) != packet_receipt or receipt(cli) != cli_receipt:
                    raise ValueError("Packet/CLI changed during queue")
                job_directory = args.output / "jobs" / job["id"]
                row = dict(job_id=job["id"], labels=job["labels"], seed=job["seed"], status="PENDING")
                state["current"] = row
                publish(progress, state)
                raw = existing_run(job_directory)
                if raw is not None and not args.resume:
                    raise ValueError("Existing raw run requires explicit --resume")
                reused = raw is not None
                if raw is None:
                    job_directory.mkdir(parents=True, exist_ok=False)
                    attempt = job_directory / "attempt-0001"
                    native(cli, args.packet, job, attempt, job_directory / "attempt-0001.log", progress, state)
                    raw = attempt / "run.json"
                # Both new and resumed evidence must pass the exact frozen
                # report importer before a job can become VERIFIED.
                partial = strict_report(cli, args.packet, packet, [raw],
                    launch / (job["id"] + "-strict-report.json"), launch / (job["id"] + "-strict-report.log"), [job["id"]])
                result = partial["runs"][0]
                row.update(status="VERIFIED", resumed=reused, run=receipt(raw),
                    strict_report=receipt(launch / (job["id"] + "-strict-report.json")), utility=result["utility"],
                    verified_utc=utc())
                state["jobs"].append(row)
                publish(progress, state)
                print(json.dumps(dict(job_id=job["id"], status="VERIFIED", resumed=reused)), flush=True)
            state.pop("current", None)
            final_path = launch / "final-report.json"
            final = strict_report(cli, args.packet, packet, [Path(row["run"]["path"]) for row in state["jobs"]],
                final_path, launch / "final-report.log", [job["id"] for job in packet["jobs"]])
            if final.get("status") != "ALL_DECLARED_JOBS_VERIFIED" or final.get("missing_job_ids") != []:
                raise ValueError("Final report is incomplete")
            state.update(status="ALL_DECLARED_JOBS_VERIFIED", final_report=receipt(final_path), completed_utc=utc())
            publish(progress, state)
            print(json.dumps(dict(status=state["status"], completed_jobs=len(state["jobs"]), report=str(final_path))))
            return state
        except BaseException as error:
            state.update(status="STOPPED", error=type(error).__name__ + ": " + str(error), stopped_utc=utc())
            publish(progress, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-source-root", required=True, type=Path,
                        help="Explicit frozen confirmation-source-v1 root; no implicit workspace imports")
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.packet = args.packet.resolve()
    args.confirmation_source_root = args.confirmation_source_root.resolve()
    if args.output is not None:
        args.output = args.output.resolve()
    if args.execute and args.output is None:
        parser.error("--execute requires --output")
    packet = read(args.packet)
    if (packet.get("schema") != "gsdeploy.spillover_confirmation.v1" or not packet.get("jobs")
            or len({job["id"] for job in packet["jobs"]}) != len(packet["jobs"])
            or any(not isinstance(job["id"], str) or len(job["id"]) != 64
                   or any(c not in "0123456789abcdef" for c in job["id"]) for job in packet["jobs"])):
        raise ValueError("Expected a nonempty spillover packet with unique SHA256 job identities")
    cli = source_cli(args.confirmation_source_root, packet)
    if not args.execute:
        result = validate(cli, args.packet, packet)
        print(json.dumps(dict(status="CPU_VALIDATED_GPU_NOT_STARTED", validation=result,
                              packet=receipt(args.packet), confirmation_cli=receipt(cli), gpu_started=False)))
        return
    def interrupted(signum, frame):
        raise InterruptedError("Queue interrupted by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    execute(args, packet, cli)


if __name__ == "__main__":
    main()
