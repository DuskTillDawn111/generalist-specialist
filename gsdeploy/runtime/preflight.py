"""Read-only native execution checks; no worker launch or model loading."""

import os
from pathlib import Path
import shutil
import subprocess

from .workload import SamplePool
from ..evidence import digest, verify_admission


def preflight(config, winner, samples, registry, admission=None, require_gpu=False):
    config.validate_plan(winner["deployment"], winner["routing"])
    pool = SamplePool(samples, list(config.workload), 42)
    issues = []
    devices = set()
    for model in winner["deployment"]:
        if model not in registry:
            issues.append(f"No worker registered for {model}")
            continue
        entry = registry[model]
        command = entry.get("command", [])
        env = entry.get("env", {})
        python = shutil.which(command[0]) if command else None
        if not python or not os.access(python, os.X_OK):
            issues.append(f"Worker interpreter unavailable: {model}")
        cwd = Path(entry.get("cwd", Path(__file__).resolve().parents[2]))
        for key in ("GS_MODEL_PATH", "GS_PROMPT_FILE"):
            value = env.get(key)
            path = Path(value) if value else None
            if path is not None and not path.is_absolute():
                path = cwd / path
            if path is None or not path.exists():
                issues.append(f"Worker {key} unavailable: {model}")
            elif key == "GS_MODEL_PATH" and not Path(value).is_absolute():
                issues.append(
                    f"GS_MODEL_PATH must be absolute for fingerprinting: {model}"
                )
        devices.add(env.get("CUDA_VISIBLE_DEVICES", "0"))
    for task, items in pool.samples.items():
        for item in items:
            sample = item["sample"]
            if task != "text":
                asset = sample.get("input")
                if not isinstance(asset, str) or not Path(asset).is_file():
                    issues.append(f"Sample asset unavailable: {item['pool_id']}")
    if len(devices) != 1 or any(
        not isinstance(d, str) or not d or "," in d or d == "-1" for d in devices
    ):
        issues.append("All workers must select the same single CUDA device")
    gpu = None
    device = next(iter(devices)) if len(devices) == 1 else "0"
    if require_gpu and not issues:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    device,
                    "--query-gpu=uuid,name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=5,
                check=True,
            )
            rows = [
                line.split(",") for line in result.stdout.splitlines() if line.strip()
            ]
            if len(rows) != 1 or len(rows[0]) != 4:
                raise ValueError("Expected one GPU")
            uuid, name, driver, memory = (v.strip() for v in rows[0])
            gpu = {
                "uuid": uuid,
                "name": name,
                "driver": driver,
                "memory_mib": float(memory),
            }
            if gpu["memory_mib"] < config.gpu_memory_mib:
                issues.append("Configured memory budget exceeds GPU memory")
            processes = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    device,
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=5,
                check=True,
            )
            if processes.stdout.strip():
                issues.append("GPU has existing compute processes")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            issues.append(f"GPU unavailable: {exc}")
    evidence = {"verified": False, "reason": "PREFLIGHT_FAILED", "pairs": []}
    if not issues:
        try:
            evidence = verify_admission(config, winner, registry, admission)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"Admission evidence invalid: {exc}")
    return {
        "ready": not issues,
        "scope": "native inputs and optional GPU availability; model imports/inference untested",
        "issues": issues,
        "sample_pool_sha256": digest(samples),
        "task_sample_counts": {
            task: len(items) for task, items in pool.samples.items()
        },
        "gpu_checked": require_gpu,
        "gpu": gpu,
        "device": device,
        "admission_evidence": evidence,
    }
