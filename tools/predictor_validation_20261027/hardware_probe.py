"""Small explicit GPU profiling permission probe, never loads model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def workload() -> None:
    import torch

    torch.set_num_threads(1)
    a = torch.ones((256, 256), device="cuda", dtype=torch.float32)
    b = torch.ones_like(a)
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("hardware_probe_matmul"):
        for _ in range(2):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "output": float(c[0, 0]),
                      "shape": [256, 256], "iterations": 2,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated()}))


def run(output: Path, python: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "receipt.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    source = Path(__file__).resolve()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0")
    target = [python, str(source), "--workload"]
    commands = {
        "nsys": ["/usr/local/bin/nsys", "profile", "--trace=cuda,nvtx", "--sample=none",
                 "--cpuctxsw=none", "--cuda-memory-usage=true", "--force-overwrite=false",
                 "--output", str(output / "tiny-cuda-nvtx"), *target],
        "ncu": ["/opt/nvidia/nsight-compute/2026.1.1/ncu", "--target-processes", "all",
                "--metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct",
                "--launch-count", "1", "--csv", "--log-file", str(output / "tiny-counters.csv"),
                "--export", str(output / "tiny-counters"), *target],
    }
    records = []
    for name, command in commands.items():
        started = time.time()
        proc = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
        stdout = output / f"{name}.stdout.log"
        stderr = output / f"{name}.stderr.log"
        stdout.write_text(proc.stdout)
        stderr.write_text(proc.stderr)
        records.append({"name": name, "command": command, "returncode": proc.returncode,
                        "elapsed_s": time.time() - started,
                        "stdout": str(stdout), "stderr": str(stderr)})
    files = []
    for path in sorted(output.glob("*")):
        if path.is_file():
            files.append({"path": str(path), "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    receipt.write_text(json.dumps({"schema": "gsdeploy.hardware_probe.v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "environment_overrides": {k: env[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
        "commands": records, "files": files}, indent=2) + "\n")
    print(receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", default="/root/autodl-tmp/gsdeploy-assets/envs/compact-standard/bin/python")
    args = parser.parse_args()
    if args.workload:
        workload()
    elif args.output is not None:
        run(args.output.resolve(), args.python)
    else:
        parser.error("--output or --workload required")
