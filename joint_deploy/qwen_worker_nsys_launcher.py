import os
from pathlib import Path

ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

PYTHON = (
    "/root/miniconda3/envs/gsdeploy/bin/python"
)

NSYS = "/usr/local/bin/nsys"

WORKER = str(
    ROOT / "joint_deploy/qwen_worker.py"
)

out = os.environ.get(
    "QWEN_NSYS_OUT",
    str(
        ROOT
        / "benchmark/results/characterization/"
          "qwen_worker_profile"
    ),
)

# Tell qwen_worker that this is a profiling run.
os.environ[
    "QWEN_NSYS_CAPTURE"
] = "1"

cmd = [
    NSYS,
    "profile",

    "--trace=cuda,nvtx",

    # Do not collect model loading / warmup.
    "--capture-range=cudaProfilerApi",

    # Once request_id=0 starts capture,
    # continue until worker exits normally.
    "--capture-range-end=none",

    "--sample=none",
    "--cpuctxsw=none",

    "--force-overwrite=true",

    "-o",
    out,

    PYTHON,
    WORKER,
]

os.execv(
    NSYS,
    cmd,
)
