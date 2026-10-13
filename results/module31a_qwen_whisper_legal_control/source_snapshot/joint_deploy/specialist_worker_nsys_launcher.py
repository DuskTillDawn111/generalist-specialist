import os

NSYS = "/usr/local/bin/nsys"

real_python = os.environ[
    "SPECIALIST_NSYS_REAL_PYTHON"
]

real_script = os.environ[
    "SPECIALIST_NSYS_REAL_SCRIPT"
]

out = os.environ[
    "SPECIALIST_NSYS_OUT"
]

# Enables __profile_start__ in worker_protocol.
os.environ[
    "SPECIALIST_NSYS_CAPTURE"
] = "1"

cmd = [
    NSYS,
    "profile",

    "--trace=cuda,nvtx",

    # Ignore loading and warmup.
    "--capture-range=cudaProfilerApi",

    # Keep capturing formal requests
    # until worker exits normally.
    "--capture-range-end=none",

    "--sample=none",
    "--cpuctxsw=none",

    "--force-overwrite=true",

    "-o",
    out,

    real_python,
    real_script,
]

os.execv(
    NSYS,
    cmd,
)
