import json
import os
import subprocess
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

PYTHON = (
    ROOT
    / "envs/specialist_env/bin/python"
)

WORKER = (
    ROOT
    / "joint_deploy/resnet_worker.py"
)

MANIFEST = (
    ROOT
    / "benchmark/manifests/"
    "image_classification.jsonl"
)


samples = []

with open(
    MANIFEST,
    "r",
    encoding="utf-8"
) as f:
    for line in f:
        line = line.strip()

        if line:
            samples.append(
                json.loads(
                    line
                )
            )


print(
    "Samples:",
    len(samples)
)


env = os.environ.copy()

env[
    "PYTHONUNBUFFERED"
] = "1"


proc = subprocess.Popen(
    [
        str(PYTHON),
        str(WORKER),
    ],
    cwd=str(ROOT),
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=env,
)


# ---------------------------------------
# Wait for READY
# ---------------------------------------

while True:
    line = proc.stdout.readline()

    if not line:
        raise RuntimeError(
            "Worker exited before READY"
        )

    line = line.rstrip()

    print(line)

    if line.startswith(
        "READY|"
    ):
        break


# ---------------------------------------
# Submit all 200 samples
# ---------------------------------------

for i, sample in enumerate(
    samples
):

    request = {
        "request_id":
            i,

        "pool_id":
            "validate:"
            + sample["id"],

        "arrival_index":
            i,

        "task_type":
            "image",

        "sample":
            sample,
    }

    proc.stdin.write(
        json.dumps(
            request,
            ensure_ascii=False
        )
        + "\n"
    )

proc.stdin.flush()


# ---------------------------------------
# Collect
# ---------------------------------------

results = []
errors = []


while (
    len(results)
    + len(errors)
    < len(samples)
):

    line = proc.stdout.readline()

    if not line:
        raise RuntimeError(
            "Worker exited early"
        )

    line = line.rstrip()

    if line.startswith(
        "RESULT|"
    ):

        payload = json.loads(
            line.split(
                "|",
                1
            )[1]
        )

        results.append(
            payload
        )

    elif line.startswith(
        "ERROR|"
    ):

        payload = json.loads(
            line.split(
                "|",
                1
            )[1]
        )

        errors.append(
            payload
        )


correct = sum(
    bool(
        row[
            "correct"
        ]
    )
    for row in results
)


accuracy = (
    correct
    / len(results)
    if results
    else 0.0
)


print()
print(
    "===== RESNET WORKER VALIDATION ====="
)

print(
    "Results :",
    len(results)
)

print(
    "Errors  :",
    len(errors)
)

print(
    "Correct :",
    correct
)

print(
    "Accuracy:",
    accuracy
)


proc.stdin.write(
    json.dumps(
        {
            "op":
                "shutdown"
        }
    )
    + "\n"
)

proc.stdin.flush()

proc.wait(
    timeout=10
)
