"""CPU-only protocol backend for functional tests; no model or GPU results."""

import json
import os
import time
from gsdeploy.runtime.protocol import run_worker


def handler(request):
    sample = request["sample"]
    time.sleep(sample.get("mock_delay_seconds", 0))
    if sample.get("mock_error"):
        raise RuntimeError("Requested mock failure")
    return {"prediction": sample.get("label", "mock"), "backend": "mock"}


if __name__ == "__main__":
    run_worker(
        "mock",
        handler,
        {"tasks": json.loads(os.environ["GS_TASKS"]), "backend": "mock"},
    )
