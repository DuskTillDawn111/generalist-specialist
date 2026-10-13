import os
import queue
import subprocess
import time
from pathlib import Path

import run_pure_qwen
import run_active_single_profile

from nvml_sampler import (
    NvmlSampler,
)


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

run_pure_qwen.WORKER_SCRIPT = (
    ROOT
    / "joint_deploy/qwen_worker_nsys_launcher.py"
)


class ProfilingQwenWorker(
    run_pure_qwen.Worker
):

    def __init__(self):

        super().__init__()

        self.profile_control_done = False

        self.resource_sampler = None

    def _start_profile_control(self):

        if self.profile_control_done:
            return

        super().send({
            "request_id": -2,
            "pool_id":
                "__profile_start__",
            "arrival_index": -2,
            "task_type":
                "__profile_start__",
            "sample": {},
        })

        deadline = (
            time.time()
            + 120.0
        )

        while True:

            remaining = (
                deadline
                - time.time()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting "
                    "for profile control."
                )

            try:
                kind, payload = (
                    self.events.get(
                        timeout=min(
                            remaining,
                            1.0,
                        )
                    )
                )

            except queue.Empty:

                if (
                    self.process.poll()
                    is not None
                ):
                    raise RuntimeError(
                        "Qwen/NSYS exited "
                        "during profile control."
                    )

                continue

            if kind == "ERROR":
                raise RuntimeError(
                    payload
                )

            if kind == "RESULT":

                print(
                    "[PROFILE] control result:",
                    payload,
                    flush=True,
                )

                break

        self.profile_control_done = True

        nvml_out = os.environ.get(
            "RESOURCE_NVML_OUT"
        )

        if nvml_out:

            self.resource_sampler = (
                NvmlSampler(
                    output_base=nvml_out,
                    interval_s=0.1,
                    gpu_index=0,
                )
            )

            self.resource_sampler.start()

            print(
                "[RESOURCE] "
                "NVML 100ms sampling started",
                flush=True,
            )

    def send(
        self,
        request,
    ):

        if (
            request.get(
                "request_id"
            )
            == 0
            and not
            self.profile_control_done
        ):
            self._start_profile_control()

        super().send(
            request
        )

    def stop_resource_sampler(self):

        if (
            self.resource_sampler
            is not None
        ):

            self.resource_sampler.stop()

            self.resource_sampler = None

    def terminate(self):

        self.stop_resource_sampler()

        if (
            self.process is None
            or self.process.poll()
            is not None
        ):
            return

        try:
            self.shutdown()
        except Exception:
            pass

        try:
            self.process.wait(
                timeout=120
            )

            return

        except subprocess.TimeoutExpired:
            pass

        self.process.terminate()

        try:
            self.process.wait(
                timeout=10
            )

        except subprocess.TimeoutExpired:
            self.process.kill()


OriginalCollectQwen = (
    run_active_single_profile
    .collect_qwen
)


def collect_qwen_profiled(
    worker,
    count,
    timeout_s,
):

    try:

        return OriginalCollectQwen(
            worker,
            count,
            timeout_s,
        )

    finally:

        if hasattr(
            worker,
            "stop_resource_sampler",
        ):
            worker.stop_resource_sampler()


run_active_single_profile.QwenWorker = (
    ProfilingQwenWorker
)

run_active_single_profile.collect_qwen = (
    collect_qwen_profiled
)


if __name__ == "__main__":

    run_active_single_profile.main()
