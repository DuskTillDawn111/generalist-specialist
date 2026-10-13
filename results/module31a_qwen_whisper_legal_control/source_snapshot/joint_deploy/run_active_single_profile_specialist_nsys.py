import os
import queue
import subprocess
import time
from pathlib import Path

import run_static_hybrid
import run_active_single_profile


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

LAUNCHER = (
    ROOT
    / "joint_deploy/"
      "specialist_worker_nsys_launcher.py"
)


class ProfilingSharedWorker(
    run_static_hybrid.Worker
):

    def __init__(
        self,
        name,
        config,
        event_queue,
    ):

        self.real_config = dict(
            config
        )

        launcher_config = {
            # Launch launcher with the
            # same Python environment as
            # the actual specialist.
            "python":
                config["python"],

            "script":
                LAUNCHER,
        }

        super().__init__(
            name,
            launcher_config,
            event_queue,
        )

        self.profile_control_done = False

    def start(self):

        out = os.environ.get(
            "SPECIALIST_NSYS_OUT"
        )

        if not out:
            raise RuntimeError(
                "SPECIALIST_NSYS_OUT "
                "is not set."
            )

        keys = [
            "SPECIALIST_NSYS_REAL_PYTHON",
            "SPECIALIST_NSYS_REAL_SCRIPT",
        ]

        old = {
            k: os.environ.get(k)
            for k in keys
        }

        os.environ[
            "SPECIALIST_NSYS_REAL_PYTHON"
        ] = str(
            self.real_config["python"]
        )

        os.environ[
            "SPECIALIST_NSYS_REAL_SCRIPT"
        ] = str(
            self.real_config["script"]
        )

        try:
            super().start()

        finally:
            # Worker.start() already copied
            # the environment into Popen.
            for k, value in old.items():
                if value is None:
                    os.environ.pop(
                        k,
                        None,
                    )
                else:
                    os.environ[k] = value

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
                    "for specialist "
                    "profile control."
                )

            try:
                (
                    worker_name,
                    kind,
                    payload,
                ) = (
                    self.event_queue.get(
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
                        f"{self.name}/NSYS "
                        "exited during "
                        "profile control."
                    )

                continue

            if worker_name != self.name:
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

    def send(
        self,
        request,
    ):

        if (
            request.get("request_id")
            == 0
            and not
            self.profile_control_done
        ):
            self._start_profile_control()

        super().send(
            request
        )

    def terminate(self):

        if (
            self.process is None
            or self.process.poll()
            is not None
        ):
            return

        # Important:
        # let worker exit normally so nsys
        # can finalize the .nsys-rep.
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


run_active_single_profile.SharedWorker = (
    ProfilingSharedWorker
)


if __name__ == "__main__":
    run_active_single_profile.main()
