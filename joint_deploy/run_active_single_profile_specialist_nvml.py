import os

import run_static_hybrid
import run_active_single_profile

from nvml_sampler import NvmlSampler


class NvmlSharedWorker(
    run_static_hybrid.Worker
):

    def __init__(
        self,
        name,
        config,
        event_queue,
    ):
        super().__init__(
            name,
            config,
            event_queue,
        )

        self.resource_sampler = None
        self.resource_started = False

    def start_resource_sampler(self):

        if self.resource_started:
            return

        out = os.environ.get(
            "RESOURCE_NVML_OUT"
        )

        if not out:
            raise RuntimeError(
                "RESOURCE_NVML_OUT is not set."
            )

        self.resource_sampler = NvmlSampler(
            output_base=out,
            interval_s=0.1,
            gpu_index=0,
        )

        self.resource_sampler.start()

        self.resource_started = True

        print(
            "[RESOURCE] "
            "NVML 100ms sampling started",
            flush=True,
        )

    def stop_resource_sampler(self):

        if self.resource_sampler is None:
            return

        self.resource_sampler.stop()

        self.resource_sampler = None

        print(
            "[RESOURCE] "
            "NVML sampling stopped",
            flush=True,
        )

    def send(
        self,
        request,
    ):

        # Warmup requests use other IDs.
        # Start immediately before the first
        # formal request.
        if (
            request.get("request_id")
            == 0
            and not self.resource_started
        ):
            self.start_resource_sampler()

        super().send(
            request
        )

    def terminate(self):

        self.stop_resource_sampler()

        super().terminate()


OriginalCollectShared = (
    run_active_single_profile
    .collect_shared
)


def collect_shared_nvml(
    worker_name,
    worker,
    events,
    count,
    timeout_s,
):

    try:
        return OriginalCollectShared(
            worker_name,
            worker,
            events,
            count,
            timeout_s,
        )

    finally:
        if hasattr(
            worker,
            "stop_resource_sampler",
        ):
            worker.stop_resource_sampler()


run_active_single_profile.SharedWorker = (
    NvmlSharedWorker
)

run_active_single_profile.collect_shared = (
    collect_shared_nvml
)


if __name__ == "__main__":
    run_active_single_profile.main()
