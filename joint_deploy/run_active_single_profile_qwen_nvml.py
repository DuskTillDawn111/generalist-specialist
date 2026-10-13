import os

import run_active_single_profile
import run_pure_qwen

from nvml_sampler import NvmlSampler


class NvmlQwenWorker(
    run_pure_qwen.Worker
):

    def __init__(self):
        super().__init__()

        self.resource_sampler = None
        self.resource_started = False

    def start_resource_sampler(self):

        if self.resource_started:
            return

        out = os.environ.get(
            "RESOURCE_NVML_OUT"
        )

        if not out:
            return

        self.resource_sampler = (
            NvmlSampler(
                output_base=out,
                interval_s=0.1,
                gpu_index=0,
            )
        )

        self.resource_sampler.start()
        self.resource_started = True

        print(
            "[RESOURCE] "
            "NVML-only 100ms sampling started",
            flush=True,
        )

    def send(
        self,
        request,
    ):

        # Warmup uses request_id=-1.
        # Start immediately before formal request 0.
        if (
            request.get("request_id")
            == 0
            and not self.resource_started
        ):
            self.start_resource_sampler()

        super().send(
            request
        )

    def stop_resource_sampler(self):

        if self.resource_sampler is not None:

            self.resource_sampler.stop()

            self.resource_sampler = None


OriginalCollectQwen = (
    run_active_single_profile.collect_qwen
)


def collect_qwen_nvml(
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
    NvmlQwenWorker
)

run_active_single_profile.collect_qwen = (
    collect_qwen_nvml
)


if __name__ == "__main__":

    run_active_single_profile.main()
