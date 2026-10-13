import json
import math
import statistics
import threading
import time
from pathlib import Path

import pynvml


def percentile(xs, p):
    if not xs:
        return None

    xs = sorted(xs)

    if len(xs) == 1:
        return xs[0]

    k = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))

    if lo == hi:
        return xs[lo]

    w = k - lo

    return (
        xs[lo] * (1.0 - w)
        + xs[hi] * w
    )


class NvmlSampler:

    def __init__(
        self,
        output_base,
        interval_s=0.1,
        gpu_index=0,
    ):
        self.output_base = Path(
            output_base
        )

        self.interval_s = interval_s
        self.gpu_index = gpu_index

        self.samples = []

        self.stop_event = (
            threading.Event()
        )

        self.thread = None
        self.handle = None

        self.start_mono_ns = None
        self.end_mono_ns = None

    def start(self):

        pynvml.nvmlInit()

        self.handle = (
            pynvml.nvmlDeviceGetHandleByIndex(
                self.gpu_index
            )
        )

        self.start_mono_ns = (
            time.monotonic_ns()
        )

        # Sample once immediately.
        self._sample_once()

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )

        self.thread.start()

    def _loop(self):

        while not self.stop_event.wait(
            self.interval_s
        ):
            self._sample_once()

    def _sample_once(self):

        now_mono = time.monotonic_ns()

        util = (
            pynvml
            .nvmlDeviceGetUtilizationRates(
                self.handle
            )
        )

        mem = (
            pynvml
            .nvmlDeviceGetMemoryInfo(
                self.handle
            )
        )

        try:
            power_w = (
                pynvml
                .nvmlDeviceGetPowerUsage(
                    self.handle
                )
                / 1000.0
            )
        except Exception:
            power_w = None

        try:
            sm_clock = (
                pynvml
                .nvmlDeviceGetClockInfo(
                    self.handle,
                    pynvml.NVML_CLOCK_SM,
                )
            )
        except Exception:
            sm_clock = None

        try:
            mem_clock = (
                pynvml
                .nvmlDeviceGetClockInfo(
                    self.handle,
                    pynvml.NVML_CLOCK_MEM,
                )
            )
        except Exception:
            mem_clock = None

        try:
            temp = (
                pynvml
                .nvmlDeviceGetTemperature(
                    self.handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
            )
        except Exception:
            temp = None

        self.samples.append({
            "wall_time_ns":
                time.time_ns(),

            "monotonic_ns":
                now_mono,

            "elapsed_ms":
                (
                    now_mono
                    - self.start_mono_ns
                )
                / 1e6,

            "gpu_util_pct":
                int(util.gpu),

            "mem_util_pct":
                int(util.memory),

            "power_w":
                power_w,

            "sm_clock_mhz":
                sm_clock,

            "mem_clock_mhz":
                mem_clock,

            "vram_used_mib":
                mem.used
                / 1024.0
                / 1024.0,

            "vram_total_mib":
                mem.total
                / 1024.0
                / 1024.0,

            "temperature_c":
                temp,
        })

    def stop(self):

        if self.start_mono_ns is None:
            return

        self.end_mono_ns = (
            time.monotonic_ns()
        )

        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(
                timeout=5.0
            )

        # Final sample near end boundary.
        try:
            self._sample_once()
        except Exception:
            pass

        self._write_outputs()

        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def _summary_one(
        self,
        key,
    ):

        xs = [
            float(row[key])
            for row in self.samples
            if row.get(key)
            is not None
        ]

        if not xs:
            return None

        return {
            "mean":
                statistics.mean(xs),

            "p50":
                percentile(xs, 50),

            "p95":
                percentile(xs, 95),

            "min":
                min(xs),

            "max":
                max(xs),
        }

    def _energy_j(self):

        rows = [
            row
            for row in self.samples
            if row.get("power_w")
            is not None
        ]

        if len(rows) < 2:
            return None

        energy = 0.0

        for a, b in zip(
            rows,
            rows[1:],
        ):
            dt_s = (
                b["monotonic_ns"]
                - a["monotonic_ns"]
            ) / 1e9

            avg_power = (
                a["power_w"]
                + b["power_w"]
            ) / 2.0

            energy += (
                avg_power
                * dt_s
            )

        return energy

    def _write_outputs(self):

        self.output_base.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        trace_path = Path(
            str(self.output_base)
            + ".nvml.jsonl"
        )

        summary_path = Path(
            str(self.output_base)
            + ".nvml_summary.json"
        )

        with trace_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            for row in self.samples:

                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        duration_s = (
            (
                self.end_mono_ns
                - self.start_mono_ns
            )
            / 1e9
            if (
                self.start_mono_ns
                is not None
                and self.end_mono_ns
                is not None
            )
            else None
        )

        summary = {
            "interval_ms":
                self.interval_s * 1000.0,

            "samples":
                len(self.samples),

            "duration_s":
                duration_s,

            "estimated_energy_j":
                self._energy_j(),

            "gpu_util_pct":
                self._summary_one(
                    "gpu_util_pct"
                ),

            "mem_util_pct":
                self._summary_one(
                    "mem_util_pct"
                ),

            "power_w":
                self._summary_one(
                    "power_w"
                ),

            "sm_clock_mhz":
                self._summary_one(
                    "sm_clock_mhz"
                ),

            "mem_clock_mhz":
                self._summary_one(
                    "mem_clock_mhz"
                ),

            "vram_used_mib":
                self._summary_one(
                    "vram_used_mib"
                ),

            "temperature_c":
                self._summary_one(
                    "temperature_c"
                ),
        }

        summary_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "[RESOURCE] NVML saved:",
            trace_path,
            flush=True,
        )

        print(
            "[RESOURCE] summary saved:",
            summary_path,
            flush=True,
        )
