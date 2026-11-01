"""NVML whole-board counters and time-aligned raw samples."""

import threading
import time

from gsdeploy.energy import integrate_power


class BoardMonitor:
    def __init__(self, device=0, interval=0.05):
        import pynvml as nv

        nv.nvmlInit()
        self.nv = nv
        self.handle = nv.nvmlDeviceGetHandleByIndex(device)
        self.interval = interval
        self.samples = []
        self.errors = []
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = None

    def sample(self):
        nv, h = self.nv, self.handle
        try:
            start = time.perf_counter_ns()
            watts = nv.nvmlDeviceGetPowerUsage(h) / 1000
            energy = None
            try:
                energy = nv.nvmlDeviceGetTotalEnergyConsumption(h) / 1000
            except nv.NVMLError:
                pass
            memory = nv.nvmlDeviceGetMemoryInfo(h)
            processes = nv.nvmlDeviceGetComputeRunningProcesses(h)
            value = {
                "time_ns": (start + time.perf_counter_ns()) // 2,
                "watts": watts,
                "counter_joules": energy,
                "used_mib": memory.used / 1024**2,
                "processes": [
                    {"pid": p.pid, "used_mib": p.usedGpuMemory / 1024**2}
                    for p in processes
                ],
            }
            # Diagnostic operating state; unavailable optional fields do not
            # replace or invalidate an otherwise covered board-power integral.
            for key, read in {
                'temperature_c': lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                'sm_clock_mhz': lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                'memory_clock_mhz': lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                'power_limit_watts': lambda: nv.nvmlDeviceGetPowerManagementLimit(h) / 1000,
            }.items():
                try:
                    value[key] = read()
                except nv.NVMLError:
                    value[key] = None
            with self.lock:
                self.samples.append(value)
            return value
        except Exception as exc:
            self.errors.append(type(exc).__name__)
            return None

    def start(self):
        self.sample()

        def loop():
            while not self.stop.wait(self.interval):
                self.sample()

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.sample()

    def window(self, start, end):
        # Boundaries are explicit NVML samples, not inferred request timestamps.
        counter = None
        if start["counter_joules"] is not None and end["counter_joules"] is not None:
            delta = end["counter_joules"] - start["counter_joules"]
            if delta >= 0:
                counter = delta
        with self.lock:
            samples = list(self.samples)
        integral = integrate_power(samples, start["time_ns"], end["time_ns"])
        return {
            "scope": "whole_gpu_board",
            "kind": "measured",
            "start_ns": start["time_ns"],
            "end_ns": end["time_ns"],
            "counter_joules": counter,
            "power_integral_joules": integral,
            # Keep the cumulative counter for diagnostics. On the measured card
            # it systematically differs from board power integration, especially
            # at startup/idle. The existing utility contract uses board power.
            "joules": integral,
            "method": "nvml_board_power_integral",
            "coverage_valid": not self.errors and integral is not None,
        }
