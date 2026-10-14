"""Optional process-owned GPU memory sampling; never blocks request dispatch."""

import subprocess
import threading
import time


class MemoryMonitor:
    def __init__(self, runtime, budget, enabled=True, device="0"):
        self.runtime, self.budget, self.enabled = runtime, budget, enabled
        self.device = device
        self.samples = []
        self.latest = None
        self.failed = False
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        if not self.enabled:
            return None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    self.device,
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            owned = {w.process.pid for w in self.runtime.workers.values() if w.process}
            rows = [
                line.split(",") for line in result.stdout.splitlines() if line.strip()
            ]
            memory = {int(pid.strip()): float(value.strip()) for pid, value in rows}
            value = {
                "owned_used_mib": sum(v for p, v in memory.items() if p in owned),
                "foreign_pids": sorted(p for p in memory if p not in owned),
                "unix_ns": time.time_ns(),
            }
            if owned and not owned.issubset(memory):
                self.failed = True
                return None
            self.latest = value
            self.samples.append(value)
            return value
        except (OSError, ValueError, subprocess.SubprocessError):
            self.failed = True
            return None

    def start(self):
        def loop():
            while not self.stop.wait(0.5):
                self.sample()

        if self.enabled:
            self.thread = threading.Thread(target=loop, daemon=True)
            self.thread.start()

    def check(self):
        value = self.latest
        if value and (value["owned_used_mib"] > self.budget or value["foreign_pids"]):
            raise RuntimeError(
                "GPU memory budget exceeded or foreign GPU processes observed"
            )

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)


class PowerMonitor:
    """Whole-board power; disabled entirely for CPU/mock execution."""

    def __init__(self, enabled=True, device="0"):
        self.enabled, self.device = enabled, device
        self.samples, self.errors = [], []
        self.failed = False
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        if not self.enabled:
            return
        try:
            start = time.perf_counter_ns()
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    self.device,
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            end = time.perf_counter_ns()
            watts = float(result.stdout.strip())
            self.samples.append({"time_ns": (start + end) // 2, "watts": watts})
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.failed = True
            self.errors.append(str(exc))

    def start(self):
        if self.enabled:

            def loop():
                while not self.stop.wait(0.25):
                    self.sample()

            self.thread = threading.Thread(target=loop, daemon=True)
            self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.sample()
