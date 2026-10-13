"""Units: arrival/throughput in req/s, latency in ms, GPU memory in MiB."""

from dataclasses import dataclass
import math

TASKS = ("text", "image", "ocr", "asr", "visual_qa", "lid")
MODEL_NAMES = ("qwen", "distilbert", "resnet50", "paddleocr", "whisper")


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass(frozen=True)
class Workload:
    arrival_rates: dict[str, float]

    def __post_init__(self):
        unknown = set(self.arrival_rates) - set(TASKS)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        for task, rate in self.arrival_rates.items():
            nonnegative(rate, f"arrival_rates.{task}")
        object.__setattr__(self, "arrival_rates", {
            task: float(self.arrival_rates.get(task, 0)) for task in TASKS
        })

    @property
    def active_tasks(self):
        return {task for task, rate in self.arrival_rates.items() if rate > 0}

    @property
    def total_rps(self):
        return sum(self.arrival_rates.values())

    @property
    def key(self):
        return tuple(self.arrival_rates[task] for task in TASKS)


@dataclass(frozen=True)
class Model:
    name: str
    supported_tasks: frozenset[str]
    memory_mib: float

    def __post_init__(self):
        if not self.name or not self.supported_tasks or set(self.supported_tasks) - set(TASKS):
            raise ValueError("Model requires a name and known supported tasks")
        nonnegative(self.memory_mib, "memory_mib")


@dataclass(frozen=True)
class Hardware:
    gpu_memory_mib: float
    reserved_memory_mib: float = 0

    def __post_init__(self):
        nonnegative(self.gpu_memory_mib, "gpu_memory_mib")
        nonnegative(self.reserved_memory_mib, "reserved_memory_mib")
        if self.reserved_memory_mib > self.gpu_memory_mib:
            raise ValueError("Reserved memory exceeds GPU memory")


@dataclass(frozen=True)
class Deployment:
    model_counts: dict[str, int]

    def __post_init__(self):
        for name, count in self.model_counts.items():
            if not name or isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("Deployment counts must be positive integers")
        object.__setattr__(self, "model_counts", dict(sorted(self.model_counts.items())))

    @property
    def key(self):
        return tuple(self.model_counts.items())


@dataclass(frozen=True)
class PerformanceResult:
    mean_latency_ms: float
    p95_latency_ms: float
    throughput_rps: float
    completed_requests: int | None = None
    slo_violation_rate: float | None = None

    def __post_init__(self):
        for name in ("mean_latency_ms", "p95_latency_ms", "throughput_rps"):
            nonnegative(getattr(self, name), name)
        if self.completed_requests is not None:
            if (isinstance(self.completed_requests, bool)
                    or not isinstance(self.completed_requests, int)
                    or self.completed_requests < 0):
                raise ValueError("completed_requests must be a nonnegative integer")
        if self.slo_violation_rate is not None:
            nonnegative(self.slo_violation_rate, "slo_violation_rate")
            if self.slo_violation_rate > 1:
                raise ValueError("slo_violation_rate must be <= 1")
