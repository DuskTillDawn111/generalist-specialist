from dataclasses import dataclass, asdict
import math


def number(value, name, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"Invalid {name}: expected finite {'positive' if positive else 'nonnegative'} number")


@dataclass
class MeasurementConfig:
    deployment: dict[str, int]
    workload_rps: dict[str, float]
    routing_rps: dict[str, dict[str, float]]
    warmup_seconds: float = 10
    measurement_seconds: float = 30
    arrival_process: str = "deterministic_periodic"
    random_seed: int = 42
    routing_source: str = "explicit_config"
    startup_timeout_seconds: float = 300
    drain_timeout_seconds: float = 60
    shutdown_timeout_seconds: float = 10
    max_pending_per_model: int = 10000

    def validate(self, capabilities):
        if not self.deployment or not any(self.deployment.values()):
            raise ValueError("At least one model instance must be deployed")
        for model, count in self.deployment.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("deployment counts must be nonnegative integers")
            if model not in capabilities:
                raise ValueError(f"Unknown model: {model}")
        tasks = set().union(*(set(ts) for ts in capabilities.values()))
        if set(self.workload_rps) - tasks or set(self.routing_rps) - tasks:
            raise ValueError("Unknown task in workload or routing")
        for task in set(self.workload_rps) | set(self.routing_rps):
            offered = self.workload_rps.get(task, 0)
            number(offered, f"workload.{task}")
            route = self.routing_rps.get(task, {})
            for model, rate in route.items():
                number(rate, f"routing.{task}.{model}")
                if model not in capabilities:
                    raise ValueError(f"Unknown routing model {model}")
                if rate > 0 and (self.deployment.get(model, 0) == 0 or task not in capabilities[model]):
                    raise ValueError(f"Illegal routing {task} -> {model}")
            if not math.isclose(math.fsum(route.values()), offered, rel_tol=1e-7, abs_tol=0):
                raise ValueError(f"Routing conservation failed for {task}")
        if not any(rate > 0 for rate in self.workload_rps.values()):
            raise ValueError("Measurement requires a nonzero workload")
        for key in ('warmup_seconds', 'measurement_seconds', 'startup_timeout_seconds',
                    'drain_timeout_seconds', 'shutdown_timeout_seconds'):
            number(getattr(self, key), key, positive=key != 'warmup_seconds')
        if self.arrival_process != 'deterministic_periodic':
            raise ValueError("Only deterministic_periodic arrivals are implemented")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if (isinstance(self.max_pending_per_model, bool) or not isinstance(self.max_pending_per_model, int)
                or self.max_pending_per_model <= 0):
            raise ValueError("max_pending_per_model must be a positive integer")
        if not isinstance(self.routing_source, str) or not self.routing_source:
            raise ValueError("routing_source is required")

    def as_dict(self):
        return asdict(self)
