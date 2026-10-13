"""Runtime boundary. A future GPU adapter must reuse joint_deploy/measurement."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math

from .neighborhood import deployment_key

STABILITY = {'unstable': 0, 'recoverable_congested': 1, 'steady': 2}


@dataclass(frozen=True)
class RuntimeEvaluation:
    stability_state_v2: str
    weighted_mean_e2e: float | None
    weighted_p95_e2e: float | None
    throughput: float
    completion_ratio: float
    queue_metrics: dict = field(default_factory=dict)
    raw_result_path: str | None = None

    def __post_init__(self):
        if self.stability_state_v2 not in STABILITY:
            raise ValueError('Unknown stability_state_v2')
        for name in ('weighted_mean_e2e', 'weighted_p95_e2e', 'throughput', 'completion_ratio'):
            value = getattr(self, name)
            if value is None and name.startswith('weighted_') and self.stability_state_v2 != 'steady':
                continue
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid runtime metric: {name}')
        if self.completion_ratio > 1:
            raise ValueError('completion_ratio exceeds 1')


def weighted_task_latency(workload, task_latency):
    """Arrival-rate weighted task mean or task P95 (not pooled system P95)."""
    total = sum(workload.arrival_rate.values())
    if total == 0:
        return 0.
    for k in workload.active_tasks:
        if k not in task_latency or not math.isfinite(task_latency[k]) or task_latency[k] < 0:
            raise ValueError(f'Missing/invalid active-task latency: {k}')
    return sum(workload.arrival_rate[k] * task_latency[k] for k in workload.active_tasks) / total


class RuntimeEvaluator(ABC):
    @abstractmethod
    def evaluate(self, deployment, workload, routing) -> RuntimeEvaluation:
        """Latency unit is ms; throughput and feasible routing are req/s."""


class SyntheticRuntimeEvaluator(RuntimeEvaluator):
    """Explicit offline function or table, independent of planning LP objectives."""
    def __init__(self, objective, integer_mode=False):
        self.objective = objective
        self.integer_mode = integer_mode
        self.calls = []

    def evaluate(self, deployment, workload, routing):
        from deployment_core.replicas import normalize_counts, replica_key
        normalize_counts(deployment)
        key = replica_key(deployment) if self.integer_mode or any(n > 1 for n in deployment.values()) else deployment_key(deployment)
        self.calls.append(key)
        value = (self.objective(deployment, workload, routing) if callable(self.objective)
                 else self.objective[key])
        if not isinstance(value, RuntimeEvaluation):
            raise TypeError('Synthetic objective must return RuntimeEvaluation')
        return value
