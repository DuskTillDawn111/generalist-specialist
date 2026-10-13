"""Planning bottlenecks derived exclusively from canonical LP demand."""
from dataclasses import dataclass


@dataclass
class BottleneckFeedback:
    max_pressure: float
    bottleneck_models: list[str]
    model_pressure: dict[str, float]
    capacity_slack: dict[str, float]
    task_contributions: dict[tuple[str, str], float]
    pressure_tasks: list[str]


def bottleneck_feedback(canonical, deployment, tolerance=1e-8):
    ordered = sorted((m for m, n in deployment.items() if n),
                     key=lambda m: (-canonical.model_pressure[m], m))
    bottlenecks = [m for m in ordered
                   if canonical.model_pressure[m] >= canonical.max_pressure - tolerance]
    demand = {}
    for (k, m), value in canonical.task_contributions.items():
        if m in bottlenecks:
            demand[k] = demand.get(k, 0.) + value
    tasks = sorted((k for k in demand if demand[k] > 0), key=lambda k: (-demand[k], k))
    return BottleneckFeedback(canonical.max_pressure, bottlenecks,
                              dict(canonical.model_pressure), dict(canonical.capacity_slack),
                              dict(canonical.task_contributions), tasks)
