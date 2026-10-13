"""Exhaustive feasibility enumeration: test oracle / small-scale baseline only.

Not used by the production/research search path (MilpDeploymentSeeder).
"""

from dataclasses import dataclass, field
from itertools import combinations

from .deployment import Deployment, FeasibleDeployment
from .models import Hardware, ModelProfile
from .routing_solver import solve_capacity_lp, validate_rho_max
from .workload import Workload


def enumerate_deployments(model_catalog):
    names = sorted(model_catalog)
    for count in range(1, len(names) + 1):
        for subset in combinations(names, count):
            yield Deployment(subset)


def missing_coverage(deployment, model_catalog, workload):
    covered = set().union(*(model_catalog[name].supported_tasks for name in deployment.models))
    return tuple(task for task in workload.active_tasks if task not in covered)


def deployment_memory_mib(deployment, model_catalog):
    return sum(model_catalog[name].memory_mib * n for name, n in deployment.counts.items())


@dataclass
class RejectedDeployment:
    deployment: Deployment
    stage: str
    reason: str


@dataclass
class ScreeningReport:
    feasible_deployments: list[FeasibleDeployment] = field(default_factory=list)
    rejected_deployments: list[RejectedDeployment] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: {
        "all_combinations": 0,
        "coverage_rejected": 0, "coverage_feasible": 0,
        "memory_rejected": 0, "memory_feasible": 0,
        "capacity_rejected": 0, "capacity_feasible": 0,
    })


def screen_deployments(model_catalog: dict[str, ModelProfile], workload: Workload,
                       hardware: Hardware, rho_max=0.8) -> ScreeningReport:
    validate_rho_max(rho_max)
    if not model_catalog or any(name != model.name for name, model in model_catalog.items()):
        raise ValueError("Catalog must be nonempty and keys must match model names")
    report = ScreeningReport()
    for deployment in enumerate_deployments(model_catalog):
        report.counts["all_combinations"] += 1
        missing = missing_coverage(deployment, model_catalog, workload)
        if missing:
            report.counts["coverage_rejected"] += 1
            report.rejected_deployments.append(RejectedDeployment(
                deployment, "coverage", "coverage missing: " + ", ".join(missing)))
            continue
        report.counts["coverage_feasible"] += 1
        memory = deployment_memory_mib(deployment, model_catalog)
        if memory > hardware.gpu_memory_mib:
            report.counts["memory_rejected"] += 1
            report.rejected_deployments.append(RejectedDeployment(
                deployment, "memory", f"memory {memory:g} MiB exceeds {hardware.gpu_memory_mib:g} MiB"))
            continue
        report.counts["memory_feasible"] += 1
        routing = solve_capacity_lp(deployment, model_catalog, workload, rho_max)
        if routing is None:
            report.counts["capacity_rejected"] += 1
            report.rejected_deployments.append(RejectedDeployment(
                deployment, "capacity", "capacity infeasible under standalone service demands"))
            continue
        report.counts["capacity_feasible"] += 1
        report.feasible_deployments.append(FeasibleDeployment(deployment, memory, routing))
    return report


def generate_feasible_deployments(model_catalog, workload, hardware, rho_max=0.8):
    """Return every feasible deployment with one routing witness, without ranking."""
    return screen_deployments(model_catalog, workload, hardware, rho_max).feasible_deployments
