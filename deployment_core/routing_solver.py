"""Zero-objective LP using standalone service demands, not response-time prediction."""

import math
import numpy as np
from scipy.optimize import linprog

from .deployment import Deployment, RoutingResult
from .models import ModelProfile
from .workload import Workload, validate_number

LP_TOLERANCE = 1e-9
ROUTING_REL_TOLERANCE = 1e-7


class RoutingSolverError(RuntimeError):
    """Numerical/solver failure is not proof of capacity infeasibility."""


def validate_rho_max(rho_max):
    validate_number(rho_max, "rho_max", positive=True)
    if rho_max > 1:
        raise ValueError("rho_max must be in (0, 1]")


def solve_capacity_lp(deployment: Deployment, model_catalog: dict[str, ModelProfile],
                      workload: Workload, rho_max=0.8, capacity_mode='linear', calibration=None) -> RoutingResult | None:
    validate_rho_max(rho_max)
    models = deployment.models
    if set(models) - set(model_catalog):
        raise ValueError("Deployment contains models absent from the catalog")
    from .replica_calibration import ReplicaPlanningPolicy
    from .replicas import normalize_counts
    policy = ReplicaPlanningPolicy(model_catalog, capacity_mode, 'additive', calibration)
    if capacity_mode == 'empirical':
        normalize_counts(deployment.counts, model_catalog, policy.calibration.evidence_bounds)
    effective = {m: policy.capacity(m, deployment.counts[m]) for m in models}
    tasks = workload.active_tasks
    if not tasks:
        return RoutingResult({}, {model: 0.0 for model in models})
    pairs = [(task, model) for task in tasks for model in models
             if task in model_catalog[model].supported_tasks]
    if any(not any(k == task for k, _ in pairs) for task in tasks):
        return None

    # Variables are r[k,m] in req/s. No variables for unsupported/absent models.
    a_eq = np.zeros((len(tasks), len(pairs)))
    a_ub = np.zeros((len(models), len(pairs)))
    task_index = {task: i for i, task in enumerate(tasks)}
    model_index = {model: i for i, model in enumerate(models)}
    for j, (task, model) in enumerate(pairs):
        a_eq[task_index[task], j] = 1.0
        a_ub[model_index[model], j] = model_catalog[model].service_time_ms[task] / 1000.0
    rates = np.array([workload.arrival_rate[task] for task in tasks])
    bounds = [(0.0, workload.arrival_rate[task]) for task, _ in pairs]
    result = linprog(
        c=np.zeros(len(pairs)), A_eq=a_eq, b_eq=rates,
        A_ub=a_ub, b_ub=np.array([rho_max * effective[m] for m in models]), bounds=bounds, method="highs",
        options={"primal_feasibility_tolerance": LP_TOLERANCE,
                 "dual_feasibility_tolerance": LP_TOLERANCE},
    )
    if result.status == 2:
        return None
    if not result.success:
        raise RoutingSolverError(f"HiGHS status {result.status}: {result.message}")
    if result.x is None or not np.all(np.isfinite(result.x)):
        raise RoutingSolverError("HiGHS returned a non-finite routing")

    # Clip only numerical bound noise, then independently verify the witness.
    routing = {(task, model): float(np.clip(value, 0.0, workload.arrival_rate[task]))
               for (task, model), value in zip(pairs, result.x)}
    for task in tasks:
        routed = math.fsum(value for (k, _), value in routing.items() if k == task)
        if not math.isclose(routed, workload.arrival_rate[task],
                            rel_tol=ROUTING_REL_TOLERANCE, abs_tol=0.0):
            raise RoutingSolverError(f"Routing conservation failed for {task}: {routed}")
    utilization = {
        model: math.fsum(value * model_catalog[model].service_time_ms[task] / 1000.0
                         for (task, m), value in routing.items() if m == model)
        for model in models
    }
    if any(value > rho_max * effective[m] + LP_TOLERANCE for m, value in utilization.items()):
        raise RoutingSolverError("Returned routing exceeds the utilization limit")
    return RoutingResult(routing, utilization)
