"""Canonical min-max standalone pressure LP; no performance predictor dependency."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linprog

from deployment_core.routing_solver import (
    LP_TOLERANCE, ROUTING_REL_TOLERANCE, RoutingSolverError, validate_rho_max,
)


@dataclass
class CanonicalRouting:
    routing: dict[tuple[str, str], float]
    max_pressure: float
    model_pressure: dict[str, float]
    capacity_slack: dict[str, float]
    task_contributions: dict[tuple[str, str], float]


def solve_canonical_routing(deployment, workload, model_catalog, rho_max=.8, replica_bounds=None,
                            capacity_mode='linear', calibration=None):
    """Return a verified optimum, or None only for proven infeasibility.

    r is req/s; s0 is converted from milliseconds to seconds. Absent/unsupported
    pairs are fixed to zero. Sorted variables make repeated solves reproducible;
    the min-max formulation does not promise a unique routing witness.
    """
    return _solve_routing(deployment, workload, model_catalog, rho_max, relaxed=False, replica_bounds=replica_bounds,
                          capacity_mode=capacity_mode, calibration=calibration)


def solve_pressure_relaxation(deployment, workload, model_catalog, rho_max=.8, replica_bounds=None,
                              capacity_mode='linear', calibration=None):
    """Diagnostic LP with no pressure cap; never a feasible serving witness."""
    return _solve_routing(deployment, workload, model_catalog, rho_max, relaxed=True, replica_bounds=replica_bounds,
                          capacity_mode=capacity_mode, calibration=calibration)


def _solve_routing(deployment, workload, model_catalog, rho_max, relaxed, replica_bounds=None,
                   capacity_mode='linear', calibration=None):
    validate_rho_max(rho_max)
    from deployment_core.replicas import normalize_counts
    from deployment_core.replica_calibration import ReplicaPlanningPolicy
    policy = ReplicaPlanningPolicy(model_catalog, capacity_mode, 'additive', calibration)
    if capacity_mode == 'empirical':
        evidence = policy.calibration.evidence_bounds
        if replica_bounds is None:
            replica_bounds = dict(evidence)
        elif any(replica_bounds.get(m, evidence[m]+1) > evidence[m] for m in model_catalog):
            raise ValueError('Routing count bound exceeds scaling evidence')
        # Exact R=1 reduction retains old LP matrices, routing ties and schema.
        if all(n == 1 for n in replica_bounds.values()):
            return _solve_routing(deployment, workload, model_catalog, rho_max, relaxed)
    bounds = {m: 1 for m in model_catalog} if replica_bounds is None else replica_bounds
    deployment = normalize_counts(deployment, model_catalog, bounds)
    models = sorted(model_catalog)
    tasks = workload.active_tasks
    pairs = [(k, m) for k in tasks for m in models
             if deployment.get(m, 0) and k in model_catalog[m].supported_tasks]
    if any(not any(k == t for k, _ in pairs) for t in tasks):
        return None
    zidx = len(pairs)
    eq = np.zeros((len(tasks), zidx + 1))
    ub = np.zeros((len(models), zidx + 1))
    ti = {k: i for i, k in enumerate(tasks)}
    mi = {m: i for i, m in enumerate(models)}
    for j, (k, m) in enumerate(pairs):
        # Normalize conservation to preserve tiny positive arrival rates.
        eq[ti[k], j] = 1 / workload.arrival_rate[k]
        ub[mi[m], j] = model_catalog[m].service_time_ms[k] / 1000
    # Fixed x keeps D_m <= z*x_m linear; z is average pressure per replica.
    effective = {m: policy.capacity(m, deployment[m]) for m in models}
    ub[:, zidx] = -1 if replica_bounds is None else [-effective[m] for m in models]
    result = linprog(
        np.r_[np.zeros(zidx), 1.],
        A_eq=eq if tasks else None, b_eq=np.ones(len(tasks)) if tasks else None,
        A_ub=ub, b_ub=np.zeros(len(models)),
        bounds=[(0., workload.arrival_rate[k]) for k, _ in pairs] + [(0., None if relaxed else rho_max)],
        method='highs', options={'primal_feasibility_tolerance': LP_TOLERANCE,
                                 'dual_feasibility_tolerance': LP_TOLERANCE},
    )
    if result.status == 2:
        return None
    if not result.success or result.x is None or not np.isfinite(result.x).all():
        raise RoutingSolverError(f'Canonical LP failed: {result.message}')
    routing = {(k, m): 0. for k in workload.arrival_rate for m in models}
    for (k, m), value in zip(pairs, result.x[:zidx]):
        bound = workload.arrival_rate[k]
        if value < -LP_TOLERANCE or value > bound + LP_TOLERANCE:
            raise RoutingSolverError('Canonical routing outside linkage bounds')
        routing[k, m] = float(np.clip(value, 0., bound))
    for k in tasks:
        if not math.isclose(math.fsum(routing[k, m] for m in models), workload.arrival_rate[k],
                            rel_tol=ROUTING_REL_TOLERANCE, abs_tol=0.):
            raise RoutingSolverError(f'Canonical conservation failed: {k}')
    contributions = {(k, m): r * model_catalog[m].service_time_ms.get(k, 0.) / 1000
                     for (k, m), r in routing.items()}
    demand = {m: math.fsum(contributions[k, m] for k in workload.arrival_rate) for m in models}
    pressure = {m: demand[m] / effective[m] if deployment[m] else 0. for m in models}
    z = float(result.x[zidx])
    if ((not relaxed and any(v > rho_max + LP_TOLERANCE for v in pressure.values()))
            or not math.isclose(max(pressure.values(), default=0.), z, abs_tol=1e-8, rel_tol=1e-7)):
        raise RoutingSolverError('Canonical pressure/objective verification failed')
    return CanonicalRouting(routing, z, pressure,
                            {m: rho_max * effective[m] - demand[m] for m in models},
                            contributions)
