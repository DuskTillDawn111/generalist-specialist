"""Tiny exhaustive validation oracle, intentionally separate from production search."""
from itertools import product
import math
from deployment_core.replicas import normalize_counts, validate_bounds
from deployment_search.neighborhood import memory_mib, missing_coverage
from deployment_search.routing import solve_canonical_routing


def small_integer_oracle(workload, catalog, hardware, bounds, evaluator, rho_max=.8):
    validate_bounds(bounds, catalog)
    if not 1 <= len(catalog) <= 3 or any(n > 2 for n in bounds.values()):
        raise ValueError('Oracle restricted to <=3 types and R_m<=2 (at most 27 vectors)')
    names = sorted(catalog)
    checked, feasible, best, best_deployment = 0, 0, math.inf, None
    for values in product(*(range(bounds[m] + 1) for m in names)):
        checked += 1
        deployment = dict(zip(names, values))
        if memory_mib(deployment, catalog) > hardware.gpu_memory_mib or missing_coverage(deployment, catalog, workload):
            continue
        canonical = solve_canonical_routing(deployment, workload, catalog, rho_max, replica_bounds=bounds)
        if canonical is None:
            continue
        feasible += 1
        objective = evaluator.evaluate(deployment, workload, canonical.routing).weighted_mean_e2e
        if objective < best:
            best, best_deployment = objective, deployment
    return {'checked_vectors': checked, 'feasible_vectors': feasible, 'J_star': best if best_deployment is not None else None,
            'optimal_deployment': best_deployment, 'scope': 'validation oracle only; never called by IntegerGuidedDeploymentSearch'}
