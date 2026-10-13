"""Small synthetic oracle only. Never imported by the GuidedDeploymentSearch path."""
from itertools import combinations

from deployment_search.neighborhood import memory_mib, missing_coverage
from deployment_search.routing import solve_canonical_routing


def exhaustive_baseline(workload, catalog, hardware, evaluator, rho_max=.8):
    if len(catalog) > 5:
        raise ValueError('Exhaustive baseline is restricted to at most five synthetic models')
    names = sorted(catalog)
    best = None
    checked = 0
    feasible = 0
    for size in range(len(names) + 1):
        for subset in combinations(names, size):
            checked += 1
            x = {m: int(m in subset) for m in names}
            if missing_coverage(x, catalog, workload) or memory_mib(x, catalog) > hardware.gpu_memory_mib:
                continue
            canonical = solve_canonical_routing(x, workload, catalog, rho_max)
            if canonical is None:
                continue
            runtime = evaluator.evaluate(x, workload, canonical.routing)
            if runtime.stability_state_v2 != 'steady':
                raise ValueError('This mean-latency oracle requires all feasible objectives steady')
            feasible += 1
            if best is None or runtime.weighted_mean_e2e < best['J_star']:
                best = dict(J_star=runtime.weighted_mean_e2e, deployment=x)
    return dict(best, combinations_checked=checked, runtime_evaluations=feasible)
