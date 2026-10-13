"""Parameter-free congestion surrogate, in seconds per offered request."""
import math

def evaluate_surrogate(deployment, routing, workload, catalog, calibration):
    rates = workload.arrival_rate if hasattr(workload, 'arrival_rate') else workload
    demand = {m: 0.0 for m, n in deployment.items() if n}
    for (k, m), value in routing.items():
        if not math.isfinite(value) or value < -1e-9:
            raise ValueError('Invalid routing rate')
        if abs(value) <= 1e-12:
            continue
        if m not in demand or k not in catalog[m].supported_tasks:
            raise ValueError('Absent or unadmitted routing path')
        demand[m] += max(0., value) * catalog[m].service_time_ms[k] / 1000.
    for k, rate in rates.items():
        if not math.isclose(math.fsum(v for (t, m), v in routing.items() if t == k), rate,
                            rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError('Routing conservation failure')
    if any(k not in rates and abs(v)>1e-12 for (k,m),v in routing.items()):
        raise ValueError('Unknown task')
    capacity = {m: calibration.capacity_factors[m][deployment[m]] for m in demand}
    rho = {m: demand[m]/capacity[m] for m in demand}
    total = math.fsum(rates.values())
    defined = all(v < 1. for v in rho.values())
    return {'D': demand, 'rho': rho, 'z': max(rho.values(), default=0.),
            'J_seconds': math.fsum(demand[m]/(1.-rho[m]) for m in demand)/total
                         if total and defined else 0. if not total else None,
            'domain_valid': defined, 'Lambda': total}
