"""One-hot per-family counts; O(sum R_m) variables, no count-vector enumeration."""
from dataclasses import dataclass, field, fields
import math
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from .milp_seeder import IntegerSeedResult, SeedResult, SeedSolverError, validate_seed
from .routing_solver import LP_TOLERANCE, validate_rho_max


@dataclass
class DiscreteSeedResult(IntegerSeedResult):
    capacity_mode: str = 'empirical'
    memory_mode: str = 'empirical'
    count_selection: dict = field(default_factory=dict)
    effective_capacity: dict = field(default_factory=dict)
    memory_by_model_mib: dict = field(default_factory=dict)


def solve_discrete_seed(catalog, workload, hardware, rho_max, bounds, policy):
    validate_rho_max(rho_max)
    if not catalog or any(m != p.name for m, p in catalog.items()):
        raise ValueError('Nonempty catalog and matching model names required')
    models, tasks = sorted(catalog), workload.active_tasks
    size = len(models)
    xi = {m: i for i, m in enumerate(models)}
    yi = {m: size + i for i, m in enumerate(models)}
    choices = [(m, n) for m in models for n in range(bounds[m] + 1)]
    zi = {pair: 2 * size + i for i, pair in enumerate(choices)}
    pairs = [(k, m) for k in tasks for m in models if k in catalog[m].supported_tasks]
    ri = {pair: 2 * size + len(choices) + i for i, pair in enumerate(pairs)}
    nvars = 2 * size + len(choices) + len(pairs)
    rows, columns, values, lower, upper = [], [], [], [], []
    def row(entries, lb=-np.inf, ub=np.inf):
        index = len(lower)
        for col, value in entries:
            rows.append(index); columns.append(col); values.append(value)
        lower.append(lb); upper.append(ub)
    for m in models:
        row([(zi[m,n], 1.) for n in range(bounds[m]+1)], lb=1., ub=1.)
        row([(xi[m], 1.)] + [(zi[m,n], -n) for n in range(bounds[m]+1)], lb=0., ub=0.)
        row([(yi[m], 1.), (zi[m,0], 1.)], lb=1., ub=1.)
        row([(yi[m], 1.), (xi[m], -1.)], ub=0.)
        row([(xi[m], 1.), (yi[m], -bounds[m])], ub=0.)
    for k in tasks:
        row([(yi[m], 1.) for m in models if k in catalog[m].supported_tasks], lb=1.)
    memory_entries = [(zi[m,n], policy.memory(m,n)) for m,n in choices]
    row(memory_entries, ub=hardware.gpu_memory_mib)
    for k in tasks:
        row([(ri[k,m], 1. / workload.arrival_rate[k]) for m in models if (k,m) in ri], lb=1., ub=1.)
    for k,m in pairs:
        row([(ri[k,m], 1. / workload.arrival_rate[k]), (yi[m], -1.)], ub=0.)
    for m in models:
        row([(ri[k,m], catalog[m].service_time_ms[k]/1000.) for k in tasks if (k,m) in ri] +
            [(zi[m,n], -rho_max * policy.capacity(m,n)) for n in range(bounds[m]+1)], ub=0.)
    costs = np.zeros(nvars)
    for m,n in choices:
        costs[zi[m,n]] = policy.memory(m,n)
    integrality = np.r_[np.ones(2*size+len(choices)), np.zeros(len(pairs))]
    var_bounds = Bounds(np.zeros(nvars), np.array([bounds[m] for m in models] + [1.]*(size+len(choices)) + [workload.arrival_rate[k] for k,m in pairs]))
    start, calls = time.perf_counter(), 0
    def run(cost):
        nonlocal calls
        calls += 1
        matrix = coo_matrix((values, (rows, columns)), shape=(len(lower), nvars)).tocsc()
        if not np.isfinite(matrix.data).all():
            raise ValueError('Inputs exceed representable MILP range')
        result = milp(c=cost, integrality=integrality, bounds=var_bounds,
            constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)), options={'mip_rel_gap': 0.})
        if result.status == 2:
            return None
        if result.status != 0 or not result.success or result.x is None or not np.isfinite(result.x).all():
            raise SeedSolverError(f'Discrete MILP failed: {result.message}')
        return result
    def witness(result, memory_objective=True):
        if len(result.x) != nvars:
            raise SeedSolverError('Invalid discrete MILP witness dimension')
        x = {m: int(round(result.x[xi[m]])) for m in models}
        selection = {}
        for m in models:
            chosen = {n: int(round(result.x[zi[m,n]])) for n in range(bounds[m]+1)}
            if any(abs(result.x[zi[m,n]]-v) > LP_TOLERANCE or v not in (0,1) for n,v in chosen.items()):
                raise SeedSolverError('Nonbinary count-selection witness')
            if (sum(chosen.values()) != 1 or x[m] != sum(n*v for n,v in chosen.items()) or
                abs(result.x[xi[m]]-x[m]) > LP_TOLERANCE or not 0 <= x[m] <= bounds[m] or
                abs(result.x[yi[m]]-int(x[m]>0)) > LP_TOLERANCE):
                raise SeedSolverError('Count/presence/selection linkage failed')
            selection[m] = chosen
        routing = {(k,m): 0. for k in workload.arrival_rate for m in models}
        for k,m in pairs:
            limit = workload.arrival_rate[k] * int(x[m]>0)
            value = float(result.x[ri[k,m]])
            if value < -LP_TOLERANCE or value > limit+LP_TOLERANCE:
                raise SeedSolverError('Routing/presence linkage failed')
            routing[k,m] = min(limit, max(0., value))
        demand = {(k,m): value * catalog[m].service_time_ms.get(k,0)/1000. for (k,m), value in routing.items()}
        total = {m: math.fsum(demand[k,m] for k in workload.arrival_rate) for m in models}
        effective = {m: policy.capacity(m,x[m]) for m in models}
        utilization = {m: total[m]/effective[m] if x[m] else 0. for m in models}
        slack = {m: rho_max * effective[m] - total[m] for m in models}
        memory = policy.deployment_memory(x)
        if memory_objective and not math.isclose(float(result.fun), memory, rel_tol=1e-10, abs_tol=LP_TOLERANCE):
            raise SeedSolverError('Discrete memory objective mismatch')
        seed = DiscreteSeedResult(True, x, routing, utilization, slack, demand, memory,
            'minimum_discrete_memory_then_replica_count_feasible_seed', (time.perf_counter()-start)*1000,
            solver_message=str(result.message), num_variables=nvars, num_constraints=len(lower),
            presence={m: int(n>0) for m,n in x.items()}, replica_bounds=dict(bounds),
            capacity_mode=policy.capacity_mode, memory_mode=policy.memory_mode, count_selection=selection,
            effective_capacity=effective, memory_by_model_mib={m: policy.memory(m,x[m]) for m in models})
        errors = validate_seed(seed, catalog, workload, hardware, rho_max, bounds, planning_policy=policy)
        if errors:
            raise SeedSolverError('; '.join(errors))
        return seed
    primary = run(costs)
    if primary is None:
        return DiscreteSeedResult(False, None, None, {}, {}, {}, None, 'infeasible',
            (time.perf_counter()-start)*1000, num_variables=nvars, num_constraints=len(lower),
            milp_call_count=calls, replica_bounds=dict(bounds), capacity_mode=policy.capacity_mode, memory_mode=policy.memory_mode)
    first = witness(primary)
    row(memory_entries, lb=first.objective_value, ub=first.objective_value)
    secondary_cost = np.r_[np.ones(size), np.zeros(nvars-size)]
    secondary = run(secondary_cost)
    if secondary is None:
        raise SeedSolverError('Replica tie solve lost verified memory optimum')
    seed = witness(secondary, False)
    if not math.isclose(seed.objective_value, first.objective_value, rel_tol=0., abs_tol=LP_TOLERANCE):
        raise SeedSolverError('Secondary solve changed memory optimum')
    if not math.isclose(float(secondary.fun), sum(seed.deployment.values()), rel_tol=0., abs_tol=LP_TOLERANCE):
        raise SeedSolverError('Replica tie objective mismatch')
    seed.milp_call_count = calls
    seed.secondary_objective_value = sum(seed.deployment.values())
    return seed
