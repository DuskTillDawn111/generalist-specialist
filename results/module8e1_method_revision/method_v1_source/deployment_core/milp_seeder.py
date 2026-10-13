"""Joint deployment/routing MILP initialization, independent of exhaustive baselines."""

from dataclasses import dataclass, field, fields
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .models import Hardware, ModelProfile
from .routing_solver import LP_TOLERANCE, ROUTING_REL_TOLERANCE, validate_rho_max
from .workload import Workload


class SeedSolverError(RuntimeError):
    """Solver failure or invalid witness; never interpreted as proven infeasibility."""


@dataclass
class SeedResult:
    feasible: bool
    deployment: dict[str, int] | None
    routing_rps: dict[tuple[str, str], float] | None
    utilization: dict[str, float]
    capacity_slack: dict[str, float]
    service_demand_by_task_model: dict[tuple[str, str], float]
    objective_value: float | None
    solver_status: str
    solver_time_ms: float
    solver_message: str = ''
    num_variables: int = 0
    num_constraints: int = 0


@dataclass
class IntegerSeedResult(SeedResult):
    """Serialized integer witness metadata; legacy SeedResult schema is retained."""
    milp_call_count: int = 0
    presence: dict[str, int] = field(default_factory=dict)
    replica_bounds: dict[str, int] = field(default_factory=dict)
    secondary_objective_value: int | None = None


def validate_seed(result, catalog, workload, hardware, rho_max, replica_bounds=None, planning_policy=None):
    """Recompute all constraints and feedback from the returned witness, without solving."""
    errors = []
    x, routing = result.deployment, result.routing_rps
    if not result.feasible or x is None or routing is None:
        return ['No feasible witness to validate']
    from .replicas import normalize_counts
    bounds = {m: 1 for m in catalog} if replica_bounds is None else replica_bounds
    try:
        normalize_counts(x, catalog, bounds)
    except ValueError:
        return ['Invalid deployment counts or replica bounds']
    if set(x) != set(catalog):
        return ['Missing deployment counts']
    expected = {(k, m) for k in workload.arrival_rate for m in catalog}
    if set(routing) != expected or set(result.service_demand_by_task_model) != expected:
        return ['Missing or unknown task/model pairs']
    if set(result.utilization) != set(catalog) or set(result.capacity_slack) != set(catalog):
        return ['Missing or unknown feedback models']
    if any(not math.isfinite(v) or v < 0 for v in routing.values()):
        return ['Invalid routing rate']
    memory = (planning_policy.deployment_memory(x) if planning_policy is not None else
              math.fsum(catalog[m].memory_mib * n for m, n in x.items()))
    if memory > hardware.gpu_memory_mib + LP_TOLERANCE:
        errors.append('Memory exceeded')
    if result.objective_value is None or not math.isclose(memory, result.objective_value, rel_tol=0, abs_tol=LP_TOLERANCE):
        errors.append('Memory objective inconsistent')
    for k, rate in workload.arrival_rate.items():
        if rate > 0 and sum(int(x[m] > 0) for m in catalog if k in catalog[m].supported_tasks) < 1:
            errors.append(f'Coverage: {k}')
        if not math.isclose(math.fsum(routing[k,m] for m in catalog), rate, rel_tol=ROUTING_REL_TOLERANCE, abs_tol=0):
            errors.append(f'Conservation: {k}')
        for m, model in catalog.items():
            supported = k in model.supported_tasks
            bound = rate * int(x[m] > 0) if supported else 0
            value = routing[k,m]
            # Zero workload / unsupported / undeployed pairs must remain exactly zero.
            if (bound == 0 and value != 0) or value > bound + LP_TOLERANCE:
                errors.append(f'Capability/deployment linkage: {k}/{m}')
            demand = value * model.service_time_ms[k] / 1000 if supported else 0.
            stored = result.service_demand_by_task_model[k,m]
            if not math.isfinite(stored) or not math.isclose(demand, stored, rel_tol=0, abs_tol=LP_TOLERANCE):
                errors.append(f'Demand contribution: {k}/{m}')
    for m, model in catalog.items():
        rho = math.fsum(routing[k,m] * model.service_time_ms[k] / 1000 for k in model.supported_tasks)
        effective = planning_policy.capacity(m, x[m]) if planning_policy is not None else x[m]
        if rho > rho_max*effective + LP_TOLERANCE:
            errors.append(f'Capacity: {m}')
        for label, stored, expected_value in (
                ('Utilization', result.utilization[m], rho / effective if x[m] else 0.),
                ('Slack', result.capacity_slack[m], rho_max*effective-rho)):
            if not math.isfinite(stored) or not math.isclose(stored, expected_value, rel_tol=0, abs_tol=LP_TOLERANCE):
                errors.append(f'{label}: {m}')
    return errors


class MilpDeploymentSeeder:
    """Feasibility seed. Explicit integer mode minimizes memory, then replica count.

    Legacy calls without bounds retain Module 3's binary solve exactly. Supply
    replica_bounds or binary_mode=False for integer allocation. No serving code.
    """
    def solve(self, model_catalog: dict[str, ModelProfile], workload: Workload,
              hardware: Hardware, rho_max=0.8, replica_bounds=None, binary_mode=None,
              capacity_mode='linear', memory_mode=None, calibration=None) -> SeedResult:
        from .replicas import derive_replica_bounds, validate_bounds
        if capacity_mode not in ('linear', 'empirical') or memory_mode not in (None, 'additive', 'empirical'):
            raise ValueError('Unknown capacity/memory planning mode')
        if capacity_mode == 'empirical' or memory_mode == 'empirical':
            from .replica_calibration import ReplicaPlanningPolicy
            from .empirical_milp import solve_discrete_seed
            policy = ReplicaPlanningPolicy(model_catalog, capacity_mode, memory_mode, calibration)
            selected, _ = policy.bounds(hardware, replica_bounds, binary_mode=binary_mode is True)
            if all(n == 1 for n in selected.values()):
                return self._solve_binary(model_catalog, workload, hardware, rho_max)
            return solve_discrete_seed(model_catalog, workload, hardware, rho_max, selected, policy)
        if replica_bounds is None and binary_mode is not False:
            return self._solve_binary(model_catalog, workload, hardware, rho_max)
        if replica_bounds is None:
            replica_bounds, _ = derive_replica_bounds(model_catalog, hardware)
        validate_bounds(replica_bounds, model_catalog)
        if binary_mode is True and any(n > 1 for n in replica_bounds.values()):
            raise ValueError('binary_mode conflicts with replica upper bounds')
        # Exact legacy reduction avoids numerical/tie differences in R_m=1 mode.
        if all(n == 1 for n in replica_bounds.values()):
            return self._solve_binary(model_catalog, workload, hardware, rho_max)
        return self._solve_integer(model_catalog, workload, hardware, rho_max, replica_bounds)

    def _solve_binary(self, model_catalog, workload, hardware, rho_max):
        validate_rho_max(rho_max)
        if not model_catalog or any(k != p.name for k,p in model_catalog.items()):
            raise ValueError('Catalog must be nonempty and keys must match model names')
        models = sorted(model_catalog)
        tasks = workload.active_tasks
        pairs = [(k,m) for k in tasks for m in models if k in model_catalog[m].supported_tasks]
        nmodels, nvars = len(models), len(models) + len(pairs)
        xi = {m:i for i,m in enumerate(models)}
        ri = {pair:nmodels+i for i,pair in enumerate(pairs)}
        row_ids, col_ids, values, lower, upper = [], [], [], [], []

        def row(entries, lb=-np.inf, ub=np.inf):
            index = len(lower)
            for j,v in entries:
                row_ids.append(index); col_ids.append(j); values.append(v)
            lower.append(lb); upper.append(ub)

        # Coverage remains explicit, including unsupported-task infeasibility.
        for k in tasks:
            row([(xi[m],1.) for m in models if k in model_catalog[m].supported_tasks], lb=1.)
        row([(xi[m],model_catalog[m].memory_mib) for m in models], ub=hardware.gpu_memory_mib)
        for k in tasks:
            # Divide the equality by lambda for robust small-rate conservation;
            # r variables themselves are still req/s, not normalized fractions.
            row([(ri[k,m],1./workload.arrival_rate[k]) for m in models if (k,m) in ri], lb=1., ub=1.)
        for k,m in pairs:
            row([(ri[k,m],1./workload.arrival_rate[k]),(xi[m],-1.)], ub=0.)
        for m in models:
            row([(ri[k,m],model_catalog[m].service_time_ms[k]/1000.) for k in tasks if (k,m) in ri]
                + [(xi[m],-rho_max)], ub=0.)
        matrix = coo_matrix((values,(row_ids,col_ids)),shape=(len(lower),nvars)).tocsc()
        objective = np.array([model_catalog[m].memory_mib for m in models]+[0.]*len(pairs))
        integrality = np.array([1]*nmodels+[0]*len(pairs))
        bounds = Bounds(np.zeros(nvars), np.array([1.]*nmodels+[workload.arrival_rate[k] for k,m in pairs]))
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError('Input rates exceed numerically representable MILP range')
        start = time.perf_counter()
        result = milp(c=objective, integrality=integrality, bounds=bounds,
                      constraints=LinearConstraint(matrix,np.array(lower),np.array(upper)),
                      options={'mip_rel_gap':0.0})
        elapsed = (time.perf_counter()-start)*1000
        meta = dict(solver_time_ms=elapsed,solver_message=str(result.message),
                    num_variables=nvars,num_constraints=len(lower))
        if result.status == 2:
            return SeedResult(False,None,None,{},{},{},None,'infeasible',**meta)
        if result.status != 0 or not result.success:
            raise SeedSolverError(f'MILP status {result.status}: {result.message}')
        if result.x is None or len(result.x) != nvars or not np.all(np.isfinite(result.x)):
            raise SeedSolverError('MILP returned a missing/non-finite witness')
        counts = {m:int(round(result.x[xi[m]])) for m in models}
        if any(abs(result.x[xi[m]]-counts[m])>LP_TOLERANCE or counts[m] not in (0,1) for m in models):
            raise SeedSolverError('MILP returned invalid integer counts')
        routing = {(k,m):0. for k in workload.arrival_rate for m in models}
        for k,m in pairs:
            value = float(result.x[ri[k,m]])
            bound = workload.arrival_rate[k]*counts[m]
            if value < -LP_TOLERANCE or value > bound+LP_TOLERANCE:
                raise SeedSolverError('MILP returned routing outside deployment bounds')
            routing[k,m] = min(bound,max(0.,value))
        demand = {(k,m):v*model_catalog[m].service_time_ms.get(k,0)/1000. for (k,m),v in routing.items()}
        utilization = {m:math.fsum(demand[k,m] for k in workload.arrival_rate) for m in models}
        slack = {m:rho_max*counts[m]-utilization[m] for m in models}
        memory = math.fsum(model_catalog[m].memory_mib*counts[m] for m in models)
        if result.fun is None or not math.isclose(float(result.fun),memory,rel_tol=1e-10,abs_tol=LP_TOLERANCE):
            raise SeedSolverError('MILP objective inconsistent with deployment')
        seed = SeedResult(True,counts,routing,utilization,slack,demand,memory,'minimum_memory_feasible_seed',**meta)
        failures = validate_seed(seed,model_catalog,workload,hardware,rho_max)
        if failures:
            raise SeedSolverError('; '.join(failures))
        return seed

    def _solve_integer(self, catalog, workload, hardware, rho_max, replica_bounds):
        validate_rho_max(rho_max)
        if not catalog or any(m != p.name for m, p in catalog.items()):
            raise ValueError('Catalog must be nonempty and keys must match model names')
        models, tasks = sorted(catalog), workload.active_tasks
        pairs = [(k, m) for k in tasks for m in models if k in catalog[m].supported_tasks]
        size = len(models)
        xi = {m: i for i, m in enumerate(models)}
        yi = {m: size + i for i, m in enumerate(models)}
        ri = {pair: 2 * size + i for i, pair in enumerate(pairs)}
        nvars = 2 * size + len(pairs)
        row_ids, columns, values, lower, upper = [], [], [], [], []

        def row(entries, lb=-np.inf, ub=np.inf):
            index = len(lower)
            for col, value in entries:
                row_ids.append(index); columns.append(col); values.append(value)
            lower.append(lb); upper.append(ub)

        for m in models:
            row([(yi[m], 1.), (xi[m], -1.)], ub=0.)
            row([(xi[m], 1.), (yi[m], -replica_bounds[m])], ub=0.)
        for k in tasks:
            row([(yi[m], 1.) for m in models if k in catalog[m].supported_tasks], lb=1.)
        memory_entries = [(xi[m], catalog[m].memory_mib) for m in models]
        row(memory_entries, ub=hardware.gpu_memory_mib)
        for k in tasks:
            row([(ri[k, m], 1. / workload.arrival_rate[k]) for m in models if (k, m) in ri], lb=1., ub=1.)
        for k, m in pairs:
            row([(ri[k, m], 1. / workload.arrival_rate[k]), (yi[m], -1.)], ub=0.)
        for m in models:
            row([(ri[k, m], catalog[m].service_time_ms[k] / 1000.) for k in tasks if (k, m) in ri]
                + [(xi[m], -rho_max)], ub=0.)
        costs = np.array([catalog[m].memory_mib for m in models] + [0.] * (nvars - size))
        integrality = np.array([1] * (2 * size) + [0] * len(pairs))
        bounds = Bounds(np.zeros(nvars), np.array([replica_bounds[m] for m in models] +
                       [1.] * size + [workload.arrival_rate[k] for k, m in pairs]))
        start = time.perf_counter()
        calls = 0

        def run(cost):
            nonlocal calls
            calls += 1
            matrix = coo_matrix((values, (row_ids, columns)), shape=(len(lower), nvars)).tocsc()
            if not np.isfinite(matrix.data).all():
                raise ValueError('Inputs exceed representable MILP range')
            result = milp(c=cost, integrality=integrality, bounds=bounds,
                          constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
                          options={'mip_rel_gap': 0.0})
            if result.status == 2:
                return None
            if result.status != 0 or not result.success or result.x is None or not np.isfinite(result.x).all():
                raise SeedSolverError(f'Integer MILP failed: {result.message}')
            return result

        primary = run(costs)
        if primary is None:
            seed = IntegerSeedResult(False, None, None, {}, {}, {}, None, 'infeasible',
                              (time.perf_counter() - start) * 1000, num_variables=nvars, num_constraints=len(lower),
                              milp_call_count=calls, replica_bounds=dict(replica_bounds))
            return seed
        # Validate the primary witness before using its exact memory as the tie constraint.
        first = self._integer_witness(primary, xi, yi, ri, catalog, workload, hardware, rho_max, replica_bounds,
                                      start, nvars, len(lower))
        row(memory_entries, lb=first.objective_value, ub=first.objective_value)
        secondary = run(np.r_[np.ones(size), np.zeros(nvars - size)])
        if secondary is None:
            raise SeedSolverError('Secondary replica-count solve lost the verified primary memory optimum')
        seed = self._integer_witness(secondary, xi, yi, ri, catalog, workload, hardware, rho_max, replica_bounds,
                                     start, nvars, len(lower), memory_objective=False)
        if not math.isclose(seed.objective_value, first.objective_value, abs_tol=LP_TOLERANCE, rel_tol=0):
            raise SeedSolverError('Secondary objective changed the primary memory optimum')
        if not math.isclose(float(secondary.fun), sum(seed.deployment.values()), abs_tol=LP_TOLERANCE, rel_tol=0):
            raise SeedSolverError('Replica-count objective inconsistent')
        return IntegerSeedResult(**{f.name: getattr(seed, f.name) for f in fields(SeedResult)},
            milp_call_count=calls, presence={m: int(n > 0) for m, n in seed.deployment.items()},
            replica_bounds=dict(replica_bounds), secondary_objective_value=sum(seed.deployment.values()))

    @staticmethod
    def _integer_witness(result, xi, yi, ri, catalog, workload, hardware, rho_max, replica_bounds,
                         start, nvars, nconstraints, memory_objective=True):
        if len(result.x) != nvars:
            raise SeedSolverError('Wrong integer MILP witness dimension')
        counts = {m: int(round(result.x[i])) for m, i in xi.items()}
        for m in xi:
            y = int(round(result.x[yi[m]]))
            if (abs(result.x[xi[m]] - counts[m]) > LP_TOLERANCE or
                    not 0 <= counts[m] <= replica_bounds[m] or
                    abs(result.x[yi[m]] - y) > LP_TOLERANCE or y not in (0, 1) or
                    y != int(counts[m] > 0)):
                raise SeedSolverError('Invalid integer count/presence witness')
        routing = {(k, m): 0. for k in workload.arrival_rate for m in catalog}
        for (k, m), index in ri.items():
            bound = workload.arrival_rate[k] * int(counts[m] > 0)
            value = float(result.x[index])
            if value < -LP_TOLERANCE or value > bound + LP_TOLERANCE:
                raise SeedSolverError('Routing violates presence linkage')
            routing[k, m] = min(bound, max(0., value))
        demand = {(k, m): r * catalog[m].service_time_ms.get(k, 0.) / 1000. for (k, m), r in routing.items()}
        totals = {m: math.fsum(demand[k, m] for k in workload.arrival_rate) for m in catalog}
        pressure = {m: totals[m] / counts[m] if counts[m] else 0. for m in catalog}
        slack = {m: rho_max * counts[m] - totals[m] for m in catalog}
        memory = math.fsum(catalog[m].memory_mib * counts[m] for m in catalog)
        if memory_objective and (result.fun is None or not math.isclose(float(result.fun), memory, abs_tol=LP_TOLERANCE, rel_tol=1e-10)):
            raise SeedSolverError('Primary integer memory objective inconsistent')
        seed = SeedResult(True, counts, routing, pressure, slack, demand, memory,
                          'minimum_memory_then_replica_count_feasible_seed', (time.perf_counter() - start) * 1000,
                          str(result.message), nvars, nconstraints)
        errors = validate_seed(seed, catalog, workload, hardware, rho_max, replica_bounds)
        if errors:
            raise SeedSolverError('; '.join(errors))
        return seed
