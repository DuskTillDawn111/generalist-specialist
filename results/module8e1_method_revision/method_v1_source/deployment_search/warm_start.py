"""Bounded one-hop ADD repair for an explicitly supplied deployment."""
from dataclasses import dataclass

from deployment_core.workload import Workload
from .feedback import bottleneck_feedback
from .neighborhood import Mutation, deployment_key, memory_mib, missing_coverage
from .routing import solve_canonical_routing, solve_pressure_relaxation


@dataclass
class FeasibilityInspection:
    feasible: bool
    canonical: object
    diagnostic_routing: object
    memory_mib: float
    memory_feasible: bool
    missing_capability: list
    unserved_workload_rps: dict
    model_pressure: dict
    max_pressure: float
    capacity_slack: dict
    pressure_tasks: list
    overloaded_pressure_tasks: list
    lp_solve_count: int
    reason: str


@dataclass
class WarmStartSeed:
    feasible: bool
    deployment: dict
    solver_status: str = 'warm_start_without_milp'


def validate_initial_deployment(deployment, catalog, hardware):
    if not isinstance(deployment, dict) or set(deployment) - set(catalog) or any(
            type(n) is not int or n not in (0, 1) for n in deployment.values()):
        raise ValueError('initial_deployment must be a binary deployment within the catalog')
    x = {m: deployment.get(m, 0) for m in sorted(catalog)}
    if memory_mib(x, catalog) > hardware.gpu_memory_mib:
        raise ValueError('initial_deployment exceeds hardware memory')
    return x


def inspect_deployment(deployment, workload, catalog, hardware, rho_max=.8):
    missing = missing_coverage(deployment, catalog, workload)
    memory = memory_mib(deployment, catalog)
    fits = memory <= hardware.gpu_memory_mib
    canonical = solve_canonical_routing(deployment, workload, catalog, rho_max)
    count = 1  # Same LP-call accounting as Module 3, including coverage prechecks.
    diagnostic = canonical
    if canonical is None:
        covered = Workload({k: rate for k, rate in workload.arrival_rate.items() if k not in missing})
        diagnostic = solve_pressure_relaxation(deployment, covered, catalog, rho_max)
        count += 1
    feedback = bottleneck_feedback(diagnostic, deployment)
    overloaded = sorted({k for (k, m), contribution in diagnostic.task_contributions.items()
                         if contribution > 0 and diagnostic.model_pressure[m] > rho_max + 1e-9})
    feasible = fits and canonical is not None
    return FeasibilityInspection(feasible, canonical, None if feasible else diagnostic,
        memory, fits, missing, {k: workload.arrival_rate[k] for k in missing},
        feedback.model_pressure, feedback.max_pressure, feedback.capacity_slack,
        feedback.pressure_tasks, overloaded, count,
        'feasible' if feasible else 'memory' if not fits else 'coverage' if missing else 'capacity')


def generate_repair_candidates(deployment, inspection, catalog):
    """One ADD per absent model supporting an unserved or overloaded task."""
    tasks = set(inspection.missing_capability) | set(inspection.overloaded_pressure_tasks)
    return [Mutation('ADD', m, dict(deployment, **{m: 1})) for m in sorted(catalog)
            if not deployment.get(m, 0) and catalog[m].supported_tasks & tasks]


def repair_rank(inspection):
    # Serving all demand is prerequisite to interpreting the relaxed pressure.
    # Among partially covered states, reduce unserved rate before pressure.
    return (not inspection.feasible, sum(inspection.unserved_workload_rps.values()),
            inspection.max_pressure)


def initialize_warm_search(search, result, initial_deployment, workload, catalog, hardware,
                           config, phase_visited_deployments=None):
    current = validate_initial_deployment(initial_deployment, catalog, hardware)
    result.start_mode = 'warm'
    result.initial_deployment = dict(current)
    inspection = inspect_deployment(current, workload, catalog, hardware, config.rho_max)
    result.initial_inspection = inspection
    result.initial_feasible = inspection.feasible
    result.lp_solve_count += inspection.lp_solve_count
    result.deployment = dict(current)
    result.visited_deployments.update(phase_visited_deployments or ())
    result.visited_deployments.add(deployment_key(current))
    cache = {deployment_key(current): inspection.canonical}
    search._checkpoint(result)
    for iteration in range(config.max_iterations):
        if inspection.feasible:
            break
        mutations = generate_repair_candidates(current, inspection, catalog)
        row = dict(iteration=iteration, mode='repair', deployment=dict(current),
                   inspection=inspection, candidates=[], selected_top_b=[], accepted=None)
        result.repair_trace.append(row)
        result.repair_iterations += 1
        ranked = []
        for mutation in mutations:
            result.num_candidates_generated += 1
            key = deployment_key(mutation.deployment)
            item = dict(mutation=mutation, inspection=None, rank=None, selected_top_b=False,
                        status='GENERATED', reason='')
            row['candidates'].append(item)
            if key in result.visited_deployments:
                item.update(status='SKIP_TABU', reason='already inspected in this phase')
                continue
            if memory_mib(mutation.deployment, catalog) > hardware.gpu_memory_mib:
                item.update(status='REJECT_INFEASIBLE', reason='memory')
                continue
            checked = inspect_deployment(mutation.deployment, workload, catalog, hardware, config.rho_max)
            result.lp_solve_count += checked.lp_solve_count
            result.num_candidates_lp_evaluated += 1
            item['inspection'] = checked
            cache[key] = checked.canonical
            ranked.append(item)
        ranked.sort(key=lambda c: (*repair_rank(c['inspection']), c['mutation'].model))
        for rank, item in enumerate(ranked, 1):
            item.update(rank=rank, selected_top_b=rank <= config.top_b,
                        status='SELECTED_TOP_B' if rank <= config.top_b else 'PRUNED_TOP_B')
        selected = ranked[:config.top_b]
        row['selected_top_b'] = [deployment_key(c['mutation'].deployment) for c in selected]
        # Repairs have no valid incumbent runtime. Accept only a structural
        # feasibility improvement, and never evaluate a partial/infeasible state.
        best = next((c for c in selected if repair_rank(c['inspection']) < repair_rank(inspection)), None)
        if config.dry_run:
            row['planned_repair'] = deployment_key(best['mutation'].deployment) if best else None
            if best:
                best.update(status='PLANNED_REPAIR', reason='planning only; no committed switch')
            result.seed = WarmStartSeed(False, dict(current), 'repair_plan_only')
            result.status = 'DRY_RUN_REPAIR_PLANNED'
            search._checkpoint(result)
            return None, None, cache
        if best:
            old = dict(current)
            current = dict(best['mutation'].deployment)
            inspection = best['inspection']
            best.update(status='ACCEPTED_REPAIR', reason='restore_feasibility' if inspection.feasible
                        else 'reduce_unserved_workload_or_overload')
            row.update(accepted=deployment_key(current), old_deployment=old, new_deployment=dict(current))
            result.deployment = dict(current)
            result.visited_deployments.add(deployment_key(current))
        search._checkpoint(result)
        if not best:
            break
    if not inspection.feasible:
        result.warm_start_fallback = True
        row = dict(mode='milp_fallback', old_deployment=dict(current),
                   reason='repair_unavailable_or_adaptation_iteration_budget_exhausted')
        result.repair_trace.append(row)
        from deployment_core.milp_seeder import MilpDeploymentSeeder
        result.milp_solve_count = 1
        seed = MilpDeploymentSeeder().solve(catalog, workload, hardware, config.rho_max)
        result.seed = seed
        if not seed.feasible:
            result.status = 'NO_FEASIBLE_SEED'
            row['status'] = 'NO_FEASIBLE_SEED'
            search._checkpoint(result)
            return None, None, cache
        current = dict(seed.deployment)
        row.update(status='FALLBACK_TO_MILP', new_deployment=dict(current))
        result.lp_solve_count += 1
        canonical = solve_canonical_routing(current, workload, catalog, config.rho_max)
        if canonical is None:
            from deployment_core.routing_solver import RoutingSolverError
            raise RoutingSolverError('Verified fallback MILP seed failed canonical feasibility')
        result.deployment = dict(current)
        result.visited_deployments.add(deployment_key(current))
        cache[deployment_key(current)] = canonical
    else:
        canonical = inspection.canonical
        result.seed = WarmStartSeed(True, dict(current))
    result.canonical = canonical
    return current, canonical, cache
