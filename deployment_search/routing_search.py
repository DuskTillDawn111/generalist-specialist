"""Module 4A: bounded one-transfer routing search for a fixed deployment.

The production entry point never enumerates routing ratios. Its evaluator is the
same RuntimeEvaluator used by deployment search, so callers may pass the same
measurement/cache adapter. Dry runs never call it.
"""
from dataclasses import dataclass, field
import math

from deployment_core.routing_solver import LP_TOLERANCE, validate_rho_max

from .routing import solve_canonical_routing
from .runtime import RuntimeEvaluator
from .scoring import RuntimeObjective, optional_prediction
from .result import jsonable


@dataclass(frozen=True)
class RoutingSearchConfig:
    routing_step_fraction: float = .10
    routing_top_b: int = 2
    max_iterations: int = 5
    no_improvement_patience: int = 2
    runtime_evaluation_budget: int = 11  # Includes canonical routing.
    rho_max: float = .8
    mean_tolerance_ms: float = 1.
    dry_run: bool = False

    def __post_init__(self):
        if not math.isfinite(self.routing_step_fraction) or not 0 < self.routing_step_fraction <= 1:
            raise ValueError('routing_step_fraction must be in (0, 1]')
        for name in ('routing_top_b', 'max_iterations', 'no_improvement_patience',
                     'runtime_evaluation_budget'):
            value = getattr(self, name)
            minimum = 0 if name in ('max_iterations', 'runtime_evaluation_budget') else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f'Invalid {name}')
        validate_rho_max(self.rho_max)
        if not math.isfinite(self.mean_tolerance_ms) or self.mean_tolerance_ms < 0:
            raise ValueError('Invalid mean_tolerance_ms')


@dataclass
class RoutingMutation:
    task: str
    source: str
    destination: str
    delta_rps: float
    routing: dict
    source_pressure_before: float
    source_pressure_after: float
    destination_pressure_before: float
    destination_pressure_after: float
    max_pressure_before: float
    max_pressure_after: float
    delta_z: float
    status: str = 'GENERATED'
    reason: str = ''
    rank: int | None = None
    selected_top_b: bool = False
    prediction: float | None = None
    prediction_status: str = 'unavailable'
    objective: object = None


@dataclass
class RoutingSearchResult:
    termination_reason: str
    deployment: dict
    config: RoutingSearchConfig
    initial_routing: dict
    final_routing: dict
    initial_pressure: dict
    final_pressure: dict
    initial_objective: object = None
    final_objective: object = None
    mutations_generated: int = 0
    mutations_evaluated: int = 0
    accepted_mutations: list = field(default_factory=list)
    rejected_mutations: list = field(default_factory=list)
    runtime_evaluation_count: int = 0
    visited_routing_states: set = field(default_factory=set)
    num_active_tasks: int = 0
    num_transfer_candidates: int = 0
    routing_freedom: dict = field(default_factory=dict)
    iterations: list = field(default_factory=list)

    def to_dict(self):
        return dict(jsonable(self), visited_routing_state_count=len(self.visited_routing_states))


def routing_state_hash(routing, workload, deployment, catalog):
    """Normalize by task rate to merge only negligible LP/float roundoff."""
    return tuple((k, m, round(routing.get((k, m), 0.) / workload.arrival_rate[k], 9))
                 for k in workload.active_tasks for m in sorted(catalog)
                 if deployment.get(m, 0) and k in catalog[m].supported_tasks)


def routing_pressure(routing, deployment, workload, catalog):
    return {m: math.fsum(routing.get((k, m), 0.) * catalog[m].service_time_ms[k] / 1000
                         for k in workload.active_tasks if k in catalog[m].supported_tasks)
            for m in sorted(catalog) if deployment.get(m, 0)}


def validate_routing_candidate(routing, deployment, workload, catalog, rho_max):
    """Check conservation, nonnegativity, linkage and Stage-1 capacity."""
    if any(k not in workload.arrival_rate or m not in catalog or not math.isfinite(v)
           for (k, m), v in routing.items()):
        return False, 'invalid_entry'
    if any(v < 0 for v in routing.values()):
        return False, 'negative_routing'
    for (k, m), v in routing.items():
        if v > 0 and (not deployment.get(m, 0) or k not in catalog[m].supported_tasks):
            return False, 'capability_or_deployment'
    for k, rate in workload.arrival_rate.items():
        total = math.fsum(v for (task, _), v in routing.items() if task == k)
        if not math.isclose(total, rate, rel_tol=1e-7, abs_tol=max(1e-12, rate * 1e-9)):
            return False, 'workload_conservation'
    pressure = routing_pressure(routing, deployment, workload, catalog)
    if any(v > rho_max + LP_TOLERANCE for v in pressure.values()):
        return False, 'stage1_capacity'
    return True, 'feasible'


def generate_routing_mutations(routing, pressure, deployment, workload, catalog, config):
    """At most one transfer per ordered capable pair and active task."""
    z = max(pressure.values(), default=0.)
    candidates = []
    freedom = {}
    for k in workload.active_tasks:
        capable = [m for m in sorted(catalog) if deployment.get(m, 0)
                   and k in catalog[m].supported_tasks]
        freedom[k] = capable
        if len(capable) < 2:
            continue
        for source in capable:
            amount = routing.get((k, source), 0.)
            if amount <= max(1e-12, workload.arrival_rate[k] * 1e-12):
                continue
            delta = min(config.routing_step_fraction * workload.arrival_rate[k], amount)
            for destination in capable:
                if destination == source:
                    continue
                next_routing = dict(routing)
                next_routing[k, source] = max(0., amount - delta)
                next_routing[k, destination] = routing.get((k, destination), 0.) + delta
                source_after = pressure[source] - delta * catalog[source].service_time_ms[k] / 1000
                dest_after = pressure[destination] + delta * catalog[destination].service_time_ms[k] / 1000
                next_pressure = dict(pressure, **{source: source_after, destination: dest_after})
                next_z = max(next_pressure.values(), default=0.)
                candidates.append(RoutingMutation(
                    k, source, destination, delta, next_routing,
                    pressure[source], source_after, pressure[destination], dest_after,
                    z, next_z, z - next_z))
    return candidates, freedom


class GuidedRoutingSearch:
    def __init__(self, runtime_evaluator: RuntimeEvaluator | None = None, predictor=None,
                 runtime_objective=None, trace_callback=None):
        self.runtime_evaluator = runtime_evaluator
        self.predictor = predictor
        self.runtime_objective = runtime_objective
        self.trace_callback = trace_callback

    def _checkpoint(self, result):
        if self.trace_callback is not None:
            self.trace_callback(result)

    def search(self, deployment, workload, catalog, config=None):
        config = config or RoutingSearchConfig()
        if set(deployment) - set(catalog) or any(type(v) is not int or v not in (0, 1)
                                                   for v in deployment.values()):
            raise ValueError('Expected a fixed binary deployment in catalog')
        x = dict(deployment)
        canonical = solve_canonical_routing(x, workload, catalog, config.rho_max)
        if canonical is None:
            raise ValueError('Fixed deployment has no canonical feasible routing')
        initial = dict(canonical.routing)
        valid, reason = validate_routing_candidate(initial, x, workload, catalog, config.rho_max)
        if not valid:
            raise ValueError(f'Canonical routing failed validation: {reason}')
        pressure = routing_pressure(initial, x, workload, catalog)
        result = RoutingSearchResult('INITIALIZING', x, config, initial, dict(initial),
                                     dict(pressure), dict(pressure), num_active_tasks=len(workload.active_tasks))
        state = routing_state_hash(initial, workload, x, catalog)
        result.visited_routing_states.add(state)
        objective = self.runtime_objective or RuntimeObjective(config.mean_tolerance_ms)
        if not config.dry_run:
            if config.runtime_evaluation_budget == 0:
                result.termination_reason = 'RUNTIME_EVALUATION_BUDGET_REACHED'
                return result
            if self.runtime_evaluator is None:
                raise ValueError('Supply a RuntimeEvaluator or set dry_run=True')
            result.initial_objective = self.runtime_evaluator.evaluate(dict(x), workload, dict(initial))
            result.final_objective = result.initial_objective
            result.runtime_evaluation_count = 1
        self._checkpoint(result)
        current = dict(initial)
        no_improvement = 0
        fixed_memory = sum(catalog[m].memory_mib for m in catalog if x.get(m, 0))
        for iteration in range(config.max_iterations):
            if not config.dry_run and result.runtime_evaluation_count >= config.runtime_evaluation_budget:
                result.termination_reason = 'RUNTIME_EVALUATION_BUDGET_REACHED'
                break
            generated, freedom = generate_routing_mutations(current, pressure, x, workload, catalog, config)
            result.routing_freedom = freedom
            result.mutations_generated += len(generated)
            trace = dict(iteration=iteration, routing=dict(current), pressure=dict(pressure),
                         z=max(pressure.values(), default=0.), current_objective=result.final_objective,
                         num_active_tasks=len(workload.active_tasks), num_transfer_candidates=len(generated),
                         candidates=generated, selected_top_b=[], accepted=None)
            result.iterations.append(trace)
            result.num_transfer_candidates = len(generated)
            feasible = []
            for candidate in generated:
                valid, reason = validate_routing_candidate(candidate.routing, x, workload, catalog,
                                                            config.rho_max)
                if not valid:
                    candidate.status, candidate.reason = 'REJECT_INFEASIBLE', reason
                    continue
                key = routing_state_hash(candidate.routing, workload, x, catalog)
                if key in result.visited_routing_states:
                    candidate.status, candidate.reason = 'SKIP_VISITED', 'routing state already evaluated'
                    continue
                candidate.prediction, candidate.prediction_status = optional_prediction(
                    self.predictor, x, workload, candidate.routing)
                candidate.status = 'FEASIBLE'
                feasible.append(candidate)
            # Pressure relief is primary. Source/destination pressure preference
            # breaks exact ties before optional coarse predictions.
            feasible.sort(key=lambda c: (-c.delta_z,
                                          -(c.source_pressure_before - c.destination_pressure_before),
                                          c.prediction if c.prediction is not None else math.inf,
                                          c.task, c.source, c.destination))
            selected = feasible[:config.routing_top_b]
            for rank, candidate in enumerate(feasible, 1):
                candidate.rank = rank
                candidate.selected_top_b = rank <= config.routing_top_b
                candidate.status = 'SELECTED_TOP_B' if candidate.selected_top_b else 'PRUNED_TOP_B'
            trace['selected_top_b'] = [(c.task, c.source, c.destination) for c in selected]
            self._checkpoint(result)
            if not selected:
                result.termination_reason = 'NO_UNVISITED_CANDIDATE'
                break
            if config.dry_run:
                result.termination_reason = 'DRY_RUN_COMPLETE'
                break
            best = None
            for candidate in selected:
                if result.runtime_evaluation_count >= config.runtime_evaluation_budget:
                    candidate.status, candidate.reason = 'NOT_EVALUATED_BUDGET', 'runtime budget exhausted'
                    continue
                candidate.objective = self.runtime_evaluator.evaluate(dict(x), workload,
                                                                      dict(candidate.routing))
                result.runtime_evaluation_count += 1
                result.mutations_evaluated += 1
                result.visited_routing_states.add(routing_state_hash(candidate.routing, workload, x, catalog))
                improves, reason = objective.compare(candidate.objective, result.final_objective,
                                                     fixed_memory, fixed_memory)
                candidate.status, candidate.reason = 'ROLLBACK', reason
                if improves and (best is None or objective.compare(candidate.objective, best.objective,
                                                                   fixed_memory, fixed_memory)[0]):
                    if best is not None:
                        best.reason = 'another_top_b_candidate_preferred'
                    best = candidate
                elif improves:
                    candidate.reason = 'another_top_b_candidate_preferred'
                self._checkpoint(result)
            if best is not None:
                best.status, best.reason = 'ACCEPTED', best.reason
                current = dict(best.routing)
                pressure = routing_pressure(current, x, workload, catalog)
                result.final_routing = dict(current)
                result.final_pressure = dict(pressure)
                result.final_objective = best.objective
                result.accepted_mutations.append(best)
                trace['accepted'] = (best.task, best.source, best.destination)
                no_improvement = 0
            else:
                no_improvement += 1
            result.rejected_mutations.extend(c for c in selected if c.status == 'ROLLBACK')
            trace['no_improvement_iterations'] = no_improvement
            self._checkpoint(result)
            if result.runtime_evaluation_count >= config.runtime_evaluation_budget:
                result.termination_reason = 'RUNTIME_EVALUATION_BUDGET_REACHED'
                break
            if no_improvement >= config.no_improvement_patience:
                result.termination_reason = 'NO_RUNTIME_IMPROVEMENT'
                break
        else:
            result.termination_reason = 'MAX_ITERATIONS_REACHED'
        self._checkpoint(result)
        return result
