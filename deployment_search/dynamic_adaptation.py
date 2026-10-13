"""Explicit workload-change events and bounded warm adaptation, no forecasting."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .neighborhood import deployment_key
from .result import jsonable
from .search import GuidedDeploymentSearch, SearchConfig


@dataclass(frozen=True)
class DynamicState:
    """S_t = (x_t, r_t, lambda_t); routing is a canonical LP witness."""
    deployment: dict
    canonical_routing: dict
    workload_rps: dict


@dataclass(frozen=True)
class ReconfigurationCost:
    models_added: int
    models_removed: int
    added_vram_mib: float
    removed_vram_mib: float
    measured_switch_time_ms: float | None = None


class ReconfigurationCostModel(ABC):
    @abstractmethod
    def estimate(self, old_deployment, new_deployment) -> ReconfigurationCost:
        pass


class StructuralReconfigurationCost(ReconfigurationCostModel):
    def __init__(self, catalog):
        self.catalog = catalog

    def estimate(self, old_deployment, new_deployment):
        for x in (old_deployment, new_deployment):
            if set(x) - set(self.catalog) or any(type(n) is not int or n not in (0, 1) for n in x.values()):
                raise ValueError('Reconfiguration cost requires binary deployments in catalog')
        old, new = set(deployment_key(old_deployment)), set(deployment_key(new_deployment))
        added, removed = new - old, old - new
        return ReconfigurationCost(len(added), len(removed),
            sum(self.catalog[m].memory_mib for m in added),
            sum(self.catalog[m].memory_mib for m in removed), None)


@dataclass(frozen=True)
class AdaptationConfig:
    top_b: int = 2
    max_adaptation_iterations: int = 3
    max_runtime_evaluations: int = 4
    rho_max: float = .8
    remove_pressure_threshold: float = .2
    mean_tolerance_ms: float = 1.
    dry_run: bool = False

    def search_config(self):
        return SearchConfig(top_b=self.top_b, max_iterations=self.max_adaptation_iterations,
            no_improvement_patience=1, runtime_evaluation_budget=self.max_runtime_evaluations,
            rho_max=self.rho_max, remove_pressure_threshold=self.remove_pressure_threshold,
            mean_tolerance_ms=self.mean_tolerance_ms, dry_run=self.dry_run)

    def __post_init__(self):
        self.search_config()  # Reuse Module 3 validation.


@dataclass
class AdaptationHistory:
    last_deployment: dict | None = None
    previous_deployments: list = field(default_factory=list)
    accepted_switch_count: int = 0
    phase_visited_deployments: dict = field(default_factory=dict)
    phase_workloads: dict = field(default_factory=dict)


@dataclass
class DynamicSearchResult:
    old_workload: dict
    new_workload: dict
    initial_deployment: dict
    final_deployment: dict
    initial_feasible: bool
    fallback_to_milp: bool
    adaptation_trace: dict
    accepted_switches: list
    models_loaded: int
    models_unloaded: int
    vram_loaded_mib: float
    vram_released_mib: float
    lp_solve_count: int
    runtime_evaluation_count: int
    final_objective: object
    termination_reason: str
    action: str
    phase_id: str
    initial_inspection: object
    milp_solve_count: int
    initial_objective: object = None
    baseline_state: DynamicState | None = None
    final_state: DynamicState | None = None
    measured_switch_time_ms: float | None = None
    event: str = 'WORKLOAD_CHANGE'
    cost_semantics: str = 'committed structural path only; synthetic evaluation probes are not measured switching'

    def to_dict(self):
        return jsonable(self)


def committed_switches(search_result, initial_deployment, cost_model, include_cold_seed=False):
    """Include initialization/repair/fallback costs, not just local search costs."""
    switches = []
    current = dict(initial_deployment)
    def commit(x, mode, iteration=None):
        nonlocal current
        if deployment_key(x) != deployment_key(current):
            switches.append(dict(mode=mode, iteration=iteration, old_deployment=dict(current),
                new_deployment=dict(x), cost=cost_model.estimate(current, x)))
            current = dict(x)
    if include_cold_seed and search_result.seed and search_result.seed.feasible:
        commit(search_result.seed.deployment, 'cold_milp_initialization')
    for row in search_result.repair_trace:
        if row.get('new_deployment') is not None:
            commit(row['new_deployment'], row['mode'], row.get('iteration'))
    for row in search_result.iterations:
        for candidate in row['candidates']:
            if candidate.status == 'ACCEPTED':
                commit(candidate.mutation.deployment, 'normal', row['iteration'])
    return switches


class DynamicAdaptation:
    def __init__(self, runtime_evaluator=None, predictor=None, runtime_objective=None,
                 cost_model=None, history=None, trace_callback=None):
        self.runtime_evaluator = runtime_evaluator
        self.predictor = predictor
        self.runtime_objective = runtime_objective
        self.cost_model = cost_model
        self.history = history if history is not None else AdaptationHistory()
        self.trace_callback = trace_callback

    def adapt(self, current_deployment, old_workload, new_workload, model_catalog, hardware,
              config=None, phase_id='phase_1'):
        config = config or AdaptationConfig()
        phase = str(phase_id)
        new_rates = dict(new_workload.arrival_rate)
        if phase in self.history.phase_workloads and self.history.phase_workloads[phase] != new_rates:
            raise ValueError('A changed workload requires a new explicit phase_id')
        if self.history.last_deployment is not None and deployment_key(current_deployment) != deployment_key(self.history.last_deployment):
            raise ValueError('current_deployment differs from adaptation history last_deployment')
        search = GuidedDeploymentSearch(self.runtime_evaluator, self.predictor,
            self.runtime_objective, self.trace_callback).search(new_workload, model_catalog, hardware,
                config.search_config(), initial_deployment=current_deployment,
                phase_visited_deployments=self.history.phase_visited_deployments.get(phase, set()))
        cost_model = self.cost_model or StructuralReconfigurationCost(model_catalog)
        switches = committed_switches(search, search.initial_deployment, cost_model)
        final = dict(search.deployment)
        if not config.dry_run:
            self.history.phase_workloads[phase] = new_rates
            self.history.phase_visited_deployments.setdefault(phase, set()).update(search.visited_deployments)
            if not self.history.previous_deployments:
                self.history.previous_deployments.append(dict(search.initial_deployment))
            for item in switches:
                self.history.previous_deployments.append(dict(item['new_deployment']))
            self.history.accepted_switch_count += len(switches)
            self.history.last_deployment = dict(final)
        action = ('PLAN_ONLY' if config.dry_run else 'INFEASIBLE' if search.canonical is None else
                  'NO_SWITCH' if deployment_key(final) == deployment_key(search.initial_deployment) else 'SWITCHED')
        return DynamicSearchResult(dict(old_workload.arrival_rate), new_rates,
            dict(search.initial_deployment), final, search.initial_feasible, search.warm_start_fallback,
            search.to_dict(), switches,
            sum(s['cost'].models_added for s in switches), sum(s['cost'].models_removed for s in switches),
            sum(s['cost'].added_vram_mib for s in switches), sum(s['cost'].removed_vram_mib for s in switches),
            search.lp_solve_count, search.runtime_evaluation_count, search.runtime, search.status,
            action, phase, search.initial_inspection, search.milp_solve_count,
            search.seed_runtime if search.initial_feasible else None,
            baseline_state=DynamicState(dict(search.initial_deployment),
                dict(search.initial_inspection.canonical.routing), new_rates) if search.initial_feasible else None,
            final_state=DynamicState(final, dict(search.canonical.routing), new_rates) if search.canonical is not None else None)
