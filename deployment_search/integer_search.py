"""One-hop integer planning/search. No exhaustive oracle or GPU-serving imports."""
from dataclasses import dataclass, field, fields
import math

from deployment_core.milp_seeder import MilpDeploymentSeeder
from deployment_core.replicas import (derive_replica_bounds, normalize_counts, replica_key,
                                     replica_identities, replica_statistics, validate_bounds)
from .search import GuidedDeploymentSearch
from .neighborhood import deployment_key, memory_mib, missing_coverage, generate_replica_neighborhood
from .result import SearchResult
from .routing import solve_canonical_routing
from .warm_start import WarmStartSeed


@dataclass
class IntegerSearchResult(SearchResult):
    planning_mode: str = 'integer_replicas'
    replica_bounds: dict = field(default_factory=dict)
    bound_provenance: dict = field(default_factory=dict)
    replica_counts_statistics: dict | None = None
    future_replica_identities: dict = field(default_factory=dict)
    theoretical_replica_vector_count: int = 0
    exhaustive_enumeration_used: bool = False
    memory_model: str = 'planning memory surrogate: additive per-replica idle VRAM'
    capacity_mode: str = 'linear'
    memory_mode: str = 'additive'
    calibration_source_hashes: dict = field(default_factory=dict)
    capacity_model: str = 'analytical planning surrogate; no claim of linear real-GPU scaling'


class IntegerGuidedDeploymentSearch(GuidedDeploymentSearch):
    def search(self, workload, model_catalog, hardware, config=None, initial_deployment=None,
               phase_visited_deployments=None, replica_bounds=None, binary_mode=False,
               worker_limits=None, roles=None, capacity_mode=None, memory_mode=None, calibration=None):
        from deployment_core.replica_calibration import ReplicaPlanningPolicy
        # Real completed v2 inputs default to empirical; old/synthetic catalogs
        # keep their historical linear default. Explicit linear is always available.
        if capacity_mode is None:
            capacity_mode = 'empirical' if model_catalog and all(p.sources.get('completed_catalog') for p in model_catalog.values()) else 'linear'
        self._planning = ReplicaPlanningPolicy(model_catalog, capacity_mode, memory_mode, calibration)
        if capacity_mode == 'empirical' or self._planning.memory_mode == 'empirical':
            physical, provenance = self._planning.bounds(hardware, replica_bounds, worker_limits, binary_mode)
            replica_bounds = None  # already validated/intersected with evidence
        else:
            physical, provenance = derive_replica_bounds(model_catalog, hardware, worker_limits, binary_mode)
        if replica_bounds is not None:
            validate_bounds(replica_bounds, model_catalog)
            if any(replica_bounds[m] > physical[m] for m in model_catalog):
                raise ValueError('Explicit replica bound exceeds physical/process/binary upper bound')
            for m in provenance:
                provenance[m].update(explicit_experiment_bound=replica_bounds[m],
                                     bound_source=provenance[m]['bound_source'] + ' intersect explicit experiment bound',
                                     R=replica_bounds[m])
            physical = dict(replica_bounds)
        self._bounds = physical
        self._binary_reduction = all(n == 1 for n in physical.values())
        self._roles = roles
        if roles is not None:
            replica_statistics({m: 0 for m in model_catalog}, roles)
        result = super().search(workload, model_catalog, hardware, config, initial_deployment,
                                phase_visited_deployments)
        if self._binary_reduction:
            # Preserve binary operation names, keys, counters and serialization.
            return result
        extended = IntegerSearchResult(**{f.name: getattr(result, f.name) for f in fields(SearchResult)},
            replica_bounds=dict(physical), bound_provenance=provenance,
            theoretical_replica_vector_count=math.prod(n + 1 for n in physical.values()),
            capacity_mode=self._planning.capacity_mode, memory_mode=self._planning.memory_mode,
            calibration_source_hashes=self._planning.calibration.source_hashes if self._planning.calibration else {},
            capacity_model='D/g(x): evidence-calibrated family planning surrogate' if capacity_mode == 'empirical' else 'D/x: linear analytical planning surrogate',
            memory_model='M(x): measured discrete idle-memory planning surrogate' if self._planning.memory_mode == 'empirical' else 'additive measured single idle-memory planning surrogate')
        if result.deployment is not None:
            extended.future_replica_identities = replica_identities(result.deployment)
            if roles is not None:
                extended.replica_counts_statistics = replica_statistics(result.deployment, roles)
        for trace in extended.iterations:
            trace['replica_count_bound'] = 2 * len(model_catalog)
            trace['pressure_definition'] = 'family total service demand / g(x)' if capacity_mode == 'empirical' else 'family total service demand / deployed replica count'
            trace['effective_capacity'] = {m: self._planning.capacity(m, trace['deployment'][m]) for m in model_catalog}
            trace['deployment_memory_mib'] = self._memory(trace['deployment'], model_catalog)
            trace['family_total_demand'] = {m: sum(d for (k, name), d in trace['canonical'].task_contributions.items() if name == m)
                                            for m in model_catalog}
            if roles is not None:
                trace['replica_counts_statistics'] = replica_statistics(trace['deployment'], roles)
                trace['candidate_replica_counts_statistics'] = [
                    {'deployment': dict(c.mutation.deployment), 'counts': replica_statistics(c.mutation.deployment, roles)}
                    for c in trace['candidates']]
            if trace['num_candidates_generated'] > 2 * len(model_catalog):
                raise AssertionError('One-hop replica neighborhood exceeded 2M')
        self._checkpoint(extended)
        return extended

    def _key(self, deployment):
        return deployment_key(deployment) if self._binary_reduction else replica_key(deployment)

    def _memory(self, deployment, catalog):
        if self._binary_reduction or self._planning.memory_mode == 'additive':
            return super()._memory(deployment, catalog)
        return self._planning.deployment_memory(deployment)

    def _memory_increment(self, model, before, after, catalog):
        if self._binary_reduction or self._planning.memory_mode == 'additive':
            return super()._memory_increment(model, before, after, catalog)
        return self._planning.memory(model, after) - self._planning.memory(model, before)

    def _seed(self, catalog, workload, hardware, rho_max):
        return MilpDeploymentSeeder().solve(catalog, workload, hardware, rho_max, replica_bounds=self._bounds,
            capacity_mode=self._planning.capacity_mode, memory_mode=self._planning.memory_mode, calibration=self._planning.calibration)

    def _routing(self, deployment, workload, catalog, rho_max):
        return solve_canonical_routing(deployment, workload, catalog, rho_max,
                                       replica_bounds=None if self._binary_reduction and self._planning.capacity_mode == 'linear' else self._bounds,
                                       capacity_mode=self._planning.capacity_mode, calibration=self._planning.calibration)

    def _mutations(self, deployment, catalog, workload, feedback, threshold):
        if self._binary_reduction:
            return super()._mutations(deployment, catalog, workload, feedback, threshold)
        return generate_replica_neighborhood(deployment, catalog, workload, feedback, self._bounds, threshold)

    def _candidate_rejection(self, deployment, catalog, workload, hardware):
        if self._binary_reduction:
            return super()._candidate_rejection(deployment, catalog, workload, hardware)
        try:
            normalize_counts(deployment, catalog, self._bounds)
        except ValueError:
            return 'replica_bounds'
        if self._memory(deployment, catalog) > hardware.gpu_memory_mib:
            return 'memory'
        if missing_coverage(deployment, catalog, workload):
            return 'coverage'
        return None

    def _warm_start(self, result, deployment, workload, catalog, hardware, config, visited):
        if self._binary_reduction:
            return super()._warm_start(result, deployment, workload, catalog, hardware, config, visited)
        current = normalize_counts(deployment, catalog, self._bounds)
        if self._memory(current, catalog) > hardware.gpu_memory_mib:
            raise ValueError('Initial integer deployment exceeds planning memory budget')
        result.start_mode = 'warm'
        result.initial_deployment = dict(current)
        result.deployment = dict(current)
        result.visited_deployments.update(visited or ())
        result.visited_deployments.add(self._key(current))
        result.lp_solve_count += 1
        canonical = self._routing(current, workload, catalog, config.rho_max)
        result.initial_feasible = canonical is not None
        if canonical is None:
            # Never evaluate a capacity/coverage-infeasible initial state.
            result.status = 'INITIAL_REPLICA_STATE_INFEASIBLE'
            result.seed = WarmStartSeed(False, current, 'initial_integer_feasibility_failed')
            return None, None, {}
        result.seed = WarmStartSeed(True, dict(current), 'integer_initial_state_without_milp')
        result.canonical = canonical
        return current, canonical, {self._key(current): canonical}
