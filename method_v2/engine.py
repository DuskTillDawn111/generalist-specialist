"""Versioned v1 engine copy; only neighborhood metadata and selection differ."""
from .selection import diversified_order, candidate_class
from dataclasses import dataclass
import math
import random

from deployment_core.milp_seeder import MilpDeploymentSeeder
from deployment_core.routing_solver import RoutingSolverError, validate_rho_max

from deployment_search.feedback import bottleneck_feedback
from deployment_search.neighborhood import deployment_key, generate_neighborhood, memory_mib, missing_coverage
from deployment_search.result import CandidateEvaluation, SearchResult
from deployment_search.routing import solve_canonical_routing
from deployment_search.scoring import PerformanceCoarseScorer, RuntimeObjective, optional_prediction, rank_candidates
from deployment_search.runtime import RuntimeEvaluator


@dataclass(frozen=True)
class SearchConfig:
    top_b: int = 2
    max_iterations: int = 8
    no_improvement_patience: int = 2
    runtime_evaluation_budget: int = 17  # Includes seed evaluation.
    rho_max: float = .8
    remove_pressure_threshold: float = .2
    mean_tolerance_ms: float = 1.
    dry_run: bool = False

    def __post_init__(self):
        for name in ('top_b', 'max_iterations', 'no_improvement_patience', 'runtime_evaluation_budget'):
            value = getattr(self, name)
            minimum = 0 if name in ('max_iterations', 'runtime_evaluation_budget') else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f'Invalid {name}')
        validate_rho_max(self.rho_max)
        for name in ('remove_pressure_threshold', 'mean_tolerance_ms'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'Invalid {name}')


class RevisedGuidedSearch:
    def __init__(self, runtime_evaluator: RuntimeEvaluator | None = None,
                 predictor: PerformanceCoarseScorer | None = None,
                 runtime_objective=None, trace_callback=None,
                 selection_policy='lp', transition_policy='runtime', random_seed=None):
        if selection_policy not in ('lp', 'random') or transition_policy not in ('runtime', 'lp'):
            raise ValueError('Unknown search policy')
        if selection_policy == 'random' and random_seed is None:
            raise ValueError('Random selection requires a recorded seed')
        self.runtime_evaluator = runtime_evaluator
        self.predictor = predictor
        self.runtime_objective = runtime_objective
        self.trace_callback = trace_callback
        self.selection_policy, self.transition_policy = selection_policy, transition_policy
        self.random_seed = random_seed

    def _checkpoint(self, result):
        if self.trace_callback is not None:
            self.trace_callback(result)

    @staticmethod
    def _key(deployment):
        return deployment_key(deployment)

    def _memory(self, deployment, catalog):
        return memory_mib(deployment, catalog)

    def _memory_increment(self, model, before, after, catalog):
        return (after - before) * catalog[model].memory_mib

    def _seed(self, catalog, workload, hardware, rho_max):
        return MilpDeploymentSeeder().solve(catalog, workload, hardware, rho_max)

    def _routing(self, deployment, workload, catalog, rho_max):
        return solve_canonical_routing(deployment, workload, catalog, rho_max)

    def _mutations(self, deployment, catalog, workload, feedback, threshold):
        return generate_neighborhood(deployment, catalog, workload, feedback, threshold)

    def _warm_start(self, result, deployment, workload, catalog, hardware, config, visited):
        from deployment_search.warm_start import initialize_warm_search
        return initialize_warm_search(self, result, deployment, workload, catalog, hardware, config, visited)

    def _candidate_rejection(self, deployment, catalog, workload, hardware):
        if missing_coverage(deployment, catalog, workload):
            return 'coverage'
        if memory_mib(deployment, catalog) > hardware.gpu_memory_mib:
            return 'memory'
        return None

    def search(self, workload, model_catalog, hardware, config=None,
               initial_deployment: dict[str, int] | None = None,
               phase_visited_deployments=None) -> SearchResult:
        config = config or frozen_config()
        expected=frozen_config()
        for name in ('top_b','max_iterations','no_improvement_patience','runtime_evaluation_budget','rho_max','remove_pressure_threshold','mean_tolerance_ms'):
            if getattr(config,name)!=getattr(expected,name):raise ValueError('v2 frozen config changed: '+name)
        if self.predictor is not None or self.selection_policy!='lp' or self.transition_policy!='runtime':
            raise ValueError('v2 requires predictor=None and original runtime transition policy')
        result = SearchResult('INITIALIZING', config, len(model_catalog))
        if initial_deployment is None:
            # Preserve the Module 3 cold entry point and its exact counters.
            result.milp_solve_count = 1
            seed = self._seed(model_catalog, workload, hardware, config.rho_max)
            result.milp_solve_count = getattr(seed, 'milp_call_count', 1)
            result.seed = seed
            if not seed.feasible:
                result.status = 'NO_FEASIBLE_SEED'
                return result
            current = dict(seed.deployment)
            result.deployment = current
            result.visited_deployments.add(self._key(current))
            result.lp_solve_count += 1
            canonical = self._routing(current, workload, model_catalog, config.rho_max)
            if canonical is None:
                raise RoutingSolverError('Verified MILP seed failed canonical capacity feasibility')
            result.canonical = canonical
            cache = {self._key(current): canonical}
        else:
            current, canonical, cache = self._warm_start(result, initial_deployment,
                workload, model_catalog, hardware, config, phase_visited_deployments)
            if current is None:
                return result
        objective = self.runtime_objective or RuntimeObjective(config.mean_tolerance_ms)
        rng = random.Random(self.random_seed)
        if not config.dry_run:
            if config.runtime_evaluation_budget == 0:
                result.status = 'RUNTIME_EVALUATION_BUDGET_REACHED'
                return result
            if self.runtime_evaluator is None:
                raise ValueError('Supply a RuntimeEvaluator or explicitly set dry_run=True')
            result.runtime_evaluation_count += 1
            result.runtime = self.runtime_evaluator.evaluate(dict(current), workload, dict(canonical.routing))
            result.seed_runtime = result.runtime
        self._checkpoint(result)
        no_improvement = 0
        for offset in range(max(0, config.max_iterations - result.repair_iterations)):
            iteration = offset + result.repair_iterations
            if not config.dry_run and result.runtime_evaluation_count >= config.runtime_evaluation_budget:
                result.status = 'RUNTIME_EVALUATION_BUDGET_REACHED'
                break
            feedback = bottleneck_feedback(canonical, current)
            mutations = self._mutations(current, model_catalog, workload, feedback,
                                               config.remove_pressure_threshold)
            trace = dict(iteration=iteration, deployment=dict(current), current_runtime=result.runtime, canonical=canonical,
                         feedback=feedback, candidates=[], ranking=[], selected_top_b=[],
                         pruned_candidates=[], accepted=None, num_models=len(model_catalog),
                         num_candidates_generated=len(mutations), num_candidates_lp_evaluated=0,
                         num_runtime_candidates=0)
            result.iterations.append(trace)
            result.search_iterations += 1
            result.num_candidates_generated += len(mutations)
            parent = self._key(current)
            feasible = []
            for mutation in mutations:
                x = mutation.deployment
                key = self._key(x)
                candidate = CandidateEvaluation(mutation, memory_mib=self._memory(x, model_catalog))
                trace['candidates'].append(candidate)
                if key in result.visited_deployments or (parent, key) in result.rejected_mutations:
                    candidate.status, candidate.reason = 'SKIP_TABU', 'already visited or rejected'
                    continue
                rejection = self._candidate_rejection(x, model_catalog, workload, hardware)
                if rejection:
                    candidate.status, candidate.reason = 'REJECT_INFEASIBLE', rejection
                    continue
                if key not in cache:
                    result.lp_solve_count += 1
                    result.num_candidates_lp_evaluated += 1
                    trace['num_candidates_lp_evaluated'] += 1
                    cache[key] = self._routing(x, workload, model_catalog, config.rho_max)
                candidate.canonical = cache[key]
                if candidate.canonical is None:
                    candidate.status, candidate.reason = 'REJECT_INFEASIBLE', 'canonical_capacity_lp'
                    continue
                candidate.status = 'LP_FEASIBLE'
                candidate.delta_z = canonical.max_pressure - candidate.canonical.max_pressure
                if mutation.operation in ('ADD', 'SCALE_UP'):
                    increment = self._memory_increment(mutation.model, current[mutation.model], x[mutation.model], model_catalog)
                    candidate.relief_per_memory = candidate.delta_z / (increment / hardware.gpu_memory_mib) if increment > 0 else None
                else:
                    candidate.memory_freed_mib = self._memory(current,model_catalog)-candidate.memory_mib
                candidate.prediction, candidate.prediction_status = optional_prediction(
                    self.predictor, x, workload, candidate.canonical.routing)
                feasible.append(candidate)
            ranked = diversified_order(feasible,current,model_catalog,workload,self._memory(current,model_catalog))
            trace['candidate_complexity']=dict(self._generation_audit)
            trace['selection_policy']='CLASS_DIVERSIFIED_TOP_2'
            trace['candidate_classes']=[{'deployment':dict(c.mutation.deployment),'class':candidate_class(c)} for c in ranked]
            selected = ranked[:config.top_b]
            for rank, candidate in enumerate(ranked, 1):
                candidate.rank = rank
                candidate.selected_top_b = rank <= config.top_b
                candidate.status = 'SELECTED_TOP_B' if candidate.selected_top_b else 'PRUNED_TOP_B'
            trace['ranking'] = [self._key(c.mutation.deployment) for c in ranked]
            trace['selected_top_b'] = [self._key(c.mutation.deployment) for c in selected]
            trace['pruned_candidates'] = [self._key(c.mutation.deployment) for c in ranked[config.top_b:]]
            self._checkpoint(result)
            if not selected:
                result.status = 'NO_FEASIBLE_UNVISITED_CANDIDATE'
                break
            if config.dry_run:
                result.status = 'DRY_RUN_COMPLETE'
                break
            # Every candidate must improve the iteration's actual incumbent first.
            # Choose at most one accepted transition after all budgeted evaluations.
            best = None
            current_memory = self._memory(current, model_catalog)
            for candidate in selected:
                if result.runtime_evaluation_count >= config.runtime_evaluation_budget:
                    candidate.status, candidate.reason = 'NOT_EVALUATED_BUDGET', 'runtime budget exhausted'
                    continue
                result.runtime_evaluation_count += 1
                result.num_runtime_candidates += 1
                trace['num_runtime_candidates'] += 1
                x = candidate.mutation.deployment
                candidate.runtime = self.runtime_evaluator.evaluate(dict(x), workload, dict(candidate.canonical.routing))
                self._checkpoint(result)
                result.visited_deployments.add(self._key(x))
                if self.transition_policy == 'lp':
                    improves = candidate.delta_z > 0
                    reason = 'lp_delta_positive' if improves else 'lp_delta_nonpositive'
                else:
                    improves, reason = objective.compare(candidate.runtime, result.runtime,
                                                           candidate.memory_mib, current_memory)
                candidate.status, candidate.reason = 'ROLLBACK', reason
                if improves:
                    prefer = (best is None or
                              (candidate.delta_z > best.delta_z if self.transition_policy == 'lp' else
                               objective.compare(candidate.runtime, best.runtime,
                                                 candidate.memory_mib, best.memory_mib)[0]))
                    if prefer:
                        if best is not None:
                            best.reason = 'another_top_b_candidate_preferred'
                        best = candidate
                    else:
                        candidate.reason = 'another_top_b_candidate_preferred'
            if best is not None:
                best.status = 'ACCEPTED'
                current = dict(best.mutation.deployment)
                canonical = best.canonical
                result.deployment, result.canonical, result.runtime = current, canonical, best.runtime
                trace['accepted'] = self._key(current)
                no_improvement = 0
            else:
                no_improvement += 1
            for candidate in selected:
                if candidate.status == 'ROLLBACK':
                    result.rejected_mutations.add((parent, self._key(candidate.mutation.deployment)))
            trace['no_improvement_iterations'] = no_improvement
            self._checkpoint(result)
            if result.runtime_evaluation_count >= config.runtime_evaluation_budget:
                result.status = 'RUNTIME_EVALUATION_BUDGET_REACHED'
                break
            if no_improvement >= config.no_improvement_patience:
                result.status = 'NO_RUNTIME_IMPROVEMENT'
                break
        else:
            result.status = 'MAX_ITERATIONS_REACHED'
        self._checkpoint(result)
        return result


def frozen_config():
    return SearchConfig(top_b=2,max_iterations=3,no_improvement_patience=1,runtime_evaluation_budget=6,rho_max=.8,remove_pressure_threshold=.2,mean_tolerance_ms=1.)
