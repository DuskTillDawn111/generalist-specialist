"""Module 4A invariants; only offline synthetic evaluators are used."""
import unittest
from unittest.mock import patch

from deployment_core.workload import Workload
from deployment_search.routing_search import (
    GuidedRoutingSearch, RoutingSearchConfig, generate_routing_mutations,
    routing_pressure, routing_state_hash, validate_routing_candidate,
)
from deployment_search.routing_foundation import profile, synthetic_cases, synthetic_runtime
from deployment_search.routing import solve_canonical_routing
from deployment_search.tests.fixtures import forbid_enumeration


class RoutingSearchTests(unittest.TestCase):
    def test_canonical_start_one_transfer_conservation_top_b_and_no_grid(self):
        workload, catalog, objective = synthetic_cases()['case3_four_capable_models']
        deployment = {m: 1 for m in catalog}
        canonical = solve_canonical_routing(deployment, workload, catalog)
        with forbid_enumeration(), patch('deployment_search.routing_foundation.small_grid_oracle',
                                         side_effect=AssertionError('grid baseline entered')):
            result = GuidedRoutingSearch(synthetic_runtime(objective), predictor=None).search(
                deployment, workload, catalog,
                RoutingSearchConfig(max_iterations=1, runtime_evaluation_budget=3))
        self.assertEqual(result.initial_routing, canonical.routing)
        self.assertEqual(result.num_transfer_candidates, 12)
        self.assertEqual(len(result.iterations[0]['selected_top_b']), 2)
        self.assertEqual(result.runtime_evaluation_count, 3)
        self.assertEqual(deployment, result.deployment)
        for mutation in result.iterations[0]['candidates']:
            changes = {pair for pair, value in mutation.routing.items()
                       if abs(value - canonical.routing[pair]) > 1e-12}
            self.assertEqual(changes, {(mutation.task, mutation.source),
                                       (mutation.task, mutation.destination)})
            self.assertAlmostEqual(mutation.routing[mutation.task, mutation.source],
                                   canonical.routing[mutation.task, mutation.source]-mutation.delta_rps)
            self.assertAlmostEqual(mutation.routing[mutation.task, mutation.destination],
                                   canonical.routing[mutation.task, mutation.destination]+mutation.delta_rps)
            self.assertTrue(validate_routing_candidate(mutation.routing, deployment,
                                                       workload, catalog, .8)[0])
            self.assertAlmostEqual(mutation.delta_z,
                                   mutation.max_pressure_before-mutation.max_pressure_after)
            direct = routing_pressure(mutation.routing, deployment, workload, catalog)
            self.assertAlmostEqual(mutation.max_pressure_after, max(direct.values()))

    def test_invalid_linkage_negative_conservation_and_capacity(self):
        w = Workload({'text': 1.})
        c = {'a': profile('a', text=600.), 'b': profile('b', text=300.),
             'image': profile('image', image=100.)}
        x = {'a': 1, 'b': 1, 'image': 0}
        cases = [
            ({('text', 'a'): -.1, ('text', 'b'): 1.1}, 'negative_routing'),
            ({('text', 'a'): .5, ('text', 'b'): .4}, 'workload_conservation'),
            ({('text', 'a'): .9, ('text', 'image'): .1}, 'capability_or_deployment'),
        ]
        for routing, expected in cases:
            self.assertEqual(validate_routing_candidate(routing, x, w, c, .8)[1], expected)
        self.assertEqual(validate_routing_candidate({('text', 'a'): 1.}, x, w, c, .5)[1],
                         'stage1_capacity')

    def test_runtime_feedback_rollback_cycle_avoidance_and_budget(self):
        w, c, _ = synthetic_cases()['case2_runtime_correction']
        x = {m: 1 for m in c}
        # Constant runtime rejects both analytically ranked candidates. A second
        # iteration must not reevaluate either routing, even though it revisits
        # the same incumbent and generates the same one-step neighborhood.
        evaluator = synthetic_runtime(lambda routing: 100.)
        r = GuidedRoutingSearch(evaluator).search(
            x, w, c, RoutingSearchConfig(max_iterations=5,
                                        no_improvement_patience=2,
                                        runtime_evaluation_budget=10))
        self.assertEqual(r.termination_reason, 'NO_UNVISITED_CANDIDATE')
        self.assertEqual(r.final_routing, r.initial_routing)
        self.assertEqual(r.runtime_evaluation_count, 3)
        self.assertEqual(len(r.rejected_mutations), 2)
        self.assertEqual(len(r.visited_routing_states), 3)
        self.assertTrue(all(m.status == 'SKIP_VISITED' for m in r.iterations[1]['candidates']))
        r = GuidedRoutingSearch(synthetic_runtime(lambda routing: 100.)).search(
            x, w, c, RoutingSearchConfig(runtime_evaluation_budget=2))
        self.assertEqual(r.termination_reason, 'RUNTIME_EVALUATION_BUDGET_REACHED')
        self.assertEqual(r.runtime_evaluation_count, 2)
        w, c, _ = synthetic_cases()['case3_four_capable_models']
        r = GuidedRoutingSearch(synthetic_runtime(lambda routing: 100.)).search(
            {m: 1 for m in c}, w, c,
            RoutingSearchConfig(no_improvement_patience=2, runtime_evaluation_budget=15))
        self.assertEqual(r.termination_reason, 'NO_RUNTIME_IMPROVEMENT')
        self.assertEqual(len(r.iterations), 2)

    def test_synthetic_correction_and_optional_predictor(self):
        w, c, objective = synthetic_cases()['case2_runtime_correction']
        result = GuidedRoutingSearch(synthetic_runtime(objective), predictor=None).search(
            {m: 1 for m in c}, w, c, RoutingSearchConfig(max_iterations=8))
        self.assertLess(result.final_objective.weighted_mean_e2e,
                        result.initial_objective.weighted_mean_e2e)
        self.assertTrue(any(m.delta_z < 0 for m in result.accepted_mutations))
        self.assertTrue(all(m.prediction is None for trace in result.iterations
                            for m in trace['candidates']))
        first = sorted((m for m in result.iterations[0]['candidates'] if m.rank is not None),
                       key=lambda m: m.rank)
        self.assertGreater(first[0].delta_z, first[1].delta_z)
        self.assertEqual(first[0].status, 'ROLLBACK')
        self.assertEqual(first[1].status, 'ACCEPTED')

    def test_dry_run_never_calls_runtime_and_fixed_deployment(self):
        w, c, objective = synthetic_cases()['case1_specialist']
        evaluator = synthetic_runtime(objective)
        x = {m: 1 for m in c}
        r = GuidedRoutingSearch(evaluator).search(x, w, c, RoutingSearchConfig(dry_run=True))
        self.assertEqual(r.termination_reason, 'DRY_RUN_COMPLETE')
        self.assertEqual(evaluator.calls, [])
        self.assertEqual(r.deployment, x)
        self.assertEqual(r.runtime_evaluation_count, 0)


if __name__ == '__main__':
    unittest.main()
