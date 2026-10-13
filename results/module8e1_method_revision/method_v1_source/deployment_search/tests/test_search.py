import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deployment_core.models import Hardware
from deployment_core.workload import Workload
from deployment_core.milp_seeder import MilpDeploymentSeeder
from deployment_core.routing_solver import RoutingSolverError
from deployment_search import (
    GuidedDeploymentSearch, SearchConfig, SyntheticRuntimeEvaluator, RuntimeObjective,
    solve_canonical_routing, weighted_task_latency,
)
from deployment_search.feedback import bottleneck_feedback
from deployment_search.neighborhood import generate_neighborhood, deployment_key
from deployment_search.result import CandidateEvaluation
from deployment_search.neighborhood import Mutation
from deployment_search.scoring import rank_candidates
from deployment_search.routing import linprog
from deployment_search.tests.fixtures import model, synthetic_pool, synthetic_evaluator, evaluation, forbid_enumeration
from deployment_search.tests.small_space_baseline import exhaustive_baseline


class RoutingTests(unittest.TestCase):
    def test_balanced_routing_linkage_and_contributions(self):
        catalog = {'a': model('a', text=1000.), 'b': model('b', text=500.),
                   'absent': model('absent', text=1.), 'image': model('image', image=100.)}
        x = dict(a=1, b=1, absent=0, image=1)
        w = Workload({'text': 1.})
        r = solve_canonical_routing(x, w, catalog)
        self.assertAlmostEqual(r.max_pressure, 1 / 3)
        self.assertAlmostEqual(r.routing['text', 'a'], 1 / 3)
        self.assertAlmostEqual(r.routing['text', 'b'], 2 / 3)
        self.assertEqual(r.routing['text', 'absent'], 0.)
        self.assertEqual(r.routing['text', 'image'], 0.)
        self.assertAlmostEqual(r.capacity_slack['a'], .8 - 1 / 3)
        self.assertEqual(r.capacity_slack['absent'], 0.)
        f = bottleneck_feedback(r, x)
        self.assertEqual(set(f.bottleneck_models), {'a', 'b'})
        self.assertEqual(f.pressure_tasks, ['text'])
        self.assertAlmostEqual(f.task_contributions['text', 'b'], 1 / 3)

    def test_infeasibility_empty_workload_tiny_rate_and_binary(self):
        c = {'a': model('a', text=1000.)}
        self.assertIsNone(solve_canonical_routing({'a': 1}, Workload({'asr': 1}), c))
        self.assertIsNone(solve_canonical_routing({'a': 1}, Workload({'text': 2}), c))
        r = solve_canonical_routing({'a': 0}, Workload({}), c)
        self.assertEqual(r.max_pressure, 0.)
        r = solve_canonical_routing({'a': 1}, Workload({'text': 1e-10}), c)
        self.assertAlmostEqual(r.routing['text', 'a'] / 1e-10, 1.)
        with self.assertRaises(ValueError):
            solve_canonical_routing({'a': 2}, Workload({}), c)

    def test_solver_failure_is_not_infeasible(self):
        with patch('deployment_search.routing.linprog', return_value=SimpleNamespace(
                status=4, success=False, x=None, message='numerical failure')):
            with self.assertRaises(RoutingSolverError):
                solve_canonical_routing({'a': 1}, Workload({'text': 1}), {'a': model('a', text=10.)})

    def test_canonical_agrees_with_module2g(self):
        from performance_model.comparable_groups.canonical import solve_canonical_routing as old
        w, c, h = synthetic_pool(5)
        x = {m: 1 for m in c}
        r = solve_canonical_routing(x, w, c)
        _, _, z = old(x, dict(w.arrival_rate), c)
        self.assertAlmostEqual(r.max_pressure, z)


class SearchTests(unittest.TestCase):
    def test_pools_scale_without_enumeration_and_improve(self):
        for size in (5, 10, 20, 30):
            with self.subTest(size=size), forbid_enumeration() as guards:
                w, c, h = synthetic_pool(size)
                evaluator = synthetic_evaluator(c)
                with patch.object(MilpDeploymentSeeder, 'solve', wraps=MilpDeploymentSeeder().solve) as seed:
                    r = GuidedDeploymentSearch(evaluator, predictor=None).search(w, c, h, SearchConfig())
                self.assertEqual(seed.call_count, 1)
                self.assertEqual(r.milp_solve_count, 1)
                self.assertLess(r.runtime.weighted_mean_e2e, r.seed_runtime.weighted_mean_e2e)
                self.assertLess(r.runtime_evaluation_count, 2 ** size)
                self.assertLessEqual(r.runtime_evaluation_count, 17)
                self.assertEqual(len(evaluator.calls), len(set(evaluator.calls)))
                self.assertEqual(r.visited_deployment_count, r.runtime_evaluation_count)
                self.assertEqual(r.lp_solve_count, r.num_candidates_lp_evaluated + 1)
                for trace in r.iterations:
                    self.assertLessEqual(trace['num_candidates_generated'], size)
                    self.assertLessEqual(trace['num_runtime_candidates'], 2)
                    for candidate in trace['candidates']:
                        flips = sum(trace['deployment'][m] != candidate.mutation.deployment[m] for m in c)
                        self.assertEqual(flips, 1)
                    ranked = sorted((v for v in trace['candidates'] if v.rank is not None), key=lambda v: v.rank)
                    self.assertEqual([v.delta_z for v in ranked], sorted((v.delta_z for v in ranked), reverse=True))
                    for v in ranked:
                        self.assertAlmostEqual(v.delta_z, trace['canonical'].max_pressure - v.canonical.max_pressure)
                        if v.mutation.operation == 'ADD':
                            self.assertAlmostEqual(v.relief_per_memory, v.delta_z / (c[v.mutation.model].memory_mib / h.gpu_memory_mib))
                json.dumps(r.to_dict(), allow_nan=False)

    def test_infeasible_seed_exits_without_lp_or_runtime(self):
        w, c, _ = synthetic_pool(5)
        with patch('deployment_search.search.solve_canonical_routing') as lp:
            r = GuidedDeploymentSearch().search(w, c, Hardware(0), SearchConfig())
        self.assertEqual(r.status, 'NO_FEASIBLE_SEED')
        self.assertEqual(r.runtime_evaluation_count, 0)
        lp.assert_not_called()

    def test_dry_run_and_seed_witness_not_used(self):
        w, c, h = synthetic_pool(5)
        seed = MilpDeploymentSeeder().solve(c, w, h)
        seed.routing_rps = {('text', 'base'): 9999.}
        evaluator = SyntheticRuntimeEvaluator(lambda *_: self.fail('dry-run runtime call'))
        with patch.object(MilpDeploymentSeeder, 'solve', return_value=seed), \
                patch('deployment_search.routing.linprog', wraps=linprog) as lp:
            r = GuidedDeploymentSearch(evaluator).search(w, c, h, SearchConfig(dry_run=True))
        self.assertEqual(r.status, 'DRY_RUN_COMPLETE')
        self.assertEqual(r.canonical.routing['text', 'base'], 1.)
        self.assertEqual(r.runtime_evaluation_count, 0)
        self.assertEqual(r.lp_solve_count, lp.call_count)
        self.assertEqual(len(r.iterations[0]['selected_top_b']), 2)
        self.assertEqual(len(r.iterations[0]['pruned_candidates']), 2)

    def test_rollback_preserves_all_results_and_tabu(self):
        w, c, h = synthetic_pool(10)
        ev = SyntheticRuntimeEvaluator(lambda x, *_: evaluation(10 if sum(x.values()) == 1 else 100,
                                                              raw='synthetic://retained'))
        r = GuidedDeploymentSearch(ev).search(w, c, h, SearchConfig())
        self.assertEqual(r.status, 'NO_RUNTIME_IMPROVEMENT')
        self.assertEqual(r.search_iterations, 2)
        self.assertEqual(deployment_key(r.deployment), ('base',))
        self.assertEqual(len(r.rejected_mutations), 4)
        self.assertEqual(len(ev.calls), len(set(ev.calls)))
        rejected = [v for t in r.iterations for v in t['candidates'] if v.status == 'ROLLBACK']
        self.assertEqual(len(rejected), 4)
        self.assertTrue(all(v.runtime.raw_result_path and v.reason == 'mean_worse' for v in rejected))
        self.assertTrue(any(v.status == 'SKIP_TABU' for v in r.iterations[1]['candidates']))

    def test_each_termination_and_exact_budget(self):
        w, c, h = synthetic_pool(10)
        for cfg, expected, count in [
            (SearchConfig(max_iterations=0), 'MAX_ITERATIONS_REACHED', 1),
            (SearchConfig(max_iterations=1), 'MAX_ITERATIONS_REACHED', 3),
            (SearchConfig(runtime_evaluation_budget=0), 'RUNTIME_EVALUATION_BUDGET_REACHED', 0),
            (SearchConfig(runtime_evaluation_budget=1), 'RUNTIME_EVALUATION_BUDGET_REACHED', 1),
            (SearchConfig(runtime_evaluation_budget=2), 'RUNTIME_EVALUATION_BUDGET_REACHED', 2),
        ]:
            with self.subTest(cfg=cfg):
                r = GuidedDeploymentSearch(synthetic_evaluator(c)).search(w, c, h, cfg)
                self.assertEqual(r.status, expected)
                self.assertEqual(r.runtime_evaluation_count, count)
        r = GuidedDeploymentSearch(synthetic_evaluator(c)).search(w, c, Hardware(1), SearchConfig())
        self.assertEqual(r.status, 'NO_FEASIBLE_UNVISITED_CANDIDATE')
        self.assertTrue(all(v.reason == 'memory' for v in r.iterations[0]['candidates']))

    def test_pressure_task_add_and_coverage_safe_remove(self):
        c = {'base': model('base', text=100.), 'text': model('text', text=100.),
             'image': model('image', image=100.)}
        w = Workload({'text': 1})
        x = dict(base=1, text=0, image=1)
        f = bottleneck_feedback(solve_canonical_routing(x, w, c), x)
        mutations = generate_neighborhood(x, c, w, f)
        self.assertEqual({(v.operation, v.model) for v in mutations}, {('ADD', 'text'), ('REMOVE', 'image')})
        x['image'] = 0
        mutations = generate_neighborhood(x, c, w, bottleneck_feedback(solve_canonical_routing(x, w, c), x))
        self.assertEqual([(v.operation, v.model) for v in mutations], [('ADD', 'text')])

    def test_removal_capacity_still_decided_by_lp(self):
        c = {m: model(m, text=1000.) for m in ('a', 'b')}
        r = GuidedDeploymentSearch().search(Workload({'text': 1.2}), c, Hardware(4),
            SearchConfig(dry_run=True, remove_pressure_threshold=.8))
        self.assertEqual(r.status, 'NO_FEASIBLE_UNVISITED_CANDIDATE')
        self.assertEqual(r.num_candidates_lp_evaluated, 2)
        self.assertTrue(all(v.reason == 'canonical_capacity_lp' for v in r.iterations[0]['candidates']))

    def test_accept_remove_and_prevent_return_cycles(self):
        c = {'base': model('base', 1., text=600.), 'b': model('b', text=200.),
             'c': model('c', text=300.)}
        values = {('base',): 100, ('b', 'base'): 60, ('base', 'c'): 80,
                  ('b', 'base', 'c'): 20, ('b',): 50, ('c',): 70, ('b', 'c'): 10}
        evaluator = SyntheticRuntimeEvaluator({key: evaluation(v) for key, v in values.items()})
        r = GuidedDeploymentSearch(evaluator).search(Workload({'text': 1}), c, Hardware(10),
            SearchConfig(remove_pressure_threshold=.8, mean_tolerance_ms=0.))
        self.assertEqual(deployment_key(r.deployment), ('b', 'c'))
        removed = [v for t in r.iterations for v in t['candidates']
                   if v.status == 'ACCEPTED' and v.mutation.operation == 'REMOVE']
        self.assertTrue(removed)
        self.assertEqual(removed[0].mutation.model, 'base')
        self.assertEqual(removed[0].memory_freed_mib, 1.)
        self.assertLess(removed[0].delta_z, 0.)  # Negative LP relief is not a rejection rule.
        self.assertEqual(len(evaluator.calls), len(set(evaluator.calls)))

    def test_bad_prediction_does_not_reject_feasible_candidate(self):
        class Pessimistic:
            def predict(self, *_):
                return 1e12
        w, c, h = synthetic_pool(5)
        r = GuidedDeploymentSearch(synthetic_evaluator(c), Pessimistic()).search(w, c, h)
        self.assertLess(r.runtime.weighted_mean_e2e, r.seed_runtime.weighted_mean_e2e)
        self.assertTrue(any(v.status == 'ACCEPTED' and v.prediction == 1e12
                            for t in r.iterations for v in t['candidates']))

    def test_optional_scorer_failure_cannot_block_search(self):
        class Bad:
            def predict(self, *_):
                raise RuntimeError('unavailable')
        w, c, h = synthetic_pool(5)
        r = GuidedDeploymentSearch(synthetic_evaluator(c), Bad()).search(w, c, h)
        self.assertLess(r.runtime.weighted_mean_e2e, r.seed_runtime.weighted_mean_e2e)
        self.assertTrue(any('RuntimeError' in v.prediction_status for t in r.iterations for v in t['candidates']))

    def test_small_exhaustive_is_separate_quality_baseline(self):
        w, c, h = synthetic_pool(5)
        with forbid_enumeration():
            r = GuidedDeploymentSearch(synthetic_evaluator(c)).search(w, c, h, SearchConfig(mean_tolerance_ms=0))
        oracle = exhaustive_baseline(w, c, h, synthetic_evaluator(c))
        gap = (r.runtime.weighted_mean_e2e - oracle['J_star']) / oracle['J_star']
        self.assertGreaterEqual(gap, -1e-10)
        self.assertLess(gap, .2)
        self.assertEqual(oracle['combinations_checked'], 32)
        w, c, h = synthetic_pool(10)
        with self.assertRaises(ValueError):
            exhaustive_baseline(w, c, h, synthetic_evaluator(c))


class ObjectiveTests(unittest.TestCase):
    def test_lexicographic_comparison(self):
        obj = RuntimeObjective(1.)
        for a, b, ma, mb, better, reason in [
            (evaluation(100, 'steady'), evaluation(1, 'recoverable_congested'), 2, 1, True, 'stability_improved'),
            (evaluation(100, 'recoverable_congested'), evaluation(1, 'unstable'), 2, 1, True, 'stability_improved'),
            (evaluation(1, 'unstable'), evaluation(100, 'steady'), 1, 2, False, 'stability_worse'),
            (evaluation(1, 'unstable'), evaluation(100, 'unstable'), 1, 2, False, 'same_nonsteady_stability'),
            (evaluation(9, p95=100), evaluation(10, p95=10), 2, 1, True, 'mean_improved'),
            (evaluation(10.5, p95=10), evaluation(10, p95=20), 2, 1, True, 'p95_improved'),
            (evaluation(10), evaluation(10), 1, 2, True, 'memory_improved'),
            (evaluation(10), evaluation(10), 1, 1, False, 'objective_tie'),
        ]:
            with self.subTest(reason=reason):
                self.assertEqual(obj.compare(a, b, ma, mb), (better, reason))
        self.assertEqual(weighted_task_latency(Workload({'text': 3, 'asr': 1}), {'text': 10, 'asr': 30}), 15)

    def test_predictor_is_secondary_only(self):
        def candidate(m, delta, prediction):
            return CandidateEvaluation(Mutation('ADD', m, {}), delta_z=delta, prediction=prediction)
        a, b, c = candidate('a', .5, 1e9), candidate('b', .2, 1), candidate('c', .2, 0)
        self.assertEqual(rank_candidates([b, c, a]), [a, c, b])
        b.prediction = None
        self.assertEqual(rank_candidates([b, c, a]), [a, b, c])

    def test_invalid_config(self):
        for kwargs in ({'top_b': 0}, {'runtime_evaluation_budget': -1}, {'max_iterations': True},
                       {'no_improvement_patience': 0}, {'mean_tolerance_ms': float('nan')}):
            with self.assertRaises(ValueError):
                SearchConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
