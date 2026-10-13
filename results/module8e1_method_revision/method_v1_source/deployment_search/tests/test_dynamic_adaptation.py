"""Module 5A CPU-only invariants. Every evaluator is synthetic or forbidden."""
import copy
import json
import unittest
from unittest.mock import patch

from deployment_core.models import Hardware
from deployment_core.workload import Workload
from deployment_core.milp_seeder import MilpDeploymentSeeder
from deployment_search import (GuidedDeploymentSearch, SearchConfig, DynamicAdaptation,
    AdaptationConfig, AdaptationHistory, StructuralReconfigurationCost)
from deployment_search.dynamic_foundation import scenarios, synthetic_objective, NoRuntimeAllowed
from deployment_search.neighborhood import deployment_key
from deployment_search.runtime import SyntheticRuntimeEvaluator
from deployment_search.routing import solve_canonical_routing
from deployment_search.tests.fixtures import model, evaluation, synthetic_pool, synthetic_evaluator, forbid_enumeration
from deployment_search.warm_start import inspect_deployment


class DynamicAdaptationTests(unittest.TestCase):
    def test_default_and_explicit_none_preserve_exact_cold_results(self):
        w, c, h = synthetic_pool(5)
        seed = MilpDeploymentSeeder().solve(c, w, h)
        with patch('deployment_search.search.MilpDeploymentSeeder.solve', return_value=seed) as seeder, forbid_enumeration():
            a = GuidedDeploymentSearch(synthetic_evaluator(c)).search(w, c, h)
            b = GuidedDeploymentSearch(synthetic_evaluator(c)).search(w, c, h, initial_deployment=None)
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual(seeder.call_count, 2)
        self.assertEqual(a.milp_solve_count, 1)
        self.assertNotIn('start_mode', a.to_dict())

    def test_feasible_warm_start_never_invokes_milp_or_mutates_input(self):
        case = scenarios()[0]
        initial = copy.deepcopy(case['initial'])
        with patch.object(MilpDeploymentSeeder, 'solve', side_effect=AssertionError('warm MILP forbidden')), forbid_enumeration():
            result = DynamicAdaptation(synthetic_objective(case['catalog'])).adapt(initial,
                case['old'], case['new'], case['catalog'], case['hardware'])
        self.assertEqual(initial, case['initial'])
        self.assertEqual(result.action, 'NO_SWITCH')
        self.assertTrue(result.initial_feasible)
        self.assertFalse(result.fallback_to_milp)
        self.assertEqual(result.milp_solve_count, 0)
        self.assertEqual(result.models_loaded, 0)
        self.assertEqual(result.final_state.workload_rps, dict(case['new'].arrival_rate))
        json.dumps(result.to_dict(), allow_nan=False)

    def test_specialist_swap_is_two_one_hop_switches(self):
        case = scenarios()[1]
        result = DynamicAdaptation(synthetic_objective(case['catalog'])).adapt(case['initial'],
            case['old'], case['new'], case['catalog'], case['hardware'])
        self.assertEqual(deployment_key(result.final_deployment), ('generalist', 'specialist_b'))
        self.assertEqual(len(result.accepted_switches), 2)
        self.assertEqual(result.runtime_evaluation_count, 4)
        self.assertEqual((result.models_loaded, result.models_unloaded), (1, 1))
        self.assertEqual((result.vram_loaded_mib, result.vram_released_mib), (2., 2.))
        for switch in result.accepted_switches:
            self.assertEqual(len(set(deployment_key(switch['old_deployment'])) ^ set(deployment_key(switch['new_deployment']))), 1)

    def test_multi_hop_coverage_repair_before_any_runtime_evaluation(self):
        c = {'text': model('text', 1., text=100.), 'image': model('image', 1., image=100.),
             'asr': model('asr', 1., asr=100.)}
        new = Workload({'text': .1, 'image': 1., 'asr': .5})
        ev = SyntheticRuntimeEvaluator(lambda x, w, r: evaluation(50.))
        with patch.object(MilpDeploymentSeeder, 'solve', side_effect=AssertionError('premature fallback')), forbid_enumeration():
            result = DynamicAdaptation(ev).adapt({'text': 1}, Workload({'text': .1}), new, c, Hardware(5))
        self.assertFalse(result.initial_feasible)
        self.assertFalse(result.fallback_to_milp)
        self.assertEqual(result.adaptation_trace['repair_iterations'], 2)
        self.assertEqual(ev.calls, [('asr', 'image', 'text')])
        self.assertEqual(result.models_loaded, 2)
        self.assertTrue(all(row['mode'] == 'repair' for row in result.adaptation_trace['repair_trace']))

    def test_capacity_relaxation_is_diagnostic_and_can_exceed_one(self):
        case = scenarios()[3]
        inspect = inspect_deployment(case['initial'], case['new'], case['catalog'], case['hardware'])
        self.assertFalse(inspect.feasible)
        self.assertIsNone(inspect.canonical)
        self.assertGreater(inspect.max_pressure, 1.)
        self.assertIn('text', inspect.overloaded_pressure_tasks)
        self.assertLess(inspect.capacity_slack['generalist'], 0.)
        with patch.object(MilpDeploymentSeeder, 'solve', side_effect=AssertionError('no fallback needed')):
            result = DynamicAdaptation(synthetic_objective(case['catalog'], 10.)).adapt(case['initial'],
                case['old'], case['new'], case['catalog'], case['hardware'])
        self.assertEqual(result.accepted_switches[0]['mode'], 'repair')
        self.assertIsNotNone(result.final_state)

    def test_memory_blocked_add_falls_back_and_accounts_seed_swap(self):
        c = {'generalist': model('generalist', 1., text=900., image=900.),
             'specialist_a': model('specialist_a', 2., text=100.),
             'specialist_b': model('specialist_b', 2., image=100.)}
        old = Workload({'text': .5, 'image': .1})
        new = Workload({'text': .1, 'image': 1.})
        result = DynamicAdaptation(synthetic_objective(c)).adapt(
            {'generalist': 1, 'specialist_a': 1}, old, new, c, Hardware(3))
        self.assertFalse(result.initial_feasible)
        self.assertTrue(result.fallback_to_milp)
        self.assertEqual(result.milp_solve_count, 1)
        self.assertLessEqual(result.runtime_evaluation_count, 4)
        self.assertLessEqual(result.adaptation_trace['repair_iterations'] + result.adaptation_trace['search_iterations'], 3)
        self.assertEqual(result.accepted_switches[0]['mode'], 'milp_fallback')
        self.assertEqual((result.models_loaded, result.models_unloaded), (1, 1))

    def test_repair_budget_consumed_before_fallback_no_automatic_expansion(self):
        c = {'text': model('text', 1., text=100.), 'image': model('image', 1., image=100.),
             'asr': model('asr', 1., asr=100.)}
        new = Workload({'text': .1, 'image': 1., 'asr': .5})
        cfg = AdaptationConfig(max_adaptation_iterations=1, max_runtime_evaluations=1)
        result = DynamicAdaptation(SyntheticRuntimeEvaluator(lambda *args: evaluation(50.))).adapt(
            {'text': 1}, Workload({'text': .1}), new, c, Hardware(5), cfg)
        self.assertTrue(result.fallback_to_milp)
        self.assertEqual(result.adaptation_trace['repair_iterations'], 1)
        self.assertEqual(result.adaptation_trace['search_iterations'], 0)
        self.assertEqual(result.runtime_evaluation_count, 1)

    def test_global_infeasibility_never_enters_runtime_evaluator(self):
        c = {'text': model('text', 1., text=100.)}
        result = DynamicAdaptation(NoRuntimeAllowed()).adapt({'text': 1}, Workload({'text': 1.}),
            Workload({'text': 1., 'image': 1.}), c, Hardware(5))
        self.assertEqual(result.action, 'INFEASIBLE')
        self.assertTrue(result.fallback_to_milp)
        self.assertEqual(result.runtime_evaluation_count, 0)
        self.assertIsNone(result.final_objective)

    def test_phase_history_prevents_repeat_cycles_but_allows_return(self):
        case = scenarios()[1]
        history = AdaptationHistory()
        adapter = DynamicAdaptation(synthetic_objective(case['catalog']), history=history)
        a = adapter.adapt(case['initial'], case['old'], case['new'], case['catalog'], case['hardware'], phase_id='B')
        b = adapter.adapt(a.final_deployment, case['new'], case['old'], case['catalog'], case['hardware'], phase_id='A_return')
        self.assertEqual(deployment_key(b.final_deployment), deployment_key(case['initial']))
        count = history.accepted_switch_count
        repeat = adapter.adapt(b.final_deployment, case['old'], case['old'], case['catalog'], case['hardware'], phase_id='A_return')
        self.assertEqual(repeat.action, 'NO_SWITCH')
        self.assertEqual(history.accepted_switch_count, count)
        self.assertEqual(repeat.runtime_evaluation_count, 1)
        self.assertEqual(history.last_deployment, b.final_deployment)
        with self.assertRaisesRegex(ValueError, 'new explicit phase_id'):
            adapter.adapt(b.final_deployment, case['old'], case['new'], case['catalog'], case['hardware'], phase_id='A_return')

    def test_dry_run_never_commits_repairs_or_mutates_history(self):
        case = scenarios()[2]
        history = AdaptationHistory()
        with patch.object(MilpDeploymentSeeder, 'solve', side_effect=AssertionError('planning should not seed')):
            result = DynamicAdaptation(NoRuntimeAllowed(), history=history).adapt(case['initial'], case['old'],
                case['new'], case['catalog'], case['hardware'], AdaptationConfig(dry_run=True))
        self.assertEqual(result.action, 'PLAN_ONLY')
        self.assertEqual(result.final_deployment, case['initial'])
        self.assertEqual(result.runtime_evaluation_count, 0)
        self.assertEqual(result.accepted_switches, [])
        self.assertIsNone(history.last_deployment)
        self.assertEqual(result.termination_reason, 'DRY_RUN_REPAIR_PLANNED')

    def test_invalid_initial_deployment_and_config_rejected(self):
        w, c, h = synthetic_pool(5)
        for x in ({'unknown': 1}, {'base': True}, {'base': 2}, {'base': -1}):
            with self.assertRaises(ValueError):
                GuidedDeploymentSearch().search(w, c, h, SearchConfig(dry_run=True), initial_deployment=x)
        with self.assertRaisesRegex(ValueError, 'exceeds hardware memory'):
            GuidedDeploymentSearch().search(w, c, Hardware(.5), SearchConfig(dry_run=True), initial_deployment={'base': 1})
        for cfg in ({'max_runtime_evaluations': -1}, {'max_adaptation_iterations': True}, {'top_b': 0}):
            with self.assertRaises(ValueError):
                AdaptationConfig(**cfg)

    def test_cost_counts_path_not_net_and_has_no_fabricated_time(self):
        c = {'a': model('a', 2., text=100.), 'b': model('b', 3., text=100.)}
        cost = StructuralReconfigurationCost(c)
        outward = cost.estimate({'a': 1}, {'b': 1})
        back = cost.estimate({'b': 1}, {'a': 1})
        self.assertEqual(outward.added_vram_mib + back.added_vram_mib, 5.)
        self.assertEqual(outward.removed_vram_mib + back.removed_vram_mib, 5.)
        self.assertIsNone(outward.measured_switch_time_ms)
        self.assertEqual(cost.estimate({'a': 1}, {'a': 1}).models_added, 0)

    def test_predictor_none_or_failure_does_not_block_warm_start(self):
        case = scenarios()[1]
        class BadScorer:
            def predict(self, *args):
                raise RuntimeError('optional scorer unavailable')
        a = DynamicAdaptation(synthetic_objective(case['catalog']), predictor=None).adapt(case['initial'],
            case['old'], case['new'], case['catalog'], case['hardware'])
        b = DynamicAdaptation(synthetic_objective(case['catalog']), predictor=BadScorer()).adapt(case['initial'],
            case['old'], case['new'], case['catalog'], case['hardware'])
        self.assertEqual(a.final_deployment, b.final_deployment)


if __name__ == '__main__':
    unittest.main()
