"""CPU regression tests for count semantics, integer optimization and audit gaps."""
from dataclasses import asdict
import unittest

from deployment_core import Deployment, Hardware, Workload, MilpDeploymentSeeder
from deployment_core.candidate_generator import deployment_memory_mib
from deployment_core.candidate_pool_v2 import (load_candidate_pool_v2, missing_demands,
                                               mandatory_presence, MissingServiceDemand)
from deployment_core.milp_seeder import validate_seed
from deployment_core.replicas import (normalize_counts, derive_replica_bounds, replica_key,
                                      replica_identities, replica_statistics)
from deployment_core.routing_solver import solve_capacity_lp
from deployment_search import IntegerGuidedDeploymentSearch, SearchConfig, solve_canonical_routing
from deployment_search.feedback import bottleneck_feedback
from deployment_search.neighborhood import generate_replica_neighborhood, memory_mib
from deployment_search.runtime import SyntheticRuntimeEvaluator
from deployment_search.integer_foundation import (model, runtime, case_a, case_b, case_c, case_d,
                                                 scalable_case, binary_compatibility, oracle_comparison)
from deployment_search.tests.fixtures import evaluation
from deployment_search.tests.small_integer_oracle import small_integer_oracle


class CountSchemaTests(unittest.TestCase):
    def test_invalid_and_sparse_counts(self):
        names = {'G', 'S'}
        self.assertEqual(normalize_counts({'S': 2}, names), {'G': 0, 'S': 2})
        for value in (-1, True, 1., '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_counts({'S': value}, names)
        with self.assertRaises(ValueError):
            normalize_counts({'unknown': 1}, names)
        with self.assertRaises(ValueError):
            normalize_counts({'S': 3}, names, {'G': 1, 'S': 2})
        with self.assertRaises(ValueError):
            normalize_counts({'S': 1}, names, {'S': 2})

    def test_count_sensitive_identity_and_legacy_hash(self):
        self.assertNotEqual(replica_key({'S': 1}), replica_key({'S': 2}))
        self.assertEqual(replica_key({'G': 0, 'S': 2}), replica_key({'S': 2}))
        self.assertEqual(replica_identities({'G': 0, 'S': 2}), {'S': ['S#0', 'S#1']})
        self.assertEqual(len({Deployment(('S',)), Deployment({'S': 1}), Deployment({'S': 2})}), 2)
        counts = replica_statistics({'G': 2, 'S': 3}, {'G': 'generalist', 'S': 'specialist'})
        self.assertEqual((counts['N_G'], counts['N_S']), (2, 3))
        self.assertEqual((counts['active_generalist_types'], counts['active_specialist_types']), (1, 1))

    def test_memory_physical_bounds_and_explicit_worker_limit(self):
        catalog = {'G': model('G', 3., text=100.), 'S': model('S', 2., text=100.)}
        bounds, refs = derive_replica_bounds(catalog, Hardware(11.), worker_limits={'S': 2})
        self.assertEqual(bounds, {'G': 3, 'S': 2})
        self.assertEqual(refs['S']['R_mem'], 5)
        self.assertEqual(refs['S']['worker_process_limit'], 2)
        x = {'G': 1, 'S': 3}
        self.assertEqual(memory_mib(x, catalog), 9.)
        self.assertEqual(deployment_memory_mib(Deployment(x), catalog), 9.)
        with self.assertRaises(ValueError):
            IntegerGuidedDeploymentSearch().search(Workload({'text': .1}), catalog, Hardware(11.),
                SearchConfig(dry_run=True), replica_bounds={'G': 4, 'S': 5})

    def test_integer_table_evaluator_never_aliases_replica_counts(self):
        evaluator = SyntheticRuntimeEvaluator({replica_key({'S': 1}): evaluation(20),
                                               replica_key({'S': 2}): evaluation(10)}, integer_mode=True)
        w = Workload({'text': 1.})
        self.assertEqual(evaluator.evaluate({'S': 1}, w, {}).weighted_mean_e2e, 20)
        self.assertEqual(evaluator.evaluate({'S': 2}, w, {}).weighted_mean_e2e, 10)
        self.assertEqual(len(set(evaluator.calls)), 2)


class IntegerPlanningTests(unittest.TestCase):
    def test_lexicographic_memory_then_count(self):
        w, hardware = Workload({'text': 1.2}), Hardware(6.)
        for general_memory, expected in [(2., {'G': 1, 'S': 0}), (2.1, {'G': 0, 'S': 2})]:
            with self.subTest(general_memory=general_memory):
                catalog = {'G': model('G', general_memory, text=500.), 'S': model('S', 1., text=1000.)}
                bounds, _ = derive_replica_bounds(catalog, hardware)
                seed = MilpDeploymentSeeder().solve(catalog, w, hardware, binary_mode=False)
                self.assertEqual(seed.deployment, expected)
                self.assertEqual(seed.objective_value, 2.)
                self.assertEqual(seed.milp_call_count, 2)
                self.assertEqual(seed.secondary_objective_value, sum(expected.values()))
                self.assertEqual(validate_seed(seed, catalog, w, hardware, .8, bounds), [])
                serialized = asdict(seed)
                self.assertEqual(serialized['presence'], {m: int(n > 0) for m, n in expected.items()})
                self.assertEqual(serialized['replica_bounds'], bounds)

    def test_more_text_replicas_do_not_cover_visual_qa(self):
        catalog = {'S': model('S', 1., text=100.)}
        seed = MilpDeploymentSeeder().solve(catalog, Workload({'text': 1., 'visual_qa': .1}),
                                          Hardware(10.), binary_mode=False)
        self.assertFalse(seed.feasible)
        self.assertEqual(seed.milp_call_count, 1)
        self.assertIsNone(solve_canonical_routing({'S': 10}, Workload({'visual_qa': .1}),
                                                catalog, replica_bounds={'S': 10}))

    def test_per_replica_pressure_and_family_demand_closed_form(self):
        catalog = {'G': model('G', 1., text=1000.), 'S': model('S', 1., text=500.),
                   'absent': model('absent', 1., text=1.)}
        w, x = Workload({'text': 1.2}), {'G': 2, 'S': 1, 'absent': 0}
        route = solve_canonical_routing(x, w, catalog, replica_bounds={m: 3 for m in catalog})
        # G provides 2/s and S provides 2/s at unit average pressure: z=1.2/4.
        self.assertAlmostEqual(route.max_pressure, .3)
        self.assertAlmostEqual(route.routing['text', 'G'], .6)
        self.assertAlmostEqual(route.routing['text', 'S'], .6)
        self.assertEqual(route.routing['text', 'absent'], 0.)
        self.assertAlmostEqual(route.task_contributions['text', 'G'], .6)
        self.assertAlmostEqual(route.model_pressure['G'], .3)
        self.assertAlmostEqual(route.capacity_slack['G'], 1.)
        self.assertIsNotNone(solve_capacity_lp(Deployment({'G': 2}), catalog, Workload({'text': 1.2}), .8))
        self.assertIsNone(solve_capacity_lp(Deployment({'G': 1}), catalog, Workload({'text': 1.2}), .8))

    def test_neighborhood_one_count_step_and_both_presence_transitions(self):
        catalog = {'G': model('G', 2., text=1000.), 'S': model('S', 1., text=500.)}
        w, x, bounds = Workload({'text': 1.2}), {'G': 2, 'S': 0}, {'G': 3, 'S': 4}
        route = solve_canonical_routing(x, w, catalog, replica_bounds=bounds)
        mutations = generate_replica_neighborhood(x, catalog, w, bottleneck_feedback(route, x), bounds)
        self.assertLessEqual(len(mutations), 2 * len(catalog))
        self.assertEqual({(c.operation, c.model) for c in mutations},
                         {('SCALE_UP', 'G'), ('SCALE_DOWN', 'G'), ('SCALE_UP', 'S')})
        self.assertTrue(all(sum(abs(c.deployment[m] - x[m]) for m in x) == 1 for c in mutations))

    def test_infeasible_warm_state_never_evaluates_runtime(self):
        evaluator = runtime(lambda _: self.fail('infeasible runtime evaluation'))
        result = IntegerGuidedDeploymentSearch(evaluator).search(Workload({'text': 1.2}),
            {'S': model('S', 1., text=1000.)}, Hardware(4.), initial_deployment={'S': 1})
        self.assertEqual(result.status, 'INITIAL_REPLICA_STATE_INFEASIBLE')
        self.assertEqual(result.runtime_evaluation_count, 0)


class ValidationCampaignTests(unittest.TestCase):
    def test_A_B_C_D(self):
        a, _ = case_a()
        self.assertTrue(a['passed'])
        self.assertEqual(len(case_b()), 3)
        self.assertTrue(case_c()['passed'])
        self.assertTrue(case_d()['passed'])

    def test_12_20_30_no_enumeration_and_fixed_runtime_budget(self):
        for size in (12, 20, 30):
            with self.subTest(size=size):
                evidence, result = scalable_case(size)
                self.assertLessEqual(max(evidence['candidate_count_per_iteration']), 2 * size)
                self.assertLessEqual(result.runtime_evaluation_count, 17)
                self.assertFalse(result.exhaustive_enumeration_used)

    def test_prechange_binary_traces(self):
        self.assertTrue(binary_compatibility()['passed'])

    def test_small_oracle_gap_and_guard(self):
        evidence = oracle_comparison()
        self.assertEqual(evidence['checked_vectors'], 27)
        self.assertGreaterEqual(evidence['relative_gap'], 0.)
        catalog = {str(i): model(str(i), 1., text=100.) for i in range(4)}
        with self.assertRaises(ValueError):
            small_integer_oracle(Workload({'text': 1.}), catalog, Hardware(8.),
                                 {m: 2 for m in catalog}, runtime(lambda _: 1.))

    def test_frozen_pool_pending_demand_and_mandatory_qwen(self):
        records, hardware = load_candidate_pool_v2()
        self.assertEqual(len(records), 12)
        self.assertEqual(sum(p.memory_mib for p in records.values()), 63140)
        self.assertEqual(hardware.gpu_memory_mib, 49140)
        bounds, refs = derive_replica_bounds(records, hardware)
        self.assertEqual(bounds['qwen'], 2)
        self.assertEqual(refs['qwen']['memory_source']['field'], 'models.qwen.measured_idle_vram_mib')
        w = Workload({k: .1 for k in ('text', 'image', 'ocr', 'asr', 'visual_qa', 'lid')})
        gaps = missing_demands(records, w)
        self.assertEqual((len(gaps), len({v['model'] for v in gaps})), (14, 7))
        self.assertEqual(mandatory_presence(records, w)[0]['model'], 'qwen')
        for name in ('phi4mm', 'minicpm', 'convnext_tiny', 'roberta_sst2'):
            with self.subTest(name=name), self.assertRaises(MissingServiceDemand):
                records[name].profile()


if __name__ == '__main__':
    unittest.main()
