"""CPU-only adapter/cache/failure/noise tests. Fake runner never launches a worker."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from deployment_core.workload import Workload
from deployment_search import GuidedDeploymentSearch, SearchConfig
from deployment_search.measurement_runtime import MeasurementRuntimeEvaluator, failure_raw, stats
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_search.runtime_cache import save_json, digest
from deployment_search.tests.fixtures import evaluation, model, synthetic_pool, synthetic_evaluator, forbid_enumeration

SETTINGS = dict(warmup_seconds=10, measurement_seconds=210, routing_source='canonical_minmax',
                startup_timeout_seconds=300, drain_timeout_seconds=60, shutdown_timeout_seconds=10,
                max_pending_per_model=10000)


class RealRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.catalog = {'a': model('a', text=10.)}
        self.workload = Workload({'text': 1.})
        self.calls = []

    def fake_runner(self, config, raw_path, log_path, config_path):
        self.calls.append(config)
        save_json(raw_path, failure_raw(config, 'injected worker error; no GPU execution'))
        return dict(returncode=1)

    def evaluator(self, **kwargs):
        return MeasurementRuntimeEvaluator(self.tmp.name, self.catalog, {'models': 'abc'}, SETTINGS,
                                           runner=self.fake_runner, **kwargs)

    def test_cache_stores_three_failed_runs_without_retry_and_survives_restart(self):
        ev = self.evaluator()
        r = ev.evaluate({'a': 1}, self.workload, {('text', 'a'): 1.})
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(r.stability_state_v2, 'unstable')
        self.assertEqual(r.weighted_mean_e2e, None)
        self.assertEqual(len(r.raw_result_paths), 3)
        r2 = self.evaluator().evaluate({'a': 1}, self.workload, {('text', 'a'): 1.})
        self.assertEqual(r, r2)
        self.assertEqual(len(self.calls), 3)
        data = json.loads(ev.cache.path.read_text())
        self.assertEqual(data['cache_hits'], 1)
        self.assertEqual(len(data['run_attempts']), 3)

    def test_cache_key_covers_every_required_component(self):
        identity = self.evaluator().identity({'a': 1}, self.workload, {('text', 'a'): 1.})
        for field in ('deployment', 'workload', 'routing', 'measurement_settings', 'source_model_hashes'):
            other = dict(identity, **{field: {'different': True}})
            self.assertNotEqual(digest(identity), digest(other))

    def test_budget_is_durable_and_cache_hit_does_not_spend_budget(self):
        ev = self.evaluator(max_unique_runtime_evaluations=1)
        ev.evaluate({'a': 1}, self.workload, {('text', 'a'): 1.})
        ev.evaluate({'a': 1}, self.workload, {('text', 'a'): 1.})
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
            ev.evaluate({'a': 1}, Workload({'text': 2}), {('text', 'a'): 2.})
        self.assertEqual(len(self.calls), 3)

    def test_started_repetition_is_not_rerun_after_interruption(self):
        ev = self.evaluator()
        identity = ev.identity({'a': 1}, self.workload, {('text', 'a'): 1.})
        with ev.cache.locked() as cache:
            key, entry = cache.reserve(identity)
            entry['repetitions'].append(dict(index=1, seed=7101, status='started'))
            cache.data['run_attempts'].append(dict(cache_key=key, repetition=1))
            cache.flush()
        result = ev.evaluate({'a': 1}, self.workload, {('text', 'a'): 1.})
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(result.repetitions), 3)
        self.assertEqual(result.repetitions[0]['stability_state_v2'], 'unstable')

    def test_tampered_cache_is_not_remeasured(self):
        ev = self.evaluator()
        result = ev.evaluate({'a': 1}, self.workload, {('text', 'a'): 1.})
        Path(result.raw_result_paths[0]).write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            ev.evaluate({'a': 1}, self.workload, {('text', 'a'): 1.})
        self.assertEqual(len(self.calls), 3)

    def test_noise_gate_stability_and_p95_cannot_bypass_mean_gate(self):
        obj = NoiseAwareRuntimeObjective(min_relative_improvement=.05)
        self.assertEqual(obj.compare(evaluation(99), evaluation(100), 1, 1),
                         (False, 'latency_gain_below_min_relative_improvement'))
        self.assertTrue(obj.compare(evaluation(95), evaluation(100), 1, 1)[0])
        self.assertFalse(obj.compare(evaluation(99.5, p95=100), evaluation(100, p95=200), 1, 1)[0])
        self.assertTrue(obj.compare(evaluation(1000), evaluation(1, 'recoverable_congested'), 1, 1)[0])
        self.assertTrue(obj.compare(evaluation(100), evaluation(100), 1, 2)[0])
        self.assertFalse(obj.compare(evaluation(50, 'unstable'), evaluation(100), 1, 1)[0])

    def test_real_raw_aggregation_matches_existing_module2g(self):
        # Replay existing immutable raw data through the real adapter's classification/aggregation.
        from deployment_core.models import load_model_catalog
        from deployment_search.measurement_runtime import summarize_repetition, aggregate_repetitions
        source = Path(__file__).resolve().parents[2] / 'results/performance_model_comparable_groups/comparison_groups.json'
        group = json.loads(source.read_text())['groups'][0]
        config = group['configurations'][0]
        w = Workload(group['workload'])
        reps = [summarize_repetition(json.loads(Path(r['source_result_path']).read_text()), w,
                                    load_model_catalog(), {'returncode': 0}, r['source_result_path'])
                for r in config['repetitions']]
        result = aggregate_repetitions(reps, 'offline', 'offline.json')
        self.assertEqual(result.stability_state_v2, 'steady')
        self.assertAlmostEqual(result.weighted_mean_e2e, config['observed_mean']['mean'])
        self.assertAlmostEqual(result.replicate_statistics['weighted_mean_e2e']['sample_std'],
                               config['observed_mean']['sample_std'])
        self.assertAlmostEqual(result.weighted_p95_e2e, config['observed_tail']['mean'])
        reps[2] = dict(reps[2], stability_state_v2='unstable', weighted_mean_e2e=None)
        result = aggregate_repetitions(reps, 'offline', 'offline.json')
        self.assertEqual(result.stability_state_v2, 'unstable')
        self.assertIsNone(result.weighted_mean_e2e)
        self.assertEqual(result.replicate_statistics['weighted_mean_e2e']['valid_repetitions'], 2)

    def test_search_hooks_preserve_non_enumeration_and_checkpoint(self):
        w, c, h = synthetic_pool(5)
        checkpoints = []
        with forbid_enumeration():
            r = GuidedDeploymentSearch(synthetic_evaluator(c), predictor=None,
                    runtime_objective=NoiseAwareRuntimeObjective(),
                    trace_callback=lambda result: checkpoints.append(result.to_dict())).search(
                        w, c, h, SearchConfig(top_b=2, max_iterations=3, no_improvement_patience=1,
                                             runtime_evaluation_budget=6))
        self.assertLessEqual(r.runtime_evaluation_count, 6)
        self.assertTrue(checkpoints)
        self.assertIn('current_runtime', checkpoints[-1]['iterations'][0])
        self.assertEqual(checkpoints[-1]['status'], r.status)

    def test_failed_measurements_complete_closed_loop_and_replay_audit(self):
        from deployment_search.real_pilot import decision_audit
        w, c, h = synthetic_pool(5)
        ev = MeasurementRuntimeEvaluator(self.tmp.name, c, {'models': 'fake'}, SETTINGS,
                                         runner=self.fake_runner)
        objective = NoiseAwareRuntimeObjective()
        with forbid_enumeration():
            result = GuidedDeploymentSearch(ev, runtime_objective=objective).search(
                w, c, h, SearchConfig(max_iterations=3, no_improvement_patience=1, runtime_evaluation_budget=6))
        self.assertEqual(result.status, 'NO_RUNTIME_IMPROVEMENT')
        self.assertEqual(result.runtime.stability_state_v2, 'unstable')
        self.assertEqual(len(self.calls), 9)
        manifest = json.loads(ev.cache.path.read_text())
        audit = decision_audit(result.to_dict(), manifest, c, objective)
        self.assertTrue(audit['passed'], audit['errors'])
        self.assertTrue(all(d['actual_decision'] == 'ROLLBACK' for d in audit['decisions']))


if __name__ == '__main__':
    unittest.main()
