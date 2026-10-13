"""Leakage and scientific-accounting regression tests (CPU only)."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from performance_model.profiling.plan import ROOT, extractor
from .features import FoldPreprocessor, planning_vector
from .evaluate import metrics, split, training_weights, strict_eligible


class GateTests(unittest.TestCase):
    def test_train_only_preprocessing_and_missing(self):
        p = FoldPreprocessor().fit([{'a': 1., 'b': None}, {'a': 3., 'b': None}], ['a', 'b'])
        before = p.means.copy()
        np.testing.assert_allclose(p.transform([{'a': 101., 'b': 900.}]), [[99.]])
        np.testing.assert_allclose(p.means, before)
        self.assertEqual(p.audit([{'a': None, 'b': 2.}])['all_missing_training_features'], ['b'])
        self.assertEqual(p.audit([{'a': None, 'b': 2.}])['heldout_missing_values'], 1)
        np.testing.assert_allclose(p.transform([{'a': None, 'b': 2.}]), [[0.]])

    def test_planning_projection_ignores_runtime_and_group_fields(self):
        row = json.loads((ROOT / 'results/performance_model_active_sampling/dataset_rows.jsonl').read_text().splitlines()[0])
        ex = extractor()
        keys = sorted({k for p in ex.resource_profiles.values() for k in p['values']})
        expected = planning_vector(row, ex, 'C', keys)[0]
        changed = copy.deepcopy(row)
        for key in ('labels', 'system_labels', 'stability', 'deployment_id', 'configuration_id', 'run_id',
                    'repetition_id', 'colocation_pressure', 'model_features', 'standalone_service_demand'):
            changed[key] = 'intentionally corrupted observed/audit value'
        self.assertEqual(planning_vector(changed, ex, 'C', keys)[0], expected)
        self.assertFalse(any('deployment_id' in k or 'queue' in k or 'completion' in k for k in expected))

    def test_missing_s0_never_filled(self):
        row = json.loads((ROOT / 'results/performance_model_active_sampling/dataset_rows.jsonl').read_text().splitlines()[0])
        ex = extractor()
        row['task'] = 'nonexistent_task'
        with self.assertRaisesRegex(ValueError, 'standalone service time'):
            planning_vector(row, ex, 'C', [])

    def test_lodo_keeps_all_repetitions_out(self):
        rows = [dict(deployment_id=d, configuration_id='cfg'+d, run_id=d+str(rep), repetition_id=rep)
                for d in ('a', 'b') for rep in (1, 2, 3)]
        train, test, overlap = split(rows, 'a')
        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 3)
        self.assertTrue(all(not x for x in overlap.values()))
        rows[-1]['configuration_id'] = 'cfga'
        with self.assertRaisesRegex(ValueError, 'leakage'):
            split(rows, 'a')

    def test_config_run_path_balancing(self):
        rows = [dict(configuration_id='a', run_id='a1')] * 3 + [dict(configuration_id='a', run_id='a2')]
        rows += [dict(configuration_id='b', run_id='b1')]
        w = training_weights(rows)
        self.assertAlmostEqual(w[:4].sum(), w[4])
        self.assertAlmostEqual(w[:3].sum(), w[3])
        self.assertAlmostEqual(w.sum(), len(rows))

    def test_metrics_and_normalized_restoration(self):
        s0, y = np.array([10., 20.]), np.array([100., 500.])
        np.testing.assert_allclose(s0 * np.exp(np.log(y / s0)), y)
        m = metrics([10, 20], [10, 30])
        self.assertEqual(m['mae_ms'], 5)
        self.assertEqual(m['median_absolute_error_ms'], 5)
        self.assertEqual(m['p90_absolute_error_ms'], 9)
        self.assertEqual(m['smape_percent'], 20)
        self.assertIsNone(metrics([10, 20], [10, 10])['spearman'])

    def test_strict_does_not_redefine_primary_label(self):
        r = dict(stability=dict(stable=True, category='boundary'), role='formal', labels=dict(completed=30))
        self.assertFalse(strict_eligible(r))
        self.assertTrue(r['stability']['stable'])
        r['stability']['category'] = 'stable'
        self.assertTrue(strict_eligible(r))
        r['labels']['completed'] = 29
        self.assertFalse(strict_eligible(r))


@unittest.skipUnless((ROOT / 'results/performance_model_generalization/leakage_audit.json').exists(), 'Run offline evaluation first')
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ROOT / 'results/performance_model_generalization'
        cls.audit = json.loads((folder / 'leakage_audit.json').read_text())
        cls.scores = json.loads((folder / 'fold_metrics.json').read_text())
        cls.predictions = [json.loads(s) for s in (folder / 'predictions.jsonl').read_text().splitlines()]
        cls.rows = [json.loads(s) for s in Path(cls.audit['dataset_path']).read_text().splitlines()]

    def test_every_prediction_is_unique_stable_and_truly_heldout(self):
        index = {r['run_id'] + '::' + r['task'] + '::' + r['target_model']: r for r in self.rows}
        seen = set()
        for p in self.predictions:
            key = (p['subset'], p['variant'], p['row_id'])
            self.assertNotIn(key, seen)
            seen.add(key)
            r = index[p['row_id']]
            self.assertIs(r['stability']['stable'], True)
            self.assertEqual(p['heldout_deployment'], r['deployment_id'])
            if p['subset'] == 'strict': self.assertTrue(strict_eligible(r))
            if p['variant'].endswith('log_ratio'):
                self.assertAlmostEqual(p['predicted_latency_ms'], p['standalone_latency_ms'] * np.exp(p['raw_model_output']))
        for subset in ('primary', 'strict'):
            for variant in self.scores['subsets'][subset]['macro']:
                self.assertEqual(sum(p['subset'] == subset and p['variant'] == variant for p in self.predictions),
                                 self.scores['subsets'][subset]['rows'])

    def test_actual_fit_records_exclude_all_heldout_runs(self):
        for f in self.audit['folds']:
            heldout_all = {r['run_id'] for r in self.rows if r['deployment_id'] == f['heldout_deployment']}
            self.assertFalse(heldout_all & set(f['train']['run_id']))
            for k in ('deployment_id', 'configuration_id', 'run_id'):
                self.assertFalse(set(f['train'][k]) & set(f['test'][k]))
            heldout_paths = {r['source_result_path'] for r in self.rows if r['run_id'] in heldout_all}
            self.assertFalse(heldout_paths & set(f['piecewise_training_source_paths']))
            train_rows = [r for r in self.rows if r['run_id'] in f['train']['run_id'] and
                          r['run_id'] + '::' + r['task'] + '::' + r['target_model'] in f['fit_row_ids']]
            # u has no missing values: saved median and scaling must equal training rows only.
            us = np.array([r['standalone_service_demand'] for r in train_rows])
            for level in ('B', 'C'):
                prep = f['preprocessors'][level]
                self.assertAlmostEqual(prep['training_medians']['u'], np.median(us))
                j = prep['transformed_feature_order'].index('u')
                self.assertAlmostEqual(prep['training_means'][j], us.mean())
                self.assertAlmostEqual(prep['training_scales'][j], us.std())

    def test_reported_fold_metrics_match_predictions(self):
        for subset, result in self.scores['subsets'].items():
            for f in result['folds']:
                for variant, expected in f['models'].items():
                    ps = [p for p in self.predictions if p['subset'] == subset and p['variant'] == variant and
                          p['heldout_deployment'] == f['heldout_deployment']]
                    actual = metrics([p['true_latency_ms'] for p in ps], [p['predicted_latency_ms'] for p in ps])
                    for k in ('mae_ms', 'median_absolute_error_ms', 'smape_percent', 'p90_absolute_error_ms', 'spearman'):
                        if actual[k] is None: self.assertIsNone(expected[k])
                        else: self.assertAlmostEqual(actual[k], expected[k])


if __name__ == '__main__':
    unittest.main()
