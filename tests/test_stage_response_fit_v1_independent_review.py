"""Independent mathematical and provenance checks; CPU training only."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import ndtri

from tools.predictor_validation_20261027 import stage_response_fit_v1 as f
from tools.predictor_validation_20261027.stage_response_features_v1 import receipt
from tests.test_stage_response_fit_v1 import stage

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'artifacts/predictor-profile-training-evidence-v1-natural-20261028-review02'
FEATURES = DIRECTORY / 'stage-training-v2.json'


def dynamic_data(beta=(.35, -.2), n=31):
    beta = np.asarray(beta)
    rows = []
    schedules = [((0, 0), (1, 0), 4), ((1, 0), (0, 0), 4),
                 ((1, 1), (1, 0), 5), ((2, 0), (1, 1), 6),
                 ((0, 0), (2, 1), 7), ((2, 1), (0, 0), 3)]
    for first, second, switch_ms in schedules:
        rate1 = math.exp(-float(beta @ first))
        rate2 = math.exp(-float(beta @ second))
        for z in ndtri((np.arange(n) + .5) / n):
            reference = 10 * math.exp(.12 * z)
            duration = (reference / rate1 if reference <= switch_ms * rate1
                        else switch_ms + (reference - switch_ms * rate1) / rate2)
            row = stage(duration, *first)
            if duration > switch_ms:
                piece = row['full_host_wall_covariate_path'][0]
                begin, end = piece['left_ns'], piece['right_ns']
                cutoff = begin + round(switch_ms * 1e6)
                piece.update(right_ns=cutoff, duration_ns=cutoff - begin)
                row['full_host_wall_covariate_path'].append(dict(
                    left_ns=cutoff, right_ns=end, duration_ns=end - cutoff,
                    state=dict(cutoff_processing_ns=cutoff,
                               peer_visible_unconfirmed_owner_count=second[0],
                               peer_same_semantic_phase_owner_count=second[1]),
                    each_state_uses_only_its_processed_prefix=True))
            rows.append(row)
    return rows


class IndependentFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dynamic = dynamic_data()
        cls.doc = json.loads(FEATURES.read_text())

    def test_direct_lognormal_transformed_density_matches_profile(self):
        rows = f.prepared(self.dynamic, f.FEATURES)
        beta = np.asarray([.21, -.11])
        nll, mu, variance, _, _ = f.profile(beta, rows)
        expected = 0.
        for row in rows:
            virtual = sum(float(dt) * math.exp(-float(x @ beta))
                          for dt, x in zip(row['durations'], row['matrix']))
            z = math.log(virtual) - mu[row['group']]
            density = math.exp(-.5 * z * z / variance) / (virtual * math.sqrt(2 * math.pi * variance))
            density *= math.exp(-float(row['matrix'][-1] @ beta))
            expected -= math.log(density)
        self.assertAlmostEqual(nll, expected, places=9)

    def test_actual_optimizer_callback_gradient_matches_central_difference(self):
        real_minimize = f.minimize
        probes = []

        def checked_minimize(objective, start, **kwargs):
            beta = np.asarray([.2, -.1])
            _, gradient = objective(beta)
            step = 1e-6
            numerical = np.asarray([(objective(beta + np.eye(2)[j] * step)[0]
                                     - objective(beta - np.eye(2)[j] * step)[0]) / (2 * step)
                                    for j in range(2)])
            np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=2e-6)
            probes.append(start)
            return real_minimize(objective, start, **kwargs)

        with patch.object(f, 'minimize', side_effect=checked_minimize):
            result = f.fit(self.dynamic, 'visible_phase_aware')
        self.assertEqual(len(probes), 3)
        np.testing.assert_allclose(result['coefficients'], [.35, -.2], atol=1e-3)
        self.assertFalse(result['prospective_predictor_qualified'])

    def test_constant_count_two_coefficients_and_acceleration_recovery(self):
        rows = []
        for count, same in ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1)):
            for noise in (-.08, .08):
                rows.append(stage(10 * math.exp(.3 * count - .2 * same + noise), count, same))
        result = f.fit(rows, 'visible_phase_aware')
        np.testing.assert_allclose(result['coefficients'], [.3, -.2], atol=1e-5)
        self.assertAlmostEqual(result['lognormal_reference_sigma'], .08, places=6)

    def test_state_rank_two_can_be_completely_confounded_by_baselines(self):
        rows = []
        for index, (count, same) in enumerate(((0, 0), (1, 0), (1, 1))):
            for noise in (-.08, .08):
                rows.append(stage(10 * math.exp(.2 * count + noise), count, same, index=index))
        data = f.prepared(rows, f.FEATURES)
        state_matrix = np.vstack([r['matrix'] for r in data])
        self.assertEqual(np.linalg.matrix_rank(state_matrix), 2)
        nlls = [f.profile(np.asarray(beta), data)[0] for beta in ((0., 0.), (.7, -.9), (-.3, .2))]
        np.testing.assert_allclose(nlls, nlls[0], atol=1e-10)
        result = f.fit(rows, 'visible_phase_aware')
        jacobian = result['nonlinear_joint_response_jacobian']
        self.assertEqual(jacobian['rank'], 3)
        self.assertEqual(jacobian['parameters'], 5)
        self.assertFalse(jacobian['full_column_rank'])

    def test_noiseless_reference_likelihood_is_explicitly_degenerate(self):
        rows = [stage(10, 0), stage(10, 0)]
        with self.assertRaisesRegex(ValueError, 'zero-variance'):
            f.fit(rows, 'baseline')

    def test_capture_count_cannot_be_replaced_by_many_phase_rows(self):
        result = f.fit(self.dynamic, 'baseline')
        self.assertEqual(result['host_phase_rows'], 186)
        self.assertEqual(result['whole_capture_blocks'], 1)
        self.assertFalse(result['minimum_initial_three_capture_design_met'])
        self.assertFalse(result['capture_robust_parameter_uncertainty_estimated'])

    def test_duplicate_actual_capture_and_mixed_domains_refused(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate capture'):
            f.run([FEATURES, FEATURES])
        with tempfile.TemporaryDirectory() as folder:
            doc = copy.deepcopy(self.doc)
            doc['mode'] = 'kernel'
            path = Path(folder) / 'kernel-features.json'
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'Single timing domain'):
                f.run([FEATURES, path])

    def test_mutated_feature_path_rejected_by_complete_rederivation(self):
        with tempfile.TemporaryDirectory() as folder:
            doc = copy.deepcopy(self.doc)
            forward = next(s for s in doc['stages'] if s['stage_kind'] == 'language_forward')
            forward['full_host_wall_covariate_path'][0]['state']['peer_visible_unconfirmed_owner_count'] += 1
            path = Path(folder) / 'tampered-features.json'
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'rederive exactly'):
                f.run([path])

    def test_transitive_review_source_drift_rejected_without_editing_actual_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            review = json.loads((DIRECTORY / 'report.json').read_text())
            dependency = folder / 'review-dependency.py'
            dependency.write_text('original')
            review['sources'].append(receipt(dependency))
            review_path = folder / 'review.json'
            review_path.write_text(json.dumps(review))
            doc = copy.deepcopy(self.doc)
            doc['sources'][0] = receipt(review_path)
            dependency.write_text('changed')
            path = folder / 'features.json'
            path.write_text(json.dumps(doc))
            # Every direct feature source still matches. Only recursive
            # qualification finds the changed dependency in its review.
            self.assertTrue(all(receipt(ref['path']) == ref for ref in doc['sources']))
            with self.assertRaisesRegex(ValueError, 'review source changed'):
                f.run([path])


if __name__ == '__main__':
    unittest.main()
