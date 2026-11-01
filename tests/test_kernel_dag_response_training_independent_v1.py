"""Independent node-path, immutable-binding and low-memory rank checks."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from gsdeploy.resources import identity
from tests.test_kernel_resources import hardware
from tests.test_kernel_dag_response_training_v1 import fixture
from tools.predictor_validation_20261027 import kernel_dag_response_training_v1 as training
from tools.predictor_validation_20261027 import kernel_dag_response_fit_v1 as fitting
from tools.predictor_validation_20261027 import kernel_competition_features_v2 as features


def compile_paths(doc, caps):
    return training.compile_paths(doc, hardware(), caps, allow_cpu_fixture=True)


class IndependentKernelDagTrainingTests(unittest.TestCase):
    def test_full_hardware_definition_is_part_of_bundle_identity(self):
        doc, cap = fixture()
        first = hardware()
        second = copy.deepcopy(first)
        second['max_warps_per_sm'] *= 2
        b1, p1 = training.training_bundle(doc, first, [cap], allow_cpu_fixture=True)
        b2, p2 = training.training_bundle(doc, second, [cap], allow_cpu_fixture=True)
        self.assertNotEqual(p1['rows'][0]['matrix'].tolist(), p2['rows'][0]['matrix'].tolist())
        self.assertEqual(b1.get('hardware_sha256'), identity(first))
        self.assertEqual(b2.get('hardware_sha256'), identity(second))
        self.assertNotEqual(b1['bundle_sha256'], b2['bundle_sha256'])

    def test_bundle_snapshots_catalog_hardware_and_capture_before_fit(self):
        doc, cap = fixture()
        device = hardware()
        expected = identity(doc), identity(device), identity(cap)
        def fit_probe(prepared):
            doc['profiles'][0]['nodes'][0]['kernel']['grid'] = [1, 1, 1]
            device['max_warps_per_sm'] *= 2
            cap['kernels'][0]['end_ns'] += 123
            cap['capture_id'] = 'external-mutation-after-compile'
            return dict(numerical_qualification=False, reason='CPU_SNAPSHOT_PROBE')
        with patch.object(training, 'fit_paths', side_effect=fit_probe):
            bundle, prepared = training.training_bundle(doc, device, [cap], allow_cpu_fixture=True)
        self.assertEqual(bundle['kernel_catalog_sha256'], expected[0])
        self.assertEqual(bundle.get('hardware_sha256'), expected[1])
        self.assertEqual(bundle['capture_identities'][0]['sha256'], expected[2])
        self.assertEqual(prepared['full_offered_request_registry'][0]['capture_id'], 'capture0')
        self.assertFalse(bundle['native_sources_authenticated'])

    def test_invalid_catalog_role_is_rejected_before_copying_kernel_labels(self):
        class UnreadLabels:
            def __deepcopy__(self, memo):
                raise AssertionError('timing labels copied before catalog role gate')
        for role in ('training_only', 'validation_labels_read'):
            doc, cap = fixture()
            doc[role] = role == 'validation_labels_read'
            cap['kernels'] = UnreadLabels()
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, '[Tt]raining|[Rr]ole'):
                training.training_bundle(doc, hardware(), [cap], allow_cpu_fixture=True)

    def test_randomized_piecewise_paths_match_direct_half_open_active_sets(self):
        rng = np.random.default_rng(437)
        for _ in range(60):
            doc, cap = fixture()
            for launch in cap['kernels']:
                launch['start_ns'] = int(rng.integers(0, 12))
                launch['end_ns'] = launch['start_ns'] + int(rng.integers(1, 10))
            result = compile_paths(doc, [cap])
            launches = {(k['request_id'], k['node_id']): k for k in cap['kernels']}
            for observation in result['observations']:
                target = launches[observation['request_id'], observation['node_id']]
                path = observation['path']
                self.assertEqual(path[0]['left_ns'], target['start_ns'])
                self.assertEqual(path[-1]['right_ns'], target['end_ns'])
                for previous, current in zip(path, path[1:]):
                    self.assertEqual(previous['right_ns'], current['left_ns'])
                for piece in path:
                    stamp = piece['left_ns']
                    peers = [training.static_kernel(k['kernel']) for k in cap['kernels']
                             if k is not target and k['start_ns'] <= stamp < k['end_ns']]
                    oracle = features.competition_features(training.static_kernel(target['kernel']), peers, hardware())
                    self.assertEqual(piece['vector'], tuple(oracle['vector'][f] for f in features.FEATURES))
            for row in result['rows']:
                self.assertAlmostEqual(sum(row['ordered_durations']), row['response_ms'], places=15)
                self.assertAlmostEqual(sum(row['durations']), row['response_ms'], places=15)

    def test_same_name_same_owner_peer_contributes_to_all_six_feature_mapper(self):
        doc, cap = fixture()
        doc['profiles'][0]['nodes'][1]['kernel']['name'] = 'x'
        cap['kernels'][1]['kernel']['name'] = 'x'
        result = compile_paths(doc, [cap])
        piece = next(p for o in result['observations'] if o['node_id'] == 'x'
                     for p in o['path'] if p['left_ns'] == 20_000_000)
        self.assertEqual(piece['vector'][:2], (1, 1))
        self.assertEqual(len(piece['vector']), 6)

    def test_each_capture_has_an_independent_timeline_not_cross_capture_peers(self):
        doc, cap = fixture()
        other = copy.deepcopy(cap)
        other.update(capture_id='capture1', raw_sha256='b'*64)
        for launch in other['kernels']:
            launch['clock_domain'] = 'second-independent-session'
        result = compile_paths(doc, [cap, other])
        self.assertEqual((result['accepted_kernel_count'], result['offered_kernel_count']), (6, 6))
        left = [o['path'] for o in result['observations'] if o['capture_id'] == 'capture0']
        right = [o['path'] for o in result['observations'] if o['capture_id'] == 'capture1']
        self.assertEqual(left, right)
        self.assertEqual(len(result['full_offered_request_registry']), 4)

    def test_mixed_clock_capture_keeps_full_unknown_denominator_alongside_valid_capture(self):
        doc, cap = fixture()
        other = copy.deepcopy(cap)
        other.update(capture_id='capture1', raw_sha256='b'*64)
        other['kernels'][0]['clock_domain'] = 'unresolved-session'
        result = compile_paths(doc, [cap, other])
        self.assertEqual((result['offered_kernel_count'], result['accepted_kernel_count']), (6, 3))
        self.assertEqual(len(result['rejected_rows']), 3)
        self.assertEqual({r['node_id'] for r in result['rejected_rows']}, {'x', 'y', 'z'})
        with patch.object(fitting, 'fit', side_effect=AssertionError('incomplete cohort entered solver')):
            self.assertEqual(training.fit_paths(result)['reason'], 'INCOMPLETE_SUPPORTED_NODE_COHORT')

    def test_reused_raw_is_rejected_even_when_capture_and_request_ids_change(self):
        doc, cap = fixture()
        other = copy.deepcopy(cap)
        other['capture_id'] = 'reexport'
        for row in other['requests']:
            row['request_id'] += '-other'
        other['offered_request_ids'] = [r['request_id'] for r in other['requests']]
        for launch in other['kernels']:
            launch['request_id'] += '-other'
        with self.assertRaisesRegex(ValueError, 'raw identities'):
            compile_paths(doc, [cap, other])

    def test_every_offered_request_requires_a_binding_and_canonical_node_cohort(self):
        doc, cap = fixture()
        cap['offered_request_ids'].append('dropped-offered-request')
        with self.assertRaisesRegex(ValueError, 'offered request registry'):
            compile_paths(doc, [cap])


class IndependentLowMemoryCertificateTests(unittest.TestCase):
    def test_random_dense_checks_bound_and_never_false_qualify(self):
        rng = np.random.default_rng(901)
        for _ in range(120):
            n = int(rng.integers(8, 55))
            g = int(rng.integers(1, min(n, 15)+1))
            width = int(rng.integers(1, 7))
            labels = np.concatenate((np.arange(g), rng.integers(0, g, n-g)))
            means = rng.normal(size=(g, width))*float(rng.choice([1e-8, 1, 1e8]))
            derivative = means[labels] + rng.normal(size=(n, width))
            rows = [dict(group=int(label)) for label in labels]
            z = np.eye(g)[labels]
            dense = np.concatenate((z, derivative), axis=1)
            singular = np.linalg.svd(dense, compute_uv=False)
            cert = fitting.nuisance_certificate(rows, derivative)
            scale = max(1, float(singular[0]))
            self.assertLessEqual(cert['full_jacobian_min_singular_lower_bound'], singular[-1]+scale*1e-12)
            self.assertGreaterEqual(cert['full_jacobian_max_singular_upper_bound'], singular[0]-scale*1e-12)
            if cert['full_column_rank']:
                self.assertEqual(np.linalg.matrix_rank(dense), g+width)

    def test_one_observation_per_node_cannot_identify_six_slopes(self):
        rng = np.random.default_rng(123)
        rows = [dict(group=('node', i)) for i in range(6000)]
        derivative = rng.normal(size=(6000, 6))
        with patch.object(fitting.np, 'column_stack', side_effect=AssertionError('dense dummy allocation')):
            cert = fitting.nuisance_certificate(rows, derivative)
        self.assertFalse(cert['full_column_rank'])
        self.assertEqual(cert['projected_derivative_rank'], 0)
        self.assertFalse(cert['dense_nuisance_matrix_allocated'])

    def test_root_certificates_cannot_override_confounded_node_nuisance(self):
        rows = [dict(group=('a',), capture='c') for _ in range(16)]
        solved = type('Solved', (), dict(x=np.zeros(6), success=True, message='CPU solver stub'))()
        local = dict(stationary_strict_local_minimum=True, strictly_positive_curvature=True)
        with patch.object(fitting, 'minimize', return_value=solved), \
             patch.object(fitting, 'root', return_value=solved), \
             patch.object(fitting, 'differential', return_value=(1., np.zeros(6), np.eye(6))), \
             patch.object(fitting, 'local_minimum_certificate', return_value=local), \
             patch.object(fitting, 'profile', return_value=(16., {('a',): 0.}, 1., np.ones((16, 6)), np.zeros(16))):
            info, _, _, _ = fitting.fit(rows, list(features.FEATURES))
        self.assertFalse(info['numerical_qualification'])
        self.assertEqual(info['nuisance_adjusted_derivative_rank'], 0)
        self.assertEqual(len(info['gradient_root_attempts']), 3)

    def test_original_profile_differential_and_local_certificate_are_shared(self):
        from tools.predictor_validation_20261027.stage_response_fit_v1 import profile
        from tools.predictor_validation_20261027.macro_joint_response_second_order_v1 import differential, local_minimum_certificate
        self.assertIs(fitting.profile, profile)
        self.assertIs(fitting.differential, differential)
        self.assertIs(fitting.local_minimum_certificate, local_minimum_certificate)


if __name__ == '__main__':
    unittest.main()
