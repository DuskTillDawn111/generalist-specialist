"""Independent CPU identity and train/predict mapping counterexamples.

These are declared synthetic domains, not authenticated native profiles.
No device access or numerical optimizer is needed for these contracts.
"""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from gsdeploy.forecasting.kernel_provider import KernelResourceProvider
from tests.test_kernel_competition_features_v2 import capture, provider
from tests.test_kernel_provider import ALIASES, active_rows
from tools.predictor_validation_20261027 import kernel_competition_features_v2 as m


class IndependentKernelV2Review(unittest.TestCase):
    def test_cross_device_is_rejected_even_without_temporal_overlap(self):
        p = provider()
        cap = capture(p, b_start=200)
        cap['kernels'][1]['device'] = 1
        with patch.object(m, 'competition_features') as math:
            with self.assertRaisesRegex(ValueError, 'Cross-device'):
                p.training_rows([cap], allow_cpu_fixture=True)
        math.assert_not_called()

    def test_execution_types_extra_keys_and_domain_rebinding_fail_before_math(self):
        p = provider()
        for execution in ({'batch_size': True, 'backend': 'synthetic', 'weight_reuse': False},
                          {'batch_size': 1, 'backend': 'synthetic', 'weight_reuse': 0},
                          {'batch_size': 1, 'backend': 'synthetic', 'weight_reuse': False,
                           'undeclared_batch_backend': 'other'}):
            cap = capture(p)
            cap['requests'][0]['execution'] = execution
            with self.subTest(execution=execution), patch.object(m, 'competition_features') as math:
                with self.assertRaisesRegex(ValueError, 'profile/execution'):
                    p.training_rows([cap], allow_cpu_fixture=True)
                math.assert_not_called()
        first, second = capture(p), capture(p, 1)
        second['hardware_binding']['hardware_id'] = 'different-hardware'
        with patch.object(m, 'competition_features') as math:
            with self.assertRaisesRegex(ValueError, 'device-to-hardware'):
                p.training_rows([first, second], allow_cpu_fixture=True)
        math.assert_not_called()

    def test_new_input_execution_profile_uses_same_prospective_static_mapping(self):
        base = provider()
        doc = copy.deepcopy(base.provider.document)
        alternate = copy.deepcopy(doc['profiles'][0])
        alternate.update(id='a-long-batch', pool_id='long-input',
                         execution=dict(batch_size=4, backend='batch-native', weight_reuse=True))
        alternate['segments'][0]['kernel'].update(name='a-long-batch:kernel',
            static_shared_bytes=32768, allocated_shared_bytes=32768)
        doc['profiles'].append(alternate)
        p = m.KernelCompetitionFeatureProvider(KernelResourceProvider(doc, 'context', allow_cpu_fixture=True))
        cap = capture(p)
        cap['requests'][0].update(pool_id='long-input', profile_id=alternate['id'],
                                  execution=copy.deepcopy(alternate['execution']))
        kernel = alternate['segments'][0]['kernel']
        cap['kernels'][0].update(profile_id=alternate['id'], name=kernel['name'],
                                 static_kernel=copy.deepcopy(kernel))
        prepared = p.training_rows([cap], allow_cpu_fixture=True)
        active = active_rows(p.provider)
        row = dict(alias='a', request_id='a-long', task='task', pool_id='long-input')
        plan = p.provider.plan(row, ALIASES, 'mean')
        segment = plan['segments'][0]
        row.update(profile_id=plan['profile_id'], segment_id=segment['id'], segment_kind=segment['kind'],
                   segment_duration_ms=segment['duration_ms'])
        active['a'] = row
        observed = prepared['observations'][0]['path'][-1]['features']
        self.assertEqual(observed, p.features(row, active, ALIASES, 'mean'))
        self.assertEqual(prepared['rows'][0]['group'][1], 'a-long-batch')
        cap['requests'][0]['execution']['batch_size'] = 1
        with self.assertRaisesRegex(ValueError, 'profile/execution'):
            p.training_rows([cap], allow_cpu_fixture=True)

    def test_touching_peer_replacement_changes_identity_without_double_counting(self):
        base = provider()
        doc = copy.deepcopy(base.provider.document)
        third = copy.deepcopy(doc['profiles'][1])
        third.update(id='c', family='c')
        third['segments'][0]['kernel']['name'] = doc['profiles'][0]['segments'][0]['kernel']['name']
        doc['profiles'].append(third)
        p = m.KernelCompetitionFeatureProvider(KernelResourceProvider(doc, 'context', allow_cpu_fixture=True))
        cap = capture(p, b_start=0, b_duration=50)
        cap['aliases']['c'] = dict(family='c', supported_tasks=['task'])
        cap['owner_pids']['c'] = 43
        cap['requests'].append(dict(request_id='c:0', alias='c', family='c', task='task', pool_id='input',
            profile_id='c', execution=copy.deepcopy(third['execution'])))
        cap['offered_request_ids'].append('c:0')
        extra = copy.deepcopy(cap['kernels'][1])
        extra.update(owner='c', request_id='c:0', profile_id='c', pid=43, context_id=43,
            nsys_start_ns=50_000_000, nsys_end_ns=100_000_000, correlation_id=999,
            name=third['segments'][0]['kernel']['name'], static_kernel=copy.deepcopy(third['segments'][0]['kernel']))
        cap['kernels'].append(extra)
        out = p.training_rows([cap], allow_cpu_fixture=True)
        path = out['observations'][0]['path']
        self.assertEqual([(r['left_ns'], r['right_ns']) for r in path], [(0, 50_000_000), (50_000_000, 100_000_000)])
        self.assertEqual([r['features']['vector']['peer_kernel_count'] for r in path], [1, 1])
        self.assertEqual([r['features']['vector']['same_name_peer_count'] for r in path], [0, 1])
        active = active_rows(p.provider)
        early = p.features(active['a'], active, cap['aliases'], 'mean')
        late_row = dict(active['b'], alias='c', profile_id='c',
                        segment_duration_ms=third['segments'][0]['duration_ms']['mean'])
        late = p.features(active['a'], {'a': active['a'], 'c': late_row}, cap['aliases'], 'mean')
        self.assertEqual(path[0]['features'], early)
        self.assertEqual(path[1]['features'], late)
        self.assertEqual(out['accepted_kernel_count'], 3)

    def test_gpu_peer_filter_does_not_turn_host_work_into_static_sm_pressure(self):
        base = provider()
        doc = copy.deepcopy(base.provider.document)
        doc['profiles'][1]['segments'][0]['kind'] = 'host'
        p = m.KernelCompetitionFeatureProvider(KernelResourceProvider(doc, 'context', allow_cpu_fixture=True))
        active = active_rows(p.provider)
        features = p.features(active['a'], active, ALIASES, 'mean')
        self.assertEqual(features['vector']['peer_kernel_count'], 0)
        self.assertEqual(features['vector']['peer_register_block_fraction'], 0)
        self.assertFalse(p.features(active['b'], active, ALIASES, 'mean')['known'])

    def test_kernel_registry_order_does_not_change_chronological_feature_paths(self):
        p = provider(40000)
        cap = capture(p)
        original = p.training_rows([cap], allow_cpu_fixture=True)
        cap['kernels'].reverse()
        permuted = p.training_rows([cap], allow_cpu_fixture=True)
        left = {tuple(o['launch_identity']): o['path'] for o in original['observations']}
        right = {tuple(o['launch_identity']): o['path'] for o in permuted['observations']}
        self.assertEqual(left, right)
        for result in (original, permuted):
            for row in result['rows']:
                self.assertEqual(row['response_ms'], row['ordered_durations'].sum())
                np.testing.assert_allclose(row['durations'].sum(), row['ordered_durations'].sum())

    def test_profiler_domains_are_distinct_nuisance_groups_not_native_evidence(self):
        p = provider()
        first, second = capture(p), capture(p, 1)
        second['profiler_id'] = 'different-profiler'
        second['hardware_binding']['profiler_id'] = second['profiler_id']
        out = p.training_rows([first, second], allow_cpu_fixture=True)
        self.assertEqual(len({r['group'] for r in out['rows']}), 4)
        self.assertEqual(out['accepted_kernel_count'], 4)
        for field in ('native_sources_authenticated', 'natural_timing_validated',
                      'prospective_main_chain_wired', 'GPU_started', 'validation_labels_read'):
            self.assertFalse(out[field])
        with patch('tools.predictor_validation_20261027.macro_projection_contrast_training_v1.fit',
                   side_effect=AssertionError('optimizer forbidden')):
            fit, beta, mu, variance = m.fit_response(out)
        self.assertFalse(fit['numerical_qualification'])
        self.assertEqual((beta, mu, variance), (None, None, None))


if __name__ == '__main__':
    unittest.main()
