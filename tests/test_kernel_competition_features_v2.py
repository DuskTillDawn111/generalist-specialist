"""CPU identity, temporal mapping and UNKNOWN/rank gates; no optimizer runs."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.kernel_provider import KernelResourceProvider
from tests.test_kernel_competition_features_v1 import provider as provider_v1, capture as capture_v1, FutureLabels
from tests.test_kernel_provider import active_rows, ALIASES
from tests.test_kernel_resources import kernel
from tools.predictor_validation_20261027 import kernel_competition_features_v1 as previous
from tools.predictor_validation_20261027 import kernel_competition_features_v2 as m


def provider(shared=0):
    return m.KernelCompetitionFeatureProvider(provider_v1(shared).provider)


def capture(p,index=0,**kwargs):
    cap=capture_v1(p,index,**kwargs)
    cap.update(schema=m.CAPTURE_SCHEMA,aliases=copy.deepcopy(ALIASES),scenario='mean',
        hardware_binding=dict(hardware_id=p.provider.hardware['hardware_id'],device=0,
            device_identity_basis='canonical_profiler_device_index',profiler_id=cap['profiler_id'],
            execution_context_sha256=cap['execution_context_sha256']))
    cap['requests']=[dict(request_id=k['request_id'],alias=k['owner'],family=ALIASES[k['owner']]['family'],
        task='task',pool_id='input',profile_id=k['profile_id'],
        execution=copy.deepcopy(p.provider.profiles[k['profile_id']]['execution'])) for k in cap['kernels']]
    return cap


class KernelCompetitionV2Tests(unittest.TestCase):
    def test_valid_binding_preserves_v1_mathematics_without_throughput_claim(self):
        p=provider(40000)
        prepared=p.training_rows([capture(p)],allow_cpu_fixture=True)
        rows=active_rows(p.provider)
        result=p.features(rows['a'],rows,ALIASES,'mean')
        old=previous.competition_features(p.provider.segments['a','kernel']['kernel'],
            [p.provider.segments['b','kernel']['kernel']],p.provider.hardware)
        self.assertEqual(result['vector'],old['vector'])
        self.assertEqual(prepared['observations'][0]['path'][1]['features'],result)
        self.assertEqual(prepared['accepted_kernel_count'],2)
        self.assertFalse(prepared['native_sources_authenticated'])
        self.assertFalse(result['actual_concurrent_SM_residency_proven'])
        for field in ('SM_throughput','DRAM_bytes','L2_hit_rate','service_rate_multiplier'):
            self.assertIsNone(result[field])

    def test_cross_device_counterexample_rejected_before_covariate_math(self):
        p=provider();cap=capture(p);cap['kernels'][1]['device']=1
        with patch.object(m,'competition_features') as features:
            with self.assertRaisesRegex(ValueError,'Cross-device'):
                p.training_rows([cap],allow_cpu_fixture=True)
        features.assert_not_called()

    def test_foreign_owner_profile_counterexample_rejected(self):
        p=provider();cap=capture(p)
        foreign=p.provider.segments['b','kernel']['kernel']
        cap['kernels'][0].update(profile_id='b',name=foreign['name'],static_kernel=copy.deepcopy(foreign))
        with self.assertRaisesRegex(ValueError,'request/owner/profile'):
            p.training_rows([cap],allow_cpu_fixture=True)

    def test_request_domain_axes_match_prospective_identity_gates(self):
        p=provider()
        for field,value in [('family','b'),('task','other'),('pool_id','other'),
                            ('profile_id','b'),('execution',{'batch_size':99}),('alias','b')]:
            cap=capture(p);cap['requests'][0][field]=value
            with self.subTest(field=field),self.assertRaises((ValueError,UnknownEvidence)):
                p.training_rows([cap],allow_cpu_fixture=True)

    def test_hardware_binding_is_closed_and_context_profiler_specific(self):
        p=provider()
        for field,value in [('hardware_id','other'),('device',True),('profiler_id','other'),
                            ('execution_context_sha256','other'),('device_identity_basis','process_local_ordinal')]:
            cap=capture(p);cap['hardware_binding'][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'device-to-hardware'):
                p.training_rows([cap],allow_cpu_fixture=True)
        cap=capture(p);cap['hardware_binding']['unspecified_axis']=0
        with self.assertRaises(ValueError):p.training_rows([cap],allow_cpu_fixture=True)

    def test_half_open_ordered_path_reprices_peer_kernel_and_preserves_gaps(self):
        p=provider(40000)
        doc=copy.deepcopy(p.provider.document)
        light=copy.deepcopy(doc['profiles'][1]['segments'][0])
        light.update(id='light',kernel=kernel(shared=1024))
        light['kernel']['name']='light'
        doc['profiles'][1]['segments'].append(light)
        p=m.KernelCompetitionFeatureProvider(KernelResourceProvider(doc,'context',allow_cpu_fixture=True))
        cap=capture(p,b_start=0,b_duration=20)
        tail=copy.deepcopy(cap['kernels'][1]);tail.update(segment_id='light',name='light',
            static_kernel=copy.deepcopy(light['kernel']),nsys_start_ns=40_000_000,
            nsys_end_ns=60_000_000,correlation_id=999)
        cap['kernels'].append(tail)
        out=p.training_rows([cap],allow_cpu_fixture=True)
        path=out['observations'][0]['path']
        self.assertEqual([(r['left_ns'],r['right_ns']) for r in path],
            [(0,20_000_000),(20_000_000,40_000_000),(40_000_000,60_000_000),(60_000_000,100_000_000)])
        self.assertEqual([r['features']['vector']['peer_kernel_count'] for r in path],[1,0,1,0])
        self.assertEqual([r['features']['vector']['mixed_one_block_infeasible'] for r in path],[1,0,0,0])
        target=out['rows'][0]
        np.testing.assert_array_equal(target['ordered_durations'],[20,20,20,40])
        self.assertEqual(sum(target['durations']),100)
        self.assertEqual(target['durations'][-1],60)
        active=active_rows(p.provider)
        active['b'].update(segment_id='light',segment_duration_ms=light['duration_ms']['mean'])
        self.assertEqual(path[2]['features'],p.features(active['a'],active,ALIASES,'mean'))

    def test_context_or_stream_transition_is_whole_capture_unknown(self):
        p=provider()
        for field in ('context_id','stream'):
            cap=capture(p);extra=copy.deepcopy(cap['kernels'][0])
            extra.update(nsys_start_ns=100_000_000,nsys_end_ns=120_000_000,correlation_id=999)
            extra[field]+=1;cap['kernels'].append(extra)
            out=p.training_rows([cap],allow_cpu_fixture=True)
            self.assertEqual((out['offered_kernel_count'],out['accepted_kernel_count']),(3,0))
            self.assertEqual(len(out['rejected_rows']),3)

    def test_structural_rank_failure_never_calls_optimizer_or_drops_columns(self):
        p=provider()
        out=p.training_rows([capture(p,i,b_start=10+10*i) for i in range(4)],allow_cpu_fixture=True)
        with patch('tools.predictor_validation_20261027.macro_projection_contrast_training_v1.fit',
                   side_effect=AssertionError('optimizer forbidden')):
            info,beta,mu,variance=m.fit_response(out)
        self.assertEqual(info['reason'],'STRUCTURAL_KERNEL_COVARIATE_RANK_FAILURE')
        self.assertEqual(info['features'],list(m.FEATURES))
        self.assertFalse(info['columns_dropped'])
        self.assertFalse(info['effective_SM_service_identified'])
        self.assertEqual((beta,mu,variance),(None,None,None))

    def test_role_gates_all_captures_before_kernel_access_and_prediction_no_labels(self):
        p=provider();first=capture(p);second=capture(p,1)
        first['kernels']=FutureLabels();second['split']='validation'
        with self.assertRaisesRegex(ValueError,'Training role'):
            p.training_rows([first,second],allow_cpu_fixture=True)
        active=active_rows(p.provider);active['a']['future_labels']=FutureLabels()
        with patch.object(p.provider,'service',side_effect=AssertionError('no analytical speed')):
            self.assertTrue(p.features(active['a'],active,ALIASES,'mean')['known'])

    def test_request_without_trace_retains_offered_denominator_and_blocks_fit(self):
        p=provider();cap=capture(p)
        missing=copy.deepcopy(cap['requests'][0]);missing['request_id']='missing'
        cap['requests'].append(missing);cap['offered_request_ids'].append('missing')
        out=p.training_rows([cap],allow_cpu_fixture=True)
        self.assertEqual((out['offered_requests'],out['offered_kernel_count'],out['accepted_kernel_count']),(3,2,0))
        self.assertEqual(len(out['rejected_rows']),3)
        with self.assertRaisesRegex(ValueError,'Complete supported'):
            m.fit_response(out)


if __name__=='__main__':
    unittest.main()
