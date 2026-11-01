import copy
import math
import unittest

import numpy as np

from gsdeploy.forecasting.kernel_provider import KernelResourceProvider
from tests.test_kernel_provider import document, active_rows, ALIASES
from tests.test_kernel_resources import hardware, kernel
from tools.predictor_validation_20261027 import kernel_competition_features_v1 as module
from tools.predictor_validation_20261027.macro_joint_response_fit_v1 import objective


def provider(shared=0):
    doc = document(shared)
    for profile in doc['profiles']:
        profile['segments'][0]['kernel']['name'] = profile['id'] + ':kernel'
    return module.KernelCompetitionFeatureProvider(KernelResourceProvider(doc,'context',allow_cpu_fixture=True))


def capture(p, index=0, *, b_start=20, a_duration=100, b_duration=100):
    records = []
    for owner,start,duration,pid in [('a',0,a_duration,41),('b',b_start,b_duration,42)]:
        static = p.provider.segments[owner,'kernel']['kernel']
        records.append(dict(owner=owner,request_id=owner + ':' + str(index),profile_id=owner,
            segment_id='kernel',name=static['name'],static_kernel=copy.deepcopy(static),
            nsys_start_ns=start*1_000_000,nsys_end_ns=(start+duration)*1_000_000,
            pid=pid,device=0,context_id=pid,stream=7,correlation_id=index*10+pid))
    return dict(schema=module.CAPTURE_SCHEMA,capture_id='capture:' + str(index),
        execution_context_sha256='context',split='training',evidence_kind='cpu_fixture',
        validation_labels_read=False,timing_basis='profiled_kernel_lifetime',
        profiler_id='synthetic-nsys',owner_pids={'a':41,'b':42},kernels=records,
        offered_request_ids=[k['request_id'] for k in records])


class FutureLabels(dict):
    def __getitem__(self,key):
        raise AssertionError('Prospective path read target labels: ' + key)
    def __iter__(self):
        raise AssertionError('Prospective path iterated target labels')


class KernelCompetitionFeaturesTests(unittest.TestCase):
    def test_inference_uses_static_catalog_no_end_or_resource_speed_claim(self):
        p = provider(); rows = active_rows(p.provider)
        rows['a']['future_native_labels'] = FutureLabels(nsys_end_ns=100)
        features = p.features(rows['a'],rows,ALIASES,'mean')
        self.assertTrue(features['known'])
        self.assertEqual(features['vector']['peer_kernel_count'],1)
        self.assertIsNone(features['service_rate_multiplier'])
        self.assertIsNone(features['SM_throughput'])
        self.assertIsNone(features['DRAM_bytes'])
        self.assertEqual(features['compute_memory_classification'],'UNKNOWN')
        self.assertFalse(p.provider.allow_analytical_hypothesis)

    def test_training_and_prospective_use_identical_feature_mapping(self):
        p=provider(); prepared=p.training_rows([capture(p)],allow_cpu_fixture=True)
        self.assertEqual(prepared['offered_kernel_count'],2)
        self.assertEqual(prepared['accepted_kernel_count'],2)
        self.assertEqual(prepared['rejected_rows'],[])
        rows=active_rows(p.provider)
        inference=p.features(rows['a'],rows,ALIASES,'mean')
        path=prepared['observations'][0]['path']
        self.assertEqual([(p['left_ns'],p['right_ns']) for p in path],[(0,20_000_000),(20_000_000,100_000_000)])
        self.assertEqual(path[1]['features'],inference)
        self.assertEqual(prepared['rows'][0]['ordered_durations'].sum(),100)

    def test_distinct_ordered_kernels_reprice_features_when_peer_changes(self):
        p=provider(40000)
        doc=copy.deepcopy(p.provider.document)
        light=copy.deepcopy(doc['profiles'][1]['segments'][0])
        light['id']='light';light['kernel']=kernel(shared=1024);light['kernel']['name']='light'
        doc['profiles'][1]['segments'].append(light)
        p=module.KernelCompetitionFeatureProvider(KernelResourceProvider(doc,'context',allow_cpu_fixture=True))
        cap=capture(p,b_start=0,b_duration=50)
        tail=copy.deepcopy(cap['kernels'][1]);tail.update(segment_id='light',name='light',
            static_kernel=copy.deepcopy(light['kernel']),nsys_start_ns=50_000_000,
            nsys_end_ns=150_000_000,correlation_id=99)
        cap['kernels'].append(tail)
        prepared=p.training_rows([cap],allow_cpu_fixture=True)
        target=prepared['observations'][0]['path']
        self.assertEqual([p['features']['vector']['mixed_one_block_infeasible'] for p in target],[1,0])
        rows=active_rows(p.provider)
        self.assertEqual(p.features(rows['a'],rows,ALIASES,'mean')['vector']['mixed_one_block_infeasible'],1)
        rows['b'].update(segment_id='light',segment_duration_ms=light['duration_ms']['mean'])
        self.assertEqual(p.features(rows['a'],rows,ALIASES,'mean')['vector']['mixed_one_block_infeasible'],0)

    def test_mixed_static_conflict_is_not_a_throughput_prediction(self):
        p=provider(40000); rows=active_rows(p.provider)
        feature=p.features(rows['a'],rows,ALIASES,'mean')
        self.assertEqual(feature['vector']['mixed_one_block_infeasible'],1)
        self.assertFalse(feature['actual_concurrent_SM_residency_proven'])
        self.assertIsNone(feature['service_rate_multiplier'])
        self.assertFalse(p.provider.service(rows['a'],rows,ALIASES,'mean')['known'])

    def test_unknown_rounded_static_fields_retain_entire_capture_denominator(self):
        p=provider();p.provider.segments['b','kernel']['kernel'].pop('allocated_registers_per_block')
        prepared=p.training_rows([capture(p)],allow_cpu_fixture=True)
        self.assertEqual(prepared['offered_kernel_count'],2)
        self.assertEqual(prepared['accepted_kernel_count'],0)
        self.assertEqual(len(prepared['rejected_rows']),2)
        with self.assertRaisesRegex(ValueError,'Complete supported kernel cohort'):
            module.fit_response(prepared)

    def test_multistream_or_same_owner_overlap_is_whole_capture_unknown(self):
        for start,stream in [(40,9),(40,7),(100,9)]:
            p=provider();cap=capture(p)
            extra=copy.deepcopy(cap['kernels'][0]);extra.update(nsys_start_ns=start*1_000_000,
                nsys_end_ns=(start+20)*1_000_000,stream=stream,correlation_id=999)
            cap['kernels'].append(extra)
            prepared=p.training_rows([cap],allow_cpu_fixture=True)
            self.assertEqual(prepared['accepted_kernel_count'],0)
            self.assertEqual(len(prepared['rejected_rows']),3)

    def test_request_without_kernel_trace_is_not_dropped_from_training(self):
        p=provider();cap=capture(p);cap['offered_request_ids'].append('missing')
        prepared=p.training_rows([cap],allow_cpu_fixture=True)
        self.assertEqual(prepared['offered_requests'],3)
        self.assertEqual(prepared['accepted_kernel_count'],0)
        self.assertEqual(len(prepared['rejected_rows']),3)
        with self.assertRaisesRegex(ValueError,'Complete supported kernel cohort'):
            module.fit_response(prepared)

    def test_role_gate_precedes_kernel_labels_and_fixtures_need_opt_in(self):
        p=provider();cap=capture(p);cap['split']='validation';cap['kernels']=FutureLabels()
        with self.assertRaisesRegex(ValueError,'Training role required'):
            p.training_rows([cap],allow_cpu_fixture=True)
        with self.assertRaisesRegex(ValueError,'fixture opt-in'):
            p.training_rows([capture(p)])

    def test_actual_response_likelihood_consumes_compiled_six_feature_rows(self):
        p=provider();caps=[capture(p,i,b_start=10+10*i,a_duration=100+3*i,b_duration=80+7*i)
            for i in range(4)]
        prepared=p.training_rows(caps,allow_cpu_fixture=True)
        value,gradient=objective(np.zeros(len(module.FEATURES)),prepared['rows'])
        self.assertTrue(math.isfinite(value));self.assertTrue(np.isfinite(gradient).all())
        # The shared fitter adapter checks the compiled kernel paths. This fixture
        # deliberately lacks six independent resource contrasts: it must retain
        # all columns and fail numerical qualification instead of dropping them.
        info,beta,mu,variance=module.fit_response(prepared)
        self.assertEqual(info['features'],list(module.FEATURES))
        self.assertFalse(info['numerical_qualification'])
        self.assertFalse(info['effective_SM_service_identified'])
        self.assertEqual(info['reason'],'STRUCTURAL_KERNEL_COVARIATE_RANK_FAILURE')
        self.assertFalse(info['columns_dropped'])
        self.assertIsNone(beta)


if __name__=='__main__':
    unittest.main()
