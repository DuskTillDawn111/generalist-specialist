import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from performance_model.active_sampling.plan import make_plan,one_hop
from performance_model.active_sampling.common import configuration_id,deployment_id
from performance_model.active_sampling.grouping import evaluation_configuration_id
from performance_model.stability import DEFAULT_THRESHOLDS,label_stability


class ActiveSamplingTests(unittest.TestCase):
    def test_only_one_hop_and_no_enumeration(self):
        with tempfile.TemporaryDirectory() as tmp,patch('deployment_core.candidate_generator.enumerate_deployments',side_effect=AssertionError('Forbidden enumeration')),patch('deployment_core.candidate_generator.generate_feasible_deployments',side_effect=AssertionError('Forbidden enumeration')):
            p=make_plan(tmp);pool=json.loads((Path(tmp)/'candidate_pool.json').read_text())
        self.assertEqual(len(pool['mutations']),5)
        self.assertEqual(len(p['conditions']),6)
        self.assertEqual(sum(len(c['runs']) for c in p['conditions']),18)
        self.assertEqual({c['expected_category'] for c in p['conditions']},{'clearly_stable','near_boundary','likely_unstable'})
        for c in p['conditions']:
            self.assertEqual(sum(abs(c['deployment'][m]-c['source_seed'][m]) for m in c['deployment']),1)
            self.assertTrue(c['feasibility']['passed'])
            self.assertFalse(any(k.endswith(('_id','/id','/name')) for k in c['feature_vector']))
        self.assertGreaterEqual(len({c['deployment_id'] for c in p['conditions']}),3)

    def test_labels_use_operational_evidence_and_explicit_thresholds(self):
        d=dict(status='completed',errors=[],system_metrics={'completion_ratio':1.},
            request_records=[dict(status='completed',scheduled_time_ns=0,completion_time_ns=1,e2e_latency_ms=999999)],
            measurement_window={'start_time_ns':0,'end_time_ns':10000000000,'drain_seconds':0.})
        self.assertEqual(label_stability(d,DEFAULT_THRESHOLDS).category,'stable')
        d['measurement_window']['drain_seconds']=7
        self.assertEqual(label_stability(d,DEFAULT_THRESHOLDS).category,'boundary')
        d['request_records'].append(dict(status='incomplete',scheduled_time_ns=0,completion_time_ns=None))
        self.assertFalse(label_stability(d,DEFAULT_THRESHOLDS).stable)
        with self.assertRaises(ValueError):label_stability(d,{})

    def test_group_identity_does_not_include_seed(self):
        d=dict(deployment={'qwen':1},workload_rps={'text':1},routing_rps={'text':{'qwen':1}},
            warmup_seconds=10,measurement_seconds=120,drain_timeout_seconds=30,random_seed=1)
        self.assertEqual(configuration_id(d),configuration_id(dict(d,random_seed=2)))
        self.assertNotEqual(configuration_id(d),configuration_id(dict(d,measurement_seconds=180)))
        self.assertEqual(evaluation_configuration_id(d),evaluation_configuration_id(dict(d,measurement_seconds=180,random_seed=2)))
        padded=dict(d,deployment={'qwen':1,'whisper':0},workload_rps={'text':1.,'asr':0.},routing_rps={'text':{'qwen':1.,'whisper':0.},'asr':{'qwen':0.}})
        self.assertEqual(evaluation_configuration_id(d),evaluation_configuration_id(padded))
        self.assertEqual(deployment_id({'qwen':1}),deployment_id({'qwen':1,'whisper':0}))

    def test_complete_after_drain_can_still_have_growing_backlog(self):
        d=dict(status='completed',errors=[],system_metrics={'completion_ratio':1.},
            request_records=[dict(status='completed',scheduled_time_ns=i*200000000,completion_time_ns=11000000000) for i in range(50)],
            measurement_window={'start_time_ns':0,'end_time_ns':10000000000,'drain_seconds':1.})
        observed=label_stability(d,DEFAULT_THRESHOLDS)
        self.assertFalse(observed.stable)
        self.assertIn('sustained_backlog_growth',observed.reason)
        changed=dict(DEFAULT_THRESHOLDS,growth_min_requests=1000,end_backlog_unstable_min_requests=1000)
        self.assertEqual(label_stability(d,changed).category,'boundary')

if __name__=='__main__':unittest.main()
