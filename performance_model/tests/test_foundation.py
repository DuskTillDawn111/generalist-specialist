"""CPU-only synthetic math fixtures plus read-only audits of real source results."""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deployment_core.models import ModelProfile,load_model_catalog
from deployment_core.workload import Workload,TASKS
from performance_model import FeatureExtractor,PiecewiseLoadLatencyModel,LoadPoint,build_colocation_pressure
from performance_model.features import curve_context
from performance_model.resources import load_resource_profiles,FEATURE_CSV
from performance_model.dataset import build_rows,build_dataset,ROOT


def catalog():
    return {m:ModelProfile(m,frozenset(ts),memory,ts) for m,memory,ts in [
        ('a',10,{'text':100.,'asr':200.}),('b',5,{'text':50.})]}


class FeatureTests(unittest.TestCase):
    def setUp(self): self.extractor=FeatureExtractor(catalog())

    def test_service_demand_mix_and_counts_are_not_plain_request_rate(self):
        f=self.extractor.extract({'a':1,'b':1},{'text':2,'asr':1},{'text':{'a':1,'b':1},'asr':{'a':1}})
        self.assertAlmostEqual(f['per_model']['a']['standalone_service_demand'],.3)
        self.assertAlmostEqual(f['per_model']['a']['task_demand_mix']['text'],1/3)
        self.assertAlmostEqual(f['per_model']['a']['task_demand_mix']['asr'],2/3)
        self.assertEqual(f['per_model']['b']['task_demand_mix']['asr'],0)
        self.assertEqual(f['task_order'],list(TASKS))
        self.assertEqual(f['deployment_features']['total_resident_memory_mib'],15)

    def test_seeder_style_pair_routing_workload_object_and_idle_model(self):
        f=self.extractor.extract({'a':1,'b':1},Workload({'text':1}),{('text','a'):1})
        self.assertEqual(f['per_model']['b']['standalone_service_demand'],0)
        self.assertFalse(f['per_model']['b']['task_demand_mix_defined'])
        self.assertEqual(sum(f['per_model']['b']['task_demand_mix'].values()),0)

    def test_invalid_inputs_and_capability_rejected(self):
        for x,w,r in [({'a':True},{'text':1},{'text':{'a':1}}),
                      ({'a':1},{'text':1},{'text':{'a':.5}}),
                      ({'a':1},{'text':1},{'text':{'b':1}}),
                      ({'b':1},{'asr':1},{'asr':{'b':1}}),
                      ({'a':1},{'text':0},{'text':{'a':1e-10}}),
                      ({'a':1},{'text':float('nan')},{})]:
            with self.subTest(x=x,w=w),self.assertRaises(ValueError): self.extractor.extract(x,w,r)

    def test_pressure_dynamic_features_missing_positive_paths_not_zero_filled(self):
        profiles={('text','a'):{'values':{'power':20.,'new_dimension':2.}},
                  ('text','b'):{'values':{'power':None,'new_dimension':3.}}}
        p=build_colocation_pressure('a',{'a':1,'b':1},{'text':{'a':100,'b':2}},profiles)
        self.assertIsNone(p['power']['value'])
        self.assertEqual(p['power']['known_partial_sum'],0)
        self.assertEqual(p['power']['missing_paths'][0]['model'],'b')
        self.assertEqual(p['new_dimension']['value'],6)
        self.assertEqual(build_colocation_pressure('a',{'a':1,'b':1},{'text':{'a':100,'b':0}},profiles)['power']['value'],0)

    def test_missing_entire_profile_retains_partial_pressure(self):
        profiles={('text','b'):{'values':{'resource':3.}}}
        p=build_colocation_pressure('a',{'a':1,'b':1,'c':1},{'text':{'b':2,'c':4}},profiles)['resource']
        self.assertIsNone(p['value']); self.assertEqual(p['known_partial_sum'],6)
        self.assertFalse(p['complete'])

    def test_live_labels_are_not_inputs(self):
        f=self.extractor.extract({'a':1},{'text':1},{'text':{'a':1}})
        text=json.dumps(f)
        for key in ('queue_wait','queue_length','recent_latency','measured_mean'): self.assertNotIn(key,text)


class BaselineTests(unittest.TestCase):
    def setUp(self): self.extractor=FeatureExtractor(catalog())

    def point(self,u,mean,p95,throughput,task='text',model='a',context=None):
        f=self.extractor.extract({'a':1},{'text':1},{'text':{'a':1}})
        return LoadPoint(model,task,u,context or curve_context(f,model),mean,p95,throughput,(str(Path(__file__).resolve()),))

    def test_no_data_or_one_point_returns_null(self):
        for points in ([],[self.point(.1,10,20,1)]):
            p=PiecewiseLoadLatencyModel(self.extractor,points).predict({'a':1},{'text':1},{'text':{'a':1}})
            self.assertEqual(p.status,'insufficient_profile_data')
            self.assertIsNone(p.per_task['text'].mean_latency_ms)
            self.assertIsNone(p.system_throughput_rps)

    def test_two_measured_fixture_points_interpolate_on_demand_not_rps(self):
        model=PiecewiseLoadLatencyModel(self.extractor,[self.point(.1,10,20,1),self.point(.3,30,60,3)])
        p=model.predict({'a':1},{'text':2},{'text':{'a':2}})
        self.assertAlmostEqual(p.per_task['text'].mean_latency_ms,20)
        self.assertAlmostEqual(p.per_task['text'].p95_latency_ms,40)
        self.assertAlmostEqual(p.system_throughput_rps,2)
        self.assertEqual(p.model_role,'planning_predictor')
        self.assertIsNone(model.predict({'a':1},{'text':4},{'text':{'a':4}}).per_task['text'].mean_latency_ms)
        self.assertIsNone(model.predict({'a':1,'b':1},{'text':2},{'text':{'a':2}}).per_task['text'].mean_latency_ms)

    def test_partial_labels_remain_missing(self):
        model=PiecewiseLoadLatencyModel(self.extractor,[self.point(.1,10,None,None),self.point(.3,30,60,3)])
        p=model.predict({'a':1},{'text':2},{'text':{'a':2}})
        self.assertIsNone(p.per_task['text'].p95_latency_ms)
        self.assertIsNone(p.system_throughput_rps)

    def test_mixture_p95_never_averaged(self):
        x={'a':1,'b':1}; w={'text':2}; r={'text':{'a':1,'b':1}}
        f=self.extractor.extract(x,w,r); points=[]
        for m in x:
            u=f['per_model'][m]['standalone_service_demand']
            for factor in (.5,1.5):
                points.append(self.point(u*factor,10 if m=='a' else 30,20 if m=='a' else 60,factor,
                                         model=m,context=curve_context(f,m)))
        p=PiecewiseLoadLatencyModel(self.extractor,points).predict(x,w,r)
        self.assertAlmostEqual(p.per_task['text'].mean_latency_ms,20)
        self.assertIsNone(p.per_task['text'].p95_latency_ms)

    def test_replicas_declined_without_invented_load_balancing(self):
        p=PiecewiseLoadLatencyModel(self.extractor).predict({'a':2},{'text':1},{'text':{'a':1}})
        self.assertEqual(p.status,'unsupported_replica_semantics')

    def test_invalid_points_need_provenance_and_finite_values(self):
        with self.assertRaises(ValueError): PiecewiseLoadLatencyModel(self.extractor,[self.point(.1,-1,1,1)])
        p=self.point(.1,1,1,1)
        with self.assertRaises(ValueError): PiecewiseLoadLatencyModel(self.extractor,[LoadPoint(**{**asdict(p),'source_result_paths':()})])


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=load_model_catalog(); cls.resources=load_resource_profiles(cls.catalog)
        cls.extractor=FeatureExtractor(cls.catalog,cls.resources)

    def test_existing_offline_profiles_and_missing_profile(self):
        self.assertEqual(len(self.resources),11)
        self.assertTrue(all(p['sources'] for p in self.resources.values()))
        for p in self.resources.values():
            fractions=[v for k,v in p['values'].items() if k.startswith('kernel_fraction_')]
            self.assertAlmostEqual(sum(fractions),1)
        extra={**self.catalog,'unprofiled':ModelProfile('unprofiled',frozenset({'text'}),1,{'text':1})}
        profile=load_resource_profiles(extra)['text','unprofiled']
        self.assertTrue(all(v is None for v in profile['values'].values()))
        self.assertTrue(profile['missing'])

    def test_task_model_labels_derived_from_matching_requests(self):
        path=ROOT/'results/stage2a/smoke_text.json'
        rows,meta=build_rows(path,'stage2a','smoke',self.extractor)
        self.assertEqual(len(rows),2)
        self.assertEqual(sum(r['labels']['offered'] for r in rows),80)
        self.assertNotEqual(rows[0]['labels']['mean_e2e_latency_ms'],rows[1]['labels']['mean_e2e_latency_ms'])
        self.assertTrue(all(r['source_result_path']==str(path.resolve()) for r in rows))
        self.assertEqual(sum(r['labels']['throughput_rps'] for r in rows),4)

    def test_bad_source_record_is_rejected(self):
        data=json.loads((ROOT/'results/stage2a/smoke_text.json').read_text())
        data['request_records'][0]['e2e_latency_ms']+=100
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.json'; path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'E2E mismatch'): build_rows(path,'stage2a','smoke',self.extractor)

    def test_full_dataset_audit_without_enumeration_or_profiling(self):
        with tempfile.TemporaryDirectory() as folder,patch('deployment_core.candidate_generator.enumerate_deployments',side_effect=AssertionError('Enumeration forbidden')) as enumeration,patch('subprocess.run',side_effect=AssertionError('Profiling subprocess forbidden')):
            manifest,coverage=build_dataset(folder)
            enumeration.assert_not_called()
            self.assertEqual(manifest['runs'],18); self.assertEqual(manifest['rows'],148)
            self.assertEqual(manifest['request_records'],8966)
            self.assertEqual(len(coverage['deployments']),3)
            self.assertFalse(coverage['curve_audit']['sufficient_for_controlled_piecewise_curve'])
            self.assertEqual(manifest['profile_points_created'],0)
            self.assertFalse(manifest['trained']); self.assertFalse(manifest['predictor_accuracy_reported'])
            rows=[json.loads(line) for line in (Path(folder)/'dataset_rows.jsonl').read_text().splitlines()]
            requests=defaultdict(int)
            for row in rows: requests[row['source_result_path']]+=row['labels']['offered']
            self.assertEqual(sum(requests.values()),8966)
            self.assertEqual(len(requests),18)


if __name__=='__main__': unittest.main()
