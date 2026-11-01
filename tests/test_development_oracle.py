"""Independent CPU oracle contracts on frozen real inputs, with no run access."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import statistics
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('oracle_under_test',ROOT/'tools/spillover_context_20261022/development_oracle.py')
o=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(o)


class FIFO(unittest.TestCase):
    def test_queue_waiting_and_independent_alias_unknown_suffix(self):
        rows=[dict(alias='a',arrival_seconds=0.,service_ms=1000),
            dict(alias='a',arrival_seconds=.2,service_ms=1000),
            dict(alias='b',arrival_seconds=.3,service_ms=None),
            dict(alias='b',arrival_seconds=.4,service_ms=1),
            dict(alias='a',arrival_seconds=3.,service_ms=100)]
        r=o.fifo(rows,2.,2.)
        self.assertEqual(r[1]['queue_ms'],800.)
        self.assertEqual(r[1]['latency_ms'],1800.)
        self.assertIsNone(r[2]['finish_seconds']);self.assertIsNone(r[3]['finish_seconds'])
        self.assertAlmostEqual(r[4]['finish_seconds'],3.1)

    def test_convex_hull_not_axis_box(self):
        self.assertFalse(o.convex_hull([.7,5],[[.2,1],[.4,2],[.8,5]])['inside'])
        self.assertIsNone(o.convex_hull([None,5],[[.2,1],[.8,5]])['inside'])


class RealDevelopmentInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle=o.DevelopmentOracle();cls.plan=copy.deepcopy(cls.oracle.packet['jobs'][0]['plan'])

    def test_exact_original_trace_samples_aliases_quality_and_arithmetic_rho(self):
        oracle=self.oracle;r=oracle.evaluate(self.plan,107,True,True);d=r['diagnostic_extrapolation_only']
        self.assertIsNone(r['U']);self.assertFalse(d['valid_prediction']);self.assertFalse(d['execution_admission'])
        self.assertTrue(d['utility_complete']);self.assertEqual(len(d['requests']),1175)
        aliases,routes=oracle.api['trace'].expand_routes(self.plan['deployment'],self.plan['routing'])
        router=oracle.api['routing'].Router(routes);trace,cohort=oracle._cohort(107)
        for row,(arrival,task,sid) in zip(d['requests'],cohort):
            self.assertEqual((row['arrival_seconds'],row['task'],row['pool_id'],row['alias']),
                (arrival,task,sid,router.choose(task)))
        self.assertEqual(r['trace_sha256'],oracle.api['trace'].trace_digest(trace))
        Q,_=oracle.api['quality'].predict_quality(oracle.catalog,self.plan['deployment'],self.plan['routing'],oracle.config['workload'],trace,107)
        self.assertEqual(Q,d['terms']['Q'])
        self.assertAlmostEqual(d['U'],-sum(oracle.config['weights'][t]*v for t,v in d['terms'].items()))
        A,M,H,_=oracle._context(self.plan);svc,_,_,_=oracle._service(self.plan,A,M,H)
        values=[svc['text','text',2,sid] for sid in oracle.pool_ids['text']]
        mean=statistics.mean(values);geometric=math.exp(statistics.mean(math.log(v) for v in values))
        self.assertNotAlmostEqual(mean,geometric)
        self.assertAlmostEqual(d['rho_by_family']['text'],1.85*mean/1000/2)
        # Each point predicts actual worker service for exact n. No second
        # n/capacity-factor service correction may be silently applied.
        self.assertAlmostEqual(d['arithmetic_mean_point_service_ms']['text/text/n2'],mean)

    def test_new_resident_context_unknown_but_numeric_probe_kept(self):
        plan=dict(deployment={'text':2,'image':1,'ocr':1,'asr':1},
            routing={'text':{'text':1.85},'image':{'image':.01},'ocr':{'ocr':.01},'asr':{'asr':.01}})
        r=self.oracle.evaluate(plan,107,True)
        self.assertIsNone(r['U']);self.assertIsNone(r['resources']['M']['value'])
        self.assertFalse(r['resources']['M']['deployment_observed'])
        self.assertIsNotNone(r['diagnostic_extrapolation_only']['U'])
        self.assertTrue(any(x['kind']=='UNSUPPORTED_RESOURCE_CONTEXT' for x in r['unknown_reasons']))

    def test_uncovered_generalist_is_unknown_without_role_filter(self):
        plan=dict(deployment={'generalist_1':1},routing={t:{'generalist_1':q} for t,q in self.oracle.config['workload'].items()})
        r=self.oracle.evaluate(plan,107,True)
        self.assertNotEqual(r['status'],'INVALID_PLAN')
        self.assertIsNone(r['diagnostic_extrapolation_only']['U'])
        self.assertTrue(any(x['kind']=='UNSUPPORTED_EXACT_PATH_N' for x in r['unknown_reasons']))

    def test_missing_per_input_baseline_never_aggregate_filled(self):
        oracle=self.oracle;key=('text','text',2);values=oracle._baselines(*key);sid=next(iter(values));old=values[sid]
        try:
            values[sid]=None;r=oracle.evaluate(self.plan,107,True)
            self.assertIsNone(r['diagnostic_extrapolation_only']['U'])
            self.assertIsNone(r['diagnostic_extrapolation_only']['rho_by_family']['text'])
            self.assertTrue(any(x['kind']=='MISSING_COMPLETE_PER_INPUT_BASELINE' for x in r['unknown_reasons']))
        finally:values[sid]=old

    def test_route_conservation_and_reserved_seed_rejected(self):
        plan=copy.deepcopy(self.plan);plan['routing']['text']['text']=1.
        self.assertEqual(self.oracle.evaluate(plan,107,True)['status'],'INVALID_PLAN')
        self.assertEqual(self.oracle.evaluate(self.plan,71,True)['status'],'INVALID_PLAN')

    def test_idle_unknown_replica_state_and_infinite_route_rejected(self):
        plan=copy.deepcopy(self.plan);plan['deployment']['generalist_1']=900
        self.assertEqual(self.oracle.evaluate(plan,107,True)['status'],'INVALID_PLAN')
        plan=copy.deepcopy(self.plan);plan['routing']['text']['text']=float('inf')
        self.assertEqual(self.oracle.evaluate(plan,107,True)['status'],'INVALID_PLAN')

    def test_absent_baseline_source_is_unknown_but_sha_mismatch_raises(self):
        oracle=self.oracle;key=('text','text',1)
        profile=oracle.catalog['task_replica_profiles']['text']['text']['1'];old=list(profile['raw_service_samples'])
        cached=oracle._baseline_cache.pop(key,None)
        try:
            profile['raw_service_samples']=['/nonexistent/unknown-baseline.jsonl']
            self.assertIsNone(oracle._baselines(*key));oracle._baseline_cache.pop(key)
            profile['raw_service_samples']=old
            expected_path=str(Path(old[0]).resolve());expected=oracle.inventory_manifest['inputs'][expected_path]
            oracle.inventory_manifest['inputs'][expected_path]='0'*64
            try:
                with self.assertRaisesRegex(ValueError,'Frozen input changed'):oracle._baselines(*key)
            finally:oracle.inventory_manifest['inputs'][expected_path]=expected
        finally:
            profile['raw_service_samples']=old;oracle._baseline_cache.pop(key,None)
            if cached is not None:oracle._baseline_cache[key]=cached

    def test_probe_opt_in_and_no_request_dump_by_default(self):
        r=self.oracle.evaluate(self.plan,108)
        self.assertIsNone(r['diagnostic_extrapolation_only']);self.assertNotIn('requests',r)
        self.assertFalse(self.oracle.domain_summary()['full_seven_pool_coverage'])
        self.assertTrue(self.oracle.verify_inputs())

    def test_skip_strict_hulls_cannot_change_diagnostic_objective_or_claim_support(self):
        full=self.oracle.evaluate(self.plan,107,True)
        fast=self.oracle.evaluate(self.plan,107,True,strict_context_checks=False)
        self.assertEqual(full['diagnostic_extrapolation_only'],fast['diagnostic_extrapolation_only'])
        self.assertIsNone(fast['U']);self.assertIsNone(fast['supported_domain_development_arithmetic_U'])
        self.assertFalse(fast['strict_context_checks_performed'])
        self.assertTrue(all(p['hull']['status']=='NOT_CHECKED_DIAGNOSTIC_ONLY' for p in fast['service_domain']))


if __name__=='__main__':unittest.main()
