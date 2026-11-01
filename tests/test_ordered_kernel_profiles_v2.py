import copy
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.simulator import FullHorizonModel
from tests.test_kernel_provider import document as kernel_document
from tools.predictor_validation_20261027.ordered_kernel_profiles_v2 import (
    SCHEMA, OrderedKernelProfileProvider, convert_collection)
from tools.predictor_validation_20261027.kernel_sequence_audit_v2 import analyze_sequence


ALIASES = {a:dict(family='shared-family',supported_tasks=['task']) for a in ('a','b')}


def document():
    doc=kernel_document();doc['schema']=SCHEMA
    for p in doc['profiles']:
        p.update(alias=p['family'],family='shared-family')
        p['segments'].insert(0,dict(id='gap',kind='unclassified_gap',duration_ms=dict(mean=10),
                                   evidence='synthetic-unclassified-gap'))
    return doc


def active(provider,index=0):
    result={}
    for alias in ALIASES:
        row=dict(alias=alias,task='task',pool_id='input',request_id=alias,arrival_ns=0)
        plan=provider.plan(row,ALIASES,'mean');seg=plan['segments'][index]
        row.update(profile_id=plan['profile_id'],segment_id=seg['id'],segment_kind=seg['kind'],
                   segment_duration_ms=seg['duration_ms']);result[alias]=row
    return result


def collection(stream=7, split='training'):
    sql=dict(path='/fixture/sqlite',sha256='fixture',size_bytes=1)
    kernels=[dict(nsys_start_ns=10,nsys_end_ns=20,pid=42,device=0,context_id=1,stream=7,
                  correlation_id=1,source_sqlite=sql),
             dict(nsys_start_ns=30,nsys_end_ns=40,pid=42,device=0,context_id=1,stream=stream,
                  correlation_id=2,source_sqlite=sql)]
    row=dict(request_id='r',model='a',task='task',pool_id='input',split=split,profile_job_id='job',
             worker_response=dict(start_perf_ns=100,end_perf_ns=150))
    mapping=dict(valid=True,offset_ns=100,max_boundary_residual_ns=1000)
    audited=analyze_sequence(kernels,row,clock_mapping=mapping)
    audited.update(request_id='r',family='shared-family')
    raw=dict(execution_context_sha256='context')
    plan=dict(split=split,training_only=True,domain='light-kernel',
              jobs=[dict(id='job',split=split,inputs=dict(a='input'))])
    requests=dict(r=dict(row=row,kernels=kernels))
    audit=dict(requests=[audited],source_receipts=['fixture'],summary=dict(unassigned_kernel_count=0))
    return raw,plan,dict(measurement_domain='light-kernel'),requests,audit


class OrderedKernelProfileTests(unittest.TestCase):
    def provider(self,**options):
        return OrderedKernelProfileProvider(document(),'context',allow_cpu_fixture=True,**options)

    def test_same_family_owners_stay_distinct_and_unsupported_input_unknown(self):
        p=self.provider();rows=active(p)
        self.assertNotEqual(rows['a']['profile_id'],rows['b']['profile_id'])
        with self.assertRaises(UnknownEvidence):p.plan(dict(rows['a'],pool_id='unseen'),ALIASES,'mean')

    def test_gap_requires_conditional_model_and_gpu_does_not_ignore_gap(self):
        p=self.provider(allow_analytical_hypothesis=True);gaps=active(p);gpus=active(p,1)
        self.assertEqual(p.service(gaps['a'],gaps,ALIASES,'mean')['reason'],'CROSS_REQUEST_LAUNCH_DELAY_MODEL_REQUIRED')
        mixed=dict(a=gpus['a'],b=gaps['b'])
        self.assertEqual(p.service(gpus['a'],mixed,ALIASES,'mean')['reason'],
                         'UNCLASSIFIED_GAP_GPU_INTERFERENCE_MODEL_REQUIRED')

    def test_isolated_replay_explicit_and_does_not_enable_cross_request_gaps(self):
        p=self.provider(allow_isolated_profiled_replay=True);rows=active(p)
        result=p.service(rows['a'],dict(a=rows['a']),ALIASES,'mean')
        self.assertTrue(result['known']);self.assertEqual(result['service_ms'],10)
        self.assertFalse(result['natural_timing_validated'])
        self.assertFalse(p.service(rows['a'],rows,ALIASES,'mean')['known'])

    def test_gpu_projection_uses_alias_identity_and_static_model_after_gap(self):
        p=self.provider(allow_analytical_hypothesis=True);rows=active(p,1)
        result=p.service(rows['a'],rows,ALIASES,'mean')
        self.assertTrue(result['known']);self.assertEqual(result['service_ms'],200)
        bad=copy.deepcopy(rows);bad['a']['segment_duration_ms']=1
        with self.assertRaises(ValueError):p.service(bad['a'],bad,ALIASES,'mean')

    def test_full_horizon_single_owner_replay_conserves_kernel_and_gap_time(self):
        p=self.provider(allow_isolated_profiled_replay=True)
        result=FullHorizonModel(ALIASES,p).forecast(dict(a=[dict(request_id='a',task='task',pool_id='input',arrival_ns=0)]),
                                                  0,1000000000,require_energy=False)
        self.assertEqual(result['finishes']['a']['finish_ns'],110000000)

    def test_training_conversion_retains_all_gaps_and_profiler_domain(self):
        with patch('tools.predictor_validation_20261027.ordered_kernel_profiles_v2.load_collection',return_value=collection()):
            result=convert_collection('fixture',[])
        segments=result['profiles'][0]['segments']
        self.assertEqual([s['kind'] for s in segments],['unclassified_gap','gpu','unclassified_gap','gpu','unclassified_gap'])
        self.assertAlmostEqual(sum(s['duration_ms']['profiled'] for s in segments),50/1e6)
        self.assertFalse(result['natural_timing_validated'])
        self.assertIsNone(result['concurrency_evidence'])

    def test_multistream_rejected_in_denominator_and_smoke_never_fitted(self):
        with patch('tools.predictor_validation_20261027.ordered_kernel_profiles_v2.load_collection',return_value=collection(stream=13)):
            result=convert_collection('fixture',[])
        self.assertEqual(result['denominator'],dict(request_count=1,converted_requests=0,rejected_requests=1))
        self.assertEqual(result['rejected_requests'][0]['reason'],'NON_SERIAL_STREAM_DAG_REQUIRED')
        for split in ('smoke','validation'):
            with patch('tools.predictor_validation_20261027.ordered_kernel_profiles_v2.load_collection',return_value=collection(split=split)):
                with self.assertRaises(ValueError):convert_collection('fixture',[])

    def test_unbound_occupancy_patch_rejected(self):
        with patch('tools.predictor_validation_20261027.ordered_kernel_profiles_v2.load_collection',return_value=collection()):
            with self.assertRaises(ValueError):
                convert_collection('fixture',[],occupancy_resolver=lambda k:dict(static_evidence=dict(source='other')))

    def test_fixture_and_context_are_explicit(self):
        with self.assertRaises(ValueError):OrderedKernelProfileProvider(document(),'context')
        with self.assertRaises(ValueError):OrderedKernelProfileProvider(document(),'other',allow_cpu_fixture=True)


if __name__=='__main__':unittest.main()
