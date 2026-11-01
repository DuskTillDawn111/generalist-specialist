"""Semantic progress parser/fit CPU fixtures; never physical measurements."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027.semantic_profiles import parse_semantic_request,semantic_context,fit_semantic


MS=1_000_000


def row(rid='r',alias='s',family='text',pool='x',start=0,end=10*MS):
    return {'request_id':rid,'alias':alias,'family':family,'task':'text','pool_id':pool,'handler_start_ns':start,'handler_end_ns':end}


def response(count=2):
    records=[{'forward_index':i,'start_ns':(2+3*i)*MS,'end_ns':(3+3*i)*MS} for i in range(count)]
    return {'semantic_forward_observations':{'schema':'gsdeploy.semantic_forward_observation.v1','request_id':'r',
                'clock':'time.perf_counter_ns','records':records,'instrumentation_errors':[],'gpu_synchronization_added':False,
                'token_scope':'test fixture','classification_scope':'mixed Python call'},
            'observed_stages':[{'id':'post_sync','start_ns':8*MS,'end_ns':9*MS}]}


def trace(index,requests,split='training'):
    return {'id':str(index),'requests':requests,'split':split,'execution_context_sha256':'CPU-fixture'}


class SemanticTests(unittest.TestCase):
    def test_continuous_semantics_include_sampling_and_sync(self):
        parsed=parse_semantic_request(row(),response())
        segments=parsed['segments']
        self.assertEqual([s['phase'] for s in segments],['prepare','prefill','decode','finalize'])
        self.assertEqual([(s['start_ns']//MS,s['end_ns']//MS) for s in segments],[(0,2),(2,5),(5,9),(9,10)])
        self.assertTrue(all(s['kind']=='mixed' and s['demand']=={'sm':None,'dram':None,'l2':None} for s in segments))
        self.assertFalse(parsed['progress_positioning']['known'])
        self.assertFalse(parsed['progress_positioning']['gpu_progress_known'])

    def test_single_forward_has_no_fictional_decode(self):
        parsed=parse_semantic_request(row(),response(1))
        self.assertEqual([s['phase'] for s in parsed['segments']],['prepare','prefill','finalize'])
        self.assertEqual(parsed['segments'][1]['end_ns'],9*MS)

    def test_ocr_does_not_require_language_forward(self):
        parsed=parse_semantic_request(row(family='ocr'),{})
        self.assertEqual([s['phase'] for s in parsed['segments']],['ocr_native'])

    def test_no_forward_or_instrumentation_error_is_unknown(self):
        for response_value in (response(0),response(1)):
            if response_value['semantic_forward_observations']['records']:
                response_value['semantic_forward_observations']['instrumentation_errors']=['hook failed']
            with self.assertRaises(ValueError):parse_semantic_request(row(),response_value)

    def test_boundary_reference_error_is_reported_without_gpu_claim(self):
        value=response();value['semantic_boundary_references']=[{'boundary':'first_forward_start','reference_ns':2*MS-100,'basis':'independent CPU fixture reference'}]
        report=parse_semantic_request(row(),value)['progress_positioning']
        self.assertTrue(report['known']);self.assertEqual(report['errors'][0]['error_ns'],100)
        self.assertFalse(report['gpu_progress_known'])

    def test_nonmonotonic_forward_rejected(self):
        value=response();value['semantic_forward_observations']['records'][1]['start_ns']=1
        with self.assertRaises(ValueError):parse_semantic_request(row(),value)

    def fit(self,traces):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'fixture.json';source.write_text(json.dumps(traces))
            with patch('tools.predictor_validation_20261027.semantic_profiles.semantic_traces',return_value=(traces,{})):
                return fit_semantic(source)

    def simple(self,rid,alias,duration):
        value=row(rid=rid,alias=alias,end=duration)
        value['segments']=[{'phase':'prefill','start_ns':0,'end_ns':duration}]
        value['progress_positioning']={'known':False,'reason':'CPU fixture no physical progress reference'}
        return value

    def test_analytical_shared_progress_fit_and_multiway_no_extrapolation(self):
        # Solo 10ms, two concurrent handlers 20ms -> speed .5 under SS.
        traces=[trace(0,[self.simple('a','s0',10*MS)]),trace(1,[self.simple('b','s0',10*MS)]),
                trace(2,[self.simple('c','s0',20*MS),self.simple('d','s1',20*MS)])]
        fitted=self.fit(traces)
        self.assertTrue(fitted['known']);self.assertEqual(fitted['coverage']['accepted_groups'],1)
        contexts=fitted['accepted_target_contexts']
        two=next(c for c in contexts if c['members'][0]['count']==2)
        self.assertAlmostEqual(two['progress'],.5)
        self.assertTrue(all(sum(m['count'] for m in c['members'])<=2 for c in contexts))
        self.assertFalse(fitted['core_enabled'])

    def test_validation_is_excluded(self):
        traces=[trace(0,[self.simple('a','s0',10*MS)]),trace(1,[self.simple('b','s0',10*MS)]),
                trace(2,[self.simple('c','s0',1000*MS)],split='validation')]
        fitted=self.fit(traces)
        self.assertEqual(fitted['solo_baselines'][0]['mean_ms'],10)
        self.assertEqual(fitted['coverage']['validation_batches_excluded'],1)

    def test_context_preserves_phase_and_multiplicity(self):
        value=row();first={'phase':'prefill'};second={'phase':'decode'}
        a=semantic_context([(value,first),(value,first)])
        b=semantic_context([(value,first),(value,second)])
        self.assertNotEqual(a[0],b[0]);self.assertEqual(a[1][0]['count'],2)


if __name__=='__main__':unittest.main()
