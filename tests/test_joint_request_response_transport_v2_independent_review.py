"""Independent compiler, latent-draw and generated-world counterexamples."""
import copy
from fractions import Fraction as F
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import joint_request_response_transport_candidate_v2 as t
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import simulate


def value(x):return F(x['numerator'],x['denominator'])


def vector(owner='a',capture='one',work=20):
    def mark(identity,at,action,phase=None):
        return dict(identity=identity,reference_work_at=t.fraction_record(at),
            reader_delay_ns=1,dequeue_extra_ns=1,processing_extra_ns=1,
            action=action,phase=phase,event_reference_phase_id='original_prefill' if action=='query' else None,
            query_complete=True if action=='query' else None,clear_scope=None)
    return dict(public_key=[owner,'text','content'],training_capture_sha256=capture,
        dispatch_to_worker_start_ns=2,ready_lags=dict(reader_delay_ns=1,dequeue_extra_ns=1,processing_extra_ns=1),
        nodes=[dict(macro_id='forward',phase='prefill',reference_work_ns=t.fraction_record(work),
            marks=[mark('begin',0,'phase','prefill'),mark('query',5,'query'),mark('RESULT',work,'result')])])


def run(vectors,cutoff=None,beta=(0,0),arrivals=None):
    rows=[];plans={}
    for i,v in enumerate(vectors):
        owner,task,pool=v['public_key'];rows.append(dict(request_id='new:'+str(i),owner=owner,
            task=task,pool_id=pool,arrival_ns=0 if arrivals is None else arrivals[i]))
        plans[(owner,task,pool)]=t.to_reference_plan(v,public_dispatch_ordinal=0)
    return simulate(rows,plans,t.rate_table(beta),window_start_ns=0,window_end_ns=10,
        drain_end_ns=100,cutoff_ns=cutoff,producer_fifo=tuple(v['public_key'][0]+':modeled_stdout_lifetime' for v in vectors))


class JointTransportIndependentReview(unittest.TestCase):
    def test_missing_READY_leg_and_compensating_negative_mark_leg_reject(self):
        a=vector();del a['ready_lags']['dequeue_extra_ns']
        with self.assertRaises(ValueError):t.to_reference_plan(a,public_dispatch_ordinal=0)
        a=vector();a['nodes'][0]['marks'][0].update(dequeue_extra_ns=-1,processing_extra_ns=3)
        with self.assertRaises(ValueError):t.to_reference_plan(a,public_dispatch_ordinal=0)

    def test_bool_lag_and_missing_handoff_cannot_be_zero_or_one_defaults(self):
        for mode in ('bool_leg','bool_handoff','missing_handoff'):
            a=vector()
            if mode=='bool_leg':a['ready_lags']['reader_delay_ns']=True
            elif mode=='bool_handoff':a['dispatch_to_worker_start_ns']=False
            else:del a['dispatch_to_worker_start_ns']
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                t.to_reference_plan(a,public_dispatch_ordinal=0)

    def test_selected_capture_missing_key_is_UNKNOWN_without_other_capture_fallback(self):
        law=dict(captures=['one','two'],vectors=[vector('a','one'),vector('b','two')])
        draw=t.choose_vector(law,('a','text','content'),capture_quantile=.9,within_capture_quantile=.1)
        self.assertEqual(draw['status'],'UNKNOWN_EXACT_PUBLIC_INPUT_IN_SELECTED_CAPTURE')

    def test_common_capture_latent_for_different_keys_and_independent_mutable_draws(self):
        law=dict(captures=['one','two'],vectors=[vector(o,c) for c in ('one','two') for o in ('a','b')])
        before=copy.deepcopy(law)
        a=t.choose_vector(law,('a','text','content'),capture_quantile=.9,within_capture_quantile=0)
        b=t.choose_vector(law,('b','text','content'),capture_quantile=.9,within_capture_quantile=0)
        self.assertEqual(a['common_training_capture_latent'],b['common_training_capture_latent'])
        a['vector']['ready_lags']['reader_delay_ns']=999
        self.assertEqual(law,before)

    def test_peer_paths_generated_by_composition_change_work_completion(self):
        import math
        alone=run([vector('a')],beta=(math.log(2),0))
        together=run([vector('a'),vector('b')],beta=(math.log(2),0))
        self.assertEqual(value(alone['RESULT_receipts']['new:0']),23)
        self.assertEqual(value(together['RESULT_receipts']['new:0']),40)
        self.assertTrue(any(p['evidence_key'][1]==1 for p in together['response_exposure']))
        self.assertEqual(sum((value(p['right_ns'])-value(p['left_ns']))*value(p['rate'])
            for p in together['response_exposure'] if p['owner']=='a'),20)

    def test_true_query_never_finishes_work_and_RESULT_reader_is_not_processing(self):
        result=run([vector()],cutoff=10)
        self.assertFalse(result['RESULT_receipts'])
        self.assertEqual(result['active_response_states']['a']['macro_id'],'forward')
        self.assertFalse(result['query_success_ends_macro'])
        result=run([vector()],cutoff=23)
        self.assertEqual(value(result['RESULT_receipts']['new:0']),23)
        self.assertFalse(result['RESULT_processed'])
        self.assertEqual(result['unfinished_request_ids'],[])
        self.assertEqual(value(result['pending_notifications'][0]['available_ns']),25)

    def test_cutoff_keeps_public_future_arrival_and_handoff_without_macro_work(self):
        result=run([vector('a'),vector('b')],cutoff=1,arrivals=[0,9])
        self.assertEqual(result['offered_requests'],2)
        self.assertFalse(result['starts']);self.assertFalse(result['response_exposure'])
        self.assertEqual(result['unfinished_request_ids'],['new:0','new:1'])
        self.assertEqual(result['future_public_arrivals'][0]['request_id'],'new:1')

    def test_joint_vector_and_inputs_not_mutated_by_compile_or_simulation(self):
        a=vector();before=copy.deepcopy(a)
        plan=t.to_reference_plan(a,public_dispatch_ordinal=7);run([a])
        self.assertEqual(a,before)
        self.assertEqual(plan.macros[0].marks[1].event_id,'a:draw:7:original_prefill:stream_end')

    def test_inverted_notice_FIFO_rejects_entire_generated_world(self):
        a=vector();a['nodes'][0]['marks'][0]['processing_extra_ns']=40
        with self.assertRaisesRegex(UnknownEvidence,'PRODUCER_FIFO'):run([a])

    def test_missing_work_or_mark_beyond_work_never_gets_an_imputed_template(self):
        a=vector();del a['nodes'][0]['reference_work_ns']
        with self.assertRaises((ValueError,KeyError)):t.to_reference_plan(a,public_dispatch_ordinal=0)
        a=vector();a['nodes'][0]['marks'][-1]['reference_work_at']=t.fraction_record(100)
        with self.assertRaises(ValueError):t.to_reference_plan(a,public_dispatch_ordinal=0)


if __name__=='__main__':unittest.main()
