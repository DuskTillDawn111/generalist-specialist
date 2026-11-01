from fractions import Fraction
import copy
import unittest

from tools.predictor_validation_20261027 import joint_request_response_transport_candidate_v2 as transport
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import simulate


def mark(identity,at,action,*,phase=None,source=None,occurrence=None,complete=None,delay=1):
    return dict(identity=identity,reference_work_at=transport.fraction_record(Fraction(at)),
        reader_delay_ns=delay,dequeue_extra_ns=0,processing_extra_ns=0,action=action,phase=phase,
        source_macro_id=source,occurrence_macro_id=occurrence,
        query_complete=complete,event_reference_phase_id=source if action=='query' else None,
        clear_scope='request' if action=='clear' else None)


def vector(identity='a',capture='one',scale=1):
    nodes=[]
    for ident,phase,work,marks in [
        ('entry','entry',2,[mark('native_BEGIN',0,'phase',phase='entry')]),
        ('prefill_0000','prefill',10,[mark('prefill_BEGIN',0,'phase',phase='prefill'),
            mark('H',4,'diagnostic'),mark('false_query',5,'query',source='prefill_0000',complete=False)]),
        ('native_end_to_worker_ready','ready_tail',8,[mark('native_END',0,'clear'),
            mark('final_true_query',2,'query',source='prefill_0000',occurrence='native_end_to_worker_ready',complete=True),
            mark('RESULT',6,'result')])]:
        for m in marks:
            m['reference_work_at']=transport.fraction_record(transport.fraction_read(m['reference_work_at'])*scale)
        nodes.append(dict(macro_id=ident,phase=phase,reference_work_ns=transport.fraction_record(work*scale),marks=marks))
    return dict(vector_id=identity,training_capture_sha256=capture,public_key=['g2','text','exact-content'],
        nodes=nodes,ready_lags=dict(reader_delay_ns=1,dequeue_extra_ns=0,processing_extra_ns=0),
        dispatch_to_worker_start_ns=0)


class JointRequestTransportTests(unittest.TestCase):
    def test_piecewise_reference_progress_uses_only_own_occurrence_segment(self):
        macro=dict(left_ns=100,right_ns=130,causal_covariate_path=[
            dict(left_ns=100,right_ns=110,state=dict(peer_visible_macro_owner_count=0,peer_same_visible_macro_phase_owner_count=0)),
            dict(left_ns=110,right_ns=130,state=dict(peer_visible_macro_owner_count=1,peer_same_visible_macro_phase_owner_count=0))])
        beta=[.2,-.1];rate=transport.response_rate(macro['causal_covariate_path'][1]['state'],beta)
        self.assertEqual(transport.progress(macro,125,beta),10+15*rate)
        with self.assertRaises(ValueError):transport.progress(macro,131,beta)

    def test_equal_capture_selection_precedes_unequal_row_counts(self):
        law=dict(captures=['one','two'],vectors=[vector(str(i)) for i in range(9)]+[vector('rare','two')])
        self.assertEqual(transport.choose_vector(law,('g2','text','exact-content'),capture_quantile=.75,
            within_capture_quantile=0)['vector']['vector_id'],'rare')

    def test_whole_request_joint_vector_is_not_independent_macro_resampling(self):
        a=vector('short','one',1);b=vector('long','one',3)
        law=dict(captures=['one'],vectors=[a,b]);before=copy.deepcopy(law)
        drawn=transport.choose_vector(law,a['public_key'],capture_quantile=0,within_capture_quantile=.9)['vector']
        self.assertEqual([transport.fraction_read(n['reference_work_ns']) for n in drawn['nodes']],[6,30,24])
        drawn['nodes'][0]['marks'][0]['reader_delay_ns']=100
        self.assertEqual(law,before)

    def test_different_content_same_length_does_not_get_template(self):
        law=dict(captures=['one'],vectors=[vector()])
        self.assertEqual(transport.choose_vector(law,('g2','text','other-content'),capture_quantile=0,
            within_capture_quantile=0)['status'],'UNKNOWN_EXACT_PUBLIC_INPUT_IN_SELECTED_CAPTURE')

    def test_bridge_keeps_final_query_in_tail_with_source_event_identity(self):
        plan=transport.to_reference_plan(vector(),public_dispatch_ordinal=4)
        query=next(m for m in plan.macros[-1].marks if m.action=='query')
        self.assertEqual(query.work_ns,2)
        self.assertIn('prefill_0000',query.event_id)
        self.assertEqual(plan.macros[-1].identity,'native_end_to_worker_ready')
        self.assertEqual(query.producer_id,plan.ready_producer_id)
        self.assertEqual(sum(m.action=='result' for n in plan.macros for m in n.marks),1)
        self.assertTrue(all(m.reader_delay_ns is not None for node in plan.macros for m in node.marks))

    def test_missing_or_negative_delay_is_rejected_instead_of_zero(self):
        for clocks in ((1,2,3,None),(3,2,4,5),(-1,2,3,4)):
            with self.assertRaises(ValueError):transport.lags(*clocks)
        for mode in ('mark-negative','ready-missing','ready-negative','handoff-missing','handoff-bool'):
            bad=vector()
            if mode=='mark-negative':bad['nodes'][0]['marks'][0]['dequeue_extra_ns']=-1
            elif mode=='ready-missing':bad['ready_lags'].pop('dequeue_extra_ns')
            elif mode=='ready-negative':bad['ready_lags']['processing_extra_ns']=-1
            elif mode=='handoff-missing':bad.pop('dispatch_to_worker_start_ns')
            else:bad['dispatch_to_worker_start_ns']=True
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                transport.to_reference_plan(bad,public_dispatch_ordinal=0)

    def test_dispatch_handoff_and_RESULT_reader_have_distinct_clocks(self):
        drawn=vector();drawn['dispatch_to_worker_start_ns']=7
        result_mark=next(m for n in drawn['nodes'] for m in n['marks'] if m['action']=='result')
        result_mark.update(reader_delay_ns=1,dequeue_extra_ns=2,processing_extra_ns=3)
        # READY must retain the same producer's delivery order after RESULT.
        drawn['ready_lags'].update(reader_delay_ns=6,dequeue_extra_ns=1,processing_extra_ns=1)
        plan=transport.to_reference_plan(drawn,public_dispatch_ordinal=0)
        public=('g2','text','exact-content')
        output=simulate([dict(request_id='fresh',owner='g2',task='text',pool_id='exact-content',arrival_ns=0)],
            {public:plan},transport.rate_table([0,0]),window_start_ns=0,window_end_ns=1,drain_end_ns=100,
            producer_fifo=('g2:modeled_stdout_lifetime',))
        self.assertEqual(output['starts']['fresh'],transport.fraction_record(7))
        self.assertEqual(output['RESULT_receipts']['fresh'],transport.fraction_record(26))

    def test_bridge_runs_complete_future_FIFO_without_historical_request_lookup(self):
        public=('g2','text','exact-content')
        requests=[dict(request_id='new:'+str(i),owner=public[0],task=public[1],pool_id=public[2],arrival_ns=i*2) for i in range(2)]
        plans={public:tuple(transport.to_reference_plan(vector(scale=i+1),public_dispatch_ordinal=i) for i in range(2))}
        result=simulate(requests,plans,transport.rate_table([0,0]),window_start_ns=0,window_end_ns=5,drain_end_ns=100,
            producer_fifo=('g2:modeled_stdout_lifetime',))
        self.assertEqual(result['unfinished_request_ids'],[])
        self.assertEqual(result['offered_requests'],2)
        self.assertIsNone(result['global_admission_cap'])
        self.assertFalse(result['query_success_ends_macro'])


if __name__=='__main__':unittest.main()
