import copy
from fractions import Fraction
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_owned_training_review_v2 import fixture
from tools.predictor_validation_20261027 import twoFIFO_joint_request_law_v2 as m
from tools.predictor_validation_20261027.twoFIFO_composable_macros_v2 import request_rows


def source_fixture():
    plan,raw=fixture()
    # Serialize the shared parent collector genuinely across all producers;
    # fixture work occurrences stay parallel and distinct from availability.
    messages=[]
    for row in raw['priming_request_records']+raw['request_records']:
        for event in row['execution_events']:
            messages.append((event['perf_ns'],row,event))
        messages.append((row['worker_response']['end_perf_ns'],row,'RESULT'))
        messages.append((row['previous_handler_next_request_ready']['perf_ns'],row,'READY'))
    reader_free={};collector=0
    for at,row,event in sorted(messages,key=lambda item:(item[0],item[1]['model'],str(item[2]))):
        reader=max(at+2,reader_free.get(row['model'],0)+1);reader_free[row['model']]=reader
        dequeue=max(reader+1,collector);process=dequeue+1;collector=process
        if event=='RESULT':
            row.update(completion_time_ns=reader,response_reader_event_enqueue_time_ns=reader,
                response_collector_dequeued_time_ns=dequeue,response_collector_processing_time_ns=process)
        else:
            target=row['previous_handler_next_request_ready'] if event=='READY' else event
            target.update(received_time_ns=reader,collector_dequeued_time_ns=dequeue,collector_processing_time_ns=process)
    requests,events=request_rows(raw,plan,'raw-provenance')
    features=m.response.FEATURES
    doc=dict(schema=m.response.intake.MACRO_SCHEMA,validation_labels_read=False,capture_id='block',attempt_id='shard',requests=requests)
    rows=m.response.intake.prepared_rows([doc],features)
    fit=dict(features=list(features),coefficients=[0.]*12,baselines=[dict(group=list(r['group']),log_reference_ms=0.) for r in rows])
    derived=m.derive(m.source_messages(raw));slacks={x['identity'][2]:x['protocol_slacks'] for x in derived if x['identity'][:2]==('text#0','first')}
    row=raw['request_records'][0];previous=raw['priming_request_records'][-1]
    constraints=m.transport_constraints(row,previous)
    vector=m.request_vector(requests[0],row,fit,slacks,constraints)
    vector['training_block_id']='training-block-1'
    return plan,raw,requests,fit,vector


class JointLawTests(unittest.TestCase):
    def test_whole_attempt_protocol_includes_priming_and_all_original_messages(self):
        plan,raw,requests,fit,vector=source_fixture();messages=m.source_messages(raw)
        self.assertEqual(len(messages),sum(len(r['execution_events'])+2 for r in raw['priming_request_records']+raw['request_records']))
        self.assertGreater(len(messages),sum(len(r['execution_events'])+2 for r in raw['request_records']))
        self.assertEqual(len(m.derive(messages)),len(messages))
        first=raw['request_records'][0]
        self.assertEqual(sum(len(n['marks']) for n in vector['nodes'])+1,len(first['execution_events'])+2)
        self.assertEqual([n['phase'] for n in vector['nodes']],['entry','prefill','decode','ready_tail'])
        self.assertEqual(vector['dispatch_to_worker_start_ns'],0)
        # START is a real zero-work protocol mark in the entry node.
        self.assertTrue(any(x['identity']=='event:0' and m.fraction_read(x['reference_work_at'])==0 for x in vector['nodes'][0]['marks']))

    def test_missing_busy_transport_is_unknown_and_explicit_pair_compiles_zero_dispatch(self):
        *_,vector=source_fixture()
        self.assertTrue(m.compile_draw_v2(vector,0)['status'].startswith('UNKNOWN'))
        c=vector['transport_constraints'];line=Fraction(min(c['submit_to_write_record_ns'],c['observed_submit_to_START_ns']))
        extra=Fraction(c['observed_submit_to_START_ns'])-max(Fraction(c['previous_READY_minus_submit_ns']),line)
        draw=m.TransportDraw(Fraction(c['enqueue_after_arrival_ns']),Fraction(c['sender_turn_extra_ns']),
            Fraction(c['submit_to_write_record_ns']),line,extra)
        support=dict(status='SUPPLIED_FEASIBLE_LATENT_SCENARIO',vector_id=vector['vector_id'],public_key_ordinal=0,draw=draw)
        result=m.compile_draw_v2(vector,0,support)
        self.assertEqual(result['plan'].dispatch_to_worker_start_ns,0)
        self.assertEqual(len(result['protocol_slacks']),sum(len(n['marks']) for n in vector['nodes'])+1)
        self.assertFalse(result['transport_scenario_probability_qualified'])
        wrong=dict(support,public_key_ordinal=1)
        with self.assertRaises(ValueError):m.compile_draw_v2(vector,0,wrong)
        bad=m.TransportDraw(draw.enqueue_after_arrival_ns,draw.sender_turn_extra_ns,draw.submit_to_write_record_ns,
            draw.submit_to_stdin_line_available_ns,draw.worker_release_to_START_extra_ns+1)
        with self.assertRaises(ValueError):m.compile_draw_v2(vector,0,dict(support,draw=bad))

    def test_equal_whole_block_draw_moves_entire_request_without_historical_lookup(self):
        *_,vector=source_fixture();vectors=[]
        for block in range(1,4):
            for ordinal in range(block):
                v=copy.deepcopy(vector);v.update(training_block_id='training-block-'+str(block),vector_id=str(block)+':'+str(ordinal));vectors.append(v)
        law=dict(schema=m.SCHEMA,blocks=['training-block-'+str(i) for i in (1,2,3)],vectors=vectors)
        selected=[]
        for q in (.1,.4,.8):
            result=m.choose_vector(law,vector['public_key'],7,block_quantile=q,within_block_quantile=.9)
            selected.append(result['block_id']);self.assertEqual(result['block_probability'],1/3)
            self.assertEqual(result['public_key_ordinal'],7);self.assertEqual(result['vector']['nodes'],vector['nodes'])
        self.assertEqual(len(set(selected)),3)
        self.assertTrue(m.choose_vector(law,('text#0','text','unseen'),0,block_quantile=.1,within_block_quantile=0)['status'].startswith('UNKNOWN'))

    def test_protocol_cross_request_order_conflict_is_rejected(self):
        plan,raw,*_=source_fixture()
        ready=raw['priming_request_records'][-1]['previous_handler_next_request_ready']
        ready.update(received_time_ns=10**9,collector_dequeued_time_ns=10**9+1,collector_processing_time_ns=10**9+2)
        with self.assertRaises(ValueError):m.derive(m.source_messages(raw))

    def test_twelve_attempt_builder_retains_every_vector_and_checks_every_fold(self):
        plan,raw,requests,fit,vector=source_fixture();plans=[];raws=[];docs=[]
        for block in range(1,4):
            for shard in range(1,5):
                identity=dict(capture_id='training-block-'+str(block),attempt_id='block-'+str(block)+'-shard-'+str(shard),execution_id=str(block)+str(shard))
                p=copy.deepcopy(plan);p.update(identity);r=copy.deepcopy(raw);r.update(identity)
                # Real four-owner priming has each resident as predecessor;
                # this fixture originally primes text0 only. Add a real past
                # peer predecessor without manufacturing an empty owner state.
                r['priming_request_records'][0]['model']='text#1'
                docs.append(dict(requests=copy.deepcopy(requests),**identity));plans.append(p);raws.append(r)
        units=['training-block-'+str(i) for i in (1,2,3)]
        denominator=dict(measured_requests=546)
        qualified=dict(response_law_compilation_eligible=True,features=list(m.response.FEATURES),
            fitted_root_numerical_qualification=True,denominator=denominator,
            full_fit=dict(fit,numerical_qualification=True,whole_blocks=units),whole_block_leaveout=[dict(heldout_block_id=heldout,
                training_fit=dict(fit,numerical_qualification=True,whole_blocks=[u for u in units if u!=heldout]),
                evaluation=dict(summary=dict(unknown_rows=0))) for heldout in units])
        with patch.object(m.response.intake,'complete_documents',return_value=denominator):
            law=m.build_joint_law_v2(docs,plans,raws,qualified)
            self.assertEqual(len(law['vectors']),36);self.assertEqual(len(law['whole_attempt_protocol_decompositions']),12)
            self.assertEqual(law['block_weights'],[1/3]*3)
            self.assertTrue(all(d['priming_history_requests']==18 for d in law['whole_attempt_protocol_decompositions']))
            fold=m.build_joint_law_v2(docs,plans,raws,qualified,heldout_block_id='training-block-3')
            self.assertEqual(len(fold['vectors']),24);self.assertEqual(fold['block_weights'],[.5,.5])
            self.assertEqual(fold['donor_startup_attempts'],8)
            self.assertTrue(all(v['training_block_id']!='training-block-3' for v in fold['vectors']))
            bad=copy.deepcopy(qualified);bad['whole_block_leaveout'][0]['training_fit']['numerical_qualification']=False
            with self.assertRaises(ValueError):m.build_joint_law_v2(docs,plans,raws,bad)


if __name__=='__main__':unittest.main()
