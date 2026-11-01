"""Small CPU counterexamples and exact subdomain comparisons with frozen v5_2."""
import copy
from fractions import Fraction as F
from itertools import product
import unittest

from tools.predictor_validation_20261027 import twoFIFO_single_owner_transport_intervals_v1 as m
from tools.predictor_validation_20261027.actual_worker_twoFIFO_reference_v5_2 import simulate,decode
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import Mark,Macro,RequestPlan
from tools.predictor_validation_20261027.twoFIFO_paired_transport_v3 import PairedTransportConstraints as C,UnknownTransportInterval
from tools.predictor_validation_20261027.worker_stdin_fifo_contract_v1 import TransportDraw


def slacks(reader=1,dequeue=2,processing=3):
    return dict(reader_slack_ns=reader,collector_dequeue_slack_ns=dequeue,collector_processing_slack_ns=processing)


def fixture():
    owner='owner';producer='owner:stdout';key=(owner,'text','input');plans=[];timing={}
    for i in range(2):
        plan=RequestPlan((Macro('prefill'+str(i),'prefill',6,(
            Mark('BEGIN'+str(i),0,0,'phase',producer,phase='prefill'),)),
            Macro('decode'+str(i),'decode',4,(
            Mark('DECODE'+str(i),0,0,'phase',producer,phase='decode'),
            Mark('RESULT'+str(i),3,0,'result',producer),
            Mark('CLEAR'+str(i),4,0,'clear',producer,clear_scope='request')))),0,producer)
        plans.append(plan);timing[plan]={mark.identity:slacks() for macro in plan.macros for mark in macro.marks}
        timing[plan]['worker_READY']=slacks(2,1,4)
    return dict(requests=[dict(request_id='r'+str(i),owner=owner,task='text',pool_id='input',arrival_ns=i) for i in range(2)],
        plans={key:tuple(plans)},rates={(owner,'prefill',0,0):F(2),(owner,'decode',0,0):F(1)},
        protocol_slacks=timing,transport_draws={(*key,0):C(0,0,6,8,6),(*key,1):C(0,0,5,10,-3)},
        domain_certificate=copy.deepcopy(m.CERTIFICATE),initial_sender_available_ns={owner:0},
        initial_worker_read_release_ns={owner:3},initial_protocol_state=dict(reader_free={'owner:stdout':20},
            collector_free=25,last_collector_arrival=20,pending_messages=[]),
        window_start_ns=0,window_end_ns=10,drain_end_ns=200)


def concrete(kw,us):
    kw=copy.deepcopy(kw);kw.pop('domain_certificate');draws={}
    for (key,draw),u in zip(kw['transport_draws'].items(),us):
        draws[key]=TransportDraw(draw.enqueue_after_arrival_ns,draw.sender_turn_extra_ns,
            draw.submit_to_write_record_ns,u,draw.observed_submit_to_START_ns-max(draw.previous_READY_minus_submit_ns,u))
    kw['transport_draws']=draws
    return decode(simulate(**kw))


def two_owner_endpoint_counterexample():
    """A legal paired source W=H=q=20 yields target B START=u in [0,20]."""
    rows=[]
    for u in (F(0),F(10),F(20)):
        requests=[];plans={};timing={};draws={};rates={}
        for owner in ('A','B'):
            producer=owner+':stdout';key=(owner,'text','input')
            plan=RequestPlan((Macro('macro','decode',10,(Mark('BEGIN',0,0,'phase',producer,phase='decode'),
                Mark('RESULT',10,0,'result',producer))),),0,producer)
            plans[key]=plan;timing[plan]={identity:slacks(0,0,0) for identity in ('BEGIN','RESULT','worker_READY')}
            requests.append(dict(request_id=owner,owner=owner,task='text',pool_id='input',arrival_ns=0))
            draws[(*key,0)]=TransportDraw(0,0,20 if owner=='B' else 0,u if owner=='B' else 0,0)
            rates.update({(owner,'decode',0,0):F(1),(owner,'decode',1,1):F(1,2)})
        world=decode(simulate(requests,plans,rates,protocol_slacks=timing,transport_draws=draws,
            initial_sender_available_ns={'A':0,'B':0},initial_worker_read_release_ns={'A':0,'B':0},
            initial_protocol_state=dict(reader_free={},collector_free=0,last_collector_arrival=0,pending_messages=[]),
            window_start_ns=0,window_end_ns=1,drain_end_ns=100))
        rows.append(dict(u_ns=u,START_ns=world['starts'],RESULT_reader_ns=world['RESULT_receipts'],
            P95_ns=m.p95(list(world['RESULT_receipts'].values()))))
    return rows


class SingleOwnerIntervalTests(unittest.TestCase):
    def test_complete_reader_and_collector_fifo_not_constant_result_lag(self):
        kw=fixture();value=m.evaluate(**kw)
        self.assertEqual(value['status'],'EXACT_SINGLE_OWNER_PAIRED_TRANSPORT_COHORT_ENVELOPE')
        self.assertTrue(value['simultaneously_attainable_componentwise_extrema'])
        for name in ('lower_witness','upper_witness'):
            witness=value[name];us=[witness['requests'][r['request_id']]['extremizing_u_ns'] for r in kw['requests']]
            world=concrete(kw,us)
            self.assertEqual({rid:r['START_ns'] for rid,r in witness['requests'].items()},world['starts'])
            self.assertEqual({rid:r['READY_ns'] for rid,r in witness['requests'].items()},world['READY_occurrences'])
            self.assertEqual({rid:r['RESULT_reader_ns'] for rid,r in witness['requests'].items()},world['RESULT_receipts'])
            self.assertEqual({rid:r['RESULT_processed_ns'] for rid,r in witness['requests'].items()},world['RESULT_processed'])
            for message in witness['protocol_messages']:
                self.assertGreaterEqual(message['reader_ns'],message['occurrence_ns'])
                self.assertGreaterEqual(message['processing_ns'],message['dequeue_ns'])
        # Nonzero initial reader/collector backlogs are propagated, not ignored.
        first=value['lower_witness']['requests']['r0']
        self.assertGreater(first['RESULT_reader_ns']-first['START_ns'],6+1)
        self.assertGreater(first['RESULT_processed_ns'],first['RESULT_reader_ns'])

    def test_dense_feasible_grid_is_inside_sharp_jointly_attained_bounds(self):
        kw=fixture();bounds=m.evaluate(**kw);count=0
        for us in product((F(0),F(3,2),F(3),F(9,2),F(6)),(F(0),F(5,4),F(5,2),F(15,4),F(5))):
            world=concrete(kw,us);latencies=[]
            for row in bounds['requests']:
                rid=row['request_id'];lat=world['RESULT_receipts'][rid]-next(r['arrival_ns'] for r in kw['requests'] if r['request_id']==rid)
                self.assertLessEqual(row['reader_E2E_ns_bounds'][0],lat);self.assertGreaterEqual(row['reader_E2E_ns_bounds'][1],lat)
                latencies.append(lat)
            p=m.p95(latencies);a,b=bounds['complete_cohort_P95_ns_bounds'];self.assertLessEqual(a,p);self.assertLessEqual(p,b)
            count+=1
        self.assertEqual(count,25)

    def test_paired_nonsingleton_preserves_entire_cohort_when_frozen_engine_is_unknown(self):
        kw=fixture();engine=dict(kw);engine.pop('domain_certificate')
        with self.assertRaises(UnknownTransportInterval):simulate(**engine)
        out=m.evaluate(**kw)
        self.assertEqual(out['offered_request_ids'],['r0','r1']);self.assertEqual(len(out['requests']),2)
        self.assertTrue(any(row['START_ns_bounds'][0]!=row['START_ns_bounds'][1] for row in out['requests']))
        self.assertIsNotNone(out['complete_cohort_P95_ns_bounds']);self.assertFalse(out['latent_probability_assigned'])

    def test_endpoint_enumeration_misses_two_owner_p95_lower_bound(self):
        rows=two_owner_endpoint_counterexample()
        self.assertEqual([r['P95_ns'] for r in rows],[F(20),F(39,2),F(29)])
        self.assertLess(rows[1]['P95_ns'],min(rows[0]['P95_ns'],rows[2]['P95_ns']))

    def test_outside_subdomain_keeps_every_request_unknown(self):
        for mutate in (lambda kw:kw['requests'][1].update(owner='external'),
                lambda kw:kw['domain_certificate'].update(no_external_visible_or_future_producers=False),
                lambda kw:kw['initial_protocol_state'].update(pending_messages=[{'unsettled':True}]),
                lambda kw:kw['rates'].pop(('owner','decode',0,0)),
                lambda kw:kw['transport_draws'].pop(('owner','text','input',1))):
            kw=fixture();mutate(kw);out=m.evaluate(**kw)
            self.assertEqual(out['status'],'UNKNOWN_OUTSIDE_CERTIFIED_SINGLE_OWNER_SUBDOMAIN')
            self.assertEqual(out['offered_requests'],2);self.assertEqual(len(out['requests']),2)
            self.assertTrue(all(row['reader_E2E_ns_bounds'] is None for row in out['requests']))

    def test_fixed_drain_censoring_does_not_create_completed_subset_p95(self):
        kw=fixture();full=m.evaluate(**kw)
        a,b=full['requests'][0]['RESULT_reader_ns_bounds']
        kw['drain_end_ns']=a;out=m.evaluate(**kw)
        self.assertIsNone(out['complete_cohort_P95_ns_bounds']);self.assertEqual(len(out['requests']),2)
        self.assertIsNotNone(out['untruncated_model_complete_cohort_P95_ns_bounds'])
        self.assertTrue(out['P95_not_inferred_from_completed_subset'])
        self.assertFalse(out['original_fixed_drain_P95_defined_for_all_feasible_worlds'])

    def test_exact_original_score_terms_preserve_unknown_resource_utility(self):
        kw=fixture();kw.update(window_end_ns=10_000_000_000,drain_end_ns=70_000_000_000,
            scoring_plan=dict(aliases={'owner':dict(family='text',supported_tasks=['text'])},workload_rps={'text':1},
                latency_reference_ms={'text':1},window_seconds=10,startup_amortization_seconds=3600,
                memory_budget_mib=1000,memory_safe_factor=1,energy_reference_watts=10,weights={k:1/6 for k in 'QTLP ME'.replace(' ','')}),
            quality={'text':{'text':.25}})
        out=m.evaluate(**kw)
        self.assertEqual(out['original_QTLP_term_bounds']['Q'],[.25,.25])
        self.assertIsNone(out['full_U_bounds']);self.assertFalse(out['actual_native_quality_validated'])
        self.assertTrue(all(score['full_U'] is None for score in out['original_score_at_componentwise_extrema']))


if __name__=='__main__':unittest.main()
