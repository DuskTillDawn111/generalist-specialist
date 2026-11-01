"""Coverage counterexamples against the frozen concrete two-FIFO simulator."""
import copy
from fractions import Fraction as F
from itertools import product
import unittest

from tools.predictor_validation_20261027 import twoFIFO_multiowner_outer_envelope_v1 as m
from tools.predictor_validation_20261027.actual_worker_twoFIFO_reference_v5_2 import simulate, decode
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import Mark, Macro, RequestPlan
from tools.predictor_validation_20261027.twoFIFO_paired_transport_v3 import PairedTransportConstraints as C
from tools.predictor_validation_20261027.worker_stdin_fifo_contract_v1 import TransportDraw


def fixture(owners=('A', 'B', 'C'), queued=True):
    requests=[]; plans={}; timing={}; transport={}; rates={}
    for i, owner in enumerate(owners):
        producer=owner+':stdout'; key=(owner,'text','input')
        plan=RequestPlan((Macro('work','decode',F(10+i),(
            Mark('BEGIN',0,0,'phase',producer,phase='decode'),
            Mark('RESULT',F(9+i),0,'result',producer),
            Mark('CLEAR',F(10+i),0,'clear',producer,clear_scope='request'))),),0,producer)
        plans[key]=plan
        timing[plan]={identity:dict(reader_slack_ns=F(i,2),
            collector_dequeue_slack_ns=F(1,3), collector_processing_slack_ns=F(i+1,5))
            for identity in ('BEGIN','RESULT','CLEAR','worker_READY')}
        for ordinal in range(2 if queued else 1):
            requests.append(dict(request_id=owner+str(ordinal),owner=owner,task='text',pool_id='input',arrival_ns=ordinal))
            transport[(*key,ordinal)]=C(0,0,20,20,20) if ordinal==0 else C(0,0,4,5,-3)
        for n in range(len(owners)):
            for same in range(n+1):rates[(owner,'decode',n,same)]=F(4,4+2*n+same)
    return dict(requests=requests,plans=plans,rates=rates,protocol_slacks=timing,
        transport_draws=transport,domain_certificate=copy.deepcopy(m.CERTIFICATE),
        initial_sender_available_ns={o:0 for o in owners},
        initial_worker_read_release_ns={o:0 for o in owners},
        initial_protocol_state=dict(reader_free={o+':stdout':7 for o in owners},
            collector_free=9,last_collector_arrival=7,pending_messages=[]),
        window_start_ns=0,window_end_ns=10,drain_end_ns=1000)


def concrete(kw,coordinates):
    kw=copy.deepcopy(kw);kw.pop('domain_certificate');draws={}
    for (key,d),u in zip(kw['transport_draws'].items(),coordinates):
        draws[key]=TransportDraw(d.enqueue_after_arrival_ns,d.sender_turn_extra_ns,
            d.submit_to_write_record_ns,u,d.observed_submit_to_START_ns-max(d.previous_READY_minus_submit_ns,u))
    kw['transport_draws']=draws
    return decode(simulate(**kw))


class MultiownerEnvelopeTests(unittest.TestCase):
    def assert_contains(self, out, world, requests):
        arrivals={r['request_id']:r['arrival_ns'] for r in requests}
        for row in out['requests']:
            rid=row['request_id']
            for field, key in [('START_ns_bounds','starts'),('READY_ns_bounds','READY_occurrences'),
                               ('RESULT_reader_ns_bounds','RESULT_receipts')]:
                lo,hi=row[field]
                self.assertLessEqual(lo,world[key][rid]);self.assertLessEqual(world[key][rid],hi)
        actual=m.p95([at-arrivals[rid] for rid,at in world['RESULT_receipts'].items()])
        lo,hi=out['untruncated_model_complete_cohort_P95_ns_bounds']
        self.assertLessEqual(lo,actual);self.assertLessEqual(actual,hi)

    def test_three_owners_queued_interiors_backlogs_and_fractional_rates(self):
        kw=fixture();out=m.evaluate(**kw);count=0
        for first in product((F(0),F(7),F(20)),repeat=3):
            for second in (F(0),F(3,2),F(4)):
                u=[v for first_u in first for v in (first_u,second)]
                self.assert_contains(out,concrete(kw,u),kw['requests']);count+=1
        self.assertEqual(count,81)
        self.assertFalse(out['envelope_is_sharp'])
        self.assertFalse(out['extrema_simultaneously_attainable_claimed'])
        self.assertFalse(out['latent_point_selected'])

    def test_interior_p95_counterexample_is_covered_without_endpoint_claim(self):
        kw=fixture(('A','B'),queued=False)
        kw['plans']={};kw['protocol_slacks']={};kw['rates']={}
        for owner in ('A','B'):
            producer=owner+':stdout'
            plan=RequestPlan((Macro('work','decode',10,(
                Mark('BEGIN',0,0,'phase',producer,phase='decode'),
                Mark('RESULT',10,0,'result',producer))),),0,producer)
            kw['plans'][(owner,'text','input')]=plan
            kw['protocol_slacks'][plan]={identity:dict(reader_slack_ns=0,
                collector_dequeue_slack_ns=0,collector_processing_slack_ns=0)
                for identity in ('BEGIN','RESULT','worker_READY')}
            kw['rates'].update({(owner,'decode',0,0):F(1),
                (owner,'decode',1,0):F(1,2),(owner,'decode',1,1):F(1,2)})
        kw['transport_draws'][('A','text','input',0)]=C(0,0,0,0,0)
        kw['initial_protocol_state']=dict(reader_free={},collector_free=0,last_collector_arrival=0,pending_messages=[])
        out=m.evaluate(**kw)
        actual={}
        for u in (F(0),F(5),F(10),F(15),F(20)):
            world=concrete(kw,[F(0),u]);self.assert_contains(out,world,kw['requests'])
            actual[u]=m.p95(list(world['RESULT_receipts'].values()))
        self.assertEqual([actual[F(u)] for u in (0,10,20)],[F(20),F(39,2),F(29)])

    def test_missing_rate_or_external_domain_keeps_full_unknown_denominator(self):
        for mutate in (lambda k:k['rates'].pop(('A','decode',2,1)),
                lambda k:k['domain_certificate'].update(initial_visible_peers_empty=False),
                lambda k:k['initial_protocol_state'].update(pending_messages=[{'pending':True}])):
            kw=fixture();mutate(kw);out=m.evaluate(**kw)
            self.assertEqual(out['status'],'UNKNOWN_OUTSIDE_FINITE_FULL_WORLD_DOMAIN')
            self.assertEqual(out['offered_requests'],6)
            self.assertEqual([r['request_id'] for r in out['requests']],out['offered_request_ids'])
            self.assertTrue(all(r['reader_E2E_ns_bounds'] is None for r in out['requests']))

    def test_fixed_drain_uses_full_cohort_and_boundary_equality(self):
        kw=fixture(('A',),queued=False)
        out=m.evaluate(**kw);deadline=out['requests'][0]['RESULT_reader_ns_bounds'][1]
        kw['drain_end_ns']=deadline
        equality=m.evaluate(**kw)
        self.assertEqual(equality['fixed_drain_incomplete_fraction_bounds'],[F(0),F(0)])
        kw['drain_end_ns']=deadline-F(1,2);censored=m.evaluate(**kw)
        self.assertIsNone(censored['complete_cohort_P95_ns_bounds'])
        self.assertEqual(censored['offered_requests'],1)
        self.assertEqual(censored['fixed_drain_incomplete_fraction_bounds'][1],1)
        self.assertIsNone(censored['full_U_bounds'])

    def test_enqueue_order_differs_from_public_arrival_plan_ordinal(self):
        kw=fixture(('A','B'))
        for owner in ('A','B'):
            k=(owner,'text','input',0);d=kw['transport_draws'][k]
            kw['transport_draws'][k]=C(5,d.sender_turn_extra_ns,d.submit_to_write_record_ns,
                                      d.observed_submit_to_START_ns,d.previous_READY_minus_submit_ns)
        out=m.evaluate(**kw)
        for u in product((F(0),F(4)),repeat=2):
            self.assert_contains(out,concrete(kw,[F(7),u[0],F(13),u[1]]),kw['requests'])


if __name__=='__main__':unittest.main()
