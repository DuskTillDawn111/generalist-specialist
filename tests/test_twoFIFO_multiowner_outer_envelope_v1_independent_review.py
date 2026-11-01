"""Independent finite-rate envelope checks with changing and stale phases."""
import copy
from fractions import Fraction as F
from itertools import product
import unittest

from tests.test_twoFIFO_multiowner_outer_envelope_v1 import concrete
from tools.predictor_validation_20261027 import twoFIFO_multiowner_outer_envelope_v1 as m
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import Mark,Macro,RequestPlan
from tools.predictor_validation_20261027.twoFIFO_paired_transport_v3 import PairedTransportConstraints as C


def changing_phases():
    requests=[];plans={};timing={};transport={};rates={}
    for index,owner in enumerate(('A','B')):
        producer=owner+':stdout';key=(owner,'text','input');values=[]
        for ordinal in (0,1):
            suffix=str(ordinal)
            plan=RequestPlan((Macro('prefill'+suffix,'prefill',5+ordinal,(
                Mark('BEGIN'+suffix,0,0,'phase',producer,phase='prefill'),
                Mark('DIAG'+suffix,2,0,'diagnostic',producer))),
                Macro('decode'+suffix,'decode',4+2*ordinal,(
                Mark('DECODE'+suffix,0,0,'phase',producer,phase='decode'),
                Mark('RESULT'+suffix,3+ordinal,0,'result',producer),
                Mark('CLEAR'+suffix,4+2*ordinal,0,'clear',producer,clear_scope='request')))),0,producer)
            values.append(plan)
            timing[plan]={mark.identity:dict(reader_slack_ns=F(index+ordinal,2),
                collector_dequeue_slack_ns=F(1),collector_processing_slack_ns=F(3+index))
                for macro in plan.macros for mark in macro.marks}
            timing[plan]['worker_READY']=dict(reader_slack_ns=F(5),collector_dequeue_slack_ns=F(1),collector_processing_slack_ns=F(4))
            requests.append(dict(request_id=owner+suffix,owner=owner,task='text',pool_id='input',arrival_ns=ordinal))
            transport[(*key,ordinal)]=C(5 if ordinal==0 else 0,index,6,8,2) if ordinal==0 else C(0,0,3,6,-2)
        plans[key]=tuple(values)
        # Visible-peer states need not slow the owner down; only complete
        # positive finite support makes the duration relaxation sound.
        for phase in ('prefill','decode'):
            rates[(owner,phase,0,0)]=F(1,3) if phase=='prefill' else F(2)
            rates[(owner,phase,1,0)]=F(3)
            rates[(owner,phase,1,1)]=F(1,2)
    requests.reverse()
    return dict(requests=requests,plans=plans,rates=rates,protocol_slacks=timing,transport_draws=transport,
        domain_certificate=copy.deepcopy(m.CERTIFICATE),initial_sender_available_ns={'A':0,'B':0},
        initial_worker_read_release_ns={'A':0,'B':0},initial_protocol_state=dict(reader_free={'A:stdout':7,'B:stdout':11},
            collector_free=25,last_collector_arrival=7,pending_messages=[]),
        window_start_ns=0,window_end_ns=10,drain_end_ns=1_000_000)


class MultiownerEnvelopeIndependentReview(unittest.TestCase):
    def test_changing_macro_phases_arbitrary_positive_rate_order_and_queued_stdout_remain_enclosed(self):
        kw=changing_phases();out=m.evaluate(**kw);count=0
        self.assertEqual(out['status'],'CONSERVATIVE_FINITE_RATE_FULL_COHORT_OUTER_ENVELOPE')
        self.assertEqual(out['offered_request_ids'],['B1','B0','A1','A0'])
        seen_states=set()
        for coordinates in product((F(0),F(3),F(6)),(F(0),F(3,2),F(3)),
                (F(0),F(3),F(6)),(F(0),F(3,2),F(3))):
            world=concrete(kw,coordinates);arrivals={r['request_id']:r['arrival_ns'] for r in kw['requests']}
            for row in out['requests']:
                rid=row['request_id']
                for field,world_key in (('START_ns_bounds','starts'),('READY_ns_bounds','READY_occurrences'),
                        ('RESULT_reader_ns_bounds','RESULT_receipts')):
                    self.assertLessEqual(row[field][0],world[world_key][rid]);self.assertLessEqual(world[world_key][rid],row[field][1])
            tail=m.p95([value-arrivals[rid] for rid,value in world['RESULT_receipts'].items()])
            self.assertLessEqual(out['untruncated_model_complete_cohort_P95_ns_bounds'][0],tail)
            self.assertLessEqual(tail,out['untruncated_model_complete_cohort_P95_ns_bounds'][1])
            for row in world['response_exposure']:seen_states.add(tuple(row['evidence_key']))
            count+=1
        self.assertEqual(count,81)
        self.assertTrue(any(key[2]>0 for key in seen_states))
        self.assertTrue(any(key[3]==0 and key[2]>0 for key in seen_states))
        self.assertTrue(any(key[3]>0 for key in seen_states))
        self.assertIsNone(out['collector_processing_bounds'])
        self.assertFalse(out['envelope_is_sharp']);self.assertFalse(out['extrema_simultaneously_attainable_claimed'])

    def test_every_potential_rate_key_is_required_even_when_missing_state_seems_unreachable(self):
        kw=changing_phases();keys=list(kw['rates'])
        for key in keys:
            changed=copy.deepcopy(kw);del changed['rates'][key]
            out=m.evaluate(**changed)
            self.assertEqual(out['status'],'UNKNOWN_OUTSIDE_FINITE_FULL_WORLD_DOMAIN')
            self.assertEqual(out['offered_requests'],4);self.assertEqual(len(out['requests']),4)
            self.assertTrue(all(r['reader_E2E_ns_bounds'] is None for r in out['requests']))

    def test_producer_lifetime_and_external_reader_or_pending_state_fail_closed_for_all_ids(self):
        for mutation in ('shared_producer','changed_lifetime','external_reader','pending'):
            kw=changing_phases()
            if mutation=='external_reader':kw['initial_protocol_state']['reader_free']['outsider:stdout']=0
            elif mutation=='pending':kw['initial_protocol_state']['pending_messages']=[{'unknown':True}]
            else:
                owner='B' if mutation=='shared_producer' else 'A';key=(owner,'text','input')
                old=kw['plans'][key][1];replacement='A:stdout' if mutation=='shared_producer' else 'changed:stdout'
                macros=tuple(Macro(ma.identity,ma.phase,ma.reference_work_ns,tuple(
                    Mark(mark.identity,mark.work_ns,mark.processing_delay_ns,mark.action,replacement,phase=mark.phase,
                        clear_scope=mark.clear_scope) for mark in ma.marks)) for ma in old.macros)
                new=RequestPlan(macros,0,replacement);kw['plans'][key]=(kw['plans'][key][0],new)
                kw['protocol_slacks'][new]=kw['protocol_slacks'][old]
            out=m.evaluate(**kw)
            self.assertEqual(out['status'],'UNKNOWN_OUTSIDE_FINITE_FULL_WORLD_DOMAIN')
            self.assertEqual([r['request_id'] for r in out['requests']],out['offered_request_ids'])

    def test_censoring_uses_entire_cohort_and_NEVER_claims_endpoint_sharpness(self):
        kw=changing_phases();full=m.evaluate(**kw)
        drain=max(kw['window_end_ns'],min(row['RESULT_reader_ns_bounds'][0] for row in full['requests']))
        out=m.evaluate(**dict(kw,drain_end_ns=drain))
        self.assertEqual(len(out['requests']),4);self.assertIsNone(out['complete_cohort_P95_ns_bounds'])
        self.assertIsNotNone(out['untruncated_model_complete_cohort_P95_ns_bounds'])
        for row in out['requests']:
            a,b=row['RESULT_reader_ns_bounds']
            self.assertEqual(row['fixed_drain_incomplete_indicator_bounds'],[int(a>drain),int(b>drain)])
        for flag in ('latent_probability_assigned','latent_point_selected','physical_rate_bounds_certified',
                'main_queue_provider_eligible','final_gain_eligible','GPU_started','validation_labels_read'):
            self.assertIs(out[flag],False)
        self.assertIsNone(out['resource_bounds']);self.assertIsNone(out['full_U_bounds'])


if __name__=='__main__':unittest.main()
