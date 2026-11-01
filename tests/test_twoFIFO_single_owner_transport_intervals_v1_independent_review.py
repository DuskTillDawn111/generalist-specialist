"""Independent narrow certificates; frozen engine is only a concrete witness oracle."""
import copy
from fractions import Fraction as F
from itertools import product
import unittest

from tests.test_twoFIFO_single_owner_transport_intervals_v1 import fixture,concrete,slacks
from tools.predictor_validation_20261027 import twoFIFO_single_owner_transport_intervals_v1 as m
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import Mark,Macro,RequestPlan
from tools.predictor_validation_20261027.twoFIFO_paired_transport_v3 import PairedTransportConstraints as C


def reordered_three():
    kw=fixture();owner='owner';producer='owner:stdout';key=(owner,'text','input')
    third=RequestPlan((Macro('prefill2','prefill',5,(
        Mark('BEGIN2',0,0,'phase',producer,phase='prefill'),
        Mark('MID2',2,0,'diagnostic',producer))),
        Macro('decode2','decode',3,(
        Mark('DECODE2',0,0,'phase',producer,phase='decode'),
        Mark('RESULT2',F(5,2),0,'result',producer),
        Mark('CLEAR2',3,0,'clear',producer,clear_scope='request')))),0,producer)
    kw['plans'][key]=(*kw['plans'][key],third)
    kw['protocol_slacks'][third]={mark.identity:slacks(0,1,2) for macro in third.macros for mark in macro.marks}
    kw['protocol_slacks'][third]['worker_READY']=slacks(0,0,0)
    # Input order, arrival/public ordinal order, and sender enqueue order differ.
    kw['requests']=[dict(request_id='r2',owner=owner,task='text',pool_id='input',arrival_ns=2),
        kw['requests'][0],kw['requests'][1]]
    kw['transport_draws']={(*key,0):C(9,0,6,8,6),(*key,1):C(0,1,5,10,-3),(*key,2):C(1,2,4,7,2)}
    kw['rates']={(owner,'prefill',0,0):F(3,2),(owner,'decode',0,0):F(2,3)}
    kw['initial_protocol_state']=dict(reader_free={producer:0},collector_free=12,last_collector_arrival=0,pending_messages=[])
    return kw


def concrete_by_request(kw,us):
    # Existing concrete helper zips draws in insertion order: match by the exact
    # public-key ordinal, regardless of sender or original cohort input order.
    return concrete(kw,us)


class SingleOwnerIntervalsIndependentReview(unittest.TestCase):
    def test_three_request_sender_reordering_has_sharp_joint_bounds_in_every_output(self):
        kw=reordered_three();bounds=m.evaluate(**kw);by={r['request_id']:r for r in bounds['requests']}
        self.assertEqual(list(bounds['lower_witness']['requests']),['r1','r2','r0'])
        self.assertEqual(bounds['offered_request_ids'],['r2','r0','r1'])
        self.assertEqual(bounds['status'],'EXACT_SINGLE_OWNER_PAIRED_TRANSPORT_COHORT_ENVELOPE')
        extrema={k:[None,None] for k in by};p95_extrema=[None,None]
        for us in product((F(0),F(3,2),F(3),F(9,2),F(6)),
                (F(0),F(5,4),F(5,2),F(15,4),F(5)),(F(0),F(1),F(2),F(3),F(4))):
            world=concrete_by_request(kw,us);lat=[]
            for r in kw['requests']:
                rid=r['request_id'];b=by[rid]
                for field,actual in (('START_ns_bounds',world['starts'][rid]),
                        ('READY_ns_bounds',world['READY_occurrences'][rid]),
                        ('RESULT_reader_ns_bounds',world['RESULT_receipts'][rid]),
                        ('RESULT_processed_ns_bounds',world['RESULT_processed'][rid])):
                    self.assertLessEqual(b[field][0],actual);self.assertLessEqual(actual,b[field][1])
                value=world['RESULT_receipts'][rid]-r['arrival_ns'];lat.append(value)
                extrema[rid]=[value if extrema[rid][0] is None else min(extrema[rid][0],value),
                    value if extrema[rid][1] is None else max(extrema[rid][1],value)]
            tail=m.p95(lat)
            p95_extrema=[tail if p95_extrema[0] is None else min(p95_extrema[0],tail),
                tail if p95_extrema[1] is None else max(p95_extrema[1],tail)]
        self.assertEqual(p95_extrema,bounds['complete_cohort_P95_ns_bounds'])
        self.assertEqual(extrema,{rid:r['reader_E2E_ns_bounds'] for rid,r in by.items()})
        for index,name in enumerate(('lower_witness','upper_witness')):
            witness=bounds[name]
            us=[witness['requests']['r'+str(i)]['extremizing_u_ns'] for i in range(3)]
            world=concrete_by_request(kw,us)
            for rid,row in witness['requests'].items():
                self.assertEqual(row['START_ns'],world['starts'][rid])
                self.assertEqual(row['READY_ns'],world['READY_occurrences'][rid])
                self.assertEqual(row['RESULT_reader_ns'],world['RESULT_receipts'][rid])
                self.assertEqual(row['RESULT_processed_ns'],world['RESULT_processed'][rid])
                self.assertEqual(row['RESULT_reader_ns'],by[rid]['RESULT_reader_ns_bounds'][index])

    def test_recursive_endpoint_witness_is_a_feasible_paired_source_point(self):
        kw=reordered_three();out=m.evaluate(**kw)
        for name in ('lower_witness','upper_witness'):
            for rid,row in out[name]['requests'].items():
                draw=kw['transport_draws'][('owner','text','input',int(rid[1:]))]
                u=row['extremizing_u_ns'];x=row['paired_continuation_ns']
                self.assertGreaterEqual(u,0);self.assertLessEqual(u,min(draw.submit_to_write_record_ns,draw.observed_submit_to_START_ns))
                self.assertEqual(x,draw.observed_submit_to_START_ns-max(draw.previous_READY_minus_submit_ns,u))
                self.assertGreaterEqual(x,0)
        self.assertIsNone(out['full_U_bounds']);self.assertFalse(out['latent_probability_assigned'])
        self.assertFalse(out['endpoint_enumeration_used_as_multiowner_envelope'])

    def test_fixed_drain_completion_at_exact_receipt_and_fraction_bounds_are_attained(self):
        kw=fixture();kw['initial_protocol_state']=dict(reader_free={},collector_free=0,last_collector_arrival=0,pending_messages=[])
        full=m.evaluate(**kw)
        for drain in sorted({stamp for row in full['requests'] for stamp in row['RESULT_reader_ns_bounds']}):
            if drain<kw['window_end_ns']:continue
            out=m.evaluate(**dict(kw,drain_end_ns=drain));actual=[]
            for witness in ('lower_witness','upper_witness'):
                incomplete=sum(row['RESULT_reader_ns']>drain for row in out[witness]['requests'].values())
                actual.append(F(incomplete,len(kw['requests'])))
            self.assertEqual(out['fixed_drain_incomplete_fraction_bounds'],actual)
            all_done=all(row['RESULT_reader_ns_bounds'][1]<=drain for row in out['requests'])
            self.assertEqual(out['complete_cohort_P95_ns_bounds'] is not None,all_done)
            self.assertEqual(len(out['requests']),len(kw['requests']))
            for row in out['requests']:
                low,high=row['RESULT_reader_ns_bounds']
                self.assertEqual(row['fixed_drain_incomplete_indicator_bounds'],[int(low>drain),int(high>drain)])

    def test_external_producer_missing_READY_and_changed_lifetime_keep_all_ids_unknown(self):
        for mutate in ('certificate_extra','reader_external','missing_READY','changed_producer','missing_transport'):
            kw=reordered_three()
            if mutate=='certificate_extra':kw['domain_certificate']['assume_whatever']=True
            elif mutate=='reader_external':kw['initial_protocol_state']['reader_free']['other:stdout']=0
            elif mutate=='missing_READY':del kw['protocol_slacks'][kw['plans'][('owner','text','input')][-1]]['worker_READY']
            elif mutate=='missing_transport':del kw['transport_draws'][('owner','text','input',1)]
            else:
                plan=kw['plans'][('owner','text','input')][-1]
                changed=RequestPlan(plan.macros,0,'different:stdout')
                kw['plans'][('owner','text','input')]=(*kw['plans'][('owner','text','input')][:-1],changed)
            out=m.evaluate(**kw)
            self.assertEqual(out['status'],'UNKNOWN_OUTSIDE_CERTIFIED_SINGLE_OWNER_SUBDOMAIN')
            self.assertEqual(out['offered_request_ids'],['r2','r0','r1']);self.assertEqual(len(out['requests']),3)
            self.assertTrue(all(r['reader_E2E_ns_bounds'] is None and r['fixed_drain_incomplete_indicator_bounds']==[0,1]
                for r in out['requests']))


if __name__=='__main__':unittest.main()
