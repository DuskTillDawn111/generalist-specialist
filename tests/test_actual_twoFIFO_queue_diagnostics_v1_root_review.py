"""Independent area identities and translation tests for generated queues."""
import copy
from dataclasses import replace
from fractions import Fraction as F
import unittest

from tests import test_actual_twoFIFO_queue_diagnostics_v1 as fixtures
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v4 as engine
from tools.predictor_validation_20261027.actual_twoFIFO_queue_diagnostics_v1 import diagnose
from tools.predictor_validation_20261027.worker_stdin_fifo_contract_v1 import TransportDraw


def diagnostic(args):
    world=engine.simulate(**args)
    out=fixtures.decode(diagnose(world,args['requests'],window_start_ns=args['window_start_ns'],
        window_end_ns=args['window_end_ns'],drain_end_ns=args['drain_end_ns']))
    return fixtures.decode(world),out


class QueueAreaIndependentReview(unittest.TestCase):
    def assert_area_identities(self,args):
        world,out=diagnostic(args);cutoff=world['cutoff_ns']
        for event,field in [('starts','arrived_waiting_START_request_ids'),
                ('RESULT_receipts','arrived_RESULT_unfinished_request_ids')]:
            # Sample-path conservation: each request contributes its own
            # arrived waiting interval exactly once to the queue-depth area.
            expected=sum(max(F(0),min(cutoff,world[event].get(r['request_id'],cutoff))-r['arrival_ns'])
                for r in args['requests'])
            actual=sum(s['duration_ns']*len(s['state'][field]) for s in out['event_slices'])
            self.assertEqual(actual,expected,(event,cutoff))
        return world,out

    def test_waiting_and_delivery_area_conservation_under_sender_reordering_and_censoring(self):
        for cutoff in (0,F(3,2),9,17,35,70):
            args=fixtures.burst_inputs(cutoff=cutoff)
            args['transport_draws']={('owner','text','input',i):TransportDraw(
                (5,0,8,0)[i],1,7,1,2) for i in range(4)}
            with self.subTest(cutoff=cutoff):self.assert_area_identities(args)

    def test_two_parallel_owner_lanes_keep_complete_independent_queue_areas(self):
        args=fixtures.burst_inputs(cutoff=27);second=copy.deepcopy(args)
        oldkey=('owner','text','input');newkey=('other','text','input')
        otherplans=tuple(replace(p,ready_producer_id='other',macros=tuple(
            replace(m,marks=tuple(replace(mark,producer_id='other') for mark in m.marks)) for m in p.macros))
            for p in second['plans'][oldkey])
        args['requests'] += [dict(r,owner='other',request_id='other:'+r['request_id']) for r in second['requests']]
        args['plans'][newkey]=otherplans
        args['protocol_slacks'].update({p:second['protocol_slacks'][q] for p,q in zip(otherplans,second['plans'][oldkey])})
        args['transport_draws'].update({('other',*k[1:]):v for k,v in second['transport_draws'].items()})
        args['initial_sender_available_ns']['other']=0;args['initial_worker_read_release_ns']['other']=0
        world,out=self.assert_area_identities(args)
        self.assertEqual(world['maximum_parallel_handlers'],2)
        self.assertEqual(out['public_offered_requests'],8)

    def test_nonzero_origin_preserves_relative_queue_and_reader_metrics(self):
        args=fixtures.burst_inputs();world,base=self.assert_area_identities(args)
        origin=10**15;shifted=copy.deepcopy(args)
        for r in shifted['requests']:r['arrival_ns']+=origin
        for k in ('window_start_ns','window_end_ns','drain_end_ns','cutoff_ns'):shifted[k]+=origin
        for mapping in ('initial_sender_available_ns','initial_worker_read_release_ns'):
            shifted[mapping]={k:v+origin for k,v in shifted[mapping].items()}
        shifted['initial_protocol_state'].update(collector_free=origin,last_collector_arrival=origin)
        _,changed=self.assert_area_identities(shifted)
        for field in ('reader_E2E_P95_ns','max_arrived_waiting_START','max_waiting_arrival_age_ns',
                'FIFO_convoy_pending_request_ns'):
            self.assertEqual(base[field],changed[field])
        for a,b in zip(base['queue_recovery_episodes'],changed['queue_recovery_episodes']):
            self.assertEqual(a['recovery_ns'],b['recovery_ns'])
            self.assertEqual(a['backlog_start_ns']+origin,b['backlog_start_ns'])


if __name__=='__main__':unittest.main()
