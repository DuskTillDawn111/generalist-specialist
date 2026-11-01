import copy
from fractions import Fraction as F
import unittest

from tests import test_actual_worker_twoFIFO_reference_v4 as fixtures

decode = fixtures.decode
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v4 as reference
from tools.predictor_validation_20261027 import actual_twoFIFO_queue_diagnostics_v1 as diagnostics
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import Mark, Macro, RequestPlan
from tools.predictor_validation_20261027.worker_stdin_fifo_contract_v1 import TransportDraw


def burst_inputs(arrivals=(0,1,2,10), cutoff=None, drain=70):
    requests=[dict(request_id='b'+str(i),owner='owner',task='text',pool_id='input',arrival_ns=a)
              for i,a in enumerate(arrivals)]
    plans=tuple(RequestPlan((Macro('m'+str(i),'work',10,
        (Mark('result'+str(i),8,0,'result','owner'),)),),0,'owner') for i in range(len(requests)))
    return dict(requests=requests,plans={('owner','text','input'):plans},rates={('work',0,0):1},
        window_start_ns=0,window_end_ns=max(arrivals)+1,drain_end_ns=drain,cutoff_ns=drain if cutoff is None else cutoff,
        protocol_slacks={plan:{'result'+str(i):dict(reader_slack_ns=5,collector_dequeue_slack_ns=0,
            collector_processing_slack_ns=100),'worker_READY':dict(reader_slack_ns=0,collector_dequeue_slack_ns=0,
            collector_processing_slack_ns=0)} for i,plan in enumerate(plans)},
        transport_draws={('owner','text','input',i):TransportDraw(0,0,0,0,0) for i in range(len(requests))},
        initial_sender_available_ns={'owner':0},initial_worker_read_release_ns={'owner':0},
        initial_protocol_state=dict(reader_free={},collector_free=0,last_collector_arrival=0,pending_messages=()))


def call(kwargs, known=True, **options):
    world=reference.simulate(**kwargs)
    costs={('owner','text','input',i):10 for i in range(len(kwargs['requests']))} if known else None
    return decode(diagnostics.diagnose(world,kwargs['requests'],window_start_ns=kwargs['window_start_ns'],
        window_end_ns=kwargs['window_end_ns'],drain_end_ns=kwargs['drain_end_ns'],
        scales_ns=(1,2,10,11),work_by_public_key_ordinal=costs,**options))


class ActualTwoFIFOQueueDiagnosticsTests(unittest.TestCase):
    def test_complete_arrival_envelopes_include_future_and_exact_half_open_boundaries(self):
        out=call(burst_inputs(cutoff=0))
        self.assertEqual([r['max_arrival_count'] for r in out['full_public_arrival_envelopes']],[1,2,3,4])
        self.assertEqual([r['max_offered_response_reference_work_ns'] for r in out['full_public_arrival_envelopes']],[10,20,30,40])
        self.assertEqual(out['cutoff_state']['lifecycle_count']['scheduled_not_arrived'],3)
        self.assertEqual(out['cutoff_state']['complete_cohort_remaining_response_reference_work_ns'],40)
        self.assertIsNone(out['native_current_visible_risk_features'])
        self.assertIsNone(out['original_drain']['RESULT_unfinished_request_ids'])
        self.assertIsNone(out['reader_E2E_P95_ns'])

    def test_burst_conservation_recovery_convoy_and_no_resource_claim(self):
        out=call(burst_inputs())
        for row in [s['state'] for s in out['event_slices']]+[out['cutoff_state']]:
            self.assertEqual(sum(row['lifecycle_count'].values()),4)
            self.assertEqual(sum(row['RESULT_delivery_count'].values()),4)
            self.assertIsNone(row['sm_demand']);self.assertIsNone(row['power_watts'])
        self.assertEqual(out['max_arrived_waiting_START'],2)
        self.assertEqual(out['reader_E2E_P95_ns'],F(327,10))
        self.assertEqual(out['max_worker_FIFO_head_submit_wait_ns'],20)
        self.assertEqual(out['max_waiting_arrival_age_ns'],20)
        self.assertEqual(out['FIFO_convoy_pending_request_ns'],47)
        self.assertEqual(out['queue_recovery_episodes'][0]['clear_ns'],30)
        self.assertEqual(out['queue_recovery_episodes'][0]['recovery_ns'],29)
        self.assertEqual(out['original_drain']['RESULT_unfinished_request_ids'],[])
        self.assertEqual(out['cutoff_state']['complete_cohort_remaining_response_reference_work_ns'],0)
        self.assertIsNone(out['resource_feasible_HOL_blocked_ns'])
        self.assertFalse(out['stability_rule_changed']);self.assertFalse(out['main_queue_provider_eligible'])

    def test_worker_RELEASE_and_RESULT_in_transit_are_distinct(self):
        out=call(burst_inputs(cutoff=11))
        row=out['cutoff_state']
        self.assertIn('b0',row['worker_lifecycle']['worker_released'])
        self.assertIn('b0',row['RESULT_delivery']['RESULT_in_transit'])
        self.assertIn('b1',row['worker_lifecycle']['running_START_to_READY'])
        self.assertEqual(out['queue_recovery_episodes'][0]['clear_ns'],None)
        self.assertEqual(out['original_drain']['status'],'UNKNOWN_SIMULATION_STOPPED_BEFORE_ORIGINAL_DRAIN')

    def test_reader_received_can_precede_worker_release(self):
        args=burst_inputs(cutoff=9)
        for plan, marks in args['protocol_slacks'].items():
            for value in marks.values():value['reader_slack_ns']=0
        row=call(args)['cutoff_state']
        self.assertIn('b0',row['RESULT_delivery']['RESULT_reader_received'])
        self.assertIn('b0',row['worker_lifecycle']['running_START_to_READY'])

    def test_work_UNKNOWN_not_reconstructed_even_for_completed_requests(self):
        out=call(burst_inputs(),known=False)
        self.assertEqual(len(out['unknown_work_public_keys']),4)
        self.assertTrue(all(e['max_offered_response_reference_work_ns'] is None for e in out['full_public_arrival_envelopes']))
        self.assertEqual(out['full_public_arrival_envelopes'][2]['max_arrival_count'],3)
        self.assertIsNone(out['cutoff_state']['complete_cohort_remaining_response_reference_work_ns'])
        self.assertIsNone(out['cutoff_state']['arrived_remaining_response_reference_work_ns'])

    def test_foreign_or_inconsistent_work_and_public_cohort_rejected(self):
        args=burst_inputs();world=reference.simulate(**args)
        options=dict(window_start_ns=0,window_end_ns=11,drain_end_ns=70)
        for costs in ({('owner','text','input',99):10},{('owner','text','input',0):9},
                      {('owner','text','input',0):11}):
            with self.assertRaises(ValueError):
                diagnostics.diagnose(world,args['requests'],work_by_public_key_ordinal=costs,**options)
        with self.assertRaises(ValueError):diagnostics.diagnose(world,args['requests'][:-1],**options)

    def test_sender_reordering_and_equal_time_FIFO_use_actual_sequence(self):
        args=fixtures.ActualWorkerTwoFIFOTests().inputs(enqueue=[10,0],worker=0)
        args['transport_draws']={('owner','text','input',0):TransportDraw(10,0,0,0,0),
            ('owner','text','input',1):TransportDraw(0,9,0,0,0)}
        args['cutoff_ns']=11
        out=call(args);head=out['cutoff_state']['worker_FIFO_heads'][0]
        self.assertEqual(head['head_request_id'],'r0')
        self.assertEqual(head['running_predecessor_request_id'],'r1')
        self.assertEqual(out['cutoff_state']['worker_lifecycle']['arrived_not_submitted'],[])

    def test_unsubmitted_sender_backlog_and_original_drain_censoring(self):
        args=fixtures.ActualWorkerTwoFIFOTests().inputs(write=20,cutoff=5)
        args.update(drain_end_ns=5)
        out=call(args)
        row=out['cutoff_state'];self.assertEqual(row['worker_lifecycle']['arrived_not_submitted'],['r1'])
        self.assertEqual(row['parent_sender_FIFO_heads'][0]['head_enqueue_wait_ns'],4)
        self.assertEqual(out['original_drain']['RESULT_unfinished_request_ids'],['r0','r1'])
        self.assertEqual(out['queue_recovery_episodes'][0]['right_censored'],True)

    def test_exact_fractional_progress_and_input_immutability(self):
        args=burst_inputs(cutoff=F(3,2));args['rates']={('work',0,0):F(3,2)}
        world=reference.simulate(**args);before=copy.deepcopy(world)
        out=decode(diagnostics.diagnose(world,args['requests'],window_start_ns=0,window_end_ns=11,
            drain_end_ns=70,work_by_public_key_ordinal={('owner','text','input',i):10 for i in range(4)}))
        self.assertEqual(out['cutoff_state']['complete_cohort_remaining_response_reference_work_ns'],F(151,4))
        self.assertEqual(world,before)


if __name__=='__main__':unittest.main()
