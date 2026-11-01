from fractions import Fraction
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v4 as module
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import Mark,Macro,RequestPlan
from tools.predictor_validation_20261027.worker_stdin_fifo_contract_v1 import TransportDraw


def decode(value):
    if isinstance(value,dict):
        if set(value)=={'numerator','denominator'}:return Fraction(value['numerator'],value['denominator'])
        return {k:decode(v) for k,v in value.items()}
    if isinstance(value,list):return [decode(v) for v in value]
    return value


def slacks(reader=0,processing=0):
    return dict(reader_slack_ns=reader,collector_dequeue_slack_ns=0,
                collector_processing_slack_ns=processing)


class ActualWorkerTwoFIFOTests(unittest.TestCase):
    def inputs(self,write=1,line=0,worker=1,enqueue=None,cutoff=150):
        requests=[dict(request_id='r'+str(i),owner='owner',task='text',pool_id='input',arrival_ns=i)
                  for i in range(2)]
        plans=tuple(RequestPlan((Macro('macro'+str(i),'work',10,
            (Mark('result'+str(i),8,0,'result','owner'),)),),0,'owner') for i in range(2))
        return dict(requests=requests,plans={('owner','text','input'):plans},rates={('work',0,0):1},
            window_start_ns=0,window_end_ns=2,drain_end_ns=150,cutoff_ns=cutoff,
            producer_fifo=('owner',),protocol_slacks={
                plans[0]:{'result0':slacks(10,100),'worker_READY':slacks(2,0)},
                plans[1]:{'result1':slacks(9,0),'worker_READY':slacks(2,0)}},
            transport_draws={('owner','text','input',i):TransportDraw(
                0 if enqueue is None else enqueue[i],0,write if i==0 else 1,line,worker) for i in range(2)},
            initial_sender_available_ns={'owner':0},initial_worker_read_release_ns={'owner':0},
            initial_protocol_state=dict(reader_free={},collector_free=0,last_collector_arrival=0,pending_messages=()))

    def run_world(self,**kwargs):
        return decode(module.simulate(**self.inputs(**kwargs)))

    def test_actual_READY_releases_queued_START_before_parent_notices(self):
        result=self.run_world()
        self.assertEqual(result['starts'],{'r0':1,'r1':12})
        self.assertEqual(result['READY_occurrences'],{'r0':11,'r1':22})
        self.assertEqual(result['RESULT_receipts'],{'r0':19,'r1':30})
        self.assertEqual(result['RESULT_processed']['r0'],119)
        self.assertEqual(result['RESULT_processed']['r1'],119)
        ready=[row for row in result['trace'] if row['kind']=='processed_notification' and row['action']=='ready']
        self.assertEqual([row['available_ns'] for row in ready],[119,119])
        self.assertFalse(result['READY_processing_gates_worker_START'])
        self.assertEqual(result['task_E2E_latency_arrays_ns']['text'],[19,29])

    def test_frozen_v3_parent_processing_gate_changes_next_actual_START(self):
        from dataclasses import replace
        from tools.predictor_validation_20261027.macro_protocol_timing_reference_v3 import simulate as old_policy
        kwargs=self.inputs();new=decode(module.simulate(**kwargs))
        key=('owner','text','input');plans=kwargs['plans'][key]
        oldplans=tuple(replace(plan,dispatch_to_worker_start_ns=1) for plan in plans)
        kwargs['plans']={key:oldplans}
        kwargs['protocol_slacks']={newplan:kwargs['protocol_slacks'][oldplan] for oldplan,newplan in zip(plans,oldplans)}
        for field in ['transport_draws','initial_sender_available_ns','initial_worker_read_release_ns','initial_protocol_state']:
            kwargs.pop(field)
        old=decode(old_policy(**kwargs))
        self.assertEqual(old['starts']['r0'],new['starts']['r0'])
        self.assertEqual([old['starts']['r1'],new['starts']['r1']],[120,12])
        self.assertFalse(old['dispatch_contract_matches_original_sender'])

    def test_parent_write_record_not_worker_gate_and_sender_blocks_independently(self):
        result=self.run_world(write=20)
        self.assertEqual(result['starts'],{'r0':1,'r1':21})
        self.assertEqual(result['parent_dispatches'],{'r0':0,'r1':20})
        self.assertLess(result['starts']['r0'],result['transport_schedule']['r0']['parent_write_record_ns'])

    def test_original_cutoff_preserves_all_pending_queues(self):
        result=self.run_world(write=20,cutoff=5)
        self.assertEqual(result['offered_requests'],2)
        self.assertEqual(result['unfinished_request_ids'],['r0','r1'])
        self.assertEqual(result['parent_sends_not_yet_recorded'],['r0'])
        self.assertEqual(result['pending_parent_sender'][0]['request_id'],'r1')
        self.assertEqual(result['active_response_states']['owner']['virtual_age_ns'],4)
        self.assertEqual(result['original_drain_end_ns'],150)
        queued=self.run_world(cutoff=5)
        self.assertEqual(queued['per_owner_FIFO']['owner'],['r1'])

    def test_explicit_sender_enqueue_order_can_differ_from_arrival(self):
        result=self.run_world(enqueue=[10,0])
        self.assertEqual(list(result['parent_dispatches']),['r1','r0'])
        self.assertEqual(result['public_key_arrival_ordinals'],{'r0':0,'r1':1})
        self.assertEqual(result['starts'],{'r1':2,'r0':13})

    def test_equal_submit_ties_preserve_actual_sender_enqueue_order(self):
        kwargs=self.inputs(enqueue=[10,0],worker=0)
        kwargs['transport_draws']={('owner','text','input',0):TransportDraw(10,0,0,0,0),
                                  ('owner','text','input',1):TransportDraw(0,9,0,0,0)}
        result=decode(module.simulate(**kwargs))
        self.assertEqual(result['parent_dispatches'],{'r1':10,'r0':10})
        self.assertEqual(result['starts'],{'r1':10,'r0':20})

    def test_missing_transport_initial_state_and_legacy_plan_delay_rejected(self):
        for field in ['transport_draws','initial_sender_available_ns','initial_worker_read_release_ns','initial_protocol_state']:
            kwargs=self.inputs();kwargs[field]=None
            with self.assertRaises(UnknownEvidence):module.simulate(**kwargs)
        kwargs=self.inputs();key=('owner','text','input');plan=kwargs['plans'][key][0]
        kwargs['plans'][key]=(RequestPlan(plan.macros,0,'owner',1),kwargs['plans'][key][1])
        with self.assertRaisesRegex(UnknownEvidence,'PLAN_DISPATCH_DELAY_CONFLICTS'):module.simulate(**kwargs)

    def test_actual_exposure_retains_template_identity_and_can_outlive_old_notice(self):
        result=self.run_world()
        second=[row for row in result['response_exposure'] if row['request_id']=='r1']
        self.assertTrue(second)
        self.assertEqual(second[0]['actual_phase'],'work')
        self.assertEqual(second[0]['pool_id'],'input')
        self.assertEqual(second[0]['public_key_arrival_ordinal'],1)
        self.assertEqual(sum(row['right_ns']-row['left_ns'] for row in second),10)
        self.assertEqual(result['maximum_parallel_reserved_or_active_handlers'],1)

    def test_explicit_resource_timeline_adapter_preserves_actual_intervals(self):
        from tools.predictor_validation_20261027.macro_resource_timeline_v1 import timeline
        kwargs=self.inputs();reference=module.simulate(**kwargs)
        view=module.resource_timeline_view(reference)
        self.assertEqual(view['origin_schema'],module.SCHEMA)
        self.assertFalse(view['schema_adapter_confers_native_qualification'])
        resource=decode(timeline(view,kwargs['requests'],resident_owners=('owner',)))
        self.assertEqual(resource['total_wall_ns'],150)
        active=sum(row['duration_ns'] for row in resource['slices'] if row['actual_START_to_READY_handlers'])
        self.assertEqual(active,20)


if __name__=='__main__':unittest.main()
