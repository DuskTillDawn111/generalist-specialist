"""Independent continuation cases with receipt/READY reversed and drain work."""
import json
import unittest

from tests import test_lifecycle_engine_checkpoint_v59 as fixture
from tools.predictor_validation_20261027 import lifecycle_scheduler_checkpoint_v1 as scp


class EarlyReady:
    def delays(self,row,aliases,scenario):
        return dict(known=True,parent_receipt_ms=.2,worker_next_ready_ms=.005)


class IndependentContinuationTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture.EngineCheckpointTests();self.f.setUp();self.addCleanup(self.f.doCleanups)

    def run_case(self,cutoff,*,early=False,deadline=None):
        f=self.f
        if deadline is not None:f.deadline=deadline
        def engine():
            value=f.engine()
            if early:value.lifecycle_provider=EarlyReady()
            return value
        whole=f.run_engine(engine())
        paused=f.run_engine(engine(),pause_at_ns=cutoff)
        saved=json.loads(json.dumps(paused['checkpoint']))
        restored=f.run_engine(engine(),checkpoint=saved)
        self.assertEqual(whole,restored)
        self.assertFalse(saved['native_observation_assimilation_supported'])
        self.assertFalse(saved['physical_progress_observed'])
        return whole,scp.decode(saved['state']),scp.decode(saved['six_term_prefix'])

    def test_next_handler_can_start_before_previous_parent_receipt(self):
        whole,state,prefix=self.run_case(50000,early=True)
        self.assertTrue(state['receipt_pending'])
        self.assertIn('a1',state['starts'])
        self.assertNotIn('a0',state['finishes'])
        self.assertLess(state['starts']['a1'],state['receipt_pending']['a0']['finish_ns'])
        self.assertFalse(prefix['received'])
        self.assertEqual(len(whole['finishes']),4)

    def test_non_event_pause_keeps_unoffered_future_fifo(self):
        _,state,_=self.run_case(999)
        self.assertEqual({row[2]['request_id'] for row in state['arrivals']},{'a1','b0','b1'})
        self.assertEqual(set(state['cohort']),{'a0','a1','b0','b1'})
        self.assertEqual(state['clock'],999)
        self.assertEqual(state['advance']['origin_ns'],0)

    def test_energy_cutoff_and_drain_work_survive_non_event_pause(self):
        whole,state,prefix=self.run_case(9999,deadline=21000)
        self.assertLess(prefix['cutoff_ns'],21000)
        self.assertTrue(any(row['finish_ns']>21000 for row in whole['finishes'].values()))
        self.assertTrue(all(x['until_ns']>x['time_ns'] for x in whole['resource_transitions']))
        expected=sum(x['board_watts']*max(0,min(21000,x['until_ns'])-x['time_ns'])/1e9
            for x in whole['resource_transitions'] if x['time_ns']<21000)
        self.assertAlmostEqual(whole['board_energy_joules'],expected,places=14)


if __name__=='__main__':unittest.main()
