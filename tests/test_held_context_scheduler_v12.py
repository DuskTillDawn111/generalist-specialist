import json
from pathlib import Path
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode,SchedulingView
from tools.predictor_validation_20261027.held_context_scheduler_v12 import (
    ContextAvailabilityDecision,HeldContextScheduler)
from tools.predictor_validation_20261027.trained_dag_availability_v8 import TrainedDagAvailabilityScheduler


def view(t,owners=('text#0',)):
    return SchedulingView(t,tuple(ActiveGpuNode((p,'gpu'),100.,'outside_language_forwards',p,'text','x') for p in owners))


class Oracle:
    def __init__(self,callback):self.callback=callback;self.queries=[]
    def decide(self,q):self.queries.append(q);return self.callback(q)


class HeldContextTests(unittest.TestCase):
    def test_real_source_holder_differs_from_previous_ready_owner(self):
        root=Path(__file__).resolve().parents[1]
        report=json.loads((root/'artifacts/predictor-first-contention-history-v8-20261028-diagnostic01/report.json').read_text())
        capture=report['observed_captures'][1]
        entry=next(r for r in capture['all_contention_entry_observations']
            if r['offset_ns']<1_000_000 and r['observed_held_owner']=='text#1'
            and r['original_fitter_rejection_reason']=='held_context_differs_from_previous_ready_owner')
        t=capture['ready_timeline_origin_trace_ns']+entry['offset_ns']
        restored=t-entry['observed_context_envelope_age_ns']
        scheduler=HeldContextScheduler(('text#0','text#1'),allow_source_replay=True)
        scheduler.observe_context_event(restored,'RESTORE_START','text#1')
        scheduler.observe_context_event(restored,'SERVICE_READY','text#1')
        q=scheduler.snapshot(view(t))
        self.assertEqual(q.held_owner,'text#1')
        self.assertEqual(q.pending_frontier[0][0],'text#0')
        self.assertEqual(q.context_age_ns,entry['observed_context_envelope_age_ns'])
        # The old implementation directly identifies current owner with the
        # sole ready owner, reproducing the real fit/execution mismatch.
        solo=[['text#0','text','x',['outside_language_forwards']]]
        old=TrainedDagAvailabilityScheduler(dict(schema='gsdeploy.trained_DAG_availability_policy.v8',
            reference_work_domain='SOLO_ONLY_REFERENCE_NS',training_only=True,cells=[],
            initial_uncontended_cells=[],uncontended_cells=[dict(frontier=solo,dispatch_wait_ns=0)]),
            allow_unvalidated_training_candidate=True)
        old.current_owner='text#1';old.handoff_end_ns=restored
        old.decide(view(t));self.assertEqual(old.current_owner,'text#0')
        self.assertEqual(scheduler.snapshot(view(t+1,())).held_owner,'text#1')
        self.assertEqual(scheduler.snapshot(view(t+2,('text#0','text#1'))).context_age_ns,q.context_age_ns+2)

    def test_requested_switch_does_not_restore_until_its_actual_event(self):
        oracle=Oracle(lambda q:ContextAvailabilityDecision('switch',20,'text#1',3,4,6)
            if q.clock_ns==1 else ContextAvailabilityDecision('hold',20))
        s=HeldContextScheduler(('text#0','text#1'),oracle=oracle,allow_source_replay=True)
        s.observe_context_event(0,'RESTORE_START','text#0');s.observe_context_event(0,'SERVICE_READY','text#0')
        self.assertEqual(s.decide(view(1,('text#0','text#1'))).next_transition_ns,3)
        self.assertEqual(s.held_owner,'text#0')
        self.assertEqual(s.decide(view(2,('text#1',))).gpu_rates[('text#1','gpu')],0.)
        self.assertEqual(s.held_owner,'text#0');self.assertEqual(s.restore_start_ns,0)
        s.decide(view(3,('text#1',)));self.assertIsNone(s.held_owner)
        self.assertEqual(s.decide(view(4,('text#1',))).gpu_rates[('text#1','gpu')],0.)
        self.assertEqual(s.restore_start_ns,4)
        self.assertEqual(s.decide(view(6,('text#1',))).gpu_rates[('text#1','gpu')],1.)
        self.assertEqual(s.snapshot(view(8,())).context_age_ns,4)
        self.assertEqual(s.review_deadline_ns,20)

    def test_host_frontier_change_cannot_extend_existing_deadline(self):
        oracle=Oracle(lambda q:ContextAvailabilityDecision('hold',20 if len(q.pending_frontier)==1 else 100))
        s=HeldContextScheduler(('text#0','text#1'),oracle=oracle,allow_source_replay=True)
        s.observe_context_event(0,'RESTORE_START','text#0');s.observe_context_event(0,'SERVICE_READY','text#0')
        s.decide(view(1))
        with self.assertRaisesRegex(UnknownEvidence,'DEADLINE_EXTENSION_WITHOUT_RESTORE'):
            s.decide(view(5,('text#0','text#1')))
        self.assertEqual(s.held_owner,'text#0');self.assertEqual(s.restore_start_ns,0)
        self.assertEqual(s.review_deadline_ns,20)

    def test_unknown_availability_does_not_choose_the_only_ready_owner(self):
        s=HeldContextScheduler(('text#0','text#1'))
        with self.assertRaisesRegex(UnknownEvidence,'AVAILABILITY_UNIDENTIFIED'):s.decide(view(0))
        self.assertIsNone(s.held_owner);self.assertEqual(s.held_knowledge,'UNKNOWN')
        with self.assertRaises(ValueError):s.observe_context_event(1,'RESTORE_START','text#0')

    def test_repeated_and_same_clock_host_calls_preserve_restore_age(self):
        oracle=Oracle(lambda q:ContextAvailabilityDecision('hold',20))
        s=HeldContextScheduler(('text#0','text#1'),oracle=oracle,allow_source_replay=True)
        s.observe_context_event(0,'RESTORE_START','text#0');s.observe_context_event(0,'SERVICE_READY','text#0')
        for t in (1,1,2,3,5):
            s.decide(view(t));self.assertEqual(s.restore_start_ns,0);self.assertEqual(s.review_deadline_ns,20)
        self.assertEqual(len(oracle.queries),1)
        self.assertEqual(s.snapshot(view(5)).pending_age_ns_by_node[0][1],4)


if __name__=='__main__':unittest.main()
