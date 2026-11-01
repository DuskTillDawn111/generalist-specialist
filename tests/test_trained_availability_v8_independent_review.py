"""Independent scheduler tests against every frozen real trained policy cell."""
import copy
import json
from pathlib import Path
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.dag_pending_frontier_v8 import completion_upper_bounds
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode, SchedulingView
from tools.predictor_validation_20261027.trained_dag_availability_v8 import TrainedDagAvailabilityScheduler


MODEL = Path(__file__).resolve().parents[1] / (
    'artifacts/predictor-dag-availability-complete-fit-v8-20261028-training02/model.json')


def view(clock, state):
    return SchedulingView(clock, tuple(
        ActiveGpuNode((owner, f'node-{i}'), 100., phase, owner, task, pool_id)
        for owner, task, pool_id, phases in state for i, phase in enumerate(phases)))


class IndependentTrainedAvailabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads(MODEL.read_text())['policy']

    def scheduler(self):
        return TrainedDagAvailabilityScheduler(self.policy, allow_unvalidated_training_candidate=True)

    def test_every_real_transition_cell_executes_exact_residual_and_preserves_deadline(self):
        self.assertEqual(len(self.policy['cells']), 64)
        for row in self.policy['cells']:
            with self.subTest(reason=row['reason'], owner=row['current_owner'], state=row['frontier']):
                s = self.scheduler(); s.current_owner = row['current_owner']
                s.mode = 'uncontended' if row['reason'] == 'contention_entry' else 'contended'
                s.grant_end_ns = 10; s.handoff_end_ns = 0
                d = s.decide(view(10, row['frontier']))
                expected_handoff = 10 + row['handoff_ns']
                expected_end = expected_handoff + row['grant_budget_ns']
                self.assertEqual(s.current_owner, row['next_owner'])
                self.assertEqual(s.handoff_end_ns, expected_handoff)
                self.assertEqual(s.grant_end_ns, expected_end)
                self.assertEqual(s.transitions[-1]['reason'], row['reason'])
                before = copy.deepcopy(s.transitions)
                self.assertEqual(s.decide(view(10, row['frontier'])), d)
                self.assertEqual(s.transitions, before)
                self.assertEqual(s.decide(view(expected_handoff, row['frontier'])).next_transition_ns, expected_end)

    def test_every_real_initial_and_later_solo_cell_has_its_own_dispatch_wait(self):
        for initial, table in [(True, 'initial_uncontended_cells'), (False, 'uncontended_cells')]:
            for row in self.policy[table]:
                s = self.scheduler()
                if not initial:
                    s.current_owner = '__previous_context__'; s.handoff_end_ns = 0
                d = s.decide(view(20, row['frontier']))
                expected = 20 + row['dispatch_wait_ns']
                self.assertEqual(s.handoff_end_ns, expected)
                self.assertEqual(s.decide(view(20, row['frontier'])), d)
                self.assertEqual(len(s.transitions), 1)
                active = s.decide(view(expected, row['frontier']))
                self.assertTrue(all(rate == 1. for rate in active.gpu_rates.values()))

    def test_real_entry_cell_rejects_untrained_phase_without_numeric_fallback(self):
        row = next(r for r in self.policy['cells'] if r['reason'] == 'contention_entry')
        s = self.scheduler(); s.current_owner = row['current_owner']; s.mode = 'uncontended'
        bad = copy.deepcopy(row['frontier']); bad[0][3] = ['unobserved_phase']
        with self.assertRaisesRegex(UnknownEvidence, 'UNKNOWN_TRAINED_AVAILABILITY_STATE'):
            s.decide(view(20, bad))
        self.assertEqual(s.transitions, [])

    def test_empty_frontier_and_semantic_change_do_not_reset_live_grant(self):
        row = next(r for r in self.policy['cells'] if r['reason'] == 'contention_entry')
        s = self.scheduler(); s.current_owner = row['current_owner']; s.mode = 'uncontended'
        s.decide(view(20, row['frontier'])); end = s.grant_end_ns
        t = s.handoff_end_ns
        self.assertEqual(s.decide(view(t, [])).gpu_rates, {})
        changed = copy.deepcopy(row['frontier'])
        for state in changed:
            state[3] = ['new_phase_while_same_live_grant']
        self.assertEqual(s.decide(view(t, changed)).next_transition_ns, end)
        self.assertEqual(s.grant_end_ns, end); self.assertEqual(len(s.transitions), 1)

    def test_virtual_event_bound_uses_earliest_started_descendant_never_raw_zero(self):
        nodes = dict(event=dict(kind='event_record', timestamp=0),
                     wait=dict(kind='event_wait', timestamp=0),
                     copy=dict(kind='copy', observed_start_ns=12, observed_end_ns=20),
                     gpu=dict(kind='gpu', observed_start_ns=30, observed_end_ns=40),
                     orphan=dict(kind='event_record', timestamp=0))
        edges = [dict(before=a, after=b) for a, b in [('event', 'wait'), ('wait', 'copy'), ('event', 'gpu')]]
        bounds, _, _ = completion_upper_bounds(nodes, edges)
        self.assertEqual(bounds, dict(event=12, wait=12, copy=20, gpu=40, orphan=None))
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            completion_upper_bounds(nodes, edges + [dict(before='copy', after='event')])


if __name__ == '__main__':
    unittest.main()
