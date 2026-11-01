"""Regression: lifecycle changes must update the engine's actual deadline."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_cumulative_lifecycle_hazard_v24 import PiecewiseHazard
from tests.test_learned_context_oracle_v14 import Handoff
from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode
from tools.predictor_validation_20261027.held_context_scheduler_v14 import InitialContextDecision
from tools.predictor_validation_20261027.implementation_source_closure_v27 import implementation_sources, strict_references
from tools.predictor_validation_20261027.learned_lifecycle_oracle_v25 import LearnedLifecycleScheduler as OldScheduler
from tools.predictor_validation_20261027.learned_lifecycle_scheduler_v27 import LearnedLifecycleOracle, LearnedLifecycleScheduler
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.owner_lifecycle_engine_v21 import LifecycleSchedulingView
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle, RESIDENTS


class Mark:
    def __init__(self): self.calls = []
    def observe_view(self, view): pass
    def predict_given_save_now(self, **kwargs):
        self.calls.append(kwargs)
        return dict(most_likely_next_owner='text#1', ood_flags=[])


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.source = Path(tmp.name)/'synthetic'; self.source.write_text('training fixture')
        self.low = {o: OwnerLifecycle(False, False, False) for o in RESIDENTS}
        self.low['text#0'] = OwnerLifecycle(True, True, True)
        self.high = dict(self.low, **{'text#1': OwnerLifecycle(True, None, False)})

    def scheduler(self, cls=LearnedLifecycleScheduler):
        controller = CumulativeLifecycleHazard(PiecewiseHazard(),
            age_edges_ns=(0, 100_000, 1_000_000), sources=[receipt(self.source)],
            candidate_identity='a'*64, allow_unvalidated_approximation=True)
        oracle = LearnedLifecycleOracle(initial_prior=None, save_controller=controller,
            next_owner=Mark(), handoff=Handoff(), public_setup={}, public_workload=[],
            source_identity='a'*64, initial_quantile=.5,
            initial_readiness_rule='lower_bound_hypothesis',
            handoff_readiness_rule='lower_bound_reference_availability_hypothesis',
            allow_initial_pooling=True, allow_observer_transfer=True,
            allow_flagged_statistical_extrapolation=True)
        oracle.initialize = lambda query: InitialContextDecision('text#0', 0, 0, 'a'*64)
        return cls(RESIDENTS, oracle=oracle)

    def view(self, clock, high=False, empty=False):
        states = self.high if high else self.low
        if empty:
            states = dict(states, **{'text#0': OwnerLifecycle(True, None, False)})
        active = () if empty else (ActiveGpuNode(('text#0', 'gpu'), 1e9,
                                                 'decode', 'req', 'text', 'known'),)
        return LifecycleSchedulingView(clock, active, tuple(states.items()), (), ())

    def test_same_frontier_update_reproduces_old_skip_and_new_deadline_switch(self):
        old, new = self.scheduler(OldScheduler), self.scheduler()
        first_old = old.decide(self.view(0)).next_transition_ns
        first_new = new.decide(self.view(0)).next_transition_ns
        self.assertEqual(first_old, first_new)
        budget = new.oracle.hazard['hazard_budget']
        old_result = old.decide(self.view(100_000, high=True))
        result = new.decide(self.view(100_000, high=True))
        crossing = new.oracle.hazard['absolute_save_candidate_ns']
        self.assertEqual(old_result.next_transition_ns, first_old)
        self.assertLess(crossing, first_old)
        self.assertEqual(result.next_transition_ns, crossing)
        self.assertEqual(new.review_deadline_ns, crossing)
        self.assertEqual(new.oracle.hazard['hazard_budget'], budget)
        self.assertEqual(new.oracle.next_owner.calls, [])
        switched = new.decide(self.view(crossing, high=True))
        self.assertEqual(new.transitions[-1]['kind'], 'SAVE_END')
        self.assertEqual(new.transitions[-1]['clock_ns'], crossing)
        self.assertEqual(new.oracle.next_owner.calls[0]['context_age_ns'], crossing)
        self.assertEqual(switched.next_transition_ns, crossing+2)
        # The historical scheduler missed this new boundary; it fails closed.
        with self.assertRaisesRegex(UnknownEvidence, 'SKIPPED_PREDICTED_SAVE'):
            old.decide(self.view(first_old, high=True))

    def test_empty_frontier_lifecycle_update_reaches_engine_and_pending_is_immutable(self):
        new = self.scheduler(); new.decide(self.view(0))
        new.decide(self.view(50_000, empty=True))
        result = new.decide(self.view(100_000, high=True, empty=True))
        crossing = new.oracle.hazard['absolute_save_candidate_ns']
        self.assertEqual(result.next_transition_ns, crossing)
        new.decide(self.view(crossing, high=True, empty=True))
        transition = new.pending_switch
        new.decide(self.view(crossing+1, empty=True))
        self.assertEqual(new.pending_switch, transition)
        self.assertEqual(len(new.oracle.next_owner.calls), 1)

    def test_true_missed_save_remains_unknown(self):
        new = self.scheduler(); new.decide(self.view(0))
        crossing = new.oracle.hazard['absolute_save_candidate_ns']
        with self.assertRaisesRegex(UnknownEvidence, 'SKIPPED_PREDICTED_SAVE'):
            new.decide(self.view(crossing+1))
        self.assertEqual(new.oracle.next_owner.calls, [])

    def test_factory_root_closure_includes_previously_missing_transitive_sources(self):
        root = Path(__file__).resolve().parents[1]
        entry = root/'tools/predictor_validation_20261027/learned_lifecycle_forecast_v27.py'
        sources = implementation_sources(entry)
        names = {Path(ref['path']).name for ref in sources}
        self.assertTrue({'learned_availability_forecast_v20.py',
                         'idle_owner_selection_audit_v20.py',
                         'learned_lifecycle_scheduler_v27.py',
                         'owner_lifecycle_hazard_query_v24.py'} <= names)
        self.assertEqual(len(sources), len({ref['path'] for ref in sources}))
        paths = {Path(ref['path']) for ref in sources}
        self.assertIn(root/'gsdeploy/__init__.py', paths)
        self.assertIn(root/'gsdeploy/runtime/__init__.py', paths)
        self.assertTrue(all(receipt(ref['path']) == ref for ref in sources))

    def test_current_implementation_cannot_overwrite_old_training_source_identity(self):
        old = dict(path=str(self.source), sha256='a'*64)
        current = receipt(self.source)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            strict_references([{'hazard': {'sources': [old]}}, current])
        self.assertEqual(strict_references([current, {'other': [current]}]), [current])


if __name__ == '__main__': unittest.main()
