"""Reproduce frozen v27 dropping a future hazard review after expiry in empty frontier."""
from pathlib import Path
import tempfile
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_learned_context_oracle_v14 import Handoff
from tests.test_projected_handler_context_v42_independent_review import candidate
from tools.predictor_validation_20261027 import projected_handler_bridge_v44 as b
from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode
from tools.predictor_validation_20261027.learned_lifecycle_scheduler_v27 import LearnedLifecycleOracle,LearnedLifecycleScheduler
from tools.predictor_validation_20261027.owner_lifecycle_engine_v21 import LifecycleSchedulingView
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle,receipt
from tools.predictor_validation_20261027.projected_handler_context_v42 import OWNERS,EDGES_NS


class FrozenExpiredEmptyReviewDiagnostic(unittest.TestCase):
    def test_expired_old_review_hides_future_SAVE_during_empty_frontier(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        source=Path(temporary.name)/'cpu-fixture';source.write_text('CPU numeric candidate only')
        hazard=object.__new__(b.ObservableSaveHazard);hazard.candidate=candidate()
        controller=CumulativeLifecycleHazard(hazard,age_edges_ns=EDGES_NS,sources=[receipt(source)],
            candidate_identity='a'*64,allow_unvalidated_approximation=True)
        mark=object.__new__(b.ObservableReferenceMark);mark.candidate=candidate();mark.states=mark.clock=None
        oracle=LearnedLifecycleOracle(initial_prior=None,save_controller=controller,next_owner=mark,
            handoff=Handoff(),public_setup={},public_workload=[],source_identity='a'*64,
            initial_quantile=.5,initial_readiness_rule='lower_bound_hypothesis',
            handoff_readiness_rule='lower_bound_reference_availability_hypothesis',
            allow_initial_pooling=True,allow_observer_transfer=True,allow_flagged_statistical_extrapolation=True)
        scheduler=LearnedLifecycleScheduler(OWNERS,oracle=oracle)
        scheduler._event(0,'RESTORE_START','text#0','CPU_FIXTURE_MODELLED_STATE')
        scheduler._event(0,'SERVICE_READY','text#0','CPU_FIXTURE_MODELLED_STATE')
        scheduler.review_deadline_ns=1  # Prior post-service review has now expired.
        states={o:OwnerLifecycle(o=='text#1',None if o=='text#1' else False,False) for o in OWNERS}
        empty=LifecycleSchedulingView(1,(),tuple(states.items()),(),())
        decision=scheduler.decide(empty)
        future=oracle.hazard['next_review_ns']
        self.assertGreater(future,empty.clock_ns)
        self.assertEqual(oracle.hazard['consumed_hazard'],0.)
        self.assertIsNone(decision.next_transition_ns)  # Confirm exact frozen defect.
        self.assertEqual(scheduler.review_deadline_ns,1)
        active=(ActiveGpuNode(('text#1','node'),1e9,'decode','r','text','pool'),)
        later=LifecycleSchedulingView(future+1000,active,tuple(states.items()),(),())
        with self.assertRaisesRegex(UnknownEvidence,'ENGINE_SKIPPED_PREDICTED_SAVE_BOUNDARY_NO_REWIND'):
            scheduler.decide(later)


if __name__=='__main__':unittest.main()
