"""Independent real v42 numeric candidate/controller/original guard integration."""
from pathlib import Path
import tempfile
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_learned_context_oracle_v14 import Handoff
from tests.test_projected_handler_context_v42_independent_review import candidate
from tools.predictor_validation_20261027 import projected_handler_bridge_v44 as b
from tools.predictor_validation_20261027 import projected_handler_forecast_v44 as f
from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode
from tools.predictor_validation_20261027.held_context_scheduler_v14 import InitialContextDecision
from tools.predictor_validation_20261027.implementation_source_closure_v27 import implementation_sources
from tools.predictor_validation_20261027.learned_lifecycle_scheduler_v27 import LearnedLifecycleOracle,LearnedLifecycleScheduler
from tools.predictor_validation_20261027.owner_lifecycle_engine_v21 import LifecycleSchedulingView
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle,receipt
from tools.predictor_validation_20261027.projected_handler_context_v42 import OWNERS,EDGES_NS


class IndependentProjectedBridgeTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.source=Path(temporary.name)/'fixture';self.source.write_text('CPU numeric fixture')
        self.states={o:OwnerLifecycle(o=='text#0',None if o=='text#0' else False,False) for o in OWNERS}

    def controller(self):
        hazard=object.__new__(b.ObservableSaveHazard);hazard.candidate=candidate()
        return CumulativeLifecycleHazard(hazard,age_edges_ns=EDGES_NS,sources=[receipt(self.source)],
            candidate_identity='a'*64,allow_unvalidated_approximation=True)

    def test_real_v42_controller_retains_budget_and_age_across_handler_change(self):
        controller=self.controller();first=controller.start_restore('text#0',0,[],'restore',self.states)
        second=controller.update(first['state'],1000,[],self.states)
        changed=dict(self.states,**{'text#1':OwnerLifecycle(True,None,False)})
        third=controller.update(second['state'],1500,[],changed)
        self.assertEqual(third['hazard_budget'],first['hazard_budget'])
        self.assertEqual(third['origin_restore_ns'],0)
        self.assertFalse(third['budget_redrawn'])
        self.assertAlmostEqual(third['consumed_hazard'],2.*1500/1e6)
        self.assertIn('UNCALIBRATED_FIXED_MEDIAN_WORKER_HANDLER_PROJECTION',third['uncertainty_reasons'])
        self.assertIn('UNSEEN_HOLDER_WORKER_HANDLER_TOKENS',third['uncertainty_reasons'])

    def test_real_v42_bridge_single_held_domain_reaches_original_guard(self):
        bridge=object.__new__(b.ObservableReferenceMark)
        bridge.candidate=candidate();bridge.states=bridge.clock=None
        oracle=LearnedLifecycleOracle(initial_prior=None,save_controller=self.controller(),next_owner=bridge,
            handoff=Handoff(),public_setup={},public_workload=[],source_identity='a'*64,
            initial_quantile=.5,initial_readiness_rule='lower_bound_hypothesis',
            handoff_readiness_rule='lower_bound_reference_availability_hypothesis',
            allow_initial_pooling=True,allow_observer_transfer=True,allow_flagged_statistical_extrapolation=True)
        oracle.initialize=lambda query:InitialContextDecision('text#0',0,0,'a'*64)
        scheduler=LearnedLifecycleScheduler(OWNERS,oracle=oracle)
        def view(clock):
            active=(ActiveGpuNode(('text#0','node'),1e9,'decode','req','text','pool'),)
            return LifecycleSchedulingView(clock,active,tuple(self.states.items()),(),())
        scheduler.decide(view(0))
        # This short-support candidate is intentionally queried beyond support;
        # statistical extrapolation stays visibly flagged in the original oracle.
        crossing=oracle.hazard['absolute_save_candidate_ns']
        with self.assertRaisesRegex(UnknownEvidence,'SAME_CONTEXT_REACQUISITION'):
            scheduler.decide(view(crossing))
        self.assertEqual(oracle.decisions,[])

    def test_v44_static_closure_binds_v42_original_guard_and_clock_controller(self):
        names={Path(r['path']).name for r in implementation_sources(f.__file__)}
        self.assertTrue({'projected_handler_context_v42.py','projected_handler_bridge_v44.py',
            'cumulative_lifecycle_hazard_v24.py','learned_lifecycle_scheduler_v27.py',
            'owner_lifecycle_engine_v21.py'}<=names)


if __name__=='__main__':unittest.main()
