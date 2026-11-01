"""Causal lifecycle updates actually reach the composed SAVE controller."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_cumulative_lifecycle_hazard_v24 import PiecewiseHazard
from tests.test_learned_context_oracle_v14 import Handoff
from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027.learned_lifecycle_oracle_v25 import LearnedLifecycleOracle
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle, RESIDENTS


class Mark:
    def __init__(self,target='text#1'):
        self.target=target

    def predict_given_save_now(self,**kwargs):
        return dict(most_likely_next_owner=self.target,ood_flags=[])


class LifecycleOracleTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        source=Path(tmp.name)/'training';source.write_text('synthetic')
        controller=CumulativeLifecycleHazard(PiecewiseHazard(),age_edges_ns=(0,100_000,1_000_000),
            sources=[receipt(source)],candidate_identity='a'*64,allow_unvalidated_approximation=True)
        self.oracle=LearnedLifecycleOracle(initial_prior=None,save_controller=controller,next_owner=Mark(),
            handoff=Handoff(),public_setup={},public_workload=[],source_identity='a'*64,
            initial_quantile=.5,initial_readiness_rule='lower_bound_hypothesis',
            handoff_readiness_rule='lower_bound_reference_availability_hypothesis',
            allow_initial_pooling=True,allow_observer_transfer=True,allow_flagged_statistical_extrapolation=True)
        self.states={o:OwnerLifecycle(False,False,False) for o in RESIDENTS}
        self.states['text#0']=OwnerLifecycle(True,True,True)

    def observe(self,clock,states=None,frontier=None):
        if states is None:states=self.states
        if frontier is None:frontier=(('text#0','text','known',('decode',)),)
        query=SimpleNamespace(clock_ns=clock,held_owner='text#0',restore_start_ns=0,
            context_age_ns=clock,service_available=True,pending_frontier=frontier,
            existing_review_deadline_ns=None)
        self.oracle.observe_lifecycle(SimpleNamespace(clock_ns=clock,owner_lifecycle=tuple(states.items())))
        self.oracle.observe_query(query)
        return query

    def test_same_frontier_lifecycle_change_integrates_old_rate(self):
        self.observe(0)
        first=self.oracle.hazard
        high=dict(self.states,**{'text#1':OwnerLifecycle(True,None,False)})
        self.observe(100_000,high)
        second=self.oracle.hazard
        self.assertAlmostEqual(second['consumed_hazard'],.1)
        self.assertLess(second['absolute_save_candidate_ns'],first['absolute_save_candidate_ns'])
        query=self.observe(second['absolute_save_candidate_ns'],high)
        switch=self.oracle.decide(query)
        self.assertEqual(switch.kind,'switch')
        self.assertEqual(switch.target_owner,'text#1')
        self.assertEqual(self.oracle.decisions[-1]['context_age_ns'],query.clock_ns)

    def test_empty_risk_is_not_paused_and_retains_age(self):
        self.observe(0)
        self.observe(100_000,frontier=())
        high=dict(self.states,**{'text#1':OwnerLifecycle(True,None,False)})
        self.observe(200_000,high,frontier=())
        self.assertAlmostEqual(self.oracle.hazard['consumed_hazard'],.2)
        self.assertEqual(self.oracle.hazard['origin_restore_ns'],0)
        self.assertIn('EMPTY_RISK_IS_STATISTICAL',self.oracle.uncertainty)

    def test_stale_or_absent_lifecycle_rejected(self):
        query=self.observe(0)
        query.clock_ns=1
        with self.assertRaisesRegex(UnknownEvidence,'SAME_CLOCK'):
            self.oracle.observe_query(query)

    def test_same_holder_save_is_unknown_without_invented_regrant(self):
        self.oracle.next_owner=Mark('text#0')
        self.observe(0)
        query=self.observe(self.oracle.hazard['absolute_save_candidate_ns'])
        with self.assertRaisesRegex(UnknownEvidence,'SAME_CONTEXT'):
            self.oracle.decide(query)
        self.assertEqual(self.oracle.decisions,[])


if __name__=='__main__':unittest.main()
