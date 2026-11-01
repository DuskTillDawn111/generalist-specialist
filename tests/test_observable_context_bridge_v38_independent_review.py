"""Independent real-candidate bridge/controller/scheduler boundary fixtures."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_learned_context_oracle_v14 import Handoff
from tools.predictor_validation_20261027 import observable_context_bridge_v38 as b
from tools.predictor_validation_20261027 import observable_context_forecast_v38 as f
from tools.predictor_validation_20261027 import whole57_observable_context_v34 as v
from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode
from tools.predictor_validation_20261027.held_context_scheduler_v14 import InitialContextDecision
from tools.predictor_validation_20261027.implementation_source_closure_v27 import implementation_sources,strict_references
from tools.predictor_validation_20261027.learned_lifecycle_scheduler_v27 import LearnedLifecycleOracle,LearnedLifecycleScheduler
from tools.predictor_validation_20261027.owner_lifecycle_engine_v21 import LifecycleSchedulingView
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle


def model(probabilities=(.1,.1,.7,.1)):
    width=len(v.mark_names());coefs=[[0.]*4 for _ in range(width)]
    coefs[0]=[math.log(p) for p in probabilities]
    mark=dict(feature_names=v.mark_names(),coefficients=coefs,center=[0.]*width,scale=[1.]*width,
        feature_minimum=[0.]*width,feature_maximum=[100.]*width,label_counts={o:1 for o in v.OWNERS},
        state_age_ranges_ns={v.digest([o,p]):[0,100_000_000] for o in v.OWNERS for p in (None,*v.OWNERS)})
    # 6/ms in the fixed 2us--5us bin, 2/ms elsewhere; no lifecycle effect.
    hazard=dict(columns=['intercept','age_bin:2'],coefficients=[math.log(2.),math.log(3.)],
        fixed_age_edges_ns=list(v.EDGES_NS),training_vocabulary=['holder:'+o for o in v.OWNERS],
        state_age_support_ns={o:[[0,100_000_000]] for o in v.OWNERS})
    result=dict(schema=v.SCHEMA,parameters=copy.deepcopy(v.PARAMETERS),training_only=True,
        validation_labels_used=False,fitted=dict(mark=mark,hazard=hazard),sources=[])
    result['model_sha256']=v.digest(result)
    return result


class ObservableBridgeIndependentTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.source=Path(tmp.name)/'training-fixture';self.source.write_text('CPU fixture only')
        self.low={o:OwnerLifecycle(False,False,False) for o in v.OWNERS}
        self.low['text#0']=OwnerLifecycle(True,True,True)

    def controller(self):
        return CumulativeLifecycleHazard(b.ObservableSaveHazard(model()),age_edges_ns=v.EDGES_NS,
            sources=[v.receipt(self.source)],candidate_identity='a'*64,allow_unvalidated_approximation=True)

    def test_real_candidate_age_bins_integrate_without_lifecycle_or_empty_budget_reset(self):
        c=self.controller()
        first=c.start_restore('text#0',0,[],'restore-1',self.low)
        changed=dict(self.low,**{'text#1':OwnerLifecycle(True,None,False)})
        direct=c.update(first['state'],3000,[],changed)
        stepped=c.update(first['state'],1000,[],self.low)
        stepped=c.update(stepped['state'],3000,[['text#0','text','known',['decode']]],changed)
        self.assertAlmostEqual(direct['consumed_hazard'],2.*2000/1e6+6.*1000/1e6)
        self.assertAlmostEqual(direct['consumed_hazard'],stepped['consumed_hazard'])
        self.assertEqual(direct['absolute_save_candidate_ns'],first['absolute_save_candidate_ns'])
        self.assertEqual(stepped['hazard_budget'],first['hazard_budget'])
        self.assertEqual(direct['origin_restore_ns'],0)
        self.assertFalse(direct['budget_redrawn'])
        self.source.write_text('changed fixture')
        with self.assertRaises(ValueError):c.assert_sources_unchanged()

    def test_real_physical_probabilities_keep_all_classes_when_work_domain_changes(self):
        bridge=b.ObservableReferenceMark(model(),allow_unvalidated_active_service_bridge=True)
        raw=v.ObservableContextCandidate(model()).next_owner(b.feature_projection('text#0',None,3000))['physical_owner_probabilities']
        def observe(clock,states):
            bridge.observe_view(LifecycleSchedulingView(clock,(),tuple(states.items()),(),()))
        observe(3000,self.low)
        one=bridge.predict_given_save_now(held_owner='text#0',context_age_ns=3000,pending_frontier=[])
        # A running handler with readiness unknown remains a separately visible work-domain hypothesis.
        states=dict(self.low,**{'text#1':OwnerLifecycle(True,None,False)})
        observe(3001,states)
        two=bridge.predict_given_save_now(held_owner='text#0',context_age_ns=3000,pending_frontier=[],
            public_workload=[['changed-future']],pending_age_ns_by_owner={'unused':999})
        self.assertEqual(one['physical_owner_probabilities'],raw)
        self.assertEqual(two['physical_owner_probabilities'],raw)
        self.assertEqual(one['next_owner_probabilities']['text#0'],1.)
        self.assertGreater(two['next_owner_probabilities']['text#1'],0.)
        self.assertEqual(two['most_likely_next_owner'],'text#0')
        self.assertFalse(two['same_holder_probability_removed'])

    def test_same_holder_prediction_reaches_original_scheduler_unknown_guard(self):
        bridge=b.ObservableReferenceMark(model(),allow_unvalidated_active_service_bridge=True)
        oracle=LearnedLifecycleOracle(initial_prior=None,save_controller=self.controller(),next_owner=bridge,
            handoff=Handoff(),public_setup={},public_workload=[],source_identity='a'*64,
            initial_quantile=.5,initial_readiness_rule='lower_bound_hypothesis',
            handoff_readiness_rule='lower_bound_reference_availability_hypothesis',
            allow_initial_pooling=True,allow_observer_transfer=True,allow_flagged_statistical_extrapolation=True)
        oracle.initialize=lambda query:InitialContextDecision('text#0',0,0,'a'*64)
        scheduler=LearnedLifecycleScheduler(v.OWNERS,oracle=oracle)
        def view(clock):
            active=(ActiveGpuNode(('text#0','gpu'),1e9,'decode','req','text','known'),)
            return LifecycleSchedulingView(clock,active,tuple(self.low.items()),(),())
        scheduler.decide(view(0));crossing=oracle.hazard['absolute_save_candidate_ns']
        with self.assertRaisesRegex(UnknownEvidence,'SAME_CONTEXT_REACQUISITION'):
            scheduler.decide(view(crossing))
        self.assertEqual(oracle.decisions,[])

    def test_v38_closure_contains_real_bridge_fit_controller_and_old_scheduler(self):
        refs=implementation_sources(f.__file__);paths={Path(r['path']).name for r in refs}
        self.assertTrue({'observable_context_bridge_v38.py','whole57_observable_context_v34.py',
            'cumulative_lifecycle_hazard_v24.py','learned_lifecycle_scheduler_v27.py',
            'learned_lifecycle_oracle_v25.py','learned_context_oracle_v14.py',
            'owner_lifecycle_engine_v21.py','learned_availability_forecast_v20.py'}<=paths)
        ref=v.receipt(b.__file__)
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            strict_references([refs,dict(ref,sha256='0'*64)])


if __name__=='__main__':unittest.main()
