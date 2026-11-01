import unittest
from types import SimpleNamespace

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import observable_context_bridge_v38 as v
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle


class PhysicalFixture:
    model={'model_sha256':'a'*64}
    def __init__(self,probabilities):self.probabilities=probabilities;self.observed=[]
    def next_owner(self,features):
        self.observed.append(features)
        return dict(physical_owner_probabilities=dict(self.probabilities),
            most_likely_physical_owner=max(self.probabilities,key=self.probabilities.get),OOD_reasons=[])


class BridgeTests(unittest.TestCase):
    def bridge(self,probabilities):
        result=object.__new__(v.ObservableReferenceMark)
        result.candidate=PhysicalFixture(probabilities);result.clock=None;result.states=None
        return result

    def view(self,clock,active):
        return SimpleNamespace(clock_ns=clock,owner_lifecycle=tuple(
            (o,OwnerLifecycle(o in active,None if o in active else False,False)) for o in v.OWNERS))

    def test_preserves_physical_probabilities_and_separate_owned_work_hypothesis(self):
        raw=dict(zip(v.OWNERS,(.6,.1,.2,.1)))
        bridge=self.bridge(raw);bridge.observe_view(self.view(1000,{'text#0','text#1'}))
        result=bridge.predict_given_save_now(held_owner='text#0',context_age_ns=1000,
            pending_frontier=[],public_workload=[['future','ignored']],previous_held_owner='ocr#0')
        self.assertEqual(result['physical_owner_probabilities'],raw)
        self.assertEqual(result['next_owner_probabilities']['generalist_2#0'],0.)
        self.assertAlmostEqual(result['next_owner_probabilities']['text#0'],2/3)
        # The existing oracle's same-holder rejection remains applicable.
        self.assertEqual(result['most_likely_next_owner'],'text#0')
        self.assertFalse(result['same_holder_probability_removed'])
        self.assertTrue(result['active_reference_service_bridge_is_unvalidated_hypothesis'])
        self.assertTrue(all(x is None for x in bridge.candidate.observed[0]['handler_active'].values()))

    def test_unknown_or_empty_owned_work_domain_cannot_invent_service(self):
        bridge=self.bridge(dict(zip(v.OWNERS,(.25,)*4)))
        bridge.observe_view(self.view(0,set()))
        with self.assertRaisesRegex(UnknownEvidence,'NO_VISIBLE'):
            bridge.predict_given_save_now(held_owner='text#0',context_age_ns=1,pending_frontier=[])
        states={o:OwnerLifecycle(False,False,False) for o in v.OWNERS}
        states['ocr#0']=OwnerLifecycle(None,None,False)
        bridge.observe_view(SimpleNamespace(clock_ns=1,owner_lifecycle=tuple(states.items())))
        with self.assertRaisesRegex(UnknownEvidence,'UNKNOWN_OWNER'):
            bridge.predict_given_save_now(held_owner='text#0',context_age_ns=1,pending_frontier=[])

    def test_monotone_clock_and_explicit_hypothesis_required(self):
        with self.assertRaisesRegex(ValueError,'do not identify'):
            v.ObservableReferenceMark({})
        bridge=self.bridge(dict(zip(v.OWNERS,(.25,)*4)))
        bridge.observe_view(self.view(10,{'text#0'}))
        with self.assertRaises(UnknownEvidence):bridge.observe_view(self.view(9,{'text#0'}))


if __name__=='__main__':unittest.main()
