"""Independent real-candidate rate/budget and original-guard integration."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_projected_handler_context_v42_independent_review import candidate
from tools.predictor_validation_20261027 import projected_handler_bridge_v44 as bridge
from tools.predictor_validation_20261027 import projected_handler_forecast_v54 as factory
from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027.idle_physical_save_hazard_v53 import IdlePhysicalSaveHazard
from tools.predictor_validation_20261027.implementation_source_closure_v27 import implementation_sources
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle
from tools.predictor_validation_20261027.projected_handler_context_v42 import OWNERS


ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'artifacts/predictor-idle-physical-save-v53-20261028-training01/model.json'


class IndependentIdleHazardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model=json.loads(MODEL.read_text())
        fallback=object.__new__(bridge.ObservableSaveHazard);fallback.candidate=candidate()
        cls.candidate=IdlePhysicalSaveHazard(model,fallback)

    def states(self):return {o:OwnerLifecycle(False,False,False) for o in OWNERS}

    def test_actual_rates_are_positive_owner_specific_and_zero_OCR_exposure_is_UNKNOWN(self):
        expected={'generalist_2#0':1807605297,'text#0':4329668994,'text#1':8560044474}
        for owner,ns in expected.items():
            result=self.candidate.query(owner,[],self.states(),0,1_000_000_000)
            rate=.5/(ns/1e9)
            self.assertAlmostEqual(result['hazard_per_ms'],rate/1000)
            self.assertAlmostEqual(result['restricted_candidate_CDF'],-math.expm1(-rate))
            self.assertAlmostEqual(result['posterior_predictive_CDF'],1-math.sqrt((ns/1e9)/(ns/1e9+1)))
            self.assertIsNone(result['conditional_CDF'])
            self.assertIn('ZERO_EVENT_IDLE_RATE_USES_FIXED_JEFFREYS_POSTERIOR_MEAN',result['OOD_reasons'])
        with self.assertRaisesRegex(UnknownEvidence,'IDLE_PHYSICAL_SAVE_NO_HELD_OWNER_EXPOSURE'):
            self.candidate.query('ocr#0',[],self.states(),0)

    def test_every_nonidle_or_unknown_field_and_nonempty_frontier_keeps_original_query(self):
        for owner in OWNERS:
            for field in ('handler_active','issued_outstanding','gpu_ready'):
                for value in (True,None):
                    fields=dict(handler_active=False,issued_outstanding=False,gpu_ready=False)
                    fields[field]=value
                    if field=='gpu_ready' and value is True:fields['issued_outstanding']=True
                    states=self.states();states[owner]=OwnerLifecycle(**fields)
                    with self.subTest(owner=owner,field=field,value=value):
                        if field=='handler_active' and value is None:
                            with self.assertRaises(ValueError):self.candidate.query('text#0',[],states,1000,2000)
                        else:
                            self.assertEqual(self.candidate.query('text#0',[],states,1000,2000),
                                self.candidate.fallback.query('text#0',[],states,1000,2000))
        frontier=[['text#0','text','p',['decode']]]
        self.assertEqual(self.candidate.query('text#0',frontier,self.states(),1000),
            self.candidate.fallback.query('text#0',frontier,self.states(),1000))

    def test_real_controller_integrates_previous_active_and_idle_rates_without_redraw(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'fixture';source.write_text('CPU cumulative budget evidence')
            controller=CumulativeLifecycleHazard(self.candidate,age_edges_ns=[0],
                sources=[receipt(source)],candidate_identity='a'*64,allow_unvalidated_approximation=True)
            active=self.states();active['text#0']=OwnerLifecycle(True,None,False)
            started=controller.start_restore('text#0',0,[],'held-restore',active)
            idle=controller.update(started['state'],1000,[],self.states())
            self.assertAlmostEqual(idle['consumed_hazard'],2*1000/1e6)
            returned=controller.update(idle['state'],2000,[],active)
            expected=2*1000/1e6 + (.5/(4329668994/1e9)/1000)*1000/1e6
            self.assertAlmostEqual(returned['consumed_hazard'],expected)
            self.assertEqual(returned['hazard_budget'],math.log(2))
            self.assertEqual(returned['origin_restore_ns'],0)
            self.assertEqual(returned['observation_ns'],0)
            self.assertEqual(returned['state']['restore_key'],'held-restore')
            self.assertFalse(returned['budget_redrawn'])

    def test_v54_keeps_original_active_mark_and_guard_sources(self):
        self.assertIs(factory.ObservableReferenceMark,bridge.ObservableReferenceMark)
        mark=object.__new__(factory.ObservableReferenceMark)
        mark.candidate=candidate();mark.states=self.states();mark.clock=1000
        with self.assertRaisesRegex(UnknownEvidence,'NO_VISIBLE_ACTIVE_REFERENCE_TARGET'):
            mark.predict_given_save_now(held_owner='text#1',context_age_ns=1000,pending_frontier=[])
        names={Path(ref['path']).name for ref in implementation_sources(factory.__file__)}
        self.assertTrue({'idle_physical_save_hazard_v53.py','projected_handler_bridge_v44.py',
            'learned_lifecycle_scheduler_v49.py','learned_lifecycle_scheduler_v27.py'}<=names)
        self.assertNotIn('idle_physical_mark_bridge_v51.py',names)


if __name__=='__main__':unittest.main()
