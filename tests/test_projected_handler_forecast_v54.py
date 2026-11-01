"""Use actual frozen training candidates with a small original provider shell."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from tools.predictor_validation_20261027 import projected_handler_forecast_v54 as v
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle
from tools.predictor_validation_20261027.projected_handler_bridge_v44 import ObservableReferenceMark
from tools.predictor_validation_20261027.whole57_shared_observable_risk_v31 import OWNERS

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/predictor-projected-handler-v50-20261028-configuration01/configuration.json'
IDLE=ROOT/'artifacts/predictor-idle-physical-save-v53-20261028-training01/model.json'


class ActualTrainingConnectionTests(unittest.TestCase):
    def config(self):
        config=json.loads(BASE.read_text());config['schema']=v.SCHEMA
        config['options']=copy.deepcopy(v.OPTIONS_V54);config['idle_SAVE_model']=v.receipt(IDLE)
        return config

    @staticmethod
    def small_original_provider(model,config,**kwargs):
        model.compiled=SimpleNamespace(sources=[])

    def test_actual_idle_exposure_changes_only_idle_hazard_not_mark_or_budget(self):
        with mock.patch.object(v.V16Forecast,'__init__',self.small_original_provider):
            model=v.ObservableContextForecast(self.config(),allow_unvalidated_training_hypothesis=True)
        self.assertIs(type(model.mark),ObservableReferenceMark)
        states={o:OwnerLifecycle(False,False,False) for o in OWNERS}
        idle=model.save_controller._candidate.query('generalist_2#0',[],states,0,1_000_000)
        self.assertGreater(idle['hazard_per_ms'],0)
        self.assertAlmostEqual(idle['hazard_per_ms'],.5/1.807605297/1000)
        states['generalist_2#0']=OwnerLifecycle(True,None,False)
        candidate=model.save_controller._candidate
        expected=candidate.fallback.query('generalist_2#0',[],states,0,1_000_000)
        self.assertEqual(candidate.query('generalist_2#0',[],states,0,1_000_000),expected)
        started=model.save_controller.start_restore('generalist_2#0',0,[],'original_restore',states)
        self.assertFalse(started['budget_redrawn'])
        self.assertEqual(started['origin_restore_ns'],0)
        self.assertTrue(any(Path(r['path'])==IDLE for r in model._lifecycle_sources))
        self.assertTrue(any(Path(r['path']).name=='idle_physical_save_hazard_v53.py' for r in model._lifecycle_sources))

    def test_mismatched_training_capture_candidate_rejected_before_rate_use(self):
        actual=v.checked
        def wrong_capture(ref):
            value=actual(ref)
            if ref['path']==str(IDLE):value=copy.deepcopy(value);value['captures']={}
            return value
        with mock.patch.object(v.V16Forecast,'__init__',self.small_original_provider),mock.patch.object(v,'checked',side_effect=wrong_capture):
            with self.assertRaisesRegex(ValueError,'same four projected-handler'):
                v.ObservableContextForecast(self.config(),allow_unvalidated_training_hypothesis=True)


if __name__=='__main__':unittest.main()
