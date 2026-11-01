"""Independent old/new interval arithmetic and actual Fraction cross-check."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027.context_survival_v12_3 import ContextEnvelopeSurvival,SCHEMA,clip_episodes
from tools.predictor_validation_20261027.profiling import digest
from tests import test_context_survival_v12_independent_review as old_counterexamples
from tests.test_context_survival_v12_numeric_independent_review import independently_query

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'artifacts/predictor-context-survival-v12_3-20261028-training01/model.json'


class NewSurvivalIndependentTests(unittest.TestCase):
    def test_all_four_prior_statistical_counterexamples_execute_on_new_provider(self):
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(old_counterexamples.ContextSurvivalIndependentTests)
        with patch.object(old_counterexamples,'ContextEnvelopeSurvival',ContextEnvelopeSurvival),patch.object(old_counterexamples,'SCHEMA',SCHEMA):
            result=unittest.TestResult();suite.run(result)
        self.assertEqual(result.testsRun,4);self.assertEqual(result.failures,[]);self.assertEqual(result.errors,[])

    def test_clip_preserves_restore_age_censors_boundaries_and_no_overlap(self):
        self.assertEqual(clip_episodes(0,100,20,80,[30,60]),[
            dict(start_ns=20,end_ns=30,age_entry_ns=20,age_exit_ns=30,save_observed=False),
            dict(start_ns=30,end_ns=60,age_entry_ns=30,age_exit_ns=60,save_observed=False),
            dict(start_ns=60,end_ns=80,age_entry_ns=60,age_exit_ns=80,save_observed=False)])
        first=clip_episodes(0,100,20,50,[]);second=clip_episodes(0,100,50,100,[])
        self.assertEqual(first[0]['age_exit_ns'],second[0]['age_entry_ns']);self.assertTrue(second[0]['save_observed'])
        for cuts in ([60,30],[30,30]):
            with self.assertRaises(ValueError):clip_episodes(0,100,20,80,cuts)
        self.assertEqual(clip_episodes(0,100,100,110,[]),[])

    def test_actual_32_supported_cells_match_independent_exact_risksets(self):
        model=json.loads(MODEL.read_text());provider=ContextEnvelopeSurvival(model,allow_unvalidated_survival_hypothesis=True,allow_cpu_fixture=True)
        checked=0
        for (owner,_),rows in sorted(provider.groups.items(),key=lambda item:str(item[0])):
            if len({r['capture'] for r in rows})<2:continue
            selected=None
            for age in sorted({0,*[r['age_entry_ns'] for r in rows]}):
                expected=independently_query(rows,age)
                if expected is not None:selected=(age,expected);break
            if selected is None:continue
            age,(horizon,mass,tail,mean,median)=selected;actual=provider.query(owner,rows[0]['frontier'],age)
            self.assertEqual(actual['restricted_residual_horizon_ns'],horizon);self.assertEqual(actual['median_residual_ns'],median)
            self.assertEqual(len(actual['residual_event_mass']),len(mass))
            for observed,(t,p) in zip(actual['residual_event_mass'],mass):
                self.assertEqual(observed['residual_ns'],t);self.assertAlmostEqual(observed['probability'],float(p),delta=1e-11)
            self.assertAlmostEqual(actual['unresolved_tail_probability'],float(tail),delta=1e-11)
            self.assertAlmostEqual(actual['restricted_mean_residual_ns'],float(mean),delta=max(1e-4,float(mean)*1e-10))
            checked+=1
            if checked==32:break
        self.assertEqual(checked,32)


if __name__=='__main__':unittest.main()
