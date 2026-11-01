"""Fresh flow simulations and safe finite-leaf pruning, CPU fixtures only."""
import copy
import json
from unittest import mock
import unittest

from tests import test_flow_validation as fixtures
from tools.predictor_validation_20261027 import flow_search_check as check


class FlowSearchCheckTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.FlowValidationTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.plan=copy.deepcopy(self.fixture.plan)
        self.model=json.loads(self.fixture.model_path.read_text())
        self.resources=json.loads(self.fixture.resources_path.read_text())
        self.quality=json.loads(self.fixture.quality_path.read_text())

    def test_both_searches_run_fresh_full_horizon_and_retain_all_ties(self):
        with mock.patch.object(check,'FullHorizonModel',wraps=check.FullHorizonModel) as model:
            plain=check.search(self.plan,self.model,self.resources,self.quality)
            bounded=check.search(self.plan,self.model,self.resources,self.quality,bounded=True)
        self.assertEqual(model.call_count,4)
        self.assertEqual(plain['all_tie_ids'],['alternative','base'])
        self.assertEqual(plain['maximum_model_comparison_U'],bounded['maximum_model_comparison_U'])
        self.assertEqual(plain['all_tie_ids'],bounded['all_tie_ids'])
        self.assertFalse(bounded['bound_pruned'])
        self.assertEqual(plain['records'][0]['result']['cohort_size'],2)
        self.assertIsNone(plain['records'][0]['result']['physical_eligibility'])

    def test_bound_dominates_actual_metric_and_does_not_need_simulation(self):
        with mock.patch.object(check,'FullHorizonModel',side_effect=AssertionError('bound simulated')):
            bound=check.upper_bound(self.plan,self.plan['candidates'][0],self.quality)
        result=check.evaluate(self.plan,self.plan['candidates'][0],self.model,self.resources,self.quality)
        self.assertLessEqual(result['comparison_U'],bound['U_upper'])
        self.assertAlmostEqual(bound['M_lower'],.55)
        self.assertTrue(bound['Q_exact_for_training_proxy'])

    def test_unknown_quality_uses_zero_lower_not_zero_objective(self):
        bound=check.upper_bound(self.plan,self.plan['candidates'][0],{})
        self.assertEqual(bound['Q_lower'],0)
        self.assertFalse(bound['Q_exact_for_training_proxy'])
        result=check.search(self.plan,self.model,self.resources,{})
        self.assertEqual(result['unknown_ids'],['alternative','base'])
        self.assertIsNone(result['maximum_model_comparison_U'])

    def test_missing_memory_is_unknown_not_infeasible(self):
        for candidate in self.plan['candidates']:candidate.pop('predicted_prephase_peak_mib')
        result=check.search(self.plan,self.model,self.resources,self.quality,bounded=True)
        self.assertEqual(result['unknown_ids'],['alternative','base'])
        self.assertFalse(result['model_ineligible_ids'])

    def test_memory_filter_preserved_and_finite_load_is_not_original_admission(self):
        self.plan['memory_budget_mib']=1
        result=check.search(self.plan,self.model,self.resources,self.quality)
        self.assertEqual(result['model_ineligible_ids'],['alternative','base'])
        self.assertFalse(result['unknown_ids'])
        self.plan=copy.deepcopy(self.fixture.plan);self.plan['window_seconds']=.1
        result=check.search(self.plan,self.model,self.resources,self.quality)
        row=result['records'][0]['result']
        self.assertGreater(row['finite_forecast_load_proxy_by_family']['text'],.8)
        self.assertIsNone(row['planning_eligible'])
        self.assertTrue(row['model_eligible'])
        self.assertTrue(row['model_constraints']['drain_complete'])

    def test_explicit_common_startup_compares_but_absolute_unknown(self):
        for candidate in self.plan['candidates']:candidate.pop('predicted_startup_joules')
        result=check.evaluate(self.plan,self.plan['candidates'][0],self.model,self.resources,self.quality)
        self.assertIsNone(result['comparison_U'])
        self.plan['startup_comparison_policy']={'kind':'common_deployment_backend_omitted','group_id':'fixed',
            'candidate_ids':['base','alternative']}
        result=check.evaluate(self.plan,self.plan['candidates'][0],self.model,self.resources,self.quality)
        self.assertIsNotNone(result['comparison_U']);self.assertIsNone(result['absolute_U'])

    def test_strict_bound_prunes_dominated_leaf_and_preserves_equal_upper(self):
        self.plan['aliases']['bad']={'family':'bad','supported_tasks':['text']}
        self.plan['candidates'][1]['routes']=dict.fromkeys(['r1','r2'],'bad')
        quality={'text':{'text':0},'bad':{'text':1}}
        fake={'comparison_U':-.2,'model_eligible':True,'planning_eligible':None}
        with mock.patch.object(check,'evaluate',return_value=fake) as evaluate:
            bounded=check.search(self.plan,self.model,self.resources,quality,bounded=True)
        self.assertEqual(evaluate.call_count,1)
        self.assertEqual(bounded['bound_pruned'][0]['candidate_id'],'alternative')
        self.plan=copy.deepcopy(self.fixture.plan)
        equal=check.upper_bound(self.plan,self.plan['candidates'][0],self.quality)['U_upper']
        with mock.patch.object(check,'evaluate',return_value={'comparison_U':equal,'model_eligible':True,'planning_eligible':None}):
            bounded=check.search(self.plan,self.model,self.resources,self.quality,bounded=True)
        self.assertEqual(bounded['all_tie_ids'],['alternative','base'])
        self.assertFalse(bounded['bound_pruned'])

    def test_report_sources_and_changes_checked_before_simulating(self):
        output=self.fixture.root/'search'
        result=check.run(self.fixture.plan_path,self.fixture.model_path,self.fixture.resources_path,self.fixture.quality_path,output)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['plain']['fresh_forecast_calls'],2)
        self.assertFalse(result['physical_optimality_certified'])
        self.assertFalse(result['original_gate_constrained_optimality_certified'])
        self.assertEqual(result['original_rho_admission_unknown_ids'],['alternative','base'])
        self.fixture.training.write_text('{}')
        with self.assertRaisesRegex(ValueError,'source receipt'):
            check.run(self.fixture.plan_path,self.fixture.model_path,self.fixture.resources_path,self.fixture.quality_path,self.fixture.root/'changed')

    def test_original_rho_gate_is_recorded_separately_from_model_comparison(self):
        gate={'status':'DIAGNOSTIC_UNBOUND','rho_admission_passed':None,'rho_threshold_satisfied':True}
        with mock.patch.object(check,'original_rho',return_value=gate) as original:
            result=check.evaluate(self.plan,self.plan['candidates'][0],self.model,self.resources,self.quality,catalog={})
        original.assert_called_once()
        self.assertTrue(result['model_eligible'])
        self.assertIsNone(result['planning_eligible'])
        self.assertIsNone(result['original_rho']['rho_admission_passed'])


if __name__=='__main__':unittest.main()
