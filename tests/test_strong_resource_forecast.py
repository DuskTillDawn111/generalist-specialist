"""No GPU/result access: support gates and missing-data abstention contracts."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('prequential',ROOT/'tools/spillover_context_20261022/seal_strong_resource_forecast.py')
f=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(f)


class ForecastSupport(unittest.TestCase):
    def rows(self):
        return [dict(run_id=f'r{i}',deployment={'x':1},features=dict(intercept=1.,raw_resident_over_budget=r,
            sum_baseline_pressure=r,workers=n)) for i,(r,n) in enumerate([(.2,1),(.4,2),(.8,5)])]
    def model(self):
        return dict(name='M_residency_only',target='M_raw_peak_over_budget',features=['raw_resident_over_budget'],
            all_run_fit=dict(coefficients=[1.1]))

    def test_inside_individual_ranges_is_not_inside_joint_hull(self):
        features=dict(raw_resident_over_budget=.7,workers=5)
        check=f.hull_check(self.rows(),features,['raw_resident_over_budget','workers'])
        self.assertFalse(check['inside'])
        self.assertTrue(.2<features['raw_resident_over_budget']<.8)

    def test_reduced_projection_is_its_own_domain_not_shared_full_context_gate(self):
        result=f.forecast_model(self.model(),dict(raw_resident_over_budget=.7,workers=5),self.rows(),{'x':1})
        self.assertTrue(result['model_feature_projection_hull']['inside'])
        self.assertFalse(result['full_context_hull']['inside']);self.assertAlmostEqual(result['forecast'],.77)

    def test_missing_input_does_not_fill(self):
        result=f.forecast_model(self.model(),dict(raw_resident_over_budget=None,workers=2),self.rows(),{'x':1})
        self.assertEqual(result['missing_inputs'],['raw_resident_over_budget']);self.assertIsNone(result['forecast'])
        self.assertIsNone(result['diagnostic_extrapolation_only']['value'])

    def test_unseen_composition_remains_unknown_even_inside_hull(self):
        result=f.forecast_model(self.model(),dict(raw_resident_over_budget=.4,workers=2),self.rows(),{'new':2})
        self.assertTrue(result['full_context_hull']['inside']);self.assertIsNone(result['forecast'])
        self.assertIn('UNOBSERVED_DEPLOYMENT_COMPOSITION',result['reasons'])
        self.assertAlmostEqual(result['diagnostic_extrapolation_only']['value'],.44)
        self.assertFalse(result['diagnostic_extrapolation_only']['valid_prediction'])

    def test_outside_candidate_feature_hull_still_has_separate_numerical_probe(self):
        result=f.forecast_model(self.model(),dict(raw_resident_over_budget=1.2,workers=2),self.rows(),{'x':1})
        self.assertIsNone(result['forecast']);self.assertFalse(result['model_feature_projection_hull']['inside'])
        self.assertAlmostEqual(result['diagnostic_extrapolation_only']['value'],1.32)
        self.assertTrue(result['formula_extrapolation_evaluated'])
        self.assertFalse(result['diagnostic_extrapolation_only']['formal_score'])
        self.assertFalse(result['diagnostic_extrapolation_only']['execution_admission'])

    def test_nonfinite_input_does_not_create_probe(self):
        result=f.forecast_model(self.model(),dict(raw_resident_over_budget=float('nan'),workers=2),self.rows(),{'x':1})
        self.assertIsNone(result['forecast']);self.assertIsNone(result['diagnostic_extrapolation_only']['value'])

    def test_complete_observed_inside_support_can_produce_development_value(self):
        result=f.forecast_model(self.model(),dict(raw_resident_over_budget=.4,workers=2),self.rows(),{'x':1})
        self.assertAlmostEqual(result['forecast'],.44);self.assertFalse(result['activated'])

    def test_missing_exact_baseline_makes_entire_demand_unknown(self):
        packet=dict(config=dict(gpu_memory_mib=100,catalog=dict(memory_table_mib={'x':{'1':20}},capacity_factors={'x':{'1':1}})))
        plan=dict(deployment={'x':1},routing={'text':{'x':1}})
        features,receipt=f.derive_features(packet,plan,[])
        self.assertEqual(features['raw_resident_over_budget'],.2)
        self.assertIsNone(features['sum_baseline_pressure']);self.assertTrue(receipt['no_missing_value_fill'])

    def test_missing_residency_does_not_replace_with_zero(self):
        packet=dict(config=dict(gpu_memory_mib=100,catalog=dict(memory_table_mib={},capacity_factors={})))
        features,_=f.derive_features(packet,dict(deployment={'x':1},routing={}),[])
        self.assertIsNone(features['raw_resident_over_budget'])


if __name__=='__main__':unittest.main()
