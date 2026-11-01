import unittest
from tests.test_generation_dag_coverage_v5 import public,template,Provider
from tools.predictor_validation_20261027.generation_dag_coverage_v6 import select_scenarios


class ExactProvider:
    def scenarios(self,request):
        self.request=request
        return dict(scenarios=[dict(Provider().scenarios({})['scenarios'][0],weight=1.)])


class ExactContentDagSupportTests(unittest.TestCase):
    def test_content_conditioning_uses_public_hash_and_complete_mass(self):
        p=dict(public(),input_content_sha256='immutable-public-content')
        t=dict(template(),input_content_sha256=p['input_content_sha256']);provider=ExactProvider()
        result=select_scenarios(provider,p,[t])
        self.assertTrue(result['unconditional_forecast_complete'])
        self.assertEqual(result['covered_probability_mass'],1.)
        self.assertEqual(provider.request['input_content_sha256'],p['input_content_sha256'])
        self.assertNotIn('pool_id',provider.request)
        result=select_scenarios(provider,dict(p,input_content_sha256='different-content'),[t])
        self.assertEqual(result['covered_probability_mass'],0.)
        self.assertFalse(result['unconditional_forecast_complete'])
        with self.assertRaises(ValueError):select_scenarios(provider,dict(p,generated_tokens=5),[t])


if __name__=='__main__':unittest.main()
