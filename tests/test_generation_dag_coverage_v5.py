import copy
import unittest

from tools.predictor_validation_20261027.generation_dag_coverage_v5 import select_scenarios


class Provider:
    def scenarios(self,request):
        self.request=request
        return dict(scenarios=[dict(generated_tokens=n,finish_reason='eos',forward_count=n,
            prefill_forward_count=1,decode_forward_count=n-1,forward_batch_size=1,
            selected_token_forward_relation='one_forward_per_selected_token',weight=w)
            for n,w in ((5,2/3),(6,1/3))])


def public():
    return dict(alias='a',family='text',task='text',pool_id='known',input_tokens=414,
        protocol_sha256='policy',effective_generation={'max_new_tokens':32},
        native_execution_projection_sha256='natural',request_id='future')


def template():
    p=public();scenario=Provider().scenarios({})['scenarios'][0]
    return dict(**{k:p[k] for k in ('alias','pool_id','input_tokens','protocol_sha256','effective_generation')},
        work=scenario,template={'dag':'training-source'})


class GenerationDagSupportTests(unittest.TestCase):
    def test_unknown_work_mass_preserved_and_no_observed_future_output_selected(self):
        provider=Provider();result=select_scenarios(provider,public(),[template()])
        self.assertEqual(result['status'],'UNKNOWN_INCOMPLETE_WORK_SCENARIOS')
        self.assertAlmostEqual(result['covered_probability_mass'],2/3)
        self.assertFalse(result['unconditional_forecast_complete'])
        self.assertEqual(len(result['outcomes']),2)
        self.assertEqual(result['outcomes'][1]['status'],'UNKNOWN')
        self.assertNotIn('pool_id',provider.request)
        for field in ('generated_tokens','worker_response','finish_reason'):
            p=public();p[field]='future observed output'
            with self.assertRaisesRegex(ValueError,'public'):select_scenarios(provider,p,[template()])

    def test_other_input_or_beam_work_cannot_supply_missing_template(self):
        one=template();other=copy.deepcopy(one)
        other['work']=Provider().scenarios({})['scenarios'][1];other['pool_id']='different'
        result=select_scenarios(Provider(),public(),[one,other])
        self.assertFalse(result['unconditional_forecast_complete'])
        other['pool_id']='known';other['work']['forward_count']=9
        self.assertFalse(select_scenarios(Provider(),public(),[one,other])['unconditional_forecast_complete'])
        other['work']=Provider().scenarios({})['scenarios'][1]
        self.assertTrue(select_scenarios(Provider(),public(),[one,other])['unconditional_forecast_complete'])


if __name__=='__main__':unittest.main()
