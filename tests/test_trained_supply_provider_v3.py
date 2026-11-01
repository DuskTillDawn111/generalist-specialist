import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.simulator import FullHorizonModel
from tests.test_ordered_kernel_profiles_v2 import document,ALIASES
from tests.test_launch_supply_residual_v2 import residual
from tools.predictor_validation_20261027.trained_supply_provider_v3 import TrainedKernelSupplyProvider


class TrainedSupplyProviderTests(unittest.TestCase):
    def provider(self,resolver):
        doc=document();fit=residual(doc)
        for p in doc['profiles']:
            for s in p['segments']:s['duration_ms']=dict(profiled=s['duration_ms']['mean'])
        return TrainedKernelSupplyProvider(doc,fit,'context',baseline_resolver=resolver,allow_cpu_fixture=True,
            allow_independent_host_supply_hypothesis=True,device_progress_hypothesis='equal_context_time_slice')

    def test_prediction_replaces_work_and_retains_observed_target_separately(self):
        p=self.provider(lambda segment:dict(known=True,duration_ns=50000000,measured_target_fallback_used=False))
        queues={a:[dict(request_id=a,task='task',pool_id='input',arrival_ns=0)] for a in ALIASES}
        result=FullHorizonModel(ALIASES,p).forecast(queues,0,1000000000,scenario='profiled',require_energy=False)
        self.assertEqual(result['finishes']['a']['finish_ns'],110000000)
        self.assertEqual(result['finishes']['b']['finish_ns'],110000000)
        kernel=p.profiles['a']['segments'][1]
        self.assertEqual(kernel['measured_profiled_duration_ms']['profiled'],100)
        self.assertEqual(kernel['duration_ms']['profiled'],50)

    def test_unknown_regression_never_falls_back_to_measured_duration(self):
        p=self.provider(lambda segment:dict(known=False,reasons=['UNSUPPORTED_FEATURE_HULL']))
        with self.assertRaises(UnknownEvidence):
            p.plan(dict(alias='a',task='task',pool_id='input'),ALIASES,'profiled')
        self.assertEqual(p.baseline_coverage['a']['unknown_kernels'],1)
        self.assertFalse(p.baseline_coverage['a']['measured_target_fallback_used'])


if __name__=='__main__':unittest.main()
