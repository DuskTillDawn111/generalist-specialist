import unittest

from gsdeploy.forecasting.simulator import FullHorizonModel
from tests.test_ordered_kernel_profiles_v2 import document, ALIASES
from tools.predictor_validation_20261027.launch_supply_residual_v2 import (
    SCHEMA, partition_nonnegative, fit_profile, LaunchSupplyResidualProvider)


def residual(doc):
    fits=[]
    for profile in doc['profiles']:
        p=dict(profile,segments=[dict(s,duration_ms=dict(profiled=s['duration_ms']['mean']))
                                 for s in profile['segments'] if s['kind']=='gpu'])
        fits.append(fit_profile(p,round(p['segments'][0]['duration_ms']['profiled']*1e6)+10000000,[100]))
    return dict(schema=SCHEMA,execution_context_sha256='context',training_only=True,fits=fits,fit_sha256='fixture')


class LaunchSupplyResidualTests(unittest.TestCase):
    def test_exact_nonnegative_partition_and_negative_rejected(self):
        self.assertEqual(partition_nonnegative(5,[1,1,1]),[2,2,1])
        self.assertEqual(partition_nonnegative(0,[1,2]),[0,0])
        with self.assertRaises(ValueError):partition_nonnegative(-1,[1])
        with self.assertRaises(ValueError):partition_nonnegative(1,[0])

    def test_negative_natural_residual_stays_unknown_without_clip(self):
        profile=document()['profiles'][0]
        profile['segments']=[dict(profile['segments'][1],duration_ms=dict(profiled=100))]
        result=fit_profile(profile,90000000,[100])
        self.assertFalse(result['known']);self.assertEqual(result['residual_ns'],-10000000)

    def provider(self,**options):
        doc=document()
        for profile in doc['profiles']:
            for s in profile['segments']:s['duration_ms']=dict(profiled=s['duration_ms']['mean'])
        fit=residual(document())
        return LaunchSupplyResidualProvider(doc,fit,'context',allow_cpu_fixture=True,**options)

    def test_full_horizon_runs_concurrent_supply_then_context_time_slice(self):
        p=self.provider(allow_independent_host_supply_hypothesis=True,device_progress_hypothesis='equal_context_time_slice')
        queues={a:[dict(request_id=a,task='task',pool_id='input',arrival_ns=0)] for a in ALIASES}
        result=FullHorizonModel(ALIASES,p).forecast(queues,0,1000000000,scenario='profiled',require_energy=False)
        self.assertEqual(result['finishes']['a']['finish_ns'],210000000)
        self.assertEqual(result['finishes']['b']['finish_ns'],310000000)
        self.assertFalse(p.physical_evidence)

    def test_explicit_supply_opt_in_is_required(self):
        p=self.provider()
        row=dict(alias='a',task='task',pool_id='input',request_id='a')
        plan=p.plan(row,ALIASES,'profiled');segment=plan['segments'][0]
        row.update(profile_id=plan['profile_id'],segment_id=segment['id'],segment_kind=segment['kind'],
                   segment_duration_ms=segment['duration_ms'])
        self.assertEqual(p.service(row,dict(a=row),ALIASES,'profiled')['reason'],
                         'INDEPENDENT_HOST_SUPPLY_HYPOTHESIS_OPT_IN_REQUIRED')

    def test_context_switch_cost_is_separate_service_hypothesis(self):
        p=self.provider(allow_independent_host_supply_hypothesis=True,
                        device_progress_hypothesis='equal_context_time_slice',context_switch_ms=3)
        rows={}
        for alias in ALIASES:
            row=dict(alias=alias,task='task',pool_id='input',request_id=alias)
            plan=p.plan(row,ALIASES,'profiled');segment=plan['segments'][1]
            row.update(profile_id=plan['profile_id'],segment_id=segment['id'],segment_kind=segment['kind'],
                       segment_duration_ms=segment['duration_ms']);rows[alias]=row
        result=p.service(rows['a'],rows,ALIASES,'profiled')
        self.assertEqual(result['service_ms'],203)
        self.assertEqual(rows['a']['segment_duration_ms'],100)
        self.assertEqual(result['device_progress_hypothesis'],'equal_context_time_slice')


if __name__=='__main__':unittest.main()
