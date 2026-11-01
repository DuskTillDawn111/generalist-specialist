"""Independent analytic CPU review; fixed law stubs are not fitted evidence."""
import copy
import math
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.kernel_provider import KernelResourceProvider
from gsdeploy.forecasting.simulator import FullHorizonModel
from gsdeploy.resources import identity
from tests.test_kernel_competition_features_v2 import capture, provider
from tests.test_kernel_provider import ALIASES
from tools.predictor_validation_20261027 import kernel_competition_features_v2 as f
from tools.predictor_validation_20261027 import kernel_competition_response_provider_v1 as m


def row(rid, arrival=0):
    return dict(request_id=rid,task='task',pool_id='input',arrival_ns=arrival)


def response(*, catalog=None, beta=None, sigma=.4, extra_baselines=(), reference_b=200):
    catalog=provider().provider if catalog is None else catalog
    coefficients=[math.log(2),0,0,0,0,0] if beta is None else beta
    groups=[('a','a','kernel',100),('b','b','kernel',reference_b),*extra_baselines]
    fit=dict(numerical_qualification=True,features=list(f.FEATURES),feature_schema=f.SCHEMA,
        response_domain=m.DOMAIN,coefficients=coefficients,lognormal_reference_sigma=sigma,
        baselines=[dict(group=[a,p,s,'context','synthetic-nsys'],log_reference_ms=math.log(ref))
                   for a,p,s,ref in groups])
    with patch.object(f,'fit_response',return_value=(fit,None,None,None)):
        bundle=m.training_bundle(catalog,[capture(f.KernelCompetitionFeatureProvider(catalog))],allow_cpu_fixture=True)
    p=m.KernelCompetitionResponseProvider(catalog,bundle,allow_unvalidated_profiled_response=True,
        allow_cpu_fixture=True,scenarios={
            'mean':dict(catalog_scenario='mean',reference_functional='mean'),
            'median':dict(catalog_scenario='mean',reference_functional='quantile',quantile=.5),
            'upper':dict(catalog_scenario='mean',reference_functional='quantile',
                         quantile=(1+math.erf(1/math.sqrt(2)))/2)})
    return p,bundle


def active(p,scenario='median'):
    out={}
    for alias in ALIASES:
        r=dict(row(alias),alias=alias);plan=p.plan(r,ALIASES,scenario);s=plan['segments'][0]
        out[alias]=dict(r,profile_id=plan['profile_id'],segment_id=s['id'],segment_kind=s['kind'],
            segment_duration_ms=s['duration_ms'])
    return out


class KernelResponseIndependentTests(unittest.TestCase):
    def test_nonmedian_quantile_and_lognormal_mean_use_reference_clock(self):
        p,_=response(beta=[0]*6)
        for scenario,multiplier in [('median',1),('mean',math.exp(.08)),('upper',math.exp(.4))]:
            rows=active(p,scenario)
            for alias,base in [('a',100),('b',200)]:
                value=p.service(rows[alias],rows,ALIASES,scenario)
                self.assertAlmostEqual(value['reference_clock_ms'],base*multiplier)
                self.assertAlmostEqual(value['service_ms'],base*multiplier)
                self.assertFalse(value['joint_request_distribution_identified'])

    def test_all_declared_covariates_enter_fitted_log_clock_without_occupancy_speed(self):
        beta=[.11,.22,.33,.44,.55,.66];p,_=response(beta=beta)
        rows=active(p)
        static_a=p.catalog.segments['a','kernel']['kernel']
        static_b=p.catalog.segments['b','kernel']['kernel'];hardware=p.catalog.hardware
        from gsdeploy.forecasting.kernel_resources import static_occupancy,check_resident_blocks
        occupied=static_occupancy(static_b,hardware)
        vector=[1,int(static_a['name']==static_b['name']),
            occupied['allocated_registers_per_block']/hardware['registers_per_sm'],
            occupied['allocated_shared_bytes']/hardware['shared_bytes_per_sm'],
            occupied['warps_per_block']/hardware['max_warps_per_sm'],
            int(not check_resident_blocks({'a':static_a,'b':static_b},{'a':1,'b':1},hardware)['feasible_under_declared_constraints'])]
        with patch.object(p.catalog,'service',side_effect=AssertionError('analytical occupancy service forbidden')):
            actual=p.service(rows['a'],rows,ALIASES,'median')
        self.assertAlmostEqual(actual['service_ms'],100*math.exp(sum(b*x for b,x in zip(beta,vector))))
        self.assertFalse(actual['static_occupancy_used_as_throughput'])

    def test_generated_peer_kernel_change_reprices_and_conserves_work(self):
        catalog=provider().provider;doc=copy.deepcopy(catalog.document)
        doc['profiles'][1]['segments'][0]['kernel']['name']=doc['profiles'][0]['segments'][0]['kernel']['name']
        second=copy.deepcopy(doc['profiles'][1]['segments'][0]);second['id']='different'
        second['kernel']['name']='different-kernel';second['duration_ms']={'mean':7}
        doc['profiles'][1]['segments'].append(second)
        catalog=KernelResourceProvider(doc,'context',allow_cpu_fixture=True)
        p,_=response(catalog=catalog,beta=[0,math.log(2),0,0,0,0],reference_b=20,
            extra_baselines=[('b','b','different',40)])
        result=FullHorizonModel(ALIASES,p).forecast({'a':[row('a')],'b':[row('b')]},
            0,1_000_000_000,'median',require_energy=False)
        self.assertEqual(result['finishes']['b']['finish_ns'],80_000_000)
        self.assertEqual(result['finishes']['a']['finish_ns'],120_000_000)
        intervals=[x for x in result['transitions'] if x['active_requests']]
        self.assertEqual([x['until_ns'] for x in intervals],[40_000_000,80_000_000,120_000_000])
        self.assertAlmostEqual(intervals[0]['service_ms']['a'],200)
        self.assertAlmostEqual(intervals[1]['service_ms']['a'],100)
        # 40/200 + 40/100 + 40/100 = one complete reference unit.
        self.assertAlmostEqual(sum((x['until_ns']-x['time_ns'])/1e6/x['service_ms']['a']
            for x in intervals),1)

    def test_partial_cursor_plus_future_arrival_preserves_reference_fraction(self):
        p,_=response();head=row('a');head['forecast_cursor']=dict(profile_id='a',segment_index=0,
            remaining_fraction=.5,observed_at_ns=0,evidence='CPU-observed-reference-fixture')
        result=FullHorizonModel(ALIASES,p).forecast({'a':[head],'b':[]},0,1_000_000_000,'median',
            future_arrivals=[dict(row('b',25_000_000),alias='b')],route_future=lambda r,q,c:r['alias'],require_energy=False)
        self.assertEqual(result['finishes']['a']['finish_ns'],75_000_000)
        self.assertEqual(result['finishes']['b']['finish_ns'],250_000_000)
        intervals=[x for x in result['transitions'] if 'a' in x['active_requests']]
        self.assertAlmostEqual(sum((x['until_ns']-x['time_ns'])/1e6/x['service_ms']['a'] for x in intervals),.5)

    def test_response_clock_does_not_supply_power_memory_or_throughput_labels(self):
        p,_=response();rows=active(p)
        self.assertFalse(p.power(rows,ALIASES,'median')['known'])
        self.assertFalse(p.memory(rows,ALIASES,'median')['known'])
        demand=p.demand_summary('a');self.assertEqual(demand['classification'],'unknown')
        self.assertTrue(all(demand[k] is None for k in ('sm','dram','l2')))
        with self.assertRaisesRegex(UnknownEvidence,'POWER'):
            FullHorizonModel(ALIASES,p).forecast({'a':[row('a')]},0,1_000_000_000,'median')
        with self.assertRaisesRegex(UnknownEvidence,'MEMORY'):
            FullHorizonModel(ALIASES,p).forecast({'a':[row('a')]},0,1_000_000_000,'median',require_energy=False,require_memory=True)

    def test_explicit_host_law_receives_catalog_clock_and_gpu_peers(self):
        catalog=provider().provider;doc=copy.deepcopy(catalog.document)
        doc['profiles'][1]['segments'].append(dict(id='host',kind='host',duration_ms={'mean':7}))
        catalog=KernelResourceProvider(doc,'context',allow_cpu_fixture=True)
        p,bundle=response(catalog=catalog,reference_b=20)
        class Host:
            execution_context_sha256='context'
            def __init__(self):self.calls=[]
            def service(self,target,active,aliases,scenario):
                self.calls.append(copy.deepcopy((target,active,scenario)))
                return dict(known=True,context_supported=True,service_ms=40)
        host=Host();p=m.KernelCompetitionResponseProvider(catalog,bundle,scenarios=p.scenarios,
            allow_cpu_fixture=True,allow_unvalidated_profiled_response=True,non_gpu_provider=host)
        result=FullHorizonModel(ALIASES,p).forecast({'a':[row('a')],'b':[row('b')]},
            0,1_000_000_000,'median',require_energy=False)
        self.assertEqual(result['finishes']['b']['finish_ns'],80_000_000)
        self.assertEqual(result['finishes']['a']['finish_ns'],120_000_000)
        self.assertTrue(host.calls)
        target,peers,scenario=host.calls[0]
        self.assertEqual(scenario,'mean')
        self.assertEqual(target['segment_duration_ms'],7)
        self.assertEqual(peers['a']['segment_duration_ms'],100)
        self.assertEqual(set(peers),{'a','b'})

    def test_complete_future_calendar_and_fixed_drain_keep_unfinished_requests(self):
        p,_=response()
        out=FullHorizonModel(ALIASES,p).forecast({'a':[row('a')],'b':[]},0,60_000_000,'median',
            future_arrivals=[dict(row('b',50_000_000),alias='b')],route_future=lambda r,q,c:r['alias'],
            drain_deadline_ns=80_000_000,require_energy=False)
        self.assertEqual(out['incomplete_ids'],['a','b'])
        self.assertEqual(out['finishes'],{})
        self.assertEqual(len(out['cohort']),2)

    def test_prospective_service_does_not_open_future_end_body(self):
        class Poison:
            def __deepcopy__(self,memo):raise AssertionError('future label copied')
            def __float__(self):raise AssertionError('future label opened')
        p,_=response();rows=active(p);rows['a']['future_END']=Poison()
        self.assertTrue(p.service(rows['a'],rows,ALIASES,'median')['known'])
        with self.assertRaisesRegex(ValueError,'no labels or outputs'):
            FullHorizonModel(ALIASES,p).forecast({'a':[dict(row('a'),nsys_end_ns=42)]},
                0,1_000_000_000,'median',require_energy=False)

    def test_real_rank_failure_and_fitted_extreme_clocks_stay_unknown(self):
        catalog=provider().provider
        real=m.training_bundle(catalog,[capture(f.KernelCompetitionFeatureProvider(catalog))],allow_cpu_fixture=True)
        self.assertFalse(real['response_fit']['numerical_qualification'])
        self.assertEqual(real['response_fit']['reason'],'STRUCTURAL_KERNEL_COVARIATE_RANK_FAILURE')
        p=m.KernelCompetitionResponseProvider(catalog,real,allow_cpu_fixture=True,allow_unvalidated_profiled_response=True)
        with self.assertRaisesRegex(UnknownEvidence,'NUMERICALLY_QUALIFIED'):
            p.plan(dict(row('a'),alias='a'),ALIASES,'mean')
        for sigma in (100,1e200):
            p,_=response(sigma=sigma)
            with self.subTest(sigma=sigma),self.assertRaisesRegex(UnknownEvidence,'NONFINITE'):
                p.plan(dict(row('a'),alias='a'),ALIASES,'mean')

    def test_qualified_bundle_cannot_cross_profiler_or_feature_identity(self):
        p,doc=response();catalog=p.catalog
        for mutate in ('profiler','features','context','sigma'):
            altered=copy.deepcopy(doc)
            if mutate=='profiler':altered['profiler_id']='foreign-profiler'
            elif mutate=='features':altered['response_fit']['features']=list(reversed(f.FEATURES))
            elif mutate=='context':altered['execution_context_sha256']='foreign-context'
            else:altered['response_fit']['lognormal_reference_sigma']=0
            altered.pop('bundle_sha256');altered['bundle_sha256']=identity(altered)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                m.KernelCompetitionResponseProvider(catalog,altered,allow_cpu_fixture=True,
                    allow_unvalidated_profiled_response=True)


if __name__=='__main__':unittest.main()
