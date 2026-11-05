"""Reference-v2 arithmetic, evidence boundaries and negative-Q search safety."""
import copy
from dataclasses import replace
import itertools
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from gsdeploy import normalization as norm
from gsdeploy.config import SystemConfig
from gsdeploy.prediction import components, predict
from gsdeploy.quality import task_quality
from gsdeploy.system import DeploymentSystem
from gsdeploy.energy import energy_key
from gsdeploy.forecasting.metrics import score_forecast

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return SystemConfig.load(ROOT/'configs/paper_fifo_reference_v2_cpu_fixture.json')


class ReferenceNormalizationTests(unittest.TestCase):
    def test_six_formulas_fixed_task_weights_and_unclipped_quality(self):
        data = {norm.KEY: dict(schema=norm.SCHEMA, status='synthetic_fixture', tasks={
            'text': dict(best_loss=.2, worst_loss=.6, deadline_ms=1000),
            'ocr': dict(best_loss=.1, worst_loss=.5, deadline_ms=2000)}, episode_reference_joules=100)}
        q = .25*norm.quality_loss(data,'text',.1)+.75*norm.quality_loss(data,'ocr',1.3)
        self.assertAlmostEqual(q, 2.1875)
        t = norm.deadline_loss([('text',1000,True),('text',500,False),('ocr',2001,True)],
                               norm.deadlines(data,{'text':1,'ocr':1}))
        result = components({'text':dict(mean_ms=500,p95_ms=1000),
                             'ocr':dict(mean_ms=4000,p95_ms=8000)},
                            {'text':1,'ocr':3},{'text':1,'ocr':1}, t, .5,
                            {k:1/6 for k in 'QTLPME'}, E=2., Q=q, data=data)
        self.assertAlmostEqual(result['T_norm'], 2/3)
        self.assertAlmostEqual(result['L_norm'],1.625)
        self.assertAlmostEqual(result['P_norm'],3.25)
        self.assertAlmostEqual(result['U'],-(q+2/3+1.625+3.25+.5+2)/6)
        self.assertEqual(result['utility_schema'],norm.SCHEMA)
        self.assertLess(norm.quality_loss(data,'text',.1),0)

    def test_deadline_uses_response_time_and_retains_failed_denominator(self):
        # A request arriving at 950ms and finishing at 1050ms is good for D=100ms.
        self.assertAlmostEqual(norm.deadline_loss([
            ('text',100,True),('text',101,True),('text',10,False),('text',None,False)],
            {'text':100}),.75)
        self.assertIsNone(norm.deadline_loss([('text',1,True)], {'text':None}))

    def test_contract_rejects_changed_units_and_degenerate_ranges(self):
        c=fixture()
        for mutate in [lambda s:s['tasks']['text'].update(worst_loss=.2),
                       lambda s:s.update(horizon_seconds=2),
                       lambda s:s.update(memory_budget_mib=90),
                       lambda s:s['tasks']['text'].update(deadline_ms=9),
                       lambda s:s.update(episode_reference_joules=0),
                       lambda s:s.update(reference_receipts=[])]:
            d=c.as_dict(); mutate(d['catalog'][norm.KEY])
            with self.assertRaises(ValueError): SystemConfig(**d)
        with self.assertRaises(ValueError): replace(c,utility_schema='six_term_quality_v1')

    def test_pending_has_no_winner_and_does_not_fall_back_to_one_second(self):
        c=fixture(); d=c.as_dict(); d['catalog'][norm.KEY]['status']='pending'
        d['catalog'][norm.KEY]['tasks']['text']['deadline_ms']=None
        d['catalog'][norm.KEY]['episode_reference_joules']=None
        d['latency_reference_ms']['text']=None
        c=SystemConfig(**d)
        p=DeploymentSystem(c).plan()
        self.assertIsNone(p['winner']); self.assertFalse(p['optimality_certified'])
        result=DeploymentSystem(c).predict({'g':1},{'text':{'g':2}})
        self.assertIsNone(result['U']); self.assertIsNone(result['L_norm'])
        self.assertIsNone(result['T_norm']); self.assertIsNone(result['E_norm'])
        self.assertEqual(result['reference_status'],'pending')

    def test_full_episode_integration_includes_startup_and_drain(self):
        w=dict(start_time_ns=1_000_000_000,end_time_ns=2_000_000_000,drain_end_time_ns=3_000_000_000)
        e=dict(kind='measured',scope='whole_gpu_board',startup_begin_time_ns=0,
               samples=[dict(time_ns=i*1_000_000_000,watts=10) for i in range(4)])
        complete=norm.measured_episode(e,w)
        self.assertEqual(complete['episode_joules'],30)
        self.assertAlmostEqual(norm.episode_cost(fixture().catalog,complete),.6)
        incomplete=norm.measured_episode({**e,'samples':e['samples'][:-1]},w)
        self.assertIsNone(norm.episode_cost(fixture().catalog,incomplete))
        self.assertIsNone(norm.episode_cost(fixture().catalog,dict(kind='measured',scope='whole_gpu_board',
                          coverage_valid=True,run_joules=10,startup_joules=10)))

    def test_memory_is_raw_owned_sample_peak_without_safety_factor(self):
        memory=dict(measured=True,episode_coverage_valid=True,safe_memory_mib=999,
                    before=dict(owned_used_mib=10),after=dict(owned_used_mib=20),
                    samples=[dict(owned_used_mib=30)])
        self.assertEqual(norm.raw_peak(memory),30)
        self.assertIsNone(norm.raw_peak({**memory,'episode_coverage_valid':False}))

    def test_scalar_fifo_and_quality_use_new_contract(self):
        c=fixture(); d=copy.deepcopy(c.catalog)
        d.update(performance_mode='independent',memory_mode='standalone_proxy')
        x={'g':1}; r={'text':{'g':2}}
        key=energy_key(x,r,c.workload,c.horizon_seconds,c.random_seed,c.arrival_process,catalog=d)
        d['energy_profiles']={key:dict(kind='synthetic_fixture',scope='whole_gpu_board',
                            episode_coverage_valid=True,episode_joules=100)}
        p=predict(d,x,r,c.workload,c.gpu_memory_mib,c.latency_reference_ms,c.weights,
                  horizon=c.horizon_seconds,seed=c.random_seed,drain_seconds=c.drain_seconds)
        self.assertTrue(p['eligible']); self.assertEqual(p['T_norm'],0)
        self.assertAlmostEqual(p['L_norm'],.2); self.assertAlmostEqual(p['P_norm'],.2)
        self.assertAlmostEqual(p['M_norm'],.01); self.assertAlmostEqual(p['E_norm'],2)
        loss,ev=task_quality(d,'text',x,r['text'],((0,'text'),))
        self.assertAlmostEqual(loss,1.); self.assertAlmostEqual(ev['native_loss'],.8)

    def test_paper_fifo_energy_and_peak_require_complete_startup_evidence(self):
        c=fixture(); s=DeploymentSystem(c)
        p=s.predict({'g':1},{'text':{'g':2}})
        self.assertTrue(p['eligible']); self.assertFalse(p['native_eligible'])
        self.assertAlmostEqual(p['M_norm'],p['safe_memory_mib']/c.catalog['safe_factor']/c.gpu_memory_mib)
        e=p['episode_energy_evidence']
        self.assertAlmostEqual(p['E_norm'],e['episode_joules']/50.)
        d=c.as_dict()
        for entry in d['catalog']['paper_profiled_fifo_v1']['deployments']:
            entry.pop('startup_peak_memory_mib'); entry.pop('startup_energy_scope')
        missing=DeploymentSystem(SystemConfig(**d)).predict({'g':1},{'text':{'g':2}})
        self.assertIsNone(missing['U']); self.assertIsNone(missing['M_norm']); self.assertIsNone(missing['E_norm'])

    def test_milp_and_dfs_match_exhaustive_even_when_utility_positive(self):
        d=fixture().as_dict(); d['latency_reference_ms']['text']=10000
        spec=d['catalog'][norm.KEY]; spec['tasks']['text']['deadline_ms']=10000
        spec['episode_reference_joules']=1e6
        d['milp_time_limit_seconds']=10
        c=SystemConfig(**d); sys=DeploymentSystem(c)
        values=[]
        for x in ({'g':1},{'s':1},{'g':1,'s':1}):
            for share in (0,.5,1):
                r={'text':{m:2*v for m,v in [('g',share),('s',1-share)] if v and m in x}}
                if sum(r['text'].values())!=2: continue
                p=sys.predict(x,r)
                if p['U'] is not None: values.append(p['U'])
        expected=max(values); self.assertGreater(expected,0)
        for solver in ('dfs','milp'):
            p=DeploymentSystem(replace(c,solver=solver)).plan()
            self.assertTrue(p['optimality_certified'],p['status'])
            self.assertAlmostEqual(p['winner']['score'],expected)
        from gsdeploy.continuous.predictor_adapter import global_cost_floor
        upper=-global_cost_floor(SimpleNamespace(cost_lower_bound=0),SimpleNamespace(config=c))
        self.assertGreaterEqual(upper,expected)

    def test_paper_energy_integrates_the_post_window_drain(self):
        c=fixture()
        p=predict(c.catalog,{'g':1},{'text':{'g':2}},c.workload,c.gpu_memory_mib,
            c.latency_reference_ms,c.weights,horizon=1.,trace=((.999,'text'),))
        f=p['forecast']
        self.assertGreater(f['drain_finish_ns'],1_000_000_000)
        self.assertGreater(f['through_drain_board_energy_joules'],f['board_energy_joules'])
        self.assertAlmostEqual(p['E_norm'],(2+f['through_drain_board_energy_joules'])/50)

    def test_measured_evaluator_reconstructs_new_components_from_raw_records(self):
        from unittest.mock import patch
        from gsdeploy.evaluation import evaluate_run
        from gsdeploy.metrics import annotate_records
        from gsdeploy.trace import make_trace,trace_digest
        c=fixture(); d=c.as_dict()
        d['catalog'].update(performance_mode='independent',memory_mode='standalone_proxy')
        d['catalog'][norm.KEY]['tasks']['text']['deadline_ms']=600
        d['latency_reference_ms']['text']=600
        c=SystemConfig(**d)
        trace=make_trace(c.workload,1.,42,c.arrival_process)
        records=[]
        for i,(offset,task) in enumerate(trace):
            at=1_000_000_000+round(offset*1e9)
            records.append(dict(request_id=f'measurement:{i}',task=task,model='g#0',
                status='completed',scheduled_time_ns=at,actual_submit_time_ns=at,
                completion_time_ns=at+600_000_000,worker_response=dict(success=True)))
        annotate_records(records)
        raw=dict(backend='mock',status='completed',replica_to_family={'g#0':'g'},
            plan=dict(deployment={'g':1},routing={'text':{'g':2}}),
            request_records=records,transport_config=dict(random_seed=42,arrival_process=c.arrival_process),
            measurement_window=dict(duration_seconds=1.,start_time_ns=1_000_000_000,
                end_time_ns=2_000_000_000,drain_end_time_ns=2_100_000_000,
                offered_requests=2,trace_sha256=trace_digest(trace)),
            memory_evidence=dict(measured=True,resource_valid=True,safe_memory_mib=24.,
                scope='CPU fabricated telemetry fixture',episode_coverage_valid=True,
                before=dict(owned_used_mib=10),after=dict(owned_used_mib=20),samples=[]),
            energy_evidence=dict(kind='measured',scope='whole_gpu_board',startup_begin_time_ns=0,
                samples=[dict(time_ns=t,watts=10.) for t in [0,1_000_000_000,2_100_000_000]],
                profile_key=energy_key({'g':1},{'text':{'g':2}},c.workload,1.,42,c.arrival_process,catalog=c.catalog)))
        with patch('gsdeploy.evaluation.reconstruct',return_value={}), patch('gsdeploy.evaluation.classify',
                return_value={'stability_state_v2':'steady'}):
            result=evaluate_run(c,raw)['utility']
        self.assertEqual(result['T_norm'],0)
        self.assertAlmostEqual(result['L_norm'],1.)
        self.assertAlmostEqual(result['M_norm'],.2)
        self.assertAlmostEqual(result['E_norm'],21/50)
        self.assertFalse(result['eligible'])
        self.assertEqual(result['utility_schema'],norm.SCHEMA)

    def test_frozen_episode_rejects_another_warmup_protocol(self):
        protocol=dict(warmup_seconds=30,priming_policy='largest_three_per_task',
                      drain_seconds=300,arrival_process='poisson')
        data={norm.KEY:dict(status='frozen',episode_reference_joules=100.,episode_protocol=protocol)}
        e=dict(kind='predicted_episode',scope='whole_gpu_board',episode_coverage_valid=True,
               episode_joules=200.,episode_protocol=protocol)
        self.assertEqual(norm.episode_cost(data,e),2.)
        self.assertIsNone(norm.episode_cost(data,{**e,'episode_protocol':{**protocol,'warmup_seconds':0}}))
        self.assertIsNone(norm.episode_cost(data,{k:v for k,v in e.items() if k!='episode_protocol'}))

    def test_continuous_objective_identity_binds_quality_anchors(self):
        from gsdeploy.continuous.contracts import ContinuousProblem
        c=fixture(); before=ContinuousProblem(c).objective_id
        d=c.as_dict();d['catalog'][norm.KEY]['tasks']['text']['best_loss']=.1
        after=ContinuousProblem(SystemConfig(**d)).objective_id
        self.assertNotEqual(before,after)

    def test_online_forecast_uses_deadlines_raw_memory_and_full_episode(self):
        c=fixture(); spec=c.catalog[norm.KEY]
        row=dict(request_id='0',task='text',alias='g#0',pool_id='a',arrival_ns=999_000_000,finish_ns=1_001_000_000)
        forecast=dict(cohort={'0':row},finishes={'0':row},board_energy_joules=5.,
                      through_drain_board_energy_joules=8.,peak_memory_mib=20.)
        kwargs=dict(phase_start_ns=0,now_ns=0,deadline_ns=1_000_000_000,memory_budget_mib=100.,
            energy_reference_watts=300.,observed_peak_memory_mib=10.,observed_board_energy_joules=0.,
            memory_safe_factor=2.,normalization_reference=spec,utility_schema=norm.SCHEMA)
        result=score_forecast(forecast,[],{'g#0':dict(family='g',supported_tasks=['text'])},
            c.workload,c.latency_reference_ms,{'g':{'text':.8}},observed_episode_energy_joules=2.,**kwargs)
        self.assertEqual(result['terms']['T'],0)
        self.assertEqual(result['terms']['M'],.2); self.assertEqual(result['safe_memory_mib'],40)
        self.assertEqual(result['terms']['E'],.2)
        self.assertIsNotNone(result['full_U'])
        result=score_forecast(forecast,[],{'g#0':dict(family='g',supported_tasks=['text'])},
            c.workload,c.latency_reference_ms,{'g':{'text':.8}},**kwargs)
        self.assertIsNone(result['full_U']); self.assertIsNone(result['terms']['E'])


if __name__=='__main__': unittest.main()
