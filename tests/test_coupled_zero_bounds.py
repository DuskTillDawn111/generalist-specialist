"""Search safety placeholder, with explicitly injected complete CPU oracles.

The production route/load service remains UNKNOWN. Synthetic oracle injection
tests search mathematics only and never supplies a fitted production model.
"""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

from gsdeploy.config import SystemConfig
from gsdeploy.coupled_mode import MODE, PLACEHOLDER_SCHEMA
from gsdeploy.optimizer import UtilityOptimizer
from gsdeploy.milp import JointMilpOptimizer
from gsdeploy.performance import profile_key, resolve_service
from gsdeploy.prediction import predict_scenarios
from gsdeploy.static_study_benchmark import grid_plans
from gsdeploy.system import DeploymentSystem
from tests.test_milp import add_energy
from tests.test_static_dfs_v6 import six_term_config


def placeholder():
    c=replace(six_term_config(),route_grid=2,horizon_seconds=4,milp_capacity_witness=True,
              milp_time_limit_seconds=30,search_max_states=100000)
    c.catalog['performance_mode']=MODE;c.catalog['memory_mode']=MODE
    c.catalog[MODE]=dict(schema=PLACEHOLDER_SCHEMA,status='UNIMPLEMENTED')
    c.catalog['energy_profiles']={}
    # Old standalone would prune every deployment and route, yet no such
    # transfer bound has been proved for the future coupled model.
    for m in c.catalog['models']:
        c.catalog['models'][m]['service_time_ms']['text']=1_000_000.
        c.catalog['memory_table_mib'][m]['1']=100_000.
    return SystemConfig(**c.as_dict())


def optimizer(c, algorithm):
    kwargs=dict(**DeploymentSystem(c).context(),milp_capacity_witness=c.milp_capacity_witness,
                milp_time_limit_seconds=30)
    if algorithm=='milp':return JointMilpOptimizer(c.catalog,c.workload,c.gpu_memory_mib,c.latency_reference_ms,c.weights,**kwargs)
    return UtilityOptimizer(c.catalog,c.workload,c.gpu_memory_mib,c.latency_reference_ms,c.weights,
                            dfs_bound_pruning=algorithm!='plain_dfs',**kwargs)


def contextual_oracle(c):
    # Actual public profile-mode validation and prediction supply this synthetic
    # complete oracle. The solver receives coupled placeholder data, while this
    # test-only oracle has exact whole-plan nonseparable memory and service.
    data=copy.deepcopy(c.catalog);data.pop(MODE)
    data['performance_mode']='profiled';data['memory_mode']='profiled'
    ordinary=replace(c,catalog=data,milp_capacity_witness=False)
    plans=list(grid_plans(replace(ordinary,gpu_memory_mib=1_000_000)))
    for x,r in plans:
        split=len(r['text'])==2
        service=2. if split else (4. if x.get('s') else 8.)
        key=profile_key(data,x,r,ordinary.workload,ordinary.horizon_seconds,
                        ordinary.random_seed,ordinary.arrival_process)
        data.setdefault('colocation_profiles',{})[key]=dict(kind='synthetic_fixture',scope='whole_gpu_colocation',
            coverage_valid=True,service_time_ms={m:{'text':service} for m in x},safe_memory_mib=1. if split else 2.)
    # Whole-board energy depends jointly on route and residents, not a sum of
    # previously profiled standalone path/state terms.
    add_energy(ordinary,plans,lambda x,r,_: 1. if len(r['text'])==2 else 2.+len(x))
    def predict(data,x,r,rates,budget,refs,weights,**context):
        ordinary.validate_plan(x,r)  # actual complete-context rho/memory gates
        return predict_scenarios(ordinary.catalog,x,r,rates,budget,refs,weights,**context)
    return ordinary,plans,predict


class CoupledZeroBoundTests(unittest.TestCase):
    def test_config_requires_explicit_placeholder_and_complete_service_stays_unknown(self):
        c=placeholder()
        for spec in [None,{},dict(schema=PLACEHOLDER_SCHEMA,status='VALIDATED')]:
            data=copy.deepcopy(c.catalog);data[MODE]=spec
            with self.assertRaisesRegex(ValueError,'NOT_IMPLEMENTED'):replace(c,catalog=data)
        x={'g':1,'s':1};r={'text':{'g':.5,'s':.5}}
        checked=c.validate_plan(x,r)
        self.assertFalse(checked['constraints_valid']);self.assertFalse(checked['capacity_verified'])
        self.assertFalse(checked['memory_verified']);self.assertIsNone(checked['safe_memory_mib'])
        self.assertIsNone(checked['rho'])
        result=DeploymentSystem(c).predict(x,r)
        self.assertTrue(result['performance_unknown']);self.assertIsNone(result['U'])
        self.assertEqual(result['reason'],'ROUTE_LOAD_MODEL_NOT_IMPLEMENTED')
        with self.assertRaisesRegex(ValueError,'unknown colocation capacity'):
            DeploymentSystem(c).run({'winner':dict(deployment=x,routing=r)},'never-read-samples')
        bad=copy.deepcopy(c.catalog);bad[MODE]={'schema':'unexpected'}
        self.assertEqual(resolve_service(bad,x,r,c.workload)[1]['reason'],'UNSUPPORTED_ROUTE_LOAD_SCHEMA')

    def test_all_resource_bound_fallbacks_are_zero_and_witness_is_disabled(self):
        c=placeholder()
        for algorithm in ['milp','dfs','plain_dfs']:
            solver=optimizer(c,algorithm)
            self.assertFalse(solver.separable_service);self.assertTrue(solver.profiled_memory)
            self.assertFalse(solver.use_witness)
            if algorithm=='milp':
                self.assertFalse(solver.diagnostic_mode)
                self.assertEqual(solver.state_memory('g',1),0)
                self.assertEqual(solver.family_service('g','text',1),0)
                self.assertEqual(solver.pressure_service('g','text',1),0)
            else:
                self.assertEqual(solver.family_service('g',1,'text'),0)
                self.assertEqual(solver.pressure_service('g',1,'text'),0)
                self.assertEqual(solver.service_floor({'g':1},'text',{'g':1.}),0)
                self.assertEqual(solver.upper({'g':1},{},100_000),0)

    def test_unknown_complete_candidates_are_not_pruned_as_infeasible_or_certified(self):
        c=placeholder()
        for algorithm in ['milp','dfs','plain_dfs']:
            result=optimizer(c,algorithm).solve(K=2)
            self.assertTrue(result['search_completed'],result['status'])
            self.assertFalse(result['optimality_certified'])
            self.assertIsNone(result['winner'])
            self.assertGreater(result['metrics']['unknown_candidates'],0)
            self.assertEqual(result['metrics'].get('memory_pruned',result['metrics'].get('column_memory_pruned')),0)
            self.assertEqual(result['metrics'].get('infeasible_candidates',result['metrics'].get('infeasible_cuts')),0)
            self.assertEqual(result['domain']['coupled_bounds']['memory_lower_mib'],0)

    def test_contextual_complete_oracle_bounds_cover_cost_and_three_solvers_agree(self):
        c=placeholder();oracle,plans,predict=contextual_oracle(c)
        exact=[]
        milp=optimizer(c,'milp');dfs=optimizer(c,'dfs')
        for x,r in plans:
            p=predict(None,x,r,c.workload,c.gpu_memory_mib,c.latency_reference_ms,c.weights,
                      **DeploymentSystem(c).context())
            self.assertIsNotNone(p['U']);exact.append(p['U'])
            bound=sum(milp.floor(t,{m:x[m] for m in row},row) for t,row in r.items())
            self.assertLessEqual(bound,-p['U']+1e-12)
            self.assertGreaterEqual(dfs.upper(x,r,1_000_000),p['U']-1e-12)
        results=[]
        with patch('gsdeploy.milp.predict_scenarios',side_effect=predict),patch('gsdeploy.optimizer.predict_scenarios',side_effect=predict):
            for algorithm in ['milp','dfs','plain_dfs']:
                result=optimizer(c,algorithm).solve(K=2);results.append(result)
                self.assertTrue(result['optimality_certified'],result['status'])
                self.assertAlmostEqual(result['winner']['score'],max(exact),places=10)
                self.assertEqual(len(result['winner']['routing']['text']),2)
                self.assertEqual(result['metrics'].get('route_feedback_cuts',0),0)
        self.assertEqual(results[0]['winner']['deployment'],{'g':1,'s':1})

    def test_unknown_identity_keeps_certificate_open_beside_eligible_incumbent(self):
        c=placeholder();oracle,plans,predict=contextual_oracle(c)
        def partial(data,x,r,*args,**kwargs):
            if len(r['text'])==2:
                return dict(U=None,diagnostic_partial_U=None,eligible=False,constraints_eligible=False,
                    performance_unknown=True,reason='CPU_ORACLE_MISSING_CONTEXT',scenario_U_low=None)
            return predict(data,x,r,*args,**kwargs)
        with patch('gsdeploy.milp.predict_scenarios',side_effect=partial),patch('gsdeploy.optimizer.predict_scenarios',side_effect=partial):
            for algorithm in ['milp','dfs','plain_dfs']:
                result=optimizer(c,algorithm).solve(K=2)
                self.assertIsNotNone(result['winner'])
                self.assertFalse(result['optimality_certified'])
                self.assertGreater(result['metrics']['unknown_candidates'],0)


if __name__=='__main__':unittest.main()
