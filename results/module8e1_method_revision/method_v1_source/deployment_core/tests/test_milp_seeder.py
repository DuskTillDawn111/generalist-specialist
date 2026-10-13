"""Joint-seed correctness, oracle parity, numerical rejection and no-enumeration tests."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import milp

from deployment_core import MilpDeploymentSeeder, ModelProfile, Hardware, Workload, SeedSolverError
from deployment_core.milp_seeder import validate_seed
from deployment_core.models import load_model_catalog, load_hardware
from deployment_core.seed import main
from deployment_core.experiments.validate_seeder import real_validation, synthetic_validation, forbid_enumeration


def model(name,memory,times):
    return ModelProfile(name,frozenset(times),memory,times)


class SeederTests(unittest.TestCase):
    def setUp(self):
        self.seeder=MilpDeploymentSeeder()
        self.catalog={'general':model('general',10,{'text':100,'asr':200}),
                      'text':model('text',2,{'text':10}),'asr':model('asr',3,{'asr':50})}
        self.hardware=Hardware(15)

    def solve(self,rates,**kwargs):
        return self.seeder.solve(self.catalog,Workload(rates),self.hardware,**kwargs)

    def test_joint_single_milp_minimum_memory_and_feedback(self):
        with forbid_enumeration(),patch('deployment_core.milp_seeder.milp',wraps=milp) as solver:
            result=self.solve({'text':10,'asr':2})
        self.assertEqual(solver.call_count,1)
        self.assertEqual(result.deployment,{'asr':1,'general':0,'text':1})
        self.assertEqual(result.objective_value,5)
        self.assertEqual(result.routing_rps['text','text'],10)
        self.assertAlmostEqual(result.routing_rps['asr','asr'],2)
        self.assertAlmostEqual(result.utilization['asr'],.1)
        self.assertAlmostEqual(result.capacity_slack['asr'],.7)
        self.assertAlmostEqual(result.service_demand_by_task_model['asr','asr'],.1)
        self.assertEqual(validate_seed(result,self.catalog,Workload({'text':10,'asr':2}),self.hardware,.8),[])

    def test_split_routing_needed_for_capacity(self):
        catalog={m:model(m,1,{'text':100}) for m in ('a','b')}
        r=self.seeder.solve(catalog,Workload({'text':12}),Hardware(2))
        self.assertEqual(r.deployment,{'a':1,'b':1})
        self.assertAlmostEqual(sum(r.routing_rps.values()),12)
        self.assertTrue(all(4-1e-9<=r.routing_rps['text',m]<=8+1e-9 for m in catalog))

    def test_infeasible_coverage_memory_capacity(self):
        for catalog,workload,hardware in [
            (self.catalog,Workload({'visual_qa':1}),self.hardware),
            (self.catalog,Workload({'text':1}),Hardware(1)),
            (self.catalog,Workload({'asr':100}),self.hardware)]:
            with self.subTest(workload=workload):
                r=self.seeder.solve(catalog,workload,hardware)
                self.assertFalse(r.feasible)
                self.assertEqual(r.solver_status,'infeasible')
                self.assertIsNone(r.deployment); self.assertIsNone(r.routing_rps)
                self.assertIsNone(r.objective_value)

    def test_zero_workload_empty_count_vector_allowed(self):
        r=self.seeder.solve(self.catalog,Workload({}),Hardware(0))
        self.assertTrue(r.feasible)
        self.assertEqual(set(r.deployment.values()),{0})
        self.assertEqual(r.objective_value,0)
        self.assertEqual(set(r.routing_rps.values()),{0.})

    def test_configurable_rho_and_binary_instance_limit(self):
        catalog={'a':model('a',1,{'text':100})}
        self.assertTrue(self.seeder.solve(catalog,Workload({'text':8}),Hardware(2),.8).feasible)
        self.assertFalse(self.seeder.solve(catalog,Workload({'text':8}),Hardware(2),.7).feasible)
        self.assertFalse(self.seeder.solve(catalog,Workload({'text':12}),Hardware(100),.8).feasible)

    def test_primary_memory_objective_not_latency_or_model_count(self):
        # Two specialists use less total memory than one faster generalist.
        catalog={'general':model('general',5.00001,{'text':1,'asr':1}),
                 'text':model('text',2,{'text':10}), 'asr':model('asr',3,{'asr':10})}
        r=self.seeder.solve(catalog,Workload({'text':1,'asr':1}),Hardware(10))
        self.assertEqual(r.objective_value,5)
        self.assertEqual(r.deployment['general'],0)

    def test_small_workload_conservation_and_exact_zero_pairs(self):
        r=self.solve({'text':1e-8})
        self.assertAlmostEqual(sum(r.routing_rps.values())/1e-8,1)
        self.assertTrue(all(v==0 for (k,m),v in r.routing_rps.items() if k!='text' or m!='text'))

    def test_invalid_rho_catalog_rejected(self):
        for rho in (0,-1,1.1,float('nan'),True):
            with self.assertRaises(ValueError): self.solve({'text':1},rho_max=rho)
        for catalog in ({},{'wrong':model('right',1,{'text':1})}):
            with self.assertRaises(ValueError): self.seeder.solve(catalog,Workload({'text':1}),self.hardware)

    def test_solver_limit_or_failure_is_not_infeasibility(self):
        for status in (1,3,4):
            with patch('deployment_core.milp_seeder.milp',return_value=SimpleNamespace(status=status,success=False,message='test failure')):
                with self.assertRaises(SeedSolverError): self.solve({'text':1})

    def test_invalid_success_witness_not_returned_or_repaired(self):
        catalog={'a':model('a',1,{'text':100})}
        for values,fun in [(None,1),([np.nan,1],1),([.5,1],.5),([0,1],0),([1,0],1),([1,9],1),([1,1],2)]:
            result=SimpleNamespace(status=0,success=True,message='invalid fixture',x=values,fun=fun)
            with self.subTest(values=values),patch('deployment_core.milp_seeder.milp',return_value=result):
                with self.assertRaises(SeedSolverError):
                    self.seeder.solve(catalog,Workload({'text':1}),Hardware(2))

    def test_feedback_tampering_is_detected(self):
        result=self.solve({'text':1})
        result.service_demand_by_task_model['text','text']+=.1
        result.capacity_slack['text']+=.1
        errors=validate_seed(result,self.catalog,Workload({'text':1}),self.hardware,.8)
        self.assertTrue(any('Demand' in e for e in errors))
        self.assertTrue(any('Slack' in e for e in errors))

    def test_formal_import_does_not_load_exhaustive_module(self):
        code='import sys; import deployment_core.seed; assert "deployment_core.candidate_generator" not in sys.modules'
        subprocess.run([sys.executable,'-c',code],check=True)

    def test_cli_feasible_infeasible_and_saved_feedback(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()) as text:
            output=Path(d)/'seed.json'
            with forbid_enumeration():
                code=main(['--workload','text=10,image=5,ocr=3,asr=2,visual_qa=1,lid=1','--output',str(output)])
            self.assertEqual(code,0)
            data=json.loads(output.read_text())
            self.assertTrue(data['feasible'])
            self.assertEqual(data['objective_value'],20386)
            self.assertEqual(len(data['routing_rps']),30)
            self.assertEqual(len(data['service_demand_by_task_model']),30)
            self.assertIn('not a performance-optimal deployment',text.getvalue())
            self.assertEqual(main(['--workload','visual_qa=100','--output',str(output)]),0)
            self.assertFalse(json.loads(output.read_text())['feasible'])


class OracleAndScalabilityTests(unittest.TestCase):
    def test_all_19_stage1_5_scenarios_and_objectives(self):
        result=real_validation(load_model_catalog(),load_hardware())
        self.assertEqual(result['num_scenarios'],19)
        self.assertTrue(result['passed'],[(r['name'],r['failures']) for r in result['scenarios']])
        self.assertTrue(any(not r['seed']['feasible'] for r in result['scenarios']))

    def test_20_and_30_models_without_deployment_enumeration(self):
        result=synthetic_validation()
        self.assertTrue(result['passed'])
        self.assertEqual([r['num_models'] for r in result['tests']],[20,30])
        for row in result['tests']:
            self.assertFalse(row['enumeration_called'])
            self.assertEqual(row['deployment_objects_created'],0)
            self.assertLessEqual(row['num_variables'],7*row['num_models'])
            self.assertGreater(row['solver_time_ms'],0)


if __name__=='__main__':
    unittest.main()
