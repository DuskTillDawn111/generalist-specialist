"""Synthetic portable tests of observers and finite accounting, not GPU evidence."""
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

from gsdeploy.search_session import SearchSession
from tests.test_calibrated_replay_v2 import fitted_replay_fixture
from experiments.joint_scalability import ROOT
from experiments.optimization_difficulty_diagnosis import (
    observe, floor_parts, run_joint, full_rank, space)


class OptimizationDifficultyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=replace(fitted_replay_fixture(),horizon_seconds=30.,
                           milp_capacity_witness=False,milp_time_limit_seconds=20.)

    def test_observed_masters_preserve_winner_and_restore_on_exception(self):
        with SearchSession(ROOT,self.config.as_dict()) as session:
            module=session.module('milp')
            original=module.solve_master
            before=session.optimizer('milp').solve(K=2)
            opt=session.optimizer('milp')
            with observe(module,opt) as stats:
                after=opt.solve(K=2)
            self.assertIs(module.solve_master,original)
            self.assertEqual(before['winner'],after['winner'])
            self.assertEqual(before['metrics'],after['metrics'])
            self.assertEqual(len(stats['predictions']),after['metrics']['full_prediction_calls'])
            self.assertEqual(len(stats['rounds']),after['metrics']['milp_calls'])
            self.assertTrue(after['optimality_certified'])
            self.assertTrue(stats['rounds'][-1]['gap_condition'])
            with self.assertRaises(RuntimeError):
                with observe(module,session.optimizer('milp'),'without_fifo'):
                    raise RuntimeError('test observer cleanup')
            self.assertIs(module.solve_master,original)

    def test_floor_parts_reconstruct_and_lower_bound_scored_plan(self):
        with SearchSession(ROOT,self.config.as_dict()) as session:
            opt=session.optimizer('milp')
            opt.start=__import__('time').monotonic()
            d={'generalist':1,'specialist':1}
            routing={t:{'generalist':q*.37,'specialist':q*.63} for t,q in self.config.workload.items()}
            total=0.
            for task,row in routing.items():
                floor=opt.floor(task,d,row)
                parts=floor_parts(opt,task,d,row,floor)
                self.assertAlmostEqual(sum(parts[k] for k in ('quality','route_energy','no_wait_service','isolated_fifo_increment')),floor)
                self.assertGreaterEqual(parts['isolated_fifo_increment'],0.)
                total+=floor
            total+=sum(opt.weights['M']*opt.state_memory(m,n)/opt.budget
                       +opt.weights['E']*opt.surrogate.state_energy(n) for m,n in d.items())
            p=session.predict(d,routing)
            self.assertIsNotNone(p['U'])
            self.assertLessEqual(total,-p['U']+1e-9)

    def test_relaxation_ablations_preserve_full_problem_and_certificate(self):
        states={m:[1] for m in self.config.catalog['models']}
        before=run_joint(self.config,states)
        for name in ('without_fifo','without_delay','without_quality','without_route_energy','zero_route_floor'):
            after=run_joint(self.config,states,name)
            self.assertTrue(after['result']['optimality_certified'],name)
            self.assertAlmostEqual(after['result']['winner']['score'],before['result']['winner']['score'])
            self.assertEqual(after['result']['winner']['prediction'],before['result']['winner']['prediction'])
            self.assertAlmostEqual(sum(after['exclusive_seconds'].values()),after['total_seconds'],places=6)

    def test_exhaustive_small_domain_retains_idle_residents_and_counts(self):
        c=replace(self.config,route_grid=1)
        states={m:[1] for m in c.catalog['models']}
        with tempfile.TemporaryDirectory() as tmp:
            rank=full_rank(c,states,Path(tmp)/'plans')
            self.assertEqual(rank['full_prediction_calls'],6)
            self.assertEqual(rank['valid_plans'],6)
            self.assertEqual(len(rank['ranked_deployments']),3)
            self.assertTrue(any(len(r['deployment'])==2 and all(len(row)==1 and 'generalist' in row
                for row in r['routing'].values()) for r in rank['plans']))
            self.assertEqual(len(list((Path(tmp)/'plans').glob('*.json'))),6)
        audit=space(c,states)
        self.assertEqual(audit['raw_deployment_combinations'],4)
        self.assertEqual(audit['exact_memory_survivors'],4)
        self.assertEqual(audit['coverage_survivors'],3)
        self.assertEqual(audit['per_task_quality_evidence_survivors'],3)


if __name__=='__main__': unittest.main()
