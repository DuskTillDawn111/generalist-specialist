"""Small complete domains and counterexamples for candidate bounds/scheduling."""

import copy
from dataclasses import replace
import itertools
import unittest
from unittest.mock import patch

from gsdeploy.continuous import Budget, ImmutableContinuousProblem, SessionPredictorOracle
from gsdeploy.continuous.adaptive_candidate import (AdaptiveCandidateRefinementOptimizer,
    budget_slice, screen_points)
from gsdeploy.continuous.candidate_bounds import ContinuousDeploymentBounds
from gsdeploy.continuous.contracts import BudgetExhausted
from gsdeploy.continuous.refinement import FixedDeploymentRoutingEvaluator, project_shares
from tests.test_calibrated_replay_v2 import fitted_replay_fixture
from tests.test_calibrated_search_bounds_v6 import search_fixture
from tests.test_milp import grid_plans


class AdaptiveCandidateTests(unittest.TestCase):
    def test_global_continuous_relaxation_covers_independent_rational_routes_and_stress(self):
        c=replace(fitted_replay_fixture(quality_mode='per_sample_replay'),horizon_seconds=20.)
        p=ImmutableContinuousProblem(c,replica_options={m:[1,2] for m in c.catalog['models']})
        with SessionPredictorOracle(p) as oracle:
            bounds=ContinuousDeploymentBounds(p,oracle)
            self.assertTrue(bounds.available)
            checked=0
            for x,r in grid_plans(c,3):
                proof=bounds.evaluate(x)
                prediction=oracle.session.predict(x,r)
                if prediction['U'] is not None:
                    self.assertFalse(proof['safe_pruned'])
                    self.assertGreaterEqual(proof['upper_bound']+1e-10,prediction['U'])
                    if prediction.get('scenario_U_low') is not None:
                        self.assertGreaterEqual(proof['upper_bound']+1e-10,prediction['scenario_U_low'])
                    self.assertAlmostEqual(proof['memory_mib'],prediction['safe_memory_mib'])
                    checked+=1
            self.assertGreater(checked,30)

    def test_bound_covers_non_grid_routes_and_zero_flow_residency(self):
        c=replace(search_fixture(),horizon_seconds=3.)
        p=ImmutableContinuousProblem(c)
        with SessionPredictorOracle(p) as oracle:
            bounds=ContinuousDeploymentBounds(p,oracle)
            x={'generalist':1,'specialist':2};proof=bounds.evaluate(x)
            e=FixedDeploymentRoutingEvaluator(p,oracle,x,Budget(20,100,100))
            for a,b in itertools.product((0.,1.,.3141592653589793),repeat=2):
                shares=e.domain.decode([a,b]);r=e.domain.to_qps(shares)
                pred=oracle.session.predict(x,r)
                if pred['U'] is not None:
                    self.assertGreaterEqual(proof['upper_bound']+1e-10,pred['U'])
                self.assertAlmostEqual(proof['memory_mib'],pred['safe_memory_mib'])
            self.assertFalse(proof['fixed_route_feedback_reused'])

    def test_unknown_gate_and_custom_oracle_do_not_become_bounds(self):
        c=replace(fitted_replay_fixture(),horizon_seconds=20.)
        changed=replace(c,workload={t:q*2 for t,q in c.workload.items()})
        p=ImmutableContinuousProblem(changed)
        with self.assertRaisesRegex(ValueError,'WORKLOAD_CHANGED'):
            SessionPredictorOracle(p)
        p=ImmutableContinuousProblem(c)
        with SessionPredictorOracle(p) as oracle:
            with patch('gsdeploy.continuous.candidate_bounds.surrogate_bounds',return_value=None):
                bounds=ContinuousDeploymentBounds(p,oracle)
            self.assertFalse(bounds.available)
            r=bounds.evaluate({'generalist':1})
            self.assertIsNone(r['upper_bound']);self.assertFalse(r['safe_pruned'])
        self.assertFalse(ContinuousDeploymentBounds(p,object()).available)

    def test_safe_coverage_states_memory_and_incumbent_decisions(self):
        c=replace(search_fixture(),gpu_memory_mib=.01)
        p=ImmutableContinuousProblem(c,replica_options={m:[1] for m in c.catalog['models']})
        with SessionPredictorOracle(p) as oracle:
            bounds=ContinuousDeploymentBounds(p,oracle)
            self.assertEqual(bounds.evaluate({})['reason'],'ILLEGAL_REPLICA_STATE')
            self.assertEqual(bounds.evaluate({'generalist':2})['reason'],'ILLEGAL_REPLICA_STATE')
            self.assertEqual(bounds.evaluate({'generalist':1})['reason'],'RESIDENT_MEMORY')
        c=search_fixture();p=ImmutableContinuousProblem(c)
        with SessionPredictorOracle(p) as oracle:
            bounds=ContinuousDeploymentBounds(p,oracle);x={'generalist':1}
            ub=bounds.evaluate(x)['upper_bound']
            self.assertTrue(bounds.evaluate(x,ub)['safe_pruned'])
            self.assertFalse(bounds.evaluate(x,ub-1)['safe_pruned'])

    def test_screen_probes_freeze_other_rows_and_keep_closed_simplex(self):
        c=replace(fitted_replay_fixture(),horizon_seconds=20.)
        p=ImmutableContinuousProblem(c)
        with SessionPredictorOracle(p) as oracle:
            e=FixedDeploymentRoutingEvaluator(p,oracle,{'generalist':1,'specialist':1})
            shares={t:{'specialist':1.} for t in c.workload}
            points=list(itertools.islice(screen_points(e,shares,11),12))
            self.assertTrue(any(r['text']['generalist']==.5 and r['image']['specialist']==1 for r in points))
            self.assertTrue(any(r['image']['generalist']==1 and r['text']['generalist']==1 for r in points))
            for point in points:e.domain.validate(point)

    def test_budget_slices_restore_on_failure_without_renewing_counts(self):
        b=Budget(10,10,10);deadline=b.deadline
        with self.assertRaises(BudgetExhausted):
            with budget_slice(b,2,2,1):
                b.consume('predictions');b.consume('predictions');b.consume('predictions')
        self.assertEqual(b.deadline,deadline);self.assertEqual(b.counts['predictions'],2)
        self.assertEqual(b.limits['predictions'],10)

    def test_adaptive_ledger_incumbent_and_fixed_deployment_restart_costs(self):
        c=replace(fitted_replay_fixture(),horizon_seconds=20.)
        r=AdaptiveCandidateRefinementOptimizer(c,{m:[1] for m in c.catalog['models']},k=3,
            safe_filter=False,seconds=15,max_predictions=90,max_routes=100,min_screen_routes=4,
            screen_routes=6,chunk_routes=15,max_extra_routes=30,stagnation_rounds=1).solve()
        self.assertGreaterEqual(r['winner']['utility'],r['initial_winner']['grid_utility'])
        self.assertLessEqual(r['budget']['full_prediction_calls'],90)
        self.assertLessEqual(r['budget']['outer_evaluations'],100)
        self.assertEqual(r['budget']['full_prediction_calls'],r['generation_counts']['predictions']+
                         sum(s['counts']['predictions'] for s in r['stages']))
        self.assertEqual(r['budget']['milp_solve_calls'],r['generation_counts']['milp'])
        self.assertEqual(r['budget']['outer_evaluations'],sum(s['counts']['outer'] for s in r['stages']))
        self.assertTrue(all(h['milp_solve_calls']==0 for s in r['stages'] for h in s['history']))
        self.assertTrue(all(s['improvement']>=0 for s in r['stages']))
        self.assertTrue(all(s['counts']['outer']==len(s['history']) for s in r['stages']))
        self.assertFalse(r['continuous_optimality'])
        for seg in r['segments']:
            if seg['stage']=='de_restart':self.assertIn('restart',seg['search_state'])
        self.assertTrue(any(d['action']=='pause' for d in r['decisions']))
        self.assertFalse(any(s['safe_pruned'] for s in r['stages']))

    def test_all_policies_preserve_single_shared_budget(self):
        c=search_fixture()
        for policy in ('legacy_uniform','uniform','priority','adaptive'):
            r=AdaptiveCandidateRefinementOptimizer(c,k=3,policy=policy,seconds=10,
                max_predictions=16,max_routes=30,min_screen_routes=2,screen_routes=4,
                chunk_routes=8,max_extra_routes=16).solve()
            self.assertLessEqual(r['budget']['full_prediction_calls'],16)
            self.assertLessEqual(r['budget']['outer_evaluations'],30)
            if r['initial_winner']:
                self.assertGreaterEqual(r['winner']['utility'],r['initial_winner']['grid_utility'])

    def test_candidate_generation_exhaustion_returns_observed_incumbent(self):
        r=AdaptiveCandidateRefinementOptimizer(search_fixture(),k=3,seconds=10,
            max_predictions=1,max_routes=20).solve()
        self.assertEqual(r['budget']['full_prediction_calls'],1)
        self.assertIsNotNone(r['winner'])
        self.assertEqual(sum(s['counts']['predictions'] for s in r['stages']),0)
        self.assertEqual(r['status'],'PREDICTIONS_LIMIT')

    def test_configuration_errors(self):
        for kwargs in (dict(k=0),dict(policy='mads'),dict(min_screen_routes=20,screen_routes=8),
                       dict(chunk_routes=0),dict(screening_fraction=0)):
            with self.assertRaises(ValueError):AdaptiveCandidateRefinementOptimizer(search_fixture(),**kwargs)


if __name__=='__main__':unittest.main()
