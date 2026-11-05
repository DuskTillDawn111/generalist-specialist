"""Independent finite-domain/exclusion and shared-budget wrapper checks."""

import copy
from dataclasses import replace
import unittest

from gsdeploy.continuous import Budget, ImmutableContinuousProblem, SessionPredictorOracle
from gsdeploy.continuous.contracts import digest
from gsdeploy.continuous.multi_candidate import (JointDeploymentCandidates,
    MultiCandidateContinuousRefinement, select_candidates, deployment_distance)
from tests.test_calibrated_search_bounds_v6 import search_fixture
from tests.test_calibrated_replay_v2 import fitted_replay_fixture
from tests.test_milp import grid_plans
from gsdeploy.continuous.refinement import FixedDeploymentRoutingEvaluator


class MultiCandidateTests(unittest.TestCase):
    def test_excludes_whole_deployment_and_matches_independent_cartesian_ranking(self):
        c=search_fixture();states={m:[1] for m in c.catalog['models']}
        p=ImmutableContinuousProblem(c,replica_options=states)
        with SessionPredictorOracle(p) as oracle:
            expected={}
            for x,r in grid_plans(c,2):
                if any(n!=1 for n in x.values()):continue
                prediction=oracle.session.predict(x,r)
                if prediction['constraints_eligible'] and prediction['U'] is not None:
                    key=digest(x);expected[key]=max(expected.get(key,-float('inf')),prediction['U'])
            ex=JointDeploymentCandidates(p,oracle,Budget(30,1000,1000))
            candidates=ex.generate(8,K=2)
            self.assertEqual({digest(c['deployment']) for c in candidates},set(expected))
            self.assertEqual(len(candidates),len(expected))
            self.assertEqual([round(c['grid_utility'],10) for c in candidates],
                             sorted([round(v,10) for v in expected.values()],reverse=True))
            for i,candidate in enumerate(candidates):
                self.assertAlmostEqual(candidate['grid_utility'],expected[digest(candidate['deployment'])])
                self.assertTrue(candidate['restricted_grid_certified'])
                self.assertEqual(ex.solves[i]['excluded_deployments'],[c['deployment'] for c in candidates[:i]])
            self.assertIsNone(ex.solves[-1]['winner'])

    def test_cache_reuse_is_exact_and_exclusions_dont_leak_between_fresh_solves(self):
        c=replace(fitted_replay_fixture(),horizon_seconds=20.)
        p=ImmutableContinuousProblem(c,replica_options={m:[1] for m in c.catalog['models']})
        with SessionPredictorOracle(p) as oracle:
            b=Budget(30,100,100);ex=JointDeploymentCandidates(p,oracle,b)
            a=ex.solve_one(K=1,excluded=[]);predictions=b.counts['predictions']
            again=ex.solve_one(K=1,excluded=[])
            self.assertEqual(a['winner'],again['winner'])
            self.assertEqual(b.counts['predictions'],predictions)
            self.assertGreater(b.counts['cache_hits'],0)
            alt=ex.solve_one(K=1,excluded=[a['winner']['deployment']])
            self.assertNotEqual(alt['winner']['deployment'],a['winner']['deployment'])
            restored=ex.solve_one(K=1,excluded=[])
            self.assertEqual(restored['winner'],a['winner'])

    def test_selection_keeps_unrestricted_first_and_diversifies_actual_deployment(self):
        candidates=[dict(generation_index=i,grid_utility=-.1-i*.001,deployment=d)
                    for i,d in enumerate([{'g':1,'s':1},{'g':2,'s':1},{'g':1,'s':2},{'g':1}])]
        utility=select_candidates(candidates,2,'utility')
        diverse=select_candidates(candidates,2,'diversity')
        self.assertEqual([c['generation_index'] for c in utility],[0,1])
        self.assertEqual([c['generation_index'] for c in diverse],[0,3])
        candidates[-1]['grid_utility']=0
        self.assertEqual(select_candidates(candidates,1)[0]['generation_index'],0)
        self.assertGreater(deployment_distance({'g':1},{'s':1}),deployment_distance({'g':1},{'g':2}))

    def test_pipeline_keeps_initial_and_has_no_deployment_milp_in_refinement(self):
        c=replace(fitted_replay_fixture(),horizon_seconds=20.)
        r=MultiCandidateContinuousRefinement(c,{m:[1] for m in c.catalog['models']},
            k=3,seconds=10,max_predictions=70,max_routes=32,seed=11).solve()
        self.assertGreaterEqual(r['winner']['utility'],r['initial_winner']['grid_utility'])
        self.assertIn(0,r['selected_indices'])
        self.assertEqual(r['budget']['milp_solve_calls'],r['generation_counts']['milp'])
        self.assertTrue(all(s['counts']['milp']==0 for s in r['stages']))
        self.assertEqual(r['budget']['full_prediction_calls'],r['generation_counts']['predictions']+
                         sum(s['counts']['predictions'] for s in r['stages']))
        self.assertTrue(all(s['continuous_utility']>=s['grid_utility'] for s in r['stages']))
        self.assertLessEqual(r['budget']['outer_evaluations'],32)
        self.assertFalse(r['global_top_k_certified'])
        self.assertFalse(r['continuous_optimality'])

    def test_one_prediction_limit_retains_initial_without_budget_renewal(self):
        r=MultiCandidateContinuousRefinement(search_fixture(),k=3,seconds=10,max_predictions=1,max_routes=30).solve()
        self.assertLessEqual(r['budget']['full_prediction_calls'],1)
        if r['initial_winner']:
            self.assertGreaterEqual(r['winner']['utility'],r['initial_winner']['grid_utility'])
        self.assertFalse(r['global_top_k_certified'])

    def test_wrong_problem_and_invalid_selection_are_rejected(self):
        p=ImmutableContinuousProblem(search_fixture());other=ImmutableContinuousProblem(search_fixture())
        with SessionPredictorOracle(p) as oracle:
            with self.assertRaisesRegex(ValueError,'MISMATCH'):
                JointDeploymentCandidates(other,oracle,Budget())
        with self.assertRaises(ValueError):select_candidates([],0)
        with self.assertRaises(ValueError):MultiCandidateContinuousRefinement(search_fixture(),k=0)

    def test_shared_multitask_specialist_and_multiple_executors_keep_closed_domain(self):
        c=replace(fitted_replay_fixture(),horizon_seconds=20.)
        p=ImmutableContinuousProblem(c,replica_options={m:[1,2] for m in c.catalog['models']})
        with SessionPredictorOracle(p) as oracle:
            b=Budget(10,20,20)
            e=FixedDeploymentRoutingEvaluator(p,oracle,{'generalist':1,'specialist':2},b)
            for shares in ({t:{'generalist':1.} for t in c.workload},
                           {t:{'specialist':1.} for t in c.workload},
                           {t:{'generalist':.3141592653589793,'specialist':.6858407346410207} for t in c.workload}):
                row=e.evaluate(shares)
                self.assertEqual(row['deployment'],{'generalist':1,'specialist':2})
                self.assertEqual(row['milp_solve_calls'],0)
                self.assertTrue(all(abs(sum(r.values())-c.workload[t])<1e-12 for t,r in row['routing_qps'].items()))
            self.assertEqual(set(e.domain.edges['text']),{'generalist','specialist'})
            self.assertEqual(set(e.domain.edges['image']),{'generalist','specialist'})

    def test_uncertified_candidate_never_becomes_a_top_k_certificate(self):
        c=replace(fitted_replay_fixture(),horizon_seconds=20.,milp_max_iterations=1)
        p=ImmutableContinuousProblem(c,replica_options={m:[1] for m in c.catalog['models']})
        with SessionPredictorOracle(p) as oracle:
            ex=JointDeploymentCandidates(p,oracle,Budget(10,100,100))
            candidates=ex.generate(2)
            self.assertTrue(candidates)
            self.assertTrue(all(not c['restricted_grid_certified'] for c in candidates))
            self.assertTrue(all('not continuous or global Top-K' in c['certificate_scope'] for c in candidates))


if __name__=='__main__':unittest.main()
