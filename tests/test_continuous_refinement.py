"""Fixed residency, warm starts, exact caching and shared pipeline budgets."""

import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

from gsdeploy.continuous import (Budget, ContinuousProblem, ContinuousRouteDomain,
    ContinuousRoutingEvaluator, FixedDeploymentRoutingEvaluator, DERoutingOptimizer,
    PairwiseTransferSearch, JointMILPContinuousRefinement)
from gsdeploy.continuous.refinement import deployment_neighbors, project_shares
from tests.test_continuous_routing import tiny, TableOracle
from tests.test_calibrated_search_bounds_v6 import search_fixture
from tests.test_calibrated_replay_v2 import fitted_replay_fixture


class ContinuousRefinementTests(unittest.TestCase):
    def test_fixed_deployment_really_keeps_idle_residents_at_zero_flow(self):
        p=tiny();seen=[]
        def value(x,r,t):
            seen.append(copy.deepcopy(x));return -.1 if 's_shared' in x else -.9
        e=FixedDeploymentRoutingEvaluator(p,TableOracle(value),{'g':1,'s_shared':1})
        for route in ({'text':{'g':1},'image':{'g':1}},
                      {'text':{'s_shared':1},'image':{'s_shared':1}}):
            r=e.evaluate(route)
            self.assertEqual(r['deployment'],{'g':1,'s_shared':1})
            self.assertEqual(r['best_feasible_utility'],-.1)
            self.assertEqual(r['milp_solve_calls'],0)
        self.assertTrue(all(x=={'g':1,'s_shared':1} for x in seen))

    def test_undeployed_model_rejected_even_with_capacity_evidence(self):
        e=FixedDeploymentRoutingEvaluator(tiny(),TableOracle(lambda x,r,t:-.1),{'g':1})
        with self.assertRaisesRegex(ValueError,'UNSUPPORTED_ROUTE'):
            e.evaluate({'text':{'s_shared':1},'image':{'g':1}})
        with self.assertRaisesRegex(ValueError,'INVALID_FIXED_DEPLOYMENT'):
            FixedDeploymentRoutingEvaluator(tiny(),TableOracle(lambda x,r,t:-.1),{'g':99})
        e.deployment['g']=99
        with self.assertRaisesRegex(ValueError,'FIXED_DEPLOYMENT_CHANGED'):
            e.evaluate({'text':{'g':1},'image':{'g':1}})

    def test_exact_ratio_cache_distinguishes_tiny_flows(self):
        e=FixedDeploymentRoutingEvaluator(tiny(),TableOracle(lambda x,r,t:-.1),{'g':1,'s_shared':1})
        route={'text':{'g':1},'image':{'g':1}}
        a=e.evaluate(route);b=e.evaluate(route)
        self.assertEqual(a['full_prediction_calls'],1);self.assertEqual(b['full_prediction_calls'],0)
        route['text']={'g':1-1e-12,'s_shared':1e-12}
        c=e.evaluate(route)
        self.assertEqual(c['full_prediction_calls'],1)
        self.assertIn('s_shared',c['activation'])
        b['prediction']['traces'][0]['prediction']['U']=999
        self.assertEqual(e.evaluate({'text':{'g':1},'image':{'g':1}})['best_feasible_utility'],-.1)

    def test_unknown_is_not_certified_or_a_feasible_search_observation(self):
        e=FixedDeploymentRoutingEvaluator(tiny(),TableOracle(lambda x,r,t:'unknown'),{'g':1})
        r=DERoutingOptimizer(e,initial_shares={'text':{'g':1},'image':{'g':1}}).solve()
        self.assertIsNone(r['winner'])
        self.assertEqual(r['status'],'INCOMPLETE_PREDICTOR_EVIDENCE')
        self.assertFalse(e.history[0]['certified'])

    def test_encode_endpoints_faces_and_continuous_warm_start(self):
        d=ContinuousRouteDomain(tiny())
        for v in (0.,1e-12,.3141592653589793,1.):
            shares=d.decode([v,v])
            rebuilt=d.decode(d.encode(shares))
            for task in shares:
                for m in shares[task]: self.assertAlmostEqual(shares[task][m],rebuilt[task][m],places=15)
        with self.assertRaisesRegex(ValueError,'OUTSIDE_FACE'):
            d.encode(d.decode([.3,.4]),{t:('g',) for t in d.edges})
        e=FixedDeploymentRoutingEvaluator(tiny(),TableOracle(lambda x,r,t:-.1),{'g':1,'s_shared':1},Budget(10,100,1))
        initial=d.decode([.3141592653589793,.2718281828459045])
        DERoutingOptimizer(e,initial_shares=initial).solve()
        self.assertEqual(e.history[0]['routing_shares'],initial)

    def test_simple_local_search_improves_an_off_grid_test_objective(self):
        p=tiny();q=p.config.workload['text']
        o=TableOracle(lambda x,r,t:-(r['text'].get('g',0)/q-.31)**2-.1)
        e=FixedDeploymentRoutingEvaluator(p,o,{'g':1,'s_shared':1},Budget(10,1000,1000))
        r=PairwiseTransferSearch(e).solve({'text':{'g':1},'image':{'g':1}})
        self.assertGreater(r['winner']['best_feasible_utility'],-.1001)
        self.assertFalse(r['continuous_optimality'])
        self.assertEqual(e.budget.counts['milp'],0)

    def test_whole_pipeline_preserves_seed_and_does_not_resolve_deployment(self):
        c=replace(fitted_replay_fixture(),horizon_seconds=20.)
        for method in ('de','local'):
            r=JointMILPContinuousRefinement(c,seconds=10,max_routes=64,method=method).solve()
            self.assertGreaterEqual(r['winner']['utility'],r['initial_winner']['utility'])
            self.assertEqual(r['winner']['deployment'],r['initial_winner']['deployment'])
            self.assertEqual(r['budget']['milp_solve_calls'],r['initial_counts']['milp'])
            self.assertEqual(r['budget']['full_prediction_calls'],r['initial_counts']['predictions']+sum(h['predictions'] for h in r['histories']))
            self.assertTrue(all(h['milp_calls']==0 for h in r['histories']))
            self.assertFalse(r['optimality_certified'])
            self.assertFalse(r['continuous_optimality'])
            self.assertGreaterEqual(r['budget']['cache_hits'],1)

    def test_de_seed_deterministically_reproduces_route_sequence(self):
        def run():
            p=tiny();q=p.config.workload['text']
            e=FixedDeploymentRoutingEvaluator(p,TableOracle(lambda x,r,t:-(r['text'].get('g',0)/q-.314159)**2-.1),
                                               {'g':1,'s_shared':1},Budget(10,1000,64))
            r=DERoutingOptimizer(e,seed=11,initial_shares={'text':{'g':1},'image':{'g':1}}).solve()
            return r['winner'],[h['routing_shares'] for h in e.history]
        a,ra=run();b,rb=run()
        self.assertEqual(a['best_feasible_utility'],b['best_feasible_utility'])
        self.assertEqual(ra,rb)

    def test_one_prediction_budget_keeps_joint_incumbent(self):
        r=JointMILPContinuousRefinement(search_fixture(),seconds=10,max_predictions=1,max_routes=64).solve()
        self.assertEqual(r['winner'],r['initial_winner'])
        self.assertEqual(r['budget']['full_prediction_calls'],1)
        self.assertLessEqual(r['budget']['milp_solve_calls'],r['initial_counts']['milp'])

    def test_expired_setup_returns_no_incumbent_without_false_certificate(self):
        r=JointMILPContinuousRefinement(search_fixture(),seconds=1e-9).solve()
        self.assertIsNone(r['winner'])
        self.assertEqual(r['status'],'NO_VALID_JOINT_INCUMBENT')
        self.assertEqual(r['initial']['status'],'TIME_LIMIT')
        self.assertFalse(r['optimality_certified'])
        self.assertEqual(r['budget']['full_prediction_calls'],0)

    def test_neighborhood_separate_switch_legal_states_and_monotone_incumbent(self):
        p=ContinuousProblem(search_fixture())
        ns=deployment_neighbors(p,{'generalist':1},{t:{'generalist':q} for t,q in p.config.workload.items()},8)
        self.assertTrue(ns)
        self.assertTrue(all(n in p.states[m] for row in ns for m,n in row['deployment'].items()))
        for row in ns:
            d=ContinuousRouteDomain(p)
            shares=project_shares(p,row['deployment'],{t:{'generalist':q} for t,q in p.config.workload.items()})
            d.validate(shares)
        for enabled in (False,True):
            r=JointMILPContinuousRefinement(search_fixture(),seconds=10,max_routes=96,neighborhood=enabled,neighbor_limit=3).solve()
            self.assertEqual(r['neighborhood_enabled'],enabled)
            self.assertEqual(bool(r['neighbors']),enabled)
            self.assertGreaterEqual(r['winner']['utility'],r['initial_winner']['utility'])
            self.assertTrue(all(h['milp_calls']==0 for h in r['histories']))
            self.assertEqual(r['budget']['milp_solve_calls'],r['initial_counts']['milp'])


if __name__=='__main__':
    unittest.main()
