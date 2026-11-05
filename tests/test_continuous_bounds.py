"""Independent exhaustive checks of scoped pruning and candidate-only reuse."""

import copy
from dataclasses import replace
import unittest

from gsdeploy.continuous import Budget, ContinuousProblem, ContinuousRoutingEvaluator, SyntheticColocationOracle
from gsdeploy.continuous.activation import deployments
from gsdeploy.continuous.safe_bounds import candidate_bound
from gsdeploy.continuous.benchmark import make_problem, oracle_from_spec
from tests.test_continuous_routing import fixture, tiny, TableOracle


class ContinuousBoundsTests(unittest.TestCase):
    def test_bound_is_valid_for_every_deployment_nominal_robust_multitrace(self):
        spec, original = fixture()
        checked = 0
        for policy in ('nominal', 'robust'):
            for load in (dict(text=4, image=4), dict(text=36, image=8), dict(text=90, image=50)):
                p = make_problem(replace(original.config, search_policy=policy, workload=load), {'train':[101, 211]})
                e = ContinuousRoutingEvaluator(p, oracle_from_spec(spec), budget=Budget(60, 10000))
                routes = list(e.domain.endpoints()) + [e.domain.decode([u]*e.domain.dimension)
                    for u in (1e-14, .1732050807568877, .6180339887498948, 1-1e-14)]
                for shares in routes:
                    routing = e.domain.to_qps(shares)
                    for x in deployments(p, routing):
                        bound = candidate_bound(e, x, routing)
                        actual = e.candidate(x, routing)
                        if bound['infeasible']:
                            self.assertEqual(actual['constraint_state'], 'INFEASIBLE')
                        elif actual['utility'] is not None:
                            self.assertGreaterEqual(bound['upper'], actual['utility'])
                        checked += 1
        self.assertGreater(checked, 600)

    def test_all_ablations_equal_independent_exhaustive_reference(self):
        spec, p = fixture()
        evaluators = [ContinuousRoutingEvaluator(p, oracle_from_spec(spec), mode, Budget(60, 10000, 100),
                        cache=False, inner_bounds=bounds, inner_warm_start=warm)
                      for mode,bounds,warm in [('enumeration',False,False),('bounded',False,False),
                          ('bounded',True,False),('bounded',False,True),('bounded',True,True)]]
        routes = list(evaluators[0].domain.endpoints()) + [evaluators[0].domain.decode([u]*3) for u in (.24,.61)]
        for route in routes:
            rows = [e.evaluate(route) for e in evaluators]
            for r in rows:
                self.assertEqual(r['best_feasible_utility'], rows[0]['best_feasible_utility'])
                self.assertEqual(r['certified'], rows[0]['certified'])
                self.assertEqual(r['remaining_deployments'], 0)
                self.assertEqual(r['evaluated_deployments']+r['safely_pruned_deployments'], r['deployment_domain_size'])
        self.assertLess(evaluators[2].budget.counts['predictions'], evaluators[0].budget.counts['predictions'])

    def test_generic_nonseparable_fallback_retains_idle_resident_optimum(self):
        e = ContinuousRoutingEvaluator(tiny(), TableOracle(lambda x,r,t: -.1 if 's_shared' in x else -.9), 'bounded')
        r = e.evaluate({'text':{'g':1},'image':{'g':1}})
        self.assertEqual(r['best_feasible_utility'], -.1)
        self.assertIn('s_shared', r['deployment'])
        self.assertTrue(r['certified'])
        self.assertEqual(r['full_prediction_calls'], 2)

    def test_unknown_prevents_certificate_when_cached(self):
        e = ContinuousRoutingEvaluator(tiny(), TableOracle(lambda x,r,t: 'unknown' if 's_shared' in x else -.1), 'bounded')
        route = {'text':{'g':1},'image':{'g':1}}
        r = e.evaluate(route)
        self.assertFalse(r['certified'])
        self.assertEqual(len(r['unresolved_candidates']),1)
        # Completed finite traversal is cached, but preserves UNKNOWN evidence.
        self.assertFalse(e.evaluate(route)['certified'])

    def test_warm_hint_is_freshly_evaluated_and_incompatible_hint_ignored(self):
        e = ContinuousRoutingEvaluator(tiny(), TableOracle(lambda x,r,t: -r['text'].get('g',0)),
                                      'bounded', cache=False, inner_bounds=False, inner_warm_start=True)
        a = e.evaluate({'text':{'g':1},'image':{'g':1}})
        b = e.evaluate({'text':{'g':.5,'s_shared':.5},'image':{'g':1}})
        self.assertNotEqual(a['best_feasible_utility'],b['best_feasible_utility'])
        self.assertEqual(b['warm_start_evaluated_deployments'],0)  # old g-only is illegal
        c = e.evaluate({'text':{'g':.4,'s_shared':.6},'image':{'g':1}})
        self.assertEqual(c['warm_start_evaluated_deployments'],1)
        self.assertEqual(c['full_prediction_calls'],1)
        self.assertNotEqual(b['best_feasible_utility'],c['best_feasible_utility'])
        c['warm_start_candidate'].clear()
        self.assertTrue(e.deployment_hint)

    def test_hint_rejected_on_changed_oracle_or_problem(self):
        for mutate in ('oracle','problem'):
            e = ContinuousRoutingEvaluator(tiny(), TableOracle(lambda x,r,t: -.1),'bounded')
            route = {'text':{'g':1},'image':{'g':1}}
            e.evaluate(route)
            if mutate=='oracle':
                e.oracle.version='changed'
            else:
                e.problem.config.catalog['safe_factor']=2
            with self.assertRaises(ValueError):
                e.evaluate(route)

    def test_budget_and_incomplete_route_not_cached(self):
        e = ContinuousRoutingEvaluator(tiny(), TableOracle(lambda x,r,t: -.1),'bounded',Budget(20,1,10))
        r = e.evaluate({'text':{'g':1},'image':{'g':1}})
        self.assertEqual(r['status'],'BUDGET_LIMIT_WITH_INCUMBENT')
        self.assertFalse(r['certified'])
        self.assertFalse(e.routing_cache)
        self.assertEqual(r['remaining_deployments'],1)

    def test_missing_global_floor_does_not_assume_nonpositive_U(self):
        oracle = TableOracle(lambda x,r,t: 2 if 's_shared' in x else 1)
        oracle.cost_lower_bound=None
        r=ContinuousRoutingEvaluator(tiny(),oracle,'bounded').evaluate({'text':{'g':1},'image':{'g':1}})
        self.assertEqual(r['best_feasible_utility'],2)
        self.assertTrue(r['certified'])

    def test_overridden_synthetic_oracle_does_not_inherit_fixture_bound(self):
        class Different(SyntheticColocationOracle):
            pass
        spec,p=fixture()
        e=ContinuousRoutingEvaluator(p,Different(),'bounded')
        routing=e.domain.to_qps(e.domain.decode([.5]*3))
        b=candidate_bound(e,{'g':2,'s_shared':2,'s_text':2},routing)
        self.assertFalse(b['infeasible'])
        self.assertEqual(b['scope'],'declared_global_and_state_floor')

    def test_bad_declared_bound_is_solver_error(self):
        for value in (-.01, 1.0):
            oracle=TableOracle(lambda x,r,t: value)
            oracle.state_costs=lambda p:{('g',1):.5}
            r=ContinuousRoutingEvaluator(tiny(),oracle,'bounded').evaluate({'text':{'g':1},'image':{'g':1}})
            self.assertEqual(r['status'],'SOLVER_ERROR')
            self.assertFalse(r['certified'])
            self.assertIsNone(r['valid_utility_upper_bound'])

    def test_legacy_missing_evidence_stays_unknown(self):
        from gsdeploy.continuous import LegacyPredictorOracle
        r=ContinuousRoutingEvaluator(tiny(),LegacyPredictorOracle(),'bounded').evaluate({'text':{'g':1},'image':{'g':1}})
        self.assertEqual(r['status'],'INCOMPLETE_EVIDENCE')
        self.assertFalse(r['certified'])

    def test_missing_quality_never_produces_formal_utility(self):
        spec,p=fixture()
        catalog=copy.deepcopy(p.config.catalog)
        catalog.pop('quality')
        p=ContinuousProblem(replace(p.config,catalog=catalog))
        e=ContinuousRoutingEvaluator(p,oracle_from_spec(spec),'bounded')
        r=e.evaluate({'text':{'g':1},'image':{'g':1}})
        self.assertIsNone(r['best_feasible_utility'])
        self.assertFalse(r['certified'])


if __name__ == '__main__':
    unittest.main()
