import copy
import unittest
from unittest.mock import patch

from experiments.bilevel_scalability_benchmark import PROTOCOL, build_problem, read, QueueCostOracle
from experiments.verify_bilevel_scheduler import ScheduledControlledOptimizer
from gsdeploy.continuous import Budget, ContinuousRoutingEvaluator
from gsdeploy.continuous.budget_scheduled_bilevel import BudgetScheduledBilevelOptimizer
from gsdeploy.continuous.contracts import digest
from tests.test_c1_lite import evaluator


class SchedulerTests(unittest.TestCase):
    def test_EI_gets_first_opportunity_despite_unseen_activation_spaces(self):
        protocol = read(PROTOCOL)
        p = build_problem(7,protocol)
        e = ContinuousRoutingEvaluator(p,QueueCostOracle(),budget=Budget(10,64,8,128))
        fit = ScheduledControlledOptimizer(e,'BO-MADS',11,protocol).solve()
        proposals = [r for r in fit['events'] if 'routing_shares' in r and r['source']!='initialization']
        self.assertEqual(proposals[0]['source'],'bo_expected_improvement')
        self.assertTrue(any(s['visits']==0 for s in fit['activation_spaces']['spaces']))
        self.assertEqual(fit['scheduler']['post_initialization_start'],4)

    def test_partial_unknowns_never_become_exact_when_refinement_is_capped(self):
        e = evaluator(lambda *args:'unknown',outer=24)
        fit = BudgetScheduledBilevelOptimizer(e,inner_predictions=1,inner_milp_calls=1,
            refinement_fraction=.25).solve()
        self.assertIsNone(fit['winner'])
        self.assertEqual(fit['surrogate']['training_observations'],0)
        self.assertTrue(all(not r['certified'] for r in fit['history']))
        self.assertLessEqual(fit['scheduler']['charged_repeats'],fit['scheduler']['repeat_limit'])
        self.assertTrue(all(n<=2 for n in fit['scheduler']['repeats_by_route'].values()))
        self.assertLessEqual(fit['budget']['outer_evaluations'],24)

    def test_global_opportunities_bounded_and_limits_never_renewed(self):
        e = evaluator(lambda x,r,t:-1.-sum((row.get('g',0.)/8-.37)**2 for row in r.values()),outer=35)
        limits, deadline = copy.deepcopy(e.budget.limits),e.budget.deadline
        fit = BudgetScheduledBilevelOptimizer(e,inner_predictions=1,inner_milp_calls=2).solve()
        events = [r for r in fit['scheduler']['events'] if r['action']=='GLOBAL_PROPOSAL']
        self.assertGreater(len(events),1)
        for a,b in zip(events,events[1:]):
            self.assertLessEqual(b['outer_before']-a['outer_after'],4)
        self.assertEqual(e.budget.limits,limits)
        self.assertEqual(e.budget.deadline,deadline)
        self.assertLessEqual(fit['budget']['outer_evaluations'],35)
        self.assertLessEqual(fit['scheduler']['charged_repeats'],fit['scheduler']['repeat_limit'])

    def test_memory_filter_requires_explicit_complete_bound(self):
        protocol = read(PROTOCOL)
        p = build_problem(3,protocol)
        e = ContinuousRoutingEvaluator(p,QueueCostOracle())
        optimizer = BudgetScheduledBilevelOptimizer(e)
        dense = e.domain.validate(dict(text={'g':.3,'s_shared':.3,'s_text':.4}, image={'g':.5,'s_shared':.5}))
        self.assertFalse(optimizer._memory_possible(dense))
        with patch.object(e.oracle,'memory_states',return_value={}):
            self.assertTrue(optimizer._memory_possible(dense))
        with patch.object(e.oracle,'memory_states',return_value={('g',1):1.}):
            self.assertTrue(optimizer._memory_possible(dense))
        pair = e.domain.validate({t:{'g':.5,'s_shared':.5} for t in p.config.workload})
        self.assertTrue(optimizer._memory_possible(pair))

    def test_reusing_certified_observation_does_not_charge_or_discard_incumbent(self):
        e = evaluator(lambda *args:-1.,outer=20)
        optimizer = BudgetScheduledBilevelOptimizer(e,exact_inner=True)
        optimizer.spaces.initialize()
        point = next(e.domain.endpoints())
        row = optimizer._evaluate(point,'initialization')
        self.assertTrue(row['certified'])
        before = e.budget.counts.copy()
        repeated = optimizer._evaluate(point,'mads_poll')
        self.assertEqual(e.budget.counts,before)
        self.assertEqual(repeated['best_feasible_utility'],row['best_feasible_utility'])
        self.assertEqual(optimizer._winner()['best_feasible_utility'],row['best_feasible_utility'])

    def test_seed_reproducibility_and_both_mechanisms_executed(self):
        trajectories=[]
        for _ in range(2):
            e=evaluator(lambda x,r,t:-1.-sum((row.get('g',0.)/8-.37)**2 for row in r.values()),outer=24)
            fit=BudgetScheduledBilevelOptimizer(e,seed=11).solve()
            sources={r['source'] for r in fit['events']}
            self.assertIn('bo_expected_improvement',sources)
            self.assertIn('mads_poll',sources)
            trajectories.append([digest(r['routing_shares']) for r in fit['events'] if 'routing_shares' in r])
        self.assertEqual(*trajectories)

    def test_invalid_scheduler_parameters_rejected(self):
        for values in (dict(global_interval=0),dict(activation_period=1),dict(refinement_fraction=-.1),
                       dict(refinement_fraction=True),dict(max_route_refinements=False)):
            with self.assertRaises(ValueError):
                BudgetScheduledBilevelOptimizer(evaluator(lambda *args:-1.),**values)


if __name__=='__main__':
    unittest.main()
