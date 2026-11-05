"""Global prediction exhaustion must stop search, not local refinement."""
import copy
import unittest
from unittest.mock import patch

from gsdeploy.continuous.activation_bilevel import ActivationAwareBilevelOptimizer
from gsdeploy.continuous.budget_scheduled_bilevel import BudgetScheduledBilevelOptimizer
from gsdeploy.continuous.contracts import BudgetExhausted
from tests.test_c1_lite import evaluator, pure


class PredictionStopTests(unittest.TestCase):
    def test_last_prediction_keeps_incumbent_and_partial_ledger(self):
        for cls in (ActivationAwareBilevelOptimizer, BudgetScheduledBilevelOptimizer):
            with self.subTest(controller=cls.__name__):
                e = evaluator(lambda *args: -1., predictions=1, outer=1000)
                opt = cls(e, inner_predictions=1, inner_milp_calls=20)
                with patch.object(opt, '_global_step', side_effect=AssertionError('No proposal after global exhaustion')):
                    fit = opt.solve()
                self.assertEqual(fit['status'], 'PREDICTIONS_LIMIT')
                self.assertEqual(e.budget.counts['predictions'], 1)
                self.assertEqual(e.budget.counts['outer'], 1)
                self.assertEqual(fit['winner']['best_feasible_utility'], -1.)
                self.assertEqual(len(opt.inner.history), 1)
                self.assertFalse(opt.inner.history[0]['certified'])
                self.assertEqual(fit['surrogate']['training_observations'], 0)

    def test_local_quota_restored_and_more_routes_can_be_evaluated(self):
        for cls in (ActivationAwareBilevelOptimizer, BudgetScheduledBilevelOptimizer):
            with self.subTest(controller=cls.__name__):
                e = evaluator(lambda *args: -1., predictions=10, outer=20)
                opt = cls(e, inner_predictions=1, inner_milp_calls=20)
                limits, deadline = copy.deepcopy(e.budget.limits), e.budget.deadline
                opt.spaces.initialize()
                first = opt._evaluate(pure(e), 'initialization', force=True)
                self.assertEqual(first['termination_reason'], 'PREDICTIONS_LIMIT')
                self.assertEqual(e.budget.counts['predictions'], 1)
                self.assertEqual(e.budget.limits, limits)
                second = opt._evaluate(pure(e, 's_shared'), 'initialization', force=True)
                self.assertIsNotNone(second['best_feasible_utility'])
                self.assertEqual(e.budget.counts['predictions'], 2)
                self.assertEqual(e.budget.limits, limits)
                self.assertEqual(e.budget.deadline, deadline)

    def test_exhausted_entry_does_not_charge_outer_or_milp(self):
        e = evaluator(lambda *args: -1., predictions=1)
        opt = BudgetScheduledBilevelOptimizer(e)
        opt.spaces.initialize()
        opt._evaluate(pure(e), 'initialization', force=True)
        before = e.budget.counts.copy()
        with self.assertRaisesRegex(BudgetExhausted, 'PREDICTIONS_LIMIT'):
            opt._evaluate(pure(e, 's_shared'), 'new_route')
        self.assertEqual(e.budget.counts, before)
        self.assertEqual(opt._winner()['best_feasible_utility'], -1.)

    def test_unknown_last_prediction_does_not_create_fake_winner(self):
        e = evaluator(lambda *args: 'unknown', predictions=1, outer=1000)
        fit = BudgetScheduledBilevelOptimizer(e, inner_predictions=1).solve()
        self.assertEqual(fit['status'], 'PREDICTIONS_LIMIT')
        self.assertIsNone(fit['winner'])
        self.assertEqual(fit['surrogate']['training_observations'], 0)
        self.assertFalse(fit['optimality_certified'])
        self.assertEqual(fit['budget']['outer_evaluations'], 1)

    def test_exact_inner_reports_global_stop_after_preserving_result(self):
        e = evaluator(lambda *args: -1., predictions=1, outer=1000)
        fit = BudgetScheduledBilevelOptimizer(e, exact_inner=True).solve()
        self.assertEqual(fit['status'], 'PREDICTIONS_LIMIT')
        self.assertEqual(fit['winner']['best_feasible_utility'], -1.)
        self.assertEqual(fit['budget']['outer_evaluations'], 1)


if __name__ == '__main__':unittest.main()
