import unittest
from fractions import Fraction

from tools.no_p95_weight_rescore_20261023 import solve_five
from tools.unified_weight_rescore_20261023 import TERMS


class NoP95Tests(unittest.TestCase):
    def test_large_p95_losses_do_not_enter_five_term_objective(self):
        a, b = dict.fromkeys(TERMS, 0), dict.fromkeys(TERMS, 0)
        a.update(Q=1, P=-10**9)
        b.update(L=1, P=-10**9)
        fit = solve_five([a, b])
        self.assertEqual(Fraction(fit["weights_exact"]["P"]), 0)
        self.assertAlmostEqual(fit["minimum_gain"], .5)
        self.assertAlmostEqual(fit["upper_bound_certificate"]["upper_bound"], .5)

    def test_p95_only_difference_is_a_tie(self):
        g = dict.fromkeys(TERMS, 0)
        g["P"] = -100
        fit = solve_five([g], .1, .5)
        self.assertEqual(fit["minimum_gain"], 0)
        self.assertFalse(fit["all_strictly_positive"])
        self.assertEqual(Fraction(fit["upper_bound_certificate"]["upper_bound_exact"]), 0)

    def test_twenty_percent_floor_forces_five_equal_not_six_equal(self):
        g = dict.fromkeys(TERMS, 1)
        g["P"] = -100
        fit = solve_five([g], .2, .5)
        self.assertEqual(fit["minimum_gain"], 1)
        self.assertEqual(sum(Fraction(w) for w in fit["weights_exact"].values()), 1)
        for t in TERMS:
            self.assertEqual(Fraction(fit["weights_exact"][t]), 0 if t=="P" else Fraction(1, 5))

    def test_empty_five_term_boxes_are_reported(self):
        g = dict.fromkeys(TERMS, 1)
        self.assertEqual(solve_five([g], .21, 1)["status"], "EMPTY_BOX")
        self.assertEqual(solve_five([g], 0, .19)["status"], "EMPTY_BOX")


if __name__ == "__main__":
    unittest.main()
