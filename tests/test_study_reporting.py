import copy
import unittest

from gsdeploy.study_reporting import measured_summary, paired_comparisons


def run(key, seed, cost):
    return {"plan_key": key, "seed": seed, "utility": {
        "eligible": True, "U": -cost, **{term + "_norm": cost for term in "QTLPME"}}}


class PairedReportingTests(unittest.TestCase):
    def test_three_seeds_are_three_blocks_and_cannot_prove_significance(self):
        rows = [run(key, seed, cost) for key, cost in [("selected", 1), ("rival", 2)]
                for seed in (43, 44, 45)]
        result = paired_comparisons(rows, "selected", [43, 44, 45])
        pair = result["comparisons"][0]
        self.assertEqual(pair["U_differences"], [1, 1, 1])
        self.assertEqual(pair["student_interval"], [1, 1])
        self.assertEqual(pair["exact_two_sided_sign_flip_p"], .25)
        self.assertFalse(pair["sign_flip_significant_after_holm"])
        self.assertFalse(result["physical_global_optimality_certified"])

    def test_unfinished_seed_is_not_dropped_from_confirmatory_analysis(self):
        rows = [run(key, seed, cost) for key, cost in [("selected", 1), ("rival", 2)]
                for seed in (43, 44)]
        pair = paired_comparisons(rows, "selected", [43, 44, 45])["comparisons"][0]
        self.assertEqual(pair["status"], "INCOMPLETE_BLOCKS")
        self.assertEqual(pair["missing_seeds"], [45])
        self.assertNotIn("student_interval", pair)
        self.assertNotIn("exact_two_sided_sign_flip_p", pair)

    def test_candidate_with_no_finished_runs_is_retained(self):
        result = paired_comparisons([], "selected", [43, 44, 45],
                                    planned_plan_keys=["selected", "rival"])
        self.assertEqual(result["comparisons"][0]["missing_seeds"], [43, 44, 45])
        self.assertFalse(result["all_planned_blocks_complete"])

    def test_duplicate_and_invalid_measured_utilities_are_rejected(self):
        row = run("selected", 43, 1)
        with self.assertRaisesRegex(ValueError, "Repeated"):
            paired_comparisons([row, row], "selected", [43, 44, 45])
        invalid = copy.deepcopy(row)
        invalid["utility"]["E_norm"] = None
        with self.assertRaisesRegex(ValueError, "complete"):
            paired_comparisons([invalid], "selected", [43, 44, 45])

    def test_costs_match_full_utility_without_missing_term_renormalization(self):
        row = run("selected", 43, 1)
        weights = {term: 1 / 6 for term in "QTLPME"}
        self.assertAlmostEqual(sum(measured_summary([row], weights)[0]["weighted_costs"].values()), 1)
        row["utility"]["U"] = -.5
        with self.assertRaisesRegex(ValueError, "does not match"):
            measured_summary([row], weights)


if __name__ == "__main__":
    unittest.main()
