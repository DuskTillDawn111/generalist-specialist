import unittest

from experiments.scoped_v3_search_decision import summarize_selection


class ScopedV3DecisionTests(unittest.TestCase):
    def setUp(self):
        self.review = {'default': {'assessment': {'mean_U': {'observed': {'candidate-1': -.51}}}}}

    def row(self, method, value, covered):
        return dict(method=method, seed=11, utility=value, matches_archived_pure_specialist=covered)

    def test_losing_unmeasured_proposal_does_not_erase_winner_evidence(self):
        decision = summarize_selection([self.row('BO', -.52, True), self.row('Joint', -.58, False)], self.review)
        self.assertTrue(decision['selected_plan_has_existing_measurements'])
        self.assertFalse(decision['all_method_winners_have_existing_measurements'])
        self.assertEqual(decision['selected_plan_archived_mean_U'], -.51)

    def test_better_unmeasured_proposal_is_not_promoted_to_measured(self):
        decision = summarize_selection([self.row('BO', -.52, True), self.row('Joint', -.50, False)], self.review)
        self.assertFalse(decision['selected_plan_has_existing_measurements'])
        self.assertEqual(decision['unconfirmed_predicted_winner']['method'], 'Joint')
        self.assertIsNone(decision['selected_plan_archived_mean_U'])
        self.assertEqual(decision['new_gpu_episodes'], 0)

    def test_tie_prefers_existing_evidence_and_duplicate_runs_rejected(self):
        rows = [self.row('a', -.5, False), self.row('b', -.5, True)]
        self.assertEqual(summarize_selection(rows, self.review)['selected_search_result']['method'], 'b')
        with self.assertRaises(ValueError):
            summarize_selection([rows[0], rows[0]], self.review)


if __name__ == '__main__':
    unittest.main()
