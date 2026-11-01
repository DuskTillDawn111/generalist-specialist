import copy
import unittest

from tools.predictor_validation_20261027 import h600_four_group_statistics_v1 as stats


class FourGroupStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.locked = {m: dict(no_spill='balanced', allow='spill_high') for m in stats.MODES}
        # Every term equals the cost, making U exactly minus that cost.
        self.rows = [dict(mode=m, route=r, terms={k: c for k in stats.TERMS},
                          requested=1200, known=1200, original_qualification='PASS')
                     for m in stats.MODES
                     for r,c in zip(stats.ROUTES, (2., 1., .5, 3.))]

    def test_posthoc_winner_does_not_replace_locked_policy(self):
        out = stats.block(self.locked, self.rows)
        mode = out['modes']['admission2']
        self.assertEqual(mode['locked_allow_minus_locked_no_spill'], -1.)
        self.assertEqual(mode['locked_allow_minus_strong_no_spill'], -2.)
        self.assertEqual(mode['post_observation_envelope_difference'], .5)
        self.assertFalse(mode['envelope_is_deployable_gain'])
        self.assertFalse(out['statistical_significance_claim'])

    def test_allow_can_choose_identical_no_spill_policy(self):
        for policy in self.locked.values(): policy['allow'] = 'balanced'
        out = stats.block(self.locked, self.rows)
        self.assertEqual(out['modes']['shared']['locked_allow_minus_locked_no_spill'], 0.)
        self.assertEqual(out['modes']['shared']['locked_allow_minus_strong_no_spill'], -1.)

    def test_empty_qualified_domain_has_no_optimum(self):
        for row in self.rows: row['original_qualification'] = 'FAIL'
        mode = stats.block(self.locked, self.rows)['modes']['admission2']
        self.assertEqual(mode['qualified_observed_envelope'], dict(
            status='NO_QUALIFIED_CANDIDATE_IN_DECLARED_DOMAIN', U=None, all_ties=[]))
        self.rows[4]['original_qualification'] = 'UNKNOWN'
        mode = stats.block(self.locked, self.rows)['modes']['admission2']
        self.assertEqual(mode['qualified_observed_envelope']['status'], 'DOMAIN_QUALIFICATION_UNKNOWN')

    def test_missing_cell_keeps_full_denominator_and_strong_baseline_unknown(self):
        self.rows.pop(5)  # admission2 skewed
        out = stats.block(self.locked, self.rows)
        self.assertEqual((out['requested'], out['known']), (9600, 8400))
        self.assertIsNone(out['modes']['admission2']['locked_allow_minus_strong_no_spill'])
        self.assertEqual(out['status'], 'DESCRIPTIVE_INCOMPLETE')

    def test_incomplete_requests_or_missing_term_cannot_supply_full_utility(self):
        self.rows[-1]['known'] = 1199
        out = stats.block(self.locked, self.rows)
        self.assertIsNone(out['modes']['admission2']['locked_allow_minus_locked_no_spill'])
        self.rows[-1]['known'] = 1200
        self.rows[-1]['terms']['Q'] = None
        self.assertEqual(stats.block(self.locked, self.rows)['status'], 'DESCRIPTIVE_INCOMPLETE')

    def test_all_observed_ties_retained_without_granting_model_optimum(self):
        self.rows[1]['terms'] = self.rows[2]['terms'].copy()
        out = stats.block(self.locked, self.rows)
        self.assertEqual(out['modes']['shared']['observed_allow_envelope']['all_ties'],
                         ['skewed', 'spill_low'])
        self.assertFalse(out['qualified_deployment_optimum_certified'])

    def test_duplicate_foreign_invalid_terms_and_locks_rejected(self):
        with self.assertRaises(ValueError): stats.block(self.locked, self.rows + [self.rows[0]])
        invalid = copy.deepcopy(self.rows)
        invalid[0]['mode'] = 'admission1'
        with self.assertRaises(ValueError): stats.block(self.locked, invalid)
        invalid = copy.deepcopy(self.rows)
        del invalid[0]['terms']['Q']
        with self.assertRaises(ValueError): stats.block(self.locked, invalid)
        invalid = copy.deepcopy(self.rows)
        invalid[0]['terms']['Q'] = float('nan')
        with self.assertRaises(ValueError): stats.block(self.locked, invalid)
        self.locked['shared']['no_spill'] = 'spill_low'
        with self.assertRaises(ValueError): stats.block(self.locked, self.rows)


if __name__ == '__main__':
    unittest.main()
