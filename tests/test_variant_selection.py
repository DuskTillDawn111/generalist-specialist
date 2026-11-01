import unittest
from tools.predictor_validation_20261027.variant_selection import choose


class FixedVariantChoiceTests(unittest.TestCase):
    def row(self, decision=0, failed=0, normalized=0.):
        return {'decision_errors': decision, 'temporal_failed_or_unknown': failed,
                'normalized_temporal_error_mean': normalized}

    def test_decision_errors_have_priority_over_temporal_fit(self):
        self.assertEqual(choose({'old': self.row(0, 10, 10.), 'new': self.row(1, 0, 0.)}, ['old', 'new']), 'old')

    def test_violation_count_then_normalized_errors(self):
        self.assertEqual(choose({'old': self.row(0, 2, .1), 'new': self.row(0, 1, 10.)}, ['old', 'new']), 'new')
        self.assertEqual(choose({'old': self.row(0, 2, .3), 'new': self.row(0, 2, .2)}, ['old', 'new']), 'new')

    def test_exact_tie_keeps_predeclared_baseline(self):
        self.assertEqual(choose({'old': self.row(), 'new': self.row()}, ['old', 'new']), 'old')

    def test_unknown_normalized_error_is_positive_infinity(self):
        self.assertEqual(choose({'old': self.row(normalized=None), 'new': self.row(normalized=100.)}, ['old', 'new']), 'new')


if __name__ == '__main__':
    unittest.main()
