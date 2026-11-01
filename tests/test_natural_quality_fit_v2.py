import unittest
from tools.predictor_validation_20261027.natural_quality_fit_v2 import expected_totals


class QualityExpectedTotalsTests(unittest.TestCase):
    def test_expected_units_remain_per_request_and_ratio_of_sums(self):
        result=expected_totals([dict(numerator=1,denominator=1),dict(numerator=0,denominator=9)],'training-only')
        self.assertEqual(result['numerator'],.5)
        self.assertEqual(result['denominator'],5.)
        self.assertAlmostEqual(result['numerator']/result['denominator'],.1)
        self.assertNotEqual(result['numerator']/result['denominator'],.5)

    def test_unbounded_cer_retained_and_empty_training_not_zero(self):
        result=expected_totals([dict(numerator=4,denominator=1)],'training-only')
        self.assertEqual(result['numerator']/result['denominator'],4.)
        with self.assertRaises(ValueError):expected_totals([],'missing')


if __name__=='__main__':unittest.main()
