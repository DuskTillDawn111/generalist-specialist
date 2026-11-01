"""Unequal OCR/ASR reference lengths must not turn into a mean of ratios."""
import unittest

from gsdeploy.forecasting.metrics import score_forecast


def estimate(numerator, denominator):
    return dict(basis='expected_per_request_totals', numerator=numerator,
                denominator=denominator, evidence='cpu-training-only')


class ForecastQualityTotalTests(unittest.TestCase):
    def score(self, quality):
        rows = {str(i): dict(request_id=str(i), task='ocr', alias=alias, pool_id=pool,
                            arrival_ns=0, finish_ns=1000000)
                for i, (alias, pool) in enumerate([('a', 'short'), ('b', 'long')])}
        forecast = dict(cohort=rows, finishes=rows, board_energy_joules=1., peak_memory_mib=1.)
        return score_forecast(forecast, [],
            {'a': {'family': 'a', 'supported_tasks': ['ocr']},
             'b': {'family': 'b', 'supported_tasks': ['ocr']}},
            {'ocr': 1.}, {'ocr': 10.}, quality,
            phase_start_ns=0, now_ns=0, deadline_ns=1000000000,
            memory_budget_mib=100., energy_reference_watts=100.,
            observed_peak_memory_mib=1., observed_board_energy_joules=0.)

    def test_unequal_units_sum_before_divide(self):
        metric = self.score({'a': {'ocr': estimate(1., 1.)}, 'b': {'ocr': estimate(0., 100.)}})
        self.assertAlmostEqual(metric['terms']['Q'], 1./101.)
        self.assertEqual(metric['quality_totals_by_task']['ocr']['denominator'], 101.)

    def test_input_conditioning_with_explicit_family_fallback(self):
        metric = self.score({'a': {'ocr': {'by_input': {'short': estimate(2., 1.)}}},
                             'b': {'ocr': {'by_input': {}, 'fallback': estimate(0., 100.)}}})
        self.assertAlmostEqual(metric['terms']['Q'], 2./101.)

    def test_scalar_edit_ratio_cannot_establish_aggregate_denominators(self):
        metric = self.score({'a': {'ocr': 1.}, 'b': {'ocr': 0.}})
        self.assertIsNone(metric['terms']['Q'])
        self.assertIn('UNKNOWN_AGGREGATE_QUALITY:ocr', metric['unknown_reasons'])

    def test_raw_training_totals_without_per_request_semantics_are_unknown(self):
        metric = self.score({'a': {'ocr': {'numerator': 1., 'denominator': 2.}},
                             'b': {'ocr': estimate(0., 100.)}})
        self.assertIsNone(metric['terms']['Q'])


if __name__ == '__main__':
    unittest.main()
