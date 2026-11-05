import unittest

from experiments.audit_activation_aggregate_capture import summarize_quality, score_output


class AggregateCaptureAuditTests(unittest.TestCase):
    def test_repeated_input_weights_and_reference_units_are_retained(self):
        a = dict(pool_id='short', error_numerator=1., reference_denominator=1.,
                 baseline_error_numerator=0., baseline_reference_denominator=1., output_changed=True)
        b = dict(pool_id='long', error_numerator=0., reference_denominator=9.,
                 baseline_error_numerator=1., baseline_reference_denominator=9., output_changed=False)
        result = summarize_quality([a, a, b], {}, 'ocr')
        self.assertEqual(result['requests'], 3)
        self.assertEqual(result['distinct_inputs'], 2)
        self.assertAlmostEqual(result['native_loss'], 2/11)
        self.assertAlmostEqual(result['matched_solo_native_loss'], 1/11)
        self.assertEqual(result['changed_outputs'], 2)

    def test_failed_output_or_missing_quality_denominator_not_averaged_away(self):
        with self.assertRaises(ValueError):
            score_output({'success': False}, 'image', 'x', {})
        with self.assertRaises(ValueError):
            summarize_quality([], {}, 'image')


if __name__ == '__main__':
    unittest.main()
