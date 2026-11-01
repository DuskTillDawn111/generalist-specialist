import unittest

from tools.predictor_validation_20261027.context_initial_transition_v13 import (
    holder_at, next_kernel_bound, support_summary,
)


class ContextInitialTransitionTests(unittest.TestCase):
    def test_holder_survives_until_source_save_and_gap_is_unknown(self):
        intervals = [(10, 20, 'previous'), (25, 35, 'next')]
        starts = [10, 25]
        self.assertEqual(holder_at(intervals, starts, 19), (10, 20, 'previous'))
        self.assertIsNone(holder_at(intervals, starts, 20))
        self.assertIsNone(holder_at(intervals, starts, 24))
        self.assertEqual(holder_at(intervals, starts, 25)[2], 'next')
        self.assertIsNone(holder_at(intervals, starts, 35))

    def test_original_restore_age_is_not_query_age(self):
        intervals = [(10, 100, 'owner')]
        first = holder_at(intervals, [10], 50)
        second = holder_at(intervals, [10], 80)
        self.assertEqual((50-first[0], 80-second[0]), (40, 70))

    def test_restore_does_not_supply_exact_service_time(self):
        bounds = next_kernel_bound([9, 29, 40], 20, 35)
        self.assertEqual((bounds['lower_ns'], bounds['upper_ns']), (20, 29))
        self.assertFalse(bounds['exact'])
        self.assertEqual(bounds['upper_witness'], 'first_owned_measured_kernel_START')

    def test_kernel_outside_envelope_cannot_bound_service(self):
        bounds = next_kernel_bound([10, 35, 40], 20, 35)
        self.assertIsNone(bounds['upper_ns'])
        self.assertIsNone(bounds['upper_witness'])
        self.assertEqual(bounds['lower_ns'], 20)

    def test_integer_context_boundaries_required(self):
        for start, end in [(20, 20), (21, 20), (-1, 20), (10.5, 20)]:
            with self.assertRaises(ValueError):
                next_kernel_bound([], start, end)
        with self.assertRaises(ValueError):
            holder_at([], [], 1.5)

    def test_thousands_of_rows_are_still_one_independent_capture(self):
        rows = [dict(capture='one', held_owner='A', state='same') for _ in range(1000)]
        summary = support_summary(rows, lambda r:r['state'])
        self.assertEqual(summary['groups_with_two_captures'], 0)
        rows.append(dict(capture='two', held_owner='B', state='same'))
        summary = support_summary(rows, lambda r:r['state'])
        self.assertEqual(summary['groups_with_two_captures'], 1)
        self.assertEqual(summary['observations_with_two_capture_exact_group'], 1001)
        self.assertEqual(summary['groups'][0]['labels'], {'A':1000, 'B':1})
        self.assertTrue(summary['exact_matching_is_diagnostic_not_final_predictor'])


if __name__ == '__main__':
    unittest.main()
