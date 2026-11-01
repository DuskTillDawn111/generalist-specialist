"""CPU comparisons do not imply new-shape measurements or context equivalence."""
import unittest

from tools.predictor_validation_20261027.kernel_template_compare_v4 import signature_comparison


class TemplateComparisonTests(unittest.TestCase):
    def test_grid_and_resource_mismatch_stays_explicit(self):
        r=signature_comparison([dict(name='a',grid=[1,1,1],block=[128,1,1])],
            [dict(name='a',grid=[2,1,1],block=[256,1,1])])
        self.assertFalse(r['all_positions_match']);self.assertEqual(r['changed_field_counts'],dict(grid=1,block=1))

    def test_count_mismatch_after_matching_prefix_is_not_a_match(self):
        r=signature_comparison([dict(name='a')],[dict(name='a'),dict(name='b')])
        self.assertFalse(r['all_positions_match']);self.assertEqual(r['first_mismatch_index'],1)

    def test_order_mismatch_is_detected_even_when_multiset_matches(self):
        r=signature_comparison([dict(name='a'),dict(name='b')],[dict(name='b'),dict(name='a')])
        self.assertEqual(r['matched_position_count'],0);self.assertEqual(r['first_mismatch_index'],0)


if __name__=='__main__':unittest.main()
