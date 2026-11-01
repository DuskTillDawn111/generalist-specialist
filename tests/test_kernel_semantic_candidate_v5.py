"""Public feature projection fixtures, not execution measurements."""
import unittest

from tools.predictor_validation_20261027.kernel_semantic_candidate_v5 import scalar_relation,evaluate


class FeatureProjectionTests(unittest.TestCase):
    def test_real_integer_linear_operator_bytes_project_exactly(self):
        rule=scalar_relation(414,414*8192,4514,4514*8192)
        self.assertEqual(evaluate(rule,963),963*8192)

    def test_constant_head_positions_and_weight_bytes_stay_constant(self):
        self.assertEqual(evaluate(scalar_relation(414,1,4514,1),1025),1)

    def test_unsupported_relation_is_rejected_instead_of_interpolating_metadata(self):
        with self.assertRaises(ValueError):scalar_relation(414,7,4514,71)


if __name__=='__main__':unittest.main()
