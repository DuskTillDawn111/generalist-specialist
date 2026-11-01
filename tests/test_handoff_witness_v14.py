import unittest

from tools.predictor_validation_20261027.handoff_witness_v14 import prequeued_witness


def row(start,ready,known=True):
    return dict(observed_start_ns=start,certified_ready_bound_ns=ready,recorded_dependencies_known=known)


class PrequeuedHandoffTests(unittest.TestCase):
    def test_future_host_supply_cannot_label_a_handoff_cost(self):
        rows = [row(12,11),row(15,9)]
        self.assertEqual(prequeued_witness(rows,[12,15],10,20),rows[1])

    def test_actual_start_is_not_a_substitute_for_ready_certificate(self):
        rows = [row(12,None),row(13,8,False)]
        self.assertIsNone(prequeued_witness(rows,[12,13],10,20))

    def test_ready_at_restore_is_valid_but_start_at_next_save_is_outside(self):
        rows = [row(10,10),row(20,9)]
        self.assertEqual(prequeued_witness(rows,[10,20],10,20),rows[0])
        self.assertIsNone(prequeued_witness(rows[1:],[20],10,20))

    def test_preexisting_started_kernel_cannot_identify_next_restore(self):
        rows = [row(5,4)]
        self.assertIsNone(prequeued_witness(rows,[5],10,20))


if __name__ == '__main__':unittest.main()
