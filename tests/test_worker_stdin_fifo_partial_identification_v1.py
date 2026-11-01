from fractions import Fraction
import unittest

from tools.predictor_validation_20261027 import worker_stdin_fifo_partial_identification_v1 as module


class PairedPartialIdentificationTests(unittest.TestCase):
    def bounds(self,d,w,s,r):
        return module.transferred_START_bounds(d,w,s,r,assume_idle_pair_transfers=True)['START_ns_bounds']

    def test_interval_endpoints_are_attained_and_cover_all_paired_draws(self):
        for d in range(6):
            for w in range(6):
                for s in range(4):
                    for r in range(8):
                        u=min(w,d)
                        starts=[max(r,s+Fraction(i,4))+d-Fraction(i,4) for i in range(4*u+1)]
                        self.assertEqual(self.bounds(d,w,s,r),[min(starts),max(starts)])

    def test_idle_reproduces_same_START_but_busy_remains_interval(self):
        self.assertEqual(self.bounds(5,3,10,4),[15,15])
        self.assertEqual(self.bounds(5,3,10,20),[22,25])
        self.assertEqual(self.bounds(5,10,0,10),[10,15])

    def test_unknown_busy_transfer_is_not_implicitly_accepted(self):
        with self.assertRaisesRegex(ValueError,'UNKNOWN_BUSY_TRANSFER'):
            module.transferred_START_bounds(5,3,0,10,assume_idle_pair_transfers=False)
        with self.assertRaises(TypeError):module.transferred_START_bounds(5,3,0,10)
        with self.assertRaises(ValueError):module.paired_idle_components(5,3,4)

    def test_joint_pair_preserves_idle_d_and_rational_arithmetic(self):
        pair=module.paired_idle_components(Fraction(5,3),1,Fraction(1,4))
        self.assertEqual(pair['worker_pre_START_ns'],Fraction(17,12))
        self.assertEqual(self.bounds(Fraction(5,3),1,0,10),[Fraction(32,3),Fraction(35,3)])

    def test_positive_continuation_and_uncertainty_accumulation_examples(self):
        examples=module.counterexamples()
        first,second=examples['queued_START_before_parent_notices']['rows']
        self.assertEqual([first['worker_START_ns'],first['READY_occurrence_ns'],second['worker_START_ns']],[1,11,12])
        self.assertLess(second['worker_START_ns'],first['RESULT_reader_ns'])
        self.assertLess(second['worker_START_ns'],first['READY_processed_ns'])
        first,second=examples['sender_blocking_separate_from_worker_FIFO']['rows']
        self.assertEqual([first['parent_write_record_ns'],second['parent_submit_ns'],second['worker_START_ns']],[20,20,21])
        worlds=examples['two_millisecond_uncertainty_propagation']['coherent_worlds']
        self.assertEqual(worlds[0]['START_ns'],[12_000_000,24_000_000])
        self.assertEqual(worlds[1]['START_ns'],[10_000_000,20_000_000])


if __name__=='__main__':unittest.main()
