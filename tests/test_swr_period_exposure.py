"""Meaningful CPU checks for finite resonance and missing baseline accounting."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('swr_period',ROOT/'tools/spillover_context_20261022/swr_period_exposure.py')
f=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(f)


class PeriodExposure(unittest.TestCase):
    def test_two_offsets_in_period32_reach_only_two_pool32_inputs(self):
        self.assertEqual(f.periodic_residues(32,[7,24],32),[7,24])

    def test_coprime_period_reaches_all_pool_inputs(self):
        self.assertEqual(f.periodic_residues(63,[31],32),list(range(32)))

    def test_observed_prefix_requires_two_cycles_and_exact_whole_tape(self):
        self.assertEqual(f.finite_period(list('ababa')),2)
        self.assertIsNone(f.finite_period(list('abac')))
        self.assertIsNone(f.finite_period(['a']))

    def test_missing_b_does_not_make_partial_sum_complete(self):
        self.assertIsNone(f.bsum([1.,None,3.]))
        self.assertIsNone(f.imbalance(None,3.)['absolute_gap'])

    def test_imbalance_separates_count_invariant_work_difference(self):
        self.assertAlmostEqual(f.imbalance(150.,250.)['relative_gap_to_mean'],.5)
        self.assertEqual(f.imbalance(0.,0.)['relative_gap_to_mean'],0.)

    def test_all_declared20_are_fixed(self):
        self.assertEqual(len(f.SHARES)*len(f.SEEDS),20)
        self.assertEqual(f.SEEDS,[107,108]);self.assertEqual(f.SHARES,['0','1/65','1/64','1/63','1/33','1/32','1/31','1/17','1/16','1/15'])


if __name__=='__main__':unittest.main()
