"""A fixed zero-peer state has an exact duration-variation obstruction."""
import copy
import unittest

from experiments.diagnose_activation_zero_peer_variation import diagnose
from tests.test_activation_semantic_bands import band_template
from tests.test_band_support_audit import document


class ZeroPeerVariationTests(unittest.TestCase):
    def fixture(self):
        first=document('one',False);second=document('two',False)
        spans=second['requests'][0]['segments']
        spans[0]['end_ns']=16_000_000
        spans[1].update(start_ns=16_000_000,end_ns=22_000_000)
        spans[2].update(start_ns=22_000_000,end_ns=23_000_000)
        second['requests'][1]['segments'][0].update(start_ns=30_000_000,end_ns=40_000_000)
        second['requests'][1]['segments'][1].update(start_ns=40_000_000,end_ns=41_000_000)
        return dict(profiles=band_template(),normalized_runs=[first,second])

    def test_same_input_fourfold_duration_cannot_fit_twenty_percent_at_zero_peers(self):
        fit=self.fixture();before=copy.deepcopy(fit);report=diagnose(fit)
        self.assertEqual(fit,before)
        group=next(g for g in report['groups'] if g['family']=='a' and g['role']=='first')
        self.assertAlmostEqual(group['exact_minimax_work_error'],.6)
        self.assertTrue(group['constant_progress_cannot_meet_point2'])
        self.assertFalse(report['physical_predictor_qualified'])

    def test_any_actual_peer_interval_prevents_zero_peer_classification(self):
        fit=self.fixture();spans=fit['normalized_runs'][1]['requests'][1]['segments']
        spans[0].update(start_ns=8_000_000,end_ns=9_000_000)
        spans[1].update(start_ns=9_000_000,end_ns=10_000_000)
        report=diagnose(fit)
        self.assertFalse(any(g['family']=='a' and g['role']=='first' for g in report['groups']))


if __name__=='__main__':unittest.main()
