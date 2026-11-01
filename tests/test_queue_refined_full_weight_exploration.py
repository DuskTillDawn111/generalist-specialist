import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from fractions import Fraction
from tools.queue_refined_full_weight_exploration_20261023 import (
    TERMS,gains,pareto_certificate,solve_lp,exact_witness,
    mean_preference_thresholds,two_pair_obstruction,analyze)


def row(**changes):return {t:changes.get(t,0.) for t in TERMS}


class FullWeightExplorationTests(unittest.TestCase):
    def test_gain_sign_and_pareto_proves_no_strict_positive(self):
        g=gains(row(L=2.),row(L=1.))
        self.assertEqual(g['L'],-1.)
        cert=pareto_certificate(g)
        self.assertEqual(cert['status'],'CONTROL_WEAKLY_DOMINATES_SPILL')
        self.assertTrue(cert['strict_positive_spill_gain_impossible_for_all_nonnegative_weights'])
        result=solve_lp([g,row(E=1.)])
        self.assertEqual(result['exact_witness_status'],'EXACT_FEASIBLE')
        self.assertEqual(result['minimum_gain']['display'],0.)
        self.assertFalse(result['all_known_pairs_strictly_positive'])
    def test_maximin_has_common_strict_witness_and_exact_simplex(self):
        result=solve_lp([row(Q=1.,T=-1.),row(Q=-1.,T=1.,L=1.)])
        self.assertTrue(result['all_known_pairs_strictly_positive'])
        self.assertGreater(result['minimum_gain']['display'],0.)
        w={t:Fraction(v) for t,v in result['weights_exact'].items()}
        self.assertEqual(sum(w.values()),1)
        self.assertTrue(all(v>=0 for v in w.values()))
    def test_unknown_and_impossible_box_preserved(self):
        self.assertEqual(solve_lp([None])['status'],'UNKNOWN')
        self.assertEqual(solve_lp([row(Q=1.)],floor=.2)['status'],'INFEASIBLE_BOX')
        self.assertEqual(solve_lp([row(Q=1.)],cap=.1)['status'],'INFEASIBLE_BOX')
        self.assertEqual(pareto_certificate(None)['status'],'UNKNOWN')
        self.assertIsNone(gains(row(E=None),row()))
    def test_nearest_equal_nonnegative_and_positive_target(self):
        rows=[row(Q=1.,T=-2.)]
        a=solve_lp(rows,objective='nearest_mean',target=0.)
        self.assertEqual(a['exact_witness_status'],'EXACT_FEASIBLE')
        self.assertGreaterEqual(a['mean_gain']['display'],0.)
        self.assertLess(a['L1_distance_to_original_equal']['display'],.2)
        b=solve_lp(rows,objective='nearest_mean',target=.1)
        self.assertEqual(b['exact_witness_status'],'EXACT_FEASIBLE')
        self.assertGreaterEqual(Fraction(b['mean_gain']['exact']),Fraction.from_float(.1))
    def test_energy_and_equal_floor_analytic_thresholds(self):
        rows=[row(Q=-1.,T=-1.,L=-1.,P=-1.,M=-1.,E=.1)]
        report=mean_preference_thresholds(rows)
        energy=Fraction(report['energy_only_family']['break_even']['exact'])
        self.assertGreater(energy,Fraction(9,10));self.assertLess(energy,1)
        threshold=Fraction(report['equal_floor_cap1']['strict_positive_floor_threshold']['exact'])
        self.assertGreater(threshold,0);self.assertLess(threshold,Fraction(1,50))
        self.assertEqual(report['equal_floor_cap1']['best_terms'],['E'])
    def test_two_pair_conflict_exact_certificate(self):
        a=row(E=1.,L=-2.);b=row(E=-2.,L=1.)
        cert=two_pair_obstruction(a,b)
        self.assertTrue(cert['both_multipliers_strictly_positive'])
        self.assertTrue(all(Fraction(v['exact'])<=0 for v in cert['combined_gains'].values()))
        self.assertIsNone(two_pair_obstruction(row(E=1.),row(E=2.)))
    def test_projection_preserves_exact_box_and_rejects_large_change(self):
        raw=[.1,.1,.1,.1,.1,.5000000000000001]
        result=exact_witness(raw,.1,.5)
        self.assertIsNotNone(result);self.assertEqual(sum(result['weights'].values()),1)
        self.assertTrue(all(Fraction.from_float(.1)<=v<=Fraction.from_float(.5) for v in result['weights'].values()))
        self.assertIsNone(exact_witness([2.,0,0,0,0,0],0.,1.))
    def test_end_to_end_unknown_fifth_and_leave_one_out_mean_dependence(self):
        from tests.test_queue_refined_weight_sensitivity import WeightSensitivityTests
        native=WeightSensitivityTests().native()
        native=[r for r in native if not(r['stage']=='final_confirmation' and r['summary']['seed']==209 and r['role']=='risk_matched_no_spill')]
        for r in native:
            r['summary']['raw']=dict(path='/synthetic/never-opened',sha256='synthetic')
            if not r['stage'].startswith('development'):
                seed=r['summary']['seed'];u=r['summary']['utility']
                for t in TERMS:u[t+'_norm']=2.
                if r['role']=='calibrated_spill':
                    u['E_norm']=1. if seed==208 else 2.1
                    u['L_norm']=2.1
                if seed==209:u.update(E_norm=None,U=None,eligible=False)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('tools.queue_refined_full_weight_exploration_20261023.load_completed',return_value=(native,{},[])):
                result=analyze('none','none','none',Path(tmp)/'out',allow_interrupted=True)
            self.assertEqual(result['all_five_stable_positive'],'UNKNOWN')
            omitted=next(r for r in result['leave_one_pair_out'] if r['excluded_seed']==208)
            self.assertTrue(omitted['mean_pareto']['strict_positive_spill_gain_impossible_for_all_nonnegative_weights'])
            self.assertEqual(len(result['development']),8)
            self.assertEqual(len(result['restricted_symmetric_boxes']),11)
            self.assertTrue((Path(tmp)/'out'/'六项权重探索报告.md').exists())


if __name__=='__main__':unittest.main()
