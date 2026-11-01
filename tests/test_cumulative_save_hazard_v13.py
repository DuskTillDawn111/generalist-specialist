import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from scipy.integrate import quad

from tools.predictor_validation_20261027.cumulative_save_hazard_v13 import CumulativeSaveHazardController
from tools.predictor_validation_20261027.native_collect import receipt,save
from tools.predictor_validation_20261027.save_hazard_v13 import EDGES_NS,SCHEMA,frontier_features

A=[['a','text','pool',['decode']]]
B=[['b','text','pool',['decode']]]


class CumulativeHazardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'model.json'
        fs=sorted(set(frontier_features('holder',A))|set(frontier_features('holder',B)))
        columns=['intercept',*fs,*['age_bin:'+str(i) for i in range(len(EDGES_NS))]]
        coefficients=[math.log(2.)]+[math.log(.25) if f=='pending_owner:b' else 0. for f in fs]+[0.]*len(EDGES_NS)
        self.model=dict(schema=SCHEMA,mode='age_additive',columns=columns,coefficients=coefficients,
                        training_only=True,fixed_age_edges_ns=list(EDGES_NS),
                        training_vocabulary=fs,exact_frontier_keys=[],age_support_ns=[[0,1_000_000_000]])
        save(self.path,self.model)

    def tearDown(self):self.temp.cleanup()

    def controller(self,**kwargs):
        return CumulativeSaveHazardController(receipt(self.path),allow_unvalidated_approximation=True,**kwargs)

    def rewrite(self):
        self.path.unlink();save(self.path,self.model)

    def test_constant_rate_analytic_median_and_absolute_clock(self):
        c=self.controller();r=c.start_restore('holder',10_000_000,A,'restore1')
        self.assertEqual(r['absolute_save_candidate_ns'],10_000_000+math.ceil(math.log(2.)/2*1e6))
        self.assertEqual(r['next_review_ns'],r['absolute_save_candidate_ns'])
        self.assertTrue(r['approximation']);self.assertFalse(r['physical_execution_gate'])

    def test_piecewise_integral_matches_independent_numerical_quadrature(self):
        self.model['coefficients'][self.model['columns'].index('age_bin:8')]=math.log(3.)
        self.rewrite();c=self.controller();r=c.start_restore('holder',0,A,'restore1')
        end=300_000;r=c.update(r['state'],end,A)
        def independent(age):return 6. if 200_000<=age<500_000 else 2.
        expected=quad(independent,0,end,points=[200_000],epsabs=1e-9)[0]/1e6
        # This clock already passes the median crossing, which must be emitted
        # before a host refresh may apply a new rate or claim an actual restore.
        self.assertEqual(r['status'],'SAVE_BUDGET_REACHED')
        predicted=r['absolute_save_candidate_ns']
        independent_integral=quad(independent,0,predicted,points=[200_000],epsabs=1e-9)[0]/1e6
        self.assertGreaterEqual(independent_integral,math.log(2.))
        self.assertLess(independent_integral-math.log(2.),6e-6)
        self.assertGreater(expected,math.log(2.))

    def test_lower_rate_preserves_old_review_and_budget(self):
        c=self.controller();first=c.start_restore('holder',0,A,'restore1');old=first['next_review_ns']
        changed=c.update(first['state'],100_000,B)
        expected=100_000+math.ceil((math.log(2.)-.2)/.5*1e6)
        self.assertEqual(changed['absolute_save_candidate_ns'],expected)
        self.assertEqual(changed['next_review_ns'],old)
        self.assertEqual(changed['hazard_budget'],first['hazard_budget'])
        self.assertAlmostEqual(changed['consumed_hazard'],.2)
        reviewed=c.update(changed['state'],old,B)
        self.assertEqual(reviewed['absolute_save_candidate_ns'],expected)
        self.assertEqual(reviewed['next_review_ns'],expected)
        self.assertEqual(reviewed['state']['restore_ns'],0)
        self.assertFalse(reviewed['budget_redrawn'])

    def test_rate_increase_brings_review_forward_without_age_reset(self):
        c=self.controller();r=c.start_restore('holder',50_000,B,'r')
        updated=c.update(r['state'],150_000,A)
        expected=150_000+math.ceil((math.log(2.)-.05)/2*1e6)
        self.assertEqual(updated['absolute_save_candidate_ns'],expected)
        self.assertEqual(updated['next_review_ns'],expected)
        self.assertEqual(updated['restore_age_ns'],100_000)

    def test_host_refreshes_and_restore_budget_are_invariant(self):
        c=self.controller();first=c.start_restore('holder',0,A,'r')
        state=first['state']
        for now in range(1000,300_001,1000):state=c.update(state,now,A)['state']
        last=c.update(state,310_000,A)
        direct=c.update(first['state'],310_000,A)
        self.assertAlmostEqual(last['consumed_hazard'],direct['consumed_hazard'],places=14)
        self.assertEqual(last['absolute_save_candidate_ns'],first['absolute_save_candidate_ns'])
        self.assertEqual(last['state']['restore_key'],'r')

    def test_empty_unknown_is_not_a_physical_no_switch_rule(self):
        c=self.controller();r=c.start_restore('holder',0,A,'r')
        empty=c.update(r['state'],100_000,[])
        self.assertIsNone(empty['absolute_save_candidate_ns'])
        later=c.update(empty['state'],200_000,B)
        self.assertEqual(later['status'],'UNKNOWN_HAZARD_HISTORY')
        self.assertIsNone(later['absolute_save_candidate_ns'])
        self.assertIn('EMPTY_FRONTIER_HAZARD_UNKNOWN',later['uncertainty_reasons'])

    def test_explicit_noquery_pause_does_not_reset_restore_age(self):
        c=self.controller(empty_frontier_policy='NO_QUERY_PAUSE')
        first=c.start_restore('holder',0,A,'r');empty=c.update(first['state'],100_000,[])
        resumed=c.update(empty['state'],200_000,A)
        expected=200_000+math.ceil((math.log(2.)-.2)/2*1e6)
        self.assertEqual(resumed['absolute_save_candidate_ns'],expected)
        self.assertEqual(resumed['state']['restore_ns'],0)
        self.assertIn('NO_QUERY_PAUSE_ASSUMPTION',resumed['uncertainty_reasons'])
        self.assertFalse(resumed['source_query_support_flags_clear'])

    def test_seed_is_deterministic_once_per_restore_and_bound_to_solver(self):
        c=self.controller(budget_policy='SEEDED_UNIFORM',seed=42)
        a=c.start_restore('holder',0,A,'r');same=c.start_restore('holder',0,A,'r')
        other=c.start_restore('holder',0,A,'other')
        self.assertEqual(a['hazard_budget'],same['hazard_budget'])
        self.assertNotEqual(a['hazard_budget'],other['hazard_budget'])
        change=c.update(a['state'],0,B)
        self.assertEqual(change['hazard_budget'],a['hazard_budget'])

    def test_seal_foreign_solver_model_mutation_and_clock_rejected(self):
        c=self.controller();r=c.start_restore('holder',0,A,'r')
        bad=copy.deepcopy(r['state']);bad['consumed_hazard']=.5
        with self.assertRaises(ValueError):c.update(bad,100,A)
        foreign=self.controller(empty_frontier_policy='NO_QUERY_PAUSE')
        with self.assertRaises(ValueError):foreign.update(r['state'],100,A)
        with self.assertRaises(ValueError):c.update(r['state'],-1,A)
        self.model['coefficients'][0]=1.;self.rewrite()
        with self.assertRaises(ValueError):c.assert_sources_unchanged()

    def test_tail_and_new_category_only_give_marked_approximation(self):
        self.model['age_support_ns']=[[0,1000]];self.rewrite();c=self.controller()
        r=c.start_restore('holder',0,A,'r')
        self.assertIn('HORIZON_OUTSIDE_OBSERVED_EXPOSURE',r['uncertainty_reasons'])
        self.assertFalse(r['complete_finite_distribution'])
        new=c.start_restore('holder',0,[['a','new-task','new-pool',['new-phase']]],'new')
        self.assertIsNotNone(new['absolute_save_candidate_ns'])
        self.assertIn('UNSEEN_VISIBLE_CATEGORIES',new['uncertainty_reasons'])
        self.assertFalse(new['candidate_empirical_support_claim'])

    def test_due_save_is_terminal_and_cannot_be_renewed_by_host_update(self):
        c=self.controller();r=c.start_restore('holder',0,A,'r');deadline=r['absolute_save_candidate_ns']
        due=c.update(r['state'],deadline+1,B)
        self.assertEqual(due['status'],'SAVE_BUDGET_REACHED')
        self.assertEqual(due['absolute_save_candidate_ns'],deadline)
        self.assertEqual(due['next_review_ns'],deadline)
        later=c.update(due['state'],deadline+10_000,[])
        self.assertEqual(later['absolute_save_candidate_ns'],deadline)
        self.assertEqual(later['state']['frontier'],A)

    def test_optin_and_budget_policy_must_be_explicit(self):
        with self.assertRaises(ValueError):CumulativeSaveHazardController(receipt(self.path))
        with self.assertRaises(ValueError):self.controller(budget_policy='SEEDED_UNIFORM')
        with self.assertRaises(ValueError):self.controller(seed=42)
        with self.assertRaises(ValueError):self.controller(empty_frontier_policy='PHYSICAL_NO_SWITCH')


if __name__=='__main__':unittest.main()
