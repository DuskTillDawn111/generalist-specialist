import math
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tools.predictor_validation_20261027.save_hazard_v13 import (
    EDGES_NS, SCHEMA, SaveHazardCandidate, aggregate, capture_weights,
    evaluate, fit, frontier_features, objective, split_exposure)
from tools.predictor_validation_20261027.native_collect import receipt


F=[['one','text','pool-a',['prefill']]]


def episode(capture='a', envelope='e', entry=0, exit=2000, event=True, holder='one', frontier=None):
    return dict(capture=capture,envelope_id=envelope,age_entry_ns=entry,
                age_exit_ns=exit,save_observed=event,held_owner=holder,frontier=frontier or F)


class SaveHazardTests(unittest.TestCase):
    def test_event_on_edge_belongs_left_and_entry_is_excluded(self):
        self.assertEqual(list(split_exposure(episode(exit=2000))),[(0,1000,0),(1,1000,1)])
        self.assertEqual(list(split_exposure(episode(entry=1000,exit=2000))),[(1,1000,1)])
        self.assertEqual(list(split_exposure(episode(entry=999,exit=1000))),[(0,1,1)])

    def test_censored_exposure_is_retained_and_tail_not_clipped(self):
        r=episode(entry=49_999_999,exit=60_000_000,event=False)
        pieces=list(split_exposure(r))
        self.assertEqual(sum(n for _,n,_ in pieces),10_000_001)
        self.assertEqual(sum(e for _,_,e in pieces),0)
        self.assertEqual(pieces[-1],(len(EDGES_NS)-1,10_000_000,0))

    def test_same_envelope_splitting_does_not_gain_capture_weight(self):
        rows=[episode('a','1',exit=1000,event=False),episode('a','1',entry=1000),
              episode('b','2'),episode('b','3')]
        data=aggregate(rows);weights=capture_weights(data)
        self.assertEqual(weights,{'a':.5,'b':.25})
        self.assertEqual(data['counts']['episodes'],4)
        self.assertEqual(data['counts']['exposure_ns'],6000)
        self.assertEqual(data['counts']['events'],3)

    def test_full_exposure_loss_and_gradient_independent_finite_difference(self):
        x=csr_matrix([[1.,1.],[1.,0.]])
        exp=np.array([2.,5.]);events=np.array([1.,0.]);w=np.array([.5,.5]);beta=np.array([.1,.2])
        value,gradient=objective(beta,x,exp,events,w)
        expected=.5*(2*math.exp(.3)-.3)+.5*5*math.exp(.1)+.5*.2**2
        self.assertAlmostEqual(value,expected)
        for i in range(2):
            lo=beta.copy();hi=beta.copy();lo[i]-=1e-6;hi[i]+=1e-6
            finite=(objective(hi,x,exp,events,w)[0]-objective(lo,x,exp,events,w)[0])/2e-6
            self.assertAlmostEqual(gradient[i],finite,places=7)

    def test_constant_fit_equals_weighted_events_over_exposure(self):
        data=aggregate([episode(exit=1000),episode('b','2',exit=3000)])
        model=fit(data,'constant')
        self.assertTrue(model['optimizer']['success'])
        self.assertAlmostEqual(math.exp(model['coefficients'][0]),500.)

    def candidate(self):
        other=[['two','text','pool-b',['decode']]]
        rows=[episode(exit=3000),episode('b','2',exit=3000,holder='two',frontier=other)]
        return SaveHazardCandidate(fit(aggregate(rows),'age_additive'))

    def test_new_combination_can_predict_without_exact_cell(self):
        model=self.candidate()
        novel=[['one','text','pool-b',['decode']]]
        result=model.query('one',novel,1000,1000)
        self.assertFalse(result['exact_frontier_seen'])
        self.assertFalse(result['OOD'])
        self.assertGreater(result['hazard_per_ms'],0)
        self.assertGreater(result['conditional_CDF'],0)

    def test_new_categories_and_unobserved_tail_mark_ood_and_keep_candidate(self):
        model=self.candidate()
        result=model.query('one',[['one','image','new',['prefill']]],0,1000)
        self.assertTrue(result['OOD']);self.assertIsNone(result['conditional_CDF'])
        self.assertGreater(result['hazard_per_ms'],0)
        tail=model.query('one',F,1000,5000)
        self.assertEqual(tail['restricted_horizon_ns'],2000)
        self.assertIsNone(tail['conditional_CDF'])
        self.assertFalse(tail['complete_finite_distribution'])
        outside=model.query('one',F,5000,1000)
        self.assertEqual(outside['restricted_horizon_ns'],0)
        self.assertIn('AGE_OUTSIDE_OBSERVED_EXPOSURE',outside['OOD_reasons'])

    def test_support_gap_not_filled_and_all_unseen_fold_rows_score(self):
        train=aggregate([episode(exit=1000),episode('a','2',entry=3000,exit=4000)])
        model=fit(train,'age_additive');candidate=SaveHazardCandidate(model)
        self.assertTrue(candidate.query('one',F,2000,100)['OOD'])
        test=aggregate([episode('b','3',frontier=[['three','image','new',['encoder']]])])
        result=evaluate(model,test)
        self.assertEqual(result['unknown_rows_removed'],0)
        self.assertEqual(result['counts']['episodes'],1)
        self.assertTrue(result['unseen_features'])
        self.assertTrue(math.isfinite(result['full_exposure_poisson_loss_without_parameter_independent_constants']))

    def test_invalid_intervals_and_hidden_nonfeature_inputs_rejected(self):
        for row in (episode(entry=1000,exit=1000),episode(entry=-1),episode(exit=2.5)):
            with self.assertRaises(ValueError):list(split_exposure(row))
        with self.assertRaises(ValueError):frontier_features('one',[])
        with self.assertRaises(ValueError):frontier_features('one',[['one','text','pool',[]]])


ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/predictor-save-hazard-v13-20261028-training01/report.json'


@unittest.skipUnless(CAMPAIGN.exists(),'Actual CPU training campaign not present')
class FullSourceCampaignTests(unittest.TestCase):
    def test_complete_source_accounting_and_capture_separation(self):
        report=json.loads(CAMPAIGN.read_text())
        original=json.loads(Path(report['sources'][0]['path']).read_text())
        rows=original['episodes'];stats=json.loads(Path(report['sufficient_statistics']['path']).read_text())
        self.assertEqual(len(rows),329093)
        self.assertEqual(report['counts']['episodes'],len(rows))
        self.assertEqual(report['counts']['events'],sum(r['save_observed'] for r in rows))
        self.assertEqual(sum(r['exposure_ns'] for r in stats['rows']),sum(r['age_exit_ns']-r['age_entry_ns'] for r in rows))
        self.assertEqual(sum(r['events'] for r in stats['rows']),sum(r['save_observed'] for r in rows))
        for fold in report['folds']:
            held=fold['held_out_training_capture']
            self.assertNotIn(held,fold['training_captures'])
            for record in fold['models'].values():
                self.assertTrue(record['optimizer']['success'])
                self.assertEqual(record['held_training_capture']['counts']['episodes'],sum(r['capture']==held for r in rows))
                self.assertEqual(record['held_training_capture']['unknown_rows_removed'],0)
        self.assertFalse(report['validation_labels_used']);self.assertFalse(report['GPU_started'])
        for ref in report['sources']:
            self.assertEqual(receipt(ref['path']),ref)


if __name__=='__main__':unittest.main()
