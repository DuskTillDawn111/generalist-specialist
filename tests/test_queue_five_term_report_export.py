"""Reporting must never turn pending or mismatched evidence into a benefit."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.queue_five_term_report_20261023 import Inputs, checked_pairs, csv_text, role_rankings


class ReportEvidenceTests(unittest.TestCase):
    def pair_fixture(self):
        runs=[]
        for role,value in [('coupled_spill',1.),('matched_no_spill',2.)]:
            runs.append(dict(label=role+'-seed401',audit_verified=True,eligible=True,
                             U=-value,cohort_signature='same',**{t+'_norm':value for t in 'QTLME'}))
        p=dict(strategy='coupled_spill',seed=401,both_evidence_valid_eligible=True,same_cohort=True,
               delta_U=1.,term_deltas_spill_minus_control={t:-1. for t in 'QTLME'},
               contribution_to_delta_U={t:.2 for t in 'QTLME'})
        return dict(summary=dict(paired=[p])),runs

    def test_complete_pair_and_sum(self):
        audit,runs=self.pair_fixture()
        pair=checked_pairs(audit,runs,'development')[0]
        self.assertEqual(pair['status'],'VALID')
        self.assertAlmostEqual(pair['delta_U'],pair['contribution_sum'])

    def test_unaudited_run_keeps_pair_pending(self):
        audit,runs=self.pair_fixture()
        runs[1]['audit_verified']=False
        pair=checked_pairs(audit,runs,'development')[0]
        self.assertEqual(pair['status'],'PENDING_OR_INVALID')
        self.assertIsNone(pair['delta_U'])
        self.assertNotIn('contribution_Q',pair)

    def test_wrong_contribution_refused(self):
        audit,runs=self.pair_fixture()
        audit['summary']['paired'][0]['contribution_to_delta_U']['Q']=-.2
        with self.assertRaisesRegex(ValueError,'contribution mismatch'):
            checked_pairs(audit,runs,'development')

    def test_wrong_cohort_refused(self):
        audit,runs=self.pair_fixture()
        runs[1]['cohort_signature']='different'
        with self.assertRaisesRegex(ValueError,'cohort mismatch'):
            checked_pairs(audit,runs,'development')

    def test_source_receipt_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.json'
            path.write_text('{}')
            ref=dict(path=str(path),sha256=hashlib.sha256(b'{}').hexdigest())
            self.assertEqual(Inputs().read(path,ref),{})
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'Receipt mismatch'):
                Inputs().read(path,ref)

    def ranking_fixture(self):
        roles=('coupled_spill','matched_no_spill','legacy_latency','legacy_no_spill')
        return [dict(role=role,seed=seed,status='completed',eligible=True,audit_verified=True,
                     causal_verified=True,U=-float(index+seed-400))
                for index,role in enumerate(roles) for seed in (401,402)]

    def test_four_roles_complete_rank_larger_U_first(self):
        rows=role_rankings(self.ranking_fixture(),[401,402])
        self.assertEqual([r['rank'] for r in rows],[1,2,3,4])
        self.assertEqual(rows[0]['mean_U'],-1.5)
        self.assertEqual(rows[0]['seed_U'],{'401':-1.,'402':-2.})

    def test_missing_one_seed_blocks_domain_ranking(self):
        rows=role_rankings(self.ranking_fixture()[:-1],[401,402])
        self.assertTrue(all(r['rank'] is None for r in rows))
        self.assertIsNone(rows[-1]['mean_U'])

    def test_missing_external_audit_blocks_mean_and_ranking(self):
        runs=self.ranking_fixture()
        runs[0]['audit_verified']=False
        rows=role_rankings(runs,[401,402])
        self.assertTrue(all(r['rank'] is None for r in rows))
        self.assertIsNone(rows[0]['mean_U'])
        self.assertIsNone(rows[0]['seed_U']['401'])

    def test_csv_retains_pending_empty_U(self):
        text=csv_text([dict(label='pending',U=None,status='PENDING')])
        self.assertIn('pending,,PENDING',text)
        self.assertNotIn('0.0',text)


if __name__=='__main__':
    unittest.main()
