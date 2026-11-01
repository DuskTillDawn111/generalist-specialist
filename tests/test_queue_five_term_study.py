import copy
import unittest

from tools.queue_five_term_20261023.study import analyze, ROLES


class FiveTermStudyGateTests(unittest.TestCase):
    def observations(self):
        packet=dict(phase='development',seeds=[401,402],jobs=[dict(role=r) for r in ROLES for s in (401,402)])
        rows={}
        for r in ROLES:
            for s in (401,402):
                rows[f'{r}-seed{s}']=dict(role=r,seed=s,status='completed',eligible=True,
                    causal_verified=True,U=-1.+(.001 if r=='coupled_spill' else 0.),
                    actual_text_spills=2 if r in ('coupled_spill','legacy_latency') else 0)
        return packet,rows

    def test_complete_fixed_primary_can_advance(self):
        packet,rows=self.observations()
        self.assertTrue(analyze(packet,rows)['confirmation_gate_passed'])

    def test_good_legacy_cannot_rescue_negative_primary(self):
        packet,rows=self.observations()
        rows['coupled_spill-seed402']['U']=-2
        rows['legacy_latency-seed402']['U']=100
        self.assertFalse(analyze(packet,rows)['confirmation_gate_passed'])

    def test_missing_or_invalid_comparator_blocks_confirmation(self):
        packet,rows=self.observations()
        missing=copy.deepcopy(rows);del missing['legacy_no_spill-seed402']
        self.assertFalse(analyze(packet,missing)['confirmation_gate_passed'])
        rows['legacy_no_spill-seed402']['eligible']=False
        self.assertFalse(analyze(packet,rows)['confirmation_gate_passed'])

    def test_positive_score_without_actual_spill_does_not_prove_mechanism(self):
        packet,rows=self.observations()
        rows['coupled_spill-seed401']['actual_text_spills']=0
        self.assertFalse(analyze(packet,rows)['confirmation_gate_passed'])


if __name__=='__main__':unittest.main()
