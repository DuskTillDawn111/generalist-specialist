"""Independent score-boundary counterexamples; native authentication is stubbed.

Actual resource integration, archived-world replay and six-score math execute.
"""
import copy
from fractions import Fraction as F
from pathlib import Path
import unittest

from tests import test_twoFIFO_matching_resource_rescore_v1 as fixtures
from tools.predictor_validation_20261027 import twoFIFO_matching_resource_rescore_v1 as m

SECOND=fixtures.SECOND


class IndependentMatchingScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=fixtures.MatchingRescoreTests();fixtures.MatchingRescoreTests.setUpClass()

    @classmethod
    def tearDownClass(cls):fixtures.MatchingRescoreTests.tearDownClass()

    def raw(self):
        data=self.fixture.data;plan=data['plans'][0]
        review=data['fit']['intake_evidence']['attempts'][0]['training_review']
        raw=m.checked(m.checked(review)['raw'])
        case=plan['cases'][0];epoch=raw['cases'][0]['observation_start_ns']
        score=m.checked(data['design']['scoring_plan']);quality=m.checked(data['design']['quality'])
        return raw,plan,case,epoch,score,quality

    def label(self,args):
        return m.measured_score_case(*args,source_evidence={'CPU_fixture':'explicitly_not_native_authentication'})

    def test_drain_peak_counts_for_M_but_neither_drain_power_nor_later_peak_counts_for_E(self):
        args=self.raw();raw,plan,case,epoch,score,quality=args
        baseline=self.label(args)
        for row in raw['board_samples']:
            if epoch+11*SECOND<=row['time_ns']<=epoch+69*SECOND:
                row['watts']=400.;row['owned_used_mib']=8000.
            if row['time_ns']>epoch+70*SECOND:row['owned_used_mib']=1000000.
        changed=self.label(args)
        self.assertEqual(changed['measured_resources']['measurement_board_energy_joules'],200.)
        self.assertEqual(changed['measured_resources']['startup_energy_joules'],40.)
        self.assertEqual(changed['measured_resources']['phase_start_through_drain_owned_peak_mib'],8000.)
        self.assertAlmostEqual(changed['measured_original_score']['terms']['M'],
            8000*score['memory_safe_factor']/score['memory_budget_mib'])
        for term in 'QTLPE':
            self.assertEqual(changed['measured_original_score']['terms'][term],baseline['measured_original_score']['terms'][term])

    def test_startup_owned_peak_carries_forward_and_safety_factor_is_applied_once(self):
        args=self.raw();raw,plan,case,epoch,score,quality=args
        raw['board_samples'][0]['owned_used_mib']=9000.
        changed=self.label(args)
        self.assertEqual(changed['measured_resources']['startup_through_phase_start_owned_peak_mib'],9000.)
        self.assertAlmostEqual(changed['measured_original_score']['terms']['M'],
            9000*score['memory_safe_factor']/score['memory_budget_mib'])
        self.assertFalse(changed['measured_resources']['memory_safety_factor_already_applied'])
        self.assertFalse(changed['measured_resources']['startup_energy_already_amortized'])

    def test_late_reader_receipt_retains_offered_denominator_and_is_unfinished(self):
        args=self.raw();raw,plan,case,epoch,score,quality=args
        wanted={r['request_id'] for r in case['arrivals']}
        for row in raw['request_records']:
            if row['request_id'] in wanted:row['completion_time_ns']=epoch+71*SECOND
        label=self.label(args)
        self.assertEqual(label['offered_requests'],len(wanted))
        self.assertTrue(all(not r['completed_within_fixed_drain'] for r in label['reader_labels']['requests']))
        # The preserved objective minimizes throughput loss: no timely result
        # produces loss one. It is never normalized over surviving requests.
        self.assertEqual(label['measured_original_score']['terms']['T'],1.)

    def test_resource_replacement_preserves_each_world_weight_unknown_reason_and_QTLP(self):
        data=self.fixture.generated(unknown=True);before=copy.deepcopy(data[4])
        result=self.fixture.score_generated(data)
        old=m.batch.single.decode(before);new=m.batch.single.decode(result['ensemble'])
        self.assertEqual([r['weight'] for r in old['draws']],[r['weight'] for r in new['draws']])
        self.assertEqual(new['draws'][1],old['draws'][1])
        self.assertEqual(new['six_terms']['E']['known_probability_mass'],F(1,3))
        self.assertIsNone(new['six_terms']['E']['expected_value'])
        for term in 'QTLP':
            self.assertEqual(new['draws'][0]['forecast']['scoring']['original_score']['terms'][term],
                old['draws'][0]['forecast']['scoring']['original_score']['terms'][term])

    def test_same_count_duplicate_case_and_wrong_block_are_rejected(self):
        plans=self.fixture.data['plans']
        for mode in ('duplicate','wrongblock'):
            report=copy.deepcopy(self.fixture.data['report']);fold=report['whole_block_leaveout'][0]
            if mode=='duplicate':fold['cases'][1]=copy.deepcopy(fold['cases'][0])
            else:fold['cases'][0]['attempt_id']=fold['donor_attempt_ids'][0]
            with self.assertRaises(ValueError):m._layout(report,plans)


if __name__=='__main__':unittest.main()
