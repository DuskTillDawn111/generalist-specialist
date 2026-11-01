"""Independent counterexamples for the finite public-case resource interface."""
import copy
import unittest

from tests import test_twoFIFO_public181_resource_campaign_v1 as fixtures
from tools.predictor_validation_20261027 import twoFIFO_public181_resource_campaign_v1 as m


class ForbiddenMemory:
    def __float__(self):
        raise AssertionError('future resource value inspected before clock gate')


class IndependentResourceCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Public181ResourceTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.Public181ResourceTests.tearDownClass()

    def rows(self):
        return fixtures.Public181ResourceTests(methodName='runTest').rows()

    def test_joint_startup_memory_energy_stays_in_same_complete_candidate_block(self):
        rows = self.rows()
        expected = {}
        for index, row in enumerate(rows):
            row['joint_tuple'] = dict(zip(m.TUPLE_FIELDS, [index*10+offset for offset in range(4)]))
            expected[(row['capture_id'], row['candidate_id'])] = copy.deepcopy(row['joint_tuple'])
        result = m.paired_resource_draws(list(reversed(rows)), fixtures.Public181ResourceTests.manifest)
        self.assertEqual(len(result['draws']), 2)
        for draw in result['draws']:
            for candidate, data in draw['candidates'].items():
                self.assertEqual(data['joint_tuple'], expected[(draw['block_id'], candidate)])
                self.assertFalse(result['queue_resource_joint_law_qualified'])
                self.assertFalse(result['probability_calibrated'])

    def test_future_poll_is_rejected_before_any_owned_value_is_read(self):
        polls = [dict(poll_begin_ns=1, poll_end_ns=3, sample_ns=2, owned_mib=ForbiddenMemory()),
                 dict(poll_begin_ns=3, poll_end_ns=7, sample_ns=4, owned_mib=ForbiddenMemory())]
        with self.assertRaisesRegex(ValueError, 'boundary-crossing'):
            m.audit_causal_prefix_polls(polls, cutoff_ns=5, forecast_origin_ns=10)

    def test_missing_protocol_or_foreign_block_cannot_be_paired_by_candidate_name(self):
        manifest = fixtures.Public181ResourceTests.manifest
        for mode in ('foreign-block', 'swap-candidate', 'plan', 'extra-term'):
            rows = self.rows()
            if mode == 'foreign-block':
                rows[0]['capture_id'] = 'foreign-block'
            elif mode == 'swap-candidate':
                rows[0]['candidate_id'] = rows[1]['candidate_id']
            elif mode == 'plan':
                rows[0]['plan'] = rows[1]['plan']
            else:
                rows[0]['joint_tuple']['amortized_energy'] = 0
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                m.paired_resource_draws(rows, manifest)

    def test_drain_peak_remains_in_memory_and_startup_energy_is_unamortized(self):
        rows = self.rows()
        rows[0]['joint_tuple'] = dict(startup_joules=7200, premeasurement_owned_sampled_peak_mib=2,
            measurement_through_drain_owned_sampled_peak_mib=999, measurement_board_joules=90)
        draws = m.paired_resource_draws(rows, fixtures.Public181ResourceTests.manifest)['draws']
        draw = next(d for d in draws if d['block_id'] == rows[0]['capture_id'])
        bridge = m.resource_bridge_payload(draw, rows[0]['candidate_id'], window_start_ns=123,
            resident_owners=list(m.fresh.OWNERS))
        self.assertEqual(bridge['startup_energy_joules'], 7200)
        self.assertEqual(bridge['phase_start_through_drain_owned_peak_mib'], 999)
        self.assertEqual(bridge['original_drain_end_ns'], 210_000_000_123)
        self.assertFalse(bridge['memory_safety_factor_already_applied'])
        self.assertFalse(bridge['startup_energy_already_amortized'])


if __name__ == '__main__':
    unittest.main()
