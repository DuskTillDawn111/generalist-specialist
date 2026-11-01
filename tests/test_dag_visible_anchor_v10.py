"""Scientific boundary tests for exact host anchors and lag-only support."""
import copy
import unittest

from tools.predictor_validation_20261027.dag_busy_posterior_v8 import key
from tools.predictor_validation_20261027.dag_visible_anchor_v10 import (
    ConditionalVisibleStartSupply, UnsupportedStartObservation, start_lag_radii)


class VisibleAnchorTests(unittest.TestCase):
    def setUp(self):
        self.rows = {rid: dict(request_id=rid, alias='a', task='text', pool_id='x', arrival_ns=at)
                     for rid, at in (('observed', 100), ('future', 120))}
        self.snapshot = dict(full_public_cohort=self.rows, snapshot_sha256='bound-visible-reader-facts',
            visible_submissions={'observed': 104},
            observed_host_state=dict(requests={'observed': dict(handler_start_ns=108)}))
        self.coordinate = key('a', 'text', 'x')
        self.scenario = dict(id='training-A', training_run_ids=['training-A'],
            request_start_delay_ns_by_input={self.coordinate: [10, 5]})

    def supply(self, radii=None):
        return ConditionalVisibleStartSupply(self.snapshot, self.scenario,
            {(self.coordinate, 0): 2} if radii is None else radii)

    def test_nonidentical_training_lag_conditions_to_exact_observed_start(self):
        supply = self.supply()
        before = copy.deepcopy(supply.__dict__)
        self.assertEqual(supply.from_owner_ready(self.rows['observed'], owner_ready_ns=90), 108)
        self.assertEqual(supply.__dict__, before)
        self.assertEqual(self.rows['observed']['arrival_ns'], 100)

    def test_future_start_retains_trained_lag_after_actual_scenario_ready(self):
        supply = self.supply()
        self.assertEqual(supply.from_owner_ready(self.rows['future'], owner_ready_ns=150), 155)
        self.assertEqual(self.rows['future']['arrival_ns'], 120)

    def test_anchor_cannot_bypass_previous_owner_ready_or_actual_submit(self):
        supply = self.supply()
        with self.assertRaisesRegex(UnsupportedStartObservation, 'OWNER_READY_OR_SUBMIT'):
            supply.from_owner_ready(self.rows['observed'], owner_ready_ns=109)
        self.snapshot['visible_submissions']['observed'] = 109
        with self.assertRaisesRegex(UnsupportedStartObservation, 'OWNER_READY_OR_SUBMIT'):
            self.supply().from_owner_ready(self.rows['observed'], owner_ready_ns=90)

    def test_out_of_support_start_is_rejected_instead_of_relaxed_or_shifted(self):
        supply = self.supply({(self.coordinate, 0): 1})
        with self.assertRaisesRegex(UnsupportedStartObservation, 'OUTSIDE_INPUT_OCCURRENCE'):
            supply.from_owner_ready(self.rows['observed'], owner_ready_ns=90)
        self.assertEqual(supply.anchors['observed'], 108)

    def test_only_same_input_occurrence_start_lag_error_supplies_radius(self):
        support = dict(observation_tolerance_ns=99999999, leave_one_run_out_errors=[
            dict(input_key=self.coordinate, occurrence=0,
                 absolute_channel_errors_ns=dict(start_lag=2, handler_wall_diagnostic=99999999)),
            dict(input_key=self.coordinate, occurrence=1,
                 absolute_channel_errors_ns=dict(start_lag=1000, phase_worker=99999999)),
            dict(input_key=key('a', 'text', 'other'), occurrence=0,
                 absolute_channel_errors_ns=dict(start_lag=1000000))])
        radii = start_lag_radii(support)
        self.assertEqual(radii[self.coordinate, 0], 2)
        self.assertEqual(self.supply(radii).from_owner_ready(self.rows['observed'], owner_ready_ns=90), 108)
        with self.assertRaisesRegex(UnsupportedStartObservation, 'SUPPORT_UNAVAILABLE'):
            self.supply({(self.coordinate, 1): 1000000})

    def test_zero_worker_start_is_observed_not_falsey(self):
        self.rows['observed']['arrival_ns'] = 0
        self.snapshot['visible_submissions']['observed'] = 0
        self.snapshot['observed_host_state']['requests']['observed']['handler_start_ns'] = 0
        self.scenario['request_start_delay_ns_by_input'][self.coordinate][0] = 0
        self.assertEqual(self.supply({(self.coordinate, 0): 0}).from_owner_ready(
            self.rows['observed'], owner_ready_ns=0), 0)


if __name__ == '__main__':
    unittest.main()
