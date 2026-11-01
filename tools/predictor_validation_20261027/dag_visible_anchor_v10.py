"""Condition host START supply on an exact reader-visible START observation.

This anchors a host fact, never a CUDA cursor. A training draw supplies future
dispatch lags. Its observed lag residual must be inside a separately derived,
public-input/occurrence-specific training support interval. These intervals
are empirical diagnostics, not calibrated probabilities or risk bounds.
"""
from __future__ import annotations

import copy


class UnsupportedStartObservation(ValueError):
    """The exact host observation has no support under this scenario."""


def start_lag_radii(support):
    """Extract only raw-rederived LOO START-lag errors, never wall/phase errors.

    The caller must have verified the support against its immutable raw fit.
    Keeping this extraction separate makes its narrower scope inspectable.
    """
    radii = {}
    for row in support.get('leave_one_run_out_errors', []):
        error = row['absolute_channel_errors_ns'].get('start_lag')
        if type(error) is not int or error < 0:
            raise ValueError('Nonnegative raw-derived START-lag LOO errors required')
        coordinate = (row['input_key'], row['occurrence'])
        radii[coordinate] = max(radii.get(coordinate, 0), error)
    return radii


class ConditionalVisibleStartSupply:
    """Pure owner-READY-relative START supply with supported exact anchors."""

    def __init__(self, snapshot, scenario, radii):
        from .dag_busy_posterior_v8 import input_draw, key, occurrence_indices, _ns
        self.cohort = copy.deepcopy(snapshot['full_public_cohort'])
        self.visible_snapshot_sha256 = snapshot['snapshot_sha256']
        self.training_run_ids = copy.deepcopy(scenario['training_run_ids'])
        self.scenario_id = scenario['id']
        self.lags = {}; self.submissions = {}; self.anchors = {}; self.radii = {}
        indices = occurrence_indices(self.cohort)
        for rid, row in self.cohort.items():
            self.lags[rid] = _ns(input_draw(scenario['request_start_delay_ns_by_input'], row, indices))
            self.submissions[rid] = _ns(snapshot['visible_submissions'].get(rid, 0))
            observed = snapshot['observed_host_state']['requests'].get(rid, {})
            anchor = observed.get('handler_start_ns')
            if anchor is None:
                continue
            self.anchors[rid] = _ns(anchor)
            coordinate = (key(row['alias'], row['task'], row['pool_id']), indices[rid])
            if coordinate not in radii:
                raise UnsupportedStartObservation('PUBLIC_INPUT_OCCURRENCE_START_LAG_SUPPORT_UNAVAILABLE')
            self.radii[rid] = _ns(radii[coordinate])

    def __call__(self, row):
        return self.from_owner_ready(row, owner_ready_ns=row['arrival_ns'])

    def from_owner_ready(self, row, *, owner_ready_ns):
        from .dag_busy_posterior_v8 import _ns
        rid = row['request_id']
        if self.cohort.get(rid) != row:
            raise ValueError('Conditional START supply changed original public request')
        eligible = max(row['arrival_ns'], _ns(owner_ready_ns))
        if rid not in self.anchors:
            return max(eligible + self.lags[rid], self.submissions[rid])
        anchor = self.anchors[rid]
        if anchor < max(eligible, self.submissions[rid]):
            raise UnsupportedStartObservation('VISIBLE_START_PRECEDES_SCENARIO_OWNER_READY_OR_SUBMIT')
        residual = anchor - eligible - self.lags[rid]
        if abs(residual) > self.radii[rid]:
            raise UnsupportedStartObservation('VISIBLE_START_LAG_OUTSIDE_INPUT_OCCURRENCE_TRAINING_SUPPORT')
        return anchor
