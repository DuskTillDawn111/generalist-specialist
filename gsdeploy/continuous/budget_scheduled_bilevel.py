"""Finite-budget scheduling for the existing certified BO/MADS controller.

The legacy controller is unchanged. This subclass bounds repeat evaluations,
interleaves activation discovery with EI, and yields local work often enough
for global proposals. Feasible incumbents and all inner certificates retain
their original meaning; skipped work never becomes an exact observation.
"""
import copy
import math

import numpy as np

from .activation import active_models
from .activation_bilevel import ActivationAwareBilevelOptimizer
from .contracts import digest


class BudgetScheduledBilevelOptimizer(ActivationAwareBilevelOptimizer):
    def __init__(self, evaluator, *, global_interval=4, activation_period=3,
                 refinement_fraction=.25, max_route_refinements=2,
                 local_batch_trials=1, **kwargs):
        for value in (global_interval, activation_period, max_route_refinements, local_batch_trials):
            if type(value) is not int or value < 1:
                raise ValueError('INVALID_SCHEDULER_COUNT')
        if activation_period < 2 or isinstance(refinement_fraction, bool) or not math.isfinite(refinement_fraction) or not 0 <= refinement_fraction <= 1:
            raise ValueError('INVALID_SCHEDULER_FRACTION_OR_PERIOD')
        super().__init__(evaluator, **kwargs)
        self.global_interval = global_interval
        self.activation_period = activation_period
        self.refinement_fraction = refinement_fraction
        self.max_route_refinements = max_route_refinements
        self.local_batch_trials = local_batch_trials
        self._schedule_start = None
        self._last_global = None
        self._global_count = 0
        self._repeat_limit = 0
        self._repeat_count = 0
        self._route_repeats = {}
        self._schedule_events = []
        self._rejected_centers = set()
        self._pruned_proposals = 0

    def _begin_schedule(self):
        if self._schedule_start is None:
            self._schedule_start = self.e.budget.counts['outer']
            self._last_global = self._schedule_start
            remaining = max(0, self.e.budget.limits['outer']-self._schedule_start)
            self._repeat_limit = int(math.floor(self.refinement_fraction*remaining))

    def _global_due(self):
        self._begin_schedule()
        return self._global_count == 0 or self.e.budget.counts['outer']-self._last_global >= self.global_interval

    def _repeat_reason(self, key):
        if self.e.domain.dimension == 0:
            return None  # There is no continuous search to reserve work for.
        self._begin_schedule()
        if self._repeat_count >= self._repeat_limit:
            return 'REPEAT_QUOTA'
        if self._route_repeats.get(key, 0) >= self.max_route_refinements:
            return 'ROUTE_REPEAT_QUOTA'
        if self._global_due():
            return 'GLOBAL_OPPORTUNITY_DUE'
        return None

    def _evaluate(self, shares, source, **kwargs):
        normalized = self.e.domain.validate(shares)
        key = digest(normalized)
        repeated = source != 'initialization' and key in self.rows
        if repeated:
            current = self.rows[key]
            reason = 'ALREADY_CERTIFIED' if current['certified'] else self._repeat_reason(key)
            if reason:
                self._schedule_events.append(dict(action='REUSE_EXISTING_OBSERVATION', source=source,
                    reason=reason, outer=self.e.budget.counts['outer'], routing_sha256=key,
                    certified=current['certified']))
                return copy.deepcopy(current)
        before = self.e.budget.counts['outer']
        try:
            return super()._evaluate(normalized, source, **kwargs)
        finally:
            used = self.e.budget.counts['outer']-before
            if repeated and used:
                self._repeat_count += used
                self._route_repeats[key] = self._route_repeats.get(key, 0)+used

    def _promote(self):
        if self.e.domain.dimension == 0:
            return super()._promote()
        if self._global_due():
            return False
        before = self.e.budget.counts['outer']
        super()._promote()
        # A quota-deferred observation must not count as scheduler progress.
        return self.e.budget.counts['outer'] > before

    def _refresh(self, row):
        # Leave room for the poll's new candidate. A stale pool version does
        # not invalidate a route-local lower/upper bound or its certificate.
        self._begin_schedule()
        if self.e.budget.counts['outer']-self._last_global >= self.global_interval-1:
            return self.rows[digest(row['routing_shares'])]
        return super()._refresh(row)

    def _local_step(self):
        if self._global_due():
            return False
        before = self.e.budget.counts['outer']
        old = self.poll_evaluations
        self.poll_evaluations = min(old, self.local_batch_trials)
        try:
            super()._local_step()
        finally:
            self.poll_evaluations = old
        # All polls may be already known. Let the enclosing scheduler try a
        # fresh global route rather than spinning without consuming work.
        return self.e.budget.counts['outer'] > before

    def _memory_possible(self, shares):
        """Reject only when an explicitly supplied additive memory floor proves it.

        No standalone estimate is substituted when the oracle lacks this
        capability. Ignoring inactive models weakens this necessary condition.
        """
        table = self.e.oracle.memory_states(self.e.problem)
        if not table:
            return True
        active = active_models(self.e.domain.to_qps(shares))
        minima = []
        for m in active:
            values = [table.get((m, n)) for n in self.e.problem.states[m] if n > 0]
            if not values or any(v is None or not math.isfinite(v) or v < 0 for v in values):
                return True
            minima.append(min(values))
        possible = math.fsum(minima) <= self.e.problem.config.gpu_memory_mib+1e-6
        if not possible:
            self._pruned_proposals += 1
        return possible

    def _activation_candidate(self):
        for key, space in list(self.spaces.spaces.items()):
            if space['visits'] or key in self._rejected_centers:
                continue
            shares = self.spaces.center(key)
            if digest(shares) in self.rows:
                continue
            if not self._memory_possible(shares):
                self._rejected_centers.add(key)
                continue
            self.spaces.expand(key)
            return shares
        return None

    def _proposal_candidates(self):
        candidates = {}
        for key, space in list(self.spaces.spaces.items()):
            self.e.budget.check()
            face, center = space['face'], self.spaces.center(key)
            points = [center]
            points.extend(self.spaces.random(key, self.rng) for _ in range(self.bo_candidates))
            vector = self.e.domain.encode(center, face)
            for j in range(len(vector)):
                for edge in (0., 1.):
                    trial = list(vector); trial[j] = edge
                    points.append(self.e.domain.decode(trial, face))
            for point in points:
                key_point = digest(point)
                if key_point not in self.rows and key_point not in candidates and self._memory_possible(point):
                    candidates[key_point] = point
        return list(candidates.values())

    def _global_step(self):
        self._begin_schedule()
        self.gp.fit(self.rows.values(), self.spaces.features)
        ready = self.use_bo and self.gp.factor is not None
        # First two global opportunities can use EI; every third reserves
        # activation exploration. Empty activation slots fall through to EI.
        activation_turn = not ready or (self._global_count+1) % self.activation_period == 0
        shares = self._activation_candidate() if activation_turn else None
        source = 'activation_exploration'
        if shares is None:
            candidates = self._proposal_candidates()
            if not candidates:
                return None
            if ready:
                features = np.asarray([self.spaces.features(p) for p in candidates])
                index = int(np.argmax(self.gp.expected_improvement(features)))
                source = 'bo_expected_improvement'
            else:
                index = int(self.rng.integers(len(candidates)))
                source = 'global_exploration'
            shares = candidates[index]
        before = self.e.budget.counts['outer']
        row = self._evaluate(shares, source)
        self._global_count += 1
        self._last_global = self.e.budget.counts['outer']
        key = tuple(active_models(row['routing_qps']))
        if key in self.spaces.spaces:
            self.spaces.expand(key)
        self._schedule_events.append(dict(action='GLOBAL_PROPOSAL', source=source,
            global_opportunity=self._global_count, gp_ready=ready,
            outer_before=before, outer_after=self._last_global))
        return row

    def solve(self):
        result = super().solve()
        result['scheduler'] = dict(version='bounded_interleaving_v1',
            global_interval=self.global_interval, activation_period=self.activation_period,
            local_batch_trials=self.local_batch_trials, refinement_fraction=self.refinement_fraction,
            max_route_refinements=self.max_route_refinements,
            post_initialization_start=self._schedule_start, repeat_limit=self._repeat_limit,
            charged_repeats=self._repeat_count, repeats_by_route=self._route_repeats,
            global_opportunities=self._global_count, events=copy.deepcopy(self._schedule_events),
            memory_rejected_proposals=self._pruned_proposals,
            memory_rejected_centers=len(self._rejected_centers),
            guarantees_conditional_on_remaining_budget_and_available_proposals=True)
        return result
