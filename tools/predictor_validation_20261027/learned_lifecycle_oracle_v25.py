"""Lifecycle-conditioned SAVE timing on the existing explicit context model.

Initial prior and handoff are unchanged candidates. Binary reference progress
remains an approximation. This does not turn context envelopes into a physical
exclusive GPU gate, nor silently accept same-holder reacquisition.
"""
from __future__ import annotations

import copy

from gsdeploy.forecast_errors import UnknownEvidence
from .learned_context_oracle_v15 import LearnedContextOracle as V15Oracle
from .learned_context_oracle_v15 import LearnedContextScheduler as V15Scheduler
from .learned_context_oracle_v14 import plain_frontier
from .cumulative_lifecycle_hazard_v24 import encode_lifecycle


class LearnedLifecycleOracle(V15Oracle):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.lifecycle = None
        self.lifecycle_clock_ns = None
        self.last_lifecycle = None

    def observe_lifecycle(self,view):
        if not hasattr(view,'owner_lifecycle'):
            raise UnknownEvidence('SETTLED_CAUSAL_OWNER_LIFECYCLE_VIEW_REQUIRED')
        states = dict(view.owner_lifecycle)
        encode_lifecycle(states)
        if self.lifecycle_clock_ns is not None and view.clock_ns < self.lifecycle_clock_ns:
            raise ValueError('Monotone lifecycle view required')
        self.lifecycle,self.lifecycle_clock_ns = states,view.clock_ns

    def observe_query(self,query):
        if self.lifecycle is None or self.lifecycle_clock_ns != query.clock_ns:
            raise UnknownEvidence('SAME_CLOCK_SETTLED_LIFECYCLE_REQUIRED')
        if query.held_owner is None or query.restore_start_ns is None or not query.service_available:
            return
        frontier = plain_frontier(query)
        lifecycle = encode_lifecycle(self.lifecycle)
        key = query.held_owner,query.restore_start_ns
        if key != self.restore_key:
            self.previous_held_owner = self.last_restored_owner
            self.last_restored_owner,self.restore_key = query.held_owner,key
            self.hazard = self.controller.start_restore(query.held_owner,query.restore_start_ns,
                frontier,str(key),self.lifecycle,observation_ns=query.clock_ns)
            self.last_frontier,self.last_lifecycle = frontier,lifecycle
        elif (frontier != self.last_frontier or lifecycle != self.last_lifecycle
                or self.hazard['next_review_ns'] is not None
                and query.clock_ns >= self.hazard['next_review_ns']):
            self.hazard = self.controller.update(self.hazard['state'],query.clock_ns,frontier,self.lifecycle)
            self.last_frontier,self.last_lifecycle = frontier,lifecycle
        self.uncertainty.update(self.hazard['uncertainty_reasons'])
        if self.hazard['uncertainty_reasons'] and not self.allow_extrapolation:
            raise UnknownEvidence('FLAGGED_LIFECYCLE_HAZARD_APPROXIMATION_NOT_AUTHORIZED')

    def provenance(self):
        return dict(super().provenance(),kind='composed_lifecycle_SAVE_and_mark_candidate.v25',
            current_lifecycle=None if self.lifecycle is None else encode_lifecycle(self.lifecycle),
            current_lifecycle_clock_ns=self.lifecycle_clock_ns,
            lifecycle_changes_trigger_hazard_integration=True,
            old_lifecycle_integrated_before_update=True,empty_risk_hazard_modelled=True,
            active_replacement_timing_independently_validated=False)


class LearnedLifecycleScheduler(V15Scheduler):
    def decide(self,view):
        self.oracle.next_owner.observe_view(view)
        self.oracle.observe_lifecycle(view)
        return super().decide(view)
