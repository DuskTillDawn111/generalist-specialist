"""Propagate lifecycle-only hazard reviews into the actual event clock.

v25 updated the oracle on lifecycle changes but the inherited scheduler only
accepted a new review when its GPU frontier changed or its old review expired.
An earlier SAVE could therefore be skipped. Preserve that scheduler's context
semantics and publish every earlier source-bound review before advancing time.
"""
import copy

from .learned_lifecycle_oracle_v25 import LearnedLifecycleOracle as V25Oracle
from .learned_lifecycle_oracle_v25 import LearnedLifecycleScheduler as V25Scheduler


class LearnedLifecycleOracle(V25Oracle):
    def provenance(self):
        return dict(super().provenance(),
                    latest_cumulative_hazard=copy.deepcopy(self.hazard))


class LearnedLifecycleScheduler(V25Scheduler):
    def decide(self, view):
        self.oracle.next_owner.observe_view(view)
        self.oracle.observe_lifecycle(view)
        query = self.snapshot(view)
        self.oracle.observe_query(query)
        hazard = self.oracle.hazard
        if (self.pending_switch is None and query.service_available
                and hazard is not None
                and self.oracle.restore_key == (query.held_owner, query.restore_start_ns)):
            review = hazard['next_review_ns']
            earlier = (review is not None and
                       (self.review_deadline_ns is None or review < self.review_deadline_ns))
            if earlier or self.oracle.save_due(query):
                query = self.snapshot(view, change_reason='lifecycle_hazard_review_changed')
                self._accept(self.oracle.decide(query), query)
                self.last_frontier = query.pending_frontier
        return super().decide(view)

    def provenance(self):
        return dict(super().provenance(), lifecycle_only_reviews_reach_engine=True,
                    context_time_rewound=False, missed_SAVE_tolerance_ns=0)
