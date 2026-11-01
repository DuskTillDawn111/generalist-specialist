"""Republish future lifecycle reviews after an old empty-frontier review expires.

The frozen v27 scheduler and its physical/context assumptions are inherited.
This only publishes an existing oracle event before the engine advances. It
does not rewind time, renew a hazard budget or invent same-holder restoration.
"""
from .dag_phase_scheduler_v8 import SchedulingDecision
from .learned_lifecycle_scheduler_v27 import LearnedLifecycleOracle
from .learned_lifecycle_scheduler_v27 import LearnedLifecycleScheduler as V27Scheduler


class LearnedLifecycleScheduler(V27Scheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.expired_review_republication_count=0

    def decide(self,view):
        # Preserve all original decision/UNKNOWN paths first. In particular an
        # already skipped SAVE still reaches the original no-rewind guard.
        decision=super().decide(view)
        query=self.snapshot(view)
        hazard=self.oracle.hazard
        if (self.pending_switch is None and query.service_available and hazard is not None
                and self.oracle.restore_key==(query.held_owner,query.restore_start_ns)):
            review=hazard['next_review_ns']
            expired=self.review_deadline_ns is None or self.review_deadline_ns<=view.clock_ns
            if expired and review is not None and review>view.clock_ns:
                query=self.snapshot(view,change_reason='expired_or_absent_lifecycle_review_republished')
                self._accept(self.oracle.decide(query),query)
                self.last_frontier=query.pending_frontier
                self.expired_review_republication_count+=1
                future=[stamp for stamp,_,_ in self.context_events if stamp>view.clock_ns]
                future.extend(stamp for stamp in (decision.next_transition_ns,self.review_deadline_ns)
                    if stamp is not None and stamp>view.clock_ns)
                return SchedulingDecision(decision.gpu_rates,min(future) if future else None)
        return decision

    def provenance(self):
        return dict(super().provenance(),expired_empty_frontier_reviews_republished=True,
            expired_review_republication_count=self.expired_review_republication_count,
            existing_future_review_extended=False,context_time_rewound=False,
            same_holder_reacquisition_support_invented=False)
