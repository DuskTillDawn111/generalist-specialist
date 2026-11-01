"""Held-context state and pending work are separate in this scheduler.

An oracle must identify availability; there is no default quantum or equal
share. Source replay verifies state semantics only, never prediction accuracy.
RESTORE_START identifies an envelope age, not physical SM execution service.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import json
import math

from gsdeploy.forecast_errors import UnknownEvidence
from .conditional_availability_scheduler_v8 import frontier
from .dag_phase_scheduler_v8 import SchedulingDecision


@dataclass(frozen=True)
class HeldContextQuery:
    clock_ns: int
    held_owner: str | None
    held_knowledge: str
    restore_start_ns: int | None
    context_age_ns: int | None
    service_available: bool
    pending_frontier: tuple
    pending_remaining_work_ns_by_owner: tuple
    pending_age_ns_by_node: tuple
    existing_review_deadline_ns: int | None
    change_reason: str


@dataclass(frozen=True)
class ContextAvailabilityDecision:
    kind: str
    next_review_ns: int
    target_owner: str | None = None
    save_end_ns: int | None = None
    restore_start_ns: int | None = None
    service_ready_ns: int | None = None


class UnidentifiedContextAvailability:
    def decide(self,query):
        raise UnknownEvidence('HELD_CONTEXT_AVAILABILITY_UNIDENTIFIED')
    def provenance(self):
        return dict(kind='unidentified_held_context_availability',parameters_available=False)


class HeldContextScheduler:
    def __init__(self,owners,*,oracle=None,allow_source_replay=False):
        self.owners=frozenset(owners)
        if not self.owners or any(not isinstance(p,str) or not p for p in self.owners):
            raise ValueError('Explicit resident owners required')
        self.oracle=oracle if oracle is not None else UnidentifiedContextAvailability()
        self.allow_source_replay=allow_source_replay
        self.held_owner=None;self.held_knowledge='UNKNOWN';self.restore_start_ns=None
        self.service_available=False;self.review_deadline_ns=None
        self.pending_switch=None;self.context_events=[];self.pending_since={}
        self.last_clock_ns=None;self.last_frontier=None;self.transitions=[];self.queries=[]

    def _clock(self,t):
        if type(t) is not int or self.last_clock_ns is not None and t<self.last_clock_ns:
            raise UnknownEvidence('MONOTONIC_HELD_CONTEXT_CLOCK_REQUIRED')
        self.last_clock_ns=t

    def _event(self,t,kind,owner,knowledge):
        if owner is not None and owner not in self.owners:
            raise UnknownEvidence('UNKNOWN_RESIDENT_CONTEXT_OWNER')
        if kind=='SAVE_END':
            if self.held_owner is not None and owner!=self.held_owner:
                raise UnknownEvidence('SAVE_OF_DIFFERENT_HELD_CONTEXT')
            self.held_owner=None;self.restore_start_ns=None;self.service_available=False
        elif kind=='RESTORE_START':
            if owner is None:raise UnknownEvidence('RESTORE_OWNER_REQUIRED')
            if self.held_owner is not None:
                raise UnknownEvidence('RESTORE_WITHOUT_PRIOR_SAVE')
            self.held_owner=owner;self.restore_start_ns=t;self.service_available=False
        elif kind=='SERVICE_READY':
            if owner!=self.held_owner or self.restore_start_ns is None:
                raise UnknownEvidence('SERVICE_WITHOUT_MATCHING_RESTORE')
            self.service_available=True
        else:raise UnknownEvidence('UNDECLARED_CONTEXT_EVENT')
        self.held_knowledge=knowledge
        self.transitions.append(dict(clock_ns=t,kind=kind,owner=owner,knowledge=knowledge))

    def observe_context_event(self,clock_ns,kind,owner):
        """Explicit source-replay input; prohibited in an ordinary predictor."""
        if self.allow_source_replay is not True:raise ValueError('Source replay opt-in required')
        if self.context_events:raise ValueError('Source events cannot overwrite a modelled switch')
        self._clock(clock_ns);self._event(clock_ns,kind,owner,'OBSERVED_SOURCE_REPLAY')

    def _advance_context(self,t):
        while self.context_events and self.context_events[0][0]<=t:
            stamp,kind,owner=self.context_events.pop(0)
            self._event(stamp,kind,owner,'MODELLED_AVAILABILITY_HYPOTHESIS')
        if self.pending_switch is not None and not self.context_events:
            self.pending_switch=None

    def snapshot(self,view,*,change_reason='source_replay_snapshot'):
        self._clock(view.clock_ns);self._advance_context(view.clock_ns)
        active={(*n.key,n.request_id):n for n in view.active_gpu_nodes}
        for key in list(self.pending_since):
            if key not in active:del self.pending_since[key]
        remaining={}
        for key,node in active.items():
            if key[0] not in self.owners:raise UnknownEvidence('FOREIGN_PENDING_GPU_OWNER')
            if not math.isfinite(node.remaining_reference_work_ns) or node.remaining_reference_work_ns<0:
                raise UnknownEvidence('FINITE_NONNEGATIVE_PENDING_REFERENCE_WORK_REQUIRED')
            self.pending_since.setdefault(key,view.clock_ns)
            remaining[key[0]]=remaining.get(key[0],0.)+node.remaining_reference_work_ns
        state=frontier(view)
        # A JSON round trip is only canonical immutability, never estimation.
        frozen=tuple((p,task,pool,tuple(phases)) for p,task,pool,phases in state)
        return HeldContextQuery(view.clock_ns,self.held_owner,self.held_knowledge,self.restore_start_ns,
            None if self.restore_start_ns is None else view.clock_ns-self.restore_start_ns,
            self.service_available,frozen,tuple(sorted(remaining.items())),
            tuple((key,view.clock_ns-self.pending_since[key]) for key in sorted(active)),
            self.review_deadline_ns,change_reason)

    def _accept(self,decision,query):
        if not isinstance(decision,ContextAvailabilityDecision):raise UnknownEvidence('EXPLICIT_CONTEXT_AVAILABILITY_DECISION_REQUIRED')
        now=query.clock_ns
        if type(decision.next_review_ns) is not int or decision.next_review_ns<=now:
            raise UnknownEvidence('STRICTLY_FUTURE_CONTEXT_REVIEW_REQUIRED')
        old=self.review_deadline_ns
        if decision.kind=='hold':
            if any(v is not None for v in (decision.target_owner,decision.save_end_ns,decision.restore_start_ns,decision.service_ready_ns)):
                raise UnknownEvidence('HOLD_MUST_NOT_INVENT_CONTEXT_EVENTS')
            if old is not None and old>now and decision.next_review_ns>old:
                raise UnknownEvidence('CONTEXT_DEADLINE_EXTENSION_WITHOUT_RESTORE')
        elif decision.kind=='switch':
            values=(decision.save_end_ns,decision.restore_start_ns,decision.service_ready_ns)
            if any(type(t) is not int for t in values) or not now<=values[0]<=values[1]<=values[2]<decision.next_review_ns:
                raise UnknownEvidence('ORDERED_ABSOLUTE_SAVE_RESTORE_SERVICE_TIMES_REQUIRED')
            if decision.target_owner not in self.owners:raise UnknownEvidence('EXPLICIT_RESIDENT_SWITCH_TARGET_REQUIRED')
            if decision.target_owner==self.held_owner:raise UnknownEvidence('SAME_CONTEXT_RESTORE_IS_NOT_A_HOST_UPDATE')
            # A pending transition is immutable; callers cannot resubmit it on
            # every host arrival and postpone restoration.
            if self.pending_switch is not None:raise UnknownEvidence('PENDING_CONTEXT_SWITCH_ALREADY_EXISTS')
            self.pending_switch=copy.deepcopy(decision)
            self.context_events=[(decision.save_end_ns,'SAVE_END',self.held_owner),
                (decision.restore_start_ns,'RESTORE_START',decision.target_owner),
                (decision.service_ready_ns,'SERVICE_READY',decision.target_owner)]
        else:raise UnknownEvidence('UNDECLARED_CONTEXT_AVAILABILITY_ACTION')
        self.review_deadline_ns=decision.next_review_ns
        self.queries.append(dict(clock_ns=now,held_owner=query.held_owner,context_age_ns=query.context_age_ns,
            pending_frontier=query.pending_frontier,change_reason=query.change_reason,decision=copy.deepcopy(decision.__dict__)))
        self._advance_context(now)

    def decide(self,view):
        query=self.snapshot(view)
        state=query.pending_frontier;changed=state!=self.last_frontier
        if not view.active_gpu_nodes:
            # Even an empty ready set cannot release or renew a context.
            return SchedulingDecision({})
        due=self.review_deadline_ns is None or view.clock_ns>=self.review_deadline_ns
        if self.pending_switch is None and (due or changed):
            reason='initial_unidentified_context' if self.held_knowledge=='UNKNOWN' else 'deadline' if due else 'pending_frontier_changed'
            query=self.snapshot(view,change_reason=reason)
            self._accept(self.oracle.decide(query),query)
            self.last_frontier=state
        future=[t for t,_,_ in self.context_events if t>view.clock_ns]
        if self.review_deadline_ns is not None and self.review_deadline_ns>view.clock_ns:future.append(self.review_deadline_ns)
        boundary=min(future) if future else None
        rates={n.key:float(self.service_available and n.key[0]==self.held_owner) for n in view.active_gpu_nodes}
        if not any(rates.values()) and boundary is None:
            raise UnknownEvidence('HELD_CONTEXT_ZERO_SERVICE_WITHOUT_IDENTIFIED_EVENT')
        return SchedulingDecision(rates,boundary)

    def provenance(self):
        oracle=self.oracle.provenance() if hasattr(self.oracle,'provenance') else dict(kind='explicit_context_oracle')
        return dict(kind='held_context_separate_from_pending_work.v12',oracle=oracle,
            held_owner=self.held_owner,held_knowledge=self.held_knowledge,restore_start_ns=self.restore_start_ns,
            review_deadline_ns=self.review_deadline_ns,pending_switch=copy.deepcopy(self.pending_switch.__dict__) if self.pending_switch else None,
            transitions=copy.deepcopy(self.transitions),queries=copy.deepcopy(self.queries),
            host_events_reset_context_age=False,sole_ready_owner_implies_restore=False,
            source_replay=self.allow_source_replay,context_envelope_is_SM_service=False,
            source_replay_is_prediction_validation=False,unknown_fallback=False,checkpoint_supported=False)
