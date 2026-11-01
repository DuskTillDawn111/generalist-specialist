"""Model-only initialization with explicit held or pending-handoff scenarios."""
from dataclasses import dataclass
import copy

from gsdeploy.forecast_errors import UnknownEvidence
from .dag_phase_scheduler_v8 import SchedulingDecision
from .held_context_scheduler_v14 import HeldContextScheduler as V14Scheduler,InitialContextDecision


@dataclass(frozen=True)
class InitialTransitionDecision:
    previous_held_owner: str
    elapsed_gap_ns: int
    next_owner: str
    residual_restore_gap_ns: int
    service_ready_delay_after_restore_ns: int
    source_model_sha256: str
    source_scenario_id: str
    readiness_rule: str
    readiness_upper_after_restore_ns: int | None
    readiness_is_hypothesis: bool


class HeldContextScheduler(V14Scheduler):
    def __init__(self,owners,*,oracle=None):
        super().__init__(owners,oracle=oracle)
        self.initial_transition_decision=None

    def _initialize(self,query):
        initialize=getattr(self.oracle,'initialize',None)
        if not callable(initialize):raise UnknownEvidence('INITIAL_CONTEXT_PREDICTION_UNAVAILABLE')
        if self.initial_decision is not None or self.held_knowledge!='UNKNOWN':
            raise UnknownEvidence('INITIAL_CONTEXT_CANNOT_REPLACE_EXISTING_STATE')
        decision=initialize(query);now=query.clock_ns
        if isinstance(decision,InitialContextDecision):
            if (decision.held_owner not in self.owners or type(decision.context_age_ns) is not int
                    or decision.context_age_ns<0 or type(decision.service_ready_delay_ns) is not int
                    or decision.service_ready_delay_ns<0):
                raise UnknownEvidence('COMPLETE_SOURCE_BOUND_INITIAL_CONTEXT_HYPOTHESIS_REQUIRED')
        elif isinstance(decision,InitialTransitionDecision):
            if (decision.previous_held_owner not in self.owners or decision.next_owner not in self.owners
                    or type(decision.elapsed_gap_ns) is not int or decision.elapsed_gap_ns<0
                    or type(decision.residual_restore_gap_ns) is not int or decision.residual_restore_gap_ns<=0
                    or type(decision.service_ready_delay_after_restore_ns) is not int
                    or decision.service_ready_delay_after_restore_ns<0
                    or decision.readiness_rule not in ('lower_bound_hypothesis','upper_bound_hypothesis','witnessed_point_only')
                    or type(decision.readiness_is_hypothesis) is not bool
                    or not isinstance(decision.source_scenario_id,str) or not decision.source_scenario_id
                    or (decision.readiness_upper_after_restore_ns is not None
                        and (type(decision.readiness_upper_after_restore_ns) is not int
                             or decision.readiness_upper_after_restore_ns<decision.service_ready_delay_after_restore_ns))):
                raise UnknownEvidence('COMPLETE_SOURCE_BOUND_INITIAL_TRANSITION_HYPOTHESIS_REQUIRED')
        else:raise UnknownEvidence('EXPLICIT_HELD_OR_IN_TRANSITION_INITIAL_DECISION_REQUIRED')
        identity=decision.source_model_sha256
        if (not isinstance(identity,str) or len(identity)!=64
                or any(c not in '0123456789abcdef' for c in identity)):
            raise UnknownEvidence('COMPLETE_SOURCE_BOUND_INITIAL_CONTEXT_HYPOTHESIS_REQUIRED')
        self.initial_decision=copy.deepcopy(decision.__dict__)
        if isinstance(decision,InitialTransitionDecision):
            self.initial_transition_decision=copy.deepcopy(decision.__dict__)
            # No invented current holder, historical RESTORE, or current age.
            self.held_knowledge='MODELLED_INITIAL_IN_TRANSITION_HYPOTHESIS'
            restore=now+decision.residual_restore_gap_ns
            service=restore+decision.service_ready_delay_after_restore_ns
            self.context_events=[(restore,'RESTORE_START',decision.next_owner),
                                 (service,'SERVICE_READY',decision.next_owner)]
            self.initial_service_at_ns=service
            return
        self._event(now-decision.context_age_ns,'RESTORE_START',decision.held_owner,
                    'MODELLED_INITIAL_STATE_NOT_OBSERVED_HISTORY')
        service=now+decision.service_ready_delay_ns
        if service==now:
            self._event(now,'SERVICE_READY',decision.held_owner,
                        'MODELLED_INITIAL_REFERENCE_AVAILABILITY_HYPOTHESIS')
        else:
            self.initial_service_at_ns=service
            self.context_events=[(service,'SERVICE_READY',decision.held_owner)]

    def decide(self,view):
        query=self.snapshot(view)
        if self.held_knowledge=='UNKNOWN' and view.active_gpu_nodes:self._initialize(query)
        if self.initial_transition_decision is not None and not self.service_available and self.context_events:
            future=[t for t,_,_ in self.context_events if t>view.clock_ns]
            if not future:raise UnknownEvidence('INITIAL_TRANSITION_REQUIRES_FUTURE_RESTORE_OR_SERVICE')
            return SchedulingDecision({n.key:0. for n in view.active_gpu_nodes},min(future))
        return super().decide(view)

    def provenance(self):
        result=super().provenance()
        result.update(kind='held_or_initial_transition_statistical_candidate.v15',
            initial_transition_decision=copy.deepcopy(self.initial_transition_decision),
            initial_gap_is_physical_no_execution=False,initial_gap_has_no_fabricated_held_owner=True,
            ordinary_source_replay_enabled=False)
        return result
