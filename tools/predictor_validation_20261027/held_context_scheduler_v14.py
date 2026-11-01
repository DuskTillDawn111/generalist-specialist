"""An explicit learned initial-state hypothesis for the held-context engine.

The ordinary source-replay interface stays disabled. Initialization is a model
decision at the first simulated GPU frontier, not an observed RESTORE. A model
must supply the joint holder/age/readiness hypothesis and its frozen identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy

from gsdeploy.forecast_errors import UnknownEvidence
from .dag_phase_scheduler_v8 import SchedulingDecision
from .held_context_scheduler_v12_1 import HeldContextScheduler as BaseScheduler


@dataclass(frozen=True)
class InitialContextDecision:
    held_owner: str
    context_age_ns: int
    service_ready_delay_ns: int
    source_model_sha256: str


class HeldContextScheduler(BaseScheduler):
    def __init__(self, owners, *, oracle=None):
        super().__init__(owners, oracle=oracle, allow_source_replay=False)
        self.initial_decision = None
        self.initial_service_at_ns = None

    def _initialize(self, query):
        initialize = getattr(self.oracle, 'initialize', None)
        if not callable(initialize):
            raise UnknownEvidence('INITIAL_CONTEXT_PREDICTION_UNAVAILABLE')
        decision = initialize(query)
        if (not isinstance(decision, InitialContextDecision)
                or decision.held_owner not in self.owners
                or type(decision.context_age_ns) is not int or decision.context_age_ns < 0
                or type(decision.service_ready_delay_ns) is not int or decision.service_ready_delay_ns < 0
                or not isinstance(decision.source_model_sha256, str)
                or len(decision.source_model_sha256) != 64
                or any(c not in '0123456789abcdef' for c in decision.source_model_sha256)):
            raise UnknownEvidence('COMPLETE_SOURCE_BOUND_INITIAL_CONTEXT_HYPOTHESIS_REQUIRED')
        if self.initial_decision is not None or self.held_knowledge != 'UNKNOWN':
            raise UnknownEvidence('INITIAL_CONTEXT_CANNOT_REPLACE_EXISTING_STATE')
        now = query.clock_ns
        self.initial_decision = copy.deepcopy(decision.__dict__)
        self._event(now-decision.context_age_ns, 'RESTORE_START', decision.held_owner,
                    'MODELLED_INITIAL_STATE_NOT_OBSERVED_HISTORY')
        service = now+decision.service_ready_delay_ns
        if service == now:
            self._event(now, 'SERVICE_READY', decision.held_owner,
                        'MODELLED_INITIAL_REFERENCE_AVAILABILITY_HYPOTHESIS')
        else:
            self.initial_service_at_ns = service
            self.context_events = [(service, 'SERVICE_READY', decision.held_owner)]

    def decide(self, view):
        query = self.snapshot(view)
        if self.held_knowledge == 'UNKNOWN' and view.active_gpu_nodes:
            self._initialize(query)
        if self.initial_service_at_ns is not None:
            if view.clock_ns < self.initial_service_at_ns:
                return SchedulingDecision({n.key:0. for n in view.active_gpu_nodes},
                                          self.initial_service_at_ns)
            self.initial_service_at_ns = None
        decision = super().decide(view)
        # Pending context transitions continue even through an empty GPU-ready
        # set. They cannot be deferred to an unrelated future host arrival.
        future = [stamp for stamp,_,_ in self.context_events if stamp > view.clock_ns]
        if self.review_deadline_ns is not None and self.review_deadline_ns > view.clock_ns:
            future.append(self.review_deadline_ns)
        if decision.next_transition_ns is not None:
            future.append(decision.next_transition_ns)
        return SchedulingDecision(decision.gpu_rates, min(future) if future else None)

    def provenance(self):
        result = super().provenance()
        result.update(kind='held_context_with_explicit_initial_state_candidate.v14',
            initial_decision=copy.deepcopy(self.initial_decision),
            initial_service_at_ns=self.initial_service_at_ns,
            initial_restore_is_observed_history=False,
            initial_readiness_is_physical_measurement=False,
            initialization_entrypoint_available=callable(getattr(self.oracle, 'initialize', None)),
            complete_source_bound_oracle_qualified=False,
            checkpoint_supported=False)
        return result
