"""Node-keyed reference-clock scheduling protocol; no physical share ceiling."""
from __future__ import annotations

import copy
import math

from gsdeploy.forecast_errors import UnknownEvidence
from .dag_phase_scheduler_v8 import SchedulingDecision


def validate_decision(view, decision):
    if type(decision) is not SchedulingDecision:
        raise UnknownEvidence('EXPLICIT_NODE_RATE_DECISION_REQUIRED')
    keys = [n.key for n in view.active_gpu_nodes]
    if len(set(keys)) != len(keys):
        raise ValueError('Distinct active node keys required')
    if type(decision.gpu_rates) is not dict or set(decision.gpu_rates) != set(keys):
        raise UnknownEvidence('COMPLETE_EXACT_ACTIVE_NODE_RATE_KEYS_REQUIRED')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0
           for v in decision.gpu_rates.values()):
        raise UnknownEvidence('FINITE_NONNEGATIVE_NODE_REFERENCE_RATE_REQUIRED')
    boundary = decision.next_transition_ns
    if boundary is not None and (type(boundary) is not int or boundary <= view.clock_ns):
        raise UnknownEvidence('STRICTLY_FUTURE_INTEGER_NODE_RATE_TRANSITION_REQUIRED')
    # A zero frontier may be released by another known future arrival/dependency
    # event. The consumer checks the complete public/generated event calendar.
    return dict(decision.gpu_rates), boundary


class CheckpointNodeScheduler:
    """Bind explicit provider configuration separately from its dynamic state.

    Providers implement decide/fresh/checkpoint_configuration/saved_state/
    restore_state/provenance. Deterministic prefix replay validates saved state;
    this protocol does not authenticate native training or make an RNG safe.
    """
    def __init__(self, provider):
        methods = ('decide','fresh','checkpoint_configuration','saved_state','restore_state','provenance')
        if provider is None or not all(callable(getattr(provider,m,None)) for m in methods):
            raise ValueError('Complete explicit node-rate provider checkpoint protocol required')
        self.provider = provider.fresh()
        self.configuration = copy.deepcopy(self.provider.checkpoint_configuration())
        if type(self.configuration) is not dict or not self.configuration:
            raise ValueError('Explicit immutable node-rate configuration required')
        if self.configuration != provider.checkpoint_configuration():
            raise ValueError('Fresh node provider changed configuration')
        self.last_clock_ns = None
        self.decision_count = 0

    @classmethod
    def fresh_from(cls, scheduler):
        if type(scheduler) is not cls or scheduler.decision_count or scheduler.last_clock_ns is not None:
            raise ValueError('Initially empty explicit node-rate scheduler required')
        return cls(scheduler.provider)

    def decide(self, view):
        if self.provider.checkpoint_configuration() != self.configuration:
            raise ValueError('Node-rate provider configuration changed')
        if self.last_clock_ns is not None and view.clock_ns < self.last_clock_ns:
            raise ValueError('Node scheduler clock moved backward')
        result = self.provider.decide(view)
        validate_decision(view, result)
        self.last_clock_ns = view.clock_ns
        self.decision_count += 1
        return result

    def checkpoint_configuration(self):
        return copy.deepcopy(self.configuration)

    def saved_state(self):
        return dict(last_clock_ns=self.last_clock_ns, decision_count=self.decision_count,
                    provider_state=copy.deepcopy(self.provider.saved_state()))

    def restore_state(self, state):
        if (type(state) is not dict or set(state) != {'last_clock_ns','decision_count','provider_state'}
                or type(state['decision_count']) is not int or state['decision_count'] < 0
                or state['last_clock_ns'] is not None and
                    (type(state['last_clock_ns']) is not int or state['last_clock_ns'] < 0)):
            raise ValueError('Complete valid node scheduler state required')
        provider = self.provider.fresh()
        provider.restore_state(copy.deepcopy(state['provider_state']))
        if (provider.saved_state() != state['provider_state']
                or provider.checkpoint_configuration() != self.configuration):
            raise ValueError('Restored node-rate provider changed configuration/state')
        self.provider = provider
        self.last_clock_ns = state['last_clock_ns']
        self.decision_count = state['decision_count']

    def provenance(self):
        return dict(kind='NODE_KEYED_DYNAMIC_REFERENCE_CLOCK_NOT_SM_SHARE',
            provider=copy.deepcopy(self.provider.provenance()),
            decision_count=self.decision_count, rates_may_exceed_one=True,
            checkpoint_supported=True, native_sources_authenticated=False,
            physical_accuracy_validated=False, deployment_qualified=False)
