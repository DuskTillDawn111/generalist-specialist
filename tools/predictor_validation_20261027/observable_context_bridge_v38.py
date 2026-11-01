"""Explicit unvalidated reference-service bridge for native physical candidates.

The physical mark retains every resident class. An independently visible
engine-owned-work domain supplies a separate conditional reference-service
choice; it is a hypothesis, never a measured physical context/SM equivalence.
"""
from __future__ import annotations

import copy

from gsdeploy.forecast_errors import UnknownEvidence
from .owner_lifecycle_mark_v21 import state_features
from .whole57_observable_context_v34 import ObservableContextCandidate
from .whole57_shared_observable_risk_v31 import OWNERS, shared_features


def feature_projection(held_owner, previous_held_owner, context_age_ns):
    # Native stage status and engine handler status are not interchangeable.
    # The v34 fit deliberately discards these diagnostic-only nulls.
    return shared_features(held_owner, previous_held_owner, context_age_ns, {o:None for o in OWNERS})


class ObservableSaveHazard:
    def __init__(self, model):
        self.candidate = ObservableContextCandidate(model)

    def query(self, held_owner, pending_frontier, owner_lifecycle, context_age_ns, horizon_ns=0):
        state_features(owner_lifecycle)
        return self.candidate.save_hazard(feature_projection(held_owner,None,context_age_ns),horizon_ns)


class ObservableReferenceMark:
    def __init__(self, model, *, allow_unvalidated_active_service_bridge=False):
        if allow_unvalidated_active_service_bridge is not True:
            raise ValueError('Physical context labels do not identify active GPU service')
        self.candidate = ObservableContextCandidate(copy.deepcopy(model))
        self.states = None
        self.clock = None

    def observe_view(self, view):
        states = dict(view.owner_lifecycle)
        state_features(states)
        if self.clock is not None and view.clock_ns < self.clock:
            raise UnknownEvidence('MONOTONE_SETTLED_LIFECYCLE_REQUIRED')
        self.states,self.clock = states,view.clock_ns

    def predict_given_save_now(self, *, held_owner, context_age_ns, pending_frontier,
            previous_held_owner=None, public_workload=None,
            pending_age_ns_by_owner=None, pending_reference_work_ns_by_owner=None):
        if self.states is None:
            raise UnknownEvidence('NO_SETTLED_OWNER_WORK_DOMAIN')
        value = self.candidate.next_owner(feature_projection(held_owner,previous_held_owner,context_age_ns))
        eligibility = {owner:self.states[owner].active_service_eligible for owner in OWNERS}
        if any(v is None for v in eligibility.values()):
            raise UnknownEvidence('UNKNOWN_OWNER_WORK_DOMAIN')
        raw=value['physical_owner_probabilities']
        mass=sum(raw[owner] for owner in OWNERS if eligibility[owner])
        if mass <= 0:
            raise UnknownEvidence('NO_VISIBLE_ACTIVE_REFERENCE_TARGET')
        active={owner:raw[owner]/mass if eligibility[owner] else 0. for owner in OWNERS}
        chosen=max(active,key=active.get)
        return dict(most_likely_next_owner=chosen,next_owner_probabilities=active,
            physical_owner_probabilities=raw,physical_context_probabilities_filtered=False,
            physical_most_likely_next_owner=value['most_likely_physical_owner'],
            active_reference_probabilities=active,visible_owner_work_domain=eligibility,
            ood_flags=value['OOD_reasons'],source_clock_ns=self.clock,
            model_sha256=self.candidate.model['model_sha256'],
            physical_context_equals_active_service=False,
            active_reference_service_bridge_is_unvalidated_hypothesis=True,
            same_holder_probability_removed=False,same_holder_reacquisition_support_invented=False,
            future_public_workload_used_as_mark_feature=False,
            handler_activity_used_as_physical_model_feature=False)
