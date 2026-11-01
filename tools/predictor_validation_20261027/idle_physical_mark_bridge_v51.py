"""Separate all-idle physical transitions from active reference-service marks.

This candidate only extends the completely known, empty owned-work state.
It keeps the trained physical probabilities, inherited same-holder rejection,
hazard clock and handoff uncertainty. It does not identify physical SM service.
"""
from __future__ import annotations

from gsdeploy.forecast_errors import UnknownEvidence
from .owner_lifecycle_mark_v21 import FIELDS, state_features
from .projected_handler_bridge_v44 import (
    ObservableReferenceMark as ActiveReferenceMark, ObservableSaveHazard,
    feature_projection)
from .whole57_shared_observable_risk_v31 import OWNERS


class ObservableReferenceMark(ActiveReferenceMark):
    def predict_given_save_now(self, *, held_owner, context_age_ns, pending_frontier,
            previous_held_owner=None, public_workload=None,
            pending_age_ns_by_owner=None, pending_reference_work_ns_by_owner=None):
        if self.states is None:
            raise UnknownEvidence('NO_SETTLED_OWNER_WORK_DOMAIN')
        state_features(self.states)
        all_idle=all(getattr(self.states[owner],field) is False
            for owner in OWNERS for field in FIELDS)
        if not all_idle:
            return super().predict_given_save_now(held_owner=held_owner,
                context_age_ns=context_age_ns,pending_frontier=pending_frontier,
                previous_held_owner=previous_held_owner,public_workload=public_workload,
                pending_age_ns_by_owner=pending_age_ns_by_owner,
                pending_reference_work_ns_by_owner=pending_reference_work_ns_by_owner)
        if pending_frontier:
            raise UnknownEvidence('ALL_IDLE_LIFECYCLE_CONFLICTS_WITH_GPU_FRONTIER')
        value=self.candidate.next_owner(feature_projection(held_owner,
            previous_held_owner,context_age_ns,self.states))
        raw=dict(value['physical_owner_probabilities'])
        # This is the physical categorical point hypothesis, not a service
        # choice. No missing/idle class is removed or renormalized away.
        target=value['most_likely_physical_owner']
        return dict(most_likely_next_owner=target,next_owner_probabilities=raw,
            physical_owner_probabilities=raw,physical_context_probabilities_filtered=False,
            physical_most_likely_next_owner=target,
            active_reference_probabilities=None,active_reference_target=None,
            visible_owner_work_domain={owner:False for owner in OWNERS},
            transition_domain='ALL_KNOWN_IDLE_PHYSICAL_CONTEXT_POINT_HYPOTHESIS',
            physical_idle_transition=True,reference_service_progress_assigned=False,
            ood_flags=[*value['OOD_reasons'],'UNVALIDATED_ALL_IDLE_PHYSICAL_TRANSITION_BRIDGE'],
            source_clock_ns=self.clock,model_sha256=self.candidate.model['model_sha256'],
            physical_context_equals_active_service=False,
            active_reference_service_bridge_is_unvalidated_hypothesis=True,
            same_holder_probability_removed=False,same_holder_reacquisition_support_invented=False,
            future_public_workload_used_as_mark_feature=False,
            handler_activity_used_as_physical_model_feature=True,
            worker_handler_clock_projection_independently_validated=False,
            handoff_service_ready_is_reference_hypothesis=True,
            physical_idle_state_independently_validated=False)
