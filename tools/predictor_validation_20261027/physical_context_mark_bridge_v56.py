"""Use physical transition probabilities in their original resident domain.

SAVE/RESTORE labels identify a physical context, not a request with active
owned work. Conditioning those labels on current service eligibility changes
the statistical target and can turn tiny held-owner mass into probability one.
This bridge preserves every resident probability at every lifecycle state.
The inherited scheduler still rejects unsupported same-holder reacquisition.
"""
from __future__ import annotations

from gsdeploy.forecast_errors import UnknownEvidence
from .owner_lifecycle_mark_v21 import state_features
from .projected_handler_bridge_v44 import ObservableReferenceMark as V44Mark, feature_projection
from .whole57_shared_observable_risk_v31 import OWNERS


class PhysicalContextMark(V44Mark):
    def predict_given_save_now(self, *, held_owner, context_age_ns, pending_frontier,
            previous_held_owner=None, public_workload=None,
            pending_age_ns_by_owner=None, pending_reference_work_ns_by_owner=None):
        if self.states is None:
            raise UnknownEvidence('NO_SETTLED_OWNER_WORK_DOMAIN')
        state_features(self.states)
        value = self.candidate.next_owner(feature_projection(held_owner,
            previous_held_owner, context_age_ns, self.states))
        raw = dict(value['physical_owner_probabilities'])
        target = value['most_likely_physical_owner']
        # Eligibility is diagnostic only. It neither filters a resident class
        # nor grants progress to an absent request. The event engine applies
        # its separate active-node/reference-service hypothesis after RESTORE.
        eligibility = {owner:self.states[owner].active_service_eligible for owner in OWNERS}
        return dict(most_likely_next_owner=target, next_owner_probabilities=raw,
            physical_owner_probabilities=raw, physical_context_probabilities_filtered=False,
            physical_most_likely_next_owner=target,
            active_reference_probabilities=None, active_reference_target=None,
            visible_owner_work_domain=eligibility,
            transition_domain='UNCONDITIONED_RESIDENT_PHYSICAL_CONTEXT_POINT_HYPOTHESIS',
            physical_target_currently_service_eligible=eligibility[target],
            reference_service_progress_assigned=False,
            ood_flags=[*value['OOD_reasons'],
                'UNVALIDATED_PHYSICAL_CONTEXT_TO_REFERENCE_SERVICE_ENGINE_HYPOTHESIS'],
            source_clock_ns=self.clock, model_sha256=self.candidate.model['model_sha256'],
            physical_context_equals_active_service=False,
            active_reference_service_bridge_is_unvalidated_hypothesis=True,
            same_holder_probability_removed=False,
            same_holder_reacquisition_support_invented=False,
            future_public_workload_used_as_mark_feature=False,
            handler_activity_used_as_physical_model_feature=True,
            worker_handler_clock_projection_independently_validated=False,
            handoff_service_ready_is_reference_hypothesis=True,
            physical_idle_state_independently_validated=False)
