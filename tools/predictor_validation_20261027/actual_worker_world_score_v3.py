"""Compose a query-bound two-FIFO CPU world, board timeline and original score.

This is a reviewable integration path, not native prediction qualification.
Resource-provider evidence and execution binding are explicit; no old-domain
resource estimates, stability decision or GPU accuracy is inferred here.
"""
from __future__ import annotations

import copy

from .actual_worker_twoFIFO_query_adapter_v5_2 import query_bound_view, clock as public_clock
from .actual_worker_twoFIFO_reference_v5_2 import resource_timeline_view
from .macro_resource_timeline_v1 import timeline, clock
from .macro_six_term_bridge_v1 import whole_score
from .twoFIFO_resource_provider_v3 import digest, predict_timeline


def score_world(reference, public_requests, scoring_plan, quality, *,
                window_start_ns, window_end_ns, drain_end_ns,
                resource_provider=None, target_context=None, resource_source_evidence=None,
                resource_memory_initial_state=None):
    """Score the complete original drain world, retaining unsupported terms.

    The provider must authenticate its own training sources; supplied evidence
    is a caller declaration and does not independently certify native identity.
    Only public requests, explicit generated state and training providers enter.
    """
    bound=query_bound_view(reference,public_requests,window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,drain_end_ns=drain_end_ns)
    start,end=map(clock,bound['original_window']);drain=clock(bound['original_drain_end_ns'])
    if clock(bound['cutoff_ns'])!=drain:
        raise ValueError('Whole-world integration requires the original complete drain cutoff')
    if any(v.denominator!=1 for v in (start,end,drain)):
        raise ValueError('Original integer scoring boundaries required')
    board=timeline(resource_timeline_view(bound),public_requests,
        resident_owners=tuple(scoring_plan['aliases']))
    estimates=resources=None
    if resource_provider is not None:
        if target_context is None or not resource_source_evidence:
            raise ValueError('Explicit resource execution context and source evidence required')
        binding=dict(basis='caller_declared_explicit_reference_execution_binding',
            physical_execution_identity=resource_provider.physical_execution_identity,
            resource_timeline_sha256=digest(board))
        if resource_memory_initial_state is not None:
            binding['memory_initial_state']=copy.deepcopy(resource_memory_initial_state)
        estimates=predict_timeline(board,resource_provider,target_context=target_context,
            execution_binding=binding)
        # Keep startup and the phase-start-to-drain peak separate. The unchanged
        # scorer alone takes their maximum and applies its safety factor once.
        future_peak=(estimates['partial_known_peak_memory_mib']
            if public_clock(estimates['memory_unknown_observation_duration_ns'])==0 else None)
        resources=dict(original_window_ns=[int(start),int(end)],original_drain_end_ns=int(drain),
            resident_owners=list(scoring_plan['aliases']),
            startup_energy_joules=estimates['startup_energy_joules'],
            startup_through_phase_start_owned_peak_mib=estimates['premeasurement_owned_sampled_peak_mib'],
            phase_start_through_drain_owned_peak_mib=future_peak,
            measurement_board_energy_joules=estimates['board_energy_joules'],
            measurement_energy_includes_drain=False,memory_safety_factor_already_applied=False,
            startup_energy_already_amortized=False,source_evidence=copy.deepcopy(resource_source_evidence))
    elif (target_context is not None or resource_source_evidence is not None
          or resource_memory_initial_state is not None):
        raise ValueError('Resource context/evidence cannot imply an absent provider')
    scored=whole_score(bound,public_requests,scoring_plan,quality,resources)
    return dict(schema='gsdeploy.actual_worker_world_original_score.v3',
        query_fingerprint_sha256=bound['query_fingerprint_sha256'],
        public_query_identity_verified=True,resource_timeline=board,
        resource_prediction=estimates,original_score_bridge=scored,
        original_score=copy.deepcopy(scored['score']),
        resource_execution_binding_is_caller_declared=True,
        no_global_request_admission_cap=bound['global_admission_cap'] is None,
        actual_READY_releases_worker=True,RESULT_reader_is_E2E_endpoint=True,
        original_stability_evaluated=False,main_queue_provider_eligible=False,
        independent_prediction_accuracy_validated=False,final_gain_eligible=False,
        validation_labels_read=False,GPU_started=False)
