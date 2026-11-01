"""Compose explicit public plans into an unchanged-policy v5 CPU forecast.

The complete public trace, response plans, rates, protocol/transport draws and
initial stations are supplied, not learned here. This entry point preserves the
original fixed drain and all requests, then composes the existing score and queue
adapters. It supplies no native, physical, risk or final-gain qualification.
"""
from collections import defaultdict
from fractions import Fraction

from gsdeploy.forecast_errors import UnknownEvidence
from .actual_worker_twoFIFO_query_adapter_v5 import canonical_requests, clock
from .actual_worker_twoFIFO_reference_v5 import simulate
from .actual_worker_world_score_v2 import score_world
from .actual_twoFIFO_queue_diagnostics_v2 import diagnose, WORK_UNITS
from .macro_response_queue_reference_v1 import encode, rational, validate_plan

SCHEMA = 'gsdeploy.actual_worker_world_forecast.v3'


def public_plan_work(requests, plans):
    """Sum supplied macro coordinates by public arrival ordinal, before running.

    Ordinal order is scheduled arrival plus input position for ties, exactly as
    v5. Enqueue/submit order may subsequently differ without changing plan draws.
    No execution, generated exposure or historical request-ID lookup is accepted.
    """
    canonical_requests(requests)
    counts = defaultdict(int)
    ordered = sorted(enumerate(requests), key=lambda pair: (clock(pair[1]['arrival_ns']), pair[0]))
    for _, row in ordered:
        counts[(row['owner'], row['task'], row['pool_id'])] += 1
    for key, count in counts.items():
        if key not in plans:
            raise UnknownEvidence('UNKNOWN_PUBLIC_INPUT_RESPONSE_TEMPLATE')
        supplied = plans[key]
        if type(supplied) is tuple and len(supplied) != count:
            raise UnknownEvidence('EXPLICIT_ONE_JOINT_DRAW_PER_OFFERED_PUBLIC_KEY_ORDINAL_REQUIRED')
    ordinals = defaultdict(int); costs = {}; details = []
    for _, row in ordered:
        key = row['owner'], row['task'], row['pool_id']
        ordinal = ordinals[key]; ordinals[key] += 1
        supplied = plans[key]
        plan = supplied[ordinal] if type(supplied) is tuple else supplied
        validate_plan(plan)
        full = sum((rational(macro.reference_work_ns, positive=True) for macro in plan.macros), Fraction(0))
        costs[(*key, ordinal)] = full
        details.append(dict(owner=key[0], task=key[1], pool_id=key[2],
            public_key_arrival_ordinal=ordinal, response_reference_work_ns=full,
            macros=[dict(macro_id=macro.identity, actual_phase=macro.phase,
                         response_reference_work_ns=rational(macro.reference_work_ns, positive=True))
                    for macro in plan.macros]))
    return costs, details


def forecast(requests, plans, rates, scoring_plan, quality, *,
             window_start_ns, window_end_ns, drain_end_ns,
             protocol_slacks, transport_draws, initial_sender_available_ns,
             initial_worker_read_release_ns, initial_protocol_state, continuation_state=None,
             producer_fifo=(), resource_provider=None, target_context=None,
             resource_source_evidence=None, resource_memory_initial_state=None,
             scales_ns=(100_000_000, 1_000_000_000, 10_000_000_000)):
    """Return one full-drain CPU world, original score and public queue view.

    Resources are optional; missing support remains UNKNOWN in the unchanged
    scoring adapter. Resource execution binding remains caller declared. Original
    scoring requires integer public arrivals/window clocks; response coordinates
    and simulated times can be rational, with bridge rounding explicitly retained.
    There is deliberately no truncated cutoff, admission-cap or work-cost override.
    """
    requests = [dict(row) for row in requests]
    plans = dict(plans); rates = dict(rates)
    # Work is derived before the generated world exists, including all future
    # requests. The original plans are the only source of these reference costs.
    costs, work_details = public_plan_work(requests, plans)
    boundaries = dict(window_start_ns=window_start_ns, window_end_ns=window_end_ns,
                      drain_end_ns=drain_end_ns)
    world = simulate(requests, plans, rates, **boundaries, cutoff_ns=drain_end_ns,
        producer_fifo=producer_fifo, protocol_slacks=protocol_slacks,
        transport_draws=transport_draws, initial_sender_available_ns=initial_sender_available_ns,
        initial_worker_read_release_ns=initial_worker_read_release_ns,
        initial_protocol_state=initial_protocol_state, continuation_state=continuation_state)
    scored = score_world(world, requests, scoring_plan, quality, **boundaries,
        resource_provider=resource_provider, target_context=target_context,
        resource_source_evidence=resource_source_evidence,
        resource_memory_initial_state=resource_memory_initial_state)
    queues = diagnose(world, requests, **boundaries, scales_ns=scales_ns,
        work_by_public_key_ordinal=costs)
    return dict(schema=SCHEMA, status='EXPLICIT_PUBLIC_PLAN_CPU_FORECAST_ONLY',
        world=world, scoring=scored, queue_diagnostics=queues,
        supplied_plan_reference_work=encode(work_details), work_units=WORK_UNITS,
        work_source='sum_of_explicit_plan_macros_before_simulation_by_public_arrival_ordinal',
        query_fingerprint_sha256=scored['query_fingerprint_sha256'],
        public_offered_requests=len(requests), original_policy_unchanged=True,
        execution_policy='parent_sender_FIFO_and_actual_worker_stdin_FIFO',
        no_global_request_admission_cap=world['global_admission_cap'] is None,
        full_original_drain_and_public_cohort_preserved=True,
        resource_execution_binding_is_caller_declared=True,
        native_current_visible_risk_features=None, physical_resource_demand=None,
        native_assimilation_eligible=False, native_history_assimilation_implemented=False,
        main_queue_provider_eligible=False,
        resource_provider_qualified=False, internal_work_template_qualified=False,
        busy_transport_transfer_qualified=False, transport_decomposition_fitted=False,
        calibrated_long_term_risk=False, stability_qualified=False,
        original_stability_evaluated=False, stability_rule_changed=False,
        independent_prediction_accuracy_validated=False, final_gain_eligible=False,
        validation_labels_read=False, GPU_started=False)
