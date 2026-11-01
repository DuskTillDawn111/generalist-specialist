"""Attach conservative queue bounds to the frozen full-cohort case forecast.

The same public compiler, donor slots and settled prefix feed both paths.
Bounds are conditional on the finite-rate model, not measured GPU guarantees.
No latent point, probability law, resource value or utility is manufactured.
"""
from __future__ import annotations

import copy
from fractions import Fraction

from gsdeploy.forecast_errors import UnknownEvidence
from . import twoFIFO_generated_wholeblock_evaluation_v1 as generated
from . import twoFIFO_forecast_ensemble_v1 as ensemble
from . import twoFIFO_multiowner_outer_envelope_v1 as outer
from . import twoFIFO_settled_history_initial_v1 as history
from .actual_worker_twoFIFO_query_adapter_v5_2 import canonical_requests, clock
from .actual_worker_twoFIFO_reference_v5_2 import decode
from .macro_response_queue_reference_v1 import encode

SCHEMA = 'gsdeploy.twoFIFO_queue_envelope_companion.v1'
require = generated.require


def _unknown(requests, reason):
    return dict(status='UNKNOWN_COMPANION_QUEUE_ENVELOPE', reason=reason,
        offered_requests=len(requests), offered_request_ids=[r['request_id'] for r in requests],
        requests=[dict(request_id=r['request_id'], reader_E2E_ns_bounds=None,
            fixed_drain_incomplete_indicator_bounds=[0, 1]) for r in requests],
        complete_cohort_P95_ns_bounds=None)


def _stations(seed, requests, inputs, origin):
    """Project idle settled producers only; retain the global collector clocks."""
    seed = history.checked_seed(seed, origin_ns=origin, owner_producers={})
    active = {r['owner'] for r in requests}
    history.need(active <= set(seed['owner_producers']), 'case_owner_outside_settled_roster')
    history.need(not ({r['request_id'] for r in requests} & set(seed['prior_request_ids'])),
                 'current_cohort_contains_prior_request')
    producers = {seed['owner_producers'][o] for o in active}
    for key, supplied in inputs['plans'].items():
        for plan in supplied if type(supplied) is tuple else (supplied,):
            history.need(key[0] in active and plan.ready_producer_id == seed['owner_producers'][key[0]],
                         'case_plan_changed_settled_stdout_lifetime')
    protocol = copy.deepcopy(seed['initial_protocol_state'])
    # Every excluded producer is settled, has no pending message, and receives
    # no request in the complete public future. Its reader clock cannot affect
    # an active producer. Global collector availability is not projected away.
    protocol['reader_free'] = {p: at for p, at in protocol['reader_free'].items() if p in producers}
    return dict(initial_sender_available_ns={o: seed['initial_sender_available_ns'][o] for o in active},
        initial_worker_read_release_ns={o: seed['initial_worker_read_release_ns'][o] for o in active},
        initial_protocol_state=protocol)


def evaluate_compiled(requests, draws, settled_seed, *, window_start_ns, window_end_ns, drain_end_ns):
    """Keep every original draw and request even when an envelope is unsupported.

    This is a model-level interface, not an authentication of training evidence.
    Missing scientific support is UNKNOWN. Invalid contracts propagate errors.
    """
    requests = [dict(r) for r in requests]
    public = canonical_requests(requests)
    start, end, drain = map(clock, (window_start_ns, window_end_ns, drain_end_ns))
    require(requests and start < end <= drain and all(x.denominator == 1 for x in (start, end, drain)),
            'Nonempty complete cohort and original integer boundaries required')
    require(all(type(r['arrival_ns']) is int and start <= r['arrival_ns'] < end for r in requests),
            'Every public arrival must belong to the original measurement window')
    require(type(draws) is list and draws and all(type(d) is dict and set(d) == ensemble.DRAW_FIELDS for d in draws),
            'Exact original compiled draw catalog required')
    require(all(type(d['draw_id']) is str and d['draw_id'] and type(d['block_id']) is str and d['block_id'] for d in draws)
            and len({d['draw_id'] for d in draws}) == len(draws), 'Distinct draw and block identities required')
    require(all(type(d['weight']) in (int, Fraction) and d['weight'] > 0 for d in draws)
            and sum((d['weight'] for d in draws), Fraction(0)) == 1, 'Exact original total probability one required')
    require(all(type(d['inputs']) is ensemble.UnknownForecastInputs or
        (type(d['inputs']) is dict and set(d['inputs']) == ensemble.INPUT_FIELDS) for d in draws),
        'Only compiled original inputs or explicit missing evidence accepted')
    rows = []
    for draw in draws:
        inputs = draw['inputs']
        try:
            if type(inputs) is ensemble.UnknownForecastInputs:
                raise UnknownEvidence(inputs.reason)
            if settled_seed is None:
                raise UnknownEvidence('UNKNOWN_MISSING_IMMUTABLE_PRECASE_HISTORY')
            stations = _stations(settled_seed, requests, inputs, start)
            result = outer.evaluate(requests, **inputs, **stations,
                domain_certificate=copy.deepcopy(outer.CERTIFICATE),
                window_start_ns=start, window_end_ns=end, drain_end_ns=drain)
        except UnknownEvidence as exc:
            result = _unknown(requests, str(exc))
        rows.append(dict(draw_id=draw['draw_id'], block_id=draw['block_id'], weight=Fraction(draw['weight']),
            offered_requests=len(requests), envelope=result))
    bounded = sum((row['weight'] for row in rows if row['envelope']['status'] ==
        'CONSERVATIVE_FINITE_RATE_FULL_COHORT_OUTER_ENVELOPE'), Fraction(0))
    risks = []
    for index, request in enumerate(requests):
        lo = hi = Fraction(0)
        for row in rows:
            a, b = row['envelope']['requests'][index]['fixed_drain_incomplete_indicator_bounds']
            lo += row['weight'] * a; hi += row['weight'] * b
        risks.append(dict(request_id=request['request_id'], fixed_drain_incomplete_probability_bounds=[lo, hi]))
    return encode(dict(schema=SCHEMA, status='CONDITIONAL_MODEL_QUEUE_ENVELOPES_NOT_POINT_OR_UTILITY',
        public_requests=public, offered_requests=len(requests), draws=rows, total_probability_mass=Fraction(1),
        bounded_world_probability_mass=bounded, unknown_envelope_probability_mass=1-bounded,
        fixed_drain_risk=risks, original_window=[start, end], original_drain_end_ns=drain,
        all_original_draws_and_probability_mass_retained=True, all_offered_requests_retained=True,
        idle_reader_projection_requires_settled_prefix=True, collector_initial_clocks_retained=True,
        domain_basis='complete_public_future_and_validated_settled_seed_under_original_finite_rate_model',
        domain_declaration_is_not_physical_authentication=True,
        latent_point_selected=False, latent_probability_assigned=False, unknown_worlds_renormalized=False,
        point_forecast_or_original_six_terms_replaced=False, full_U_bounds=None, resource_bounds=None,
        expected_world_P95_is_not_mixture_P95=True, physical_rate_bounds_certified=False,
        main_queue_provider_eligible=False, independent_prediction_accuracy_validated=False,
        final_gain_eligible=False, validation_labels_read=False, GPU_started=False))


def forecast_case(case, epoch, snapshot, law, scoring_plan, quality, *, donor_blocks=None, unqualified_reason=None):
    """Use the unchanged case forecast and attach a separate, paired diagnostic."""
    point = generated.forecast_case(case, epoch, snapshot, law, scoring_plan, quality,
        donor_blocks=donor_blocks, unqualified_reason=unqualified_reason)
    requests = generated.public_case(case, epoch)
    reason = unqualified_reason or point['settled_seed_unknown_reason']
    blocks = law['blocks'] if law is not None else donor_blocks
    if reason is not None:
        draws = generated.unknown_draws(blocks, reason)
    else:
        draws, _ = generated.compile_case_draws(requests, law)
    diagnostic = evaluate_compiled(requests, draws, point['settled_seed'],
        window_start_ns=epoch, window_end_ns=epoch+generated.MEASUREMENT_NS,
        drain_end_ns=epoch+generated.MEASUREMENT_NS+generated.DRAIN_NS)
    original = decode(point['ensemble']); bounds = decode(diagnostic)
    require([(r['draw_id'], r['block_id'], r['weight']) for r in original['draws']] ==
            [(r['draw_id'], r['block_id'], r['weight']) for r in bounds['draws']], 'Original paired draw identity changed')
    require(original['public_requests'] == bounds['public_requests'], 'Original complete public cohort changed')
    for src, row in zip(original['draws'], bounds['draws']):
        result = row['envelope']
        if src['status'] != 'GENERATED_WORLD' or result['status'] != 'CONSERVATIVE_FINITE_RATE_FULL_COHORT_OUTER_ENVELOPE':
            continue
        world = src['forecast']['world']
        for request in result['requests']:
            rid = request['request_id']
            for key, field in (('starts', 'START_ns_bounds'), ('READY_occurrences', 'READY_ns_bounds'),
                               ('RESULT_receipts', 'RESULT_reader_ns_bounds')):
                if rid in world[key]:
                    low, high = request[field]
                    require(low <= clock(world[key][rid]) <= high, 'Generated world outside its finite-rate envelope')
    return dict(prediction=point, queue_envelope=diagnostic,
        unchanged_point_prediction_sha256=generated.digest(point),
        same_public_compiler_and_donor_scheme=True, bounds_used_for_utility_or_candidate_selection=False)
