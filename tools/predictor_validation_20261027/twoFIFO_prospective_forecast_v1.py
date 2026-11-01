"""Pure full-calendar adapter for fresh runs of the original public inputs.

The compiled entry propagates a supplied finite model; it authenticates neither
native templates nor transfer to a different execution context. The donor-law
entry keeps unqualified prospective transfer UNKNOWN. Neither entry reads
target outcomes, fits, launches, projects ensemble P95 to a gate point or seals.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from . import twoFIFO_prospective_contract_v1 as public
from . import twoFIFO_forecast_ensemble_v1 as ensemble
from . import twoFIFO_generated_wholeblock_evaluation_v1 as generated
from . import twoFIFO_queue_envelope_companion_v1 as envelope
from . import twoFIFO_settled_history_initial_v1 as history
from . import twoFIFO_candidate_ranking_v1 as ranking
from .actual_worker_twoFIFO_reference_v5_2 import decode
from .macro_response_queue_reference_v1 import encode

SCHEMA = 'gsdeploy.twoFIFO_prospective_full_calendar_forecast.v1'
need = public.need


def public_calendars(plan, objective, *, request_id_map, forecast_origin_ns):
    """Only IDs and the clock origin change; inputs, order and routes do not."""
    contract = public.public_contract(plan, objective)
    ranking._no_caps(contract)
    need(type(forecast_origin_ns) is int and forecast_origin_ns >= 0,
         'Explicit nonnegative integer forecast origin required')
    old = [r['request_id'] for r in contract['trace']]
    need(type(request_id_map) is dict and set(request_id_map) == set(old)
         and all(type(v) is str and v for v in request_id_map.values())
         and len(set(request_id_map.values())) == len(old)
         and not (set(old) & set(request_id_map.values())),
         'Complete bijective fresh request identity map required')
    calendars = {}
    for candidate in contract['candidates']:
        calendar = public.candidate_calendar(contract, candidate)
        calendar['public_requests'] = [dict(row,
            request_id=request_id_map[row['request_id']],
            arrival_ns=forecast_origin_ns+row['arrival_ns'])
            for row in calendar['public_requests']]
        calendar['original_window'] = [forecast_origin_ns+x for x in calendar['original_window']]
        calendar['original_drain_end_ns'] += forecast_origin_ns
        calendar['queue_sample_times_ns'] = [forecast_origin_ns+x
            for x in contract['queue_sample_offsets_ns']]
        calendars[candidate['id']] = calendar
    return contract, calendars


def _catalog(draws):
    # Validate before scientific UNKNOWN replacement; contract errors must raise.
    need(type(draws) is list and draws and all(type(d) is dict
         and set(d) == ensemble.DRAW_FIELDS for d in draws), 'Closed finite draw catalog required')
    need(all(type(d['draw_id']) is str and d['draw_id'] and type(d['block_id']) is str
         and d['block_id'] for d in draws)
         and len({d['draw_id'] for d in draws}) == len(draws), 'Distinct explicit draw identities required')
    need(all(type(d['weight']) in (int, Fraction) and d['weight'] > 0 for d in draws)
         and sum((d['weight'] for d in draws), Fraction(0)) == 1,
         'Exact positive full probability one required')
    need(all(type(d['inputs']) is ensemble.UnknownForecastInputs or
         (type(d['inputs']) is dict and set(d['inputs']) == ensemble.INPUT_FIELDS)
         for d in draws), 'Exact joint model inputs or explicit missing evidence required')
    return [(d['draw_id'], d['block_id'], d['weight']) for d in draws]


def _unknown(draws, reason):
    return [dict(d, inputs=(d['inputs'] if type(d['inputs']) is ensemble.UnknownForecastInputs
        else ensemble.UnknownForecastInputs(reason))) for d in draws]


def forecast_compiled(plan, objective, *, request_id_map, forecast_origin_ns,
        draws_by_candidate, quality, settled_seed=None, resources_by_candidate=None):
    """Propagate an explicitly supplied paired model over the entire public flow.

    Missing causal settled state is UNKNOWN, never fabricated empty. Resources
    use the unchanged ensemble shared contract and are not authenticated here.
    Source evidence qualification remains a separate mandatory gate.
    """
    contract, calendars = public_calendars(plan, objective,
        request_id_map=request_id_map, forecast_origin_ns=forecast_origin_ns)
    need(type(draws_by_candidate) is dict and set(draws_by_candidate) == set(calendars),
         'Every original candidate and no substituted candidate required')
    catalogs = [_catalog(draws_by_candidate[c]) for c in calendars]
    need(all(c == catalogs[0] for c in catalogs), 'Same paired full draw catalog required')
    resources_by_candidate = {} if resources_by_candidate is None else resources_by_candidate
    need(type(resources_by_candidate) is dict and set(resources_by_candidate) <= set(calendars),
         'Resource candidate identity mismatch')
    resource_fields = {'resource_provider', 'target_context', 'resource_source_evidence',
                       'resource_memory_initial_state'}
    need(all(type(v) is dict and set(v) <= resource_fields
         for v in resources_by_candidate.values()), 'Only original resource interface accepted')
    seed = None; seed_reason = None
    try:
        if settled_seed is None:
            raise UnknownEvidence('UNKNOWN_MISSING_IMMUTABLE_PRECOHORT_HISTORY')
        seed = history.checked_seed(settled_seed, origin_ns=forecast_origin_ns, owner_producers={})
        history.need(set(seed['owner_producers']) == set(contract['aliases']), 'complete_resident_roster')
        history.need(not (set(seed['prior_request_ids']) & set(request_id_map.values())),
                     'fresh_cohort_contains_prior_request')
    except UnknownEvidence as exc:
        seed_reason = str(exc)
    predictions = {}; bounds = {}
    for cid, calendar in calendars.items():
        draws = draws_by_candidate[cid] if seed_reason is None else _unknown(draws_by_candidate[cid], seed_reason)
        start, end = calendar['original_window']; drain = calendar['original_drain_end_ns']
        boundaries = dict(window_start_ns=start, window_end_ns=end, drain_end_ns=drain)
        predictions[cid] = ensemble.evaluate(calendar['public_requests'], draws,
            scoring_plan=contract, quality=quality, settled_initial_state=seed,
            **boundaries, **resources_by_candidate.get(cid, {}))
        bounds[cid] = envelope.evaluate_compiled(calendar['public_requests'], draws, seed, **boundaries)
        need(predictions[cid]['public_requests'] == bounds[cid]['public_requests']
             and [(d['draw_id'], d['block_id'], d['weight']) for d in predictions[cid]['draws']]
             == [(d['draw_id'], d['block_id'], d['weight']) for d in bounds[cid]['draws']],
             'Point and envelope must retain identical complete cohort and paired worlds')
    ranks = ranking.rank_candidates(predictions,
        utility_tie_tolerance=contract.get('utility_tie_tolerance', 0))
    return encode(dict(schema=SCHEMA, status='MODEL_PROPAGATION_ONLY_HOLD_NOT_PREDICTION_SEAL',
        request_id_map=deepcopy(request_id_map), public_calendars=calendars,
        original_scoring_contract={k:deepcopy(contract[k]) for k in (*ensemble.SCORE_FIELDS, 'aliases')},
        candidates=predictions, queue_envelopes=bounds, paired_candidate_ranking=ranks,
        settled_seed_unknown_reason=seed_reason, settled_seed_source_authenticated=False,
        gate_entry='known_input_gate_v5.evaluate_prediction_gate', point_projection_predeclared=False,
        gate_readiness='HOLD', gate_blockers=[
            'Native exact-input template and changed-context/schedule transfer are not authenticated by this model adapter',
            'Causal initial state and original resource evidence require independent native authentication',
            'Finite-world point projection and fresh execution seal require a separate predeclared protocol',
            'Independent timing/quality/six-term sign/rank validation and original stability gates remain required'],
        all_future_public_arrivals_used=True, original_rates_routes_windows_and_six_terms_retained=True,
        all_draws_and_unknown_mass_retained=True, mean_world_P95_is_not_mixture_P95=True,
        native_source_transfer_authenticated=False, main_queue_provider_eligible=False,
        original_stability_evaluated=False, final_gain_eligible=False,
        predictions_sealed=False, model_fit=False, validation_labels_read=False, GPU_started=False))


def forecast_donor_law(plan, objective, *, request_id_map, forecast_origin_ns,
        quality, joint_law=None, donor_blocks=None, settled_seed=None):
    """Safe native-law readiness path; current laws do not qualify fresh transfer.

    No current verifier authenticates prospective template transfer. A true
    caller Boolean is consequently rejected, not treated as a qualification.
    The numeric compiler remains available for explicit model-level studies.
    """
    _, calendars = public_calendars(plan, objective,
        request_id_map=request_id_map, forecast_origin_ns=forecast_origin_ns)
    if joint_law is None:
        blocks = donor_blocks; reason = 'UNKNOWN_MISSING_QUALIFIED_NATIVE_DONOR_LAW'
    else:
        need(joint_law.get('schema') == generated.joint.SCHEMA, 'Original whole-request native law required')
        blocks = joint_law['blocks']
        need(donor_blocks is None or donor_blocks == blocks, 'Donor catalog differs from native law')
        need(joint_law.get('heldout_block_id') not in blocks, 'Heldout block cannot become donor')
        need(joint_law.get('prospective_template_transfer_qualified') is False,
             'No authenticated prospective-transfer verifier exists; caller qualification cannot release this gate')
        reason = 'UNKNOWN_UNQUALIFIED_PROSPECTIVE_NATIVE_TEMPLATE_AND_TRANSPORT_TRANSFER'
    draws = generated.unknown_draws(blocks, reason)
    return forecast_compiled(plan, objective, request_id_map=request_id_map,
        forecast_origin_ns=forecast_origin_ns, quality=quality, settled_seed=settled_seed,
        draws_by_candidate={cid:deepcopy(draws) for cid in calendars})


def prepare_cpu(public_plan_path, objective_path, *, donor_blocks, output):
    """Publish replayable public-only readiness, not an execution/prediction seal."""
    from .implementation_source_closure_v27 import implementation_sources
    refs = dict(public_plan=public.receipt(public_plan_path), objective=public.receipt(objective_path))
    plan, objective = (public.checked(refs[k]) for k in ('public_plan', 'objective'))
    identity = {r['request_id']:'cpu-draft:'+r['request_id'] for r in plan['trace']}
    forecast = forecast_donor_law(plan, objective, request_id_map=identity,
        forecast_origin_ns=0, donor_blocks=donor_blocks, quality={})
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    forecast_ref = public.save(out/'forecast-readiness.json', forecast)
    report = dict(schema=SCHEMA+'.cpu_preparation',
        status='PUBLIC_ONLY_CPU_READINESS_NOT_NATIVE_PREDICTION_SEAL', inputs=refs,
        donor_blocks=list(donor_blocks), forecast=forecast_ref,
        sources=implementation_sources(__file__),
        quality_evidence_supplied=False, initial_state_evidence_supplied=False,
        native_law_evidence_supplied=False, draft_request_ids_are_execution_seal=False,
        independent_prediction_accuracy_validated=False, main_queue_provider_eligible=False,
        final_gain_eligible=False, predictions_sealed=False, model_fit=False,
        validation_labels_read=False, GPU_started=False)
    public.save(out/'report.json', report)
    audit_cpu(out/'report.json')
    return report


def audit_cpu(report_path):
    """Re-derive readiness solely from original public plan/objective bytes."""
    from .implementation_source_closure_v27 import implementation_sources
    report = public.checked(public.receipt(report_path))
    need(report.get('schema') == SCHEMA+'.cpu_preparation' and report.get('status') ==
         'PUBLIC_ONLY_CPU_READINESS_NOT_NATIVE_PREDICTION_SEAL', 'CPU readiness report required')
    need(report.get('sources') == implementation_sources(__file__), 'Adapter source closure changed')
    for ref in report['sources']:
        need(public.receipt(ref['path']) == ref, 'Referenced adapter source changed')
    plan, objective = (public.checked(report['inputs'][k]) for k in ('public_plan', 'objective'))
    identity = {r['request_id']:'cpu-draft:'+r['request_id'] for r in plan['trace']}
    expected = forecast_donor_law(plan, objective, request_id_map=identity,
        forecast_origin_ns=0, donor_blocks=report['donor_blocks'], quality={})
    need(public.checked(report['forecast']) == expected, 'Public readiness changed')
    for key in ('quality_evidence_supplied', 'initial_state_evidence_supplied',
            'native_law_evidence_supplied', 'draft_request_ids_are_execution_seal',
            'independent_prediction_accuracy_validated', 'main_queue_provider_eligible',
            'final_gain_eligible', 'predictions_sealed', 'model_fit', 'validation_labels_read', 'GPU_started'):
        need(report.get(key) is False, 'CPU readiness cannot claim evidence or execution')
    return report
