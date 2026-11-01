"""Pure CPU audit of an already generated, paired finite candidate domain.

This companion validates serialized six-term arithmetic, not evidence provenance
or predictor accuracy. It neither generates worlds nor evaluates qualification.
Unknown probability mass prevents a complete-domain winner even when a complete
subset is available. Tolerance groups use their highest-utility anchor, so a
chain of nearby utilities cannot create a non-transitive tie class.
"""
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
import math

from . import twoFIFO_forecast_ensemble_v1 as ensemble
from .actual_worker_twoFIFO_query_adapter_v5_2 import clock
from .macro_response_queue_reference_v1 import encode
from .macro_six_term_bridge_v1 import whole_score

SCHEMA = 'gsdeploy.twoFIFO_candidate_ranking.v1'
TERMS = 'QTLPME'
QUALIFICATION_FIELDS = {'stability_state_v2', 'stability_status', 'rho', 'g(n)',
    'g_n', 'reasons', 'memory_budget_satisfied', 'complete_cohort_drain'}


def need(value, message):
    if not value:
        raise ValueError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _no_caps(value):
    if type(value) is dict:
        for key, item in value.items():
            need(not ((not key.lower().startswith('no_')) and 'cap' in key.lower() and ('admission' in key.lower()
                or 'concurrency' in key.lower() or key.lower() in {'cap1', 'cap2', 'global_cap'})
                and item is not None), 'Admission/concurrency caps are outside this ranking contract')
            _no_caps(item)
    elif type(value) in (list, tuple):
        for item in value:
            _no_caps(item)


def _public_clock(value):
    # canonical_requests deliberately serializes public arrival clocks as [n,d].
    if type(value) is list and len(value) == 2:
        value = dict(numerator=value[0], denominator=value[1])
    return clock(value)


def _scope_score(document, reference, prediction=None):
    """Reuse the original bridge for timing and raw resource scopes, not quality.

    The serialized comparison contract omits resident alias family metadata.
    Synthetic family names deliberately make Q unsupported in this local check;
    the supplied original Q/resource evidence remains the original scorer's job.
    """
    plan = deepcopy(document['score_comparison_contract']['scoring_plan'])
    requests = [dict(row, arrival_ns=int(_public_clock(row['arrival_ns'])))
                for row in document['public_requests']]
    plan['aliases'] = {row['owner']: dict(family='timing_check_only',
        supported_tasks=list(plan['workload_rps'])) for row in requests}
    resources = None
    if prediction is not None:
        fields = {'schema', 'board_energy_joules', 'partial_known_peak_memory_mib',
            'startup_energy_joules', 'premeasurement_owned_sampled_peak_mib',
            'memory_unknown_observation_duration_ns', 'startup_energy_already_amortized',
            'safe_factor_already_applied', 'power_unknown_measurement_duration_ns',
            'total_measurement_duration_ns', 'total_observation_duration_ns'}
        need(type(prediction) is dict and fields <= set(prediction)
             and prediction['schema'] == 'gsdeploy.twoFIFO_multiwindow_resource_prediction.v3'
             and prediction['startup_energy_already_amortized'] is False
             and prediction['safe_factor_already_applied'] is False,
             'Explicit unamortized, unsafened original raw resource prediction required')
        start, end = map(clock, document['original_window'])
        drain = clock(document['original_drain_end_ns'])
        need(clock(prediction['total_measurement_duration_ns']) == end-start
             and clock(prediction['total_observation_duration_ns']) == drain-start
             and clock(prediction['power_unknown_measurement_duration_ns']) <= end-start
             and clock(prediction['memory_unknown_observation_duration_ns']) <= drain-start,
             'Resource prediction must retain original measurement and startup-to-drain scope')
        need(clock(prediction['power_unknown_measurement_duration_ns']) == 0
             or prediction['board_energy_joules'] is None, 'Unknown measurement power cannot become complete E')
        future_peak = (prediction['partial_known_peak_memory_mib']
            if clock(prediction['memory_unknown_observation_duration_ns']) == 0 else None)
        resources = dict(original_window_ns=[int(clock(x)) for x in document['original_window']],
            original_drain_end_ns=int(clock(document['original_drain_end_ns'])),
            resident_owners=list(plan['aliases']), startup_energy_joules=prediction['startup_energy_joules'],
            startup_through_phase_start_owned_peak_mib=prediction['premeasurement_owned_sampled_peak_mib'],
            phase_start_through_drain_owned_peak_mib=future_peak,
            measurement_board_energy_joules=prediction['board_energy_joules'],
            measurement_energy_includes_drain=False, memory_safety_factor_already_applied=False,
            startup_energy_already_amortized=False,
            source_evidence=['Serialized prediction arithmetic check only; provenance not authenticated'])
    return whole_score(reference, requests, plan, {}, resources)['score']


def _validate_score(document, draw):
    if draw['status'] == 'UNKNOWN_WORLD':
        return None, {term: 'UNKNOWN' for term in TERMS}
    forecast = draw['forecast']
    need(forecast.get('schema') == 'gsdeploy.actual_worker_world_forecast.v4',
         'Explicit original full-world forecast schema required')
    world = forecast.get('world', {})
    scoring = forecast.get('scoring', {})
    need(scoring.get('schema') == 'gsdeploy.actual_worker_world_original_score.v3',
         'Explicit original score-world schema required')
    score = scoring.get('original_score', {})
    bridge = scoring.get('original_score_bridge', {})
    need(bridge.get('schema') == 'gsdeploy.macro_world_original_six_term_bridge.v1'
         and bridge.get('original_scorer_reused') is True
         and bridge.get('weights_changed') is False
         and bridge.get('E2E_endpoint') == 'scheduled_arrival_to_RESULT_reader_receipt'
         and bridge.get('score') == score, 'Original six-term score bridge and reader endpoint required')
    public = document['public_requests']
    ids = [row['request_id'] for row in public]
    need(world.get('offered_requests') == len(ids)
         and world.get('original_window') == document['original_window']
         and world.get('original_drain_end_ns') == document['original_drain_end_ns']
         and world.get('cutoff_ns') == document['original_drain_end_ns'],
         'Every generated world must retain original cohort, window and full drain')
    generated = world.get('public_offered_requests')
    need(type(generated) is list and len(generated) == len(public), 'Generated public cohort required')
    need(all({k:v for k,v in a.items() if k != 'arrival_ns'} ==
             {k:v for k,v in b.items() if k != 'arrival_ns'}
             and _public_clock(a['arrival_ns']) == _public_clock(b['arrival_ns'])
             for a,b in zip(public, generated)), 'Generated route/cohort must match candidate')
    receipts = world.get('RESULT_receipts', {})
    unfinished = world.get('unfinished_request_ids', [])
    need(type(receipts) is dict and type(unfinished) is list
         and len(set(unfinished)) == len(unfinished)
         and not set(receipts) & set(unfinished)
         and set(receipts) | set(unfinished) == set(ids), 'Complete offered RESULT/unfinished partition required')
    start, end = map(clock, document['original_window'])
    drain = clock(document['original_drain_end_ns'])
    need(all(_public_clock(row['arrival_ns']) <= clock(receipts[row['request_id']]) <= drain
             for row in public if row['request_id'] in receipts), 'RESULT outside public arrival/drain')
    need(score.get('cohort_size') == len(ids) and score.get('cohort_ids') == sorted(ids)
         and score.get('incomplete_ids') == sorted(unfinished), 'Score cannot delete unfinished/offered requests')
    terms = score.get('terms')
    need(type(terms) is dict and set(terms) == set(TERMS)
         and all(v is None or _finite(v) for v in terms.values()), 'Exactly six finite or UNKNOWN terms required')
    scoped = _scope_score(document, world, scoring.get('resource_prediction'))
    need(all(terms[term] == scoped['terms'][term] for term in 'TLPME'),
         'T/L/P/M/E must use original bridge, full cohort, RESULT receipt and raw resource scope')
    complete = all(value is not None for value in terms.values())
    need(score.get('six_term_comparison_complete') is complete,
         'Six-term completion flag inconsistent with term evidence')
    utility = score.get('full_U')
    cost = score.get('cost')
    expected_cost = sum(terms[k] / 6 for k in TERMS) if complete else None
    if complete:
        need(_finite(cost) and math.isclose(cost, expected_cost, rel_tol=1e-12, abs_tol=1e-12),
             'Original six terms must each retain weight 1/6')
    else:
        need(cost is None, 'Missing terms cannot be renormalized into a conditional cost')
    need(type(score.get('absolute_utility_complete')) is bool, 'Explicit absolute-utility evidence status required')
    if score['absolute_utility_complete']:
        need(complete and _finite(utility) and math.isclose(utility, -cost, rel_tol=1e-12, abs_tol=1e-12),
             'Full U must equal negative original equal-six-term cost')
    else:
        need(utility is None, 'Missing absolute evidence must preserve UNKNOWN full U')
    return utility, {term: 'KNOWN' if terms[term] is not None else 'UNKNOWN' for term in TERMS}


def _validate_document(document):
    _no_caps(document)
    # Existing compare validates serialized mass, statuses and denominators even
    # for all-UNKNOWN domains. Self-comparison does not generate or select draws.
    ensemble.compare(document, document)
    contract = document['score_comparison_contract']
    need(set(contract) == {'scoring_plan', 'quality'} and type(contract['quality']) is dict,
         'Original score and common quality contract required')
    plan = contract['scoring_plan']
    need(type(plan) is dict and set(plan) == set(ensemble.SCORE_FIELDS)
         and plan.get('weights') == {k:1/6 for k in TERMS}
         and plan.get('startup_amortization_seconds') == 3600,
         'Fixed equal-six-term original score contract required')
    start, end = map(clock, document['original_window'])
    drain = clock(document['original_drain_end_ns'])
    need(start < end <= drain and all(x.denominator == 1 for x in (start,end,drain))
         and round(plan['window_seconds'] * 1e9) == end-start, 'Original measurement/drain boundaries required')
    public = document['public_requests']
    need(len({r['request_id'] for r in public}) == len(public), 'Distinct offered public requests required')
    for row in public:
        need(type(row) is dict and set(row) == {'request_id','owner','task','pool_id','arrival_ns'}
             and all(type(row[k]) is str and row[k] for k in ('request_id','owner','task','pool_id'))
             and start <= _public_clock(row['arrival_ns']) < end
             and _public_clock(row['arrival_ns']).denominator == 1, 'Complete public input/cohort required')
    # Validate original scoring scales even if every joint world is UNKNOWN.
    _scope_score(document, dict(original_window=document['original_window'],
        original_drain_end_ns=document['original_drain_end_ns'],
        cutoff_ns=document['original_drain_end_ns'], RESULT_receipts={},
        unfinished_request_ids=[row['request_id'] for row in public], offered_requests=len(public)))
    return [_validate_score(document, draw) for draw in document['draws']]


def _groups(rows, tolerance):
    """Highest-anchor groups; membership never extends through a nearby peer."""
    ordered = sorted(rows, key=lambda row: (-row['complete_utility'], row['candidate_id']))
    groups = []
    for row in ordered:
        if not groups or groups[-1]['anchor_utility'] - row['complete_utility'] > tolerance:
            groups.append(dict(rank=len(groups)+1, anchor_utility=row['complete_utility'], candidate_ids=[]))
        groups[-1]['candidate_ids'].append(row['candidate_id'])
    return groups


def rank_candidates(candidates, *, utility_tie_tolerance=0, qualifications=None):
    """Audit every candidate and compare every paired draw without model fitting.

    Qualifications are caller-declared original gate results, copied verbatim;
    ordinary ranking is not deployment eligibility or a final gain claim.
    """
    need(type(candidates) is dict and candidates
         and all(type(k) is str and k for k in candidates), 'Named nonempty candidate domain required')
    need(_finite(utility_tie_tolerance) and utility_tie_tolerance >= 0, 'Predeclared finite nonnegative tie tolerance required')
    qualifications = {} if qualifications is None else qualifications
    need(type(qualifications) is dict and set(qualifications) <= set(candidates), 'Qualification candidate identity mismatch')
    for qualification in qualifications.values():
        need(type(qualification) is dict and set(qualification) <= QUALIFICATION_FIELDS,
             'Only copied original stability/resource/drain qualification fields accepted')
        _no_caps(qualification)
    rows = []
    for candidate_id, document in candidates.items():
        audit = _validate_document(document)
        known = Fraction(0); values = []; statuses = []; mean = 0.
        for draw, (utility, term_status) in zip(document['draws'], audit):
            weight = clock(draw['weight'])
            identity = dict(draw_id=draw['draw_id'], block_id=draw['block_id'], weight=weight)
            values.append(dict(identity, full_U=utility))
            statuses.append(dict(identity, terms=term_status))
            if utility is not None:
                known += weight; mean += float(weight) * utility
        need(math.isfinite(mean), 'Finite weighted candidate utility required')
        qualification = deepcopy(qualifications.get(candidate_id, {}))
        rows.append(dict(candidate_id=candidate_id, offered_requests=document['offered_requests'],
            draw_count=len(audit), known_utility_probability_mass=known,
            unknown_utility_probability_mass=1-known, per_draw_full_U=values,
            per_draw_term_status=statuses, complete_utility=mean if known == 1 else None,
            conditional_on_known_utility_mean=mean/float(known) if known else None,
            qualification=qualification, stability_status=deepcopy(qualification.get('stability_status')),
            rank_status='COMPLETE_SUBDOMAIN_DIAGNOSTIC' if known == 1 else 'UNORDERED_UNKNOWN'))
    comparisons = []
    for left, right in combinations(candidates, 2):
        comparison = ensemble.compare(candidates[left], candidates[right])
        unknown = clock(comparison['unknown_delta_probability_mass'])
        deltas = [draw['delta_U'] for draw in comparison['draws']]
        relation = ('UNORDERED_UNKNOWN' if unknown else
            'CANDIDATE_DOMINATES' if all(d > utility_tie_tolerance for d in deltas) else
            'BASELINE_DOMINATES' if all(d < -utility_tie_tolerance for d in deltas) else
            'EXACTLY_TIED' if all(d == 0 for d in deltas) else 'NO_STRICT_DRAW_DOMINANCE')
        comparisons.append(dict(baseline_id=left, candidate_id=right,
            comparison=comparison, known_delta_probability_mass=1-unknown,
            unknown_delta_probability_mass=unknown, strict_draw_relation=relation))
    complete_rows = [row for row in rows if row['complete_utility'] is not None]
    groups = _groups(complete_rows, utility_tie_tolerance)
    complete = len(complete_rows) == len(rows)
    if complete:
        for row in rows:
            row['rank_status'] = 'COMPLETE_DOMAIN_RANKED'
    return encode(dict(schema=SCHEMA, candidate_count=len(rows), candidates=rows,
        utility_tie_tolerance=utility_tie_tolerance,
        tie_policy='descending_highest_anchor_maximum_group_spread_within_tolerance',
        complete_domain_ranking_status='KNOWN' if complete else 'UNKNOWN',
        complete_domain_ranking=groups if complete else None,
        winner_status='KNOWN' if complete else 'UNKNOWN',
        winner_candidate_ids=groups[0]['candidate_ids'] if complete else None,
        complete_subdomain_ranking_diagnostic=groups,
        complete_subdomain_is_not_complete_domain_optimum=not complete,
        paired_comparisons=comparisons, conditional_means_used_for_ranking=False,
        mean_of_world_P95_is_not_mixture_latency_P95=True,
        original_equal_six_term_score_reused=True, original_stability_evaluated=False,
        qualifications_copied_not_authenticated=True, score_evidence_provenance_authenticated=False,
        final_gain_eligible=False, GPU_started=False, validation_labels_read=False,
        worlds_generated=False, model_fit=False))
