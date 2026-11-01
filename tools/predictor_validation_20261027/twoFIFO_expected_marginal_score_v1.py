"""Pure expected original utility from complete, explicitly same-domain marginals.

Linearity needs no invented joint queue/resource draw. Each marginal must
already have scored its nonlinear statistic per draw (in particular max memory
and P95). This helper authenticates neither those sources nor their common
domain. A native producer must reconstruct both before calling it.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import math

from .actual_worker_twoFIFO_reference_v5_2 import decode
from .macro_response_queue_reference_v1 import encode

SCHEMA = 'gsdeploy.twoFIFO_expected_original_marginal_score.v1'
TERMS = 'QTLPME'
DISTRIBUTION_FIELDS = {'known_probability_mass', 'unknown_probability_mass',
    'expected_value', 'conditional_on_known_mean', 'conditional_mean_is_not_full_expectation'}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def _number(value):
    return type(value) in (int, float, Fraction) and math.isfinite(value) and value >= 0


def combine(marginals, *, candidate_id, candidate_domain, comparison_contract, weights):
    """Compose supplied original term summaries, never fill an UNKNOWN term.

    candidate_domain binds a single exact routed case and execution protocol.
    comparison_contract binds its route-independent offered population,
    horizon, original score normalization and execution protocol. These are
    explicit caller declarations, not authentication or execution permission.
    Marginal summaries have the original forecast ensemble distribution shape.
    """
    need(type(candidate_id) is str and candidate_id, 'Explicit candidate identity required')
    need(type(candidate_domain) is dict and candidate_domain and
         type(comparison_contract) is dict and comparison_contract,
         'Explicit candidate domain and common comparison contract required')
    need(weights == {term: 1/6 for term in TERMS}, 'Original equal six weights required')
    need(type(marginals) is dict and set(marginals) == set(TERMS),
         'Exactly all six original term marginals required')
    values = {}
    summaries = {}
    for term in TERMS:
        marginal = decode(marginals[term])
        need(type(marginal) is dict and set(marginal) == {'domain', 'distribution', 'sources'}
             and marginal['domain'] == candidate_domain and type(marginal['sources']) is list
             and marginal['sources'], 'Every marginal must explicitly bind the same candidate domain and sources')
        distribution = marginal['distribution']
        need(type(distribution) is dict and set(distribution) == DISTRIBUTION_FIELDS,
             'Original full-mass ensemble distribution summary required')
        known, unknown = (distribution[k] for k in ('known_probability_mass', 'unknown_probability_mass'))
        need(type(known) in (int, Fraction) and type(unknown) in (int, Fraction)
             and 0 <= known <= 1 and 0 <= unknown <= 1 and known+unknown == 1,
             'Exact full probability mass required for each original term')
        value = distribution['expected_value']
        need(distribution['conditional_mean_is_not_full_expectation'] is True,
             'A conditional-on-known mean cannot replace a full expectation')
        need((_number(value) if known == 1 else value is None),
             'Only full known mass can supply a finite nonnegative expected term')
        conditional = distribution['conditional_on_known_mean']
        need(conditional is None or _number(conditional), 'Finite nonnegative conditional diagnostic required')
        if known == 0:
            need(conditional is None, 'Zero known mass has no conditional mean')
        if known == 1:
            need(conditional is not None and math.isclose(float(value), float(conditional), rel_tol=1e-12, abs_tol=1e-12),
                 'Complete original distribution means disagree')
        values[term] = value
        summaries[term] = copy.deepcopy(marginal)
    complete = all(value is not None for value in values.values())
    expected = -math.fsum(weights[term]*float(values[term]) for term in TERMS) if complete else None
    return encode(dict(schema=SCHEMA, candidate_id=candidate_id,
        candidate_domain=copy.deepcopy(candidate_domain), comparison_contract=copy.deepcopy(comparison_contract),
        weights=copy.deepcopy(weights), marginals=summaries, expected_terms=values,
        expected_U_under_supplied_marginals=expected, all_six_marginal_expectations_complete=complete,
        joint_queue_resource_law_inferred=False, independence_assumed=False,
        UNKNOWN_mass_renormalized=False, conditional_means_used_as_full_expectations=False,
        stability_conditioning_applied=False, original_stability_evaluated=False,
        source_authenticated=False, independent_prediction_accuracy_evaluated=False,
        positive_delta_probability=None, full_U_qualified=False, final_gain_eligible=False,
        GPU_started=False))


def _reconstruct(document):
    document = decode(document)
    need(document.get('schema') == SCHEMA, 'Explicit marginal expected-score schema required')
    rebuilt = decode(combine(document['marginals'], candidate_id=document['candidate_id'],
        candidate_domain=document['candidate_domain'], comparison_contract=document['comparison_contract'],
        weights=document['weights']))
    need(document == rebuilt, 'Expected score must rederive from all six original marginal summaries')
    return rebuilt


def compare(baseline, candidate):
    """Compare means only; no victory probability or stability gate is inferred."""
    left, right = _reconstruct(baseline), _reconstruct(candidate)
    need(left['candidate_id'] != right['candidate_id'], 'Distinct comparison candidates required')
    need(left['comparison_contract'] == right['comparison_contract'] and left['weights'] == right['weights'],
         'Identical population, score normalization, horizon and execution protocol required')
    a, b = (row['expected_U_under_supplied_marginals'] for row in (left, right))
    delta = None if a is None or b is None else b-a
    return encode(dict(schema=SCHEMA+'.mean_comparison', baseline_id=left['candidate_id'],
        candidate_id=right['candidate_id'], expected_delta_U_under_supplied_marginals=delta,
        candidate_better_in_expected_U=None if delta is None else delta > 0,
        positive_delta_probability=None, confidence_interval=None,
        joint_law_inferred=False, original_stability_evaluated=False,
        source_authenticated=False, independent_prediction_accuracy_evaluated=False,
        final_gain_eligible=False, GPU_started=False))
