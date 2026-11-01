"""Pure arithmetic for a prospectively locked eight-configuration block.

This module neither collects GPU evidence nor certifies provenance, capacity,
prediction accuracy, stability, or a deployment optimum. A strict collection
adapter must verify those separately. One block supports descriptive contrasts
only; requests within that block are not independent experimental replicates.
"""
from __future__ import annotations

import math

MODES = ('shared', 'admission2')
NO_SPILL = ('balanced', 'skewed')
ROUTES = (*NO_SPILL, 'spill_low', 'spill_high')
TERMS = 'QTLPME'
TOLERANCE = 1e-12


def _utility(terms):
    if set(terms) != set(TERMS):
        raise ValueError('All six original terms required; no partial reweighting')
    if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v < 0)
           for v in terms.values()):
        raise ValueError('Finite nonnegative costs required')
    return -sum(terms.values()) / 6 if all(v is not None for v in terms.values()) else None


def _best(values, routes):
    if any(values[r] is None for r in routes):
        return dict(status='UNKNOWN', U=None, all_ties=[])
    best = max(values[r] for r in routes)
    return dict(status='KNOWN', U=best,
                all_ties=[r for r in routes if abs(values[r] - best) <= TOLERANCE])


def block(locked, rows):
    """Compute contrasts from verified caller data, retaining missing cells.

    ``locked`` must come from a protocol preceding final startup, with one
    no_spill and allow route per mode. ``rows`` contain mode, route, terms,
    requested, known and original_qualification (PASS/FAIL/UNKNOWN). Those
    labels are inputs to arithmetic, not evidence authenticated by this helper.
    An allow policy may select a no-spill route. Missing configurations keep
    their entire 1200-request denominator. No statistical intervals are fitted.
    """
    if set(locked) != set(MODES):
        raise ValueError('Both exact execution modes must be locked')
    for value in locked.values():
        if (set(value) != {'no_spill', 'allow'} or value['no_spill'] not in NO_SPILL
                or value['allow'] not in ROUTES):
            raise ValueError('Locks must respect nested no-spill/allow domains')
    cells = {}
    for row in rows:
        key = (row['mode'], row['route'])
        if key[0] not in MODES or key[1] not in ROUTES or key in cells:
            raise ValueError('No foreign or duplicate configuration')
        if (row['requested'] != 1200 or type(row['known']) is not int
                or not 0 <= row['known'] <= 1200
                or row['original_qualification'] not in ('PASS', 'FAIL', 'UNKNOWN')):
            raise ValueError('Full cohort denominator and explicit original qualification required')
        utility = _utility(row['terms'])
        cells[key] = dict(row, U=utility if row['known'] == 1200 else None)
    output = {}
    for mode in MODES:
        values = {r: cells.get((mode, r), {}).get('U') for r in ROUTES}
        no = _best(values, NO_SPILL)
        allow = _best(values, ROUTES)
        policy = locked[mode]
        n, a = (values[policy[k]] for k in ('no_spill', 'allow'))
        qualification = {r: cells.get((mode, r), {}).get('original_qualification', 'UNKNOWN')
                         for r in ROUTES}
        eligible = [r for r in ROUTES if qualification[r] == 'PASS']
        # Known failure is distinct from missing evidence. Empty feasible sets
        # have no maximizer; matching empty DFS outputs are not optimum proof.
        if any(v == 'UNKNOWN' for v in qualification.values()):
            qualified = dict(status='DOMAIN_QUALIFICATION_UNKNOWN', U=None, all_ties=[])
        elif not eligible:
            qualified = dict(status='NO_QUALIFIED_CANDIDATE_IN_DECLARED_DOMAIN', U=None, all_ties=[])
        else:
            qualified = _best(values, eligible)
        output[mode] = dict(locked_policy=policy, utility_by_route=values,
            locked_allow_minus_locked_no_spill=a-n if a is not None and n is not None else None,
            locked_allow_minus_each_no_spill={r: a-values[r]
                if a is not None and values[r] is not None else None for r in NO_SPILL},
            locked_allow_minus_strong_no_spill=a-no['U']
                if a is not None and no['U'] is not None else None,
            observed_no_spill_envelope=no, observed_allow_envelope=allow,
            post_observation_envelope_difference=allow['U']-no['U']
                if allow['U'] is not None and no['U'] is not None else None,
            envelope_is_deployable_gain=False, original_qualification=qualification,
            qualified_observed_envelope=qualified)
    known = sum(row['known'] for row in cells.values())
    contrasts = [output[m]['locked_allow_minus_locked_no_spill'] for m in MODES]
    return dict(schema='gsdeploy.H600_four_group_block_arithmetic.v1',
        status='DESCRIPTIVE_COMPLETE' if len(cells) == 8 and known == 9600
            and all(c['U'] is not None for c in cells.values()) else 'DESCRIPTIVE_INCOMPLETE',
        requested=9600, known=known, declared_configurations=8, supplied_configurations=len(cells),
        missing_configurations=[dict(mode=m, route=r) for m in MODES for r in ROUTES if (m,r) not in cells],
        modes=output, difference_in_locked_contrasts=contrasts[1]-contrasts[0]
            if all(v is not None for v in contrasts) else None,
        independent_blocks=1, confidence_interval=None,
        statistical_significance_claim=False, source_authentication_performed=False,
        qualified_deployment_optimum_certified=False, final_gain_eligible=False)
