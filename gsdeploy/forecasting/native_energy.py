"""Empirical whole-board energy with a causal busy-to-idle history.

Native call age is a host-clock descriptor, not a GPU power-state counter.
Fits remain unqualified until independent FIFO and energy checks pass.
"""
from __future__ import annotations
from collections import defaultdict
import math

import numpy as np
from scipy.optimize import lsq_linear

from ..energy import integrate_power
from ..forecast_errors import UnknownEvidence
from .native_context import categories, counts, design, within_support

SCHEMA = 'gsdeploy.native_idle_age_energy.v1'
# Covered 5-second energy blocks do not resolve subsecond cooldown bins.
# Keep a coarse descriptor; every saved fit carries its own discretization.
IDLE_EDGES_SECONDS = [0., 2., 10., 20., 31.]


def activity_windows(profiles, spans, start_ns, end_ns):
    """Continuous count roster, retaining all host intervals as count zero."""
    events = defaultdict(list)
    for span in spans:
        if not start_ns <= span['start_ns'] < span['end_ns'] <= end_ns:
            raise ValueError('Energy span outside its declared window')
        events[span['start_ns']].append(('start', span))
        events[span['end_ns']].append(('end', span))
    active, result, previous = {}, [], start_ns
    names = categories(profiles)
    for at in sorted({start_ns, end_ns, *events}):
        if at > previous:
            result.append((previous, at, counts(active, profiles, names)))
        for kind, span in sorted(events[at], key=lambda item: item[0]):
            alias = span['alias']
            if kind == 'end':
                if active.pop(alias) is not span:
                    raise ValueError('Unmatched energy stage')
            else:
                if alias in active:
                    raise ValueError('Overlapping energy owner')
                active[alias] = span
        previous = at
    if active:
        raise ValueError('Incomplete energy stages')
    return result


def exposure(windows, terms, last_native_end_ns, *, idle_edges=IDLE_EDGES_SECONDS):
    """Integrate activity and idle-age bases; return causal carry state.

    The initial timestamp must come from a completed prefix or a previous
    prediction. A missing initial state is never silently treated as cold.
    """
    if type(last_native_end_ns) is not int:
        raise UnknownEvidence('UNKNOWN_ENERGY_PREFIX_LAST_NATIVE_END')
    value = np.zeros(len(terms) + len(idle_edges) - 1)
    max_age = 0.
    previous = None
    for left, right, x in windows:
        if right <= left or previous is not None and left != previous:
            raise ValueError('Continuous ordered energy windows required')
        if last_native_end_ns > left:
            raise ValueError('Energy prefix state comes from the future')
        if any(x):
            value[:len(terms)] += (right-left)/1e9 * design(x, terms)
            last_native_end_ns = right
        else:
            age0, age1 = (left-last_native_end_ns)/1e9, (right-last_native_end_ns)/1e9
            max_age = max(max_age, age1)
            if age1 > idle_edges[-1] + 1e-9:
                raise UnknownEvidence('ENERGY_IDLE_AGE_OUTSIDE_DESCRIPTOR_DOMAIN')
            for i, (lo, hi) in enumerate(zip(idle_edges, idle_edges[1:])):
                value[len(terms)+i] += max(0., min(age1, hi)-max(age0, lo))
        previous = right
    return value, last_native_end_ns, max_age


def fit_idle_age_energy(profiles, runs, *, block_seconds=5.):
    """Fit covered physical blocks, with independently repeated idle support."""
    if block_seconds <= 0 or len({r['run_id'] for r in runs}) != len(runs):
        raise ValueError('Distinct training runs and positive block duration required')
    names = categories(profiles)
    rosters, vertices = [], set()
    for run in runs:
        if run['split'] != 'training' or run.get('telemetry_errors') or any(
                s.get('foreign_pids') for s in run['board_samples']):
            raise ValueError('Clean training-only energy receipts required')
        windows = activity_windows(profiles, run['spans'], *run['window_ns'])
        vertices.update(tuple(x) for _, _, x in windows)
        rosters.append((run, windows))
    used = [i for i in range(len(names)) if any(v[i] for v in vertices)]
    terms = [[]] + [[i] for i in used] + [[i, j] for i in used for j in used
              if i < j and any(v[i]*v[j] for v in vertices)]
    rows, targets, blocks, repeats = [], [], [], defaultdict(set)
    max_idle_age = 0.
    for run, windows in rosters:
        start, end = run['window_ns']
        # Carry idle age across block boundaries instead of resetting every block.
        state = run['initial_last_native_end_ns']
        left = start
        while left < end:
            right = min(end, left + round(block_seconds*1e9))
            clipped = [(max(a, left), min(b, right), x) for a, b, x in windows if a < right and b > left]
            row, state, max_age = exposure(clipped, terms, state)
            max_idle_age = max(max_idle_age, max_age)
            joules = integrate_power(run['board_samples'], left, right, max_gap_seconds=.25)
            if joules is None or joules <= 0:
                raise ValueError('Covered nonzero measured board energy required')
            if (right-left)/1e9 >= min(.5, block_seconds):
                rows.append(row); targets.append(joules)
                blocks.append(dict(run_id=run['run_id'], start_ns=left, end_ns=right, observed_joules=joules))
                for i in np.flatnonzero(row[len(terms):] > 1e-8):
                    repeats[int(i)].add(run['run_id'])
            left = right
    matrix, target = np.array(rows), np.array(targets)
    diagnostic = dict(accepted=False, independent_run_ids=sorted(r['run_id'] for r in runs),
                      blocks=len(rows), parameters=len(terms)+len(IDLE_EDGES_SECONDS)-1,
                      scope='training-only integrated board energy; no independent accuracy qualification')
    rank = int(np.linalg.matrix_rank(matrix)); diagnostic['rank'] = rank
    diagnostic['idle_bin_independent_runs'] = {str(i): sorted(repeats[i]) for i in range(len(IDLE_EDGES_SECONDS)-1)}
    if rank != matrix.shape[1] or any(len(repeats[i]) < 2 for i in range(len(IDLE_EDGES_SECONDS)-1)):
        diagnostic['reason'] = 'UNIDENTIFIABLE_OR_UNREPEATED_IDLE_AGE_ENERGY'
        return None, diagnostic
    # Relative block errors keep long cold windows visible next to busy blocks.
    lower = np.r_[np.full(len(terms), -np.inf), np.zeros(len(IDLE_EDGES_SECONDS)-1)]
    fit = lsq_linear(matrix/target[:, None], np.ones(len(target)), bounds=(lower, np.inf), max_iter=1000)
    beta = fit.x
    active_power = [float(design(v, terms) @ beta[:len(terms)]) for v in vertices if any(v)]
    if not fit.success or any(p < 0 for p in active_power):
        diagnostic['reason'] = 'NONPHYSICAL_POWER_ON_OBSERVED_ACTIVITY_OR_SOLVER_FAILURE'
        return None, diagnostic
    errors = abs(matrix@beta-target)/target
    diagnostic.update(accepted=True, maximum_relative_block_error=float(max(errors)),
                      mean_relative_block_error=float(np.mean(errors)),
                      p95_relative_block_error=float(np.quantile(errors, .95)),
                      weighted_design_condition_number=float(np.linalg.cond(matrix/target[:,None])),
                      idle_coefficient_lower_bound_hits=np.flatnonzero(beta[len(terms):]<1e-7).tolist())
    if diagnostic['idle_coefficient_lower_bound_hits']:
        diagnostic.update(accepted=False,reason='IDLE_POWER_COEFFICIENT_AT_ZERO_BOUND')
    for b, energy in zip(blocks, matrix@beta):
        b['predicted_joules'] = float(energy)
        b['relative_error'] = abs(float(energy)-b['observed_joules'])/b['observed_joules']
    spec = dict(schema=SCHEMA, status='FITTED_UNVALIDATED', admission_enabled=False,
                categories=[list(n) for n in names], active_terms=terms, coefficients=beta.tolist(),
                idle_edges_seconds=IDLE_EDGES_SECONDS, maximum_observed_idle_age_seconds=max_idle_age,
                support_vertices=[list(v) for v in sorted(vertices)],
                independent_run_ids=diagnostic['independent_run_ids'],
                descriptor_scope='host native-call activity and time since last native end; not device power-state counters')
    return spec, dict(diagnostic, blocks_detail=blocks)


class NativeIdleAgeEnergy:
    def __init__(self, spec):
        if (spec.get('schema') != SCHEMA or spec.get('status') != 'FITTED_UNVALIDATED'
                or spec.get('admission_enabled') is not False
                or not isinstance(spec.get('idle_edges_seconds'),list)
                or len(spec['idle_edges_seconds'])<2 or spec['idle_edges_seconds'][0]!=0.
                or any(not math.isfinite(x) for x in spec['idle_edges_seconds'])
                or any(a>=b for a,b in zip(spec['idle_edges_seconds'],spec['idle_edges_seconds'][1:]))
                or len(spec.get('independent_run_ids', [])) < 2
                or len(spec['coefficients']) != len(spec['active_terms'])+len(spec['idle_edges_seconds'])-1
                or any(not math.isfinite(x) for x in spec['coefficients'])
                or any(x<0 for x in spec['coefficients'][len(spec['active_terms']):])):
            raise ValueError('Explicit unvalidated causal idle energy law required')
        self.spec = spec

    def integrate(self, windows, last_native_end_ns):
        for _, _, x in windows:
            if not within_support(x, self.spec['support_vertices']):
                raise UnknownEvidence('ENERGY_ACTIVITY_OUTSIDE_TRAINING_SUPPORT')
            if any(x) and float(design(x,self.spec['active_terms']) @ np.array(
                    self.spec['coefficients'][:len(self.spec['active_terms'])]))<0:
                raise UnknownEvidence('NEGATIVE_POWER_AT_PREDICTED_ACTIVITY')
        row, state, age = exposure(windows, self.spec['active_terms'], last_native_end_ns,
                                   idle_edges=self.spec['idle_edges_seconds'])
        if age > self.spec['maximum_observed_idle_age_seconds'] + 1e-8:
            raise UnknownEvidence('ENERGY_IDLE_AGE_OUTSIDE_MEASURED_SUPPORT')
        joules = float(row @ np.array(self.spec['coefficients']))
        if not math.isfinite(joules) or joules < 0:
            raise UnknownEvidence('INVALID_IDLE_AGE_ENERGY')
        return joules, state
