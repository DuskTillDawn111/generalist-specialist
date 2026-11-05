"""Isolated parameterized copy of native idle-age energy training.

The original fitter remains frozen. Caller must declare edges; this changes the
cold-tail descriptor, never the measured-support bound used by the predictor.
"""
from collections import defaultdict
import math
import numpy as np
from scipy.optimize import lsq_linear
from gsdeploy.energy import integrate_power
from gsdeploy.forecasting.native_context import categories, design
from gsdeploy.forecasting.native_energy import activity_windows, exposure, SCHEMA

def fit_declared_idle_age_energy(profiles, runs, *, idle_edges, block_seconds=5.):
    """Fit covered physical blocks, with independently repeated idle support."""
    if block_seconds <= 0 or len({r['run_id'] for r in runs}) != len(runs):
        raise ValueError('Distinct training runs and positive block duration required')
    idle_edges = list(idle_edges)
    if (len(idle_edges)<2 or idle_edges[0]!=0. or any(not math.isfinite(x) for x in idle_edges)
            or any(a>=b for a,b in zip(idle_edges,idle_edges[1:]))):
        raise ValueError("Explicit finite increasing idle edges starting at zero required")
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
            row, state, max_age = exposure(clipped, terms, state, idle_edges=idle_edges)
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
                      blocks=len(rows), parameters=len(terms)+len(idle_edges)-1,
                      scope='training-only integrated board energy; no independent accuracy qualification')
    rank = int(np.linalg.matrix_rank(matrix)); diagnostic['rank'] = rank
    diagnostic['idle_bin_independent_runs'] = {str(i): sorted(repeats[i]) for i in range(len(idle_edges)-1)}
    if rank != matrix.shape[1] or any(len(repeats[i]) < 2 for i in range(len(idle_edges)-1)):
        diagnostic['reason'] = 'UNIDENTIFIABLE_OR_UNREPEATED_IDLE_AGE_ENERGY'
        return None, diagnostic
    # Relative block errors keep long cold windows visible next to busy blocks.
    lower = np.r_[np.full(len(terms), -np.inf), np.zeros(len(idle_edges)-1)]
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
                idle_edges_seconds=idle_edges, maximum_observed_idle_age_seconds=max_idle_age,
                support_vertices=[list(v) for v in sorted(vertices)],
                independent_run_ids=diagnostic['independent_run_ids'],
                descriptor_scope='host native-call activity and time since last native end; not device power-state counters')
    return spec, dict(diagnostic, blocks_detail=blocks)
