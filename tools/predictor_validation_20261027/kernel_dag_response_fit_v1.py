"""Shared lifetime likelihood with a linear-memory nuisance rank certificate.

Keep the original exact-Hessian/root solver, starts and tolerances. A dense
N-by-G nuisance dummy matrix is inappropriate for a baseline per DAG node.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize, root

from .stage_response_fit_v1 import profile
from .macro_joint_response_second_order_v1 import differential, local_minimum_certificate


def nuisance_certificate(rows, derivatives):
    """Conservatively certify the original full Jacobian's default rank test.

    J=[Z,D]=[Z,D_centered] [[I,M],[0,I]]. Its minimum singular value is at least
    min(sqrt(min group count), sigma_min(D_centered))/(1+||M||_2).
    Its maximum is at most ||J||_F. Use those bounds in numpy's original
    max(shape)*eps*sigma_max threshold. Inconclusive bounds never qualify.
    """
    d = np.asarray(derivatives, dtype=float)
    groups = {}
    for i, row in enumerate(rows): groups.setdefault(row['group'], []).append(i)
    centered = d.copy(); means = []; counts = []
    for indices in groups.values():
        mean = np.mean(d[indices], axis=0)
        centered[indices] -= mean; means.append(mean); counts.append(len(indices))
    singular = np.linalg.svd(centered, compute_uv=False)
    width = d.shape[1]
    projected_rank = int(np.linalg.matrix_rank(centered))
    low = (min(math.sqrt(min(counts)), float(singular[-1])) /
           (1+float(np.linalg.norm(np.asarray(means), ord=2)))) if len(singular)==width else 0.
    high = math.sqrt(len(rows)+float(np.sum(d*d)))
    threshold = max(len(rows), len(groups)+width)*np.finfo(float).eps*high
    certified = bool(projected_rank==width and math.isfinite(low) and low > threshold)
    return dict(parameters=len(groups)+width, full_column_rank=certified,
        algebraic_rank_via_projected_numerical_rank=len(groups)+projected_rank,
        projected_derivative_rank=projected_rank, projected_singular_values=singular.tolist(),
        full_jacobian_min_singular_lower_bound=low,
        full_jacobian_max_singular_upper_bound=high,
        original_default_rank_tolerance_upper_bound=threshold,
        includes_baseline_nuisance_means=True, includes_sigma=False,
        dense_nuisance_matrix_allocated=False, conservative_certificate=True)


def fit(rows, features):
    width = len(features); attempts = []; roots = []
    if width <= 0:
        raise ValueError('Explicit positive kernel feature width required')
    for start in (np.zeros(width), np.full(width,.25), np.full(width,-.25)):
        solved = minimize(lambda b:differential(b,rows)[:2], start, jac=True,
            hess=lambda b:differential(b,rows)[2], method='trust-exact',
            options={'gtol':1e-10,'maxiter':300})
        value, gradient, hessian = differential(solved.x, rows)
        attempts.append(dict(start=start.tolist(),solution=solved.x.tolist(),success=bool(solved.success),
            message=str(solved.message),mean_nll=value,**local_minimum_certificate(gradient,hessian)))
        stationary = root(lambda b:differential(b,rows)[1], solved.x,
            jac=lambda b:differential(b,rows)[2], method='lm',
            options={'ftol':1e-12,'xtol':1e-12,'gtol':1e-12,'maxiter':300})
        value, gradient, hessian = differential(stationary.x,rows)
        roots.append(dict(start=solved.x.tolist(),solution=stationary.x.tolist(),success=bool(stationary.success),
            message=str(stationary.message),mean_nll=value,**local_minimum_certificate(gradient,hessian)))
    valid = [r for r in roots if r['success'] and r['stationary_strict_local_minimum']]
    finite = [r for r in [*roots,*attempts] if math.isfinite(r['mean_nll'])]
    if not finite:
        raise ValueError('No finite diagnostic solution')
    best = min(valid or finite,key=lambda r:r['mean_nll']); beta = np.asarray(best['solution'])
    disagreement = max(float(np.linalg.norm(np.asarray(r['solution'])-beta)) for r in roots)
    qualified = len(valid)==3 and disagreement<1e-7 and all(
        r['strictly_positive_curvature'] and np.linalg.norm(np.asarray(r['solution'])-beta)<1e-7
        for r in attempts)
    nll, mu, variance, derivatives, _ = profile(beta,rows)
    certificate = nuisance_certificate(rows,derivatives)
    groups = sorted(mu,key=repr)
    return dict(features=list(features),coefficients=beta.tolist(),macro_rows=len(rows),nll=nll,
        mean_nll=nll/len(rows),lognormal_reference_sigma=math.sqrt(variance),
        baselines=[dict(group=list(g),log_reference_ms=mu[g]) for g in groups],
        captures=sorted({r['capture'] for r in rows}),trust_exact_attempts=attempts,
        gradient_root_attempts=roots,numerical_qualification=bool(qualified and certificate['full_column_rank']),
        root_max_solution_disagreement=disagreement,
        stationarity_certificate=local_minimum_certificate(*differential(beta,rows)[1:]),
        nuisance_adjusted_derivative_rank=certificate['projected_derivative_rank'],
        nuisance_adjusted_features=width,joint_response_jacobian=certificate,
        effective_SM_service_identified=False,prospective_prediction_qualified=False), beta, mu, variance
