"""Training diagnostic for positive virtual clocks; no qualified queue provider.

Lognormal reference response and interference are jointly estimated. Whole
captures remain the replication unit. Kernel-perturbed timing is never pooled
with natural timing. Coefficients are unconstrained and may encode acceleration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .stage_response_features_v1 import checked, receipt
from .stage_response_training_v2 import extract as rederive_training_channels

SCHEMA = 'gsdeploy.stage_response_virtual_clock_training_diagnostic.v1'
FEATURES = ('peer_visible_unconfirmed_owner_count', 'peer_same_semantic_phase_owner_count')
CANDIDATES = {'baseline': (), 'visible_count': FEATURES[:1], 'visible_phase_aware': FEATURES}


def require(value, message):
    if not value:
        raise ValueError(message)


def prepared(stages, features):
    rows = []
    for stage in stages:
        if stage['stage_kind'] != 'language_forward':
            continue
        require(stage['host_training_fit_diagnostic_eligible'] is True
            and stage['stream_response_fit_eligible'] is False
            and stage['host_channel_is_effective_SM_service'] is False,
            'Only explicit host-channel training diagnostics supported')
        start, end = stage['response_labels']['host_observed_interval_ns']
        path = stage['full_host_wall_covariate_path']
        require(path and path[0]['left_ns'] == start and path[-1]['right_ns'] == end
            and all(a['right_ns'] == b['left_ns'] for a,b in zip(path,path[1:])),
            'Exact complete occurrence-origin host exposure required')
        durations, matrix = [], []
        for piece in path:
            duration = piece['right_ns']-piece['left_ns']
            require(type(duration) is int and duration > 0
                and duration == piece['duration_ns']
                and piece['state']['cutoff_processing_ns'] == piece['left_ns']
                and piece['each_state_uses_only_its_processed_prefix'] is True,
                'Positive causal disjoint host intervals required')
            vector = [piece['state'][key] for key in features]
            require(all(type(x) is int and x >= 0 for x in vector), 'Nonnegative integer evidence counts required')
            durations.append(duration/1e6)
            matrix.append(vector)
        require(sum(p['duration_ns'] for p in path) == stage['response_labels']['host_call_wall_ns'],
            'Host label and covariate path clocks differ')
        # Exact known-input forward baseline avoids treating decode index/shape
        # changes as measured interference. This is not unseen-input support.
        group = (stage['owner'], stage['pool_id'], stage['semantic_phase'],
            stage['forward_index'], stage['input_sequence_length'], stage['input_batch_size'])
        rows.append(dict(group=group, capture=stage['capture_sha256'],
            request_id=stage['request_id'], phase_id=stage['phase_id'],
            durations=np.asarray(durations), matrix=np.asarray(matrix).reshape(len(path),len(features))))
    require(bool(rows), 'Language host training rows required')
    return rows


def profile(beta, rows):
    """Exact time-change likelihood with endpoint rate and profiled nuisance.

    V=sum(dt*exp(-beta*x)); f_T=f_W(V)*rate(end). Omitting the
    endpoint Jacobian would fit a different, generally biased objective.
    """
    logs, derivatives, end_rates = [], [], []
    for row in rows:
        values = np.log(row['durations']) - row['matrix'] @ beta
        logv = float(logsumexp(values))
        weights = np.exp(values-logv)
        logs.append(logv)
        derivatives.append(-(weights @ row['matrix']))
        end_rates.append(-float(row['matrix'][-1] @ beta))
    grouped = {}
    for row,value in zip(rows,logs):
        grouped.setdefault(row['group'],[]).append(value)
    mu = {key:float(np.mean(values)) for key,values in grouped.items()}
    residual = np.asarray([value-mu[row['group']] for row,value in zip(rows,logs)])
    variance = float(np.mean(residual**2))
    # A zero residual variance has no finite lognormal density MLE; do not
    # silently assign an arbitrary variance floor and declare qualification.
    require(math.isfinite(variance) and variance > 0, 'Degenerate zero-variance response likelihood')
    n = len(rows)
    nll = .5*n*(math.log(2*math.pi*variance)+1) + sum(logs)-sum(end_rates)
    return float(nll), mu, variance, np.asarray(derivatives), residual


def fit(stages, candidate):
    features = CANDIDATES[candidate]
    rows = prepared(stages,features)
    width = len(features)
    attempts = []
    if width:
        def objective(beta):
            try:
                value, _, variance, derivative, residual = profile(beta,rows)
                endpoint = np.asarray([row['matrix'][-1] for row in rows])
                gradient = np.sum(derivative*(1+residual[:,None]/variance),axis=0)+np.sum(endpoint,axis=0)
                return value, gradient
            except (ValueError, FloatingPointError, OverflowError):
                return float('inf'), np.full(width,float('nan'))
        for start in (np.zeros(width), np.full(width,.25), np.full(width,-.25)):
            result = minimize(objective,start,method='BFGS',jac=True,options={'gtol':1e-7,'maxiter':1000})
            attempts.append(result)
        best = min(attempts,key=lambda x: x.fun if math.isfinite(float(x.fun)) else float('inf'))
        beta = best.x
        converged = bool(best.success)
        gradient_norm = float(np.linalg.norm(best.jac))
    else:
        beta = np.empty(0)
        converged, gradient_norm = True, 0.
    nll,mu,variance,derivatives,residual = profile(beta,rows)
    groups = sorted(mu)
    group_index = {key:i for i,key in enumerate(groups)}
    # Residual Jacobian includes every jointly fitted baseline nuisance mean.
    baseline = np.zeros((len(rows),len(groups)))
    for i,row in enumerate(rows):
        baseline[i,group_index[row['group']]]=-1
    jacobian = np.column_stack((baseline,derivatives))
    singular = np.linalg.svd(jacobian,compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian))
    dimension = len(groups)+width
    captures = sorted({row['capture'] for row in rows})
    counts = Counter(row['group'] for row in rows)
    # Numerical profile likelihood Hessian is a diagnostic, not a replacement
    # for cluster replication or a claim that phase rows are independent.
    hessian = None
    if width:
        step=1e-4
        h=np.empty((width,width))
        for i in range(width):
            ei=np.eye(width)[i]*step
            for j in range(width):
                ej=np.eye(width)[j]*step
                h[i,j]=(profile(beta+ei+ej,rows)[0]-profile(beta+ei-ej,rows)[0]
                    -profile(beta-ei+ej,rows)[0]+profile(beta-ei-ej,rows)[0])/(4*step*step)
        hessian=dict(matrix=h.tolist(),eigenvalues=np.linalg.eigvalsh(h).tolist(),
            scope='phase_conditional_profile_likelihood_diagnostic_not_capture_robust_uncertainty')
    return dict(candidate=candidate,features=list(features),coefficients=beta.tolist(),
        coefficient_constraints='free_real_acceleration_allowed',nll=nll,
        lognormal_reference_sigma=math.sqrt(variance),
        baselines=[dict(group=list(key),log_reference_ms=mu[key],rows=counts[key]) for key in groups],
        optimizer_converged=converged,optimizer_gradient_norm=gradient_norm,
        optimizer_attempts=[dict(success=bool(x.success),message=str(x.message),nll=float(x.fun)) for x in attempts],
        nonlinear_joint_response_jacobian=dict(rank=rank,parameters=dimension,
            full_column_rank=rank==dimension,singular_values=singular.tolist(),
            condition_number=float(singular[0]/singular[-1]) if len(singular) and singular[-1]>0 else None,
            includes_baseline_nuisance_means=True,includes_response_sigma=False),
        profile_likelihood_hessian=hessian,host_phase_rows=len(rows),
        whole_capture_blocks=len(captures),capture_sha256=captures,
        minimum_initial_three_capture_design_met=len(captures)>=3,
        training_log_reference_residual_rmse=float(np.sqrt(np.mean(residual**2))),
        in_sample_diagnostic_only=True,prospective_predictor_qualified=False,
        capture_robust_parameter_uncertainty_estimated=False)


def run(paths):
    references = [receipt(path) for path in paths]
    docs = [checked(ref) for ref in references]
    require(bool(docs) and len({doc['mode'] for doc in docs}) == 1,
        'Single timing domain per training fit required')
    for doc in docs:
        require(doc['schema']=='gsdeploy.reviewed_stage_response_training.v2'
            and doc['status']=='REVIEWED_CLOCK_ALIGNED_TRAINING_DIAGNOSTICS_ONLY'
            and doc['validation_labels_read'] is False, 'Reviewed v2 training channels required')
        for ref in doc['sources']:
            require(receipt(ref['path'])==ref,'Training feature source changed')
        require(rederive_training_channels(doc['sources'][0]['path']) == doc,
            'Features must rederive exactly from the complete unchanged training review closure')
    stages = [s for d in docs for s in d['stages']]
    captures = {s['capture_sha256'] for s in stages}
    require(len(captures)==len(docs), 'Duplicate capture is not an independent repetition')
    results = [fit(stages,name) for name in CANDIDATES]
    sources = [*references,receipt(__file__)]
    require(all(receipt(r['path'])==r for r in sources), 'Source changed during fit')
    return dict(schema=SCHEMA,status='TRAINING_LIKELIHOOD_DIAGNOSTICS_NOT_MODEL_SELECTION',
        mode=docs[0]['mode'],sources=sources,candidates=results,selected_candidate=None,
        error_family='lognormal_reference_work_with_exact_positive_virtual_time_change',
        baseline_and_interference_jointly_fitted=True,endpoint_density_jacobian_included=True,
        whole_capture_leaveout_performed=False,main_queue_provider_eligible=False,
        independent_prediction_accuracy_validated=False,final_gain_eligible=False,
        GPU_started=False,validation_labels_read=False,
        limitations=['No heldout capture has been used and no candidate is selected by in-sample likelihood.',
            'Phase-row likelihood diagnostics do not establish independent capture uncertainty or P95 accuracy.',
            'Coefficients describe parent-visible evidence association, not certified physical SM/DRAM interference.',
            'Future covariates, notifications, sync and host gaps still need a jointly executable provider.',
            'Stream elapsed, physical context clocks and unavailable hardware counters were not fitted.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-features',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=run(args.training_features)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'],candidates=len(result['candidates']))))
