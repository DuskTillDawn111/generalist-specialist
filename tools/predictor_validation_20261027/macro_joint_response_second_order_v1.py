"""Independent analytic second-order solver for training macro diagnostics.

No optimizer flags are rewritten. Exact Hessian trust-region minimization and
an independently solved gradient root must agree with positive local curvature.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, root
from scipy.special import logsumexp

from . import macro_joint_response_fit_v1 as first
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA='gsdeploy.macro_joint_response_second_order_training_leaveout.v1'
STATIONARITY_TOLERANCE=1e-10
SOLUTION_AGREEMENT_TOLERANCE=1e-7


def differential(beta,rows):
    """Profile meanNLL, gradient and Hessian including endpoint Jacobian.

    l_i=log V_i, d_i=-E_w[x], C_i=Cov_w[x], r_i=l_i-mean_g(l),
    v=mean(r^2), a=mean(r*d),
    H=mean(C)+(mean[(d-mean_g d)(d-mean_g d)' ]+mean[r*C])/v
      -2*a*a'/v^2. Endpoint -log(rate) is linear and has zero Hessian.
    """
    width=len(beta);n=len(rows);logs=[];ds=[];cs=[];endpoints=[]
    for row in rows:
        terms=np.log(row['durations'])-row['matrix']@beta
        logv=float(logsumexp(terms));weights=np.exp(terms-logv)
        mean=weights@row['matrix'];center=row['matrix']-mean
        covariance=(center.T*weights)@center
        logs.append(logv);ds.append(-mean);cs.append(covariance)
        endpoints.append(row['matrix'][-1])
    logs=np.asarray(logs);ds=np.asarray(ds).reshape(n,width);cs=np.asarray(cs).reshape(n,width,width)
    groups={}
    for i,row in enumerate(rows):groups.setdefault(row['group'],[]).append(i)
    residual=logs.copy();centered=ds.copy()
    for indices in groups.values():
        residual[indices]-=np.mean(logs[indices]);centered[indices]-=np.mean(ds[indices],axis=0)
    variance=float(np.mean(residual**2))
    first.require(math.isfinite(variance) and variance>0,'Degenerate zero-variance response likelihood')
    a=np.mean(residual[:,None]*ds,axis=0)
    gradient=np.mean(ds,axis=0)+a/variance+np.mean(np.asarray(endpoints).reshape(n,width),axis=0)
    hessian=np.mean(cs,axis=0)+(centered.T@centered/n+np.mean(residual[:,None,None]*cs,axis=0))/variance\
        -2*np.outer(a,a)/(variance*variance)
    value=.5*(math.log(2*math.pi*variance)+1)+float(np.mean(logs))\
        +float(np.mean(np.asarray(endpoints).reshape(n,width)@beta))
    return value,gradient,(hessian+hessian.T)/2


def local_minimum_certificate(gradient,hessian):
    eigenvalues=np.linalg.eigvalsh(hessian)
    norm=float(np.linalg.norm(gradient))
    return dict(mean_nll_gradient_norm=norm,hessian=hessian.tolist(),eigenvalues=eigenvalues.tolist(),
        stationary_with_fixed_tolerance=norm<STATIONARITY_TOLERANCE,
        strictly_positive_curvature=bool(np.all(eigenvalues>0)),
        stationary_strict_local_minimum=bool(norm<STATIONARITY_TOLERANCE and np.all(eigenvalues>0)))


def second_order_fit(rows,candidate):
    original,original_beta,_,_=first.fit(rows,candidate)
    width=len(first.CANDIDATES[candidate])
    if not width:
        original.update(original_BFGS_diagnostic=dict(optimizer_converged=original['optimizer_converged'],
            optimizer_attempts=original['optimizer_attempts']),
            second_order_qualification=True,solver='analytic_profile_baseline_no_beta_optimization')
        _,mu,variance,_,_=first.profile(original_beta,rows)
        return original,original_beta,mu,variance
    attempts=[];roots=[]
    starts=(np.zeros(width),np.full(width,.25),np.full(width,-.25))
    for start in starts:
        solved=minimize(lambda b:differential(b,rows)[:2],start,jac=True,
            hess=lambda b:differential(b,rows)[2],method='trust-exact',
            options={'gtol':1e-10,'maxiter':300})
        value,gradient,hessian=differential(solved.x,rows)
        attempts.append(dict(start=start.tolist(),solution=solved.x.tolist(),success=bool(solved.success),
            message=str(solved.message),iterations=int(solved.nit),mean_nll=value,
            **local_minimum_certificate(gradient,hessian)))
        # Independently refine/verify the gradient at each trust solution.
        # Blind gradient-root searches can drift to infinity where both the
        # score and curvature vanish; root.success alone is insufficient.
        stationary=root(lambda b:differential(b,rows)[1],solved.x,
            jac=lambda b:differential(b,rows)[2],method='lm',
            options={'ftol':1e-12,'xtol':1e-12,'gtol':1e-12,'maxiter':300})
        value,gradient,hessian=differential(stationary.x,rows)
        roots.append(dict(original_trust_start=start.tolist(),start=solved.x.tolist(),
            solution=stationary.x.tolist(),success=bool(stationary.success),
            message=str(stationary.message),mean_nll=value,**local_minimum_certificate(gradient,hessian)))
    minimizers=[x for x in attempts if x['success'] and x['stationary_strict_local_minimum']]
    verified_roots=[x for x in roots if x['success'] and x['stationary_strict_local_minimum']]
    # Failing flags stay failing. A successful root is a separately justified
    # candidate solver result, never a relabelled BFGS/trust status.
    valid=verified_roots
    first.require(bool(valid),'No successful positive-curvature stationary root')
    best=min(valid,key=lambda x:x['mean_nll']);beta=np.asarray(best['solution'])
    nll,mu,variance,derivatives,residual=first.profile(beta,rows)
    agreement=max(float(np.linalg.norm(np.asarray(x['solution'])-beta)) for x in valid)
    all_roots_agree=len(valid)==len(starts) and agreement<SOLUTION_AGREEMENT_TOLERANCE
    trust_agreement=all(float(np.linalg.norm(np.asarray(x['solution'])-beta))<SOLUTION_AGREEMENT_TOLERANCE
        and x['strictly_positive_curvature'] for x in attempts)
    # Recompute full nuisance Jacobian at the actual root, rather than copying
    # rank from the BFGS position.
    groups=sorted(mu,key=repr);indices={g:i for i,g in enumerate(groups)}
    nuisance=np.zeros((len(rows),len(groups)))
    for i,row in enumerate(rows):nuisance[i,indices[row['group']]]=-1
    jacobian=np.column_stack((nuisance,derivatives));singular=np.linalg.svd(jacobian,compute_uv=False)
    rank=int(np.linalg.matrix_rank(jacobian));dimension=len(groups)+width
    qualified=all_roots_agree and trust_agreement and rank==dimension
    info=dict(original)
    info.update(coefficients=beta.tolist(),nll=nll,mean_nll=nll/len(rows),
        lognormal_reference_sigma=math.sqrt(variance),
        baselines=[dict(group=list(g),log_reference_ms=mu[g],rows=sum(r['group']==g for r in rows)) for g in groups],
        training_log_reference_residual_rmse=float(np.sqrt(np.mean(residual**2))),
        original_BFGS_diagnostic=original,solver='analytic_gradient_root_LM_with_independent_exact_Hessian_trust_minimization',
        original_optimizer_flags_preserved=True,optimizer_flag_scope='original_BFGS_not_relabelled_by_second_order_solver',
        trust_exact_attempts=attempts,gradient_root_attempts=roots,
        successful_trust_minimizers=len(minimizers),all_gradient_roots_agree=all_roots_agree,
        root_solution_max_distance=agreement,trust_stationary_solution_agrees=trust_agreement,
        stationarity_tolerance=STATIONARITY_TOLERANCE,solution_agreement_tolerance=SOLUTION_AGREEMENT_TOLERANCE,
        numerical_tolerances_fixed_without_heldout_labels=True,
        second_order_qualification=qualified,stationarity_certificate=local_minimum_certificate(*differential(beta,rows)[1:]),
        joint_response_jacobian=dict(rank=rank,parameters=dimension,full_column_rank=rank==dimension,
            includes_baseline_nuisance_means=True,includes_sigma=False,singular_values=singular.tolist(),
            condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else None))
    return info,beta,mu,variance


def run(paths,prior_report):
    prior_ref=first.receipt(prior_report);prior=first.checked(prior_ref)
    first.require(prior['schema']==first.SCHEMA and prior['validation_labels_read'] is False,
        'Original training diagnostic required')
    docs=[];sources=[prior_ref]
    for path in paths:
        doc,closure=first.verified_document(path);docs.append(doc);sources.extend(closure)
    captures=[{m['capture_sha256'] for r in d['requests'] for m in r['macros']} for d in docs]
    first.require(len(docs)==3 and all(len(x)==1 for x in captures) and len(set.union(*captures))==3,
        'Exactly three distinct natural training captures required')
    results=[]
    for candidate,features in first.CANDIDATES.items():
        rows=first.prepared(docs,features);full,beta,mu,variance=second_order_fit(rows,candidate)
        folds=[];records=[]
        for heldout in sorted(set.union(*captures)):
            fitted,b,m,v=second_order_fit([r for r in rows if r['capture']!=heldout],candidate)
            evaluation=first.evaluate([r for r in rows if r['capture']==heldout],b,m,v)
            records.extend(evaluation['records'])
            folds.append(dict(heldout_capture_sha256=heldout,training_fit=fitted,evaluation=evaluation))
        eligible=full['second_order_qualification'] and all(f['training_fit']['second_order_qualification']
            and f['evaluation']['summary']['unknown_rows']==0 for f in folds)
        full.update(whole_capture_leaveout=folds,pooled_leaveout=first.summarize(records),
            leaveout_selection_eligible=eligible,full_fit_and_all_folds_numerically_qualified=eligible,
            coefficient_capture_stability=dict(fold_coefficients=[f['training_fit']['coefficients'] for f in folds],
                max_minus_min=np.ptp(np.asarray([f['training_fit']['coefficients'] for f in folds]),axis=0).tolist()),
            all_capture_conditional_fit=first.evaluate(rows,beta,mu,variance))
        results.append(full)
    eligible=[r for r in results if r['leaveout_selection_eligible']]
    selected=min(eligible,key=lambda r:np.mean([f['evaluation']['summary']['mean_nll'] for f in r['whole_capture_leaveout']]))['candidate'] if eligible else None
    closure=strict_references([sources,implementation_sources(__file__)])
    first.require(all(first.receipt(r['path'])==r for r in closure),'Source changed during second order fit')
    metadata={k:v for k,v in prior.items() if k not in ('schema','status','sources','candidates','selected_candidate','selection_rule')}
    return dict(metadata,schema=SCHEMA,status='SECOND_ORDER_THREE_TRAINING_CAPTURE_CONDITIONAL_MACRO_DIAGNOSTICS',
        sources=closure,candidates=results,selected_candidate=selected,
        selection_rule='equal_capture_heldout_conditional_NLL_full_fit_AND_all_folds_stationary_positive_curvature_full_rank_supported',
        original_report_preserved=prior_ref,numerical_solver_uses_heldout_labels=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-macros',nargs='+',required=True)
    parser.add_argument('--prior-report',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.training_macros,args.prior_report)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],selected_candidate=result['selected_candidate'])))
