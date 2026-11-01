"""Independent scalar-likelihood audit of the real macro training fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import macro_joint_response_fit_v1 as first
from .implementation_source_closure_v27 import implementation_sources, strict_references


def run(report_path, macro_paths):
    ref=first.receipt(report_path);report=first.checked(ref)
    first.require(report['schema']=='gsdeploy.macro_joint_response_second_order_training_leaveout.v1'
        and report['validation_labels_read'] is False,'Second-order training report required')
    docs=[];sources=[ref]
    for path in macro_paths:
        doc,closure=first.verified_document(path);docs.append(doc);sources.extend(closure)
    results=[]
    for candidate in report['candidates']:
        rows=first.prepared(docs,tuple(candidate['features']))
        fits=[('all',candidate,rows)]
        seen=[]
        for fold in candidate['whole_capture_leaveout']:
            heldout=fold['heldout_capture_sha256']
            train=[r for r in rows if r['capture']!=heldout]
            first.require(heldout not in fold['training_fit']['captures'],'Heldout in fitted capture set')
            seen.extend((r['capture'],r['request_id'],r['macro_id']) for r in rows if r['capture']==heldout)
            fits.append((heldout,fold['training_fit'],train))
        first.require(len(seen)==len(rows) and len(set(seen))==len(rows),'Complete unique capture leaveout required')
        for capture,fit,training in fits:
            beta=np.asarray(fit['coefficients']);nll,mu,var,_,_=first.profile(beta,training)
            first.require(abs(nll-fit['nll'])<1e-9,'Scalar NLL differs from reported fitted solution')
            first.require(abs(var-fit['lognormal_reference_sigma']**2)<1e-12,'Response variance differs')
            for baseline in fit['baselines']:
                first.require(abs(mu[tuple(baseline['group'])]-baseline['log_reference_ms'])<1e-12,
                    'Profiled nuisance baseline differs')
            diagnostic=dict(candidate=candidate['candidate'],capture=capture,rows=len(training),
                scalar_nll_rederived=True,all_nuisance_means_rederived=True)
            if len(beta):
                # Differentiate only the scalar profile likelihood; this uses
                # neither the second-order solver nor its gradient/Hessian.
                objective=lambda b:first.profile(b,training)[0]/len(training)
                step=2e-4;basis=np.eye(len(beta))*step;hessian=np.empty((len(beta),len(beta)))
                gradient=np.asarray([(objective(beta+e/20)-objective(beta-e/20))/(step/10) for e in basis])
                for i,ei in enumerate(basis):
                    for j,ej in enumerate(basis):
                        hessian[i,j]=(objective(beta+ei+ej)-objective(beta+ei-ej)
                            -objective(beta-ei+ej)+objective(beta-ei-ej))/(4*step*step)
                reported=np.asarray(fit['stationarity_certificate']['hessian'])
                error=float(np.max(np.abs(hessian-reported)))
                first.require(np.linalg.norm(gradient)<1e-6 and error<1e-4
                    and np.linalg.eigvalsh(hessian)[0]>0,'Scalar finite difference audit failed')
                diagnostic.update(scalar_mean_nll_finite_difference_gradient=gradient.tolist(),
                    scalar_hessian_eigenvalues=np.linalg.eigvalsh(hessian).tolist(),
                    scalar_vs_analytic_hessian_max_error=error,
                    finite_difference_steps=dict(gradient=step/20,hessian=step),
                    finite_difference_scope='discretized_crosscheck_not_1e-10_stationarity_certificate')
            results.append(diagnostic)
    closure=strict_references([sources,implementation_sources(__file__)])
    first.require(all(first.receipt(r['path'])==r for r in closure),'Audit source changed')
    return dict(schema='gsdeploy.macro_response_scalar_independent_review.v1',status='PASS',
        sources=closure,checks=results,complete_macro_rows=len(rows),
        validation_labels_read=False,GPU_started=False,independent_accuracy_validated=False,
        main_queue_provider_eligible=False,
        limitation='Checks likelihood arithmetic and split integrity, not residual independence or physical accuracy.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True);parser.add_argument('--training-macros',nargs=3,required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run(args.report,args.training_macros)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],fits=len(result['checks']))))
