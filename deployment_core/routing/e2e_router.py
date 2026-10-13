"""Deterministic convex J solver with a linear global-optimality certificate."""
import time
import numpy as np
from scipy.optimize import minimize, linprog
from deployment_search.routing import solve_canonical_routing, solve_pressure_relaxation
from .e2e_surrogate import evaluate_surrogate

NUMERICAL_EPSILON = 1e-9
CERTIFICATE_TOLERANCE_SECONDS = 1e-8

def solve_e2e_routing(deployment, workload, catalog, calibration, canonical=None):
    start = time.perf_counter()
    models = sorted(m for m,n in deployment.items() if n)
    tasks = sorted(workload.active_tasks)
    pairs = [(k,m) for k in tasks for m in models if k in catalog[m].supported_tasks]
    if any(not any(t==k for t,m in pairs) for k in tasks):
        return {'status':'COVERAGE_INFEASIBLE','converged':False,'routing':None}
    if not tasks:
        return {'status':'SOLVED','converged':True,'routing':{},'evaluation':evaluate_surrogate(deployment,{},workload,catalog,calibration),'optimality_gap_seconds':0.,'solve_seconds':time.perf_counter()-start}
    initial = canonical or solve_pressure_relaxation(deployment,workload,catalog,
                  replica_bounds=calibration.evidence_bounds,capacity_mode='empirical',calibration=calibration)
    if initial is None or initial.max_pressure >= 1.-NUMERICAL_EPSILON:
        return {'status':'CAPACITY_INFEASIBLE','converged':False,'routing':None,'solve_seconds':time.perf_counter()-start}
    lam=np.array([workload.arrival_rate[k] for k in tasks]); total=lam.sum()
    eq=np.array([[float(t==k) for t,m in pairs] for k in tasks])
    service=np.array([[workload.arrival_rate[k]*catalog[m].service_time_ms[k]/1000. if name==m else 0.
                      for k,m in pairs] for name in models])
    capacity=np.array([calibration.capacity_factors[m][deployment[m]] for m in models])
    initial_u=np.array([initial.routing.get((k,m),0.)/workload.arrival_rate[k] for k,m in pairs])
    def objective(u):
        d=service@u;denom=1.-d/capacity
        # Smooth continuation only for infeasible trial iterates; the returned
        # witness must be in the original domain and passes a convex certificate.
        safe=np.maximum(denom,NUMERICAL_EPSILON)
        # Express the identical objective in milliseconds for conditioning.
        return float(1000.*np.sum(d/safe)/total)
    def gradient(u):
        d=service@u;denom=np.maximum(1.-d/capacity,NUMERICAL_EPSILON)
        return 1000.*service.T@(1./denom**2)/total
    constraints=[{'type':'eq','fun':lambda u:eq@u-1.,'jac':lambda u:eq},
                 {'type':'ineq','fun':lambda u:capacity*(1.-NUMERICAL_EPSILON)-service@u,'jac':lambda u:-service}]
    result=minimize(objective,initial_u,jac=gradient,bounds=[(0.,1.)]*len(pairs),
                    constraints=constraints,method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
    u=np.clip(result.x,0.,1.)
    routing={(k,m):float(v*workload.arrival_rate[k]) for (k,m),v in zip(pairs,u)}
    evaluation=evaluate_surrogate(deployment,routing,workload,catalog,calibration)
    feasible=np.max(np.abs(eq@u-1.))<=1e-7 and np.all(service@u<=capacity*(1.-NUMERICAL_EPSILON)+1e-9) and evaluation['domain_valid']
    cert=linprog(gradient(u),A_eq=eq,b_eq=np.ones(len(tasks)),A_ub=service,
                 b_ub=capacity*(1.-NUMERICAL_EPSILON),bounds=[(0.,1.)]*len(pairs),method='highs',
                 options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    gap=max(0.,float(gradient(u)@(u-cert.x))/1000.) if cert.success else None
    converged=bool(result.success and feasible and gap is not None and gap<=CERTIFICATE_TOLERANCE_SECONDS)
    initial_eval=evaluate_surrogate(deployment,initial.routing,workload,catalog,calibration)
    if evaluation['J_seconds'] is None or evaluation['J_seconds']>initial_eval['J_seconds']+1e-10:
        converged=False
    return {'status':'SOLVED' if converged else 'NUMERICAL_SOLVER_FAILURE','converged':converged,
            'routing':routing if converged else None,'evaluation':evaluation,'initial_evaluation':initial_eval,
            'solver_success':bool(result.success),'solver_message':result.message,'iterations':int(result.nit),
            'constraints_valid':bool(feasible),'optimality_gap_seconds':gap,
            'solve_seconds':time.perf_counter()-start,'initialization':'canonical minmax routing',
            'epsilon':NUMERICAL_EPSILON,'random_restarts':0}
