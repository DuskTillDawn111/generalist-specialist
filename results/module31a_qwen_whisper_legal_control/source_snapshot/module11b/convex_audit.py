"""CPU mathematical J* certificate over discrete measured deployments.

This is an offline design/numerical reference, NOT a replacement for the frozen
Module9 router. The frozen router is independently evaluated on the winning
deployment; its operational rejection remains visible and is never repaired.

f(D,g)=g*f(D/g) is convex and homogeneous. Its tangent at rho=a is
q >= (D-a*a*g)/(1-a)**2. Adaptive tangent MILPs give a global lower bound;
feasible routes give upper bounds. No feature, weight, fitted surrogate, or
unmeasured count enters this certificate.
"""
import time,math
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint,minimize,linprog
from deployment_core import Workload
from deployment_search.routing import solve_canonical_routing
from deployment_core.routing.e2e_surrogate import evaluate_surrogate
from deployment_core.routing.e2e_router import solve_e2e_routing
from module11b.common import factor
from module11.latency import certificate

TOLERANCE_SECONDS=1e-8

def reference_route(x,lam,c,cal,canonical=None):
    workload=Workload(lam);models=sorted(m for m,n in x.items() if n);tasks=sorted(lam)
    pairs=[(k,m) for k in tasks for m in models if k in c[m].supported_tasks]
    initial=canonical or solve_canonical_routing(x,workload,c,replica_bounds=cal.evidence_bounds,capacity_mode='empirical',calibration=cal)
    if initial is None:return None
    eq=np.array([[float(k==t) for k,m in pairs] for t in tasks])
    service=np.array([[lam[k]*c[m].service_time_ms[k]/1000 if model==m else 0. for k,m in pairs] for model in models])
    g=np.array([cal.capacity_factors[m][x[m]] for m in models]);total=math.fsum(lam.values())
    u0=np.array([initial.routing.get((k,m),0.)/lam[k] for k,m in pairs])
    def objective(u):
        D=service@u;return 1000*sum(D/np.maximum(1-D/g,1e-9))/total
    def gradient(u):
        D=service@u;return 1000*service.T@(1/np.maximum(1-D/g,1e-9)**2)/total
    constraints=[{'type':'eq','fun':lambda u:eq@u-1,'jac':lambda u:eq},
                 {'type':'ineq','fun':lambda u:.8*g-service@u,'jac':lambda u:-service}]
    r=minimize(objective,u0,jac=gradient,bounds=[(0,1)]*len(pairs),constraints=constraints,
        method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
    route={(k,m):float(v*lam[k]) for (k,m),v in zip(pairs,np.clip(r.x,0,1))}
    try:
        evaluation=evaluate_surrogate(x,route,workload,c,cal);cert=certificate(x,route,c,cal)
    except ValueError:return None
    if not cert['passed']:return None
    lp=linprog(gradient(r.x),A_eq=eq,b_eq=np.ones(len(tasks)),A_ub=service,b_ub=.8*g,
               bounds=[(0,1)]*len(pairs),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    gap=max(0.,float(gradient(r.x)@(r.x-lp.x))/1000) if lp.success else None
    return {'routing':route,'evaluation':evaluation,'certificate':cert,'solver_success':bool(r.success),
            'solver_message':r.message,'gap_seconds':gap,'converged':bool(r.success and gap is not None and gap<=TOLERANCE_SECONDS)}

def solve_discrete_J(lam,c,roles,cal,budget,max_G,excluded_deployments=()):
    begin=time.perf_counter();models=sorted(c);tasks=sorted(lam)
    selectors=[(m,n) for m in models for n in range(cal.evidence_bounds[m]+1)]
    paths=[(k,m) for k in tasks for m in models if k in c[m].supported_tasks]
    nsel=len(selectors);npath=len(paths);N=nsel+npath+len(models);total=math.fsum(lam.values())
    A=[];lo=[];hi=[]
    def row():return np.zeros(N)
    D={};G={};Q={};mem=row();ng=row()
    for m in models:
        select=row();g=row();d=row();q=row();q[nsel+npath+models.index(m)]=1
        for i,(name,n) in enumerate(selectors):
            if name==m:select[i]=1;g[i]=cal.capacity_factors[m][n]
        for j,(k,name) in enumerate(paths):
            if name==m:d[nsel+j]=c[m].service_time_ms[k]/1000
        A.append(select);lo.append(1);hi.append(1)
        A.append(d-.8*g);lo.append(-np.inf);hi.append(0)
        D[m]=d;G[m]=g;Q[m]=q
    for i,(m,n) in enumerate(selectors):
        mem[i]=factor()*cal.memory_table_mib[m][n]/budget;ng[i]=n*int(roles[m]=='generalist')
    A.extend([mem,ng]);lo.extend([-np.inf,-np.inf]);hi.extend([1,max_G])
    for k in tasks:
        v=row()
        for j,(t,m) in enumerate(paths):
            if t==k:v[nsel+j]=1
        A.append(v);lo.append(lam[k]);hi.append(lam[k])
    for x in excluded_deployments:
        v=row()
        for i,(m,n) in enumerate(selectors):
            if n==x.get(m,0):v[i]=1
        A.append(v);lo.append(-np.inf);hi.append(len(models)-1)
    cuts={m:set() for m in models}
    def cut(m,rho):
        rho=max(0.,min(.8,float(rho)));key=round(rho,14)
        if key in cuts[m]:return False
        cuts[m].add(key);A.append(D[m]-rho*rho*G[m]-(1-rho)**2*Q[m]);lo.append(-np.inf);hi.append(0)
        return True
    for m in models:cut(m,0);cut(m,.8)
    objective=sum(Q.values())*1000/total
    integ=np.r_[np.ones(nsel),np.zeros(npath+len(models))]
    bounds=Bounds(np.zeros(N),np.r_[np.ones(nsel),np.full(npath+len(models),np.inf)])
    best=None;history=[];fixed_cache={};iteration=0
    while True:
        iteration+=1
        cuts_before=sum(len(v) for v in cuts.values())
        result=milp(objective,integrality=integ,bounds=bounds,constraints=LinearConstraint(np.array(A),lo,hi),
                    options={'mip_rel_gap':0.,'time_limit':120.})
        if result.status==2:return {'status':'OPERATIONAL_PLANNING_INFEASIBLE','J_seconds':None,'iterations':iteration}
        assert result.success,(iteration,result.message)
        lower=float(result.mip_dual_bound)/1000
        if best:assert lower<=best['J_seconds']+TOLERANCE_SECONDS,'Numerical lower bound exceeds feasible upper bound'
        if best and best['J_seconds']-lower<=TOLERANCE_SECONDS:
            best.update(status='GLOBAL_J_CERTIFIED_CPU_REFERENCE',global_lower_bound_seconds=lower,
                global_gap_seconds=max(0.,best['J_seconds']-lower),iterations=iteration,
                seconds=time.perf_counter()-begin,history=history,mathematical_scope='same J objective, rho<=.8, all measured integer counts; CPU only')
            try:
                frozen=solve_e2e_routing(best['deployment'],Workload(lam),c,cal)
                cert=certificate(best['deployment'],frozen['routing'],c,cal) if frozen['converged'] else None
            except (ValueError,AssertionError) as exc:
                frozen={'status':'NUMERICAL_SOLVER_EXCEPTION','converged':False,'exception':repr(exc)};cert=None
            best['frozen_Module9_solver']={'status':frozen['status'],'converged':frozen['converged'],
                'evaluation':frozen.get('evaluation'),'operational_certificate':cert,
                'routing':{f'{k}::{m}':v for (k,m),v in (frozen.get('routing') or {}).items()},
                'reference_route_is_not_a_GPU_route_replacement':True}
            if 'exception' in frozen:best['frozen_Module9_solver']['exception']=frozen['exception']
            return best
        x={m:sum(n for i,(name,n) in enumerate(selectors) if name==m and result.x[i]>.5) for m in models}
        route={(k,m):float(result.x[nsel+j]) for j,(k,m) in enumerate(paths) if x[m]}
        cert=certificate(x,route,c,cal)
        values=[]
        try:evaluation=evaluate_surrogate(x,route,lam,c,cal)
        except ValueError:evaluation=None
        if cert['passed'] and evaluation:values.append((route,evaluation,cert,'MILP_epigraph_feasible_route'))
        key=tuple(x.items())
        if key not in fixed_cache:
            fixed_cache[key]=reference_route(x,lam,c,cal)
        local=fixed_cache[key]
        if local:values.append((local['routing'],local['evaluation'],local['certificate'],'fixed_state_convex_reference'))
        for r,ev,ce,source in values:
            rho=ev['rho']
            for m in rho:cut(m,rho[m])
            if best is None or ev['J_seconds']<best['J_seconds']:
                best={'deployment':x,'routing':{f'{k}::{m}':v for (k,m),v in r.items()},'J_seconds':ev['J_seconds'],
                    'rho':rho,'max_rho':ev['z'],'D':ev['D'],'safe_memory_mib':factor()*sum(cal.memory_table_mib[m][n] for m,n in x.items()),
                    'N_G':sum(n for m,n in x.items() if roles[m]=='generalist'),'certificate':ce,'upper_bound_source':source}
        for m in models:
            gg=float(G[m]@result.x);dd=float(D[m]@result.x)
            if gg:cut(m,dd/gg)
        history.append({'iteration':iteration,'MILP_lower_bound_seconds':lower,'best_upper_bound_seconds':best['J_seconds'] if best else None,
            'MILP_primal_epigraph_seconds':float(result.fun)/1000,
            'candidate_deployment':x,'tangent_cuts':sum(len(v) for v in cuts.values()),'MILP_mip_gap':float(result.mip_gap)})
        assert sum(len(v) for v in cuts.values())>cuts_before or (best and best['J_seconds']-lower<=TOLERANCE_SECONDS),'Numerical tangent certificate stalled; retain unresolved status'
        assert iteration<2000,'Numerical certificate did not converge; do not manufacture a J optimum'

def solve_frozen_J(lam,c,roles,cal,budget,max_G):
    """Exact offline frozen-router selection, pruned by mathematical lower bounds.

    No route repair: rejected deployments are retained as frozen-solver failures.
    No-good cuts exclude only already evaluated identities, not a neighborhood.
    """
    excluded=[];visited=[];best=None;first_reference=None;begin=time.perf_counter()
    while True:
        reference=solve_discrete_J(lam,c,roles,cal,budget,max_G,excluded)
        if first_reference is None:first_reference=reference
        if reference['status']=='OPERATIONAL_PLANNING_INFEASIBLE':
            return {'status':'FROZEN_ROUTER_GLOBAL_BEST_CERTIFIED' if best else 'NO_OPERATIONALLY_ACCEPTED_FROZEN_ROUTE',
                'best':best,'mathematical_reference':first_reference,'visited':visited,
                'all_other_measured_deployments_exhausted':True,'seconds':time.perf_counter()-begin,
                'failure_is_not_physical_infeasibility':True}
        lower=reference['global_lower_bound_seconds']
        if best and lower>=best['J_seconds']-TOLERANCE_SECONDS:
            return {'status':'FROZEN_ROUTER_GLOBAL_BEST_CERTIFIED','best':best,'mathematical_reference':first_reference,
                'visited':visited,'remaining_global_lower_bound_seconds':lower,
                'remaining_gap_seconds':max(0.,best['J_seconds']-lower),'seconds':time.perf_counter()-begin}
        frozen=reference['frozen_Module9_solver'];cert=frozen.get('operational_certificate')
        accepted=bool(frozen['converged'] and cert and cert['passed'])
        record={'deployment':reference['deployment'],'frozen_solver':frozen,'accepted':accepted,
            'mathematical_J_seconds':reference['J_seconds']}
        if accepted:
            record['J_seconds']=frozen['evaluation']['J_seconds']
            if best is None or record['J_seconds']<best['J_seconds']:best=record
        visited.append(record);excluded.append(reference['deployment'])
        if best and best['J_seconds']-lower<=TOLERANCE_SECONDS:
            return {'status':'FROZEN_ROUTER_GLOBAL_BEST_CERTIFIED','best':best,'mathematical_reference':first_reference,
                'visited':visited,'remaining_global_lower_bound_seconds':lower,
                'remaining_gap_seconds':max(0.,best['J_seconds']-lower),'seconds':time.perf_counter()-begin}
