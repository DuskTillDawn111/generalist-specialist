"""Exact empirical-count-space MILP, followed by a fixed-state LP witness.

The scalar direction is fixed before this solver runs. No g(x) extrapolation.
Missing empirical counts are evidence gaps, never physical replica constraints.
"""
import math
import time
import numpy as np
from scipy.optimize import milp, linprog, Bounds, LinearConstraint
from .common import safe_memory, rates, serial

def fixed_lp(x, background, heads, catalog, roles, cal, mode, alpha=None):
    tasks=sorted(background); models=sorted(m for m,n in x.items() if n)
    paths=[(k,m) for k in tasks for m in models if k in catalog[m].supported_tasks
           and not(mode=='S' and k in heads and roles[m]=='generalist')]
    if any(not any(t==k for t,m in paths) for k in tasks):return None
    extend=alpha is None
    eq=np.array([[float(t==k) for t,m in paths]+([-heads.get(k,0.)] if extend else []) for k in tasks])
    ub=np.array([[catalog[m].service_time_ms[k]/1000 if model==m else 0. for k,model in paths]
                 +([0.] if extend else []) for m in models])
    rhs=[background[k] if extend else background[k]+alpha*heads.get(k,0.) for k in tasks]
    obj=np.zeros(len(paths)+int(extend))
    if extend:obj[-1]=-1
    r=linprog(obj,A_eq=eq,b_eq=rhs,A_ub=ub,
        b_ub=[.8*cal.capacity_factors[m][x[m]] for m in models],bounds=(0,None),method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if r.status==2:return None
    assert r.success,r.message
    a=float(r.x[-1]) if extend else alpha
    route={(k,m):float(v) for (k,m),v in zip(paths,r.x)}
    D={m:math.fsum(v*catalog[m].service_time_ms[k]/1000 for (k,n),v in route.items() if n==m) for m in models}
    rho={m:D[m]/cal.capacity_factors[m][x[m]] for m in models}
    lam=rates(background,heads,a)
    conservation=max(abs(math.fsum(v for (k,m),v in route.items() if k==t)-lam[t]) for t in tasks)
    assert conservation<1e-7 and max(rho.values())<=.8+1e-9,(conservation,rho)
    return {'alpha':a,'deployment':x,'routing':serial(route),'D':D,'rho':rho,
            'safe_memory_mib':safe_memory(x,cal),'lambda_rps':lam,
            'N_G':sum(n for m,n in x.items() if roles[m]=='generalist'),
            'N_S':sum(n for m,n in x.items() if roles[m]=='specialist'),
            'specialist_replicas':{m:n for m,n in x.items() if n and roles[m]=='specialist'},
            'hotspot_G_fraction':{k:math.fsum(v for (t,m),v in route.items() if t==k and roles[m]=='generalist')/lam[k] for k in heads},
            'max_conservation_residual_rps':conservation,'max_capacity_residual':max(D[m]-.8*cal.capacity_factors[m][x[m]] for m in models)}

def solve(catalog,roles,cal,budget,background,heads,mode,safety_factor):
    start=time.perf_counter(); models=sorted(catalog);tasks=sorted(background)
    selectors=[(m,n) for m in models for n in range(cal.evidence_bounds[m]+1)]
    paths=[(k,m) for k in tasks for m in models if k in catalog[m].supported_tasks
           and not(mode=='S' and k in heads and roles[m]=='generalist')]
    N=len(selectors)+len(paths)+1; a=[];lo=[];hi=[]
    def row():return np.zeros(N)
    for m in models:
        v=row()
        for i,(model,n) in enumerate(selectors):
            if model==m:v[i]=1
        a.append(v);lo.append(1);hi.append(1)
    for k in tasks:
        v=row();v[-1]=-heads.get(k,0.)
        for j,(task,m) in enumerate(paths):
            if task==k:v[len(selectors)+j]=1
        a.append(v);lo.append(background[k]);hi.append(background[k])
    for m in models:
        v=row()
        for i,(model,n) in enumerate(selectors):
            if model==m:v[i]=-.8*cal.capacity_factors[m][n]
        for j,(k,model) in enumerate(paths):
            if model==m:v[len(selectors)+j]=catalog[m].service_time_ms[k]/1000
        a.append(v);lo.append(-np.inf);hi.append(0)
    memory=row();counts=row();G=row()
    for i,(m,n) in enumerate(selectors):
        memory[i]=safety_factor*cal.memory_table_mib[m][n]/budget
        counts[i]=n;G[i]=n*int(roles[m]=='generalist')
    a.append(memory);lo.append(-np.inf);hi.append(1.)
    if mode!='S':a.append(G);lo.append(-np.inf);hi.append(int(mode[0]))
    stages=[]
    def run(obj):
        r=milp(obj,integrality=np.r_[np.ones(len(selectors)),np.zeros(len(paths)+1)],
               bounds=Bounds(np.zeros(N),np.r_[np.ones(len(selectors)),np.full(len(paths)+1,np.inf)]),
               constraints=LinearConstraint(np.array(a),lo,hi),options={'mip_rel_gap':0.,'time_limit':120.})
        stages.append({'status':int(r.status),'message':r.message,'gap':float(r.mip_gap) if r.success else None})
        return r
    av=row();av[-1]=1;first=run(-av)
    if first.status==2:return {'status':'BACKGROUND_INFEASIBLE','alpha':None,'stages':stages}
    assert first.success,stages
    optimum=float(first.x[-1]);a.append(av);lo.append(max(0.,optimum-1e-8));hi.append(np.inf)
    r=run(memory);assert r.success,stages
    mem=sum(memory[i] for i in range(len(selectors)) if r.x[i]>.5)
    a.append(memory);lo.append(-np.inf);hi.append(mem+1e-9)
    r=run(counts);assert r.success,stages
    a.append(counts);lo.append(round(float(counts@r.x)));hi.append(lo[-1])
    vector=row();stride=1
    for m in reversed(models):
        for i,(model,n) in enumerate(selectors):
            if model==m:vector[i]=stride*n
        stride*=cal.evidence_bounds[m]+1
    r=run(vector);assert r.success,stages
    x={m:sum(n for i,(model,n) in enumerate(selectors) if model==m and r.x[i]>.5) for m in models}
    witness=fixed_lp(x,background,heads,catalog,roles,cal,mode)
    assert witness and abs(witness['alpha']-optimum)<1e-7,(mode,optimum,witness)
    assert witness['safe_memory_mib']<=budget+1e-6
    assert mode=='S' or witness['N_G']<=int(mode[0])
    return {'status':'OPTIMAL_EMPIRICAL_SPACE',**witness,'milp_alpha_upper_bound':optimum,
            'stages':stages,'seconds':time.perf_counter()-start,'rho_limit':.8}
