"""Exact discrete capacity envelope: design audit, never a runtime neighborhood.

HiGHS MILP selects every evidence-supported count directly. It is equivalent to
enumerating the frozen count space and solving a capacity LP for each deployment.
Capacity ceilings use D <= g, not the methods' separate frozen rho_max=0.8.
"""
import math,itertools,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint,linprog

def solve_envelope(catalog,roles,cal,budget,background,target,s_only,safety_factor,capacity_ceiling=1.):
    models=sorted(catalog);tasks=sorted(background)
    selectors=[(m,n) for m in models for n in range(cal.evidence_bounds[m]+1)]
    paths=[(k,m) for k in tasks for m in models if k in catalog[m].supported_tasks and not(s_only and k==target and roles[m]=='generalist')]
    N=len(selectors)+len(paths);a=[];lo=[];hi=[]
    def row():return np.zeros(N)
    for m in models:
        v=row()
        for i,(model,n) in enumerate(selectors):
            if model==m:v[i]=1
        a.append(v);lo.append(1);hi.append(1)
    for k in tasks:
        if k==target:continue
        v=row()
        for j,(task,m) in enumerate(paths):
            if task==k:v[len(selectors)+j]=1
        a.append(v);lo.append(background[k]);hi.append(background[k])
    for m in models:
        v=row()
        for i,(model,n) in enumerate(selectors):
            if model==m:v[i]=-capacity_ceiling*cal.capacity_factors[m][n]
        for j,(k,model) in enumerate(paths):
            if model==m:v[len(selectors)+j]=catalog[m].service_time_ms[k]/1000
        a.append(v);lo.append(-np.inf);hi.append(0)
    memory=row()
    for i,(m,n) in enumerate(selectors):memory[i]=safety_factor*cal.memory_table_mib[m][n]/budget
    a.append(memory);lo.append(-np.inf);hi.append(1.)
    lb=np.zeros(N);ub=np.r_[np.ones(len(selectors)),np.full(len(paths),np.inf)]
    integrality=np.r_[np.ones(len(selectors)),np.zeros(len(paths))]
    target_vector=row()
    for j,(k,m) in enumerate(paths):
        if k==target:target_vector[len(selectors)+j]=1
    start=time.perf_counter();solutions=[]
    def run(objective):
        result=milp(objective,integrality=integrality,bounds=Bounds(lb,ub),constraints=LinearConstraint(np.array(a),lo,hi),options={'mip_rel_gap':0.,'presolve':True,'time_limit':120.})
        solutions.append({'status':int(result.status),'message':result.message,'mip_gap':getattr(result,'mip_gap',None)})
        return result
    optimum=run(-target_vector)
    if optimum.status==2:return {'status':'BACKGROUND_INFEASIBLE','capacity_rps':None,'solver_stages':solutions}
    assert optimum.success,(target,budget,solutions)
    capacity=float(target_vector@optimum.x)
    # Lexicographic design ties: least safe memory, then replica count, then the
    # integer deployment vector. No fitted or weighted performance score.
    tol=1e-6  # req/s: solver feasibility safety, never a workload/performance knob.
    a.append(target_vector);lo.append(capacity-tol);hi.append(np.inf)
    result=run(memory);assert result.success,solutions
    integer_memory=sum(memory[i] for i in range(len(selectors)) if result.x[i]>.5)
    a.append(memory);lo.append(-np.inf);hi.append(float(integer_memory)+1e-9)
    counts=row()
    for i,(m,n) in enumerate(selectors):counts[i]=n
    result=run(counts);assert result.success,solutions
    a.append(counts);lo.append(round(float(counts@result.x)));hi.append(lo[-1])
    vector=row();stride=1
    for m in reversed(models):
        for i,(model,n) in enumerate(selectors):
            if model==m:vector[i]=stride*n
        stride*=cal.evidence_bounds[m]+1
    result=run(vector);assert result.success,solutions
    x={m:sum(n for i,(model,n) in enumerate(selectors) if model==m and result.x[i]>.5) for m in models}
    # Fix the chosen deployment and maximize again, removing MILP tie tolerances
    # from the reported witness routing and pressure.
    active=[p for p in paths if x[p[1]]]
    ae=[];be=[]
    for k in tasks:
        if k==target:continue
        ae.append([float(t==k) for t,m in active]);be.append(background[k])
    au=[[catalog[m].service_time_ms[k]/1000 if model==m else 0. for k,model in active] for m in models]
    bu=[capacity_ceiling*cal.capacity_factors[m][x[m]] for m in models]
    lp=linprog([-float(k==target) for k,m in active],A_ub=au,b_ub=bu,A_eq=ae,b_eq=be,bounds=(0,None),method='highs')
    assert lp.success,(target,x,lp.message)
    routing={f'{k}::{m}':float(v) for (k,m),v in zip(active,lp.x)}
    final_capacity=-float(lp.fun)
    assert abs(final_capacity-capacity)<=max(1e-5,capacity*1e-7),(final_capacity,capacity)
    D={m:math.fsum(v*catalog[m].service_time_ms[k]/1000 for (k,model),v in zip(active,lp.x) if model==m) for m in models if x[m]}
    rho={m:D[m]/cal.capacity_factors[m][x[m]] for m in D}
    assert max(rho.values())<=capacity_ceiling+1e-7
    for k in tasks:
        if k!=target:assert abs(sum(v for (t,m),v in zip(active,lp.x) if t==k)-background[k])<1e-7
    safe=safety_factor*math.fsum(cal.memory_table_mib[m][x[m]] for m in models)
    assert safe<=budget+1e-5
    assert not s_only or all(v<=1e-8 for (k,m),v in zip(active,lp.x) if k==target and roles[m]=='generalist')
    target_G=sum(v for (k,m),v in zip(active,lp.x) if k==target and roles[m]=='generalist')
    target_S=final_capacity-target_G
    return {'status':'OPTIMAL','capacity_rps':final_capacity,'milp_upper_bound_rps':capacity,'deployment':x,'routing':routing,
            'safe_memory_mib':safe,'D':D,'rho':rho,'target_G_rps':target_G,'target_S_rps':target_S,
            'target_G_fraction':target_G/final_capacity if final_capacity else None,'target_S_fraction':target_S/final_capacity if final_capacity else None,
            'N_G':sum(x[m] for m in models if roles[m]=='generalist'),'N_S':sum(x[m] for m in models if roles[m]=='specialist'),
            'G_types':sum(x[m]>0 for m in models if roles[m]=='generalist'),'specialist_replicas':{m:x[m] for m in models if roles[m]=='specialist' and x[m]},
            'solver_stages':solutions,'wall_seconds':time.perf_counter()-start,'capacity_ceiling_rho':capacity_ceiling}
