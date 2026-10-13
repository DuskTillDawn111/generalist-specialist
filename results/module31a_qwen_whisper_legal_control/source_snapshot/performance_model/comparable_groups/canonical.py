"""Capacity-feasible canonical load-balancing routing (planning only)."""
import math
import numpy as np
from scipy.optimize import linprog
from deployment_core.workload import TASKS, Workload


def solve_canonical_routing(deployment, workload, catalog, rho_max=.8):
    """Minimize maximum standalone utilization z, preserving formal req/s routing."""
    models = sorted(m for m,n in deployment.items() if n)
    tasks = [k for k in TASKS if workload.get(k, 0) > 0]
    pairs = [(k,m) for k in tasks for m in models if k in catalog[m].supported_tasks]
    if not tasks or any(not any(k == t for t,m in pairs) for k in tasks):
        raise ValueError('Canonical routing has uncovered task')
    # r variables followed by z. This is the exact min-max LP formulation.
    zidx = len(pairs); n = zidx + 1
    eq = np.zeros((len(tasks), n)); beq=[]
    task_i={k:i for i,k in enumerate(tasks)}
    for j,(k,m) in enumerate(pairs): eq[task_i[k],j]=1.
    beq=np.array([float(workload[k]) for k in tasks])
    ub=[]; bub=[]
    for m in models:
        row=np.zeros(n)
        for j,(k,mm) in enumerate(pairs):
            if mm==m: row[j]=catalog[m].service_time_ms[k]/1000.
        row[zidx]=-1.; ub.append(row); bub.append(0.)
        # x_m=1 in the prototype, so rho <= rho_max.
        ub.append(np.eye(1,n,zidx)[0]); bub.append(rho_max)
    bounds=[(0.,float(workload[k])) for k,m in pairs]+[(0.,rho_max)]
    result=linprog(np.r_[np.zeros(zidx),1.],A_ub=np.array(ub),b_ub=np.array(bub),
                   A_eq=eq,b_eq=beq,bounds=bounds,method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success or result.x is None or not np.isfinite(result.x).all():
        raise ValueError(f'Canonical routing infeasible or solver failure: {result.message}')
    routing={k:{m:0. for m in catalog} for k in TASKS}
    for (k,m),value in zip(pairs,result.x[:zidx]): routing[k][m]=float(np.clip(value,0.,workload[k]))
    utilization={m:math.fsum(routing[k][m]*catalog[m].service_time_ms[k]/1000. for k in tasks if k in catalog[m].supported_tasks) for m in models}
    if any(v>rho_max+1e-8 for v in utilization.values()): raise ValueError('Canonical capacity violation')
    for k in tasks:
        if not math.isclose(sum(routing[k].values()),workload[k],rel_tol=1e-7,abs_tol=1e-12): raise ValueError('Canonical conservation violation')
    for k in TASKS:
        if k not in tasks: routing[k]={m:0. for m in catalog}
    return routing, utilization, float(result.x[zidx])
