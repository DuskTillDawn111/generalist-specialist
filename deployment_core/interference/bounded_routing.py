"""Module 9B: fixed-deployment, bounded routing under a frozen evaluator.

This is a surrogate routing heuristic, not a steady-state or E2E certificate.
No worker launch, model selection, coefficient fit or deployment search occurs.
"""
from copy import deepcopy
import math
import time
import numpy as np
from scipy.optimize import linprog

class BoundedRoutingError(RuntimeError):
    pass

def _score(evaluation, routing, workload):
    peak=max(evaluation['effective_utilization'].values(),default=0.)
    total=math.fsum(workload.values())
    service=math.fsum(rate*evaluation['effective_service_time'][k][m]
                      for k,rates in routing.items() for m,rate in rates.items())/total if total else 0.
    return peak,service

def _improves(candidate,current):
    return candidate[0]<current[0]-1e-9 or (abs(candidate[0]-current[0])<=1e-9 and candidate[1]<current[1]-1e-7)

def _lp(estimator,x,workload,effective,rho_max,remaining):
    models=sorted(m for m,n in x.items() if n)
    tasks=sorted(k for k,rate in workload.items() if rate>0)
    pairs=[(k,m) for k in tasks for m in models if (m,k) in estimator.by_pair]
    if any(not any(k==task for k,m in pairs) for task in tasks):return None,'UNSUPPORTED_TASK'
    if not tasks:return {},'ZERO_WORKLOAD'
    dimension=len(pairs)+1
    aeq=np.zeros((len(tasks),dimension));aub=np.zeros((len(models),dimension))
    latency=np.zeros(dimension)
    for index,(k,m) in enumerate(pairs):
        aeq[tasks.index(k),index]=1.
        aub[models.index(m),index]=effective[k][m]/1000
        latency[index]=effective[k][m]
    for index,m in enumerate(models):aub[index,-1]=-estimator.capacity[m][str(x[m])]
    bounds=[(0.,workload[k]) for k,m in pairs]+[(0.,rho_max)]
    objective=np.zeros(dimension);objective[-1]=1.
    options={'time_limit':max(remaining,1e-4),'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    started=time.perf_counter()
    first=linprog(objective,A_eq=aeq,b_eq=[workload[k] for k in tasks],A_ub=aub,b_ub=np.zeros(len(models)),bounds=bounds,method='highs',options=options)
    if first.status==2:return None,'SURROGATE_LP_INFEASIBLE'
    if first.status==1:return None,'LP_TIME_LIMIT'
    if not first.success:raise BoundedRoutingError(f'HiGHS {first.status}: {first.message}')
    # Tie-break on weighted service time within the minimum-pressure solution.
    # Keep the primary bound up to numerical tolerance; never raise rho_max.
    bounds[-1]=(0.,min(rho_max,float(first.x[-1])+1e-9))
    remaining-=time.perf_counter()-started
    if remaining>1e-4:
        options['time_limit']=remaining
        second=linprog(latency,A_eq=aeq,b_eq=[workload[k] for k in tasks],A_ub=aub,b_ub=np.zeros(len(models)),bounds=bounds,method='highs',options=options)
        if second.success:chosen=second
        elif second.status==1:chosen=first
        else:raise BoundedRoutingError(f'Secondary HiGHS {second.status}: {second.message}')
    else:chosen=first
    if chosen.x is None or not np.all(np.isfinite(chosen.x)):
        raise BoundedRoutingError('Nonfinite LP solution')
    route={k:{} for k in workload}
    for (k,m),value in zip(pairs,chosen.x):route[k][m]=float(np.clip(value,0.,workload[k]))
    estimator.evaluate(x,route,workload) # Independently verify support and conservation.
    for m in models:
        demand=math.fsum(rates.get(m,0)*effective[k][m]/1000 for k,rates in route.items() if m in effective.get(k,{}))
        if demand>rho_max*estimator.capacity[m][str(x[m])]+1e-8:
            raise BoundedRoutingError('LP witness exceeds frozen capacity bound')
    return route,'SOLVED'

def solve_fixed_deployment_routing(estimator,deployment,workload,initial_routing,
                                   safe_memory_by_model,memory_budget_mib,
                                   rho_max=.8,max_iterations=3,damping=.5,
                                   runtime_budget_seconds=6.):
    """Nested routing in req/s. x, lambda, empirical g and memory budget stay fixed."""
    x=deepcopy(deployment.counts if hasattr(deployment,'counts') else deployment)
    rates=dict(workload.arrival_rate if hasattr(workload,'arrival_rate') else workload)
    original=deepcopy(initial_routing);best=deepcopy(initial_routing)
    if not (0<rho_max<=1) or not isinstance(max_iterations,int) or isinstance(max_iterations,bool) or max_iterations<1:
        raise ValueError('Invalid routing bounds')
    if not math.isfinite(damping) or not 0<damping<=1 or not math.isfinite(runtime_budget_seconds) or runtime_budget_seconds<=0:
        raise ValueError('Invalid iteration limits')
    if not math.isfinite(memory_budget_mib) or memory_budget_mib<=0:
        raise ValueError('Invalid memory budget')
    evaluation=estimator.evaluate(x,best,rates)
    memory=math.fsum(safe_memory_by_model[m][str(n)] for m,n in x.items() if n)
    if memory>memory_budget_mib+1e-6:raise ValueError('Deployment exceeds the unchanged safe memory budget')
    start=time.perf_counter();initial_score=best_score=_score(evaluation,best,rates)
    trace=[];reason='MAX_ITERATIONS';seen=[]
    for iteration in range(1,max_iterations+1):
        remaining=runtime_budget_seconds-(time.perf_counter()-start)
        if remaining<=0:reason='RUNTIME_BUDGET';break
        proposed,status=_lp(estimator,x,rates,evaluation['effective_service_time'],rho_max,remaining)
        if proposed is None:
            trace.append({'iteration':iteration,'LP_status':status,'accepted':False});reason=status;break
        candidate={k:{m:(1-damping)*best.get(k,{}).get(m,0.)+damping*proposed.get(k,{}).get(m,0.)
                      for m in sorted(set(best.get(k,{}))|set(proposed.get(k,{})))} for k in rates}
        candidate_eval=estimator.evaluate(x,candidate,rates);candidate_score=_score(candidate_eval,candidate,rates)
        accepted=_improves(candidate_score,best_score)
        trace.append({'iteration':iteration,'LP_status':status,'previous_score':list(best_score),'candidate_score':list(candidate_score),'accepted':accepted,
                      're_evaluated_after_routing_update':True})
        if not accepted:reason='NO_IMPROVEMENT';break
        signature=tuple((k,m,round(rate,12)) for k,r in sorted(candidate.items()) for m,rate in sorted(r.items()))
        if signature in seen:reason='NUMERICAL_CYCLE';break
        seen.append(signature);best=candidate;evaluation=candidate_eval;best_score=candidate_score
    assert original==initial_routing
    final=estimator.evaluate(x,best,rates)
    assert not _improves(initial_score,_score(final,best,rates)), 'Best surrogate witness worsened'
    return {'deployment':x,'workload':rates,'routing_rps':best,'evaluation':final,'initial_score':list(initial_score),'final_score':list(best_score),
            'score_definition':['maximum effective utilization / empirical g','arrival-weighted mean effective service time ms'],
            'rho_max':rho_max,'feasible_by_surrogate':best_score[0]<=rho_max+1e-8,'stop_reason':reason,'trace':trace,
            'iterations':len(trace),'runtime_seconds':time.perf_counter()-start,'memory_safe_mib':memory,'memory_budget_mib':memory_budget_mib,
            'actual_runtime_or_E2E_certificate':False,'selected_model':estimator.variant,'coefficients_refitted':False,
            'no_hard_candidate_pruning':True,'multi_model_interference_validated':False}
