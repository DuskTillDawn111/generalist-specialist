"""Exact finite oracle, unchanged U predictor, explicitly bounded domain adapter."""
from .common import *
import numpy as np,types
from module20.optimizer import Generator,predict_scenarios,fingerprint
from scipy.optimize import Bounds,LinearConstraint,milp
from module14c.common import catalog as historical_catalog
from deployment_core.routing.e2e_surrogate import evaluate_surrogate

def catalog(d):
    # Match native ModelProfile's exact service-key invariant without editing
    # frozen source values: the excluded ASR profile remains in the manifest.
    projected=copy.deepcopy(d)
    for model in projected['models'].values():
        model['service_time_ms']={t:v for t,v in model['service_time_ms'].items() if t in model['supported_tasks']}
    return historical_catalog(projected)

def compositions(k,total):
    if k==1:yield (total,);return
    for first in range(total+1):
        for tail in compositions(k-1,total-first):yield (first,*tail)

def candidate_routes(d,x,rates,K):
    options={t:[m for m in sorted(x) if t in d['models'][m]['supported_tasks']] for t in rates}
    if not all(options.values()):return
    for rows in itertools.product(*(list(compositions(len(v),K)) for v in options.values())):
        yield {t:{m:rates[t]*u/K for m,u in zip(options[t],units) if u} for t,units in zip(options,rows)}

def evaluate(d,x,rr,rates,budget,refs,weights,calparts=None):
    c,roles,cal=calparts or catalog(d)
    errors=[]
    if not x:errors.append('EMPTY_DEPLOYMENT')
    for m,n in x.items():
        if str(n) not in d['capacity_factors'][m] or n<=0:errors.append('MEASURED_COUNT')
    j=evaluate_surrogate(x,{(t,m):q for t,row in rr.items() for m,q in row.items()},rates,c,cal)
    p=predict_scenarios(d,x,rr,rates,budget,refs,weights)
    if p.get('max_rho',float('inf'))>.8+1e-8:errors.append('COMMON_RHO08')
    if errors:p.update(eligible=False,U=None)
    return p,j,errors

class GridGenerator(Generator):
    """Only MILP domain/variable-unit adaptation; original solve code executes."""
    def __init__(self,d,refs,K):
        super().__init__(d,refs);self.K=K
    def solve(self,rates,budget,target,fraction,cap,weights):
        N=len(self.states);units=np.array([1.]*N+[rates[t]/self.K for t,m in self.paths])
        def finite_milp(c,integrality,bounds,constraints,options):
            fit=milp(c*units,integrality=np.ones(len(c)),
                bounds=Bounds(bounds.lb/units,bounds.ub/units),
                constraints=LinearConstraint(constraints.A*units,constraints.lb,constraints.ub),options=options)
            if fit.x is not None:fit.x=fit.x*units
            return fit
        fn=types.FunctionType(Generator.solve.__code__,{**Generator.solve.__globals__,'milp':finite_milp})
        return fn(self,rates,budget,target,fraction,cap,weights)

def optimizer(d,rates,budget,refs,weights,K,native=False):
    started=time.perf_counter();gen=Generator(d,refs) if native else GridGenerator(d,refs,K);candidates={};calls=[]
    for fraction,cap,wm in itertools.product([None,0.,.01,.05,.2,.5,1.],[.8,.95],[.05,.15,.3]):
        fit=gen.solve(rates,budget,'asr',fraction,cap,{**weights,'M':wm});calls.append(fit)
        if 'deployment' in fit:candidates.setdefault(fingerprint(fit['deployment'],fit['routing']),dict(deployment=fit['deployment'],routing=fit['routing']))
    calparts=catalog(d)
    for candidate in candidates.values():
        p,j,errors=evaluate(d,candidate['deployment'],candidate['routing'],rates,budget,refs,weights,calparts)
        candidate.update(prediction=p,J=j['J_seconds'],id=identity(candidate['deployment'],candidate['routing']),errors=errors)
    legal=[c for c in candidates.values() if c['prediction']['eligible']]
    # Original search tie: max U, lower safe memory, deterministic candidate order.
    winner=max(legal,key=lambda c:(c['prediction']['U'],-c['prediction']['safe_memory_mib'])) if legal else None
    return dict(winner=winner,candidates=list(candidates.values()),generation_calls=calls,
        seconds=time.perf_counter()-started,objective_calls=4*len(candidates),unique_evaluated=len(candidates),
        MILP_calls=len(calls),domain='NATIVE_CONTINUOUS' if native else 'FROZEN_FINITE_GRID',core_source=ref(ROOT/'module20/optimizer.py'))

def formal_v3(d,rates,budget,refs,weights,grid=False,K=1):
    """Production V3 autonomous search, with explicit CPU forecast runtime backend."""
    from module9bc.frozen_v3 import create,snapshot
    from deployment_search.runtime import RuntimeEvaluator,RuntimeEvaluation,weighted_task_latency
    from deployment_core import Workload,Hardware
    from method_v2 import frozen_config
    class Backend(RuntimeEvaluator):
        def __init__(self):self.calls=[]
        def evaluate(self,x,workload,routing):
            rr={t:{} for t in rates}
            for (t,m),q in routing.items():
                if q>1e-9:rr[t][m]=q
            p=predict_scenarios(d,{m:n for m,n in x.items() if n},rr,rates,budget,refs,weights)
            self.calls.append(dict(deployment=x,routing=rr,prediction=p))
            if not p['eligible']:return RuntimeEvaluation('unstable',None,None,0.,0.)
            mean=sum(rates[t]*p['per_task'][t]['mean_ms'] for t in rates)/sum(rates.values())
            tail=sum(rates[t]*p['per_task'][t]['p95_ms'] for t in rates)/sum(rates.values())
            return RuntimeEvaluation('steady',mean,tail,p['predicted_within_window_QPS'],1.,raw_result_path=None)
    backend=Backend();c,roles,cal=catalog(d);engine=create(backend,c,cal);started=time.perf_counter()
    if grid:
        from .grid_control import install
        install(engine,d,c,cal,K)
    try:
        result=engine.search(Workload(rates),c,Hardware(budget/d['safe_factor']),frozen_config(),roles=roles,
            capacity_mode='empirical',memory_mode='empirical',calibration=cal)
        seconds=time.perf_counter()-started;data=snapshot(engine,result,Workload(rates))
        x={m:n for m,n in result.deployment.items() if n};fit=engine.serving_routing_adapter.ranker.evaluate(result.deployment,Workload(rates))
        rr={t:{} for t in rates}
        for (t,m),q in fit['routing'].items():
            if q>1e-9:rr[t][m]=q
        p,j,errors=evaluate(d,x,rr,rates,budget,refs,weights,(c,roles,cal))
        return dict(status='COMPLETED_CPU_FORECAST_BACKEND',decision=dict(deployment=x,routing=rr,prediction=p,J=j['J_seconds'],errors=errors),grid_control=grid,
            result=data,seconds=seconds,objective_calls=4*len(backend.calls),backend_calls=backend.calls,
            J_route_solves=len(engine.e2e_ranker.cache)+len(engine.serving_routing_adapter.ranker.cache),
            scope='Unmodified formal V3, CPU performance forecast replaces physical RuntimeEvaluator; acceptance remains original mean/P95/memory and 5% gate, not U or J global optimization.')
    except Exception as exc:
        import traceback
        return dict(status='FORMAL_V3_FAILED',error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-started,backend_calls=backend.calls)
