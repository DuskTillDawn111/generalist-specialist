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
    if not x or not {'generalist','specialist'}<=set(roles[m] for m in x):errors.append('MIXED_RESIDENCY')
    for m,n in x.items():
        if str(n) not in d['capacity_factors'][m] or n<=0:errors.append('MEASURED_COUNT')
    j=evaluate_surrogate(x,{(t,m):q for t,row in rr.items() for m,q in row.items()},rates,c,cal)
    p=predict_scenarios(d,x,rr,rates,budget,refs,weights)
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

def formal_v3(d,rates,budget,refs,weights):
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
    try:
        result=engine.search(Workload(rates),c,Hardware(budget/d['safe_factor']),frozen_config(),roles=roles,
            capacity_mode='empirical',memory_mode='empirical',calibration=cal)
        seconds=time.perf_counter()-started;data=snapshot(engine,result,Workload(rates))
        x={m:n for m,n in result.deployment.items() if n};fit=engine.serving_routing_adapter.ranker.evaluate(result.deployment,Workload(rates))
        rr={t:{} for t in rates}
        for (t,m),q in fit['routing'].items():
            if q>1e-9:rr[t][m]=q
        p,j,errors=evaluate(d,x,rr,rates,budget,refs,weights,(c,roles,cal))
        return dict(status='COMPLETED_CPU_FORECAST_BACKEND',decision=dict(deployment=x,routing=rr,prediction=p,J=j['J_seconds'],errors=errors),
            result=data,seconds=seconds,objective_calls=4*len(backend.calls),backend_calls=backend.calls,
            J_route_solves=len(engine.e2e_ranker.cache)+len(engine.serving_routing_adapter.ranker.cache),
            scope='Unmodified formal V3, CPU performance forecast replaces physical RuntimeEvaluator; acceptance remains original mean/P95/memory and 5% gate, not U or J global optimization.')
    except Exception as exc:
        import traceback
        return dict(status='FORMAL_V3_FAILED',error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-started,backend_calls=backend.calls)

def main():
    manifest=read(OUT/'frozen_problem_manifest.json');space=read(manifest['space']['path']);definition=read(manifest['utility_definition']['path']);refs=read(manifest['utility_reference']['path'])['latency_reference_ms'];weights=definition['weights']
    oracles=[];decisions=[];regrets=[];runtime=[];deprows=[];allgpu=[]
    for name,pool in space['pools'].items():
        d=read(pool['path']);cp=catalog(d);models=sorted(d['models']);K=space['route_grid_denominator'];budget=space['budget_mib']
        for load,L in space['loads'].items():
            rates=L['rates'];started=time.perf_counter();rows=[];states=0;covered=0;capacity_rejected=0
            for counts in itertools.product(*(sorted(map(int,d['capacity_factors'][m])) for m in models)):
                states+=1;x={m:n for m,n in zip(models,counts) if n}
                mem=d['safe_factor']*sum(d['memory_table_mib'][m][str(n)] for m,n in x.items())
                cover={t for m in x for t in d['models'][m]['supported_tasks']}
                reasons=[]
                if not set(rates)<=cover:reasons.append('COVERAGE')
                if mem>budget+1e-6:reasons.append('MEMORY')
                if not {'generalist','specialist'}<=set(cp[1][m] for m in x):reasons.append('MIXED')
                deprows.append(dict(pool=name,load=load,deployment=x,safe_memory_mib=mem,pre_route_valid=not reasons,reject_reasons=reasons))
                # Model/count states failing independent route-free constraints have zero legal routes.
                # Retain every state; enumerate each structurally legal routing, including capacity rejects.
                if reasons:continue
                covered+=1
                for rr in candidate_routes(d,x,rates,K):
                    p,j,errors=evaluate(d,x,rr,rates,budget,refs,weights,cp)
                    row=dict(pool=name,load=load,candidate_id='C'+hashlib.sha256(identity(x,rr).encode()).hexdigest()[:16],
                        deployment=x,routing=rr,predicted_U=p['U'],predicted_J_seconds=j['J_seconds'],eligible=p['eligible'],
                        V3_rho08_eligible=p['eligible'] and p['max_rho']<=.8+1e-8,
                        safe_memory_mib=p.get('safe_memory_mib'),max_rho=p.get('max_rho'),prediction=p)
                    rows.append(row)
                    capacity_rejected+=not p['eligible']
            seconds=time.perf_counter()-started;valid=[r for r in rows if r['eligible']];v3valid=[r for r in valid if r['V3_rho08_eligible']]
            assert valid,'ORACLE_SPACE_INVALID'
            Ubest=min(valid,key=lambda r:(-r['predicted_U'],r['safe_memory_mib'],identity(r['deployment'],r['routing'])))
            Jbest=min(v3valid,key=lambda r:(r['predicted_J_seconds'],r['safe_memory_mib'],identity(r['deployment'],r['routing'])))
            oracle=dict(pool=name,load=load,total_deployment_vectors=states,coverage_memory_legal_deployments=covered,
                enumerated_deployment_route_candidates=len(rows),feasible_candidates=len(valid),capacity_rejected=capacity_rejected,
                objective_calls=4*len(rows),seconds=seconds,U_oracle=Ubest,J_rho08_oracle=Jbest,
                verification='Every finite routing evaluated with identical original predict_scenarios. Structural invalid vectors retained in feasible_deployments.csv.')
            oracles.append(oracle);save('cpu/'+name+'_'+load+'_oracle.json',oracle)
            csvsave('cpu/'+name+'_'+load+'_all_candidates.csv',[{k:v for k,v in r.items() if k!='prediction'} for r in rows])
            print('ORACLE',name,load,'vectors',states,'routes',len(rows),'legal',len(valid),'seconds',seconds,flush=True)
            fits=[]
            for rep in range(manifest['runtime_benchmark_repetitions']):
                fit=optimizer(d,rates,budget,refs,weights,K);fits.append(fit)
                runtime.append(dict(pool=name,load=load,repetition=rep+1,models=len(models),algorithm='20A_FINITE_DOMAIN_ADAPTER',seconds=fit['seconds'],MILP_calls=fit['MILP_calls'],
                    prediction_calls=fit['objective_calls'],unique_candidates=fit['unique_evaluated'],oracle_seconds=seconds,speedup=seconds/fit['seconds'],
                    visited_feasible_ratio=sum(c['prediction']['eligible'] for c in fit['candidates'])/len(valid),all_grid_routes=len(rows)))
            fit=fits[0];save('cpu/'+name+'_'+load+'_optimizer_grid.json',fit)
            native=optimizer(d,rates,budget,refs,weights,K,native=True);save('cpu/'+name+'_'+load+'_optimizer_native20A.json',native)
            formal=formal_v3(d,rates,budget,refs,weights);save('cpu/'+name+'_'+load+'_formal_V3.json',formal)
            for label,f in [('20A_FINITE_DOMAIN_ADAPTER',fit),('20A_NATIVE_CONTINUOUS',native)]:
                w=f['winner'];assert w
                inrows=[r for r in valid if identity(r['deployment'],r['routing'])==w['id']]
                decisions.append(dict(pool=name,load=load,algorithm=label,deployment=w['deployment'],routing=w['routing'],predicted_U=w['prediction']['U'],predicted_J_seconds=w['J'],
                    seconds=f['seconds'],objective_calls=f['objective_calls'],MILP_calls=f['MILP_calls'],in_finite_space=bool(inrows),constraint_violations=w['errors']))
                if inrows:
                    r=inrows[0];regrets.append(dict(pool=name,load=load,algorithm=label,oracle_U=Ubest['predicted_U'],selected_U=r['predicted_U'],U_regret=Ubest['predicted_U']-r['predicted_U'],
                        normalized_cost_regret=(Ubest['predicted_U']-r['predicted_U'])/abs(Ubest['predicted_U']),same_structure=set(Ubest['deployment'])==set(r['deployment']),
                        same_replicas=Ubest['deployment']==r['deployment'],same_routing=identity(Ubest['deployment'],Ubest['routing'])==identity(r['deployment'],r['routing']),
                        J_regret_vs_rho08_oracle=r['predicted_J_seconds']-Jbest['predicted_J_seconds'],finite_scope=True))
                else:regrets.append(dict(pool=name,load=load,algorithm=label,U_regret=None,finite_scope=False,reason='NATIVE_CONTINUOUS_DOMAIN_MISMATCH; no rounding or false finite optimality claim'))
            if formal.get('decision'):
                w=formal['decision'];inrows=[r for r in valid if identity(r['deployment'],r['routing'])==identity(w['deployment'],w['routing'])]
                decisions.append(dict(pool=name,load=load,algorithm='FORMAL_V3_CPU_BACKEND',deployment=w['deployment'],routing=w['routing'],predicted_U=w['prediction']['U'],predicted_J_seconds=w['J'],seconds=formal['seconds'],objective_calls=formal['objective_calls'],in_finite_space=bool(inrows),constraint_violations=w['errors']))
                regrets.append(dict(pool=name,load=load,algorithm='FORMAL_V3_CPU_BACKEND',oracle_U=Ubest['predicted_U'],selected_U=w['prediction']['U'],
                    U_regret=Ubest['predicted_U']-w['prediction']['U'] if inrows else None,
                    J_regret_vs_rho08_oracle=w['J']-Jbest['predicted_J_seconds'] if inrows else None,
                    finite_scope=bool(inrows),reason='Formal acceptance is noise-aware raw latency, not scalar J/U; off-grid choices not certified by grid oracle'))
            # Explicit comparator algorithms, never used to prove generator coverage.
            static=[r for r in valid if all(n==1 for n in r['deployment'].values()) and all(len(rr)==1 for rr in r['routing'].values())]
            for label,r in [('J_ROUTING_GRID_REFERENCE',Jbest),('MIN_MEMORY_THEN_GRID_J',min(v3valid,key=lambda r:(r['safe_memory_mib'],r['predicted_J_seconds']))),
                ('FASTEST_SERVICE_STATIC',min(static,key=lambda r:sum(q*d['models'][m]['service_time_ms'][t] for t,rr in r['routing'].items() for m,q in rr.items()))),
                ('BEST_ONE_REPLICA_NO_SPLIT_STATIC',max(static,key=lambda r:r['predicted_U']))]:
                decisions.append(dict(pool=name,load=load,algorithm=label,deployment=r['deployment'],routing=r['routing'],predicted_U=r['predicted_U'],predicted_J_seconds=r['predicted_J_seconds'],in_finite_space=True))
            if name=='SMALL6' and load=='ASR32B':
                assert len(valid)==manifest['GPU_protocol']['expected_feasible_candidates']
                allgpu=valid;save('gpu_candidates.json',valid)
            print('SOLVERS',name,load,'gridU',fit['winner']['prediction']['U'],'oracleU',Ubest['predicted_U'],'V3',formal['status'],flush=True)
    save('cpu_oracles.json',oracles);csvsave('feasible_deployments.csv',deprows)
    csvsave('cpu_oracle_results.csv',[{k:v for k,v in o.items() if k not in ['U_oracle','J_rho08_oracle']}|dict(U_oracle=o['U_oracle'],J_rho08_oracle=o['J_rho08_oracle']) for o in oracles])
    csvsave('optimizer_decisions.csv',decisions);save('optimizer_decisions.json',decisions);csvsave('predicted_regret.csv',regrets);save('predicted_regret.json',regrets)
    csvsave('optimizer_runtime_benchmark.csv',runtime);save('CPU_stage_completed.json',dict(passed=True,oracle_cells=len(oracles),new_GPU_runs=0,grid_regrets=[r for r in regrets if r['algorithm']=='20A_FINITE_DOMAIN_ADAPTER']))
if __name__=='__main__':main()
