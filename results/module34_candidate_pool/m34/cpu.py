"""Nine fixed cells. Enumerate all structures, retaining every failure."""
from .common import *
from .cpu_core import evaluate,catalog,candidate_routes,optimizer,formal_v3

def cid(x,r):return 'D'+hashlib.sha256(identity(x,r).encode()).hexdigest()[:16]
def on_grid(r,rates,K):return all(abs(q/rates[t]*K-round(q/rates[t]*K))<1e-7 for t,row in r.items() for q in row.values())
def pick(rows):
    ok=[r for r in rows if r['eligible']]
    return min(ok,key=lambda r:(-r['predicted_U'],r['safe_memory_mib'],identity(r['deployment'],r['routing']))) if ok else None
def static(d,rates,minimum=False,budget=49140.):
    models=sorted(d['models']);states=[]
    if minimum:
        for bits in itertools.product([0,1],repeat=len(models)):
            x={m:1 for m,b in zip(models,bits) if b}
            if set(rates)<={t for m in x for t in d['models'][m]['supported_tasks']}:
                states.append((d['safe_factor']*sum(d['memory_table_mib'][m]['1'] for m in x),tuple(x)))
        choices=list(min(states)[1]) if states else []
    else:choices=models
    route={}
    for t,q in rates.items():
        available=[m for m in choices if t in d['models'][m]['supported_tasks']]
        if not available:return None
        m=min(available,key=lambda m:(d['models'][m]['service_time_ms'][t],d['memory_table_mib'][m]['1'],m));route[t]={m:q}
    return {m:1 for m in {m for row in route.values() for m in row}},route

def main():
    fm=read(OUT/'frozen_problem_manifest.json');sp=read(fm['space']['path']);weights=read(fm['utility_definition']['path'])['weights'];refs=read(fm['utility_reference']['path'])['latency_reference_ms'];K=sp['route_grid_denominator'];budget=sp['budget_mib']
    oracles=[];decisions=[];timings=[];states=[];expansion=[]
    for scenario,L in sp['scenarios'].items():
        previous=None
        for pool,pref in sp['pools'].items():
            d=read(pref['path']);cp=catalog(d);models=sorted(d['models']);rows=[];start=time.perf_counter();census=0
            for bits in itertools.product([0,1],repeat=len(models)):
                x={m:1 for m,b in zip(models,bits) if b};mem=d['safe_factor']*sum(d['memory_table_mib'][m]['1'] for m in x);cover={t for m in x for t in d['models'][m]['supported_tasks']};why=[]
                if not set(L['rates'])<=cover:why.append('COVERAGE')
                if mem>budget+1e-6:why.append('MEMORY')
                states.append(dict(pool=pool,scenario=scenario,deployment=x,safe_memory_mib=mem,reasons=why))
                if why:continue
                for r in candidate_routes(d,x,L['rates'],K):
                    p,j,err=evaluate(d,x,r,L['rates'],budget,refs,weights,cp);census+=1
                    rows.append(dict(pool=pool,scenario=scenario,candidate_id=cid(x,r),deployment=x,routing=r,structure=structure(d,x),predicted_U=p['U'],predicted_J_seconds=j['J_seconds'],eligible=p['eligible'],safe_memory_mib=p.get('safe_memory_mib'),max_rho=p.get('max_rho'),errors=err))
            elapsed=time.perf_counter()-start;best=pick(rows);jbest=min((r for r in rows if r['eligible']),key=lambda r:(r['predicted_J_seconds'],r['safe_memory_mib'],identity(r['deployment'],r['routing'])),default=None)
            oracle=dict(pool=pool,scenario=scenario,replica_vectors=2**len(models),evaluated_routes=census,feasible_routes=sum(r['eligible'] for r in rows),seconds=elapsed,prediction_calls=4*census,U_oracle=best,J_oracle=jbest)
            oracles.append(oracle);save(f'cpu/{scenario}_{pool}_oracle.json',oracle);csvsave(f'cpu/{scenario}_{pool}_candidates.csv',rows)
            print('ORACLE',scenario,pool,'routes',census,'seconds',elapsed,'winner',best['structure'] if best else None,'U',best['predicted_U'] if best else None,flush=True)
            def record(alg,plan,seconds=None,calls=None,status=None):
                if plan is None:
                    row=dict(pool=pool,scenario=scenario,algorithm=alg,status=status or 'INFEASIBLE',eligible=False,predicted_U=None,regret=None,seconds=seconds,objective_calls=calls)
                else:
                    x,r=plan;p,j,err=evaluate(d,x,r,L['rates'],budget,refs,weights,cp);grid=on_grid(r,L['rates'],K);member=any(identity(x,r)==identity(z['deployment'],z['routing']) for z in rows if z['eligible'])
                    row=dict(pool=pool,scenario=scenario,algorithm=alg,status='FEASIBLE' if p['eligible'] and grid and member else 'DOMAIN_MISMATCH' if not grid else 'CONSTRAINT_REJECT',
                        deployment=x,routing=r,candidate_id=cid(x,r),structure=structure(d,x),predicted_U=p['U'],predicted_J_seconds=j['J_seconds'],eligible=p['eligible'],finite_admissible=grid and member,
                        regret=best['predicted_U']-p['U'] if best and p['eligible'] and grid and member else None,
                        safe_memory_mib=p.get('safe_memory_mib'),max_rho=p.get('max_rho'),seconds=seconds,objective_calls=calls,errors=err)
                decisions.append(row);return row
            record('CPU_U_ORACLE',(best['deployment'],best['routing']) if best else None,elapsed,4*census)
            record('CPU_J_ORACLE',(jbest['deployment'],jbest['routing']) if jbest else None)
            for alg,grid in [('V3_NATIVE_CONTINUOUS_PRESERVED',False),('V3_J_GRID_CONTROL',True)]:
                fit=formal_v3(d,L['rates'],budget,refs,weights,grid=grid,K=K);save(f'cpu/{scenario}_{pool}_{alg}.json',fit)
                w=fit.get('decision');rec=record(alg,(w['deployment'],w['routing']) if w else None,fit['seconds'],fit.get('objective_calls'),fit['status'])
                timings.append(dict(pool=pool,scenario=scenario,algorithm=alg,repetition=1,seconds=fit['seconds'],prediction_calls=fit.get('objective_calls'),J_route_solves=fit.get('J_route_solves'),visited=len(fit['backend_calls']),grid=grid))
            for alg,nat in [('20A_U_GRID',False),('20A_NATIVE_CONTINUOUS_PRESERVED',True)]:
                reps=1 if nat else fm['runtime_repetitions']
                for rep in range(reps):
                    fit=optimizer(d,L['rates'],budget,refs,weights,K,native=nat);save(f'cpu/{scenario}_{pool}_{alg}_{rep+1}.json',fit)
                    w=fit['winner']
                    if rep==0:record(alg,(w['deployment'],w['routing']) if w else None,fit['seconds'],fit['objective_calls'])
                    timings.append(dict(pool=pool,scenario=scenario,algorithm=alg,repetition=rep+1,seconds=fit['seconds'],prediction_calls=fit['objective_calls'],MILP_calls=fit['MILP_calls'],visited=fit['unique_evaluated'],grid=not nat))
            for alg,minimum in [('FASTEST_PATH_STATIC',False),('MIN_MEMORY_COVER_FASTEST',True)]:
                begin=time.perf_counter();plan=static(d,L['rates'],minimum,budget);record(alg,plan,time.perf_counter()-begin,4 if plan else 0)
            for role in ['G_ONLY','S_ONLY']:
                b=pick([r for r in rows if r['structure']==role]);record(role+'_CLASS_ORACLE',(b['deployment'],b['routing']) if b else None,status='COVERAGE_OR_BUDGET_INFEASIBLE')
            expansion.append(dict(scenario=scenario,pool=pool,oracle_U=best['predicted_U'] if best else None,previous_oracle_U=previous,oracle_monotone=previous is None or best and best['predicted_U']>=previous-1e-9))
            if best:previous=best['predicted_U']
            save('cpu_oracles.json',oracles);save('optimizer_decisions.json',decisions);csvsave('optimizer_decisions.csv',decisions);csvsave('optimizer_runtime_benchmark.csv',timings);csvsave('feasible_deployments.csv',states);save('pool_expansion_audit.json',expansion)
    csvsave('cpu_oracle_results.csv',[{k:v for k,v in o.items()} for o in oracles]);csvsave('predicted_regret.csv',decisions)
    save('CPU_stage_completed.json',dict(cells=len(oracles),all_oracle_monotone=all(r['oracle_monotone'] for r in expansion),algorithm_refit=False))
if __name__=='__main__':main()
