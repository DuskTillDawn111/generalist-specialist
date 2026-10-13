"""Guarded extension of the project utility optimizer; never a new objective.

Stage D is deliberately unreachable without the frozen predictive gate. This
adapter keeps Module20 Generator candidates and competes every legal measured
placement with unconstrained all-task routes scored by the new predictor.
"""
from .common import *
from module20.optimizer import Generator
from scipy.optimize import differential_evolution,minimize

def legal_deployments(d,rates,budget):
    models=sorted(d['models']);tasks=set(rates);suffix=[set() for _ in range(len(models)+1)]
    for i in range(len(models)-1,-1,-1):suffix[i]=suffix[i+1]|set(d['models'][models[i]]['supported_tasks'])
    def visit(i,x,memory,covered):
        if not tasks<=covered|suffix[i]:return
        if i==len(models):
            if tasks<=covered:yield dict(x)
            return
        m=models[i]
        for n in sorted(map(int,d['capacity_factors'][m])):
            cost=d['safe_factor']*d['memory_table_mib'][m][str(n)]
            if memory+cost>budget+1e-6:continue
            if n:x[m]=n
            yield from visit(i+1,x,memory+cost,covered|(set(d['models'][m]['supported_tasks']) if n else set()))
            x.pop(m,None)
    yield from visit(0,{},0.,set())

def domain(d,x,rates,target,mode):
    options={}
    for t in rates:
        paths=[m for m in sorted(x) if t in d['models'][m]['supported_tasks']]
        if t==target and mode!='FREE':
            role='generalist' if mode=='G-only' else 'specialist'
            paths=[m for m in paths if d['models'][m]['role']==role]
        if not paths:return None
        options[t]=paths
    return options

def routes(options,rates,z):
    cursor=0;rr={}
    for t,paths in options.items():
        remaining=rates[t];row={}
        for m in paths[:-1]:
            q=remaining*float(z[cursor]);cursor+=1
            if q>1e-12:row[m]=q
            remaining-=q
        if remaining>1e-12:row[paths[-1]]=remaining
        rr[t]=row
    return rr

def optimize(d,rates,budget,refs,weights,predictor,validation,target='asr'):
    if not validation['passed']:raise RuntimeError('MODULE25_PREDICTIVE_GATE_FAILED: Stage D forbidden')
    started=time.perf_counter();records=[];seeds=[]
    for cap in [.8,.95]:
        candidate=Generator(d,refs).solve(rates,budget,target,None,cap,weights)
        if 'deployment' in candidate:
            candidate['prediction']=predictor.predict(d,candidate['deployment'],candidate['routing'],rates,budget,refs,weights)
        seeds.append(candidate)
    for x in legal_deployments(d,rates,budget):
        modes={}
        for mode in ['G-only','S-only','FREE']:
            options=domain(d,x,rates,target,mode)
            if options is None:modes[mode]=dict(status='QUALITY_COVERAGE_INFEASIBLE');continue
            dimensions=sum(len(v)-1 for v in options.values());candidates=[]
            def objective(z):
                rr=routes(options,rates,z);p=predictor.predict(d,x,rr,rates,budget,refs,weights)
                if p['U'] is not None:candidates.append(dict(deployment=x,routing=rr,prediction=p,mode=mode))
                return -p['U'] if p['U'] is not None else 1e6
            if dimensions:
                for z in itertools.product([0.,1.],repeat=dimensions):objective(z)
                objective(np.full(dimensions,.5))
                for seed in [25051,25052]:
                    fit=differential_evolution(objective,[(0.,1.)]*dimensions,seed=seed,popsize=5,maxiter=25,
                        tol=1e-7,atol=1e-10,polish=False);objective(fit.x)
                if candidates:
                    leader=max(candidates,key=lambda c:c['prediction']['U'])
                    # DE already competes all endpoint classes; no phase/split constraint.
            else:objective([])
            modes[mode]=max(candidates,key=lambda c:c['prediction']['U']) if candidates else dict(status='NO_ELIGIBLE_PREDICTION')
        records.append(dict(deployment=x,modes=modes))
    winners=[r for row in records for r in row['modes'].values() if 'prediction' in r]
    winner=min(winners,key=lambda c:(-c['prediction']['U'],c['prediction']['safe_memory_mib'],identity(c['deployment']))) if winners else None
    return dict(winner=winner,records=records,original_Generator=seeds,wall_seconds=time.perf_counter()-started,
        no_forced_phases=True,all_legal_replica_deployments_competed=True,continuous_routing_global_certificate=False,
        utility_definition='Original20A components; quality hard; new predictor changes forecasts only')
