"""Exhaustive measured placements, free route optimization with original U.

All endpoints compete. Multiple deterministic starts, DE seeds and a dense ASR
line provide numerical convergence checks, not a global continuous certificate.
"""
from .common import *
from .replay import replay
from scipy.optimize import differential_evolution, minimize, linprog
from module20.optimizer import Generator,predict,predict_scenarios

def best(fs):
    valid=[f for f in fs if f.get('prediction',{}).get('eligible')]
    return min(valid,key=lambda f:(-f['prediction']['U'],f['prediction']['safe_memory_mib'],identity([f['deployment'],f['routing']]))) if valid else None

def search(d,refs,weights,model=None,label='original'):
    start=time.perf_counter();states=[];legal=[]
    for counts in itertools.product(*(list(map(int,d['capacity_factors'][m])) for m in MODELS)):
        x={m:n for m,n in zip(MODELS,counts) if n};mem=d['safe_factor']*sum(d['memory_table_mib'][m][str(n)] for m,n in x.items())
        cover={t for m in x for t in d['models'][m]['supported_tasks']}
        reasons=[]
        if mem>BUDGET+1e-6:reasons.append('SAFE_MEMORY_BUDGET')
        if set(RATES['LOW'])-cover:reasons.append('TASK_COVERAGE')
        states.append(dict(deployment=x,safe_memory_mib=mem,reject_reasons=reasons))
        if not reasons:legal.append(x)
    save(label+'_all_deployment_states.json',states)
    stages={};perf=[]
    def evaluate(x,rr,rates):
        if model:return replay(d,x,rr,rates,BUDGET,refs,weights,model,detail=False)[0]
        return predict(d,x,rr,rates,BUDGET,refs,weights,phase=.37)
    for stage,rates in RATES.items():
        ss=time.perf_counter();fits=[];counts=0
        for xi,x in enumerate(legal):
            bounds=[(0.,1.) if sp in x else (1.,1.) for sp in [TEXT,IMAGE,S]]
            free=[i for i,b in enumerate(bounds) if b[0]!=b[1]];base=np.array([b[0] for b in bounds]);candidates=[];cache={}
            def ev(z):
                nonlocal counts
                key=tuple(round(float(v),11) for v in z)
                if key not in cache:
                    rr=route(d,x,rates,z);p=evaluate(x,rr,rates);counts+=1
                    cache[key]=(p,rr)
                    if p['eligible']:candidates.append(dict(deployment=x,routing=rr,shares=list(map(float,z)),ASR_G_fraction=float(z[2]),prediction=p,candidate_id=f'{stage}_D{xi:03d}'))
                return -cache[key][0]['U'] if cache[key][0]['eligible'] else 1e4+max(0,cache[key][0].get('max_rho',1)-1)*1e4
            def expand(v):
                z=base.copy();z[free]=v;return z
            def obj(v):return ev(expand(v))
            # Vertices and interior grid explicitly include pure routes.
            for v in itertools.product([0.,.5,1.],repeat=len(free)):obj(v)
            diag=[]
            if free:
                for seed in [24001,24002]:
                    fit=differential_evolution(obj,[bounds[i] for i in free],seed=seed,popsize=5,maxiter=25,
                                               tol=1e-7,atol=1e-10,polish=False)
                    obj(fit.x);diag.append(dict(seed=seed,nfev=fit.nfev,success=bool(fit.success),best_negative_U=float(fit.fun)))
                leaders=sorted(candidates,key=lambda f:-f['prediction']['U'])[:2]
                for c in leaders:
                    fit=minimize(obj,np.array(c['shares'])[free],method='Powell',bounds=[bounds[i] for i in free],
                                 options=dict(maxiter=20,xtol=1e-7,ftol=1e-9));obj(fit.x)
                # Pure ASR lines optimize other shares too: all corners, dense ASR line at leaders.
                for c in (leaders[:1] or [dict(shares=base)]):
                    if S in x:
                        for f in np.linspace(0,1,101):z=np.array(c['shares']);z[2]=f;ev(z)
                for f in ([0.,1.] if S in x else [1.]):
                    other=[i for i in free if i!=2];z=base.copy();z[2]=f
                    if other:
                        def pureobj(v):
                            zz=z.copy();zz[other]=v;return ev(zz)
                        ft=differential_evolution(pureobj,[(0.,1.)]*len(other),seed=24003,popsize=4,maxiter=15,polish=False)
                        pureobj(ft.x)
                    else:ev(z)
            winner=best(candidates)
            modes={name:best([c for c in candidates if rule(c['ASR_G_fraction'])]) for name,rule in
                   [('G-only',lambda f:f>=1-1e-9),('S-only',lambda f:f<=1e-9),('G/S',lambda f:1e-9<f<1-1e-9)]}
            fits.append(dict(deployment=x,candidate_id=f'{stage}_D{xi:03d}',winner=winner,
                             route_class_best=modes,evaluations=len(cache),solver_diagnostics=diag))
            if (xi+1)%15==0:print(label,stage,xi+1,'/',len(legal),flush=True)
        allfits=[c for f in fits for c in [f['winner'],*f['route_class_best'].values()] if c]
        winner=best(allfits)
        if winner is None:
            detail=dict(rates=rates,winner=None,choice='NO_ELIGIBLE_PREDICTION',route_class_best={k:None for k in ['G-only','S-only','G/S']},
                        candidates=fits,autonomous=True,phase_constraints=None,
                        reason='Every measured placement lacks an eligible prediction; no artificial finite U assigned',
                        evaluations=counts,wall_seconds=time.perf_counter()-ss,routing_global_certificate=False)
            stages[stage]=detail;save(label+'_stage_'+stage+'.json',detail)
            perf.append(dict(stage=stage,algorithm=label,seconds=detail['wall_seconds'],evaluations=counts))
            print('SELECTED',label,stage,'NO_ELIGIBLE_PREDICTION',flush=True)
            continue
        # Further refine all routing dimensions in top placements, with independent seed.
        contenders=sorted([f['winner'] for f in fits if f['winner']],key=lambda f:-f['prediction']['U'])[:5]
        refine=[]
        for c in contenders:
            x=c['deployment'];bounds=[(0,1) if sp in x else (1,1) for sp in [TEXT,IMAGE,S]]
            def finalobj(z):
                nonlocal counts
                rr=route(d,x,rates,z);p=evaluate(x,rr,rates);counts+=1
                if p['eligible']:refine.append(dict(deployment=x,routing=rr,shares=list(map(float,z)),ASR_G_fraction=float(z[2]),prediction=p,candidate_id=c['candidate_id']+'_refined'))
                return -p['U'] if p['eligible'] else 1e5
            ft=differential_evolution(finalobj,bounds,seed=24011,popsize=7,maxiter=60,tol=1e-9,polish=False);finalobj(ft.x)
        before=winner;winner=best([*allfits,*refine]);improvement=winner['prediction']['U']-before['prediction']['U']
        allfits+=refine
        # Original free MILP candidates, with unchanged generation weights.
        seeds=[]
        if not model:
            for cap in [.8,.95]:
                fit=Generator(d,refs).solve(rates,BUDGET,'asr',None,cap,weights)
                if 'deployment' in fit:
                    fit['prediction']=predict_scenarios(d,fit['deployment'],fit['routing'],rates,BUDGET,refs,weights)
                    fit['ASR_G_fraction']=fit['routing']['asr'].get(G,0)/rates['asr']
                    fit['shares']=[fit['routing'][t].get(G,0)/rates[t] for t in ['text','image','asr']];allfits.append(fit)
                seeds.append(fit)
            winner=best(allfits)
        modes={name:best([c for c in allfits if rule(c['ASR_G_fraction'])]) for name,rule in
               [('G-only',lambda f:f>=1-1e-9),('S-only',lambda f:f<=1e-9),('G/S',lambda f:1e-9<f<1-1e-9)]}
        f=winner['ASR_G_fraction'];choice='G-only' if f>=1-1e-9 else 'S-only' if f<=1e-9 else 'G/S'
        p,paths=replay(d,winner['deployment'],winner['routing'],rates,BUDGET,refs,weights,model)
        if not model:assert abs(p['U']-winner['prediction']['U'])<1e-10
        detail=dict(rates=rates,winner=winner,choice=choice,route_class_best=modes,candidates=fits,
                    refinement_improvement_U=improvement,refinement_evaluations=len(refine),original_free_MILP=seeds,
                    predicted_worker_paths=paths,autonomous=True,phase_constraints=None,
                    reason='Highest eligible predicted U among all measured placements and freely competing routes; no phase or 18% share requirement',
                    routing_global_certificate=False,route_numerical_convergence_U=improvement,
                    evaluations=counts,wall_seconds=time.perf_counter()-ss)
        stages[stage]=detail;save(label+'_stage_'+stage+'.json',detail)
        print('SELECTED',label,stage,choice,winner['deployment'],winner['shares'],winner['prediction']['U'],flush=True)
        perf.append(dict(stage=stage,algorithm=label,seconds=detail['wall_seconds'],evaluations=counts))
    result=dict(algorithm=label,CPU_only=True,weights=weights,refs_ms=refs,budget_mib=BUDGET,pool=MODELS,
                all_states=len(states),legal_deployments=len(legal),all_legal_counts_competed=True,
                routing_method='Corners/interior grid, two DE seeds, Powell, 101-point ASR lines, pure-route solves, independent top-five refinement',
                routing_limits='Numerically converged search; discontinuous periodic objective has no global continuous certificate',
                predicted_arrivals='Module20 periodic surrogate, phase=.37, horizon=120s',stages=stages,
                performance=perf,total_seconds=time.perf_counter()-start)
    return result

def main():
    assert not (OUT/'original_optimizer_audit.json').exists(),'Do not overwrite a completed audit'
    OUT.mkdir(parents=True,exist_ok=True);save('historical_metadata_before.json',metadata())
    d,refs,weights=inputs();save('selected_pool_catalog.json',d)
    sources=[SOURCE,ROOT/'module20/optimizer.py',ROOT/'results/module20_utility_optimizer/utility_definition.json',
             ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json']
    save('cpu_protocol.json',dict(frozen_before_search=True,pool=MODELS,rates=RATES,budget_mib=BUDGET,
        weights=weights,inputs=[ref(p) for p in sources],fixed_QPS=True,no_forced_phases=True,
        optimizer='Unmodified module20.optimizer.Generator/predict',development_negatives=[19,22],
        holdout='All Module16 model combinations, including same combination in Module17; all sessions/workloads together',
        GPU_probes='Only after CPU audit and offline validation; HIGH S-only and G/S=18%, at most two sessions each; diagnostic development only'))
    result=search(d,refs,weights);save('original_optimizer_audit.json',result)

if __name__=='__main__':main()
