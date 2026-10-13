import sys, os, time, itertools, copy, hashlib, csv, json
from pathlib import Path
ROOT=Path('/root/autodl-tmp/generalist-specialist')
OUT=ROOT/'results/module35_utility_routing';OLD=ROOT/'results/module34_candidate_pool'
sys.path[:0]=[str(OUT),str(OLD),str(ROOT)]
from m35.isolation import install
install('supervisor')
from m35.common import read,save,csvsave,ref,identity,structure
from m34.cpu_core import evaluate,catalog,candidate_routes,optimizer,formal_v3
from m34.cpu import static
from u_aligned_optimizer import UAlignedOptimizer

fm=read(OLD/'frozen_problem_manifest.json');sp=read(fm['space']['path'])
w=read(fm['utility_definition']['path'])['weights'];refs=read(fm['utility_reference']['path'])['latency_reference_ms'];budget=sp['budget_mib']
hold=read(OUT/'frozen_holdout_manifest.json')
save('optimizer_source_freeze.json',dict(at_unix_ns=time.time_ns(),sources=[ref(OUT/'u_aligned_optimizer.py'),ref(OUT/'run_cpu.py')],holdout=ref(OUT/'frozen_holdout_manifest.json'),no_oracle_route_injection=True))

def enumerate_oracle(d,rates):
    cp=catalog(d);rows=[];states=0;start=time.perf_counter()
    for bits in itertools.product((0,1),repeat=len(d['models'])):
        states+=1;x={m:1 for m,b in zip(sorted(d['models']),bits) if b}
        mem=d['safe_factor']*sum(d['memory_table_mib'][m]['1'] for m in x)
        if mem>budget+1e-6 or not set(rates)<={t for m in x for t in d['models'][m]['supported_tasks']}:continue
        for r in candidate_routes(d,x,rates,1):
            p,j,err=evaluate(d,x,r,rates,budget,refs,w,cp)
            rows.append(dict(deployment=x,routing=r,predicted_U=p['U'],eligible=p['eligible'],errors=err))
    elapsed=time.perf_counter()-start
    ok=[r for r in rows if r['eligible']]
    best=min(ok,key=lambda r:(-r['predicted_U'],d['safe_factor']*sum(d['memory_table_mib'][m]['1'] for m in r['deployment']),identity(r['deployment'],r['routing']))) if ok else None
    return best,rows,elapsed,states

comparisons=[];holds=[];timings=[];correct=[];attributions=[]
def record(dest,scenario,pool,alg,plan,d,rates,oracle,seconds=None,metrics=None,status=None):
    if plan is None:
        row=dict(scenario=scenario,pool=pool,algorithm=alg,status=status or 'NO_PLAN',eligible=False,predicted_U=None,regret=None,seconds=seconds)
    else:
        x,r=plan;p,j,err=evaluate(d,x,r,rates,budget,refs,w,catalog(d))
        grid=all(abs(q/rates[t]-round(q/rates[t]))<1e-7 for t,rr in r.items() for q in rr.values())
        legal=p['eligible'] and grid
        row=dict(scenario=scenario,pool=pool,algorithm=alg,status='FEASIBLE' if legal else 'DOMAIN_MISMATCH' if not grid else 'CONSTRAINT_REJECT',
            eligible=legal,deployment=x,routing=r,structure=structure(d,x),predicted_U=p['U'],regret=oracle['predicted_U']-p['U'] if legal else None,
            seconds=seconds,safe_memory_mib=p.get('safe_memory_mib'),max_rho=p.get('max_rho'),errors=err)
    row.update(metrics or {});dest.append(row);return row

allcases=[(s,L['rates'],False) for s,L in sp['scenarios'].items()]+[('HOLDOUT_TEXT3',hold['rates'],True)]
for scenario,rates,is_hold in allcases:
    for pool,pref in sp['pools'].items():
        d=read(pref['path']);dest=holds if is_hold else comparisons
        cold=[];fits=[]
        # All algorithms run before the new acceptance Oracle is inspected.
        for rep in range(3):
            solver=UAlignedOptimizer(d,rates,budget,refs,w)
            f=solver.solve(replica_options={m:[1] for m in d['models']});fits.append(f)
            cold.append(f['seconds']);save(f'cpu/{scenario}_{pool}_U_ALIGNED_{rep+1}.json',f)
        fit=fits[0];save(f'cpu/{scenario}_{pool}_20A_GRID.json',a:=optimizer(d,rates,budget,refs,w,1))
        v= formal_v3(d,rates,budget,refs,w,grid=True,K=1);save(f'cpu/{scenario}_{pool}_V3_GRID.json',v)
        vn=formal_v3(d,rates,budget,refs,w,grid=False,K=1);save(f'cpu/{scenario}_{pool}_V3_NATIVE.json',vn)
        an=optimizer(d,rates,budget,refs,w,1,native=True);save(f'cpu/{scenario}_{pool}_20A_NATIVE.json',an)
        best,rows,sec,vectors=enumerate_oracle(d,rates)
        csvsave(f'cpu/{scenario}_{pool}_oracle_candidates.csv',rows)
        save(f'cpu/{scenario}_{pool}_oracle.json',dict(best=best,seconds=sec,evaluated_routes=len(rows),deployment_vectors=vectors))
        new=record(dest,scenario,pool,'U_ALIGNED_DP',None if not fit['winner'] else (fit['winner']['deployment'],fit['winner']['routing']),d,rates,best,
            float(__import__('numpy').median(cold)),fit['metrics'],fit['status'])
        record(dest,scenario,pool,'CPU_U_ORACLE',(best['deployment'],best['routing']),d,rates,best,sec,
               dict(search_states=len(rows),deployment_vectors=vectors,full_prediction_calls=4*len(rows),cache_hits=0))
        for name,f in [('20A_U_GRID',a),('20A_NATIVE_CONTINUOUS_PRESERVED',an),('V3_J_GRID_CONTROL',v),('V3_NATIVE_CONTINUOUS_PRESERVED',vn)]:
            z=f.get('winner') if 'winner' in f else f.get('decision')
            record(dest,scenario,pool,name,None if not z else (z['deployment'],z['routing']),d,rates,best,f['seconds'],
                   dict(full_prediction_calls=f.get('objective_calls'),search_states=f.get('unique_evaluated',len(f.get('backend_calls',[])))),f.get('status'))
        for name,minmem in [('FASTEST_PATH_STATIC',False),('MIN_MEMORY_COVER_FASTEST',True)]:
            st=time.perf_counter();plan=static(d,rates,minmem,budget);ss=time.perf_counter()-st
            record(dest,scenario,pool,name,plan,d,rates,best,ss,dict(full_prediction_calls=4 if plan else 0))
        # Repeated Oracle timing done separately to avoid GPU overlap. Exactly
        # same environment, fresh new solver cache, no imported-result timings.
        times=[sec]
        for rep in range(2):
            bb,_,ss,_=enumerate_oracle(d,rates);times.append(ss)
            assert abs(bb['predicted_U']-best['predicted_U'])<1e-9
        for alg,tt,met in [('CPU_U_ORACLE',times,dict(full_prediction_calls=4*len(rows),search_states=len(rows))),('U_ALIGNED_DP',cold,fit['metrics'])]:
            for rep,ss in enumerate(tt,1):timings.append(dict(scenario=scenario,pool=pool,algorithm=alg,repetition=rep,seconds=ss,candidate_space=len(rows),holdout=is_hold,**met))
        # Check EVERY original finite row using cached worker bundles; complete
        # nominal utility cannot silently change the score of any candidate.
        sim=UAlignedOptimizer(d,rates,budget,refs,w);maxerr=0.;mismatch=0
        for row in rows:
            x,r=row['deployment'],row['routing'];cost=0.;within=0;valid=True
            for m,n in x.items():
                mask=sum(1<<i for i,t in enumerate(rates) if r[t].get(m,0)>1e-9)
                if not mask:cost+=w['M']*d['safe_factor']*d['memory_table_mib'][m][str(n)]/budget;continue
                b=sim.bundle(m,n,mask)
                if b is None:valid=False;break
                cost+=b['cost'];within+=b['within']
            mismatch+=valid!=row['eligible']
            if valid and row['eligible']:
                uu=-cost-w['T']*max(0.,1-within/(sum(rates.values())*120));maxerr=max(maxerr,abs(uu-row['predicted_U']))
        test=dict(scenario=scenario,pool=pool,holdout=is_hold,rows_checked=len(rows),eligibility_mismatches=mismatch,max_nominal_U_error=maxerr,
                  new_regret=new['regret'],passed=mismatch==0 and maxerr<1e-8 and new['eligible'] and abs(new['regret'])<1e-8,
                  worker_cache=sim.metrics)
        if not is_hold:
            oldrow=next(r for r in read(OLD/'cpu_oracles.json') if r['scenario']==scenario and r['pool']==pool)
            test['historical_oracle_agreement']=abs(best['predicted_U']-oldrow['U_oracle']['predicted_U'])<1e-9
            olda=next(r for r in read(OLD/'optimizer_decisions.json') if r['scenario']==scenario and r['pool']==pool and r['algorithm']=='20A_U_GRID')
            if olda.get('regret',0)>1e-8:
                # Fixed-deployment optimizer proves the entire gap can be
                # closed without changing model selection or replica counts.
                fix=UAlignedOptimizer(d,rates,budget,refs,w).solve(deployment=olda['deployment'])
                oracle_key=identity(best['deployment'],best['routing'])
                candidates=read(OLD/f'cpu/{scenario}_{pool}_20A_U_GRID_1.json')['candidates']
                legal=[z for z in candidates if z['prediction']['eligible']]
                inside=max(z['prediction']['U'] for z in legal)
                diagnostic=dict(scenario=scenario,pool=pool,old_U=olda['predicted_U'],oracle_U=best['predicted_U'],regret=olda['regret'],
                    same_deployment=olda['deployment']==best['deployment'],old_routing=olda['routing'],oracle_routing=best['routing'],
                    oracle_route_generated=any(identity(z['deployment'],z['routing'])==oracle_key for z in candidates),generated_count=len(candidates),
                    generated_ranking_loss=inside-olda['predicted_U'],candidate_omission_loss=best['predicted_U']-inside,
                    fixed_deployment_repaired_U=fix['winner']['prediction']['U'],fixed_route_search=fix,
                    changed_task_routes=[t for t in rates if olda['routing'][t]!=best['routing'][t]])
                attributions.append(diagnostic)
        correct.append(test)
        print('CELL',scenario,pool,'regret',new['regret'],'new_s',new['seconds'],'oracle_s',sec,'test',test['passed'],flush=True)
        csvsave('cpu_oracle_comparison.csv',comparisons);csvsave('holdout_results.csv',holds);csvsave('solver_runtime_comparison.csv',timings)
        save('optimizer_correctness_tests.json',dict(cells=correct,all_passed=all(t['passed'] for t in correct)))
        save('routing_gap_attribution.json',attributions)

# No-benefit expansion control with a real existing, admitted Specialist.
d=read(sp['pools']['C']['path']);minus=copy.deepcopy(d)
bart=next(m for m in d['models'] if 'bart-large' in m)
for key in ['models','capacity_factors','memory_table_mib']:minus[key].pop(bart,None)
rates=sp['scenarios']['TEXT_HALF_MID18']['rates']
control=UAlignedOptimizer(minus,rates,budget,refs,w).solve(replica_options={m:[1] for m in minus['models']})
full=next(r for r in comparisons if r['scenario']=='TEXT_HALF_MID18' and r['pool']=='C' and r['algorithm']=='U_ALIGNED_DP')
legacy=optimizer(minus,rates,budget,refs,w,1);save('unhelpful_candidate_control.json',dict(added_specialist=bart,new_before=control,new_after=full,
    same_plan=identity(control['winner']['deployment'],control['winner']['routing'])==identity(full['deployment'],full['routing']),
    legacy_before=legacy,oracle_before_U=max(r['predicted_U'] for r in rows if r['eligible'] and bart not in r['deployment']) if scenario=='TEXT_HALF_MID18' else None,
    historical_Oracle_control=read(OLD/'unhelpful_candidate_control.json'),scope='C without Bart -> C; real admitted addition, same TEXT10 load; A/B not fabricated to remain G-only'))

# Fixed-deployment K=2 test exercises optional G/S split and replica handling.
d=read(sp['pools']['B']['path']);rates=sp['scenarios']['ASR32B']['rates']
x=next(r for r in comparisons if r['scenario']=='ASR32B' and r['pool']=='B' and r['algorithm']=='CPU_U_ORACLE')['deployment']
ss=UAlignedOptimizer(d,rates,budget,refs,w);split=ss.solve(deployment=x,K=2)
br=[]
for r in candidate_routes(d,x,rates,2):
    p,_,_=evaluate(d,x,r,rates,budget,refs,w,catalog(d))
    if p['eligible']:br.append((p['U'],r))
split_test=dict(deployment=x,K=2,new=split,oracle_U=max(u for u,r in br),legal_routes=len(br),passed=abs(split['winner']['prediction']['U']-max(u for u,r in br))<1e-9)
# Repeated solve checks cache correctness and counts, not cold runtime.
again=ss.solve(deployment=x,K=2);split_test['warm_metrics']=again['metrics'];split_test['repeat_same_U']=again['winner']['prediction']['U']==split['winner']['prediction']['U']
save('split_routing_correctness.json',split_test)
checks=read(OUT/'optimizer_correctness_tests.json');checks.update(split_test_passed=split_test['passed'],
    all_passed=checks['all_passed'] and split_test['passed'],source_freeze=ref(OUT/'optimizer_source_freeze.json'))
save('optimizer_correctness_tests.json',checks)
save('CPU_stage_completed.json',dict(cells=len(correct),passed=checks['all_passed'],holdout_used_for_tuning=False,attributions=len(attributions)))
