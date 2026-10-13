"""Exhaustive deployment audit with rigorous task-wise J lower-bound pruning.

Each task independently gets the full capacity of every admitted model. Since
f_m(sum D_k) >= sum f_m(D_k), this relaxation is a lower bound, not a new router.
Only the original frozen router produces accepted routing witnesses.
"""
import math
import time
from collections import Counter
from decimal import Decimal, localcontext
from deployment_core import Workload
from deployment_search.routing import solve_canonical_routing
from deployment_core.routing.e2e_router import solve_e2e_routing
from .common import *

def lower_bound(x,lam,c,cal):
    total=0.
    for k,rate in lam.items():
        options=sorted((c[m].service_time_ms[k]/1000,cal.capacity_factors[m][n])
            for m,n in x.items() if n and k in c[m].supported_tasks)
        if not options:return math.inf
        a=b=0.;used=[]
        for i,(s,g) in enumerate(options):
            a+=g/s;b+=g/math.sqrt(s);used.append((s,g))
            if a<=rate:continue
            root=b/(a-rate)
            if i+1<len(options) and root*root>options[i+1][0]:continue
            subtotal=0.
            for st,cap in used:
                d=max(0.,cap*(1-math.sqrt(st)/root))
                subtotal+=d/(1-d/cap)
            total+=subtotal;break
        else:return math.inf
    return total/math.fsum(lam.values())

def certificate(x,route,c,cal):
    with localcontext() as ctx:
        ctx.prec=60
        D={m:sum((Decimal(str(v))*Decimal(str(c[m].service_time_ms[k]))/1000
                 for (k,n),v in route.items() if n==m and v),Decimal(0)) for m,n in x.items() if n}
        res={m:D[m]-Decimal('.8')*Decimal(str(cal.capacity_factors[m][x[m]])) for m in D}
        bound={m:min(Decimal('1e-8'),Decimal(str(32*math.ulp(max(1.,float(D[m])))))) for m in D}
        passed=all(res[m]<=bound[m] for m in D)
    return {'status':'PASS' if passed else 'OPERATIONAL_ROUTE_INFEASIBLE','passed':passed,
            'capacity_residual_decimal':{m:str(v) for m,v in res.items()},'fixed_tolerance_seconds':1e-8}

def witness(x,fit,lam,heads,roles,cal,cert):
    route=fit['routing'];evaluation=fit['evaluation']
    frac={k:math.fsum(v for (t,m),v in route.items() if t==k and roles[m]=='generalist')/lam[k] for k in heads}
    return {'deployment':x,'routing':serial(route),'safe_memory_mib':safe_memory(x,cal),
        'J_seconds':evaluation['J_seconds'],'max_rho':evaluation['z'],'rho':evaluation['rho'],'D':evaluation['D'],
        'N_G':sum(n for m,n in x.items() if roles[m]=='generalist'),
        'specialist_replicas':{m:n for m,n in x.items() if n and roles[m]=='specialist'},
        'hotspot_G_fraction':frac,'overflow_tasks':[k for k,v in frac.items() if v>0],
        'certificate':cert,'optimality_gap_seconds':fit['optimality_gap_seconds']}

def main():
    assert not (OUT/'latency_stage_status.json').exists(),'Latency audit already complete'
    c,hw,roles,cal=inputs();freeze=read(OUT/'input_freeze.json');bg=freeze['background']
    windows=read(OUT/'windows.json')['rows'];directions=read(OUT/'normalized_directions.json')['heads_by_budget']
    states=read(OUT/'deployment_space.json');cache={};start=time.perf_counter();raw_path=OUT/'J_solver_diagnostics.jsonl'
    rows=read(OUT/'latency_witnesses.json')['rows'] if (OUT/'latency_witnesses.json').exists() else []
    done={(r['budget'],r['set_id']) for r in rows};stats=Counter()
    # Resume completed CPU contexts without overwriting diagnostics or rerunning
    # their solver outcomes. A frozen solver exception remains a failed state.
    if raw_path.exists():
        for line in raw_path.open():
            rec=json.loads(line);x=rec['deployment'];lam=rec['lambda_rps']
            fit={'status':rec['solver_status'],'converged':rec['solver_status']=='SOLVED',
                 'evaluation':rec['evaluation'],'optimality_gap_seconds':rec['gap_seconds'],
                 'constraints_valid':rec.get('constraints_valid',rec['solver_status']=='SOLVED'),
                 'routing':{tuple(k.split('::')):v for k,v in rec['routing'].items()} if rec['routing'] else None}
            cache[(tuple(sorted(x.items())),tuple(sorted(lam.items())))]=(fit,rec['certificate'])
    stats['unique_state_solves']=len(cache)
    raw=raw_path.open('a')
    for win in windows:
        if (win['budget'],win['set_id']) in done:continue
        if win['alpha_1G'] is None:continue
        bid=win['budget'];tasks=win['tasks'];heads={k:directions[bid][k] for k in tasks}
        s,a,w=win['alpha_S'],win['alpha_1G'],win['W_1G']
        proposed=[('S80',.8*s),('S98',.98*s),('O30',s+.3*w),('O70',s+.7*w),('G80',.8*a),('G98',.98*a)]
        points=[]
        for label,alpha in proposed:
            if alpha<0 or alpha>=a or any(abs(alpha-v)<1e-12 for _,v in points):continue
            points.append((label,alpha))
        for label,alpha in points:
            lam=rates(bg,heads,alpha);workload=Workload(lam);best={1:None,2:None};counts=Counter();uncertain={1:[],2:[]}
            pool=[]
            for state in states[bid]:
                if state['N_G']>2:continue
                lb=lower_bound(state['deployment'],lam,c,cal)
                if math.isfinite(lb):pool.append((lb,state))
                else:counts['taskwise_relaxation_infeasible']+=1
            pool.sort(key=lambda row:(row[0],row[1]['safe_memory_mib'],tuple(sorted(row[1]['deployment'].items()))))
            for lb,state in pool:
                x=state['deployment'];families=[2] if state['N_G']==2 else [1,2]
                if all(best[f] is not None and lb>best[f]['J_seconds']+1e-12 for f in families):
                    counts['rigorous_lower_bound_pruned']+=1;continue
                key=(tuple(sorted(x.items())),tuple(sorted(lam.items())))
                if key in cache:
                    fit,cert=cache[key];counts['exact_CPU_solve_cache_reused']+=1
                else:
                    canonical=solve_canonical_routing(x,workload,c,replica_bounds=cal.evidence_bounds,capacity_mode='empirical',calibration=cal)
                    if canonical is None:
                        fit={'status':'OPERATIONAL_PLANNING_INFEASIBLE','converged':False};cert=None
                    else:
                        # Default initialization is deliberately identical to the
                        # frozen current-state ranker; no custom route is passed.
                        try:
                            fit=solve_e2e_routing(x,workload,c,cal)
                        except (ValueError,ArithmeticError) as exc:
                            fit={'status':'NUMERICAL_SOLVER_EXCEPTION','converged':False,'exception':repr(exc)}
                        cert=certificate(x,fit['routing'],c,cal) if fit['converged'] else None
                    cache[key]=(fit,cert);stats['unique_state_solves']+=1
                    rec={'deployment':x,'lambda_rps':lam,'solver_status':fit['status'],
                         'certificate':cert,'evaluation':fit.get('evaluation'),
                         'gap_seconds':fit.get('optimality_gap_seconds'),'solver_message':fit.get('solver_message'),
                         'constraints_valid':fit.get('constraints_valid',False),'exception':fit.get('exception'),
                         'routing':serial(fit['routing']) if fit.get('routing') else None}
                    raw.write(json.dumps(rec,sort_keys=True)+'\n');raw.flush()
                status=cert['status'] if cert else fit['status'];counts[status]+=1
                if fit['converged'] and cert['passed']:
                    found=witness(x,fit,lam,heads,roles,cal,cert)
                    assert lb<=found['J_seconds']+1e-9,(lb,found)
                    for family in families:
                        prev=best[family]
                        ordering=lambda v:(v['J_seconds'],v['max_rho'],v['safe_memory_mib'],tuple(sorted(v['deployment'].items())))
                        if prev is None or ordering(found)<ordering(prev):best[family]=found
                elif status!='OPERATIONAL_PLANNING_INFEASIBLE':
                    value=(fit.get('evaluation') or {}).get('J_seconds');gap=fit.get('optimality_gap_seconds')
                    # A convex gradient certificate bounds the unconstrained
                    # optimum, even for a returned route above rho=.8.
                    bound=max(lb,value-gap) if value is not None and gap is not None and fit.get('constraints_valid') else lb
                    for family in families:uncertain[family].append({'lower_bound_seconds':bound,'status':status})
            j1,j2=[best[f]['J_seconds'] if best[f] else None for f in [1,2]]
            delta=j1-j2 if j1 is not None and j2 is not None else None
            censored={str(f):sum(best[f] is None or q['lower_bound_seconds']<best[f]['J_seconds']-1e-8 for q in uncertain[f]) for f in [1,2]}
            # No invented benefit threshold: the existing 10 ns solver certificate
            # gives a numerical interval, and positivity must survive that interval.
            positive=bool(delta is not None and best[2]['N_G']==2 and delta>best[1]['optimality_gap_seconds']+best[2]['optimality_gap_seconds'])
            row={'budget':bid,'set_id':win['set_id'],'tasks':tasks,'landmark':label,'alpha':alpha,'lambda_rps':lam,
                'J_1G_seconds':j1,'J_2G_seconds':j2,'DeltaJ_2G_seconds':delta,
                'best_1G':best[1],'best_2G':best[2],'replica_certificate':win['replica_certificate'],
                'observed_two_G_J_advantage':positive,
                'latency_beneficial_in_empirical_space':bool(positive and not any(censored.values())),
                'SECOND_GENERALIST_LATENCY_BENEFICIAL':bool(positive and win['replica_certificate'] and not any(censored.values())),
                'unresolved_better_state_count':censored,'enumeration_status_counts':dict(counts),
                'J_star_scope':'best operationally accepted frozen-router state; unreturned/overloaded potential optima counted separately',
                'new_GPU_runs':0}
            rows.append(row);stats.update(counts)
        save('latency_witnesses.json',{'rows':rows});csvsave('oneG_vs_twoG_j.csv',rows)
        print(json.dumps({'budget':bid,'set':win['set_id'],'landmarks':len(points),'unique_solves':stats['unique_state_solves'],
            'elapsed_seconds':round(time.perf_counter()-start,1)}),flush=True)
    raw.close()
    totals=Counter()
    for row in rows:totals.update(row['enumeration_status_counts'])
    save('latency_stage_status.json',{'status':'COMPLETE','comparisons':len(rows),'statistics':dict(totals),
         'unique_state_solves':len(cache),'last_execution_seconds':time.perf_counter()-start,
         'GPU_runs':0,'original_J_solver_unchanged':True})

if __name__=='__main__':main()
