"""Independent Cartesian route census and scalar FIFO evaluation checks."""
from .common import *
import numpy as np
from module20.optimizer import components,predict
from m33.cpu import identity

def scalar(d,x,rr,rates,budget,refs,weights):
    stats={t:[] for t in rates};within=0;offered=0;late=0
    for m,n in x.items():
        events=[]
        for ti,t in enumerate(rates):
            q=rr.get(t,{}).get(m,0)
            if q<=1e-9:continue
            # Independent scalar loop; no cumsum/maximum-accumulate identity.
            a=(-10*q+.37*(ti+1))/q;index=0
            while a<120-1e-10:
                if a>=-10-1e-10:events.append((a,t,index%n))
                index+=1;a=(-10*q+.37*(ti+1)+index)/q
        events.sort(key=lambda v:(v[0],list(rates).index(v[1])));finish=[-float('inf')]*n
        for a,t,i in events:
            finish[i]=max(a,finish[i])+d['models'][m]['service_time_ms'][t]/1000*n/d['capacity_factors'][m][str(n)]
            if a>=0:
                stats[t].append((finish[i]-a)*1000);offered+=1;within+=finish[i]<=120;late+=finish[i]>180
    per={t:dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95))) for t,v in stats.items()}
    T=1-min(1.,(offered-late)/offered,within/(sum(rates.values())*120))
    mem=d['safe_factor']*sum(d['memory_table_mib'][m][str(n)] for m,n in x.items())
    return components(per,rates,refs,T,mem/budget,weights)

def main():
    fm=read(OUT/'frozen_problem_manifest.json');sp=read(fm['space']['path']);refs=read(fm['utility_reference']['path'])['latency_reference_ms'];weights=read(fm['utility_definition']['path'])['weights'];records=[]
    source=read(ROOT/'results/module32b_vqa_aware_routing/selected_pool_catalog.json')
    for pool,pr in sp['pools'].items():
        d=read(pr['path']);models=sorted(d['models']);states=math.prod(len(d['capacity_factors'][m]) for m in models)
        assert all(d[k][m][n]==source[k][m][n] for k in ['capacity_factors','memory_table_mib'] for m in models for n in d[k][m])
        assert all(d['models'][m]['service_time_ms']==source['models'][m]['service_time_ms'] for m in models)
        expected=set();totalroutes=0;structuralroutes=0
        # All deployments first; independently enumerate normalized route vectors
        # by Cartesian product [0,.5,1] then exact conservation (not weak recursion).
        for counts in itertools.product(*(sorted(map(int,d['capacity_factors'][m])) for m in models)):
            x={m:n for m,n in zip(models,counts) if n};opts={t:[m for m in sorted(x) if t in d['models'][m]['supported_tasks']] for t in sp['loads']['ASR32B']['rates']}
            if not x or not all(opts.values()):continue
            choices={t:[dict(zip(ms,z)) for z in itertools.product([0.,.5,1.],repeat=len(ms)) if sum(z)==1.] for t,ms in opts.items()}
            nr=math.prod(len(v) for v in choices.values());totalroutes+=nr
            mixed={'generalist','specialist'}<=set(d['models'][m]['role'] for m in x)
            if not mixed:continue
            structuralroutes+=nr
            mem=d['safe_factor']*sum(d['memory_table_mib'][m][str(n)] for m,n in x.items())
            if mem>sp['budget_mib']+1e-6:continue
            for rows in itertools.product(*(choices[t] for t in opts)):
                rr={t:{m:q for m,q in row.items() if q} for t,row in zip(opts,rows)}
                expected.add(identity(x,rr))
        for load,L in sp['loads'].items():
            rows=list(csv.DictReader((OUT/'cpu'/f'{pool}_{load}_all_candidates.csv').open()));seen=set();rates=L['rates']
            for r in rows:
                x=json.loads(r['deployment']);rr=json.loads(r['routing']);norm={t:{m:q/rates[t] for m,q in row.items()} for t,row in rr.items()}
                seen.add(identity(x,norm))
                assert all(abs(sum(row.values())-rates[t])<1e-9 for t,row in rr.items())
                assert all(q>=0 and m in x and t in d['models'][m]['supported_tasks'] for t,row in rr.items() for m,q in row.items())
            assert len(seen)==len(rows) and seen==expected
            checks=[]
            # Endpoints, mixed routes and multi-replica examples from frozen
            # enumeration positions; no selection on prediction accuracy.
            ix=sorted(set([0,len(rows)//4,len(rows)//2,3*len(rows)//4,len(rows)-1]))
            for i in ix:
                r=rows[i];x=json.loads(r['deployment']);rr=json.loads(r['routing'])
                a=scalar(d,x,rr,rates,sp['budget_mib'],refs,weights);b=predict(d,x,rr,rates,sp['budget_mib'],refs,weights,phase=.37)
                error=abs(a['U']-b['U']);assert error<1e-9,(pool,load,i,error)
                checks.append(dict(index=i,scalar_U=a['U'],native_U=b['U'],abs_error=error))
            records.append(dict(pool=pool,load=load,all_vectors=states,total_covered_route_candidates_before_memory=totalroutes,
                total_mixed_route_candidates_before_memory=structuralroutes,enumerated_after_memory=len(rows),
                independent_candidate_set_equal=True,original_profiles_exact=True,scalar_checks=checks))
    save('CPU_oracle_correctness.json',dict(passed=True,checks=records,algorithm_independent_census=True,scalar_FIFO_check=True))
    print('CPU_ORACLE_INDEPENDENT_VERIFICATION_PASS',flush=True)
if __name__=='__main__':main()
