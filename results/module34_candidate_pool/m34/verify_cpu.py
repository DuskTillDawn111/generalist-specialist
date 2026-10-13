"""Independent Cartesian route census and scalar FIFO evaluation checks."""
from .common import *
import numpy as np
from module20.optimizer import components,predict
from .common import identity

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
    fm=read(OUT/'frozen_problem_manifest.json');sp=read(fm['space']['path']);refs=read(fm['utility_reference']['path'])['latency_reference_ms'];weights=read(fm['utility_definition']['path'])['weights'];checks=[]
    old=read(fm['historical_profiles'][0]['path']);small=read(fm['historical_profiles'][1]['path'])
    for pool,pr in sp['pools'].items():
        d=read(pr['path']);models=sorted(d['models']);tasks=list(next(iter(sp['scenarios'].values()))['rates'])
        for m in models:
            source=small if m=='whisper-small' else old
            assert d['models'][m]==source['models'][m]
            assert all(d[k][m][n]==source[k][m][n] for k in ['memory_table_mib','capacity_factors'] for n in d[k][m])
        # Independent inverse enumeration: assign paths first, then all supersets
        # of routed models. Unlike original deployment-first recursion.
        paths={t:[m for m in models if t in d['models'][m]['supported_tasks']] for t in tasks};expected=set();before_memory=0
        for assigned in itertools.product(*(paths[t] for t in tasks)):
            required=set(assigned);unused=[m for m in models if m not in required]
            rr={t:{m:1.} for t,m in zip(tasks,assigned)}
            for include in itertools.product([False,True],repeat=len(unused)):
                x={m:1 for m in sorted(required|{m for m,b in zip(unused,include) if b})};before_memory+=1
                mem=d['safe_factor']*sum(d['memory_table_mib'][m]['1'] for m in x)
                if mem<=sp['budget_mib']+1e-6:expected.add(identity(x,rr))
        for scenario,L in sp['scenarios'].items():
            rows=list(csv.DictReader((OUT/f'cpu/{scenario}_{pool}_candidates.csv').open()));seen=set()
            for r in rows:
                x=json.loads(r['deployment']);rr=json.loads(r['routing']);norm={t:{m:q/L['rates'][t] for m,q in row.items()} for t,row in rr.items()};seen.add(identity(x,norm))
            assert seen==expected and len(seen)==len(rows)
            scalar_checks=[]
            # Every route in A/B, and predeclared census positions + algorithm
            # endpoints in C. Sets/completeness checked for every candidate.
            ix=list(range(len(rows))) if pool!='C' else sorted(set([0,len(rows)//4,len(rows)//2,3*len(rows)//4,len(rows)-1]))
            for i in ix:
                r=rows[i];x=json.loads(r['deployment']);rr=json.loads(r['routing']);a=scalar(d,x,rr,L['rates'],sp['budget_mib'],refs,weights);b=predict(d,x,rr,L['rates'],sp['budget_mib'],refs,weights,phase=.37)
                error=abs(a['U']-b['U']);assert error<1e-8
                scalar_checks.append(dict(index=i,abs_error=error))
            checks.append(dict(pool=pool,scenario=scenario,all_profiles_original=True,independent_route_set_equal=True,before_memory_routes=before_memory,after_memory_routes=len(rows),scalar_checks=scalar_checks))
    # Every evaluated/accepted runtime route of controlledV3 obeys the domain.
    for p in (OUT/'cpu').glob('*_V3_J_GRID_CONTROL.json'):
        r=read(p)
        for call in r['backend_calls']:
            assert all(len(row)==1 for row in call['routing'].values())
    save('CPU_oracle_correctness.json',dict(passed=True,independent_inverse_enumeration=True,controlled_V3_all_runtime_routes_on_grid=True,checks=checks))
    print('CPU_VERIFICATION_PASS',flush=True)
if __name__=='__main__':main()
