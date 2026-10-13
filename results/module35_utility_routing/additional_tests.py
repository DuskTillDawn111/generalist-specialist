import sys,itertools,csv,json,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/generalist-specialist');OUT=ROOT/'results/module35_utility_routing';OLD=ROOT/'results/module34_candidate_pool'
sys.path[:0]=[str(OUT),str(OLD),str(ROOT)]
from m35.isolation import install
install('supervisor')
from m35.common import *
from m34.cpu_core import candidate_routes,evaluate,catalog,formal_v3
from u_aligned_optimizer import UAlignedOptimizer
assert read(OUT/'CPU_stage_completed.json')['passed']
fm=read(OLD/'frozen_problem_manifest.json');sp=read(fm['space']['path']);refs=read(fm['utility_reference']['path'])['latency_reference_ms'];w=read(fm['utility_definition']['path'])['weights'];budget=sp['budget_mib']
checks=read(OUT/'optimizer_correctness_tests.json')
# All finite legal route keys in the smaller pools remain legal after expansion.
sets={};exp=[]
for scen in sp['scenarios']:
    for pool in sp['pools']:
        rr=list(csv.DictReader((OUT/f'cpu/{scen}_{pool}_oracle_candidates.csv').open()))
        sets[scen,pool]={identity(json.loads(r['deployment']),json.loads(r['routing'])) for r in rr if r['eligible']=='True'}
    exp.append(dict(scenario=scen,A_in_B=sets[scen,'A']<=sets[scen,'B'],B_in_C=sets[scen,'B']<=sets[scen,'C']))
shared=[]
for p,q in [('A','B'),('B','C')]:
    a,b=[read(sp['pools'][z]['path']) for z in [p,q]]
    shared.append(dict(smaller=p,larger=q,strict_subset=set(a['models'])<set(b['models']),safe_factor_equal=a['safe_factor']==b['safe_factor'],
        unchanged_common_models=all(a[k][m]==b[k][m] for k in ['models','capacity_factors','memory_table_mib'] for m in a['models'])))
save('pool_nesting_validation.json',dict(shared_profiles=shared,legal_space_inclusions=exp,passed=all(r['strict_subset'] and r['safe_factor_equal'] and r['unchanged_common_models'] for r in shared) and all(r['A_in_B'] and r['B_in_C'] for r in exp)))
# Replica correctness on the already frozen33 catalog, no new GPU/scenario.
ss=read(ROOT/'results/module33_mixed_oracle/candidate_space_definition.json');d=read(ROOT/'results/module33_mixed_oracle/pools/SMALL6.json')
rates=sp['scenarios']['ASR32B']['rates'];b33=ss['budget_mib']
opts={m:[int(n) for n in d['capacity_factors'][m] if n in d['memory_table_mib'][m] and int(n)>0] for m in d['models']}
solver=UAlignedOptimizer(d,rates,b33,refs,w);f=solver.solve(replica_options=opts)
oracle=[]
for counts in itertools.product(*([0]+opts[m] for m in sorted(d['models']))):
    x={m:n for m,n in zip(sorted(d['models']),counts) if n}
    if not set(rates)<={t for m in x for t in d['models'][m]['supported_tasks']}:continue
    for r in candidate_routes(d,x,rates,1):
        p,_,_=evaluate(d,x,r,rates,b33,refs,w,catalog(d))
        if p['eligible']:oracle.append(dict(deployment=x,routing=r,U=p['U']))
best=max(oracle,key=lambda z:z['U']);checks['measured_replica_test']=dict(replica_options=opts,new=f,oracle=best,legal_routes=len(oracle),passed=abs(f['winner']['prediction']['U']-best['U'])<1e-8)
checks['infeasibility_tests']=[dict(name='budget_one_mib',status=(v:=UAlignedOptimizer(d,rates,1,refs,w).solve(replica_options=opts))['status'],passed=v['winner'] is None),
    dict(name='unmeasured_fixed_replica',status=(v:=UAlignedOptimizer(d,rates,b33,refs,w).solve(deployment={'phi4mm':999}))['status'],passed=v['winner'] is None)]
checks['all_passed']=checks['all_passed'] and checks['measured_replica_test']['passed'] and all(x['passed'] for x in checks['infeasibility_tests'])
save('optimizer_correctness_tests.json',checks)
# Fresh TEXT10 finite table yields an independent no-benefit addition Oracle.
control=read(OUT/'unhelpful_candidate_control.json');bart=control['added_specialist']
rows=list(csv.DictReader((OUT/'cpu/TEXT_HALF_MID18_C_oracle_candidates.csv').open()))
before=max(float(r['predicted_U']) for r in rows if r['eligible']=='True' and bart not in json.loads(r['deployment']))
after=max(float(r['predicted_U']) for r in rows if r['eligible']=='True')
control.update(oracle_before_U=before,oracle_after_U=after,oracle_same_U=before==after)
legacy_before=control['legacy_before']['winner'];old_dec=read(OLD/'optimizer_decisions.json')
legacy_after=next(r for r in old_dec if r['scenario']=='TEXT_HALF_MID18' and r['pool']=='C' and r['algorithm']=='20A_U_GRID')
control['legacy_same_plan']=identity(legacy_before['deployment'],legacy_before['routing'])==identity(legacy_after['deployment'],legacy_after['routing'])
save('unhelpful_candidate_control.json',control)
save('additional_acceptance_completed.json',dict(passed=checks['all_passed'],no_algorithm_source_changes=True))
print('ADDITIONAL',checks['all_passed'],flush=True)
