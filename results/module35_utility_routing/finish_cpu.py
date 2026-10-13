"""Continue auxiliary tests after a catalog-projection harness error.

All twelve primary cells and their timings were completed before that error.
The optimizer source is unchanged. Original failed run/log/freeze retained.
"""
import sys,csv,json,copy,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/generalist-specialist');OUT=ROOT/'results/module35_utility_routing';OLD=ROOT/'results/module34_candidate_pool'
sys.path[:0]=[str(OUT),str(OLD),str(ROOT)]
from m35.isolation import install
install('supervisor')
from m35.common import *
from m34.cpu_core import optimizer,evaluate,catalog,candidate_routes
from u_aligned_optimizer import UAlignedOptimizer
fm=read(OLD/'frozen_problem_manifest.json');sp=read(fm['space']['path']);w=read(fm['utility_definition']['path'])['weights'];refs=read(fm['utility_reference']['path'])['latency_reference_ms'];budget=sp['budget_mib']
comparisons=list(csv.DictReader((OUT/'cpu_oracle_comparison.csv').open()))
for r in comparisons:
    for k in ['deployment','routing']:
        if r.get(k):r[k]=json.loads(r[k])
    for k in ['predicted_U','regret']:
        if r.get(k):r[k]=float(r[k])
checks=read(OUT/'optimizer_correctness_tests.json');assert len(checks['cells'])==12 and checks['all_passed']
freeze=read(OUT/'optimizer_source_freeze.json');assert ref(OUT/'u_aligned_optimizer.py')['sha256']==freeze['sources'][0]['sha256']
save('infrastructure_events.json',[dict(event='Auxiliary no-Bart catalog projection omitted evidence_bounds/model_evidence',
    original_failure_log=ref(OUT/'cpu.log'),impact='Occurred after all nine primary and threeholdout cells; no scientific measurements/timings overwritten; no optimizer changes',
    repair='Remove excluded Bart from all catalog-keyed maps in new in-memory pool projection',at_unix_ns=time.time_ns())])
d=read(sp['pools']['C']['path']);minus=copy.deepcopy(d);bart=next(m for m in d['models'] if 'bart-large' in m)
for key in ['models','capacity_factors','memory_table_mib','evidence_bounds','model_evidence']:minus[key].pop(bart,None)
rates=sp['scenarios']['TEXT_HALF_MID18']['rates']
control=UAlignedOptimizer(minus,rates,budget,refs,w).solve(replica_options={m:[1] for m in minus['models']})
full=next(r for r in comparisons if r['scenario']=='TEXT_HALF_MID18' and r['pool']=='C' and r['algorithm']=='U_ALIGNED_DP')
legacy=optimizer(minus,rates,budget,refs,w,1)
save('unhelpful_candidate_control.json',dict(added_specialist=bart,new_before=control,new_after=full,
    same_plan=identity(control['winner']['deployment'],control['winner']['routing'])==identity(full['deployment'],full['routing']),
    legacy_before=legacy,historical_Oracle_control=read(OLD/'unhelpful_candidate_control.json'),scope='C without Bart -> C; actual admitted addition; fixedTEXT10'))
d=read(sp['pools']['B']['path']);rates=sp['scenarios']['ASR32B']['rates']
x=next(r for r in comparisons if r['scenario']=='ASR32B' and r['pool']=='B' and r['algorithm']=='CPU_U_ORACLE')['deployment']
ss=UAlignedOptimizer(d,rates,budget,refs,w);split=ss.solve(deployment=x,K=2)
br=[]
for r in candidate_routes(d,x,rates,2):
    p,_,_=evaluate(d,x,r,rates,budget,refs,w,catalog(d))
    if p['eligible']:br.append((p['U'],r))
split_test=dict(deployment=x,K=2,new=split,oracle_U=max(u for u,r in br),legal_routes=len(br),passed=abs(split['winner']['prediction']['U']-max(u for u,r in br))<1e-9)
again=ss.solve(deployment=x,K=2);split_test['warm_metrics']=again['metrics'];split_test['repeat_same_U']=again['winner']['prediction']['U']==split['winner']['prediction']['U']
save('split_routing_correctness.json',split_test)
checks.update(split_test_passed=split_test['passed'],all_passed=checks['all_passed'] and split_test['passed'],source_freeze=ref(OUT/'optimizer_source_freeze.json'))
save('optimizer_correctness_tests.json',checks)
save('CPU_stage_completed.json',dict(cells=12,passed=checks['all_passed'],holdout_used_for_tuning=False,attributions=len(read(OUT/'routing_gap_attribution.json'))))
print('CPU_ACCEPTANCE',checks['all_passed'],flush=True)
