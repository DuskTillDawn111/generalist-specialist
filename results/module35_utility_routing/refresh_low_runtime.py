"""Replace early LOW18 timings after history inventory / GPU are finished.

The first timings may overlap SHA inventory CPU/I/O. Preserve them verbatim.
No decision/optimizer changes; all other primary timings predate GPU launch.
"""
import sys,time,csv,itertools,shutil
from pathlib import Path
ROOT=Path('/root/autodl-tmp/generalist-specialist');OUT=ROOT/'results/module35_utility_routing';OLD=ROOT/'results/module34_candidate_pool'
sys.path[:0]=[str(OUT),str(OLD),str(ROOT)]
from m35.isolation import install
install('supervisor')
from m35.common import *
from m34.cpu_core import evaluate,catalog,candidate_routes
from u_aligned_optimizer import UAlignedOptimizer
from replica_scaling.telemetry import snapshot
g=snapshot();assert not g['processes'] and g['used_mib']<=128
assert read(OUT/'history_integrity_after.json')['passed']
freeze=read(OUT/'optimizer_source_freeze.json');assert ref(OUT/'u_aligned_optimizer.py')['sha256']==freeze['sources'][0]['sha256']
rows=list(csv.DictReader((OUT/'solver_runtime_comparison.csv').open()))
assert not (OUT/'solver_runtime_initial.csv').exists()
shutil.copyfile(OUT/'solver_runtime_comparison.csv',OUT/'solver_runtime_initial.csv')
fm=read(OLD/'frozen_problem_manifest.json');sp=read(fm['space']['path']);w=read(fm['utility_definition']['path'])['weights'];refs=read(fm['utility_reference']['path'])['latency_reference_ms'];budget=sp['budget_mib'];rates=sp['scenarios']['LOW18']['rates']
for pool in ['A','B','C']:
    d=read(sp['pools'][pool]['path']);cp=catalog(d)
    for rep in range(1,4):
        f=UAlignedOptimizer(d,rates,budget,refs,w).solve(replica_options={m:[1] for m in d['models']})
        best=None;cnt=0;start=time.perf_counter()
        for bits in itertools.product((0,1),repeat=len(d['models'])):
            x={m:1 for m,b in zip(sorted(d['models']),bits) if b}
            mem=d['safe_factor']*sum(d['memory_table_mib'][m]['1'] for m in x)
            if mem>budget+1e-6 or not set(rates)<={t for m in x for t in d['models'][m]['supported_tasks']}:continue
            for r in candidate_routes(d,x,rates,1):
                p,_,_=evaluate(d,x,r,rates,budget,refs,w,cp);cnt+=1
                if p['eligible']:best=p['U'] if best is None else max(best,p['U'])
        sec=time.perf_counter()-start;assert abs(best-f['winner']['prediction']['U'])<1e-8
        for alg,ss in [('U_ALIGNED_DP',f['seconds']),('CPU_U_ORACLE',sec)]:
            row=next(r for r in rows if r['scenario']=='LOW18' and r['pool']==pool and r['algorithm']==alg and int(r['repetition'])==rep)
            row['seconds']=ss;row['timing_batch']='POST_GPU_NO_HISTORY_INVENTORY'
        save(f'cpu/LOW18_{pool}_runtime_refresh_{rep}.json',dict(new=f,oracle_seconds=sec,oracle_routes=cnt,oracle_U=best))
    print('REFRESHED_LOW18',pool,flush=True)
csvsave('solver_runtime_comparison.csv',rows)
save('runtime_refresh_record.json',dict(reason='Initial LOW18 timing could overlap history SHA inventory; initial data preserved',optimizer_source_unchanged=True,
    refreshed='LOW18 A/B/C both algorithms three repeats',remaining='ASR32B,TEXT10,holdout all beforeGPU after inventory',GPU_idle=g))
