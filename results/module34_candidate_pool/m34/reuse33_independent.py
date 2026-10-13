from .common import *
from module26.common import arithmetic
from .analyze_gpu import stats
import numpy as np
from scipy.stats import t as tdist

def main():
 fm=read(OUT/'frozen_problem_manifest.json');sp=read(fm['space']['path']);d=read(sp['pools']['C']['path']);refs=read(fm['utility_reference']['path'])['latency_reference_ms'];weights=read(fm['utility_definition']['path'])['weights'];old=read(ROOT/'results/module33_mixed_oracle/pair_observations.json');rows=[]
 for o in old:
  raw=read(o['raw']['path']);s=raw['slot'];assert o['eligible'] and all(n==1 for n in s['deployment'].values()) and all(m in d['models'] for m in s['deployment']) and all(len(rr)==1 for rr in s['routing'].values());assert s['lambda_rps']==sp['scenarios']['ASR32B']['rates']
  origbudget=s['safe_memory_budget_mib'];raw['slot']['safe_memory_budget_mib']=sp['budget_mib'];a=arithmetic(raw,d,refs,weights)
  rows.append(dict(run_id=o['run_id'],candidate_id=o['candidate_id'],old_U=o['U'],rebased_U=a['raw_arithmetic_U'],old_budget=origbudget,new_budget=sp['budget_mib'],seed=s['seed'],seconds=s['measurement_seconds'],deployment=s['deployment'],routing=s['routing'],environment=o['environment'],baseline_GPU=raw['baseline_GPU'],input_trace=s['trajectories']['measurement'],source_raw=ref(o['raw']['path']),scope='Original33 independent LargeASR+Phi6 endpoint plan admitted in C; rebaseM only; not paired to34 seeds.'))
 save('module33_GPU_reuse_independent.json',rows);csvsave('module33_GPU_reuse_independent.csv',rows)
 current=read(OUT/'gpu_results_table.json') if (OUT/'gpu_results_table.json').exists() else [];new=[r for r in current if r['group']=='C_ASR32B' and r['stage']=='INDEPENDENT' and r['arm']=='ORACLE' and r['eligible']];sv=stats([r['rebased_U'] for r in rows]);sn=stats([r['U'] for r in new]);comparison=dict(old_statistics=sv,new_statistics=sn,paired=False,secondary_exploratory=True,full_pool_GPU_oracle=False,scope='Independent cohorts and different seed/media realizations. Unpaired Welch uncertainty; not a primary Holm comparison or all-C Oracle. Same original6-task rates/240s protocol/native workers, current samephysicalGPU verified. No algorithm changes.')
 if len(new)>=2:
  x=np.asarray([r['U'] for r in new]);y=np.asarray([r['rebased_U'] for r in rows]);vx=x.var(ddof=1)/len(x);vy=y.var(ddof=1)/len(y);df=(vx+vy)**2/(vx**2/(len(x)-1)+vy**2/(len(y)-1));diff=float(x.mean()-y.mean());half=float(tdist.ppf(.975,df)*np.sqrt(vx+vy));te=abs(np.mean([r['environment']['temperature_gpu_c'] for r in new])-np.mean([r['environment']['temperature_gpu_c'] for r in rows]));cl=abs(np.mean([r['environment']['sm_clock_mhz'] for r in new])/np.mean([r['environment']['sm_clock_mhz'] for r in rows])-1)
  samegpu=all(r['baseline_GPU']['uuid']==read(OUT/'frozen_gpu_manifest_v2.json')['GPU']['uuid'] for r in rows);comparison.update(new_minus_old_U=diff,welch_df=float(df),ci95=[diff-half,diff+half],temperature_mean_delta_c=float(te),clock_relative_delta=float(cl),same_GPU_UUID=samegpu,environment_comparable=samegpu and te<=5 and cl<=.15)
 save('historical_large_vs_new_small_analysis.json',comparison);print('REUSE_INDEPENDENT',len(rows),'new',len(new))
if __name__=='__main__':main()
