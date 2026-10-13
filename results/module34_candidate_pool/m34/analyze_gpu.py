from .common import *
import numpy as np
from scipy.stats import t as tdist,ttest_1samp

def stats(v):
 v=np.asarray(v,float);n=len(v);mean=float(v.mean()) if n else None;sd=float(v.std(ddof=1)) if n>1 else None;half=float(tdist.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
 return dict(n=n,mean=mean,sd=sd,ci95=[mean-half,mean+half] if half is not None else None,p_two_sided=float(ttest_1samp(v,0).pvalue) if n>1 and sd>0 else None)
def main():
 f=read(OUT/'frozen_gpu_manifest_v2.json');by={s['run_id']:s for s in f['slots']};obs=read(OUT/'gpu_observations.json') if (OUT/'gpu_observations.json').exists() else [];rows=[];errors=[]
 for o in obs:
  s=by[o['run_id']];r={**{k:s[k] for k in ['run_id','group','arm','pool','scenario','stage','repetition','seed','predicted_U']},**{k:o[k] for k in ['U','eligible','technical_valid','stable','environment']},'checks':o['checks']};r['U_prediction_error']=r['U']-r['predicted_U'] if r['U'] is not None else None;rows.append(r)
 csvsave('gpu_oracle_runs.csv',rows);save('gpu_results_table.json',rows)
 contrasts=[]
 for group in sorted({s['group'] for s in f['slots']}):
  dev=[r for r in rows if r['group']==group and r['stage']=='DEVELOPMENT'];inde=[r for r in rows if r['group']==group and r['stage']=='INDEPENDENT'];matched=[]
  for rep in [1,2,3]:
   a=next((r for r in inde if r['repetition']==rep and r['arm']=='ORACLE'),None);b=next((r for r in inde if r['repetition']==rep and r['arm']=='COMPETITOR'),None)
   if not a or not b or not a['eligible'] or not b['eligible']:continue
   assert a['seed']==b['seed'];sa=by[a['run_id']];sb=by[b['run_id']];assert sa['trajectories']==sb['trajectories']
   temp=abs(a['environment']['temperature_gpu_c']-b['environment']['temperature_gpu_c']);clock=abs(a['environment']['sm_clock_mhz']/b['environment']['sm_clock_mhz']-1)
   matched.append(dict(repetition=rep,seed=a['seed'],oracle_U=a['U'],competitor_U=b['U'],delta_U=a['U']-b['U'],temperature_mean_delta_c=temp,clock_relative_delta=clock,environment_comparable=temp<=5 and clock<=.15))
  ss=stats([r['delta_U'] for r in matched]);ci=ss['ci95'];prediction=next(s['predicted_U'] for s in f['slots'] if s['group']==group and s['arm']=='ORACLE')-next(s['predicted_U'] for s in f['slots'] if s['group']==group and s['arm']=='COMPETITOR');complete=len(matched)==3
  contrast=dict(group=group,matched=matched,delta_statistics=ss,predicted_delta_U=prediction,development_valid=len(dev)==2 and all(r['eligible'] for r in dev),independent_complete=complete,
   environment_comparable=complete and all(r['environment_comparable'] for r in matched),development_best=max((r for r in dev if r['eligible']),key=lambda r:r['U'])['arm'] if any(r['eligible'] for r in dev) else None,
   verdict='INSUFFICIENT_INDEPENDENT_DATA' if not complete else 'ENVIRONMENT_CONFOUNDED' if not all(r['environment_comparable'] for r in matched) else 'PREDICTED_ORACLE_SUPPORTED' if ci and ci[0]>0 else 'GPU_RANKING_REVERSED' if ci and ci[1]<0 else 'STATISTICALLY_UNRESOLVED')
  contrasts.append(contrast)
 # Same-request A/B pool expansion witness at TEXT10. No added experiments.
 exp=[]
 for rep in [1,2,3]:
  a=next((r for r in rows if r['group']=='A_TEXT_HALF_MID18' and r['stage']=='INDEPENDENT' and r['arm']=='ORACLE' and r['repetition']==rep and r['eligible']),None);b=next((r for r in rows if r['group']=='B_TEXT_HALF_MID18' and r['stage']=='INDEPENDENT' and r['arm']=='ORACLE' and r['repetition']==rep and r['eligible']),None)
  if a and b:
   assert by[a['run_id']]['trajectories']==by[b['run_id']]['trajectories'];exp.append(dict(rep=rep,G_only_U=a['U'],Mixed_U=b['U'],Mixed_minus_G=b['U']-a['U'],temperature_mean_delta_c=abs(a['environment']['temperature_gpu_c']-b['environment']['temperature_gpu_c']),clock_relative_delta=abs(a['environment']['sm_clock_mhz']/b['environment']['sm_clock_mhz']-1)))
 exps=stats([r['Mixed_minus_G'] for r in exp]);save('gpu_pool_expansion_pair.json',dict(matched=exp,statistics=exps,scope='Two preselected competing plans admitted in B; not all-B measured oracle.'))
 # Holm correction across three predeclared within-pool pair contrasts.
 eligible=[c for c in contrasts if c['independent_complete'] and c['environment_comparable'] and c['delta_statistics']['p_two_sided'] is not None];eligible.sort(key=lambda c:c['delta_statistics']['p_two_sided']);running=0.
 for i,c in enumerate(eligible):running=max(running,min(1.,(3-i)*c['delta_statistics']['p_two_sided']));c['holm_adjusted_p']=running;c['significant_familywise_05']=running<.05
 save('gpu_contrasts.json',contrasts);csvsave('gpu_empirical_regret.csv',[dict(group=c['group'],predicted_delta_U=c['predicted_delta_U'],independent_complete=c['independent_complete'],environment_comparable=c['environment_comparable'],verdict=c['verdict'],observed_delta=c['delta_statistics']['mean'],ci95=c['delta_statistics']['ci95'],n=c['delta_statistics']['n'],competitor_observed_regret=max(0.,c['delta_statistics']['mean']) if c['delta_statistics']['mean'] is not None else None,oracle_observed_regret=max(0.,-c['delta_statistics']['mean']) if c['delta_statistics']['mean'] is not None else None,holm_adjusted_p=c.get('holm_adjusted_p'),scope='Best among frozen measured pair; no full-pool GPU Oracle') for c in contrasts]);print('GPU_ANALYSIS',len(rows),flush=True)
if __name__=='__main__':main()
