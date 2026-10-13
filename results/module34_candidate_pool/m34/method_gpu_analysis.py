from .common import *
import numpy as np

def main():
 ds=read(OUT/'optimizer_decisions.json');f=read(OUT/'frozen_gpu_manifest_v2.json');slots={s['run_id']:s for s in f['slots']};observations=read(OUT/'gpu_results_table.json');cs=read(OUT/'gpu_contrasts.json');decomposition=[];regrets=[]
 for c in cs:
  delta=c['delta_statistics']['mean'];ci=c['delta_statistics']['ci95'];decomposition.append(dict(group=c['group'],predicted_oracle_minus_competitor=c['predicted_delta_U'],observed_signed_difference=delta,differential_prediction_error=delta-c['predicted_delta_U'] if delta is not None else None,prediction_error_ci95=[v-c['predicted_delta_U'] for v in ci] if ci else None,
   interpretation='Observed signed difference = frozen predicted difference + differential prediction error. Can be negative; not an additive allocation of nonnegative causes. Oracle-selected plan is not guaranteed GPU best.'))
 for pool,scenario,groups in [('A','TEXT_HALF_MID18',['A_TEXT_HALF_MID18']),('B','TEXT_HALF_MID18',['A_TEXT_HALF_MID18','B_TEXT_HALF_MID18']),('C','ASR32B',['C_ASR32B'])]:
  cohort=[r for r in observations if r['stage']=='INDEPENDENT' and r['group'] in groups];plans={}
  for r in cohort:
   s=slots[r['run_id']];key=identity(s['deployment'],s['routing']);plans.setdefault(key,[]).append(r)
  complete={key:rr for key,rr in plans.items() if len(rr)==3 and all(r['eligible'] for r in rr)};means={key:float(np.mean([r['U'] for r in rr])) for key,rr in complete.items()};expected=2 if pool!='B' else 4;allcomplete=len(complete)==expected;best=max(means.values()) if allcomplete else None
  for r in [r for r in ds if r['pool']==pool and r['scenario']==scenario]:
   key=identity(r['deployment'],r['routing']) if r.get('deployment') else None;rr=plans.get(key,[]);mean=means.get(key)
   regrets.append(dict(pool=pool,scenario=scenario,algorithm=r['algorithm'],CPU_status=r['status'],measured_same_plan=key in plans,valid_independent_windows=sum(x['eligible'] for x in rr),all_required_small_measured_set_complete=allcomplete,observed_mean_U=mean,measured_set_best_mean_U=best,empirical_point_regret=best-mean if best is not None and mean is not None else None,
    measured_set_expected_configs=expected,scope='Two frozen measured plans in A/C; four TEXT10 plans also all admittedB. Point regret relative to highest independent observed mean, not fullCPU pool Oracle; no CI for post-selected winner. Use predeclared pairedCI/Holm for ranking claims. Native20A equivalent-fingerprint routes shared, nativeV3 missing/failed result never imputed.'))
 csvsave('prediction_search_decomposition.csv',decomposition);csvsave('method_empirical_regret.csv',regrets);save('method_empirical_regret.json',regrets)
 rawobs=read(OUT/'gpu_observations.json');stability=[]
 for o in rawobs:
  s=slots[o['run_id']];stability.append(dict(run_id=o['run_id'],group=s['group'],stage=s['stage'],arm=s['arm'],system_state=o['system_stability']['stability_state_v2'],system_recovery_confirmed=o['system_stability']['final_recovery_confirmed'],worker_states={m:r['stability_state_v2'] for m,r in o['worker_stability'].items()},eligible=o['eligible'],U=o['U'],policy='Original hard unstable disallowed; recoverable_congested permitted only with fullcohort and confirmed recovery. Legal U does not imply all windows steady.'))
 csvsave('gpu_stability_audit.csv',stability);save('gpu_stability_audit.json',stability);print('METHOD_GPU_ANALYSIS',len(regrets))
if __name__=='__main__':main()
