"""Precommitted finite-domain error comparison; calibration never uses pair validation."""
import math, statistics, time
import numpy as np
from scipy.stats import spearmanr,kendalltau
from .common import *

def samples(key):
 arrays=[]
 protocol=load(OUT/'protocol_identity.json');s=protocol['states'][key]
 for rep in [1,2,3]:
  raw=load(OUT/'raw_runs'/f'{key}__rep{rep}.json')
  assert raw['status']=='COMPLETE' and raw['gpu_release']['verified']
  assert raw['protocol_sha256']==sha(OUT/'protocol_identity.json')
  assert len(raw['target_records'])==30 and len(raw['warmup_records'])==s['warmup_count']
  records=raw['target_records']
  assert [r['pool_id'] for r in records]==[r['pool_id'] for r in s['target_items']]
  assert all(r['task']==s['task'] if 'task' in r else r['task_type']==s['task'] for r in records)
  assert len({v['pid'] for v in raw['ready'].values()})==len(s['worker_specs'])
  if s['co_model']:
   assert min(r['active_overlap_ratio'] for r in records)>=.9
  arrays.append([(r['end_ns']-r['start_ns'])/1e6 for r in records])
 return np.asarray(arrays)

def estimate(entries,features):
 by_path={};by_model={};rows=[]
 for e in entries:
  p='__'.join(e['target']);q='__'.join(e['co_path']);n=features['paths'][p]['kernel_count']
  pair=float(samples(e['id']).mean());baseline=float(samples(p+'__none').mean())
  residual=(pair-baseline)/n
  by_path.setdefault(q,[]).append(residual);by_model.setdefault(e['co_path'][0],[]).append(residual)
  rows.append({'id':e['id'],'target':e['target'],'co_path':e['co_path'],'pair_ms':pair,'standalone_ms':baseline,'kernel_count':n,'pressure_ms_per_kernel':residual})
 return {q:statistics.median(v) for q,v in by_path.items()},{m:statistics.median(v) for m,v in by_model.items()},rows

def freeze_fit():
 assert not (OUT/'calibration_fit_freeze.json').exists(),'No calibration refit'
 entries=load(OUT/'pair_split_manifest.json')['entries'];cal=[e for e in entries if e['split']=='CALIBRATION']
 assert not any((OUT/'raw_runs'/f"{e['id']}__rep1.json").exists() for e in entries if e['split']=='VALIDATION'), 'Validation must follow coefficient freeze'
 features=load(OUT/'path_features.json')
 for key,p in features['paths'].items():
  if (OUT/'raw_runs'/f'{key}__none__rep1.json').exists():
   p['frozen_catalog_standalone_ms']=p['standalone_ms'];p['standalone_ms']=float(samples(key+'__none').mean());p['standalone_anchor']='MATCHED_UNPROFILED_3_PROCESS_30_REQUEST'
  else:p['standalone_anchor']='FROZEN_UNPROFILED_CATALOG'
 dpath,dmodel,rows=estimate(cal,features)
 fit={'frozen_at_utc':now(),'delta_path_ms_per_kernel':dpath,'delta_model_ms_per_kernel':dmodel,'calibration_rows':rows,'validation_targets':sorted({tuple(e['target']) for e in entries if e['split']=='VALIDATION'}),'selected_model':None,'coefficient_source':'CALIBRATION_ONLY','negative_pressure_paths':[q for q,d in dpath.items() if d<0]}
 save('path_features.json',features);save('calibration_fit_freeze.json',fit)
 csvsave('calibration_pressure.csv',rows)
 from deployment_core.interference.scheduling_pressure import SchedulingPressureEstimator
 predictions=[]
 for e in entries:
  row={**e}
  for variant in ['I0','I1','I2']:
   pred=SchedulingPressureEstimator(features,fit,variant).pair(e['target'],e['co_path'])
   row[variant+'_mu']=pred['predicted_slowdown'];row[variant+'_latency_ms']=pred['effective_service_time_ms']
  predictions.append(row)
 save('pair_prediction_freeze.json',{'frozen_before_validation_pair_measurements':True,'at_utc':now(),'fit_sha256':sha(OUT/'calibration_fit_freeze.json'),'features_sha256':sha(OUT/'path_features.json'),'predictions':predictions})
 print(json.dumps({'phase':'COEFFICIENTS_AND_PREDICTIONS_FROZEN','delta_path':dpath,'delta_model':dmodel}),flush=True)

def ci(values):return [float(v) for v in np.quantile(values,[.025,.975])]

def block_means(array,rng,draws,indices=None):
 # Process blocks first; input IDs matched within the selected blocks.
 processes=rng.integers(0,array.shape[0],(draws,array.shape[0]))
 if indices is None:indices=rng.integers(0,array.shape[1],(draws,array.shape[1]))
 return array[processes[:,:,None],indices[:,None,:]].mean(axis=(1,2))

def metrics(pred,real):
 errors=np.asarray(pred)-np.asarray(real);absolute=np.abs(errors)
 spearman=float(spearmanr(pred,real).statistic) if len(set(pred))>1 and len(set(real))>1 else None
 kendall=float(kendalltau(pred,real).statistic) if len(set(pred))>1 and len(set(real))>1 else None
 return {'MAE':float(absolute.mean()),'median_AE':float(np.median(absolute)),'RMSE':float(np.sqrt(np.mean(errors**2))),'Spearman':spearman,'Kendall':kendall,
         'MAPE_abs_mu_gt_0_05':float(np.mean(absolute[np.abs(real)>.05]/np.abs(np.asarray(real)[np.abs(real)>.05]))) if np.any(np.abs(real)>.05) else None,'identity_count':len(real)}

def closeout():
 freeze=load(OUT/'pair_prediction_freeze.json');assert freeze['fit_sha256']==sha(OUT/'calibration_fit_freeze.json') and freeze['features_sha256']==sha(OUT/'path_features.json')
 split=load(OUT/'pair_split_manifest.json')['entries'];features=load(OUT/'path_features.json');fit=load(OUT/'calibration_fit_freeze.json')
 ids={e['id']:e for e in split};pred={e['id']:e for e in freeze['predictions']}
 ground=[];rng=np.random.default_rng(91023);draws=50000;baseline_draws={};request_indices={}
 for e in split:
  p='__'.join(e['target']);pair=samples(e['id']);base=samples(p+'__none')
  if p not in baseline_draws:
   request_indices[p]=rng.integers(0,base.shape[1],(draws,base.shape[1]));baseline_draws[p]=block_means(base,rng,draws,request_indices[p])
  pair_draws=block_means(pair,rng,draws,request_indices[p]);mu=pair_draws/baseline_draws[p]-1
  ground.append({**e,'pair_ms':float(pair.mean()),'standalone_ms':float(base.mean()),'mu_real':float(pair.mean()/base.mean()-1),'mu_real_95CI':ci(mu),'process_pair_means_ms':list(pair.mean(axis=1)),'process_standalone_means_ms':list(base.mean(axis=1))})
 validation=[g for g in ground if g['split']=='VALIDATION'];real=np.asarray([g['mu_real'] for g in validation])
 predictions={v:np.asarray([pred[g['id']][v+'_mu'] for g in validation]) for v in ['I0','I1','I2']}
 errors={v:np.abs(y-real) for v,y in predictions.items()}
 results={v:metrics(y,real) for v,y in predictions.items()}
 for v,y in predictions.items():
  results[v]['latency_MAE_ms']=float(np.mean([abs(pred[g['id']][v+'_latency_ms']-g['pair_ms']) for g in validation]))
  for role in sorted({features['paths']['__'.join(g['target'])]['role'] for g in validation}):
   selected=[i for i,g in enumerate(validation) if features['paths']['__'.join(g['target'])]['role']==role]
   results[v][role]=metrics(y[selected],real[selected])
 indices=rng.integers(0,len(validation),(draws,len(validation)));comparison={}
 for complicated,simple in [('I1','I0'),('I2','I0'),('I2','I1')]:
  delta=errors[complicated]-errors[simple];interval=ci(delta[indices].mean(axis=1))
  comparison[complicated+'-'+simple]={'mean_paired_MAE_difference':float(delta.mean()),'95CI':interval,'complex_model_supported':interval[1]<0,'method':'equal-weight pair-identity paired bootstrap, 50000 draws','conditional_on_calibration':True}
 eligible=['I0']
 for v,simpler in [('I1',['I0']),('I2',['I0','I1'])]:
  if all(comparison[v+'-'+s]['95CI'][1]<0 for s in simpler):eligible.append(v)
 selected=min(eligible,key=lambda v:(results[v]['MAE'],['I0','I1','I2'].index(v)))
 fit['selected_model']=selected;fit['selection_eligible']=eligible;fit['selected_at_utc']=now();save('fitted_model.json',fit)
 csvsave('pair_ground_truth.csv',ground);csvsave('pair_predictions.csv',freeze['predictions']);save('pair_validation_metrics.json',results);save('paired_bootstrap_model_comparison.json',comparison)
 ordering=[]
 for q in sorted({tuple(e['co_path']) for e in validation}):
  group=[g for g in validation if tuple(g['co_path'])==q]
  for a,b in zip(group,group[1:]):
   row={'co_path':q,'target_a':a['target'],'target_b':b['target'],'real_mu_difference':a['mu_real']-b['mu_real']}
   for v in predictions:
    difference=pred[a['id']][v+'_mu']-pred[b['id']][v+'_mu'];row[v+'_difference']=difference;row[v+'_sign_agrees']=bool(np.sign(difference)==np.sign(row['real_mu_difference']))
   ordering.append(row)
 csvsave('task_conditioned_ordering.csv',ordering)
 save('model_selection.json',{'selected_model':selected,'eligible_models':eligible,'metrics':results,'rule':load(OUT/'model_comparison_freeze.json')['selection'],'9A_complete':True,'no_feature_expansion':True,'finite_domain_only':True,'pair_level_bootstrap_dependence_limit':'shared targets/co-paths; not independent co-model sampling','multi_model_GPU_validation':'NOT_RUN_FIRST_LIGHTWEIGHT_VERSION','uncalibrated_paths':'zero pressure fallback and low-confidence metadata; not hard-pruned','9B_authorized_by_latest_user_request':True})
 from .uncertainty_audit import main as uncertainty_audit
 uncertainty_audit()
 from .validation import validate_and_benchmark
 validate_and_benchmark()
 save('campaign_state.json',{'phase':'9A_COMPLETE_MODEL_SELECTED_NO_FEATURE_EXPANSION','selected_model':selected,'new_completed_runs':load(OUT/'protocol_identity.json')['max_new_runs'],'at_utc':now()})
 print(json.dumps({'phase':'9A_COMPLETE','selected_model':selected,'metrics':results,'comparison':comparison}),flush=True)

if __name__=='__main__':closeout()
