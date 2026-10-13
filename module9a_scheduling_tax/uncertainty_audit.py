"""Predeclared robustness checks; never changes features, coefficients or selection."""
import statistics
import numpy as np
from .common import *
from .analysis import samples,block_means,ci

def main():
 entries=load(OUT/'pair_split_manifest.json')['entries'];cal=[e for e in entries if e['split']=='CALIBRATION'];val=[e for e in entries if e['split']=='VALIDATION']
 features=load(OUT/'path_features.json');fit=load(OUT/'calibration_fit_freeze.json')
 rng=np.random.default_rng(91024);draws=20000;base={};request_indices={};pair={}
 for e in entries:
  p='__'.join(e['target'])
  if p not in base:
   a=samples(p+'__none');request_indices[p]=rng.integers(0,a.shape[1],(draws,a.shape[1]));base[p]=block_means(a,rng,draws,request_indices[p])
  pair[e['id']]=block_means(samples(e['id']),rng,draws,request_indices[p])
 residual={e['id']:(pair[e['id']]-base['__'.join(e['target'])])/features['paths']['__'.join(e['target'])]['kernel_count'] for e in cal}
 paths={};models={}
 for q in {tuple(e['co_path']) for e in cal}:
  rows=[residual[e['id']] for e in cal if tuple(e['co_path'])==q]
  paths['__'.join(q)]=np.median(np.stack(rows),axis=0)
 for m in {e['co_path'][0] for e in cal}:
  rows=[residual[e['id']] for e in cal if e['co_path'][0]==m]
  models[m]=np.median(np.stack(rows),axis=0)
 real=np.stack([pair[e['id']]/base['__'.join(e['target'])]-1 for e in val],axis=1)
 predicted={'I0':np.zeros_like(real),'I1':np.stack([features['paths']['__'.join(e['target'])]['kernel_count']*models[e['co_path'][0]]/base['__'.join(e['target'])] for e in val],axis=1),'I2':np.stack([features['paths']['__'.join(e['target'])]['kernel_count']*paths['__'.join(e['co_path'])]/base['__'.join(e['target'])] for e in val],axis=1)}
 errors={v:np.abs(y-real) for v,y in predicted.items()};identity_indices=rng.integers(0,len(val),(draws,len(val)));comparisons={}
 freeze=load(OUT/'pair_prediction_freeze.json');point={r['id']:r for r in freeze['predictions']}
 gt={r['id']:float(samples(r['id']).mean()/samples('__'.join(r['target'])+'__none').mean()-1) for r in val}
 # Shared co-paths/targets are explicitly clustered in alternative CIs.
 for complex_,simple in [('I1','I0'),('I2','I0'),('I2','I1')]:
  diff=errors[complex_]-errors[simple];hier=diff[np.arange(draws)[:,None],identity_indices].mean(axis=1)
  pointdiff=np.asarray([abs(point[e['id']][complex_+'_mu']-gt[e['id']])-abs(point[e['id']][simple+'_mu']-gt[e['id']]) for e in val])
  clustered={}
  for label,key in [('co_path',lambda e:tuple(e['co_path'])),('target_path',lambda e:tuple(e['target']))]:
   groups=sorted({key(e) for e in val});groupsums=np.asarray([pointdiff[[i for i,e in enumerate(val) if key(e)==g]].sum() for g in groups]);sizes=np.asarray([sum(key(e)==g for e in val) for g in groups]);chosen=rng.integers(0,len(groups),(draws,len(groups)))
   clustered[label]={'clusters':len(groups),'95CI':ci(groupsums[chosen].sum(axis=1)/sizes[chosen].sum(axis=1))}
  comparisons[complex_+'-'+simple]={'process_block_refit_plus_identity_95CI':ci(hier),'clustered_point_fit_CIs':clustered}
 save('uncertainty_sensitivity.json',{'predeclared_before_calibration_and_validation_pairs':True,'purpose':'robustness audit only; no change to frozen primary selection','process_repetitions_resampled_with_replacement':3,'matched_request_indices_joint_across_pair_and_baseline':True,'refit_from_calibration_only':True,'draws':draws,'comparisons':comparisons,'delta_path_process_refit_95CI':{q:ci(a) for q,a in paths.items()},'delta_model_process_refit_95CI':{m:ci(a) for m,a in models.items()},'limitations':'co-path clusters=3, target clusters=4; shared fitted coefficients and narrow co-model sample prevent population-level claims; no multiplicity correction in the original 95% primary comparison'})

if __name__=='__main__':main()
