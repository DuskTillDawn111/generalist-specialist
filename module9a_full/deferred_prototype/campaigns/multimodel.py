"""Development-only per-path sanity using three mechanically selected 8F2 states."""
from collections import defaultdict
import time
import numpy as np
from scipy.stats import spearmanr
from .common import *
from deployment_core.interference.routing_conditioned import InterferenceEstimator

def prepare():
 assert not (OUT/'multimodel_validation_manifest.json').exists()
 estimator=InterferenceEstimator.from_directory(OUT);bank=load(ROOT/'results/module8f2_resource_static/physical_measurements.json')['runs'];groups=defaultdict(list)
 for r in bank:
  if r['valid'] and r['stability']=='steady' and sum(n>0 for n in r['deployment'].values())>=3:groups[r['state_identity_sha256']].append(r)
 candidates=[]
 for ident,rr in groups.items():
  if len(rr)<3:continue
  config=rr[0]['identity']['config'];pred=estimator.evaluate(deployment=config['deployment'],routing=config['routing_rps'],workload=config['workload_rps']);demand=[]
  for k,route in config['routing_rps'].items():
   for m,v in route.items():
    if v>0:demand.append((v*estimator.s0[(m,k)],pred['predicted_slowdown'][k][m]))
  score=sum(w*s for w,s in demand)/sum(w for w,s in demand)
  candidates.append({'state_identity':ident,'score':score,'prediction':pred,'config':config,'runs':sorted(rr,key=lambda r:r['physical_raw_id'])[:3]})
 candidates.sort(key=lambda r:(r['score'],r['state_identity']));assert len(candidates)>=3
 indices=[0,(len(candidates)-1)//2,len(candidates)-1];selection=[]
 for label,index in zip(['low','median','high'],indices):selection.append({'stratum':label,**candidates[index]})
 save('multimodel_validation_manifest.json',{'frozen_at_utc':now(),'selection_rule':'min/lower median/max predicted demand-weighted slowdown over complete repeated steady 8F2 3+ model states; tie state identity hash; first three raw IDs per state','new_GPU_runs':0,'evidence_use':'DEVELOPMENT_PER_PATH_SERVICE_TIME_SANITY','source_protocol':'8F2 periodic mixed-task benchmark, original frozen routing and replicas, unprofiled native handler service times','same_input_protocol_as_pair_validation':False,'input_matching_caveat':'historical benchmark per-path input cohorts/counts differ from fresh 9A pair fixed n30; not pair predictor accuracy ground truth; direct service timing rather than E2E, input mix is a confounder','all_candidates':[{k:r[k] for k in ['state_identity','score']} for r in candidates],'states':selection})

def analyze():
 manifest=load(OUT/'multimodel_validation_manifest.json');estimator=InterferenceEstimator.from_directory(OUT);rows=[];states=[]
 for state in manifest['states']:
  config=state['config'];values=defaultdict(list);runmeans=defaultdict(list)
  for run in state['runs']:
   raw=load(run['raw_path']);assert sha(run['raw_path'])==run['raw_sha256'];local=defaultdict(list)
   for r in raw['request_records']:
    if r['status']=='completed':local[(r['model_family'],r['task'])].append(r['worker_latency_ms'])
   for p,v in local.items():values[p].extend(v);runmeans[p].append(float(np.mean(v)))
  pathrows=[]
  for k,route in config['routing_rps'].items():
   for m,rate in route.items():
    if rate<=0:continue
    p=(m,k);means=np.array(runmeans[p]);assert len(means)==3
    predicted=state['prediction']['predicted_slowdown'][k][m];observed=float(np.mean(values[p])/estimator.s0[p]);rng=np.random.default_rng(int(digest([state['state_identity'],m,k])[:8],16));boot=means[rng.integers(0,3,size=(10000,3))].mean(axis=1)/estimator.s0[p];ci=np.quantile(boot,[.025,.975]);row={'state_identity':state['state_identity'],'stratum':state['stratum'],'resident_models':sum(n>0 for n in config['deployment'].values()),'model':m,'task':k,'replicas':config['deployment'][m],'predicted_slowdown':predicted,'historical_service_time_slowdown':observed,'historical_service_time_CI95':ci.tolist(),'historical_run_slowdowns':(means/estimator.s0[p]).tolist(),'observed_service_mean_ms':float(np.mean(values[p])),'frozen_standalone_s0_ms':estimator.s0[p],'absolute_slowdown_error':abs(predicted-observed),'n_requests':len(values[p]),'repetitions':3,'base_utilization':state['prediction']['base_utilization'][m],'evidence_type':'DEVELOPMENT_PER_PATH_SANITY_UNMATCHED_INPUTS','pair_accuracy_ground_truth':False,'SYSTEM_LEVEL_SANITY_CHECK':False,'input_cohort_protocol_matches_pair':False};rows.append(row);pathrows.append(row)
  a=np.array([r['predicted_slowdown'] for r in pathrows]);b=np.array([r['historical_service_time_slowdown'] for r in pathrows]);rank=float(spearmanr(a,b).statistic) if len(set(a))>1 and len(set(b))>1 else None;opposite=sum(r['historical_service_time_CI95'][1]<1 and r['predicted_slowdown']>1 for r in pathrows);above=sum(r['predicted_slowdown']>max(r['historical_run_slowdowns']) for r in pathrows);catastrophe=opposite>len(pathrows)/2 or (above==len(pathrows) and rank is not None and rank<=0)
  states.append({'state_identity':state['state_identity'],'stratum':state['stratum'],'path_count':len(pathrows),'mean_absolute_slowdown_error':float(np.mean(abs(a-b))),'Spearman':rank,'predicted_real_ratio_by_path':[float(x/y) for x,y in zip(a,b)],'opposite_direction_CI_paths':opposite,'prediction_above_all_repetitions_paths':above,'catastrophic_mismatch':catastrophe,'not_heldout_validation':True})
 csvsave('multimodel_validation.csv',rows);save('multimodel_sanity_summary.json',{'states':states,'new_GPU_runs':0,'acceptable_sanity':not any(r['catastrophic_mismatch'] for r in states),'caveat':manifest['input_matching_caveat'],'accuracy_headline':False,'effective_service_ms_includes_native_CPU_GPU_path':True})
 print(json.dumps({'multi_model_sanity':states}))
if __name__=='__main__':prepare();analyze()
