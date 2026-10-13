"""Build all cross-model path predictions offline, then freeze bounded validation."""
import time
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr,kendalltau
from .common import *
from deployment_core.interference.task_path_profile import TaskPathProfile
from deployment_core.interference.occupancy import DeviceLimits
from deployment_core.interference.pairwise_predictor import predict_pair

def limits():
 a=load(OUT/'hardware_metric_audit.json')['device_attributes'];return DeviceLimits(a['max_threads_per_sm']//a['warp_size'],a['max_blocks_per_sm'],a['regs_per_sm'],a['shared_per_sm'],a['sm_count'],a['reserved_shared_per_block'])
def main():
 assert not (OUT/'pair_predictions.csv').exists(),'no outcome-dependent predictor revision'
 c=load(OUT/'catalog.json');profiles={(r['model'],r['task']):TaskPathProfile.load(OUT/'task_path_profiles'/(r['model']+'__'+r['task']+'.json')) for r in c['paths']};device=limits();rows=[];rawrows=[];started=time.perf_counter()
 for p in sorted(profiles):
  for q in sorted(profiles):
   if p[0]==q[0]:continue
   pred=predict_pair(profiles[p],profiles[q],device).as_dict();raw=predict_pair(profiles[p],profiles[q],device,raw=True).as_dict();rows.append(pred);rawrows.append(raw)
  print(json.dumps({'event':'PREDICTIONS_TARGET_COMPLETE','target':p,'pairs_done':len(rows)}),flush=True)
 csvsave('pair_predictions.csv',rows);csvsave('pair_predictions_raw_diagnostic.csv',rawrows)
 corrected=np.array([r['mu_hat'] for r in rows]);raw=np.array([r['mu_hat'] for r in rawrows]);order=[(np.sign(corrected[i]-corrected[j]),np.sign(raw[i]-raw[j])) for i in range(len(rows)) for j in range(i)];comparable=[(a,b) for a,b in order if a!=0 and b!=0]
 audit=load(OUT/'duration_reconstruction_audit.json');audit['raw_vs_corrected_sensitivity']={'Spearman':float(spearmanr(corrected,raw).statistic),'Kendall':float(kendalltau(corrected,raw).statistic),'comparable_pair_rankings':len(comparable),'rankings_changed':sum(a!=b for a,b in comparable),'ranking_agreement':float(np.mean([a==b for a,b in comparable])),'ranking_changed':any(a!=b for a,b in comparable),'scope':'all profiled cross-model pairs; raw prediction diagnostic only','correction_selected_before_outcomes':True};save('duration_reconstruction_audit.json',audit)
 save('offline_prediction_cost.json',{'cross_model_pairs':len(rows),'elapsed_seconds':time.perf_counter()-started,'online_GPU_required':False,'online_simulation_required':False})
 print(json.dumps({'pairs':len(rows),'offline_seconds':time.perf_counter()-started}))
if __name__=='__main__':main()
