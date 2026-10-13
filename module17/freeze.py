"""Pre-register at most two diagnostic conditions and two independent pairs."""
import copy
from pathlib import Path
from scipy.stats import t
from .common import *
def main():
 assert not (OUT/'diagnostic_manifest.json').exists();e=read(OUT/'runtime_bottleneck_evidence.json');assert e['stage']=='A_CPU_ONLY_COMPLETE_BEFORE_ANY_NEW_GPU' and e['need_minimal_GPU_diagnostics']
 old=read(ROOT/'results/module16_fullpool_gpu/frozen_gpu_manifest.json');base=next(s for s in old['slots'] if s['run_id']=='mid_rep1');slots=[]
 for rep,seed in enumerate([17101,17202],1):
  for condition in ['FULL_MID_REFERENCE','SPEECH_ONLY_SAME_RESIDENCY']:
   slot=copy.deepcopy(base);slot.update(run_id=condition.lower()+'_rep'+str(rep),condition=condition,stage='DIAGNOSTIC',repetition=rep,seed=seed)
   if condition=='SPEECH_ONLY_SAME_RESIDENCY':
    for k in slot['lambda_rps']:
     if k not in ['asr','lid']:slot['lambda_rps'][k]=0.;slot['routing'][k]={}
   slots.append(slot)
 for s in slots:
  assert s['lambda_rps']['asr']==1. and s['lambda_rps']['lid']==12. and s['deployment']==base['deployment'] and s['routing']['asr']==base['routing']['asr'] and s['routing']['lid']==base['routing']['lid']
 oldtasks=list(csv.DictReader((OUT/'mid_task_path_diagnosis.csv').open()));bounds=[]
 for (m,k) in [(next(m for m in base['deployment'] if m.startswith('openai/whisper-medium')),'lid'),('whisper_turbo','asr'),('whisper_turbo','lid')]:
  means=[float(r['worker_service_mean_ms']) for r in oldtasks if r['worker'].split('#')[0]==m and r['task']==k];assert len(means)==3
  upper=max(max(means),statistics.mean(means)+float(t.ppf(.95,2))*statistics.stdev(means)/math.sqrt(3))
  bounds.append({'model':m,'task':k,'replica_count':1,'training_run_means_ms':means,'proposed_service_mean_bound_ms':upper,'rule':'max(observed repetition mean, one-sided 95% t upper confidence limit on repetition mean); diagnostic proposed bound, not a deterministic per-request upper bound','accepted':False})
 save('capacity_bound_proposal_freeze.json',{'at_utc':now(),'trained_only_on_Module16_MID_three_runs':True,'bounds':bounds,'heldout_runs':[s['run_id'] for s in slots],'heldout_acceptance':'Every heldout complete independent run must have each path cohort mean <= training-only bound, conserved counts, unchanged worker semantics and valid session. No refitting/raising bounds after seeing diagnostics.','transfer_limit':'At most observed original/zero non-speech background under identical six resident checkpoints/counts, ASR=1,LID=12 and unchanged speech routes. These tests do not certify arbitrary new pool selections, replica counts, different speech task mixtures or changed speech QPS. Do not authorize a new three-phase witness based solely on this fitted scope.'})
 implementation=[ref(p) for p in sorted((ROOT/'module17').glob('*.py'))];reused=old['implementation']+old['reused_implementation'];verify(reused+[old['CPU'],old['catalog'],old['plan']['sample_pool']])
 save('diagnostic_manifest.json',{'at_utc':now(),'status':'FROZEN_AFTER_STAGE_A_BEFORE_GPU','stage_A':ref(OUT/'runtime_bottleneck_evidence.json'),'training_proposal':ref(OUT/'capacity_bound_proposal_freeze.json'),'catalog':old['catalog'],'plan':old['plan'],'slots':slots,'conditions':2,'maximum_repetitions_per_condition':2,'maximum_sessions':4,'experiment_type':'DIAGNOSIS_ONLY_NOT_FORMAL_THREE_PHASE_VALIDATION','changed_asr_lid_QPS_routes':False,'unchanged_all_six_residencies':True,'controls':'Paired same seeds and identical task-specific sample sequences/periodic phases for ASR and LID. Only text/image/OCR/VQA offered traffic is disabled in isolation; all six checkpoints remain resident and receive identical premeasurement prime calls.','stop_condition':'Runtime/resource failure, interference by foreign GPU PID, or failed release stops diagnosis. Diagnostic unstable is an outcome, not a reason to retune or retry.','runtime_observation':'Read-only per-worker /proc task schedstat, context switches and CPU cgroup/affinity. No worker/model inference changes. NVML and original request timestamp telemetry unchanged.','implementation':implementation,'reused_implementation':reused,'GPU_protocol':{'warmup_seconds':10,'measurement_seconds':120,'drain_seconds':60,'seeds':[17101,17202]},'no_new_model_search':True,'no_Module9A_restart':True,'no_point_movement_or_outcome_retry':True})
 print('DIAGNOSTIC_FROZEN',[(s['run_id'],s['lambda_rps']) for s in slots],flush=True)
if __name__=='__main__':main()
