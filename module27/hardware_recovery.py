"""One pre-inference identity correction; budget loses one scheduled grid cell."""
from .common import *
from . import gpu,analyze

def main():
 amendment=read(OUT/'hardware_identity_amendment.json');original=read(OUT/'discovery_protocol.json')
 verify(original['implementation']+[original[k] for k in ['catalog','utility_definition','utility_reference','quality_policy']]);verify(amendment['implementation'])
 assert not (OUT/'campaign.json').exists(),'No GPU outcome campaign retry'
 f=copy.deepcopy(original);f['plan']['GPU']=amendment['current_GPU']
 ledger=read(OUT/'initial_prelaunch_failure/attempt_ledger.json');ledger[0]['GPU_workers_started']=False
 executed_b=[];executed_c=[];fatal=None;queue_stop=None
 save('attempt_ledger.json',ledger)
 gpu_save=gpu.save
 def tagged(name,value):
  if name.startswith('raw_runs/'):
   value['hardware_identity_amendment']=ref(OUT/'hardware_identity_amendment.json')
  gpu_save(name,value)
 gpu.save=tagged
 analysis_save=analyze.save
 def tagged_analysis(name,value):
  if name=='frozen_positive_candidate.json':
   value['hardware_identity_amendment']=ref(OUT/'hardware_identity_amendment.json');value['actual_GPU']=amendment['current_GPU']
  analysis_save(name,value)
 analyze.save=tagged_analysis
 def execute(s):
  nonlocal fatal,queue_stop
  cap=6
  assert sum(v['stage']==s['stage'] for v in ledger)<cap,'No budget expansion even counting original identity check'
  ledger.append(dict(run_id=s['run_id'],stage=s['stage'],at_utc=now(),state='STARTING',slot=s,GPU_workers_started=None));save('attempt_ledger.json',ledger)
  print('GPU_START',s['run_id'],now(),flush=True)
  try:raw=gpu.run(s,f)
  except Exception as exc:
   fatal=dict(run_id=s['run_id'],reason='PRELAUNCH_OR_RUNTIME_FAILURE_NO_SECOND_RECOVERY',error=repr(exc))
   save('prelaunch_failures/'+s['run_id']+'.json',dict(slot=s,error=repr(exc),normal_U=None));ledger[-1]['state']='FAILED';save('attempt_ledger.json',ledger);return None
  ledger[-1].update(state='FINISHED',GPU_workers_started=bool(raw['worker_identities']),raw=ref(OUT/'raw_runs'/(s['run_id']+'.json')));save('attempt_ledger.json',ledger)
  ob=observe(raw,OUT/'raw_runs'/(s['run_id']+'.json'));mem=gpu.memory_values(raw,f['safe_factor'])
  print('GPU_END',s['run_id'],'eligible',ob['eligible'],'unstable',ob['hard_unstable'],'U',ob['U'],'errors',raw['errors'],now(),flush=True)
  if not ob['eligible']:
   if raw['valid_session'] and not raw['errors'] and mem['passed'] and raw['gpu_release']['verified'] and ob['hard_unstable']:
    queue_stop=dict(run_id=s['run_id'],reason='UNSTABLE_QUEUE_NO_NORMAL_U',point_id=s['point_id'],G_fraction=s['G_fraction'])
   else:fatal=dict(run_id=s['run_id'],reason='ILLEGAL_RUNTIME_QUALITY_MEMORY_OR_RELEASE_FAILURE',memory=mem,errors=raw['errors'])
  return raw
 for s in amendment['remaining_exploration_slots']:
  raw=execute(s)
  if raw is not None:executed_b.append(s['run_id'])
  if fatal or queue_stop:break
 save('stage_b_campaign.json',dict(at_utc=now(),executed=executed_b,fatal=fatal,queue_stop=queue_stop,
  first_prelaunch_check_retained=ref(OUT/'initial_prelaunch_failure/attempt_ledger.json'),
  skipped=amendment['cancelled_grid_cells']+[s['run_id'] for s in amendment['remaining_exploration_slots'] if s['run_id'] not in executed_b],
  budget_used_including_failed_identity_check=sum(v['stage']=='B' for v in ledger),budget_fixed=6,
  actual_model_startup_attempts=sum(bool(v.get('GPU_workers_started')) for v in ledger if v['stage']=='B')))
 b=analyze.stage_b(executed_b,f,fatal)
 if b['Stage_C_permitted']:
  queue_stop=None;candidate=read(OUT/'frozen_positive_candidate.json')
  for s in candidate['slots']:
   raw=execute(s)
   if raw is not None:executed_c.append(s['run_id'])
   if fatal or queue_stop:break
  analyze.stage_c(executed_c,f)
 else:save('independent_not_executed.json',dict(reason='No qualified stable exploratory positive',Stage_C=False))
 save('campaign.json',dict(at_utc=now(),executed_B=executed_b,executed_C=executed_c,fatal=fatal,queue_stop=queue_stop,
  total_GPU_attempts=len(ledger)-1,B_attempts=sum(v['stage']=='B' for v in ledger)-1,C_attempts=sum(v['stage']=='C' for v in ledger),
  initial_pre_inference_identity_check_count=1,B_budget_checks_used=sum(v['stage']=='B' for v in ledger),
  hardware_identity_amendment=ref(OUT/'hardware_identity_amendment.json'),
  no_budget_expansion=True,no_outcome_retries=True,no_extra_seeds=True))
 print('GPU_CAMPAIGN_STOPPED',len(executed_b),len(executed_c),flush=True)

if __name__=='__main__':main()
