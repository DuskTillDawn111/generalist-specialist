"""Detached bounded campaign; no new trial after a negative outcome."""
from .common import *
from .gpu import run,memory_values
from .analyze import stage_b,stage_c

def main():
 f=read(OUT/'discovery_protocol.json');assert not (OUT/'campaign.json').exists() and not (OUT/'attempt_ledger.json').exists()
 verify(f['implementation']+[f[k] for k in ['catalog','utility_definition','utility_reference','quality_policy']]+[f['plan']['sample_pool']])
 ledger=[];executed_b=[];executed_c=[];fatal=None;queue_stop=None
 def execute(s):
  nonlocal fatal,queue_stop
  cap=f['maximum_exploration_startup_attempts'] if s['stage']=='B' else 6
  assert sum(x['stage']==s['stage'] for x in ledger)<cap,'PREREGISTERED_GPU_BUDGET_EXCEEDED'
  ledger.append(dict(run_id=s['run_id'],stage=s['stage'],at_utc=now(),state='STARTING',slot=s));save('attempt_ledger.json',ledger)
  print('GPU_START',s['run_id'],now(),flush=True)
  try:raw=run(s,f)
  except Exception as exc:
   fatal=dict(run_id=s['run_id'],reason='PRELAUNCH_OR_RUNTIME_EXCEPTION_NO_RETRY',error=repr(exc))
   save('prelaunch_failures/'+s['run_id']+'.json',dict(slot=s,error=repr(exc),valid_U=None));ledger[-1]['state']='FAILED';save('attempt_ledger.json',ledger);return None
  ledger[-1].update(state='FINISHED',raw=ref(OUT/'raw_runs'/(s['run_id']+'.json')));save('attempt_ledger.json',ledger)
  o=observe(raw,OUT/'raw_runs'/(s['run_id']+'.json'));mem=memory_values(raw,f['safe_factor'])
  print('GPU_END',s['run_id'],'eligible',o['eligible'],'unstable',o['hard_unstable'],'U',o['U'],'errors',raw['errors'],now(),flush=True)
  if not o['eligible']:
   if raw['valid_session'] and not raw['errors'] and mem['passed'] and raw['gpu_release']['verified'] and o['hard_unstable']:
    queue_stop=dict(run_id=s['run_id'],reason='UNSTABLE_QUEUE_NO_NORMAL_U',point_id=s['point_id'],G_fraction=s['G_fraction'])
   else:fatal=dict(run_id=s['run_id'],reason='ILLEGAL_RUNTIME_QUALITY_MEMORY_OR_RELEASE_FAILURE',memory=mem,errors=raw['errors'],observation=o)
  return raw
 for s in f['exploration_slots']:
  raw=execute(s)
  if raw is not None:executed_b.append(s['run_id'])
  if fatal or queue_stop:break
 save('stage_b_campaign.json',dict(at_utc=now(),executed=executed_b,fatal=fatal,queue_stop=queue_stop,
  skipped=[s['run_id'] for s in f['exploration_slots'] if s['run_id'] not in executed_b],budget_used=sum(x['stage']=='B' for x in ledger),budget_fixed=6))
 result=stage_b(executed_b,f,fatal)
 if result['Stage_C_permitted']:
  independent=read(OUT/'frozen_positive_candidate.json');queue_stop=None
  for s in independent['slots']:
   raw=execute(s)
   if raw is not None:executed_c.append(s['run_id'])
   if fatal or queue_stop:break
  stage_c(executed_c,f)
 else:save('independent_not_executed.json',dict(reason='No qualified stable exploratory positive',Stage_C=False))
 save('campaign.json',dict(at_utc=now(),executed_B=executed_b,executed_C=executed_c,fatal=fatal,queue_stop=queue_stop,
  total_GPU_attempts=len(ledger),B_attempts=sum(x['stage']=='B' for x in ledger),C_attempts=sum(x['stage']=='C' for x in ledger),
  no_budget_expansion=True,no_outcome_retries=True,no_extra_seeds=True))
 print('GPU_CAMPAIGN_STOPPED',len(executed_b),len(executed_c),flush=True)

if __name__=='__main__':main()
