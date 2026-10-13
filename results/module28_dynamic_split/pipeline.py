"""Durable bounded campaign. No automatic next module, no outcome retries."""
from audit import *
from gpu import run
from report import evaluate,summarize_groups,write_tables,finalize
import traceback

def main():
 f=json.loads((OUT/'frozen_candidate_manifest.json').read_text())
 assert f['status']=='FROZEN_BEFORE_GPU'
 for r in f['implementation']+[f['catalog'],f['utility_definition'],f['utility_reference'],f['quality_policy']]:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 ledger=[];raws=[];fatal=None
 def execute(slots):
  nonlocal fatal
  for slot in slots:
   assert len(ledger)<f['maximum_total_attempts']
   assert not (OUT/'raw_runs'/(slot['run_id']+'.json')).exists(),'No outcome retry'
   ledger.append(dict(run_id=slot['run_id'],stage=slot['stage'],order=len(ledger)+1,status='STARTING',start_unix_ns=time.time_ns()));save('attempt_ledger.json',ledger)
   print('START',slot['run_id'],flush=True)
   try:raw=run(slot,f);raws.append(raw);v=evaluate(raw,f)
   except Exception as exc:
    fatal=dict(run_id=slot['run_id'],reason=repr(exc),traceback=traceback.format_exc());ledger[-1].update(status='EXCEPTION',error=fatal);save('attempt_ledger.json',ledger);return
   ledger[-1].update(status=raw['status'],end_unix_ns=time.time_ns(),eligible=v['eligible'],U=v['U']);save('attempt_ledger.json',ledger);write_tables(raws,f)
   print('FINISH',slot['run_id'],'eligible',v['eligible'],'U',v['U'],'errors',raw['errors'],flush=True)
   if not v['eligible']:
    fatal=dict(run_id=slot['run_id'],reason='ILLEGAL_UNSTABLE_OR_INFRASTRUCTURE_FAILURE',errors=raw['errors'],worker_states=v['worker_states']);return
 execute(f['exploration_slots'])
 cs,_=summarize_groups(raws,f,'C');qualified=[r for r in cs if r['qualifies']]
 save('exploratory_result.json',dict(groups=cs,development_only=True,fatal=fatal))
 if qualified and not fatal:
  winner=max(qualified,key=lambda r:(r['delta_to_strongest'],r['profile']=='steady'));slots=[s for s in f['independent_slots'] if s['arrival_profile']==winner['profile']]
  save('frozen_independent_manifest.json',dict(status='FROZEN_BEFORE_INDEPENDENT_GPU',at_unix_ns=time.time_ns(),candidate=winner,slots=slots,router_source=ref(OUT/'dynamic_router.py'),parameters=f['parameters'],utility_definition=f['utility_definition'],utility_reference=f['utility_reference'],original_manifest=ref(OUT/'frozen_candidate_manifest.json'),selection_rule=f['selection'],acceptance=f['independent_protocol'],all_models_residency_replicas_QPS_unchanged=True))
  print('EXPLORATORY_POSITIVE',winner['profile'],winner['delta_to_strongest'],flush=True);execute(slots)
 else:save('independent_not_executed.json',dict(reason='NO_QUALIFIED_POSITIVE_EXPLORATORY_CANDIDATE' if not fatal else 'FATAL_STOP',no_seed_or_threshold_change=True))
 gate=finalize(raws,f,fatal);save('pipeline_completed.json',dict(at_unix_ns=time.time_ns(),gate=gate['gate'],executed=[r['slot']['run_id'] for r in raws],fatal=fatal,completed_and_stopped=True));print('FINAL_GATE',gate['gate'],flush=True)
if __name__=='__main__':main()
