import os,subprocess,sys
from .common import *
def main():
 assert load(OUT/'campaign_state.json')['status']=='PREPARED','No restart or automatic retry'
 p=load(OUT/'protocol_identity.json');keys=list(p['states']);plan=[]
 for rep in range(1,6):
  offset=2*(rep-1)%len(keys);order=keys[offset:]+keys[:offset]
  if rep%2==0:order=order[::-1]
  plan.extend({'state':key,'rep':rep} for key in order)
 sources=[{'path':str(f.resolve()),'sha256':sha(f)} for f in sorted(Path(__file__).parent.glob('*.py'))];core=[{'path':str((OUT/n).resolve()),'sha256':sha(OUT/n)} for n in ['protocol_identity.json','anchor_selection.json']]
 freeze={'frozen_at_utc':now(),'execution_sources':sources,'core_files':core,'plan':plan,'GPU_runs_before_freeze':0,'planned_timed_runs':len(plan),'retries':0,'old_gate_immutable':True};save('execution_freeze.json',freeze);account={'planned_timed_runs':len(plan),'completed_timed_runs':0,'new_attempts':0,'retries':0,'reused_GPU_measurements':0,'process_resets_per_run':True};save('run_accounting.json',account)
 try:
  for i,row in enumerate(plan):
   verify(sources+core);account['new_attempts']+=1;save('run_accounting.json',account);save('campaign_state.json',{'status':'RUNNING','plan_index':i,'current':row,'completed_runs':account['completed_timed_runs'],'new_GPU_runs':account['new_attempts']})
   env=dict(os.environ,**p['environment_variables'],PYTHONUNBUFFERED='1')
   if p['states'][row['state']]['protocol_family']=='historical_full_state_replay':env.update(p['historical_environment_overrides'])
   cmd=[sys.executable,'-u','-m','module9a2_repeatability.runtime','--state',row['state'],'--rep',str(row['rep'])];log=OUT/'run_logs'/f'{i+1:03d}.log';log.parent.mkdir(exist_ok=True)
   print(json.dumps({'event':'START','index':i,**row}),flush=True)
   with log.open('w') as stream:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT);save('active_process.json',{'pid':proc.pid,'log':str(log),'command':cmd});code=proc.wait()
   assert code==0,f'Runtime failure; preserve evidence and stop without retry: {log}'
   account['completed_timed_runs']+=1;save('run_accounting.json',account);print(json.dumps({'event':'COMPLETE','index':i,'completed':account['completed_timed_runs']}),flush=True)
  save('campaign_state.json',{'status':'ANALYZING','completed_runs':account['completed_timed_runs'],'new_GPU_runs':account['new_attempts']});verify(sources+core);verify(p['frozen_refs'])
  from .analyze import final
  final()
  from candidate_pool_audit.provenance import environment_snapshot
  assert environment_snapshot()==p['runtime_environment'],'Post-run environment drift'
  assert sha(PILOT/'pilot_gate.json')==p['source_9A1_gate_hash'];assert load(PILOT/'pilot_gate.json')['status']=='GATE_PARTIAL'
  save('validation_results.json',{'status':'PASS','all_old_artifacts_and_sources_unchanged':True,'frozen_reference_count':len(p['frozen_refs']),'source_9A1_gate':'GATE_PARTIAL','all_50_process_level_runs_complete':True,'all_exact_input_ID_and_warmup_count_checks_passed':True,'no_method_routing_ranking_workload_budget_revision':True,'full_Module9A_or_heldout_validation_started':False})
  save('campaign_state.json',{'status':'COMPLETE','completed_runs':account['completed_timed_runs'],'new_GPU_runs':account['new_attempts'],'completed_at_utc':now()});save('artifact_hashes.json',{str(f.relative_to(OUT)):{'path':str(f.resolve()),'sha256':sha(f)} for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['artifact_hashes.json','execution.log']})
 except BaseException as e:save('campaign_state.json',{'status':'INCOMPLETE','error':f'{type(e).__name__}: {e}','completed_runs':account['completed_timed_runs'],'new_GPU_runs':account['new_attempts'],'old_gate_unchanged':True});raise
 print(json.dumps({'event':'COMPLETE','gate':load(OUT/'repeatability_gate.json')['status'],'accounting':account}),flush=True)
if __name__=='__main__':main()
