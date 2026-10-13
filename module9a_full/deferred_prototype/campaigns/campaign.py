"""Independent protocol-identical process blocks. Frozen list; no top-up pairs."""
import os,subprocess,time
from .common import *

def main():
 assert not (OUT/'measurement_execution_freeze.json').exists()
 p=load(OUT/'protocol_identity.json');manifest=load(OUT/'pair_validation_manifest.json');sources=[{'path':str(f.resolve()),'sha256':sha(f)} for directory in [ROOT/'module9a_full',ROOT/'deployment_core/interference'] for f in sorted(directory.iterdir()) if f.suffix in ['.py','.cpp','.so']]
 core=[{'path':str((OUT/n).resolve()),'sha256':sha(OUT/n)} for n in ['protocol_identity.json','pair_validation_manifest.json','pair_split_manifest.json','scheduler_policy_audit.json','gate_rule_freeze.json','catalog.json','pair_predictions.csv','profile_protocol.json','roomie_formula_audit.md','prior_art_boundary.md','routing_conditioned_model.md']]
 stages={'STANDALONE':[k for k,s in p['states'].items() if not s['co_model']], 'CALIBRATION':[k for k,s in p['states'].items() if s.get('split')=='CALIBRATION'],'VALIDATION':[k for k,s in p['states'].items() if s.get('split')=='VALIDATION']};plan=[]
 for stage,keys in stages.items():
  keys=sorted(keys,key=lambda k:digest({'order':'9AF_EXECUTE','key':k}))
  for rep in range(1,4):
   order=keys[(rep-1)%len(keys):]+keys[:(rep-1)%len(keys)]
   if rep%2==0:order=order[::-1]
   plan.extend({'stage':stage,'state':k,'rep':rep} for k in order)
 freeze={'frozen_at_utc':now(),'sources':sources,'core':core,'plan':plan,'new_validation_runs_before_freeze':0,'retries':0,'no_result_based_extra_pairs':True};save('measurement_execution_freeze.json',freeze);save('run_accounting.json',{'planned_timed_runs':len(plan),'completed_timed_runs':0,'new_attempts':0,'retries':0,'profile_prelaunch_failures':2,'completed_profiles':len(load(OUT/'catalog.json')['paths'])})
 completed=0
 try:
  for i,r in enumerate(plan):
   verify(sources+core)
   if r['stage']=='VALIDATION' and not (OUT/'I1_calibration_freeze.json').exists():
    from .analysis import freeze_baseline
    freeze_baseline()
   account=load(OUT/'run_accounting.json');account['new_attempts']+=1;save('run_accounting.json',account);save('campaign_state.json',{'phase':r['stage'],'plan_index':i,'current':r,'completed_timed_runs':completed,'planned_timed_runs':len(plan)})
   cmd=[str(PYTHON),'-u','-m','module9a_full.pair_runtime','--state',r['state'],'--rep',str(r['rep'])];env=dict(os.environ,**p['environment_variables']);log=OUT/'run_logs'/f'{i+1:03d}.log';log.parent.mkdir(exist_ok=True)
   print(json.dumps({'event':'START','index':i,**r}),flush=True)
   with log.open('w') as stream:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT);save('active_process.json',{'pid':proc.pid,'command':cmd,'log':str(log)});code=proc.wait()
   assert code==0,f'failed measurement preserved; no retry: {log}'
   completed+=1;account['completed_timed_runs']=completed;save('run_accounting.json',account);print(json.dumps({'event':'COMPLETE','index':i,'completed':completed}),flush=True)
  verify(sources+core);save('campaign_state.json',{'phase':'GPU_COMPLETE_ANALYSIS_PENDING','completed_timed_runs':completed})
 except BaseException as e:save('campaign_state.json',{'phase':'INCOMPLETE','error':repr(e),'completed_timed_runs':completed});raise
if __name__=='__main__':main()
