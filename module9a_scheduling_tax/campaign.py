"""One bounded sequence, stops on any failure; no outcome-based additions."""
import os, subprocess, time
from .common import *

def main():
 protocol=load(OUT/'protocol_identity.json');split=load(OUT/'pair_split_manifest.json')
 env={**os.environ,**protocol['environment_variables']}
 env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_LOG_DIRECTORY',None)
 assert not any(k.startswith('CUDA_MPS') for k in env), 'Unexpected MPS environment'
 assert subprocess.run(['pgrep','-f','^nvidia-cuda-mps'],capture_output=True).returncode!=0
 phases=[('STANDALONE',[k for k,s in protocol['states'].items() if s['co_model'] is None]),
         ('CALIBRATION',[e['id'] for e in split['entries'] if e['split']=='CALIBRATION']),
         ('VALIDATION',[e['id'] for e in split['entries'] if e['split']=='VALIDATION'])]
 completed=0
 # Interleave identities within independent-process repetitions, fixed hash order.
 for phase,keys in phases:
  for rep in [1,2,3]:
   for key in sorted(keys,key=lambda k:digest({'key':k,'rep':rep,'phase':phase})):
    output=OUT/'raw_runs'/f'{key}__rep{rep}.json'
    if output.exists():
     assert load(output)['status']=='COMPLETE','No retries of failed runs'
     completed+=1;continue
    state={'phase':phase,'active_state':key,'active_repetition':rep,'new_completed_runs':completed,'planned_runs':protocol['max_new_runs'],'updated_at_utc':now()}
    save('campaign_state.json',state);print(json.dumps(state),flush=True)
    log=OUT/'run_logs'/f'{key}__rep{rep}.log';log.parent.mkdir(exist_ok=True)
    with log.open('w') as f:
     result=subprocess.run([str(PYTHON),'-m','module9a_scheduling_tax.pair_runtime','--state',key,'--rep',str(rep)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
     save('campaign_state.json',{**state,'phase':'FAILED_NO_RETRY','log':str(log),'returncode':result.returncode});raise RuntimeError(str(log))
    completed+=1
  if phase=='CALIBRATION':
   from .analysis import freeze_fit
   freeze_fit()
 save('campaign_state.json',{'phase':'MEASUREMENTS_COMPLETE','new_completed_runs':completed,'planned_runs':protocol['max_new_runs'],'updated_at_utc':now()})
 from .analysis import closeout
 closeout()

if __name__=='__main__':main()
