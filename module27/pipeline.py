"""Durable CPU supervision; never restarts lost GPU or selects more seeds."""
from .common import *
import subprocess

def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(')')[1].split()[0]!='Z'
 except FileNotFoundError:return False
def main():
 launch=read(OUT/'durable_controller_launch.json')
 while not (OUT/'campaign.json').exists():
  if not alive(launch['pid']):
   ledger=read(OUT/'attempt_ledger.json') if (OUT/'attempt_ledger.json').exists() else [];raws=[]
   for r in ledger:
    path=OUT/'raw_runs'/(r['run_id']+'.json')
    if path.exists():raws.append(read(path))
   fatal=dict(reason='DURABLE_CONTROLLER_DISAPPEARED_NO_GPU_RETRY',journal_retained=True,normal_U_for_unfinished_attempt=None)
   save('infrastructure_failure.json',dict(at_utc=now(),fatal=fatal,ledger=ledger))
   from .analyze import stage_b,stage_c
   f=read(OUT/'discovery_protocol.json');bb=[r['slot']['run_id'] for r in raws if r['slot']['stage']=='B'];cc=[r['slot']['run_id'] for r in raws if r['slot']['stage']=='C']
   if not (OUT/'exploratory_result.json').exists():stage_b(bb,f,fatal)
   if (OUT/'frozen_positive_candidate.json').exists() and read(OUT/'frozen_positive_candidate.json').get('slots') and not (OUT/'independent_result.json').exists():stage_c(cc,f)
   save('campaign.json',dict(at_utc=now(),executed_B=bb,executed_C=cc,fatal=fatal,queue_stop=None,total_GPU_attempts=len(ledger),
    B_attempts=sum(r['stage']=='B' for r in ledger),C_attempts=sum(r['stage']=='C' for r in ledger),no_budget_expansion=True,no_outcome_retries=True));break
  time.sleep(10.)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 def cpu(module,log):
  with (OUT/log).open('w') as stream:
   result=subprocess.run(['/root/miniconda3/envs/gsdeploy/bin/python','-u','-m',module],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
  assert result.returncode==0,(module,result.returncode)
 c=read(OUT/'independent_result.json') if (OUT/'independent_result.json').exists() else None
 if c and c['Stage_D_permitted']:cpu('module27.autonomy','optimizer.log')
 else:save('optimizer_not_executed.json',dict(reason='Independent positive split gate not satisfied',Stage_D=False))
 cpu('module27.finalize','finalize.log');cpu('module27.verify','final_verification.log')
 save('pipeline_completed.json',dict(at_utc=now(),status=read(OUT/'module27_gate.json')['status'],verification_passed=True,no_next_Module=True))

if __name__=='__main__':main()
