"""Finish one documented pre-inference repair; no GPU restart on loss."""
from .common import *
from .pipeline import alive
import subprocess

def main():
 amendment=read(OUT/'hardware_identity_amendment.json');launch=read(OUT/'hardware_recovery_controller_launch.json')
 while not (OUT/'campaign.json').exists():
  if not alive(launch['pid']):
   ledger=read(OUT/'attempt_ledger.json');raws=[]
   for r in ledger:
    path=OUT/'raw_runs'/(r['run_id']+'.json')
    if path.exists():raws.append(read(path))
   fatal=dict(reason='RECOVERY_CONTROLLER_DISAPPEARED_NO_SECOND_REPAIR',unfinished_U=None,request_journals_retained=True)
   save('infrastructure_failure.json',dict(at_utc=now(),fatal=fatal,ledger=ledger))
   from .analyze import stage_b,stage_c
   f=read(OUT/'discovery_protocol.json');bb=[r['slot']['run_id'] for r in raws if r['slot']['stage']=='B'];cc=[r['slot']['run_id'] for r in raws if r['slot']['stage']=='C']
   if not (OUT/'exploratory_result.json').exists():stage_b(bb,f,fatal)
   if read(OUT/'frozen_positive_candidate.json').get('slots') and not (OUT/'independent_result.json').exists():stage_c(cc,f)
   save('campaign.json',dict(at_utc=now(),executed_B=bb,executed_C=cc,fatal=fatal,queue_stop=None,
    total_GPU_attempts=len(ledger)-1,B_attempts=sum(r['stage']=='B' for r in ledger)-1,C_attempts=sum(r['stage']=='C' for r in ledger),
    B_budget_checks_used=sum(r['stage']=='B' for r in ledger),initial_pre_inference_identity_check_count=1,hardware_identity_amendment=ref(OUT/'hardware_identity_amendment.json'),
    no_budget_expansion=True,no_outcome_retries=True));break
  time.sleep(10.)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 def cpu(module,log):
  with (OUT/log).open('w') as stream:r=subprocess.run(['/root/miniconda3/envs/gsdeploy/bin/python','-u','-m',module],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
  assert r.returncode==0,(module,r.returncode)
 c=read(OUT/'independent_result.json') if (OUT/'independent_result.json').exists() else None
 if c and c['Stage_D_permitted']:cpu('module27.autonomy','optimizer.log')
 else:save('optimizer_not_executed.json',dict(reason='Independent positive gate not passed',Stage_D=False))
 verify(amendment['implementation']);cpu('module27.finalize','finalize.log')
 gate=read(OUT/'module27_gate.json');campaign=read(OUT/'campaign.json')
 gate.update(initial_pre_inference_identity_failure_preserved=True,initial_identity_check_not_GPU_measurement=True,
  B_budget_checks_used=campaign['B_budget_checks_used'],remaining_exploration_cap_after_check=5,
  historical_GPU_UUID_difference_explicit=True,hardware_identity_amendment=ref(OUT/'hardware_identity_amendment.json'))
 save('module27_gate.json',gate)
 with (OUT/'module27_summary.md').open('a') as stream:
  stream.write('\n设备与运行会计：第一次预启动仅做NVML身份核验，发现Module18历史GPU UUID与当前Module26 GPU不同；模型未启动，GPU推理/测量数为0，U不存在。该失败的全部原始输出及首次gate单独原样归档于initial_prelaunch_failure/。\n')
  stream.write('当前设备身份在任何模型推理前另行冻结于hardware_identity_amendment.json；模型/QPS/预算/效用/输入/种子不变。为了不增加原6次预算，这次失败占1个检查名额，最多5次实际探索；预先取消11.5QPS的20%格点，保留0/10%，不是按GPU收益事后筛除。\n')
  stream.write('历史Module18/19不同GPU的服务时间只是筛选先验，不是当前GPU容量认证；同机当前GPU配对才是效用证据。GPU型号、总显存及driver相同仍不能证明不同UUID性能等价。\n')
 cpu('module27.verify','final_verification.log');verify(amendment['implementation'])
 save('pipeline_completed.json',dict(at_utc=now(),status=gate['status'],verification_passed=True,no_next_Module=True,identity_failure_preserved=True))

if __name__=='__main__':main()
