import subprocess,sys
from .common import *
def main():
 attempts=[]
 for path in read(POOL_OUT/'pending_service_demand_paths.json')['paths']:
  name=path['model_id']
  for rep in range(1,4):
   output=POOL_OUT/f'vqa_new_path_raw_measurements/r{rep}/active_single_{name}_visual_qa_n30.jsonl'
   validation=Path(str(output)+'.validation.json')
   if validation.exists():
    v=read(validation);attempts.append({'model':name,'repetition':rep,'reused':True,'status':v['status']})
    if v['status']!='measured':raise RuntimeError('Saved failed attempt remains pending; cannot continue')
    continue
   output.parent.mkdir(parents=True,exist_ok=True)
   command=[sys.executable,'-m','module8b_quality_pool_finalization.profile','--pool',str(TASK_POOL),
            '--target-worker',name,'--target-task','visual_qa','--target-count','30','--output',str(output),'--timeout-s','3600']
   print('Starting',name,'VQA standalone repetition',rep,flush=True)
   with Path(str(output)+'.profile.log').open('w') as log:
    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   v=read(validation) if validation.exists() else {'status':'MEASUREMENT_FAILED','GPU_memory_released':False}
   attempts.append({'model':name,'repetition':rep,'status':v['status'],'returncode':result.returncode,'command':command,'output':str(output)})
   write(POOL_OUT/'campaign_progress.json',{'attempts':attempts,'planned_attempts':6})
   if result.returncode or v['status']!='measured' or not v['GPU_memory_released']:
    raise RuntimeError('Measurement failure: leave pending; no Part D')
   print('Completed',name,'repetition',rep,flush=True)
 write(POOL_OUT/'campaign_progress.json',{'attempts':attempts,'planned_attempts':6,'complete':True})
if __name__=='__main__':main()
