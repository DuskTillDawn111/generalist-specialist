"""Honor automatic continuation; downstream modules have separate directories."""
import os,subprocess,time
from .common import *
PYTHON='/root/miniconda3/envs/gsdeploy/bin/python'
def invoke(module,cpu=False,log=None):
 env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
 if cpu:env['CUDA_VISIBLE_DEVICES']=''
 print('STAGE_START',module,now(),flush=True)
 if log:
  with Path(log).open('a') as stream:r=subprocess.run([PYTHON,'-m',module],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
 else:r=subprocess.run([PYTHON,'-m',module],cwd=ROOT,env=env)
 assert r.returncode==0,(module,r.returncode)
 print('STAGE_COMPLETE',module,now(),flush=True)
def main():
 expected=read(OUT/'replica_measurement_manifest.json')['slots']
 while not all((OUT/'raw_runs'/(s['run_id']+'.json')).exists() for s in expected):time.sleep(10)
 invoke('module14d.analyze_measurements')
 invoke('module14d.replay',cpu=True,log=OUT/'replay.log')
 invoke('module14d.finalize',cpu=True)
 if read(OUT/'module14d_gate.json')['automatic_GPU_validation_required']:
  invoke('module14d.gpu_validation',log=OUT/'continuation_gpu.log')
  if read(ROOT/'results/module14e_heavy_gpu_validation/module14e_gate.json')['continue_stable_overflow_search']:
   invoke('module14d.overflow',cpu=True,log=OUT/'continuation_overflow.log')
 invoke('module14d.closeout',cpu=True)
 print('AUTHORIZED_PIPELINE_COMPLETE',now(),flush=True)
if __name__=='__main__':main()
