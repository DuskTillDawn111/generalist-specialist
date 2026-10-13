"""Continue exactly through x3 closure/replay/proposals, without new research execution."""
import os,subprocess,time
from .common import *
PYTHON='/root/miniconda3/envs/gsdeploy/bin/python'
def invoke(name,cpu=False,log=None):
 env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
 if cpu:env['CUDA_VISIBLE_DEVICES']=''
 print('STAGE_START',name,now(),flush=True)
 if log:
  with Path(log).open('a') as f:r=subprocess.run([PYTHON,'-m',name],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 else:r=subprocess.run([PYTHON,'-m',name],cwd=ROOT,env=env)
 assert r.returncode==0,(name,r.returncode);print('STAGE_COMPLETE',name,now(),flush=True)
def main():
 slots=read(OUT/'profiling_freeze.json')['slots']
 while not all((OUT/'raw_runs'/(s['run_id']+'.json')).exists() for s in slots):time.sleep(10)
 invoke('module14f.analyze',cpu=True);invoke('module14f.replay',cpu=True,log=OUT/'replay.log');invoke('module14f.finalize',cpu=True)
 print('MODULE14F_REQUESTED_SCOPE_COMPLETE',now(),flush=True)
if __name__=='__main__':main()
