"""Finish reporting after the already frozen nine observations; then stop."""
import os,time,subprocess
from .common import *
def main():
 while True:
  p=OUT/'gpu_campaign.log'
  if p.exists() and 'FROZEN_NINE_SESSIONS_COMPLETE' in p.read_text():break
  time.sleep(10)
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
 r=subprocess.run(['/root/miniconda3/envs/gsdeploy/bin/python','-m','module15.finalize'],cwd=ROOT,env=env);assert r.returncode==0
 print('MODULE15_REQUESTED_SCOPE_COMPLETE_AND_STOPPED',now(),flush=True)
if __name__=='__main__':main()
