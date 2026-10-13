"""Profiling-only launch adapter; native inference handlers are unchanged."""
import os,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'joint_deploy'),str(ROOT/'benchmark/scripts')]
import worker_protocol
original=worker_protocol.run_worker

def instrumented(worker_name,handler,ready_payload=None):
 import torch
 def wrapped(request):
  if request['task_type']=='__pilot_profile_stop__':
   torch.cuda.synchronize();torch.cuda.profiler.stop();return {'profile_stopped':True}
  label='PILOT|'+request['task_type']+'|'+str(request['request_id'])
  torch.cuda.nvtx.range_push(label)
  try:return handler(request)
  finally:torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
 return original(worker_name,wrapped,ready_payload)
def start_capture():
 import torch
 torch.cuda.synchronize();torch.cuda.profiler.start()
 return {'profile_control':True,'profile_started':True,'backend':'active native torch CUDA runtime'}
worker_protocol._start_cuda_profiler=start_capture
worker_protocol.run_worker=instrumented
os.environ['SPECIALIST_NSYS_CAPTURE']='1';os.environ['QWEN_NSYS_CAPTURE']='1'
runpy.run_path(os.environ['PILOT_NATIVE_WORKER'],run_name='__main__')
