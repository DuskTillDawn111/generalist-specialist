"""Profiling-only wrapper. Native worker/runtime source and inference stay unchanged."""
import ctypes,os,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'joint_deploy'),str(ROOT/'benchmark/scripts')]
import worker_protocol
original=worker_protocol.run_worker

def read_driver_context():
 driver=ctypes.CDLL('libcuda.so.1');current=ctypes.c_void_p();rc=driver.cuCtxGetCurrent(ctypes.byref(current));result={'pid':os.getpid(),'cuCtxGetCurrent_rc':rc,'current_context_handle':hex(current.value) if current.value else None,'MPS_environment':{k:v for k,v in os.environ.items() if k.startswith('CUDA_MPS_')}}
 if rc==0 and current.value:
  if hasattr(driver,'cuCtxGetId'):
   ident=ctypes.c_ulonglong();code=driver.cuCtxGetId(current,ctypes.byref(ident));result.update(cuCtxGetId_rc=code,driver_context_id=ident.value if code==0 else None)
  priority=ctypes.c_int();code=driver.cuStreamGetPriority(ctypes.c_void_p(0),ctypes.byref(priority));result.update(default_stream_priority_rc=code,default_stream_priority=priority.value if code==0 else None)
 return result

# Paddle environment has no torch. Use its existing runtime and native NVTX.
paddle_worker=Path(os.environ['PILOT_NATIVE_WORKER']).name=='ocr_worker.py'
if paddle_worker:
 nvtx=ctypes.CDLL('/usr/local/cuda/lib64/libnvToolsExt.so.1');nvtx.nvtxRangePushA.argtypes=[ctypes.c_char_p]
 cudart=ctypes.CDLL('/root/miniconda3/envs/ocr_env/lib/python3.11/site-packages/nvidia/cu13/lib/libcudart.so.13')
 def sync():
  import paddle
  paddle.device.synchronize()
 def push(label):nvtx.nvtxRangePushA(label.encode())
 def pop():nvtx.nvtxRangePop()
 def control(start):
  sync();rc=cudart.cudaProfilerStart() if start else cudart.cudaProfilerStop()
  if rc!=0:raise RuntimeError(('CUDA profiler control',rc))
else:
 def sync():
  import torch
  torch.cuda.synchronize()
 def push(label):
  import torch
  torch.cuda.nvtx.range_push(label)
 def pop():
  import torch
  torch.cuda.nvtx.range_pop()
 def control(start):
  import torch
  sync()
  if start:torch.cuda.profiler.start()
  else:torch.cuda.profiler.stop()

def instrumented(worker_name,handler,ready_payload=None):
 def wrapped(request):
  if request['task_type']=='__pilot_profile_stop__':control(False);return {'profile_stopped':True}
  push('PILOT|'+request['task_type']+'|'+str(request['request_id']))
  try:return handler(request)
  finally:sync();pop()
 payload=dict(ready_payload or {});payload['runtime_sharing_semantics']=read_driver_context()
 return original(worker_name,wrapped,payload)
def start_capture():
 control(True);return {'profile_control':True,'profile_started':True,'backend':'existing native CUDA runtime','runtime_sharing_semantics':read_driver_context()}
worker_protocol._start_cuda_profiler=start_capture;worker_protocol.run_worker=instrumented
os.environ['SPECIALIST_NSYS_CAPTURE']='1';os.environ['QWEN_NSYS_CAPTURE']='1'
sys.path.insert(0,str(Path(os.environ['PILOT_NATIVE_WORKER']).resolve().parent))
runpy.run_path(os.environ['PILOT_NATIVE_WORKER'],run_name='__main__')
