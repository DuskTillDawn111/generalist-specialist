"""Fresh exec process, CUDA init + synchronized tensor arithmetic; no model."""
import sys,json,os,time,traceback
from pathlib import Path
from m32f.isolation import install,VIOLATIONS,ROOT
layer=sys.argv[1] if len(sys.argv)>1 else 'full'
install(layer)
result=dict(layer=layer,pid=os.getpid(),parent_pid=os.getppid(),before_cuda={},environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','HF_HOME','TMPDIR','CUDA_CACHE_PATH']})
try:
 import torch
 result.update(torch_version=torch.__version__,CUDA_runtime=torch.version.cuda)
 result['before_cuda']=dict(initialized=torch.cuda.is_initialized(),bad_fork=torch.cuda._is_in_bad_fork())
 print('MIN_CUDA_INIT_BEGIN',flush=True)
 tick=time.perf_counter_ns();torch.cuda.init();result['initialize_ms']=(time.perf_counter_ns()-tick)/1e6
 result['device']=dict(name=torch.cuda.get_device_name(0),count=torch.cuda.device_count(),properties=str(torch.cuda.get_device_properties(0)))
 a=torch.arange(1024,device='cuda:0',dtype=torch.float32);value=float((a*a).sum().cpu());torch.cuda.synchronize()
 assert value==sum(x*x for x in range(1024)),value
 result.update(passed=True,initialized=torch.cuda.is_initialized(),synchronized_tensor_sum=value,allocated_bytes=torch.cuda.memory_allocated())
except BaseException as exc:result.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
result['unexpected_python_writes']=list(VIOLATIONS)
result['after_cuda']=dict(initialized=torch.cuda.is_initialized(),bad_fork=torch.cuda._is_in_bad_fork()) if 'torch' in locals() else {}
p=ROOT/'minimal_cuda'/f'{layer}_{os.getpid()}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
sys.exit(0 if result.get('passed') else 1)
