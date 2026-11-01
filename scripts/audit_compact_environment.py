"""CPU ABI/import check in one isolated interpreter, without checkpoints or GPU."""
import argparse
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--environment',required=True)
p.add_argument('--uv',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
import torch

def forbidden(*args,**kwargs):
    raise RuntimeError('GPU/checkpoint access forbidden in dependency audit')

torch.cuda._lazy_init=forbidden
torch.cuda.init=forbidden
torch.cuda.get_device_capability=forbidden
torch.cuda.device_count=lambda:0
torch.cuda.is_available=lambda:False
torch.load=forbidden
import safetensors
safetensors.safe_open=forbidden
modules=['transformers','accelerate','bitsandbytes','peft','PIL','soundfile','librosa','sentencepiece','pyarrow','pynvml','rapidfuzz']
if a.environment in {'common','deepseek'}:modules+=['flash_attn','flash_attn_2_cuda']
if a.environment=='qwen':modules+=['transformers.models.qwen2_5_omni.processing_qwen2_5_omni','transformers.models.qwen2_5_omni.modeling_qwen2_5_omni']
if a.environment=='minicpm':modules+=['datasets','vocos','vector_quantize_pytorch','auto_gptq','autogptq_cuda_64','autogptq_cuda_256','exllama_kernels']
errors={}
for name in modules:
 try:importlib.import_module(name)
 except Exception as e:errors[name]=type(e).__name__+': '+str(e)
check=subprocess.run([a.uv,'pip','check','--python',sys.executable],capture_output=True,text=True)
if check.returncode:errors['dependency_check']=check.stdout+check.stderr
result={'status':'PASS_CPU_IMPORTS_AND_DEPENDENCIES' if not errors else 'FAIL_CPU_DEPENDENCIES',
 'environment':a.environment,'python':sys.executable,'python_version':sys.version,'cuda_initialized':torch.cuda.is_initialized(),
 'weights_loaded':0,'gpu_started':False,'modules':modules,'errors':errors,
 'torch_cuda_build':torch.version.cuda,'cxx11_abi':torch._C._GLIBCXX_USE_CXX11_ABI,
 'dependency_check':check.stdout+check.stderr,
 'installed_packages':dict(sorted((d.metadata['Name'].lower().replace('_','-'),d.version) for d in metadata.distributions()))}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','environment','errors','cuda_initialized']}))
if errors:raise SystemExit(1)
