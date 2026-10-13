"""CPU-only native dynamic-module cache resolution under write confinement."""
import sys,json
from pathlib import Path
from m32e.isolation import install,VIOLATIONS,ROOT
install()
from transformers.dynamic_module_utils import get_cached_module_file
import torch
files={}
for name in ['configuration_phi4mm.py','processing_phi4mm.py','modeling_phi4mm.py']:
 files[name]=get_cached_module_file('/root/autodl-tmp/generalist-specialist/models/phi4-multimodal',name,local_files_only=True)
result=dict(passed=not VIOLATIONS and not torch.cuda.is_initialized(),modules=files,unexpected_writes=VIOLATIONS,CUDA_initialized=torch.cuda.is_initialized())
(ROOT/'isolated_cache_CPU_acceptance.json').write_text(json.dumps(result)+'\n')
assert result['passed'],result
print('ISOLATED_CACHE_CPU_PASS',flush=True)
