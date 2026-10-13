import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m34.isolation import install,ROOT
install()
from transformers.dynamic_module_utils import get_cached_module_file
model=sys.argv[1];files={}
for f in Path('/root/autodl-tmp/generalist-specialist/models',model).glob('*.py'):
 if model=='phi4-multimodal' and f.name not in ['configuration_phi4mm.py','processing_phi4mm.py','modeling_phi4mm.py']:continue
 files[f.name]=get_cached_module_file(str(f.parent),f.name,local_files_only=True)
(ROOT/('cache_CPU_'+model+'.json')).write_text(json.dumps(dict(passed=True,model=model,files=files,no_GPU=True)))
print('CACHE_ACCEPTANCE',model,len(files))
