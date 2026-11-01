"""Run repository tests with explicit GPU/checkpoint guards and save a receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True)
a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch

def forbidden(*args,**kwargs):
    raise AssertionError('GPU initialization or checkpoint reads forbidden in CPU regression tests')

torch.cuda._lazy_init=forbidden
torch.cuda.init=forbidden
torch.cuda.get_device_capability=forbidden
torch.cuda.is_available=lambda:False
torch.cuda.device_count=lambda:0
torch.Tensor.cuda=forbidden
torch.load=forbidden
import safetensors
safetensors.safe_open=forbidden
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
suite=unittest.defaultTestLoader.discover(str(root/'tests'),top_level_dir=str(root))
result=unittest.TextTestRunner(verbosity=2).run(suite)
receipt={'status':'PASS' if result.wasSuccessful() and not result.skipped else 'FAIL_OR_SKIPPED',
         'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
         'skipped':len(result.skipped),'gpu_started':False,'cuda_initialized':torch.cuda.is_initialized(),
         'models_loaded':0,
         'source_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
             for pattern in ['gsdeploy/**/*.py','tests/**/*.py','configs/compact*.json','configs/study*.json',
                             'scripts/run_compact_cpu_tests.py','scripts/run_compact_gpu_queue.py']
             for p in root.glob(pattern)}}
Path(a.output).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
if receipt['status']!='PASS':raise SystemExit(1)
