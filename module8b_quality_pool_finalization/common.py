import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/module8b_quality_pool_finalization'
POOL_OUT=ROOT/'results/candidate_pool_v2_1'
QUALITY=ROOT/'results/quality_workload_audit'
SERVICE_OLD=ROOT/'results/candidate_pool_v2_service_demand/updated_service_demand_catalog.json'
TASK_POOL=ROOT/'benchmark/task_pools/mixed_formal_1200_6task.jsonl'
TASKS=['text','image','ocr','asr','visual_qa','lid']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def write(p,d,freeze=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 text=json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
 if freeze and p.exists() and p.read_text()!=text:raise RuntimeError('Refuse frozen overwrite '+str(p))
 p.write_text(text)
def registry():
 from candidate_pool_audit.registry import registry as v2_registry
 result=v2_registry()
 for name in ('phi4mm','minicpm'):
  result[name]=dict(result[name],script=ROOT/f'candidate_pool_v2_1_adapters/worker_{name}.py')
 return result
