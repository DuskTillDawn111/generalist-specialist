import csv,datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/module14_high_quality_heavy';RAW=OUT/'raw';MODELS=ROOT/'models/module14'
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def ref(p):
 p=Path(p);return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2)+'\n')
def csvsave(name,rows,columns=None):
 columns=columns or list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
def key(e):return e['task']+'__'+e['model_id'].replace('/','--')
def audit(p,baseline):
 task=p['task'];b=baseline[task];q=p.get('quality');s=p.get('s0_ms');m=p.get('safe_memory_mib');valid=q is not None and s is not None and m is not None
 flags={'QUALITY_IMPROVED':(q<b['quality'] if task=='asr' else q>b['quality']) if valid else False,'LATENCY_INCREASED':s>b['s0_ms'] if valid else False,'MEMORY_INCREASED':m>b['safe_memory_mib'] if valid else False}
 failed=[x for x,k in [('QUALITY','QUALITY_IMPROVED'),('LATENCY','LATENCY_INCREASED'),('MEMORY','MEMORY_INCREASED')] if not flags[k]]
 return {**p,'baseline_model':b['model_id'],'baseline_quality':b['quality'],'baseline_s0_ms':b['s0_ms'],'baseline_safe_memory_mib':b['safe_memory_mib'],**flags,'HQ_HEAVY_ELIGIBLE':valid and not failed,'status':'HQ_HEAVY_ELIGIBLE' if valid and not failed else 'FAIL_'+failed[0] if len(failed)==1 else 'FAIL_MULTIPLE','failed_criteria':failed,'measurement_complete':valid,'quality_gain':b['quality']-q if task=='asr' and q is not None else q-b['quality'] if q is not None else None,'latency_ratio':s/b['s0_ms'] if s is not None else None,'safe_memory_ratio':m/b['safe_memory_mib'] if m is not None else None}
