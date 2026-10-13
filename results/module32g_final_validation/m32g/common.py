import json,csv,hashlib,time,sys,collections,math,copy,os,tempfile
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/generalist-specialist');OUT=ROOT/'results/module32g_final_validation';OLD=ROOT/'results/module32b_vqa_aware_routing';sys.path.insert(0,str(ROOT))
G='phi4mm#0';S='openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1#0'
def read(p):return json.loads(Path(p).read_text())
def native(x):
 if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [native(v) for v in x]
 if isinstance(x,np.generic):return x.item()
 return x
def save(n,v):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=p.name+'.',suffix='.tmp',dir=p.parent)
 try:
  with os.fdopen(fd,'w') as f:
   json.dump(native(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,p);d=os.open(p.parent,os.O_DIRECTORY);os.fsync(d);os.close(d)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def csvsave(n,rows,fields=None):
 rows=list(rows);fields=list(dict.fromkeys([*(fields or []),*(k for r in rows for k in r)]));p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(native(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else native(v) for k,v in r.items()})
def ref(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=p.stat().st_size)
def stats(a):
 a=list(a);return dict(n=len(a),mean=float(np.mean(a)) if a else None,p95=float(np.percentile(a,95)) if a else None,max=max(a) if a else None)
def interval(r):return (r['worker_response']['start_ns'],r['worker_response']['end_ns'])
def overlap(a,b):return max(0,min(a[1],b[1])-max(a[0],b[0]))/1e6
