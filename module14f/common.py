"""New 14F artifacts only; all earlier modules remain immutable."""
import csv,hashlib,json,math,statistics
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/module14f_memory_ocr_x3'
D=ROOT/'results/module14_heavy_replica'
V=ROOT/'results/module14e_heavy_gpu_validation'
C=ROOT/'results/module14_heavy_structural'
E='EFFICIENCY_OPTIMAL_SPECIALIST_REGIME'
H='HEAVY_SPECIALIST_REGIME'
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def ref(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 s=p.stat();return {'path':str(p),'sha256':h.hexdigest(),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def verify(rs):
 for r in rs:assert ref(r['path'])['sha256']==r['sha256'],r['path']
def save(n,v):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvsave(n,rows):
 with (OUT/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,sort_keys=True,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
