import csv, hashlib, json, math, statistics
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/module14_heavy_replica'
OLD=ROOT/'results/module14_heavy_structural'
def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def save(name,data):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def ref(path):
 p=Path(path).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 s=p.stat();return {'path':str(p),'sha256':h.hexdigest(),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def verify(refs):
 for r in refs:assert ref(r['path'])['sha256']==r['sha256'],r['path']
def csvsave(name,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,sort_keys=True,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
