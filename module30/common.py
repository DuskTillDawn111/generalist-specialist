import os,json,csv,hashlib,tempfile
from pathlib import Path
from module25.common import json_native
ROOT=Path('/root/autodl-tmp/generalist-specialist')
OUT=ROOT/'results/module30_positive_split_feasibility'
def read(p):return json.loads(Path(p).read_text())
def ref(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)
def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix=p.name+'.',suffix='.tmp',dir=p.parent)
 try:
  with os.fdopen(fd,'w') as f:
   json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False,default=json_native);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)
  d=os.open(p.parent,os.O_DIRECTORY);os.fsync(d);os.close(d)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def csvsave(name,rows,fields=()):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys([*fields,*(k for r in rows for k in r)])));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,default=json_native) if isinstance(v,(dict,list)) else v for k,v in r.items()})
