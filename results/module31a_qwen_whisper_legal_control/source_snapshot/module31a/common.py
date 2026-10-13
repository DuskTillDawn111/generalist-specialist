from module30.common import ROOT,read,ref,json_native
from pathlib import Path
import os,json,csv,tempfile
OUT=ROOT/'results/module31a_qwen_whisper_legal_control'
OLD=ROOT/'results/module30_positive_split_feasibility'
def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix=p.name+'.',dir=p.parent)
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
def configure():
 from module30 import gpu,telemetry,report
 for m in [gpu,telemetry,report]:m.OUT=OUT;m.save=save;m.csvsave=csvsave
 return gpu,report
