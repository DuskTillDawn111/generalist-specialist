from module14f.common import ROOT,read,ref,verify,now
import json,csv,math,statistics
OUT=ROOT/'results/module16_fullpool_gpu'
def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvsave(name,rows,fields=()):
 with (OUT/name).open('w',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=list(dict.fromkeys([*fields,*(k for r in rows for k in r)])));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
