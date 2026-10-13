from module20.common import ROOT,read,ref,now
import json,csv,math,time
OUT=ROOT/'results/module22_mid_high_utility'
def save(n,d):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvsave(n,rows,fields=()):
 with (OUT/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys([*fields,*(k for r in rows for k in r)])));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
MODELS=['qwen','roberta_sst2','convnext_tiny','baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt','facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a']
G=MODELS[0];MMS=MODELS[-1]
BUDGET=32019.2544
BASE={'text':4.,'image':2.,'ocr':2.,'asr':.05,'visual_qa':.2}
RATES={st:{**BASE,'lid':q} for st,q in [('MID',3.),('HIGH',12.)]}
