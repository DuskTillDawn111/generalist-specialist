from module25.common import *
import os
from datetime import datetime,timezone
from module26.common import cohort
OUT=ROOT/'results/module27_positive_split_discovery'
MEDIUM='openai/whisper-medium@abdf7c39ab9d0397620ccaea8974cc764cd0953e'
MODELS=[G,MEDIUM,TEXT,IMAGE,OCR]
TARGET='lid'
def now():return datetime.now(timezone.utc).isoformat()
def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=json_native)+'\n');tmp.replace(p)
def csvsave(name,rows,fields=()):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys([*fields,*(k for r in rows for k in r)])));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True,default=json_native) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def protected_metadata():
 roots=[ROOT/'results',*[p for p in ROOT.glob('module*') if p.is_dir() and p.name!='module27']]
 return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for root in roots for p in root.rglob('*') if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts}
def inputs():
 definition,refs=utility_inputs();return read(OUT/'selected_pool_catalog.json'),refs,definition['weights']
def observe(raw,path):
 from module24.calibration import extract
 d,refs,weights=inputs()
 return extract(raw,27,path,refs,weights,d=d,split='EXPLORATION' if raw['slot']['stage']=='B' else 'INDEPENDENT_NEVER_TRAIN')
def arithmetic(raw):
 from module26.common import arithmetic as original
 d,refs,weights=inputs();clean={k:v for k,v in raw.items() if k!='_path'};value=original(clean,d,refs,weights)
 ob=observe(raw,OUT/'raw_runs'/(raw['slot']['run_id']+'.json'))
 value.update(U=value.get('raw_arithmetic_U') if ob['eligible'] else None,eligible=ob['eligible'])
 if ob['eligible']:assert abs(value['U']-ob['U'])<1e-12
 return value,ob
def routing(rates,fraction):
 return {'text':{TEXT:rates['text']},'image':{IMAGE:rates['image']},'ocr':{OCR:rates['ocr']},
         'asr':{MEDIUM:rates['asr']},'visual_qa':{G:rates['visual_qa']},
         'lid':{m:q for m,q in [(MEDIUM,rates['lid']*(1-fraction)),(G,rates['lid']*fraction)] if q>0}}
