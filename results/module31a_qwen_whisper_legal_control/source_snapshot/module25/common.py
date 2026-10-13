from pathlib import Path
import json,csv,hashlib,time,copy,itertools,math
from collections import defaultdict,Counter
import numpy as np
csv.field_size_limit(128*1024*1024)
from module24.common import ref,identity,G,S,TEXT,IMAGE,OCR,BUDGET,RATES,SOURCE,ROOT

OUT=ROOT/'results/module25_utility_difference'
DIRECTORIES={19:'module19_strategy_benchmark',20:'module20_utility_optimizer',21:'module21_positive_utility',
    '21D':'module21_low_diagnosis',22:'module22_mid_high_utility',23:'module23_positive_three_phase',
    '24A':'module24_utility_calibration','24B':'module24b_high_validation'}

def read(path):return json.loads(Path(path).read_text())
def json_native(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)
def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=json_native)+'\n')
def csvsave(name,rows,fields=()):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys([*fields,*(k for r in rows for k in r)])))
        writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def verify(refs):
    for r in refs:assert ref(r['path'])['sha256']==r['sha256'],r['path']
def protected_metadata():
    roots=[ROOT/'results',*[p for p in ROOT.glob('module*') if p.is_dir() and p.name!='module25']]
    return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for root in roots for p in root.rglob('*')
        if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts}
def deployment_key(x):return identity(sorted((m,n) for m,n in x.items() if n))
def checkpoint_key(x):return identity(sorted(m for m,n in x.items() if n))
def group_key(row):return identity([row['deployment'],row['rates'],row.get('arrival_process'),row.get('hardware_uuid')])
def utility_inputs():
    definition=read(ROOT/'results/module20_utility_optimizer/utility_definition.json')
    refs=read(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json')['latency_reference_ms']
    assert definition['weights']==dict(Q=0.,T=.25,L=.3,P=.3,M=.15)
    return definition,refs
def stats(values):
    return dict(count=len(values),mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95))) if len(values) else None
