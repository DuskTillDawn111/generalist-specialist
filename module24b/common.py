"""Module24B owns its output; all earlier experiment files are read-only."""
from module24.common import *

HISTORY = ROOT/'results/module24_utility_calibration'
OUT = ROOT/'results/module24b_high_validation'

def save(name, value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def csvsave(name, rows, fields=()):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys([*fields,*(k for r in rows for k in r)])))
        writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def historical_metadata():
    roots=[ROOT/'results',*[p for p in ROOT.glob('module*') if p.is_dir() and p.name!='module24b']]
    return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for root in roots for p in root.rglob('*')
            if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts}

def model_inputs():
    from module24.calibration import ResidualModel
    d=read(HISTORY/'selected_pool_catalog.json')
    refs=read(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json')['latency_reference_ms']
    weights=read(ROOT/'results/module20_utility_optimizer/utility_definition.json')['weights']
    saved=read(HISTORY/'residual_model_final_development.json');model=ResidualModel()
    for key in ['exact','path','training_groups','fit_seconds']:setattr(model,key,saved[key])
    assert weights==dict(Q=0.,T=.25,L=.3,P=.3,M=.15)
    return d,refs,weights,model

def verify(references):
    for expected in references:
        assert ref(expected['path'])['sha256']==expected['sha256'],expected['path']
