import json,csv,math
from pathlib import Path
from datetime import datetime,timezone
from module8c_final_real_search.common import inputs
from deployment_search.runtime_cache import file_hash,save_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/module10_specialist_saturation'
def read(p):return json.loads(Path(p).read_text())
def save(name,data):OUT.mkdir(parents=True,exist_ok=True);save_json(OUT/name,data)
def ref(p):return {'path':str(Path(p).resolve()),'sha256':file_hash(p)}
def now():return datetime.now(timezone.utc).isoformat()
def csvsave(name,rows,columns=None):
    columns=columns or list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def factor():return read(ROOT/'results/module8f1a_memory_breakpoints/memory_safety_factor.json')['safety_factor']
def contexts():
    cells=read(ROOT/'results/module8f1b_intensity_grid_freeze/campaign_cells.json')['cells']
    budgets=sorted({(r['memory_budget_mib'],r['planning_budget_input_mib']) for r in cells})
    backgrounds={r['intensity_level']:r['lambda_rps'] for r in cells if r['mixture']=='mix_03' and r['intensity_level'] in ['I1','I2']}
    return [{'id':f'B{i+1}_{level}','mixture':'mix_03','background':level,'background_lambda':lam,'memory_budget_mib':b,'planning_budget_input_mib':raw}
            for i,(b,raw) in enumerate(budgets) for level,lam in sorted(backgrounds.items())]
def verify(refs):
    for r in refs:assert file_hash(r['path'])==r['sha256'],('Frozen reference changed',r['path'])
def protected():
    refs=read(ROOT/'results/module9a_scheduling_tax/protected_references.json')
    refs+=read(ROOT/'results/module9bc_e2e_routing_ranking/experiment_freeze.json')['protected_references']
    refs+=[ref(p) for d in ['module9bc','results/module9bc_e2e_routing_ranking'] for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix not in ['.pyc']]
    return list({r['path']:r for r in refs}.values())
