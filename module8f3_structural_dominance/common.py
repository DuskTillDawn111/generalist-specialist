import csv, hashlib, json, math
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
OUT = R / 'module8f3_structural_dominance'
A = R / 'module8f1a_memory_breakpoints'
F = R / 'module8f1b_intensity_grid_freeze'
DEV = R / 'module8f2_resource_static'
TOL = 1e-9
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(n, obj):
    OUT.mkdir(exist_ok=True)
    (OUT/n).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def csvsave(n, rows):
    with (OUT/n).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
def now(): return datetime.now(timezone.utc).isoformat()
def route_dict(r): return {f'{k}::{m}':v for (k,m),v in r.routing.items() if v>0}
def inputs():
    from module8c_final_real_search.common import inputs as original
    catalog, _, roles, cal = original()
    return catalog, roles, cal
def counts(x,roles):
    ng=sum(x.get(m,0)>0 for m in roles if roles[m]=='generalist')
    ns=sum(x.get(m,0)>0 for m in roles if roles[m]=='specialist')
    regime='Specialist-heavy' if ns>ng else 'Hybrid' if ng and ns else 'Generalist-heavy' if ng else 'Specialist-only'
    return ng,ns,regime
def verify(refs):
    for r in refs:
        assert sha(r['path'])==r['sha256'],r['path']
