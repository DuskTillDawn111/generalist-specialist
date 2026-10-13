import csv,json,hashlib,math
from pathlib import Path
from datetime import datetime,timezone
from module8c_final_real_search.common import inputs
from deployment_search.runtime_cache import file_hash,digest,save_json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/module9bc_e2e_routing_ranking'
HISTORY=ROOT/'results/module8f2_resource_static'
PROTOCOL=ROOT/'results/module8e2_heldout_freeze/method_comparison_freeze.json'
def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def save(name,data):OUT.mkdir(parents=True,exist_ok=True);save_json(OUT/name,data)
def ref(path):return {'path':str(Path(path).resolve()),'sha256':file_hash(path)}
def csvsave(name,rows,columns=None):
    columns=columns or list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def flat(route):
    if not route:return {}
    return {tuple(k.split('::')) if isinstance(k,str) else k:v for k,v in route.items()}
def serial(route):return {'::'.join(k):v for k,v in route.items()}
def statekey(cell,x,route):return digest({'cell':cell,'x':x,'r':{k:round(v,10) for k,v in serial(route).items() if v>1e-10}})
def cells():return {c['cell_id']:c for c in read(ROOT/'results/module8f1b_intensity_grid_freeze/campaign_cells.json')['cells']}
def verify(refs):
    for r in refs:
        if file_hash(r['path'])!=r['sha256']:raise RuntimeError('Frozen file changed: '+r['path'])
TIER={'unstable':0,'recoverable_congested':1,'steady':2}
def labels(runtime):
    return {'mean_e2e_ms':runtime['weighted_mean_e2e'],'weighted_p95_ms':runtime.get('weighted_p95_e2e'),
            'stability':runtime['stability_state_v2']}
