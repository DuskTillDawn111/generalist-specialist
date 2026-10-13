import csv,json,math,statistics
from pathlib import Path
from module11.common import ROOT, inputs, read, ref, now, factor, verify
from deployment_search.runtime_cache import save_json,file_hash,digest

OUT=ROOT/'results/module11_replica_and_cost'
OLD=ROOT/'results/module11_multihotspot'
def save(name,data):
    OUT.mkdir(parents=True,exist_ok=True);save_json(OUT/name,data)
def csvsave(name,rows):
    OUT.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def stat(values):
    v=[float(x) for x in values if x is not None]
    return {'n':len(v),'mean':statistics.mean(v) if v else None,'median':statistics.median(v) if v else None,
            'min':min(v) if v else None,'max':max(v) if v else None,'sample_std':statistics.stdev(v) if len(v)>1 else None}
def fast_verify(refs):
    for r in refs:
        p=Path(r['path']);s=p.stat()
        if s.st_size!=r['bytes'] or s.st_mtime_ns!=r['mtime_ns']:
            assert file_hash(p)==r['sha256'],('Frozen file changed',r['path'])
