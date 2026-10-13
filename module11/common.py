import csv
import json
import math
from pathlib import Path
from module10.common import ROOT, inputs, read, ref, now, factor, contexts, verify, protected

OUT = ROOT / 'results/module11_multihotspot'

def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))

def csvsave(name, rows, columns=None):
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns); w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(v, sort_keys=True) if isinstance(v, (dict,list)) else v for k,v in r.items()})

def rates(background, heads, alpha):
    return {k:v + alpha*heads.get(k, 0.) for k,v in background.items()}

def safe_memory(x, cal):
    return factor()*math.fsum(cal.memory_table_mib[m][n] for m,n in x.items())

def serial(route):
    return {f'{k}::{m}':float(v) for (k,m),v in route.items()}
