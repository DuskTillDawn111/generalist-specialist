"""Module 28R CPU audit. Reads original sources; writes only the 28R directory."""
import time, copy, traceback
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, json, csv, hashlib, collections
from pathlib import Path
import numpy as np
ROOT = Path('/root/autodl-tmp/generalist-specialist')
OUT = ROOT / 'results/module29_dynamic_replication'
OLD = ROOT / 'results/module28_dynamic_split'
sys.path.insert(0, str(ROOT))
from performance_model.local_calibration.stability_v2 import reconstruct, classify, DEFAULT_THRESHOLDS, timeline
from types import SimpleNamespace
from module25.common import json_native
from module26.common import arithmetic

def save(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=json_native) + '\n')
    tmp.replace(p)

def ref(p):
    p = Path(p)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)

def csvsave(name, rows, fields=()):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys([*fields, *(k for r in rows for k in r)]))
    with p.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k,v in r.items()})

def metadata():
    roots = [ROOT / 'results', ROOT / 'joint_deploy', ROOT / 'performance_model', ROOT / 'deployment_search', ROOT / 'deployment_core']
    roots += [p for p in ROOT.glob('module*') if p.is_dir()]
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for root in roots for p in root.rglob('*')
            if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts}

def stats(rows, key):
    a = [r[key] for r in rows if r.get(key) is not None]
    return dict(n=len(a), mean=float(np.mean(a)), p95=float(np.percentile(a,95))) if a else dict(n=0,mean=None,p95=None)

def curve(raw):
    ph = raw['phases']['measurement']; win = ph['window']; rows = ph['request_records']
    start, end = win['start_time_ns'], win['end_time_ns']
    events = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        for field, delta in [('scheduled_time_ns',1), ('enqueued_time_ns',1), ('actual_submit_time_ns',1), ('completion_time_ns',-1)]:
            at = r.get(field)
            if at is None or (field=='completion_time_ns' and r['status']!='completed'): continue
            scope = {'scheduled_time_ns':'offered','enqueued_time_ns':'host','actual_submit_time_ns':'submitted','completion_time_ns':None}[field]
            for s in ([scope] if scope else ['offered','host','submitted']):
                events[at][s + ':system'] += delta
                events[at][s + ':' + r['model']] += delta
    state = collections.Counter(); out=[]
    for at, ds in sorted(events.items()):
        state.update(ds)
        out.append(dict(run_id=raw['slot']['run_id'], offset_seconds=(at-start)/1e9,
                        within_measurement=at<=end, **dict(state)))
    return out

