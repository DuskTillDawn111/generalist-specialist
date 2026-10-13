import hashlib
import json
from pathlib import Path
from candidate_pool_audit.provenance import sha
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/replica_scaling_real'
PLAN = OUT / 'replica_scaling_plan.json'

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)

def reference(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def frozen_plan():
    plan = read(PLAN)
    if sha(OUT / 'source_hashes_before.json') != read(OUT / 'plan_freeze.json')['source_hashes_sha256']:
        raise RuntimeError('Frozen provenance manifest changed')
    if sha(PLAN) != read(OUT / 'plan_freeze.json')['plan_sha256']:
        raise RuntimeError('Frozen run plan changed')
    return plan

def verify_refs(refs, full=False):
    changed = []
    for name, ref in refs.items():
        path = Path(ref['path'])
        stat = path.stat()
        if full or stat.st_size != ref['bytes'] or stat.st_mtime_ns != ref['mtime_ns']:
            if sha(path) != ref['sha256']:
                changed.append(name)
    if changed:
        raise RuntimeError('Frozen assets/source changed: ' + str(changed))
    return len(refs)
