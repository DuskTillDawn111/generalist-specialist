import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/module14_heavy_structural'
PREVIOUS = ROOT / 'results/module14_high_quality_heavy'
PLAN = ROOT / 'results/module12_specialist_sensitivity/experiment_freeze.json'
E = 'EFFICIENCY_OPTIMAL_SPECIALIST_REGIME'
H = 'HEAVY_SPECIALIST_REGIME'


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def table(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def csvsave(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def ref(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def now():
    return datetime.now(timezone.utc).isoformat()


def verify(refs):
    for record in refs:
        assert ref(record['path'])['sha256'] == record['sha256'], record['path']


def catalog(data):
    from deployment_core.models import ModelProfile
    from deployment_core.replica_calibration import ReplicaCalibration
    c = {m: ModelProfile(m, frozenset(p['supported_tasks']), p['idle_memory_mib'], p['service_time_ms'], p.get('sources', {})) for m, p in data['models'].items()}
    cal = ReplicaCalibration(
        {m: {int(n): v for n, v in t.items()} for m, t in data['capacity_factors'].items()},
        {m: {int(n): v for n, v in t.items()} for m, t in data['memory_table_mib'].items()},
        data['evidence_bounds'], data['model_evidence'], data['source_hashes']).validate(c)
    return c, {m: p['role'] for m, p in data['models'].items()}, cal
