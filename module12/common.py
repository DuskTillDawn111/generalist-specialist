import csv, json, math, copy, hashlib
from pathlib import Path
from deployment_search.runtime_cache import save_json, file_hash
from module10.common import ROOT, inputs, factor, read, ref, now, verify
from deployment_core.models import ModelProfile
from deployment_core.replica_calibration import ReplicaCalibration

OUT = ROOT / 'results/module12_specialist_sensitivity'
BASE = ROOT / 'results/module11_replica_and_cost'
GAMMAS = [1., 1.1, 1.25, 1.5, 2., 2.5, 3., 4.]
MUS = [1., 1.1, 1.25, 1.5, 2., 2.5, 3.]


def save(name, value):
    path = OUT / name; path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, value)


def csvsave(name, rows):
    path = OUT / name; path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def baseline():
    c, hw, roles, old = inputs()
    data = read(BASE / 'efficiency_optimal_baseline/replica_scaling.json')
    cal = ReplicaCalibration(
        {m: {int(n): v for n, v in tab.items()} for m, tab in data['capacity_factors'].items()},
        {m: {int(n): v for n, v in tab.items()} for m, tab in data['memory_table_mib'].items()},
        data['evidence_bounds'], data['model_evidence'], data['source_hashes']).validate(c)
    return c, hw, roles, cal


def scaled(c, roles, cal, gamma, mu, family=None):
    selected = {m for m in c if roles[m] == 'specialist'} if family is None else set(family)
    assert all(roles[m] == 'specialist' for m in selected)
    profiles = {m: ModelProfile(m, p.supported_tasks,
        p.memory_mib * (mu if m in selected else 1.),
        {k: v * (gamma if m in selected else 1.) for k, v in p.service_time_ms.items()},
        dict(p.sources)) for m, p in c.items()}
    memory = {m: {n: v * (mu if m in selected else 1.) for n, v in tab.items()} for m, tab in cal.memory_table_mib.items()}
    counterfactual = ReplicaCalibration(copy.deepcopy(cal.capacity_factors), memory,
        dict(cal.evidence_bounds), copy.deepcopy(cal.model_evidence), dict(cal.source_hashes)).validate(profiles)
    return profiles, counterfactual


def query_key(workload, budget, gamma, mu, family):
    raw = {'workload': workload, 'budget': budget, 'gamma': gamma, 'mu': mu, 'family': family}
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
