"""Read the frozen capability/idle-memory inventory, without inventing s0.

Stage-1 service time requires the original standalone worker measurement protocol.
Formal quality benchmark E2E timing and minimal-load timings cannot replace it.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .models import Hardware, ModelProfile, load_model_catalog

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / 'results/candidate_pool_v2'


class MissingServiceDemand(ValueError):
    pass


@dataclass(frozen=True)
class CandidatePlanningRecord:
    name: str
    role: str
    supported_tasks: frozenset[str]
    memory_mib: float
    service_time_ms: dict[str, float | None]
    sources: dict

    def profile(self):
        missing = [k for k, value in self.service_time_ms.items() if value is None]
        if missing:
            raise MissingServiceDemand(f'{self.name}: missing frozen Stage-1 s0 for {missing}')
        return ModelProfile(self.name, self.supported_tasks, self.memory_mib, self.service_time_ms, self.sources)


def reference(path, field):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'field': field}


def load_candidate_pool_v2(directory=DEFAULT_POOL):
    directory = Path(directory)
    summary = json.loads((directory / 'pool_resource_summary.json').read_text())
    if summary.get('frozen') is not True or summary['gpu_capacity_mib'] != 49140:
        raise ValueError('Expected frozen v2 pool and its actual 49140 MiB GPU capacity')
    models = json.loads((directory / 'candidate_models.json').read_text())['models']
    measured = json.loads((directory / 'measured_vram.json').read_text())['models']
    matrix = json.loads((directory / 'compatibility_matrix.json').read_text())['models']
    old = load_model_catalog()
    aliases = {'whisper-small': 'whisper'}
    records = {}
    for model in models:
        name = model['id']
        if name not in summary['formal_model_ids']:
            continue
        capabilities = frozenset(model['capabilities_verified'])
        memory = model['measured_idle_vram_mib']
        if (model['adapter_status'] != 'verified' or memory is None or memory <= 0 or
                measured[name]['measured_idle_vram_mib'] != memory or
                measured[name]['measurement_status'] != 'passed' or
                any(matrix[name][k]['status'] != 'verified' for k in capabilities)):
            raise ValueError('Inconsistent frozen model capability/resource evidence: ' + name)
        original = old.get(aliases.get(name, name))
        services = {k: original.service_time_ms[k] if original and k in original.supported_tasks else None
                    for k in sorted(capabilities)}
        records[name] = CandidatePlanningRecord(name, model['role'], capabilities, memory, services,
            {'memory': reference(directory / 'measured_vram.json', f'models.{name}.measured_idle_vram_mib'),
             'capability': reference(directory / 'compatibility_matrix.json', f'models.{name}'),
             'service_time': original.sources['service_time'] if original else {},
             'missing_service_policy': 'pending; no inference from smoke latency, file size, architecture, or another model'})
    if set(records) != set(summary['formal_model_ids']):
        raise ValueError('Frozen pool model IDs are inconsistent')
    hardware = Hardware(49140, 'Frozen candidate pool v2 GPU',
                         reference(directory / 'pool_resource_summary.json', 'gpu_capacity_mib'))
    return records, hardware


def missing_demands(records, workload):
    return [{'model': name, 'task': k, 'status': 'pending', 'missing': 'frozen Stage-1 s0[k,m]'}
            for name, model in sorted(records.items()) for k in sorted(model.supported_tasks)
            if workload.arrival_rate[k] > 0 and model.service_time_ms[k] is None]


def mandatory_presence(records, workload):
    result = []
    for task in workload.active_tasks:
        capable = [m for m, p in sorted(records.items()) if task in p.supported_tasks]
        if len(capable) == 1:
            result.append({'task': task, 'model': capable[0], 'constraint': f'y[{capable[0]}]=1',
                           'reason': 'mandatory model induced by capability coverage'})
    return result
