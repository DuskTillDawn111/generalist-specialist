"""Evidence-bound discrete planning data. No serving or GPU imports."""
from dataclasses import dataclass, field
import hashlib
import json
import math
import statistics
from pathlib import Path
from .replicas import derive_replica_bounds, validate_bounds, normalize_counts

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / 'results/replica_scaling_real'


def reference(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@dataclass(frozen=True)
class ReplicaCalibration:
    capacity_factors: dict[str, dict[int, float]]
    memory_table_mib: dict[str, dict[int, float]]
    evidence_bounds: dict[str, int]
    model_evidence: dict = field(default_factory=dict)
    source_hashes: dict = field(default_factory=dict)

    def validate(self, catalog):
        names = set(catalog)
        validate_bounds(self.evidence_bounds, names)
        if set(self.capacity_factors) != names or set(self.memory_table_mib) != names:
            raise ValueError('Calibration must exactly cover the planning catalog')
        for m in names:
            bound = self.evidence_bounds[m]
            if bound < 1:
                raise ValueError('Every candidate must retain a single-replica baseline')
            for label, table in [('capacity', self.capacity_factors[m]), ('memory', self.memory_table_mib[m])]:
                if set(table) != set(range(bound + 1)) or any(type(n) is not int for n in table):
                    raise ValueError(f'Contiguous evidence-bound {label} table required: {m}')
                if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 or (n > 0 and v == 0)
                       for n, v in table.items()) or table[0] != 0:
                    raise ValueError(f'Invalid discrete {label} values: {m}')
            if self.capacity_factors[m][1] != 1:
                raise ValueError('Single-replica relative capacity must equal one')
            if self.memory_table_mib[m][1] != catalog[m].memory_mib:
                raise ValueError('M(1) must preserve frozen single-replica idle VRAM / binary behavior')
        return self


def load_replica_calibration(catalog, directory=DEFAULT_EVIDENCE):
    """Read full precision frozen high-load paired means; no extrapolation.

    Qwen/Whisper evidence is ASR-only; family-level transfer to other tasks/mixes
    remains an analytical surrogate. M(1) anchors to frozen candidate-pool idle
    memory for exact binary compatibility. M(2) is measured post-warm idle mean.
    """
    directory = Path(directory)
    manifest = json.loads((directory / 'artifact_manifest.json').read_text())
    if not manifest.get('frozen') or manifest.get('valid_full_runs') != 54:
        raise ValueError('Completed frozen Module 7C-1 evidence required')
    required = ('replica_scaling_metrics.json', 'memory_scaling.json', 'planning_surrogate_validation.json',
                'replica_scaling_plan.json', 'measurement_validation.json')
    data, sources = {}, {'artifact_manifest.json': reference(directory / 'artifact_manifest.json')}
    for name in required:
        path = directory / name
        sources[name] = reference(path)
        if sources[name]['sha256'] != manifest['artifacts'][name]['sha256']:
            raise ValueError('Frozen 7C-1 artifact hash mismatch: ' + name)
        data[name] = json.loads(path.read_text())
    audit = data['measurement_validation.json']
    if audit['status'] != 'passed' or not audit['environments_unchanged'] or not audit['frozen_inputs_and_7B_unchanged']:
        raise ValueError('Invalid original measurement provenance')
    plan = data['replica_scaling_plan.json']
    if sources['replica_scaling_plan.json']['sha256'] != audit['frozen_plan_sha256']:
        raise ValueError('Plan/audit mismatch')
    # Verify actual raw repetitions; an edited aggregate cannot create evidence.
    for ref in audit['raw_file_hashes'].values():
        if reference(ref['path'])['sha256'] != ref['sha256']:
            raise ValueError('Frozen 7C-1 raw measurement changed')
    comparisons = data['replica_scaling_metrics.json']['scaling_comparisons']
    observed_memory = data['memory_scaling.json']['conditions']
    measured = {m['model_id']: m for m in plan['models']}
    if set(measured) != {'qwen', 'whisper-small', 'resnet50'}:
        raise ValueError('Unexpected measured model identities')
    g, memory, evidence_bounds, evidence = {}, {}, {}, {}
    for m, profile in sorted(catalog.items()):
        g[m], memory[m] = {0: 0., 1: 1.}, {0: 0., 1: profile.memory_mib}
        evidence_bounds[m] = 1
        evidence[m] = {'replica_scaling_status': 'pending', 'measured_replica_counts': [1],
                       'reason': 'No frozen multi-replica scaling evidence; 0/1 baseline remains available',
                       'single_memory_source': profile.sources.get('memory', {})}
        if m not in measured:
            continue
        if measured[m]['measured_idle_vram_mib'] != profile.memory_mib:
            raise ValueError('Frozen v2 memory identity mismatch: ' + m)
        pair = next(p for p in comparisons if p['model_id'] == m and p['load_level'] == 'high')
        if not pair['all_three_pairs_eligible'] or not pair['baseline_capacity_pressure_observed']:
            raise ValueError('Three comparable pressure measurements required: ' + m)
        values = pair['paired_ratios']['throughput_gain']['values']
        value = pair['paired_ratios']['throughput_gain']['mean']
        if len(values) != 3 or statistics.mean(values) != value:
            raise ValueError('Full-precision paired aggregation inconsistent')
        assessment = next(row for row in data['planning_surrogate_validation.json']['model_assessments'] if row['model_id'] == m)
        if assessment['exploratory_g2'] != value:
            raise ValueError('Surrogate report/full aggregate mismatch')
        one = next(row for row in observed_memory if (row['model_id'], row['replica_count'], row['load_level']) == (m, 1, 'high'))
        two = next(row for row in observed_memory if (row['model_id'], row['replica_count'], row['load_level']) == (m, 2, 'high'))
        if not one['release_verified'] or not two['release_verified']:
            raise ValueError('Memory isolation/unload evidence incomplete')
        observation = two['measurements']['idle_before_workload_owned_mib']
        if observation['count'] != 3:
            raise ValueError('Three idle measurements required')
        g[m][2], memory[m][2], evidence_bounds[m] = value, observation['mean'], 2
        evidence[m] = {'replica_scaling_status': 'verified_representative_task', 'measured_replica_counts': [1, 2],
            'task': measured[m]['task'], 'load_level': 'high', 'precision': measured[m]['precision'],
            'source_revision': measured[m]['source_revision'], 'model_manifest_sha256': measured[m]['model_manifest_sha256'],
            'adapter_sha256': measured[m]['adapter']['sha256'], 'environment_hash': measured[m]['environment_hash'],
            'capacity_field': f'scaling_comparisons[{m},high].paired_ratios.throughput_gain.mean',
            'capacity_full_precision_values': values, 'capacity_sample_std': pair['paired_ratios']['throughput_gain']['sample_std'],
            'single_memory_source': profile.sources.get('memory', {}), 'M1_policy': 'frozen candidate pool measured single-replica idle VRAM; retain binary compatibility',
            'observed_7c1_single_idle_mib': one['measurements']['idle_before_workload_owned_mib'],
            'observed_7c1_double_idle_mib': observation,
            'observed_7c1_single_peak_mib': one['measurements']['peak_owned_mib'],
            'observed_7c1_double_peak_mib': two['measurements']['peak_owned_mib'],
            'capacity_scope': 'Measured representative task only; applying the factor to aggregate family demand / other tasks is a transfer surrogate, pending real mixed-workload validation.',
            'memory_scope': 'Measured single-family post-warm idle mean; not a guarantee of joint heterogeneous peak VRAM.'}
    return ReplicaCalibration(g, memory, evidence_bounds, evidence, sources).validate(catalog)


class ReplicaPlanningPolicy:
    def __init__(self, catalog, capacity_mode='linear', memory_mode=None, calibration=None):
        if capacity_mode not in ('linear', 'empirical'):
            raise ValueError('capacity_mode must be linear or empirical')
        memory_mode = memory_mode or ('empirical' if capacity_mode == 'empirical' else 'additive')
        if memory_mode not in ('additive', 'empirical'):
            raise ValueError('memory_mode must be additive or empirical')
        self.catalog, self.capacity_mode, self.memory_mode = catalog, capacity_mode, memory_mode
        self.calibration = calibration
        if capacity_mode == 'empirical' or memory_mode == 'empirical':
            self.calibration = (calibration or load_replica_calibration(catalog)).validate(catalog)
        elif calibration is not None:
            calibration.validate(catalog)

    def capacity(self, model, count):
        if type(count) is not int or count < 0:
            raise ValueError('Nonnegative integer replica count required')
        if self.capacity_mode == 'linear':
            return float(count)
        try:
            return self.calibration.capacity_factors[model][count]
        except KeyError as exc:
            raise ValueError('Replica scaling pending/outside evidence bound') from exc

    def memory(self, model, count):
        if type(count) is not int or count < 0:
            raise ValueError('Nonnegative integer replica count required')
        if self.memory_mode == 'additive':
            return self.catalog[model].memory_mib * count
        try:
            return self.calibration.memory_table_mib[model][count]
        except KeyError as exc:
            raise ValueError('Replica memory pending/outside evidence bound') from exc

    def deployment_memory(self, deployment):
        return math.fsum(self.memory(m, n) for m, n in normalize_counts(deployment, self.catalog).items())

    def bounds(self, hardware, replica_bounds=None, worker_limits=None, binary_mode=False):
        physical, provenance = derive_replica_bounds(self.catalog, hardware, worker_limits, binary_mode)
        use_evidence = self.capacity_mode == 'empirical' or self.memory_mode == 'empirical'
        selected = {}
        for m, bound in physical.items():
            evidence = self.calibration.evidence_bounds[m] if use_evidence else None
            selected[m] = min(bound, evidence) if evidence is not None else bound
            provenance[m].update(R_evidence=evidence, R=selected[m], capacity_mode=self.capacity_mode,
                memory_mode=self.memory_mode, replica_scaling_status=self.calibration.model_evidence.get(m, {}).get('replica_scaling_status', 'declared_fixture') if use_evidence else 'linear_analytical_baseline',
                bound_source=provenance[m]['bound_source'] + (' intersect measured replica evidence' if use_evidence else ''))
        if replica_bounds is not None:
            validate_bounds(replica_bounds, self.catalog)
            if any(replica_bounds[m] > selected[m] for m in self.catalog):
                raise ValueError('Explicit replica bound exceeds physical/process/evidence upper bound')
            for m in selected:
                provenance[m].update(explicit_experiment_bound=replica_bounds[m], R=replica_bounds[m])
            selected = dict(replica_bounds)
        return selected, provenance
