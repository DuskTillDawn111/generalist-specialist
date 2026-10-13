"""Authenticate all prior evidence and freeze two bounded real campaigns."""
from pathlib import Path
import json
from candidate_pool_service_demand.dry_run import load_completed_catalog
from candidate_pool_audit.provenance import environment_snapshot
from deployment_core import Workload
from deployment_core.candidate_pool_v2 import mandatory_presence
from deployment_core.replica_calibration import load_replica_calibration, ReplicaPlanningPolicy
from deployment_search import IntegerGuidedDeploymentSearch, SearchConfig
from deployment_search.runtime_cache import save_json, file_hash
from replica_scaling.telemetry import snapshot
from integer_replica_pilot.prepare import ROOT, OUT as PILOT, CONFIG, read, ref, verify_freeze as verify_pilot

OUT = ROOT / 'results/integer_replica_search_multiworkload'
JOBS = (('w1', 'W1_balanced'), ('w3', 'W3_specialist_friendly'))
ALL = (('w1', 'W1_balanced'), ('w2', 'W2_generalist_heavy'), ('w3', 'W3_specialist_friendly'))


def verify_references(references):
    for name, value in references.items():
        if file_hash(value['path']) != value['sha256']:
            raise RuntimeError('Frozen evidence changed: ' + name)


def verify_freeze():
    freeze = read(OUT / 'experiment_freeze.json')
    verify_references(freeze['frozen_files'])
    verify_references(freeze['implementation_hashes'])
    verify_pilot()
    return freeze


def imported_protocol(base, workload, records):
    # Retain measurement fields, source labels, seeds and acceptance semantics
    # verbatim. Only the frozen workload and absence of new smoke runs differ.
    protocol = dict(base)
    protocol.update(workload=workload, adapter_smoke_plan=[], smoke_physical_attempt_cap=0,
        total_physical_attempt_cap=22, mandatory_presence=mandatory_presence(records, Workload(workload['arrival_rates_rps'])),
        adapter_correctness_source=ref(PILOT / 'replica_adapter_validation.json'))
    if protocol['measurement_settings']['warmup_seconds'] != workload['warmup_seconds'] or protocol['measurement_settings']['measurement_seconds'] != workload['measurement_seconds']:
        raise RuntimeError('Frozen workload timing differs from 7C-3')
    return protocol


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'experiment_freeze.json').exists():
        return verify_freeze()
    verify_pilot()
    if read(PILOT / 'campaign_status.json')['status'] != 'complete' or read(PILOT / 'pilot_validation.json')['status'] != 'passed':
        raise RuntimeError('W2 pilot is not completed and audited')
    if not read(PILOT / 'confirmation_results.json')['complete'] or read(PILOT / 'replica_adapter_validation.json')['status'] != 'passed':
        raise RuntimeError('W2 confirmation/adapter evidence incomplete')
    verify_references(read(PILOT / 'artifact_manifest.json')['artifacts'])
    catalog, hardware, roles, records, _ = load_completed_catalog()
    calibration = load_replica_calibration(catalog)
    base = read(PILOT / 'pilot_protocol.json')
    context = read(PILOT / 'runtime_context.json')
    bounds, _ = ReplicaPlanningPolicy(catalog, 'empirical', calibration=calibration).bounds(hardware)
    if bounds != context['replica_bounds'] or base['search_config'] != read(PILOT / 'search_trace.json')['config']:
        raise RuntimeError('7C-3 evidence/search configuration differs from current inputs')
    env = environment_snapshot()
    if env != read(PILOT / 'environment_manifest.json')['environments']:
        raise RuntimeError('Frozen environments changed')
    before = read(PILOT / 'before_hashes.json')
    verify_references(before['frozen_files'])
    for index, (name, value) in enumerate(before['native_models_and_adapters'].items(), 1):
        if file_hash(ROOT / name) != value['sha256']:
            raise RuntimeError('Model/adapter changed: ' + name)
        if index % 50 == 0:
            print(json.dumps({'event': 'source_hash_check', 'checked': index, 'total': len(before['native_models_and_adapters'])}), flush=True)
    gpu = snapshot()
    if gpu['processes'] or gpu['used_mib'] > 128 or gpu['uuid'] != context['gpu_uuid'] or gpu['total_mib'] != hardware.gpu_memory_mib:
        raise RuntimeError('GPU identity/isolation/capacity differs from 7C-3')
    workload_file = ROOT / 'results/deployment_search_multiworkload/benchmark_workloads.json'
    if file_hash(workload_file) != read(ROOT / 'results/deployment_search_multiworkload/freeze_manifest.json')['files']['benchmark_workloads.json']:
        raise RuntimeError('Module 3C workloads changed')
    workloads = {row['name']: row for row in read(workload_file)['workloads']}
    if workloads['W2_generalist_heavy'] != base['workload']:
        raise RuntimeError('W2 result does not use the required Module 3C workload')
    w2_files = {str(p.relative_to(PILOT)): ref(p) for p in sorted(PILOT.rglob('*'))
                if p.is_file() and not p.name.endswith('.lock')}
    references = dict(before['frozen_files'])
    references.update({'7C3/' + name: value for name, value in w2_files.items()})
    references.update({'7C3_code/' + str(p.relative_to(ROOT)): ref(p) for p in sorted((ROOT / 'integer_replica_pilot').glob('*.py'))})
    save_json(OUT / 'before_hashes.json', {'frozen_files': references, 'native_models_and_adapters': before['native_models_and_adapters']})
    save_json(OUT / 'environment_manifest.json', {'environments': env, 'matches_frozen_7C3': True})
    preflight = {}
    for short, name in JOBS:
        directory = OUT / short
        directory.mkdir(exist_ok=True)
        protocol = imported_protocol(base, workloads[name], records)
        save_json(directory / 'protocol.json', protocol)
        (directory / 'runtime_context.json').write_bytes((PILOT / 'runtime_context.json').read_bytes())
        (directory / 'runtime_context.json.sha256').write_bytes((PILOT / 'runtime_context.json.sha256').read_bytes())
        result = IntegerGuidedDeploymentSearch().search(Workload(workloads[name]['arrival_rates_rps']), catalog, hardware,
            SearchConfig(dry_run=True, **{k: v for k, v in base['search_config'].items() if k != 'dry_run'}),
            roles=roles, capacity_mode='empirical', memory_mode='empirical', calibration=calibration)
        preflight[short] = result.to_dict()
        save_json(directory / 'planning_preflight.json', result.to_dict())
    (OUT / 'w2').mkdir(exist_ok=True)
    save_json(OUT / 'w2/reference.json', {'source_module': '7C-3', 'workload': workloads['W2_generalist_heavy'],
        'search_trace': ref(PILOT / 'search_trace.json'), 'confirmation_results': ref(PILOT / 'confirmation_results.json'),
        'runtime_evaluations': ref(PILOT / 'runtime_evaluations.json'), 'artifact_manifest': ref(PILOT / 'artifact_manifest.json'),
        'all_source_files': w2_files, 'new_GPU_runs': 0, 'reuse_policy': 'Read immutable recorded outcomes and raw traces; never call W2 runtime/search.'})
    frozen = {str(p.relative_to(OUT)): ref(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json', '.sha256')}
    implementation = {str(p.relative_to(ROOT)): ref(p) for d in ('integer_replica_multiworkload', 'integer_replica_pilot', 'deployment_core', 'deployment_search', 'joint_deploy/measurement') for p in sorted((ROOT / d).glob('*.py'))}
    freeze = {'module': '7C-4', 'frozen_before_first_new_GPU_run': True, 'gpu': gpu,
        'workload_source': ref(workload_file), 'workloads': workloads, 'new_GPU_workloads': [name for _, name in JOBS],
        'execution_order': ['w1', 'w3'], 'W2_reused_without_GPU': True, 'adapter_smoke_reused_without_GPU': True,
        'search_config': base['search_config'], 'measurement_settings': base['measurement_settings'],
        'capacity_mode': 'empirical', 'memory_mode': 'empirical', 'predictor': None, 'replica_bounds': bounds,
        'input_hashes': read(PILOT / 'pilot_freeze.json')['input_hashes'], 'source_W2_freeze': ref(PILOT / 'pilot_freeze.json'),
        'per_new_workload_budgets': {'unique_search_states': 6, 'search_physical_attempts': 18, 'search_repetitions_per_state': 3,
            'confirmation_physical_attempts': 4, 'confirmation_repetitions_per_state': 2, 'max_iterations': 3, 'top_b': 2, 'no_improvement_patience': 1},
        'total_new_GPU_physical_attempt_cap': 44, 'no_retry_or_budget_growth': True,
        'objective': base['objective'], 'frozen_files': frozen, 'implementation_hashes': implementation}
    save_json(OUT / 'experiment_freeze.json', freeze)
    print(json.dumps({'event': 'experiment_frozen', 'sha256': file_hash(OUT / 'experiment_freeze.json'), 'new_GPU_runs': 0,
        'preflight_seeds': {k: t['deployment'] for k, t in preflight.items()}}), flush=True)
    return freeze


if __name__ == '__main__':
    prepare()
