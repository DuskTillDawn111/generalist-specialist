import copy
import json
from pathlib import Path
from datetime import datetime, timezone

from deployment_search.runtime_cache import digest, file_hash, save_json
from module8c_final_real_search.common import inputs

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / 'results/module8e2_heldout_freeze'
OUT = ROOT / 'results/module8e3_heldout_validation'
METHODS = ['M1', 'M2', 'B2']


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_hash(path)}


def utc():
    return datetime.now(timezone.utc).isoformat()


def immutable(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, 'Frozen execution record changed: ' + str(path)
    else:
        save_json(path, value)
    return ref(path)


def verify_preregistration():
    for p in F.iterdir():
        if p.is_file() and not p.name.endswith('.sha256'):
            sidecar = Path(str(p) + '.sha256')
            assert sidecar.exists() and file_hash(p) == sidecar.read_text().strip(), str(p)
            if p.suffix == '.json': read(p)
            else: p.read_text()
    manifest = read(F / 'artifact_manifest.json')
    for r in list(manifest['artifacts'].values()) + [manifest['generator_source']]:
        assert file_hash(r['path']) == r['sha256'], r['path']
    for w in read(F / 'heldout_workloads.json')['workloads']:
        h = read(F / 'heldout_workload_hashes.json')['workloads'][w['id']]
        assert digest(w) == h['workload_sha256'] and digest(w['lambda_rps']) == h['lambda_sha256']
    from module8e2_freeze import verify_sources
    verify_sources()
    assert read(F / 'heldout_campaign_gate.json')['HELDOUT_GPU_CAMPAIGN_READY']
    return True


def verify_manifest():
    manifest = read(OUT / 'campaign_manifest.json')
    for group in ['preregistration', 'execution_sources', 'frozen_shared_inputs']:
        for r in manifest[group].values():
            assert file_hash(r['path']) == r['sha256'], r['path']
    for r in [manifest['context'], manifest['tests']]:
        assert file_hash(r['path']) == r['sha256'], r['path']
    return manifest


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    verify_preregistration()
    if (OUT / 'campaign_manifest.json').exists():
        return verify_manifest()
    from candidate_pool_audit.provenance import environment_snapshot
    from replica_scaling.telemetry import snapshot
    protocol = read(F / 'method_comparison_freeze.json')
    checked = []
    model_sources = read(protocol['frozen_inputs']['model_sources']['path'])
    for model, record in model_sources.items():
        for name, r in record['model_files'].items():
            p = ROOT / name
            actual = file_hash(p)
            checked.append({'path': str(p), 'expected': r['sha256'], 'actual': actual,
                            'match': actual == r['sha256'], 'model': model})
            assert actual == r['sha256'], str(p)
        for key in ['adapter', 'vqa_extension']:
            if key in record:
                r = record[key]
                assert file_hash(r['path']) == r['sha256'], r['path']
        print(json.dumps({'event': 'model_assets_verified', 'model': model,
                          'files': len(record['model_files'])}), flush=True)
    # Task payload and handler sources: full frozen native manifest, not only version labels.
    native = read(ROOT / 'results/integer_replica_search_real_pilot/before_hashes.json')['native_models_and_adapters']
    verified = {r['path'] for r in checked}
    for name, r in native.items():
        p = ROOT / name
        if str(p) not in verified:
            assert file_hash(p) == r['sha256'], str(p)
    env = environment_snapshot()
    assert env == read(protocol['frozen_inputs']['environment']['path']), 'Environment identity mismatch'
    gpu = snapshot()
    old = read(ROOT / 'results/module8c_final_real_search/experiment_freeze.json')['gpu']
    for key in ['uuid', 'name', 'driver', 'total_mib']:
        assert gpu[key] == old[key], 'GPU identity mismatch: ' + key
    assert not gpu['processes'] and gpu['used_mib'] <= 128, 'GPU is not isolated'
    context = copy.deepcopy(read(protocol['runtime_context_template']['path']))
    # Bind the old serving template to the new immutable workload provenance only.
    context['source_hashes']['workloads'] = ref(F / 'heldout_workloads.json')
    context['context_sha256'] = digest({k: v for k, v in context.items() if k != 'context_sha256'})
    context_path = OUT / 'runtime_context.json'
    immutable(context_path, context)
    Path(str(context_path) + '.sha256').write_text(file_hash(context_path) + '\n')
    tests = read(OUT / 'adapter_validation.json')
    assert tests['status'] == 'PASS' and tests['GPU_runs'] == 0
    verification = read(OUT / 'hash_verification.json')
    assert verification['status'] == 'PASS'
    verification.update(model_asset_checks=checked, native_manifest_files_verified=len(native),
                        live_environment_matches=True, live_GPU=gpu, verification_complete_at_utc=utc())
    save_json(OUT / 'hash_verification.json', verification)
    sources = {str(p.relative_to(ROOT)): ref(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    manifest = {'module': '8E-3', 'prepared_at_utc': utc(), 'preregistration': {
        p.name: ref(p) for p in sorted(F.iterdir()) if p.is_file()},
        'execution_sources': sources, 'frozen_shared_inputs': protocol['frozen_inputs'],
        'context': ref(context_path), 'tests': ref(OUT / 'adapter_validation.json'),
        'hash_verification': ref(OUT / 'hash_verification.json'), 'GPU': gpu,
        'environment': env, 'methods': METHODS,
        'logical_search_evaluation_cap': 78, 'unique_physical_state_cap': 72,
        'physical_attempt_cap_including_confirmations': 264,
        'retries_allowed': 0, 'only_permitted_method_difference': 'M2 targeted neighborhood + diversified Top-B',
        'context_template_binding': 'Only source_hashes.workloads and derived context_sha256 change, mechanically binding the held-out matrix. All serving/routing/model/memory settings identical to the frozen template.',
        'scheduler': 'Unmodified original methods run behind blocking RuntimeEvaluator request adapters. Main scheduler delegates one frozen logical slot at a time to unchanged ReplicaRuntimeEvaluator. No method sees the other method\'s observations before its own independent selection.',
        'ROUTING_REVISION_PENDING': True, 'extra_experiments_allowed': False}
    immutable(OUT / 'campaign_manifest.json', manifest)
    return manifest
