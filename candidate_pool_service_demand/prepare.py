"""Audit immutable inputs and the exact historical standalone protocol."""
import hashlib
import json
from pathlib import Path

from candidate_pool_audit.inventory import SOURCES, selected_weights
from candidate_pool_audit.provenance import environment_snapshot, sha
from candidate_pool_audit.registry import registry
from deployment_core.candidate_pool_v2 import load_candidate_pool_v2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/candidate_pool_v2_service_demand'
POOL = ROOT / 'benchmark/task_pools/mixed_formal_1200_6task.jsonl'
REPETITIONS = ('active_single_v3_current', 'active_single_v3_r2', 'active_single_v3_r3')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def reference(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def targets(task):
    return [json.loads(line) for line in POOL.read_text().splitlines()
            if line and json.loads(line)['task_type'] == task][:30]


def protected_files():
    files = {ROOT / 'deployment_core/catalog_sources.json', POOL,
             ROOT / 'benchmark/prompts/task_prompts.json'}
    for directory in ('deployment_core', 'deployment_search', 'joint_deploy',
                      'benchmark/scripts', 'candidate_pool_audit'):
        files.update((ROOT / directory).rglob('*.py'))
        files.update((ROOT / directory).glob('*.md'))
    files.update((ROOT / 'benchmark/manifests').glob('*.jsonl'))
    files.update((ROOT / 'benchmark/profiles').glob('*.json'))
    files.update((ROOT / 'results/candidate_pool_v2').rglob('*'))
    files = {p for p in files if p.is_file()}
    for rep in REPETITIONS:
        files.update((ROOT / 'benchmark/results/placement' / rep).glob('*'))
    # Keep the original memory snapshots and service-data references unchanged too.
    from deployment_core.models import load_model_catalog
    for profile in load_model_catalog().values():
        for source in (profile.sources['memory'], *profile.sources['service_time'].values()):
            if isinstance(source, dict) and source.get('path'):
                files.add(Path(source['path']))
            elif isinstance(source, list):
                files.update(Path(item['path']) for item in source)
    return sorted(files)


def model_files(name):
    directory = ROOT / 'models' / SOURCES[name][1]
    files = set(selected_weights(SOURCES[name][1]))
    files.update(p for p in directory.rglob('*') if p.is_file() and
                 p.suffix in ('.json', '.py', '.txt', '.model', '.tiktoken') and '.cache' not in p.parts)
    return sorted(files)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'before_hashes.json').exists():
        raise RuntimeError('Preparation already frozen; do not replace before hashes')
    records, hardware = load_candidate_pool_v2()
    matrix = read(ROOT / 'results/candidate_pool_v2/compatibility_matrix.json')['models']
    models = {m['id']: m for m in read(ROOT / 'results/candidate_pool_v2/candidate_models.json')['models']}
    configs = registry()
    missing = [(m, k) for m, p in sorted(records.items()) for k, v in sorted(p.service_time_ms.items()) if v is None]
    if len(missing) != 14:
        raise ValueError(f'Expected precisely 14 pending verified paths, got {len(missing)}')
    frozen_env = read(ROOT / 'results/candidate_pool_v2/environment_manifest.json')
    current_env = environment_snapshot()
    if current_env != frozen_env['environments']:
        raise RuntimeError('Frozen candidate runtime environment drift; no installs or measurements permitted')
    write(OUT / 'environment_manifest.json', {'environments': current_env, 'matches_frozen_v2': True,
          'frozen_manifest': reference(ROOT / 'results/candidate_pool_v2/environment_manifest.json')})
    expected_sources = read(ROOT / 'results/candidate_pool_v2/source_hashes.json')['files']
    paths = []
    model_hashes = {}
    for name in sorted({m for m, _ in missing}):
        hashes = {}
        for path in model_files(name):
            rel = str(path.relative_to(ROOT))
            current = reference(path)
            if rel in expected_sources and current['sha256'] != expected_sources[rel]['sha256']:
                raise RuntimeError('Frozen model file changed: ' + rel)
            hashes[rel] = current
        model_hashes[name] = {'source': models[name]['source'], 'source_revision': models[name]['source_revision'],
                              'files': hashes,
                              'model_manifest_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
    for name, task in missing:
        if matrix[name][task]['status'] != 'verified':
            raise ValueError('Cannot measure unsupported or pending capability')
        adapter = Path(configs[name]['script'])
        adapter_ref = reference(adapter)
        rel = str(adapter.relative_to(ROOT))
        if adapter_ref['sha256'] != expected_sources[rel]['sha256']:
            raise RuntimeError('Frozen adapter changed: ' + rel)
        if name in ('convnext_tiny', 'roberta_sst2', 'paddleocr_vl', 'whisper_turbo', 'distil_whisper'):
            common = ROOT / 'candidate_pool_audit/new_worker.py'
            if sha(common) != expected_sources[str(common.relative_to(ROOT))]['sha256']:
                raise RuntimeError('Frozen shared adapter changed')
        samples = targets(task)
        if len(samples) != 30:
            raise ValueError('Exactly 30 historical targets required')
        paths.append({'model_id': name, 'task': task, 'capability_status': 'verified', 'existing_value': None,
            'source': {'capability': reference(ROOT / 'results/candidate_pool_v2/compatibility_matrix.json'),
                       'existing_stage1_catalog': reference(ROOT / 'deployment_core/catalog_sources.json')},
            'runtime_environment': {'python': str(configs[name]['python']),
                                    'package_manifest_sha256': current_env[str(configs[name]['python'])]['package_manifest_sha256'],
                                    'executable_sha256': current_env[str(configs[name]['python'])]['executable_sha256']},
            'adapter': adapter_ref, 'model': model_hashes[name], 'precision': models[name]['precision'],
            'reason_pending': 'No frozen active_single_profile_v1 standalone worker-latency measurement for this verified path',
            'target_pool_ids': [s['pool_id'] for s in samples],
            'input_subset_sha256': hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(),
            'repetitions': 3, 'requests_per_repetition': 30, 'warmups_per_repetition': 1})
    write(OUT / 'pending_service_demand_paths.json', {'schema_version': 1, 'unit': 'milliseconds',
          'pending_path_count': len(paths), 'paths': paths})
    historical = []
    for name, profile in records.items():
        for task, value in profile.service_time_ms.items():
            if value is None:
                continue
            historical_name = 'whisper' if name == 'whisper-small' else name
            expected_ids = [s['pool_id'] for s in targets(task)]
            repetition_means = []
            for rep in REPETITIONS:
                path = ROOT / f'benchmark/results/placement/{rep}/active_single_{historical_name}_{task}_n30.jsonl'
                summary = read(str(path) + '.summary.json')
                raw = [json.loads(line) for line in path.read_text().splitlines()]
                if (summary['protocol'] != 'active_single_profile_v1' or summary['target_count'] != 30
                    or len(raw) != 30 or [r['pool_id'] for r in raw] != expected_ids):
                    raise ValueError('Historical protocol/input subset mismatch: ' + str(path))
                recomputed = sum(r['worker_latency_ms'] for r in raw) / 30
                if abs(recomputed - summary['mean_worker_latency_ms']) > 1e-9:
                    raise ValueError('Historical raw/summary mean mismatch')
                repetition_means.append(recomputed)
            if abs(sum(repetition_means) / 3 - value) > 1e-9:
                raise ValueError('Historical weighted aggregation mismatch')
            historical.append({'model_id': name, 'task': task, 'existing_value_ms': value,
                               'repetition_means_ms': repetition_means, 'measurement_count': 90,
                               'target_pool_ids': expected_ids, 'source': profile.sources['service_time'][task]})
    write(OUT / 'historical_protocol_audit.json', {'protocol': 'active_single_profile_v1',
          'implementation': reference(ROOT / 'joint_deploy/run_active_single_profile.py'),
          'timer_implementation': reference(ROOT / 'joint_deploy/worker_protocol.py'),
          'quantity': 'worker_latency_ms = (handler end_ns - handler start_ns)/1e6',
          'includes': 'decode/local input read, model preprocessing, CUDA transfer, inference, synchronization and task output processing inside handler',
          'excludes': 'model load, one warmup, IPC queue_wait_ms, submit-to-completion E2E latency',
          'summary_statistic': 'arithmetic mean; no median substitution',
          'aggregation': 'sum(mean_worker_latency_ms * target_count) / sum(target_count)',
          'repetitions': 3, 'requests_per_repetition': 30, 'warmup': 'first target sample once, request_id=-1; excluded from measured records',
          'submission_policy': 'enqueue the same first 30 task rows; existing worker processes sequentially; no arrival-rate/workload sweep',
          'pool': reference(POOL), 'existing_service_paths': historical,
          'input_note': 'ASR first 30 historical service-pool rows include general durations; do not substitute 4–6 s smoke/quality subset.',
          'adapter_policy': 'Frozen native model adapters, same task datasets/label semantics and output settings. Model-native tokenizers, prescribed image weight transforms and chat templates are retained, not replaced.',
          'generation_limits': {'phi4mm/minicpm': {'text': 16, 'image': 16, 'ocr': 32, 'asr': 256},
              'paddleocr_vl': {'ocr': 32, 'element_prompt': 'OCR:', 'do_sample': False},
              'whisper_turbo/distil_whisper': 'Frozen shared eval_whisper_librispeech.generate_transcription: language=en, task=transcribe, do_sample=False, native length defaults; LID uses acoustic detection without forced language'},
          'hardware_capacity_mib': hardware.gpu_memory_mib})
    write(OUT / 'before_hashes.json', {'files': {str(p.relative_to(ROOT)): reference(p) for p in protected_files()},
          'models': model_hashes, 'existing_service_values_ms': {m: dict(p.service_time_ms) for m, p in records.items()}})
    print(json.dumps({'pending_verified_paths': len(paths), 'unchanged_existing_paths': len(historical),
                      'environments_match': True}), flush=True)


if __name__ == '__main__':
    prepare()
