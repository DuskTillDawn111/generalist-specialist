"""Read-only hashes of model bytes, inputs, source, interpreter and package metadata."""
import json
from pathlib import Path
import subprocess

from joint_deploy.measurement.deployment_runtime import default_registry
from joint_deploy.measurement.run_measurement import DEFAULT_POOL, DEFAULT_PROFILE
from .runtime_cache import digest, file_hash, save_json

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRS = dict(qwen='qwen2.5-omni-7b', distilbert='distilbert-sst2',
                  paddleocr='paddleocr/official_models/PP-OCRv6_medium_rec',
                  resnet50='resnet50', whisper='whisper-small')


def asset_hashes():
    files = {}
    models = {}
    for name, directory in MODEL_DIRS.items():
        paths = sorted(p for p in (ROOT / 'models' / directory).rglob('*')
                       if p.is_file() and not any(part.startswith('.') for part in p.relative_to(ROOT).parts))
        entries = {}
        for p in paths:
            info = dict(sha256=file_hash(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            files[str(p)] = info
            entries[str(p.relative_to(ROOT / 'models' / directory))] = info['sha256']
        if not entries:
            raise ValueError('Missing model files: ' + name)
        models[name] = dict(content_sha256=digest(entries), files=entries)
    input_paths = {DEFAULT_POOL, DEFAULT_PROFILE, ROOT / 'benchmark/prompts/task_prompts.json'}
    for line in DEFAULT_POOL.read_text().splitlines():
        sample = json.loads(line)['sample']
        for value in sample.values():
            if isinstance(value, str) and value.startswith('/') and Path(value).is_file():
                input_paths.add(Path(value))
    inputs = {}
    for p in sorted(input_paths):
        info = dict(sha256=file_hash(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        files[str(p)] = info
        inputs[str(p)] = info['sha256']
    return dict(models=models, input_sha256=digest(inputs), inputs=inputs, files=files)


def source_hashes(assets):
    paths = set((ROOT / 'joint_deploy/measurement').glob('*.py'))
    paths.update(ROOT / 'joint_deploy' / n for n in
                 ('worker_protocol.py', 'run_static_hybrid.py', 'run_pure_qwen.py',
                  'qwen_worker.py', 'distilbert_worker.py', 'resnet_worker.py', 'whisper_worker.py', 'ocr_worker.py'))
    paths.update((ROOT / 'benchmark/scripts').glob('eval_qwen*.py'))
    paths.add(ROOT / 'benchmark/scripts/eval_paddleocr_iiit5k.py')
    paths.add(ROOT / 'performance_model/local_calibration/stability_v2.py')
    paths.add(ROOT / 'deployment_search/measurement_runtime.py')
    source = {str(p): file_hash(p) for p in sorted(paths)}
    interpreters = {}
    registry = default_registry()
    for m in MODEL_DIRS:
        executable = Path(registry[m]['python']).resolve()
        if str(executable) in interpreters:
            continue
        # Metadata lookup never imports torch/paddle or initializes a GPU context.
        command = [str(executable), '-c',
                   'import json,importlib.metadata as m,sys; print(json.dumps({"python":sys.version,"packages":sorted((d.metadata["Name"],d.version) for d in m.distributions())}))']
        metadata = json.loads(subprocess.check_output(command, text=True))
        interpreters[str(executable)] = dict(executable_sha256=file_hash(executable), metadata=metadata)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total',
                                    '--format=csv,noheader'], text=True).strip()
    return dict(model_hashes={m: v['content_sha256'] for m, v in assets['models'].items()},
                input_hash=assets['input_sha256'], source_hashes=source,
                interpreters=interpreters, gpu_identity=gpu,
                environment=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))


def verify_assets_unchanged(assets):
    for path, info in assets['files'].items():
        p = Path(path)
        if p.stat().st_size != info['bytes'] or p.stat().st_mtime_ns != info['mtime_ns']:
            if file_hash(p) != info['sha256']:
                raise RuntimeError('Frozen asset changed: ' + path)


if __name__ == '__main__':
    out = ROOT / 'results/deployment_search_real_pilot'
    save_json(out / 'asset_hashes.json', asset_hashes())
    print('Model and workload asset content hashes saved.', flush=True)
