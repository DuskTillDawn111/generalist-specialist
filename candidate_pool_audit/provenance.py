import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .inventory import ROOT, SOURCES, selected_weights
from .registry import registry

OUT = ROOT / 'results/candidate_pool_v2'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def environment_snapshot():
    result = {}
    for launcher in registry().values():
        executable = str(launcher['python'])
        if executable in result:
            continue
        code = 'import json,sys,importlib.metadata as m; print(json.dumps({"python":sys.version,"executable":sys.executable,"packages":sorted([(d.metadata["Name"],d.version) for d in m.distributions()])}))'
        raw = subprocess.check_output([executable, '-c', code], text=True)
        data = json.loads(raw)
        data['package_manifest_sha256'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        data['executable_sha256'] = sha(executable)
        data['isolation'] = 'Existing environment reused without installation or mutation'
        result[executable] = data
    return result


def source_snapshot(include_weights):
    result = {}
    files = []
    for directory in ('deployment_core', 'deployment_search', 'performance_model', 'joint_deploy', 'benchmark/scripts', 'candidate_pool_audit'):
        files.extend((ROOT / directory).rglob('*.py'))
    files += list((ROOT / 'benchmark/manifests').glob('*.jsonl'))
    files += list((ROOT / 'benchmark/profiles').glob('*.json'))
    files += [ROOT / 'benchmark/prompts/task_prompts.json', ROOT / 'benchmark/task_pools/mixed_formal_1200_6task.jsonl']
    files += list((OUT / 'sources').glob('*'))
    files += list((ROOT / 'benchmark/results/formal').glob('*_summary.json'))
    for _, (source, directory, *_) in SOURCES.items():
        if directory:
            base = ROOT / 'models' / directory
            files += list(base.glob('*.json'))
            files += list(base.glob('*.py'))
            if include_weights:
                files += selected_weights(directory)
    for path in sorted(set(files)):
        result[str(path.relative_to(ROOT))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['before', 'after'])
    args = parser.parse_args()
    environments = environment_snapshot()
    sources = source_snapshot(args.phase == 'after')
    if args.phase == 'before':
        (OUT / 'provenance_before.json').write_text(json.dumps({'environments': environments, 'sources': sources}, indent=2))
    else:
        before = json.loads((OUT / 'provenance_before.json').read_text())
        environment_unchanged = environments == before['environments']
        protected = ('deployment_core/', 'deployment_search/', 'performance_model/', 'joint_deploy/', 'benchmark/')
        changed = [path for path, value in before['sources'].items() if path.startswith(protected) and sources.get(path) != value]
        (OUT / 'environment_manifest.json').write_text(json.dumps({'environments': environments, 'unchanged_from_before': environment_unchanged,
            'new_environments': [], 'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
            'environment_by_candidate': {name: str(cfg['python']) for name, cfg in registry().items()}}, indent=2))
        (OUT / 'source_hashes.json').write_text(json.dumps({'algorithm_and_baseline_files_unchanged': not changed,
            'changed_protected_files': changed, 'files': sources}, indent=2))
        if not environment_unchanged or changed:
            raise RuntimeError('Frozen source/environment drift')
    print(json.dumps({'phase': args.phase, 'files': len(sources), 'environments': len(environments)}), flush=True)


if __name__ == '__main__':
    main()
