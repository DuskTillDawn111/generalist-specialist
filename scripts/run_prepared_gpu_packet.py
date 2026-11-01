"""Launch a sealed diagnostic packet from its own source tree after GPU release."""
import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import json
import os
from pathlib import Path
import subprocess

from resume_gpu_diagnostics import digest, read, require_gpu_release, verify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_GPU_LOCK = PROJECT_ROOT/'artifacts/.gpu-experiment.lock'


@contextmanager
def shared_gpu_lease():
    """Coordinate source-isolated jobs with the original repository's lease."""
    SHARED_GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with SHARED_GPU_LOCK.open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def validate_smoke(manifest, full_manifest, results):
    smoke = read(manifest)
    if smoke['code_hashes'] != full_manifest['code_hashes'] or len(smoke['jobs']) != 4:
        raise ValueError('Smoke must use the same source revision and cover all four tasks')
    verify({**smoke['sources'], **smoke['code_hashes']})
    full_specs = [read(entry['spec']) for entry in full_manifest['workers'].values()]
    tasks = set()
    for entry in smoke['jobs']:
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Changed smoke job')
        job = read(entry['path'])
        if len(job['samples']) != 3 or len({s['task'] for s in job['samples']}) != 1:
            raise ValueError('Smoke must contain three complete requests for each task')
        task = job['samples'][0]['task']
        tasks.add(task)
        spec = read(job['spec'])
        matching = [full for full in full_specs if full['candidate']['model'] == spec['candidate']['model']
                    and full['candidate']['revision'] == spec['candidate']['revision']
                    and full['execution'] == spec['execution']
                    and read(full['resources']['protocol']) == read(spec['resources']['protocol'])]
        if not matching:
            raise ValueError('Smoke uses another model, execution setting or baseline task protocol')
        folder = Path(results)/entry['id']
        receipt, outcome = read(folder/'receipt.json'), read(folder/'diagnostic-result.json')
        if (receipt.get('status') != 'COMPLETE' or receipt.get('manifest_sha256') != digest(manifest)
                or receipt.get('job_sha256') != entry['sha256'] or outcome.get('collection_complete') is not True
                or outcome.get('expected_requests') != 3 or outcome.get('returned_requests') != 3
                or outcome.get('valid_completions') != 3):
            raise ValueError('Full calibration withheld: smoke outputs are incomplete or invalid')
    if tasks != {'text', 'image', 'ocr', 'asr'}:
        raise ValueError('Missing a native task in smoke evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['manifest', 'source-root', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--smoke-manifest')
    parser.add_argument('--smoke-results')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('GPU launch requires --execute and release of the user hold')
    require_gpu_release()
    manifest_path, source_root = Path(args.manifest).resolve(), Path(args.source_root).resolve()
    manifest = read(manifest_path)
    if not manifest.get('diagnostic_only'):
        raise ValueError('This launcher accepts only diagnostic packets')
    verify({**manifest['sources'], **manifest['code_hashes']})
    expected_code = {str(p.resolve()): digest(p) for p in source_root.glob('gsdeploy/**/*.py')}
    if manifest['code_hashes'] != expected_code:
        raise ValueError('Execution source tree differs from the frozen packet')
    for entry in manifest['jobs']:
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Changed packet job')
        job = read(entry['path'])
        if read(job['spec'])['execution']['family'] == 'minicpm45' and len(job['samples']) > 3:
            if not args.smoke_manifest or not args.smoke_results:
                raise ValueError('Full Mini45 calibration requires the corresponding complete four-task smoke evidence')
            validate_smoke(Path(args.smoke_manifest), manifest, args.smoke_results)
            break
    if Path(args.output).exists():
        raise FileExistsError('Preserve existing evidence; use a new result directory')
    controller = '/root/autodl-tmp/gsdeploy-assets/envs/compact-standard/bin/python'
    # Root diagnostics already acquire this same lock. Source-isolated
    # diagnostics also acquire their local lock, so add the shared outer lease.
    with shared_gpu_lease() if source_root != PROJECT_ROOT else nullcontext():
        process = subprocess.run([controller, '-m', 'gsdeploy.profiling.diagnostics', '--manifest', str(manifest_path),
                                  '--output', str(Path(args.output).resolve()), '--execute'], cwd=source_root,
                                 env={**os.environ, 'PYTHONPATH': str(source_root), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    if process.returncode:
        raise SystemExit(process.returncode)
    state = read(Path(args.output)/'diagnostic-queue-state.json')
    if (state.get('status') != 'COMPLETE_DIAGNOSTICS_NOT_ADMITTED' or state.get('failed') or state.get('blocked')
            or set(state.get('completed', [])) != {entry['id'] for entry in manifest['jobs']}):
        raise SystemExit('Diagnostic collection failed; original results retained for review')


if __name__ == '__main__':
    main()
