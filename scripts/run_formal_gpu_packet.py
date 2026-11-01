"""Run a sealed formal90 batch from its isolated source after explicit GPU release."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from resume_gpu_diagnostics import digest, read, require_gpu_release, verify
from run_prepared_gpu_packet import shared_gpu_lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['manifest', 'source-root', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--through', choices=['smoke', 'memory', 'quality', 'profile', 'matched_cost'], default='matched_cost')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Formal GPU execution requires --execute')
    require_gpu_release()
    source = Path(args.source_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = read(manifest_path)
    if manifest.get('diagnostic_only') or not manifest.get('memory_validation'):
        raise ValueError('A formal batch requires fresh full calibration memory and independent quality')
    expected = {str(p.resolve()): digest(p) for p in source.glob('gsdeploy/**/*.py')}
    expected[str(source / 'scripts/run_compact_gpu_queue.py')] = digest(source / 'scripts/run_compact_gpu_queue.py')
    if manifest['code_hashes'] != expected:
        raise ValueError('Formal batch source differs from the sealed source revision')
    verify({**manifest['sources'], **manifest['code_hashes']})
    for entry in manifest['jobs']:
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Changed formal job: ' + entry['id'])
    for worker in manifest['workers'].values():
        if digest(worker['spec']) != worker['spec_sha256']:
            raise ValueError('Changed formal worker')
        policy = read(read(worker['spec'])['resources']['protocol'])
        if policy.get('diagnostic_only'):
            raise ValueError('Diagnostic protocols cannot grant formal admission')
    if Path(args.output).exists():
        raise FileExistsError('Retain prior evidence; use a fresh formal output folder')
    # The queue holds its isolated source lease; this outer lease coordinates
    # it with every diagnostic collector without changing any frozen collector.
    with shared_gpu_lease():
        result = subprocess.run([
            '/root/autodl-tmp/gsdeploy-assets/envs/compact-standard/bin/python',
            str(source / 'scripts/run_compact_gpu_queue.py'), '--manifest', str(manifest_path),
            '--output', str(Path(args.output).resolve()), '--through', args.through, '--execute'],
            cwd=source, env={**os.environ, 'PYTHONPATH': str(source),
                            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
