"""Prepare an immutable continuation packet on CPU; execution respects GPU hold.

Completed jobs remain in their original folders and keep their original receipt
identities. Interrupted jobs start in a new folder using the exact original job.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
HOLD = ROOT / 'artifacts/continuous-gpu-study-20261018/gpu-execution-hold.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def verify(bindings):
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError('Frozen source/evidence changed: ' + path)


def require_gpu_release(hold_path=HOLD):
    if not Path(hold_path).exists() or read(hold_path).get('status') != 'RELEASED_BY_USER':
        raise ValueError('GPU is held at user request; availability must be confirmed before execution')


def prepare(manifest_path, previous_results, output):
    manifest_path, previous_results, output = map(lambda p: Path(p).resolve(),
                                                (manifest_path, previous_results, output))
    original = read(manifest_path)
    if not original.get('diagnostic_only'):
        raise ValueError('Only diagnostic jobs can use this continuation path')
    verify({**original['sources'], **original['code_hashes']})
    completed, remaining, bindings = [], [], {str(manifest_path): digest(manifest_path), str(Path(__file__).resolve()): digest(__file__)}
    for entry in original['jobs']:
        job_path = Path(entry['path'])
        if digest(job_path) != entry['sha256']:
            raise ValueError('Changed original diagnostic job')
        expected_requests = len(read(job_path)['samples'])
        folder = previous_results / entry['id']
        receipt_path, result_path = folder / 'receipt.json', folder / 'diagnostic-result.json'
        if receipt_path.exists() and result_path.exists():
            receipt, result = read(receipt_path), read(result_path)
            if (receipt.get('status') == 'COMPLETE' and receipt.get('manifest_sha256') == digest(manifest_path)
                    and receipt.get('job_sha256') == entry['sha256'] and result.get('collection_complete') is True
                    and expected_requests > 0 and result.get('returned_requests') == expected_requests
                    and result.get('expected_requests') == expected_requests):
                evidence = [p for p in folder.iterdir() if p.is_file()]
                bindings.update({str(p): digest(p) for p in evidence})
                completed.append({'id': entry['id'], 'result': str(result_path), 'quality_summary': result['quality_summary']})
                continue
        remaining.append(entry)
    if not remaining:
        raise ValueError('No unfinished diagnostic jobs')
    remaining_ids = {entry['id'] for entry in remaining}
    for entry in remaining:
        parent = read(entry['path']).get('parent')
        if parent and parent not in remaining_ids:
            raise ValueError('Remaining contrast depends on an external completed parent; prepare a separately bound contrast')
    packet = copy.deepcopy(original)
    packet.update(jobs=remaining, workers={key: value for key, value in original['workers'].items() if key in remaining_ids},
                  status='CPU_PREPARED_GPU_DISABLED', gpu_started=False)
    packet['sources'].update(bindings)
    output.mkdir(parents=True, exist_ok=False)
    packet_path = output / 'experiment-manifest.json'
    write(packet_path, packet)
    report = {'status': 'CPU_PREPARED_GPU_HELD', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'gpu_started': False, 'pool_admitted': False, 'parent_manifest': str(manifest_path),
              'parent_results': str(previous_results), 'retained_completed': completed,
              'remaining_jobs': [e['id'] for e in remaining],
              'remaining_requests': sum(len(read(e['path'])['samples']) if isinstance(read(e['path']).get('samples'), list)
                                        else read(e['path']).get('expected_requests', 0) for e in remaining),
              'manifest': str(packet_path), 'manifest_sha256': digest(packet_path),
              'preserved_evidence': bindings,
              'resume_rule': 'Use a new output directory; retain all original complete and interrupted evidence.'}
    write(output / 'continuation-plan.json', report)
    return report


def execute(plan_path, output):
    require_gpu_release()
    plan = read(plan_path)
    verify(plan['preserved_evidence'])
    if digest(plan['manifest']) != plan['manifest_sha256']:
        raise ValueError('Changed continuation packet')
    if Path(output).exists():
        raise FileExistsError('Continuation output must be new')
    sys.path.insert(0, str(ROOT))
    from gsdeploy.profiling.diagnostics import run
    return run(plan['manifest'], output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for key in ['manifest', 'previous-results', 'output']:
        prep.add_argument('--' + key, required=True)
    run = sub.add_parser('run')
    run.add_argument('--plan', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        report = prepare(args.manifest, args.previous_results, args.output)
        print(json.dumps({key: report[key] for key in ['status', 'remaining_jobs', 'remaining_requests', 'gpu_started']}))
    else:
        if not args.execute:
            parser.error('GPU execution requires --execute and release of the user hold')
        result = execute(args.plan, args.output)
        if result['status'] != 'COMPLETE_DIAGNOSTICS_NOT_ADMITTED':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
