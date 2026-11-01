"""Read-only CPU verification of prepared native candidates; never imports torch.

Hashing checkpoint bytes verifies storage only. It cannot establish GPU loading,
quality, memory, latency, energy or model-pool admission.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check_bindings(bindings, base=None):
    for name, expected in bindings.items():
        path = Path(name)
        if not path.is_absolute():
            path = Path(base) / path
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Changed or missing bound input/source: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = {'status': 'RUNNING_CPU_ONLY', 'gpu_started': False, 'weights_loaded': 0,
              'pool_admitted': False, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'controller_sha256': digest(__file__), 'checks': {}, 'failures': []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(output)

    def run(name, callback):
        try:
            report['checks'][name] = {'status': 'PASS', **callback()}
        except Exception as error:
            report['checks'][name] = {'status': 'FAIL', 'error': str(error)}
            report['failures'].append(name)
        save()

    def regression(stage, receipt):
        path = ROOT / stage / receipt
        data = read(path)
        if (data['status'] != 'PASS' or data['failures'] or data['errors'] or data['skipped']
                or data['gpu_started'] or data['cuda_initialized'] or data['models_loaded']):
            raise ValueError('CPU regression receipt failed or skipped tests')
        check_bindings(data['source_hashes'], ROOT / stage)
        return {'tests': data['tests_run'], 'receipt': str(path), 'sha256': digest(path)}

    for stage, receipt in [
        ('artifacts/formal-pool-release-stage-20261018', 'cpu-tests-preprocessing-pinned.json'),
        ('artifacts/dots-adapter-stage-20261018', 'cpu-tests.json'),
        ('artifacts/minicpm45-adapter-stage-20261018', 'cpu-tests-final-v2.json'),
    ]:
        run(stage, lambda stage=stage, receipt=receipt: regression(stage, receipt))

    base = ROOT / 'artifacts/formal-pool-release-stage-20261018/input-preparation'
    jobs = read(base / 'audit-jobs.json')
    for name, job in jobs.items():
        def audit(name=name, job=job):
            folder = base / (name + '-quality-inputs')
            path, contract_path = folder / 'preprocessing.json', folder / 'audit-contract.json'
            data, contract = read(path), read(contract_path)
            if (data['status'] != 'PASS_CPU_PREPROCESSING' or data['failures']
                    or data['samples_checked'] != job['expected_requests']
                    or data['successful'] != job['expected_requests'] or data['limited_diagnostic']
                    or data['cuda_initialized'] or data['weights_loaded'] or data['gpu_started']
                    or data['audit_contract_sha256'] != digest(contract_path)):
                raise ValueError('Full native input audit failed or has changed')
            check_bindings(contract['source_hashes'])
            for key in ['study', 'inventory']:
                if digest(job[key]) != contract[key + '_sha256']:
                    raise ValueError('Changed input audit ' + key)
            study = read(job['study'])
            if digest(job['protocol']) != study['task_adapter_protocol']['sha256']:
                raise ValueError('Changed task protocol')
            observations = data['observations']
            if len(observations) != job['expected_requests']:
                raise ValueError('Incomplete per-request observations')
            return {'requests': len(observations), 'by_task': dict(Counter(r['task'] for r in observations)),
                    'receipt': str(path), 'sha256': digest(path), 'contract_sha256': digest(contract_path)}
        run('native_inputs_' + name, audit)

    inventories = [
        'artifacts/generalist-replacement-20261018/qwen7b-local-resources/resource-inventory.json',
        'artifacts/ocr-reserve-20261018/dots-resources/resource-inventory.json',
        'artifacts/minicpm45-replacement-20261018/resources/resource-inventory.json',
    ]
    for relative in inventories:
        def inventory(relative=relative):
            path = ROOT / relative
            before = digest(path)
            data = read(path)
            if data['status'] != 'COMPLETE':
                raise ValueError('Checkpoint inventory is not complete')
            observed = []
            for model in data['models'].values():
                for item in model['files']:
                    target = Path(model['path']) / item['name']
                    if item['status'] != 'VERIFIED' or target.stat().st_size != item['size']:
                        raise ValueError('Checkpoint file missing or size mismatch: ' + str(target))
                    expected = item.get('sha256')
                    algorithm = 'sha256' if expected else 'git_blob_sha1'
                    value = hashlib.sha256() if expected else hashlib.sha1()
                    if not expected:
                        expected = item.get('git_blob')
                        if not expected:
                            raise ValueError('Checkpoint file lacks source hash')
                        value.update(f'blob {item["size"]}\0'.encode())
                    with target.open('rb') as stream:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                            value.update(block)
                    if value.hexdigest() != expected:
                        raise ValueError('Checkpoint source hash mismatch: ' + str(target))
                    observed.append({'path': str(target), 'size': item['size'],
                                     'algorithm': algorithm, 'digest': value.hexdigest()})
            if digest(path) != before:
                raise ValueError('Checkpoint inventory changed during audit')
            return {'inventory': str(path), 'inventory_sha256': before, 'files': observed,
                    'bytes_verified': sum(item['size'] for item in observed)}
        run('checkpoint_' + relative.split('/')[1], inventory)
    report.update(status='PASS_CPU_PREREQUISITES_GPU_HELD' if not report['failures'] else 'CPU_PREREQUISITES_NEED_REPAIR',
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  limitations=['GPU unavailable per user; no GPU action attempted.',
                               'Native GPU loading and all missing quality/cost/system evidence remain unverified.'])
    save()
    print(json.dumps({k: report[k] for k in ['status', 'failures', 'gpu_started', 'pool_admitted']}))
    if report['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
