"""CPU prerequisite audit. No torch import, model loading or GPU discovery."""
import argparse
import hashlib
import json
from pathlib import Path

from .prepare import digest
from .runner import save
from .evaluation import read_rows
from ..backends.compact import CompactAdapter


def inspect_cpu(study_path, inventory_path, environments_path, preprocessing_root, experiments_path,
                test_report, output):
    study = json.loads(Path(study_path).read_text())
    inventory = json.loads(Path(inventory_path).read_text())
    envs = json.loads(Path(environments_path).read_text())
    errors, checks, hashes = [], {}, {}
    sources = [Path(study_path), Path(inventory_path), Path(environments_path), Path(experiments_path), Path(test_report)]
    contract = study['task_adapter_protocol']
    execution_path = Path('configs/compact_execution.json')
    execution = json.loads(execution_path.read_text())
    sources.append(execution_path)
    protocol_path = Path(contract['path'])
    jobs = json.loads(Path(experiments_path).read_text())
    classmap_path = next(Path(p) for p in jobs['sources'] if Path(p).name == 'imagenet-class-index.json')
    protocol_fingerprint = CompactAdapter(json.loads(protocol_path.read_text()), json.loads(classmap_path.read_text())).policy_sha256
    sources.append(protocol_path)
    if digest(protocol_path) != contract['sha256'] or digest(environments_path) != contract['environment_plan_sha256']:
        errors.append('PROTOCOL_OR_ENVIRONMENT_HASH_CHANGED')
    if study['execution_catalog'] is not None:
        errors.append('UNMEASURED_CATALOG_MUST_REMAIN_NULL')
    checks['resources'] = inventory['status'] == 'COMPLETE'
    for slot, model in inventory['models'].items():
        directory = Path(model['path'])
        for item in model['files']:
            path = directory / item['name']
            okay = item['status'] == 'VERIFIED' and path.is_file() and path.stat().st_size == item['size']
            if okay:
                if item.get('sha256'):
                    okay = digest(path) == item['sha256']
                else:
                    raw = path.read_bytes()
                    okay = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == item['git_blob']
            if not okay:
                errors.append('MISSING_OR_CHANGED_RESOURCE:' + slot + ':' + item['name'])
    checks['environments'] = envs['state'] == 'CPU_INSTALLED_LOCKED_GPU_UNVALIDATED'
    checks['environments'] &= len(envs.get('installed_environments', {})) == 5
    for entry in envs.get('installed_environments', {}).values():
        receipt = Path(entry['cpu_audit'])
        sources.append(receipt)
        evidence = json.loads(receipt.read_text())
        if evidence['status'] != 'PASS_CPU_IMPORTS_AND_DEPENDENCIES' or evidence['cuda_initialized'] or evidence['weights_loaded']:
            errors.append('INVALID_ENVIRONMENT_CPU_AUDIT:' + str(receipt))
        if not Path(entry['python']).is_file():
            errors.append('MISSING_INTERPRETER:' + entry['python'])
        for source, sha in entry['lock_hashes'].items():
            if digest(source) != sha:
                errors.append('CHANGED_ENVIRONMENT_LOCK:' + source)
            sources.append(Path(source))
    expected_total, observed_total = 0, 0
    inputs = {}
    for phase, key, filename in [('quality', 'quality_preparation', 'evaluation-status.json'),
                                  ('calibration', 'calibration_preparation', 'sample-manifests.json')]:
        manifest = Path(study[key]['manifest']); sources.append(manifest)
        if digest(manifest) != study[key]['manifest_sha256']:
            errors.append('CHANGED_INPUT_MANIFEST:' + phase)
        data = json.loads(manifest.read_text()); inputs[phase] = data
        for task, entry in data['tasks'].items():
            if digest(entry['path']) != entry['sha256']:
                errors.append('CHANGED_TASK_INPUT:' + phase + ':' + task)
            sources.append(Path(entry['path']))
            for row in read_rows(entry['path']):
                if task != 'text' and digest(row['input']) != row['input_sha256']:
                    errors.append('CHANGED_MEDIA:' + row['id'])
        for candidate in study['generalists'] + study['specialists']:
            slot = candidate['slot']
            tasks = study['active_tasks'] if slot.startswith('generalist_') else [slot]
            report = Path(preprocessing_root) / ('preprocessing-' + phase + '-' + slot) / 'preprocessing.json'
            sources.append(report)
            if not report.is_file():
                errors.append('MISSING_FULL_CPU_PREPROCESSING:' + phase + ':' + slot)
                continue
            audit = json.loads(report.read_text())
            wanted = sum(data['tasks'][t]['samples'] for t in tasks)
            expected_total += wanted
            observed_total += audit['successful']
            if (audit['status'] != 'PASS_CPU_PREPROCESSING' or audit['limited_diagnostic']
                    or audit['successful'] != wanted or audit['cuda_initialized'] or audit['weights_loaded']
                    or audit['protocol_fingerprint'] != protocol_fingerprint
                    or audit['input_manifest_sha256'] != {t: data['tasks'][t]['sha256'] for t in tasks}):
                errors.append('INVALID_FULL_CPU_PREPROCESSING:' + phase + ':' + slot)
            bound = execution['models'][slot].get('max_input_length')
            if bound and any((row['input_tokens'] or 0) > bound for row in audit['observations']):
                errors.append('NATIVE_PREFILL_BUFFER_TOO_SMALL:' + phase + ':' + slot)
    checks['preprocessing'] = observed_total == expected_total == 7840
    tests = json.loads(Path(test_report).read_text())
    checks['tests'] = tests['status'] == 'PASS' and tests['failures'] == tests['errors'] == tests['skipped'] == 0
    for path, expected in tests['source_hashes'].items():
        if digest(path) != expected:
            errors.append('TESTED_SOURCE_CHANGED:' + path)
    checks['jobs_prepared'] = jobs['status'] == 'PREPARED_GPU_DISABLED' and len(jobs['jobs']) == 83
    for source, sha in {**jobs['sources'], **jobs['code_hashes']}.items():
        if digest(source) != sha:
            errors.append('CHANGED_EXPERIMENT_SOURCE:' + source)
    for job in jobs['jobs']:
        if digest(job['path']) != job['sha256']:
            errors.append('CHANGED_EXPERIMENT_JOB:' + job['id'])
    for key, okay in checks.items():
        if not okay:
            errors.append('INCOMPLETE:' + key)
    for p in sources:
        if p.is_file():hashes[str(p.resolve())] = digest(p)
    code = {str(p): digest(p) for p in Path('gsdeploy').rglob('*.py')}
    result = {'schema': 'gsdeploy.compact_cpu_readiness.v1',
              'status': 'CPU_PREREQUISITES_COMPLETE_GPU_UNVALIDATED' if not errors else 'CPU_PREREQUISITES_INCOMPLETE',
              'checks': checks, 'issues': sorted(set(errors)), 'source_hashes': hashes, 'code_hashes': code,
              'preprocessed_requests': observed_total, 'gpu_started': False, 'models_loaded': 0,
              'quality_admitted': False, 'runtime_catalog': None, 'utility': None,
              'remaining_gpu_work': ['Native load/quantization/kernel compatibility and complete output EOS',
                  'Independent quality, paired all-three-generalist comparisons and matched board energy',
                  'Measured residence/peak memory, n1/n2/n3 feasibility and actual capacity',
                  'Whole-plan colocation/routing/replica evidence, then catalog and static search validation']}
    save(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', default='configs/study.json')
    p.add_argument('--inventory', required=True)
    p.add_argument('--environments', default='configs/compact_environments.json')
    p.add_argument('--preprocessing-root', required=True)
    p.add_argument('--experiments', required=True)
    p.add_argument('--test-report', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    r = inspect_cpu(a.study, a.inventory, a.environments, a.preprocessing_root, a.experiments, a.test_report, a.output)
    print(json.dumps({k: r[k] for k in ['status', 'checks', 'issues', 'preprocessed_requests', 'gpu_started']}, indent=2))
    if r['issues']:raise SystemExit(1)


if __name__ == '__main__':
    main()
