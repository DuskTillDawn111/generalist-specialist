"""CPU-only preparation of all compact model/task request paths.

Builds and fingerprints call blueprints; never invokes an executor or loads a
model. Model admission, runtime readiness and total utility remain unset.
"""

import argparse
import json
from pathlib import Path

from ..backends.compact import CompactAdapter, fingerprint
from .evaluation import read_rows
from .prepare import digest


def prepare_compact(study_path, policy_path, environments_path, classmap_path, output):
    study_path, policy_path, environments_path, classmap_path, output = [
        Path(p).resolve() for p in (study_path, policy_path, environments_path, classmap_path, output)]
    if output.exists():
        raise FileExistsError(output)
    study = json.loads(study_path.read_text())
    contract = study['task_adapter_protocol']
    if digest(policy_path) != contract['sha256'] or digest(environments_path) != contract['environment_plan_sha256']:
        raise ValueError('Changed task protocol or environment plan')
    policy = json.loads(policy_path.read_text())
    environments = json.loads(environments_path.read_text())
    if study['active_tasks'] != ['text', 'image', 'ocr', 'asr'] or study['execution_catalog'] is not None:
        raise ValueError('Expected the four-task unadmitted study')
    candidates = study['generalists'] + study['specialists']
    if len(study['generalists']) != 3 or len(study['specialists']) != 4:
        raise ValueError('Expected three generalists and four specialists')
    if environments['state'] not in {'PARTIAL_SOURCE_PINS_NOT_AN_INSTALLATION_LOCK',
                                     'CPU_INSTALLED_LOCKED_GPU_UNVALIDATED'}:
        raise ValueError('Environment plan cannot certify execution')
    for candidate in candidates:
        if candidate['state'] != 'METADATA_SCREENED_NOT_ADMITTED':
            raise ValueError('This entry cannot certify admission')
        if environments['models'][candidate['model']]['model_revision'] != candidate['revision']:
            raise ValueError('Environment source pin does not match the model')
    adapter = CompactAdapter(policy, json.loads(classmap_path.read_text()))
    manifest = Path(study['quality_preparation']['manifest'])
    if not manifest.is_absolute():
        manifest = study_path.parent.parent / manifest
    if digest(manifest) != study['quality_preparation']['manifest_sha256']:
        raise ValueError('Quality manifest changed')
    evaluation = json.loads(manifest.read_text())
    if evaluation['status'] not in {'FOUR_TASK_INPUTS_FROZEN_PROTOCOL_REVIEW_PENDING_NOT_ADMITTED',
                                    'FOUR_TASK_INPUTS_AND_PROTOCOL_FROZEN_NOT_ADMITTED'}:
        raise ValueError('Expected the frozen four-task input manifest')
    pinned_classes = [sha for name, sha in evaluation['source_hashes'].items()
                      if Path(name).name == 'imagenet-class-index.json']
    if len(pinned_classes) != 1 or digest(classmap_path) != pinned_classes[0]:
        raise ValueError('Class catalog does not match the frozen quality inputs')
    samples = {}
    for task in study['active_tasks']:
        entry = evaluation['tasks'][task]
        path = Path(entry['path'])
        if digest(path) != entry['sha256']:
            raise ValueError(f'Changed quality input manifest: {task}')
        rows = read_rows(path)
        if len(rows) != entry['samples'] or len({row['id'] for row in rows}) != len(rows):
            raise ValueError(f'Invalid independent sample cohort: {task}')
        for sample in rows:
            if sample['task'] != task:
                raise ValueError('Cross-task sample manifest')
            if task != 'text' and digest(sample['input']) != sample['input_sha256']:
                raise ValueError(f"Changed media: {sample['id']}")
        samples[task] = rows
    output.mkdir(parents=True)
    paths = []
    for candidate in candidates:
        model_dir = output / candidate['slot']
        model_dir.mkdir()
        for task in adapter.supported_tasks(candidate):
            hashes = []
            preview = None
            # Each blueprint contains only task inputs, never quality references.
            for sample in samples[task]:
                request = adapter.build(candidate, sample)
                preview = request if preview is None else preview
                hashes.append({'sample_id': sample['id'], 'request_sha256': fingerprint(request)})
            path = model_dir / (task + '-requests.json')
            path.write_text(json.dumps(hashes, indent=2) + '\n')
            example = model_dir / (task + '-example.json')
            example.write_text(json.dumps(preview, ensure_ascii=False, indent=2) + '\n')
            paths.append({'model': candidate['model'], 'revision': candidate['revision'], 'task': task,
                          'constructed_requests': len(hashes), 'executed_requests': 0,
                          'request_hashes': str(path), 'request_hashes_sha256': digest(path),
                          'native_api': preview['native']['api'],
                          'state': 'CPU_BLUEPRINT_VALIDATED_NATIVE_EXECUTION_UNVERIFIED'})
    for name, source in [('study.json', study_path), ('task-protocol.json', policy_path),
                         ('environments.json', environments_path)]:
        (output / name).write_bytes(source.read_bytes())
    result = {'schema': 'gsdeploy.compact_adapter_preparation.v1',
              'status': 'CPU_BLUEPRINTS_PREPARED_NOT_EXECUTED_OR_ADMITTED',
              'gpu_started': False, 'models_loaded': 0, 'executed_requests': 0,
              'native_execution_ready': False, 'utility': None,
              'quality_samples': {task: len(rows) for task, rows in samples.items()},
              'paths': paths, 'total_paths': len(paths),
              'constructed_requests': sum(path['constructed_requests'] for path in paths),
              'source_hashes': {str(p): digest(p) for p in [study_path, policy_path, environments_path, classmap_path, manifest]},
              'code_hashes': {str(p): digest(p) for p in [
                  Path(__file__), Path(__file__).parents[1] / 'tasks.py',
                  Path(__file__).parents[1] / 'backends/compact.py']},
              'protocol_fingerprint': adapter.policy_sha256,
              'classmap_fingerprint': adapter.classmap_sha256,
              'remaining_requirements': [
                  'GPU compatibility, measured memory/latency/quality/energy, feasible replicas and colocation',
                  'All-three-generalist admission comparisons and evidence-backed runtime catalog',
                  'GPU measurements only after the user resumes GPU experiments',
              ]}
    (output / 'preparation-status.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='configs/study.json')
    parser.add_argument('--protocol', default='configs/compact_protocol.json')
    parser.add_argument('--environments', default='configs/compact_environments.json')
    parser.add_argument('--classmap', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = prepare_compact(args.study, args.protocol, args.environments, args.classmap, args.output)
    print(json.dumps({k: result[k] for k in ['status', 'total_paths', 'constructed_requests', 'executed_requests', 'gpu_started']}, indent=2))


if __name__ == '__main__':
    main()
