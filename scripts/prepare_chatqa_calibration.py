"""CPU preparation for the pinned QA replacement; never downloads or loads weights."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsdeploy.backends.compact import CompactAdapter
from gsdeploy.backends.compact_native import NativeExecutor
from gsdeploy.profiling.diagnostics import prepare
from gsdeploy.profiling.prepare import digest
from gsdeploy.profiling.runner import save


def prepare_candidate(output, *, audit_only=False):
    import torch
    from transformers import AutoTokenizer
    study_path = ROOT / 'configs/study_repaired.json'
    study = json.loads(study_path.read_text())
    candidate = next(c for c in study['specialists'] if c['slot'] == 'text')
    inventory_path = ROOT / 'artifacts/qa-replacement-20261018/resources/resource-inventory.json'
    inventory = json.loads(inventory_path.read_text())
    checkpoint = inventory['models']['text']
    if (checkpoint['model'], checkpoint['revision']) != (candidate['model'], candidate['revision']):
        raise ValueError('QA replacement checkpoint identity mismatch')
    directory = Path(checkpoint['path'])
    for entry in checkpoint['files']:
        if entry['name'] in {'tokenizer.json', 'tokenizer_config.json', 'config.json'}:
            if entry['status'] != 'VERIFIED':
                raise ValueError('QA tokenizer/config not verified')
    policy = json.loads(Path(study['task_adapter_protocol']['path']).read_text())
    if digest(study['task_adapter_protocol']['path']) != study['task_adapter_protocol']['sha256']:
        raise ValueError('Changed repair protocol')
    classmap = Path('/root/autodl-tmp/gsdeploy-assets/datasets/imagenet-class-index.json')
    adapter = CompactAdapter(policy, json.loads(classmap.read_text()))
    calibration_path = ROOT / study['calibration_preparation']['manifest']
    if digest(calibration_path) != study['calibration_preparation']['manifest_sha256']:
        raise ValueError('Changed calibration manifest')
    entry = json.loads(calibration_path.read_text())['tasks']['text']
    if digest(entry['path']) != entry['sha256']:
        raise ValueError('Changed QA calibration inputs')
    samples = [json.loads(line) for line in Path(entry['path']).read_text().splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    config = json.loads((directory / 'config.json').read_text())
    executor = NativeExecutor(adapter, candidate, SimpleNamespace(config=SimpleNamespace(**config)), tokenizer,
                              torch=torch, device='cpu', dtype=torch.bfloat16)
    audit = []
    for sample in samples:
        request = adapter.build(candidate, sample)
        changed = copy.deepcopy(sample)
        changed.update(answers=['FORBIDDEN_REFERENCE'], label='FORBIDDEN_REFERENCE')
        if adapter.build(candidate, changed) != request:
            raise ValueError('QA references entered model input')
        prepared = executor.prepare(request)
        ids = prepared['inputs']['input_ids'][0].tolist()
        if ids[:2] == [tokenizer.bos_token_id] * 2:
            raise ValueError('Duplicate BOS at native prompt start')
        audit.append({'sample_id': sample['id'], 'input_tokens': prepared['input_tokens'],
                      'output_budget': request['generation']['max_new_tokens']})
    terminators = [tokenizer.convert_tokens_to_ids(t) for t in ['<|end_of_text|>', '<|eot_id|>']]
    if terminators != [128001, 128009]:
        raise ValueError('Unexpected pinned QA native terminators')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'cpu-input-audit.json', {'status': 'PASS', 'gpu_started': False, 'models_loaded': 0,
          'samples': audit, 'bos_id': tokenizer.bos_token_id, 'eos_ids': terminators,
          'context_limit': config['max_position_embeddings'], 'scope': 'native_tokenization_only_not_quality'})
    if audit_only:
        return
    if inventory['status'] != 'COMPLETE' or not checkpoint['files'] or any(f['status'] != 'VERIFIED' for f in checkpoint['files']):
        raise ValueError('QA calibration cannot freeze before complete verified weights')
    execution_path = Path(study['execution_configuration'])
    execution = json.loads(execution_path.read_text())['models']['text']
    template = {'id': 'text', 'kind': 'compact', 'adapter': 'compact', 'candidate': candidate,
                'execution': execution, 'resources': {'checkpoint': checkpoint,
                    'protocol': study['task_adapter_protocol']['path'], 'classmap': str(classmap),
                    'request_output': str(output / 'temporary')}}
    worker = output / 'template-worker.json'
    save(worker, template)
    base = output / 'template-manifest.json'
    save(base, {'study': str(study_path), 'workers': {'text': {'spec': str(worker),
               'python': '/root/autodl-tmp/gsdeploy-assets/envs/compact-standard/bin/python'}}})
    diagnostic_policy = copy.deepcopy(policy)
    diagnostic_policy['diagnostic_only'] = True
    cases = [{'id': 'qa-chatqa-nf4-native-calibration', 'parent': None, 'factor': None,
              'candidate': candidate, 'execution': execution, 'policy': diagnostic_policy, 'samples': samples}]
    prepare(cases, base, output / 'experiments', extra_sources=[__file__, inventory_path, study_path,
            execution_path, calibration_path, entry['path'], study['task_adapter_protocol']['environment_plan']])
    print(json.dumps({'status': 'CPU_PREPARED_GPU_DISABLED', 'requests': len(samples),
                      'manifest': str(output / 'experiments/experiment-manifest.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    prepare_candidate(args.output, audit_only=args.audit_only)
