"""Verify a derived QA adapter's immutable identity before any model loading."""
import hashlib
import json
from pathlib import Path
import re


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verified_qa_adapter(profile, candidate):
    binding = profile.get('peft_adapter')
    identity = candidate.get('adapter_sha256')
    if binding is None:
        if identity is not None:
            raise ValueError('Derived candidate is missing its adapter execution binding')
        return None
    if (candidate['model'] != 'nvidia/Llama3-ChatQA-1.5-8B' or candidate['slot'] != 'text'
            or not isinstance(binding, dict) or set(binding) != {'manifest', 'sha256'}
            or not isinstance(identity, str) or re.fullmatch('[0-9a-f]{64}', identity) is None
            or binding['sha256'] != identity):
        raise ValueError('QA adapter execution and request identities must match')
    path = Path(binding['manifest']).resolve()
    if sha256(path) != identity:
        raise ValueError('Changed pinned QA adapter manifest')
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 'gsdeploy.local_qa_adapter.v1'
            or manifest.get('model') != candidate['model'] or manifest.get('revision') != candidate['revision']):
        raise ValueError('QA adapter is bound to a different base model/revision')
    files = manifest['files']
    if len(files) != 2 or {f['name'] for f in files} != {'adapter_config.json', 'adapter_model.safetensors'}:
        raise ValueError('Only the declared LoRA config and safetensors weights are allowed')
    directory = Path(manifest['path']).resolve()
    for entry in files:
        target = directory / entry['name']
        if (not target.is_file() or target.stat().st_size != entry['size']
                or sha256(target) != entry['sha256']):
            raise ValueError('Changed pinned QA adapter file: ' + entry['name'])
    config = json.loads((directory / 'adapter_config.json').read_text())
    if (config.get('peft_type') != 'LORA' or config.get('task_type') != 'CAUSAL_LM'
            or config.get('bias', 'none') != 'none' or config.get('modules_to_save')):
        raise ValueError('The derived QA candidate permits only the declared causal-LM LoRA adapter')
    return {'path': str(directory), 'sha256': identity, 'manifest': str(path)}
